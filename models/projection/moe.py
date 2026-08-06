
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Optional, Tuple, Union

SUPPORTED_ROUTER_TYPE = "mlp_moe"


def expert_etf_loss(expert_outputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Equiangular tight frame loss on per-patch expert delta directions.

    expert_outputs: [B, N, E, C] or [N, E, C]
    """
    if expert_outputs.dim() == 3:
        expert_outputs = expert_outputs.unsqueeze(0)
    b, n, e, c = expert_outputs.shape
    if e <= 1:
        return expert_outputs.new_tensor(0.0)

    z = F.normalize(expert_outputs, dim=-1, eps=eps)
    z = z.reshape(b * n, e, c)
    gram = torch.bmm(z, z.transpose(1, 2))

    target = torch.full((e, e), -1.0 / (e - 1), device=gram.device, dtype=gram.dtype)
    target.fill_diagonal_(1.0)
    return F.mse_loss(gram, target.unsqueeze(0).expand_as(gram))


def build_fofs_lora_A(
    d_model: int,
    num_experts: int,
    rank: int,
    device=None,
    dtype=None,
):
    """
    Frozen Orthogonal Feature Separation style initialization.

    Each expert receives an orthogonal low-rank projection focused on
    a different subspace of the input feature dimension.

    Returns:
        list of [rank, d_model] tensors.
    """
    base_chunk = d_model // num_experts
    remainder = d_model % num_experts

    fixed_As = []
    start = 0

    for i in range(num_experts):
        chunk = base_chunk + (1 if i < remainder else 0)
        end = start + chunk

        if rank > chunk:
            raise ValueError(
                f"rank={rank} should be <= chunk size={chunk}. "
                f"Use smaller rank or fewer experts."
            )

        # QR gives orthogonal columns: [chunk, rank]
        temp = torch.randn(chunk, rank, device=device, dtype=dtype)
        q, _ = torch.linalg.qr(temp, mode="reduced")

        A = torch.zeros(rank, d_model, device=device, dtype=dtype)
        A[:, start:end] = q.T

        fixed_As.append(A)
        start = end

    return fixed_As


def _topk_route(
    logits: torch.Tensor,
    top_k: int,
    tau: float,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Softmax + top-k sparsify. Returns dense, sparse, topk_idx|None."""
    prob = F.softmax(logits / tau, dim=-1)
    k = min(int(top_k), prob.shape[-1])
    if k >= prob.shape[-1]:
        return prob, prob, None
    topk_val, topk_idx = prob.topk(k, dim=-1)
    topk_val = topk_val / topk_val.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    sparse = torch.zeros_like(prob)
    sparse.scatter_(-1, topk_idx, topk_val)
    return prob, sparse, topk_idx


def _mix_expert_outputs(
    expert_outputs: torch.Tensor,
    dense_or_sparse: torch.Tensor,
    topk_idx: Optional[torch.Tensor],
    topk_val: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted sum of experts → [B, N, C]."""
    if topk_idx is None:
        return (dense_or_sparse.unsqueeze(-1) * expert_outputs).sum(dim=2)
    assert topk_val is not None
    selected = torch.gather(
        expert_outputs,
        2,
        topk_idx.unsqueeze(-1).expand(-1, -1, -1, expert_outputs.shape[-1]),
    )
    return (topk_val.unsqueeze(-1) * selected).sum(dim=2)


class LoRAExpert(nn.Module):
    """
    MoECLIP-style LoRA expert in DINO feature space.

    dim -> rank -> dim  (same input/output dimension).
    LoRA-B is zero-initialized so residual starts near zero.
    """

    def __init__(
        self,
        dim: int = 1024,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
        fixed_A: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.scaling = alpha / rank

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Linear(dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, dim, bias=False)

        if fixed_A is not None:
            with torch.no_grad():
                self.lora_A.weight.copy_(fixed_A)
            self.lora_A.weight.requires_grad = False
        else:
            init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        # Key: LoRA residual starts from zero.
        init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def _build_expert_list(
    *,
    vis_dim: int,
    num_experts: int,
    expert_rank: int,
    expert_alpha: int,
    expert_dropout: float,
    use_fofs: bool,
    device,
    dtype,
) -> nn.ModuleList:
    fixed_As = None
    if use_fofs and num_experts > 0:
        fixed_As = build_fofs_lora_A(
            d_model=vis_dim,
            num_experts=num_experts,
            rank=expert_rank,
            device=device,
            dtype=dtype,
        )
    experts = nn.ModuleList()
    for i in range(num_experts):
        fixed_A = fixed_As[i] if fixed_As is not None else None
        experts.append(
            LoRAExpert(
                dim=vis_dim,
                rank=expert_rank,
                alpha=expert_alpha,
                dropout=expert_dropout,
                fixed_A=fixed_A,
            )
        )
    return experts


def _make_linear_router(vis_dim: int, num_out: int) -> nn.Sequential:
    router = nn.Sequential(
        nn.LayerNorm(vis_dim),
        nn.Linear(vis_dim, num_out),
    )
    init.normal_(router[-1].weight, mean=0.0, std=0.02)
    init.zeros_(router[-1].bias)
    return router


class MoEVisualProjection(nn.Module):
    """
    MoECLIP-style DINO adapter + shared DINO-to-CLIP projection.

    Modes:
      - flat: single expert pool + one router (MoECLIP-like)
      - hierarchical (use_region_gate): E0 vs E1..K attribute experts
      - dual_path: additive region MoE + gated attribute MoE
            delta = region_delta + g * attr_delta
    """

    def __init__(
        self,
        vis_dim: int = 1024,
        output_dim: int = 768,
        num_experts: int = 4,
        use_learned_router: bool = True,
        router_hidden: int = 256,
        router_top_k: int = 2,
        router_type: str = SUPPORTED_ROUTER_TYPE,
        router_temperature: float = 1.0,
        expert_rank: int = 8,
        expert_alpha: int = 16,
        expert_dropout: float = 0.05,
        adapt_weight: float = 0.1,
        use_fofs: bool = True,
        normal_bank_top_k: int = 5,
        use_region_gate: bool = True,
        region_gate_mode: str = "feature",
        use_dual_path: bool = False,
        num_region_experts: Optional[int] = None,
        num_attr_experts: Optional[int] = None,
        router_attr_top_k: Optional[int] = None,
        dual_path_separate_norm: bool = False,
        dual_path_mode: str = "dual",
        **deprecated_kwargs,
    ):
        super().__init__()
        if deprecated_kwargs:
            ignored = ", ".join(sorted(deprecated_kwargs))
            print(f"MoEVisualProjection: ignoring deprecated kwargs: {ignored}")

        if router_type != SUPPORTED_ROUTER_TYPE:
            raise ValueError(
                f"Unsupported router_type={router_type!r}; "
                f"only {SUPPORTED_ROUTER_TYPE!r} is supported"
            )
        if not use_learned_router:
            raise ValueError("use_learned_router=False is no longer supported")

        mode = str(region_gate_mode).lower().strip()
        if mode not in {"feature", "cpa", "mean_dev"}:
            raise ValueError(
                f"region_gate_mode must be feature|cpa|mean_dev, got {region_gate_mode!r}"
            )

        self.vis_dim = vis_dim
        self.output_dim = output_dim
        self.use_dual_path = bool(use_dual_path)
        self.use_region_gate = bool(use_region_gate) and not self.use_dual_path
        valid_path_modes = {
            "base", "region", "attr_gated", "attr_force", "dual", "dual_force",
        }
        self.dual_path_mode = str(dual_path_mode).lower().strip()
        if self.dual_path_mode not in valid_path_modes:
            raise ValueError(
                f"dual_path_mode must be one of {sorted(valid_path_modes)}, "
                f"got {dual_path_mode!r}"
            )
        self.dual_path_separate_norm = bool(dual_path_separate_norm)
        self.region_gate_mode = mode
        self.use_cpa_router = bool(
            (self.use_region_gate or self.use_dual_path) and mode == "cpa"
        )
        self.router_tau = router_temperature
        self.expert_rank = expert_rank
        self.expert_alpha = expert_alpha
        self.expert_dropout = expert_dropout
        self.adapt_weight = adapt_weight
        self.use_fofs = use_fofs
        self.normal_bank_top_k = normal_bank_top_k
        self.normal_bank_tau: float = 0.07

        self._last_routing_weights: Optional[torch.Tensor] = None
        self._last_dense_routing_weights: Optional[torch.Tensor] = None
        self._last_ab_routing_weights: Optional[torch.Tensor] = None
        self._last_region_gate: Optional[torch.Tensor] = None
        self._last_expert_outputs: Optional[torch.Tensor] = None
        self._last_attr_expert_outputs: Optional[torch.Tensor] = None
        self._last_region_delta: Optional[torch.Tensor] = None
        self._last_attr_delta: Optional[torch.Tensor] = None
        self._last_gated_attr_delta: Optional[torch.Tensor] = None
        self._last_attr_projected: Optional[torch.Tensor] = None
        self._last_attr_routing_weights: Optional[torch.Tensor] = None
        self._last_attr_topk_idx: Optional[torch.Tensor] = None
        self._last_region_gate_logits: Optional[torch.Tensor] = None
        self._last_expert_deltas: Optional[torch.Tensor] = None
        self._last_patch_logits: Optional[torch.Tensor] = None
        self._last_batch_size: int = 1

        # Shared DINO → CLIP projection.
        self.base_proj = nn.Sequential(
            nn.LayerNorm(vis_dim),
            nn.Linear(vis_dim, output_dim),
        )
        init.zeros_(self.base_proj[1].bias)

        _ = router_hidden  # kept for API compatibility
        device = self.base_proj[1].weight.device
        dtype = self.base_proj[1].weight.dtype

        if self.use_dual_path:
            self.num_region_experts = int(
                num_region_experts if num_region_experts is not None else num_experts
            )
            self.num_attr_experts = int(
                num_attr_experts if num_attr_experts is not None else max(num_experts - 1, 1)
            )
            if self.num_region_experts < 2:
                raise ValueError("dual_path needs num_region_experts >= 2")
            if self.num_attr_experts < 1:
                raise ValueError("dual_path needs num_attr_experts >= 1")
            # Public alias used by viz / region processor: region pool size.
            self.num_experts = self.num_region_experts
            self.num_abnormal = self.num_attr_experts
            self.router_top_k = min(int(router_top_k), self.num_region_experts)
            self.router_attr_top_k = min(
                int(router_attr_top_k if router_attr_top_k is not None else router_top_k),
                self.num_attr_experts,
            )

            self.router_region = _make_linear_router(vis_dim, self.num_region_experts)
            self.router_attr = _make_linear_router(vis_dim, self.num_attr_experts)
            self.region_gate = nn.Sequential(
                nn.LayerNorm(vis_dim),
                nn.Linear(vis_dim, 1),
            )
            init.zeros_(self.region_gate[-1].weight)
            init.constant_(self.region_gate[-1].bias, -1.0)  # g≈0.27 early
            self.router_ab = None
            self.router_patch = None
            self.experts = None  # unused; dual path uses two pools

            self.region_experts = _build_expert_list(
                vis_dim=vis_dim,
                num_experts=self.num_region_experts,
                expert_rank=expert_rank,
                expert_alpha=expert_alpha,
                expert_dropout=expert_dropout,
                use_fofs=use_fofs,
                device=device,
                dtype=dtype,
            )
            self.attr_experts = _build_expert_list(
                vis_dim=vis_dim,
                num_experts=self.num_attr_experts,
                expert_rank=expert_rank,
                expert_alpha=expert_alpha,
                expert_dropout=expert_dropout,
                use_fofs=use_fofs,
                device=device,
                dtype=dtype,
            )
            mode_tag = f"dual_path(R={self.num_region_experts},A={self.num_attr_experts})"
            print(
                f"MoEVisualProjection MoECLIP-style [{mode_tag}]: "
                f"delta=region + g*attr | region_top_k={self.router_top_k}, "
                f"attr_top_k={self.router_attr_top_k}, tau={router_temperature}, "
                f"adapt_weight={adapt_weight} (norm-matched), use_fofs={use_fofs}, "
                f"separate_norm={self.dual_path_separate_norm}, "
                f"path_mode={self.dual_path_mode}, "
                f"projection={vis_dim}→{output_dim}"
            )
            return

        # ---- legacy hierarchical / flat ----
        if num_experts < 2:
            raise ValueError("MoE needs num_experts >= 2")
        self.num_experts = num_experts
        self.num_abnormal = num_experts - 1
        self.num_region_experts = num_experts
        self.num_attr_experts = self.num_abnormal if self.use_region_gate else 0
        self.router_top_k = min(
            router_top_k,
            self.num_abnormal if self.use_region_gate else num_experts,
        )
        self.router_attr_top_k = self.router_top_k
        self.region_experts = None
        self.attr_experts = None
        self.router_region = None
        self.router_attr = None

        if self.use_region_gate:
            self.region_gate = nn.Sequential(
                nn.LayerNorm(vis_dim),
                nn.Linear(vis_dim, 1),
            )
            init.zeros_(self.region_gate[-1].weight)
            init.constant_(self.region_gate[-1].bias, -1.0)
            self.router_ab = _make_linear_router(vis_dim, self.num_abnormal)
            self.router_patch = None
        else:
            self.region_gate = None
            self.router_ab = None
            self.router_patch = _make_linear_router(vis_dim, num_experts)

        self.experts = _build_expert_list(
            vis_dim=vis_dim,
            num_experts=num_experts,
            expert_rank=expert_rank,
            expert_alpha=expert_alpha,
            expert_dropout=expert_dropout,
            use_fofs=use_fofs,
            device=device,
            dtype=dtype,
        )
        mode_tag = (
            f"region_gate({self.region_gate_mode})+attr"
            if self.use_region_gate
            else "flat"
        )
        print(
            f"MoEVisualProjection MoECLIP-style [{mode_tag}]: "
            f"{num_experts} LoRA experts in DINO space, "
            f"rank={expert_rank}, alpha={expert_alpha}, "
            f"top_k={self.router_top_k}, tau={router_temperature}, "
            f"adapt_weight={adapt_weight} (norm-matched), use_fofs={use_fofs}, "
            f"projection={vis_dim}→{output_dim}"
        )

    def _input_for_gate(
        self,
        x: torch.Tensor,
        cpa_context: Optional[dict] = None,
    ) -> torch.Tensor:
        """Gate cue: feature (MoECLIP), mean_dev, or CPA residual."""
        if self.region_gate_mode == "feature":
            return x
        if self.region_gate_mode == "mean_dev":
            return F.normalize(x - x.mean(dim=1, keepdim=True), dim=-1)
        if cpa_context is not None and cpa_context.get("residuals") is not None:
            res = cpa_context["residuals"]
            if res.shape[:2] == x.shape[:2]:
                return res
        return F.normalize(x - x.mean(dim=1, keepdim=True), dim=-1)

    def _router_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_dual_path:
            assert self.router_region is not None
            logits = self.router_region(x)
        elif self.use_region_gate:
            assert self.router_ab is not None
            logits = self.router_ab(x)
        else:
            assert self.router_patch is not None
            logits = self.router_patch(x)
        self._last_patch_logits = logits.detach()
        return logits

    def forward(
        self,
        x: torch.Tensor,
        normal_bank: Optional[torch.Tensor] = None,
        class_ids: Optional[torch.Tensor] = None,
        cpa_context: Optional[dict] = None,
    ) -> torch.Tensor:
        _ = normal_bank, class_ids
        orig_shape = x.shape
        self._last_routing_weights = None
        self._last_dense_routing_weights = None
        self._last_ab_routing_weights = None
        self._last_region_gate = None
        self._last_expert_outputs = None
        self._last_attr_expert_outputs = None
        self._last_region_delta = None
        self._last_attr_delta = None
        self._last_gated_attr_delta = None
        self._last_attr_projected = None
        self._last_attr_routing_weights = None
        self._last_attr_topk_idx = None
        self._last_region_gate_logits = None
        self._last_weighted_delta = None
        self._last_x_input = None
        self._last_x_adapt = None
        self._last_topk_idx = None
        self._last_patch_logits = None

        if x.dim() == 4:
            b, h, w, d = x.shape
            x = x.view(b, h * w, d)
            spatial = (h, w)
        elif x.dim() == 2:
            x = x.unsqueeze(0)
            spatial = None
        else:
            spatial = None

        b, n, d = x.shape
        self._last_batch_size = b

        if self.use_dual_path:
            delta, topk_idx = self._forward_dual_path(x, cpa_context)
        elif self.use_region_gate:
            delta, topk_idx = self._forward_hierarchical(x, cpa_context)
        else:
            delta, topk_idx = self._forward_flat(x)

        # Attribute-only view gives the binding loss a path-specific target.
        if self.use_dual_path and self._last_attr_delta is not None:
            attr_matched = self._norm_match(self._last_attr_delta, x)
            attr_adapt = x + self.adapt_weight * attr_matched
            self._last_attr_projected = F.normalize(
                self.base_proj(attr_adapt), dim=-1
            )

        if self.use_dual_path and self.dual_path_mode == "base":
            delta_matched = torch.zeros_like(x)
            x_adapt = x
        elif self.use_dual_path and self.dual_path_separate_norm:
            region = self._last_region_delta
            attr = self._last_attr_delta
            g = self._last_region_gate
            assert region is not None and attr is not None and g is not None
            region_matched = self._norm_match(region, x)
            attr_matched = self._norm_match(attr, x)
            if self.dual_path_mode == "region":
                residual = region_matched
            elif self.dual_path_mode == "attr_gated":
                residual = g.unsqueeze(-1) * attr_matched
            elif self.dual_path_mode == "attr_force":
                residual = attr_matched
            elif self.dual_path_mode == "dual_force":
                residual = region_matched + attr_matched
            else:
                residual = region_matched + g.unsqueeze(-1) * attr_matched
            delta_matched = residual
            x_adapt = x + self.adapt_weight * residual
        else:
            # Legacy MoECLIP-style combined norm matching.
            delta_matched = self._norm_match(delta, x)
            x_adapt = (1.0 - self.adapt_weight) * x + self.adapt_weight * delta_matched

        out = self.base_proj(x_adapt)
        out = F.normalize(out, dim=-1)

        self._last_weighted_delta = delta_matched
        self._last_x_input = x
        self._last_x_adapt = x_adapt
        self._last_topk_idx = topk_idx

        if spatial is not None:
            h, w = spatial
            out = out.view(b, h, w, self.output_dim)
        elif len(orig_shape) == 2:
            out = out.squeeze(0)

        return out

    @staticmethod
    def _norm_match(delta: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return delta * reference.norm(dim=-1, keepdim=True) / (
            delta.norm(dim=-1, keepdim=True) + 1e-6
        )

    def _forward_dual_path(
        self,
        x: torch.Tensor,
        cpa_context: Optional[dict],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert self.region_experts is not None and self.attr_experts is not None
        assert self.router_region is not None and self.router_attr is not None
        assert self.region_gate is not None

        b, n, _ = x.shape
        region_outputs = torch.stack(
            [expert(x) for expert in self.region_experts], dim=2
        )  # [B, N, R, C]
        attr_outputs = torch.stack(
            [expert(x) for expert in self.attr_experts], dim=2
        )  # [B, N, A, C]

        r_logits = self.router_region(x)
        r_prob, r_sparse, r_idx = _topk_route(r_logits, self.router_top_k, self.router_tau)
        if r_idx is None:
            r_val = r_prob
        else:
            r_val = r_sparse.gather(-1, r_idx)
            r_val = r_val / r_val.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        region_delta = _mix_expert_outputs(region_outputs, r_sparse, r_idx, r_val)

        gate_in = self._input_for_gate(x, cpa_context)
        gate_logits = self.region_gate(gate_in).squeeze(-1)
        g = torch.sigmoid(gate_logits)  # [B, N]

        a_logits = self.router_attr(x)
        a_prob, a_sparse, a_idx = _topk_route(
            a_logits, self.router_attr_top_k, self.router_tau
        )
        if a_idx is None:
            a_val = a_prob
        else:
            a_val = a_sparse.gather(-1, a_idx)
            a_val = a_val / a_val.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attr_delta = _mix_expert_outputs(attr_outputs, a_sparse, a_idx, a_val)

        gated_attr_delta = g.unsqueeze(-1) * attr_delta
        if self.dual_path_mode == "base":
            delta = torch.zeros_like(region_delta)
        elif self.dual_path_mode == "region":
            delta = region_delta
        elif self.dual_path_mode == "attr_gated":
            delta = gated_attr_delta
        elif self.dual_path_mode == "attr_force":
            delta = attr_delta
        elif self.dual_path_mode == "dual_force":
            delta = region_delta + attr_delta
        else:
            delta = region_delta + gated_attr_delta

        self._last_patch_logits = r_logits.detach()
        self._last_region_gate = g
        self._last_region_gate_logits = gate_logits
        self._last_routing_weights = r_sparse.reshape(b * n, self.num_region_experts)
        self._last_dense_routing_weights = r_prob.reshape(b * n, self.num_region_experts)
        self._last_ab_routing_weights = a_prob.reshape(b * n, self.num_attr_experts)
        self._last_attr_routing_weights = a_sparse.reshape(b * n, self.num_attr_experts)
        self._last_attr_topk_idx = a_idx
        self._last_expert_outputs = region_outputs
        self._last_attr_expert_outputs = attr_outputs
        self._last_region_delta = region_delta
        self._last_attr_delta = attr_delta
        self._last_gated_attr_delta = gated_attr_delta
        return delta, r_idx

    def _forward_hierarchical(
        self,
        x: torch.Tensor,
        cpa_context: Optional[dict],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert self.experts is not None
        assert self.region_gate is not None and self.router_ab is not None
        b, n, _ = x.shape

        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=2
        )  # [B, N, E, C]

        gate_in = self._input_for_gate(x, cpa_context)
        g = torch.sigmoid(self.region_gate(gate_in)).squeeze(-1)  # [B, N]
        ab_logits = self.router_ab(x)  # [B, N, K]
        ab_prob, ab_sparse, topk_idx = _topk_route(
            ab_logits, self.router_top_k, self.router_tau
        )
        self._last_patch_logits = ab_logits.detach()

        if topk_idx is None:
            topk_val = ab_prob
        else:
            topk_val = ab_sparse.gather(-1, topk_idx)
            topk_val = topk_val / topk_val.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        ab_experts = expert_outputs[:, :, 1:, :]
        ab_delta = _mix_expert_outputs(ab_experts, ab_sparse, topk_idx, topk_val)
        e0_delta = expert_outputs[:, :, 0, :]
        delta = (1.0 - g.unsqueeze(-1)) * e0_delta + g.unsqueeze(-1) * ab_delta

        router_prob = torch.cat(
            [(1.0 - g).unsqueeze(-1), g.unsqueeze(-1) * ab_prob],
            dim=-1,
        )
        router_sparse = torch.cat(
            [(1.0 - g).unsqueeze(-1), g.unsqueeze(-1) * ab_sparse],
            dim=-1,
        )
        if topk_idx is not None:
            topk_idx = topk_idx + 1

        self._last_region_gate = g
        self._last_ab_routing_weights = ab_prob.reshape(b * n, self.num_abnormal)
        self._last_routing_weights = router_sparse.reshape(b * n, self.num_experts)
        self._last_dense_routing_weights = router_prob.reshape(b * n, self.num_experts)
        self._last_expert_outputs = expert_outputs
        return delta, topk_idx

    def _forward_flat(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert self.experts is not None
        b, n, _ = x.shape
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=2
        )
        router_logits = self._router_logits(x)
        router_prob, router_sparse, topk_idx = _topk_route(
            router_logits, self.router_top_k, self.router_tau
        )
        if topk_idx is None:
            topk_val = router_prob
        else:
            topk_val = router_sparse.gather(-1, topk_idx)
            topk_val = topk_val / topk_val.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        delta = _mix_expert_outputs(expert_outputs, router_sparse, topk_idx, topk_val)
        self._last_routing_weights = router_sparse.reshape(b * n, self.num_experts)
        self._last_dense_routing_weights = router_prob.reshape(b * n, self.num_experts)
        self._last_expert_outputs = expert_outputs
        return delta, topk_idx

    def routing_regularization_loss(
        self,
        balance_weight: float = 0.0,
        entropy_weight: float = 0.0,
        etf_weight: float = 0.0,
        balance_abnormal_only: bool = True,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)

        if self.use_dual_path:
            # Balance region router (always) + attribute router.
            for w, e_bal in (
                (self._last_dense_routing_weights, self.num_region_experts),
                (self._last_ab_routing_weights, self.num_attr_experts),
            ):
                if w is not None and balance_weight > 0 and e_bal > 1:
                    load = w.mean(dim=0)
                    load_mean = load.mean().clamp(min=1e-6)
                    load_std = load.std(unbiased=False)
                    loss = loss + balance_weight * ((load_std / load_mean) ** 2)
                if w is not None and entropy_weight > 0 and e_bal > 1:
                    entropy = -(w.clamp(min=1e-8) * w.clamp(min=1e-8).log()).sum(dim=-1)
                    entropy = entropy / torch.log(w.new_tensor(float(e_bal)))
                    loss = loss + entropy_weight * (1.0 - entropy.mean())
            if etf_weight > 0:
                if self._last_expert_outputs is not None:
                    loss = loss + etf_weight * expert_etf_loss(self._last_expert_outputs)
                if self._last_attr_expert_outputs is not None:
                    loss = loss + etf_weight * expert_etf_loss(self._last_attr_expert_outputs)
            return loss

        if self.use_region_gate and balance_abnormal_only:
            w = self._last_ab_routing_weights
            e_bal = self.num_abnormal
        else:
            w = self._last_dense_routing_weights
            if w is None:
                w = self._last_routing_weights
            e_bal = self.num_experts

        if w is not None and balance_weight > 0 and e_bal > 1:
            load = w.mean(dim=0)
            load_mean = load.mean().clamp(min=1e-6)
            load_std = load.std(unbiased=False)
            loss = loss + balance_weight * ((load_std / load_mean) ** 2)

        if w is not None and entropy_weight > 0 and e_bal > 1:
            entropy = -(w.clamp(min=1e-8) * w.clamp(min=1e-8).log()).sum(dim=-1)
            entropy = entropy / torch.log(w.new_tensor(float(e_bal)))
            loss = loss + entropy_weight * (1.0 - entropy.mean())

        if etf_weight > 0 and self._last_expert_outputs is not None:
            outs = self._last_expert_outputs
            if self.use_region_gate and outs.shape[2] > 1:
                outs = outs[:, :, 1:, :]
            loss = loss + etf_weight * expert_etf_loss(outs)

        return loss

    def project_layer_tokens(
        self,
        layer_feat: torch.Tensor,
        include_cls: bool = True,
        normal_bank: Optional[torch.Tensor] = None,
        class_ids: Optional[torch.Tensor] = None,
        cpa_context: Optional[dict] = None,
    ) -> torch.Tensor:
        b, _, d = layer_feat.shape
        cls_tok = layer_feat[:, :1, :]
        patch_tok = layer_feat[:, 1:, :]

        patch_out = self.forward(
            patch_tok,
            normal_bank=normal_bank,
            class_ids=class_ids,
            cpa_context=cpa_context,
        )

        cls_out = F.normalize(
            self.base_proj(cls_tok.reshape(-1, d)),
            dim=-1,
        ).view(b, 1, self.output_dim)

        if include_cls:
            return torch.cat([cls_out, patch_out], dim=1)
        return patch_out


class PerLayerMoEVisualProjection(nn.Module):
    """One MoECLIP-style DINO-space MoE adapter per DINO feature level."""

    def __init__(
        self,
        num_layers: int,
        vis_dim: int = 1024,
        output_dim: int = 768,
        num_experts: int = 4,
        use_learned_router: bool = True,
        router_hidden: int = 256,
        router_top_k: int = 2,
        router_type: str = SUPPORTED_ROUTER_TYPE,
        router_temperature: float = 1.0,
        expert_rank: int = 8,
        expert_alpha: int = 16,
        expert_dropout: float = 0.05,
        adapt_weight: float = 0.1,
        use_fofs: bool = True,
        normal_bank_top_k: int = 5,
        use_region_gate: bool = True,
        region_gate_mode: str = "feature",
        use_dual_path: bool = False,
        num_region_experts: Optional[int] = None,
        num_attr_experts: Optional[int] = None,
        router_attr_top_k: Optional[int] = None,
        dual_path_separate_norm: bool = False,
        dual_path_mode: str = "dual",
        **deprecated_kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.vis_dim = vis_dim
        self.output_dim = output_dim
        self.router_top_k = router_top_k
        self.router_type = router_type
        self.expert_rank = expert_rank
        self.expert_alpha = expert_alpha
        self.expert_dropout = expert_dropout
        self.adapt_weight = adapt_weight
        self.use_fofs = use_fofs
        self.normal_bank_top_k = normal_bank_top_k
        self.use_dual_path = bool(use_dual_path)
        self.use_region_gate = bool(use_region_gate) and not self.use_dual_path
        self.region_gate_mode = str(region_gate_mode).lower().strip()
        self.use_cpa_router = bool(
            (self.use_region_gate or self.use_dual_path)
            and self.region_gate_mode == "cpa"
        )
        self.num_region_experts = int(
            num_region_experts if num_region_experts is not None else num_experts
        )
        self.num_attr_experts = int(
            num_attr_experts
            if num_attr_experts is not None
            else (max(num_experts - 1, 1) if self.use_dual_path else 0)
        )
        self.router_attr_top_k = router_attr_top_k
        self.dual_path_separate_norm = bool(dual_path_separate_norm)
        self.dual_path_mode = str(dual_path_mode).lower().strip()

        layer_kwargs = dict(
            vis_dim=vis_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            use_learned_router=use_learned_router,
            router_hidden=router_hidden,
            router_top_k=router_top_k,
            router_type=router_type,
            router_temperature=router_temperature,
            expert_rank=expert_rank,
            expert_alpha=expert_alpha,
            expert_dropout=expert_dropout,
            adapt_weight=adapt_weight,
            use_fofs=use_fofs,
            normal_bank_top_k=normal_bank_top_k,
            use_region_gate=use_region_gate,
            region_gate_mode=self.region_gate_mode,
            use_dual_path=self.use_dual_path,
            num_region_experts=self.num_region_experts,
            num_attr_experts=self.num_attr_experts,
            router_attr_top_k=router_attr_top_k,
            dual_path_separate_norm=self.dual_path_separate_norm,
            dual_path_mode=self.dual_path_mode,
            **deprecated_kwargs,
        )
        self.layers = nn.ModuleList(
            [MoEVisualProjection(**layer_kwargs) for _ in range(num_layers)]
        )
        self.num_experts = self.layers[0].num_experts
        self.num_abnormal = self.layers[0].num_abnormal
        print(
            f"PerLayerMoEVisualProjection MoECLIP-style: "
            f"{num_layers} layers × region={self.num_experts}"
            f"{f'+attr={self.num_attr_experts}' if self.use_dual_path else ''} experts, "
            f"rank={expert_rank}, alpha={expert_alpha}, "
            f"{vis_dim}→{output_dim}, top_k={router_top_k}, "
            f"adapt_weight={adapt_weight}, use_fofs={use_fofs}, "
            f"dual_path={self.use_dual_path}, region_gate={self.use_region_gate}, "
            f"gate_mode={self.region_gate_mode}, "
            f"separate_norm={self.dual_path_separate_norm}, "
            f"path_mode={self.dual_path_mode}"
        )

    def set_dual_path_mode(self, mode: str) -> None:
        """Switch eval-time path intervention without changing checkpoint weights."""
        mode = str(mode).lower().strip()
        valid = {"base", "region", "attr_gated", "attr_force", "dual", "dual_force"}
        if mode not in valid:
            raise ValueError(f"dual_path_mode must be one of {sorted(valid)}, got {mode!r}")
        self.dual_path_mode = mode
        for layer in self.layers:
            layer.dual_path_mode = mode

    def project_layer_tokens(
        self,
        layer_idx: int,
        layer_feat: torch.Tensor,
        include_cls: bool = True,
        normal_bank: Optional[torch.Tensor] = None,
        class_ids: Optional[torch.Tensor] = None,
        cpa_context: Optional[dict] = None,
    ) -> torch.Tensor:
        return self.layers[layer_idx].project_layer_tokens(
            layer_feat, include_cls=include_cls,
            normal_bank=normal_bank, class_ids=class_ids,
            cpa_context=cpa_context,
        )

    def routing_regularization_loss(
        self,
        balance_weight: float = 0.0,
        entropy_weight: float = 0.0,
        etf_weight: float = 0.0,
        balance_abnormal_only: bool = True,
    ) -> torch.Tensor:
        losses = [
            layer.routing_regularization_loss(
                balance_weight,
                entropy_weight,
                etf_weight,
                balance_abnormal_only=balance_abnormal_only,
            )
            for layer in self.layers
        ]
        return sum(losses) / max(len(losses), 1)


MoEPatchProjection = Union[MoEVisualProjection, PerLayerMoEVisualProjection]


def is_moe_patch_proj(module: nn.Module) -> bool:
    return isinstance(module, (MoEVisualProjection, PerLayerMoEVisualProjection)) or bool(
        getattr(module, "_is_per_layer_projection", False)
    )


def is_per_layer_moe(module: nn.Module) -> bool:
    return isinstance(module, PerLayerMoEVisualProjection) or bool(
        getattr(module, "_is_per_layer_projection", False)
    )


def load_patch_proj_state_dict(patch_proj: nn.Module, state_dict: dict, strict: bool = True):
    """Load weights; replicate single-layer ckpt into every per-layer copy."""
    if is_per_layer_moe(patch_proj):
        if any(k.startswith("layers.") for k in state_dict):
            return patch_proj.load_state_dict(state_dict, strict=strict)
        for layer in patch_proj.layers:
            layer.load_state_dict(state_dict, strict=strict)
        return None
    return patch_proj.load_state_dict(state_dict, strict=strict)
