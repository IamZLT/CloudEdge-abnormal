from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def torch_dtype(name: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


class DinoV3Encoder:
    """DINOv3 wrapper supporting both a Hugging Face directory and native .pth weights."""

    def __init__(
        self,
        model_path: str,
        source_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        image_size: int = 448,
        layers: list[int] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = torch_dtype(dtype)
        self.image_size = image_size
        self.layers = layers or [6, 12, 18, 24]
        self.patch_size = 16
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"DINOv3 weights not found: {path}")
        pth_files = sorted(path.glob("*.pth")) if path.is_dir() else ([path] if path.suffix == ".pth" else [])
        if pth_files:
            source_parent = str(Path(source_path).resolve().parent)
            if source_parent not in sys.path:
                sys.path.insert(0, source_parent)
            from dinov3.hub.backbones import dinov3_vitl16

            self.model = dinov3_vitl16(pretrained=True, weights=str(pth_files[0]))
            self.backend = "native"
        else:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(
                str(path), local_files_only=True, trust_remote_code=True, dtype=self.dtype
            )
            self.backend = "transformers"
            self.patch_size = int(getattr(self.model.config, "patch_size", 16))
        self.model.to(device=self.device, dtype=self.dtype).eval()
        self.model.requires_grad_(False)
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("DINOv3 parameter freezing failed")

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        tensor = TF.to_tensor(image)
        tensor = TF.resize(
            tensor, [self.image_size, self.image_size], interpolation=InterpolationMode.BICUBIC, antialias=True
        )
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        return tensor.unsqueeze(0).to(self.device, dtype=self.dtype)

    @torch.inference_mode()
    def encode(self, image: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
        x = self.preprocess(image)
        grid = (self.image_size // self.patch_size, self.image_size // self.patch_size)
        if self.backend == "native":
            # Config uses human-readable 1-based block numbers; native API is 0-based.
            native_layers = [max(0, int(layer) - 1) for layer in self.layers]
            features = self.model.get_intermediate_layers(x, n=native_layers, reshape=True, norm=True)
            patches = [f.flatten(2).transpose(1, 2) for f in features]
        else:
            output = self.model(x, output_hidden_states=True, return_dict=True)
            hidden = output.hidden_states
            patches = []
            patch_count = grid[0] * grid[1]
            for layer in self.layers:
                index = min(max(int(layer), 1), len(hidden) - 1)
                patches.append(hidden[index][:, -patch_count:, :])
        patches = [F.normalize(p.float(), dim=-1) for p in patches]
        # All ViT-L layers have the same channel space. Averaging retains the
        # multi-depth signal without making every kNN comparison 4x larger.
        embedding = F.normalize(torch.stack(patches, dim=0).mean(dim=0), dim=-1)
        return embedding.squeeze(0).cpu(), grid


def load_mask(path: Path | None, size: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    mask = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)
