#!/usr/bin/env python3
"""Cloud–edge full comparison report (224×224 unified, 970-image standard split).

Schemes (all on the 970-image `mvtec_anomaly_llm` test split, 15 categories,
image input unified to 224×224):

  - sft_edge    : standalone edge = Qwen3.5-0.8B + reason LoRA (Yes/No + reason)
  - fusion      : standalone cloud = DINOv3 (ViT-L) + Qwen3.5-2B fusion
                  (Cloud-abnormal-cx "traditional + model combined" scheme)
  - collab_sft  : cloud–edge link = SFT edge detection + kNN-score CRR routing
                  + cloud DINOv3 + Qwen3.5 fusion review

Detection metrics use per-category macro F1. The fusion scheme emits a
continuous anomaly score, so it is reported with AUROC / AP / F1-max
(threshold-optimal); the generative/routed schemes are binary decisions and use
fixed-operating-point F1 / precision / recall / accuracy.

FLOPs are per-image at 224×224 (patch 16 → 196 pre-merge tokens). Vision-tower
FLOPs are FlopCounterMode-measured (0.8B) or analytic ViT formula; LLM FLOPs use
2·N per token.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "reports" / "cloud_edge_compare"
CX_OUT = ROOT / "outputs" / "cloud_abnormal_cx_224"

DOMAIN = {"screw", "cable", "pill", "capsule", "zipper"}
N_TEST = 970


def _load(p: Path) -> dict[str, Any]:
    if not p.exists():
        print(f"[warn] missing {p}")
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and np.isnan(x):
        return "—"
    return f"{x:.{nd}f}"


def _fmt_ms(a: dict | None) -> str:
    if not a:
        return "—"
    return f"{a['mean']:.1f} (p95 {a['p95']:.1f})"


def _macro_f1(per_cat: dict[str, dict[str, Any]], key: str = "f1") -> float | None:
    vals = [v.get(key) for v in per_cat.values() if v.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def _macro_from_rows(rows: list[dict[str, Any]], p_key: str) -> dict[str, float]:
    by: dict[str, list[int]] = {}
    for r in rows:
        by.setdefault(r["category"], []).append((r["label"], r[p_key]))
    fs, ps, rs, ac = [], [], [], []
    for c in sorted(by):
        yt = [a for a, _ in by[c]]
        yp = [p for _, p in by[c]]
        fs.append(f1_score(yt, yp, zero_division=0))
        ps.append(_prec(yt, yp))
        rs.append(_rec(yt, yp))
        ac.append(float(np.mean([a == p for a, p in by[c]])))
    return {
        "f1": float(np.mean(fs)),
        "precision": float(np.mean(ps)),
        "recall": float(np.mean(rs)),
        "accuracy": float(np.mean(ac)),
    }


def _prec(yt: list[int], yp: list[int]) -> float:
    tp = sum(a == p == 1 for a, p in zip(yt, yp))
    fp = sum(a == 0 and p == 1 for a, p in zip(yt, yp))
    return tp / (tp + fp) if (tp + fp) else 0.0


def _rec(yt: list[int], yp: list[int]) -> float:
    tp = sum(a == p == 1 for a, p in zip(yt, yp))
    fn = sum(a == 1 and p == 0 for a, p in zip(yt, yp))
    return tp / (tp + fn) if (tp + fn) else 0.0


def main() -> int:
    sft = _load(OUT / "sft_edge.json")
    fusion = _load(CX_OUT / "mvtec_llm_2b_fusion_metrics.json")
    collab = _load(OUT / "collab_sft.json")

    sm = sft.get("macro") or {}
    collab_rows = collab.get("rows") or []
    csum = collab.get("summary") or {}

    # fusion image-level macro (threshold-free AUROC/AP + threshold-optimal F1-max)
    fm = (fusion.get("overall") or {}).get("macro_average", {}).get("image_level", {})
    fusion_macro = {
        "auroc": fm.get("auroc"),
        "ap": fm.get("ap"),
        "f1_max": fm.get("f1_max"),
    }

    collab_macro = _macro_from_rows(collab_rows, "final_pred")
    collab_edge_macro = _macro_from_rows(collab_rows, "edge_pred")

    print("\n# 云边协同 vs 单独云 vs 单独边（224×224 统一，970 图 / 15 类）\n")

    print("## 1. 检测指标（宏平均，15 类）\n")
    print("| 方案 | Macro-F1 | Precision | Recall | Accuracy |")
    print("|------|----------|-----------|--------|----------|")
    print(f"| 单独边 (SFT 生成式, 224) | {_fmt(sm.get('f1'))} | {_fmt(sm.get('precision'))} | {_fmt(sm.get('recall'))} | {_fmt(sm.get('accuracy'))} |")
    print(f"| 云边协同 (SFT边+kNN路由) | {_fmt(collab_macro['f1'])} | {_fmt(collab_macro['precision'])} | {_fmt(collab_macro['recall'])} | {_fmt(collab_macro['accuracy'])} |")
    print(f"| └ 协同前 SFT 边缘基线 | {_fmt(collab_edge_macro['f1'])} | {_fmt(collab_edge_macro['precision'])} | {_fmt(collab_edge_macro['recall'])} | {_fmt(collab_edge_macro['accuracy'])} |")
    print()
    print("| 方案 | AUROC | AP | F1-max |")
    print("|------|-------|----|--------|")
    print(f"| 单独云 (DINOv3+Qwen3.5-2B 融合, 224) | {_fmt(fusion_macro['auroc'])} | {_fmt(fusion_macro['ap'])} | {_fmt(fusion_macro['f1_max'])} |")

    print("\n> 说明：单独云（融合）输出连续异常分数，故按异常检测惯例报 **AUROC / AP / F1-max（阈值最优）**，\n"
          "> 而不是固定阈值的 F1/P/R/Acc。其 F1-max 与生成式/路由式的固定决策 F1 不可直接等同比较。\n")

    print("## 2. 基础指标（模型 / 计算 / 显存 / 速度）\n")
    sb = sft.get("base") or {}
    fusion_mean_s = (fusion.get("overall") or {}).get("mean_time_seconds")
    fusion_ms = f"{fusion_mean_s*1000:.0f}" if fusion_mean_s else "—"
    print("| 指标 | 单独边 SFT | 单独云 融合 | 云边协同 |")
    print("|------|-----------|------------|----------|")
    print(f"| 参数量 (M) | 860.3 (视觉塔 100.6 + LLM 759.7) | 2516.4 (DINOv3 303.1 + Qwen3.5-2B 2213.2) | 边 860.3 + 云 4454.3 |")
    print(f"| 单图视觉塔 FLOPs (G) | 36.4 | DINOv3 122 + Qwen2B 视觉 162×4图 | 同左边 |")
    print(f"| LLM FLOPs/生成 token (G) | 1.52 | 3.60 | 边 1.52 / 云 8.91 |")
    print(f"| 单图总 FLOPs (G) | ≈ 207 (0.21 T) | ≈ 2500–2900 (2.5–2.9 T) | ≈ 243 边 + 950 云(仅复核图) |")
    print(f"| 峰值显存 (MB) | {_fmt(sb.get('peak_mem_mb'), 0)} | 4800 | 边 1699 / 云 8923 |")
    print(f"| 单图延迟 (ms) | 1428 (p95 1915) | ≈ {fusion_ms} | 本地 1450 / 上云 3379 |")

    print("\n## 3. 云边协同开销与延迟\n")
    s = csum
    if s:
        print("| 指标 | 值 |")
        print("|------|----|")
        print(f"| 上传率 | {s['upload_rate']*100:.1f}% |")
        print(f"| 云端复核率 | {s['cloud_review_rate']*100:.1f}% |")
        print(f"| 云端实际调用率 | {s['cloud_used_rate']*100:.1f}% |")
        print(f"| 边缘 kNN 分数延迟 (ms) | {_fmt_ms(s.get('knn_ms'))} |")
        print(f"| 边缘 SFT 生成延迟 (ms) | {_fmt_ms(s.get('sft_ms'))} |")
        print(f"| 网络延迟 (ms, 上传时) | {_fmt_ms(s.get('net_ms_when_upload'))} |")
        print(f"| 云端推理延迟 (ms, 复核时) | {_fmt_ms(s.get('cloud_ms_when_review'))} |")
        print(f"| 端到端延迟 (ms, 本地判定) | {_fmt_ms(s.get('total_ms_local'))} |")
        print(f"| 端到端延迟 (ms, 上云复核) | {_fmt_ms(s.get('total_ms_cloud_review'))} |")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
