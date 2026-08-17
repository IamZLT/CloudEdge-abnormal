#!/usr/bin/env python3
"""Split mvtec_zero_shot.json (LLM annotations) to match the mvtec_anomaly_llm
train/test image split.

Reads the manifests produced by build_mvtec_llm_split.py (train.jsonl / test.jsonl)
to learn which raw image went to which split, then partitions the zero-shot
conversation annotations accordingly.

Output (JSON arrays, same entry schema + `split` field):
    datasets/mvtec_anomaly_llm/mvtec_zero_shot_train.json
    datasets/mvtec_anomaly_llm/mvtec_zero_shot_test.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ZS = ROOT / "datasets/mvtec_anomaly_detection/mvtec_zero_shot.json"
DEFAULT_OUT = ROOT / "datasets/mvtec_anomaly_llm"
SRC_ROOT = "/data2/zlt/datasets/anomaly_detection/mvtec_anomaly_detection/"
ZS_PREFIX = "mvtec_anomaly_detection/"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-shot", default=str(DEFAULT_ZS))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    train_rows = load_jsonl(out_root / "train.jsonl")
    test_rows = load_jsonl(out_root / "test.jsonl")

    # raw relative path -> ('train'|'test')
    assign: dict[str, str] = {}
    for r in train_rows + test_rows:
        src = r["source"]
        assert src.startswith(SRC_ROOT), src
        assign[src[len(SRC_ROOT):]] = r["split"]

    zero_shot = json.loads(Path(args.zero_shot).read_text(encoding="utf-8"))

    train_out: list[dict] = []
    test_out: list[dict] = []
    unmapped = 0

    for e in zero_shot:
        img = e["image"]
        assert img.startswith(ZS_PREFIX), img
        rel = img[len(ZS_PREFIX):]  # {cat}/{mvtc_split}/{defect}/{file}
        split = assign.get(rel)
        if split is None:
            unmapped += 1
            continue
        cat, _, sub, fname = rel.split("/", 3)
        new_img = f"{split}/{cat}/{sub}/{fname}"
        out = {
            "id": e["id"],
            "image": new_img,
            "image_orig": img,
            "split": split,
            "conversations": e["conversations"],
            "metadata": e["metadata"],
        }
        (train_out if split == "train" else test_out).append(out)

    train_path = out_root / "mvtec_zero_shot_train.json"
    test_path = out_root / "mvtec_zero_shot_test.json"
    train_path.write_text(json.dumps(train_out, ensure_ascii=False, indent=2), encoding="utf-8")
    test_path.write_text(json.dumps(test_out, ensure_ascii=False, indent=2), encoding="utf-8")

    def count(rows: list[dict]):
        n = len(rows)
        imgs = len({r["image"] for r in rows})
        ng = len({r["image"] for r in rows if r["metadata"].get("anomaly")})
        ok = imgs - ng
        return n, imgs, ok, ng

    tr_n, tr_img, tr_ok, tr_ng = count(train_out)
    te_n, te_img, te_ok, te_ng = count(test_out)

    print(f"unmapped entries: {unmapped}")
    print(f"TRAIN: entries={tr_n}, images={tr_img} (OK={tr_ok}, NG={tr_ng}) -> {train_path}")
    print(f"TEST : entries={te_n}, images={te_img} (OK={te_ok}, NG={te_ng}) -> {test_path}")
    print(f"total entries={tr_n + te_n}, total images={tr_img + te_img}")


if __name__ == "__main__":
    main()
