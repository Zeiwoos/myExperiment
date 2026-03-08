import json
import os
import random

RANDOM_SEED = 42
KEEP_RATIO = 0.5   # 保留一半正常样本

def process_split(split_dict):
    new_split = {}
    removed = 0
    kept = 0

    for cls, samples in split_dict.items():
        normals = [s for s in samples if s.get("anomaly", 0) == 0]
        abnormals = [s for s in samples if s.get("anomaly", 0) == 1]

        random.shuffle(normals)
        keep_n = int(len(normals) * KEEP_RATIO)

        keep_normals = normals[:keep_n]

        removed += len(normals) - keep_n
        kept += keep_n + len(abnormals)

        new_split[cls] = keep_normals + abnormals

    return new_split, kept, removed


def main(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    random.seed(RANDOM_SEED)

    total_removed = 0
    total_kept = 0

    for split in ["test"]:
        if split in meta:
            meta[split], kept, removed = process_split(meta[split])
            total_removed += removed
            total_kept += kept
            print(f"[{split}] kept={kept}, removed_normals={removed}")

    out_path = meta_path.replace(".json", ".half_normal.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("===================================")
    print(f"TOTAL kept    : {total_kept}")
    print(f"TOTAL removed : {total_removed}")
    print(f"Output saved  : {out_path}")
    print("===================================")


if __name__ == "__main__":
    meta_path = input("请输入 meta.json 路径: ").strip()
    assert os.path.isfile(meta_path), "meta.json 不存在"
    main(meta_path)
