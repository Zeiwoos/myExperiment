import json
import os

def clean_split(split_dict, prefix):
    removed = 0
    kept = 0
    new_split = {}

    for cls, samples in split_dict.items():
        new_list = []
        for s in samples:
            img_name = os.path.basename(s.get("img_path", ""))
            if img_name.startswith(prefix):
                removed += 1
            else:
                new_list.append(s)
                kept += 1
        new_split[cls] = new_list

    return new_split, kept, removed


def main(meta_path, prefix="songdong_"):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    total_removed = 0
    total_kept = 0

    for split in ["train", "test"]:
        if split in meta:
            meta[split], kept, removed = clean_split(meta[split], prefix)
            total_removed += removed
            total_kept += kept
            print(f"[{split}] kept={kept}, removed={removed}")

    out_path = meta_path.replace(".json", ".no_songdong.json")
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
