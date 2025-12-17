import os
import random
from pathlib import Path
import argparse


# 在此处指定根目录；也可运行时用 --root 覆盖
ROOT_DIR = Path(".").resolve()
SPLIT_RATIO = 0.9  # 训练集比例
RANDOM_SEED = 42


def write_list(path: Path, items):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(f"{it}\n")


def main(root_dir: Path):
    """
    列出 root_dir 下的所有子目录作为 case_names，按 9:1 (可通过 SPLIT_RATIO 调整) 划分训练/测试，
    并写入 root_dir：
      - all_cases.txt
      - train_cases.txt
      - test_cases.txt
    """
    if not root_dir.exists():
        print(f"[Error] root_dir not found: {root_dir}")
        return
    if not root_dir.is_dir():
        print(f"[Error] root_dir is not a directory: {root_dir}")
        return

    case_names = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])

    if not case_names:
        print(f"No case folders found under: {root_dir}")
        return

    rnd = random.Random(RANDOM_SEED)
    rnd.shuffle(case_names)

    n_train = max(1, int(len(case_names) * SPLIT_RATIO)) if len(case_names) > 1 else 1
    train_cases = case_names[:n_train]
    test_cases = case_names[n_train:]

    write_list(root_dir / "all_cases.txt", case_names)
    write_list(root_dir / "train_cases.txt", train_cases)
    write_list(root_dir / "test_cases.txt", test_cases)

    print(
        f"Total cases: {len(case_names)} (train={len(train_cases)}, test={len(test_cases)}), "
        f"saved under {root_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split case folders into train/test lists.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(ROOT_DIR),
        help="Root directory containing case folders (default: ROOT_DIR variable).",
    )
    args = parser.parse_args()
    main(Path(args.root).resolve())
