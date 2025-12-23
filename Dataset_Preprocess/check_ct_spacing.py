import os
import csv
import SimpleITK as sitk

# 根目录：包含多个 case_name，每个 case 下有 pre/ 和 intra/ 子目录，每个子目录下有 CT 文件夹
CT_ROOT = r"D:\data_space\ZhongriSecond\wjy 1-11"
# 输出文件
REPORT_CSV = "ct_spacing_report.csv"
FAILED_TXT = "ct_spacing_failed_cases.txt"


def scan_ct_files(root_dir):
    rows = []
    failed = []

    if not os.path.isdir(root_dir):
        print(f"[Error] CT_ROOT not found: {root_dir}")
        return rows, failed

    case_names = sorted(
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    )
    print(f"[Info] Found {len(case_names)} cases under {root_dir}")

    for case in case_names:
        for phase in ["pre", "intra"]:
            for mode in ["ct", "bone", "std"]:
                ct_dir = os.path.join(root_dir, case, phase, "CT", mode)
                if not os.path.exists(ct_dir):
                    continue
                nii_files = sorted(os.listdir(ct_dir))
                for fname in nii_files:
                    fpath = os.path.join(ct_dir, fname)
                    try:    
                        img = sitk.ReadImage(fpath)
                        spacing = img.GetSpacing()  # (x, y, z)
                        size = img.GetSize()        # (x, y, z)
                        min_spacing = min(spacing)
                        status = "fail" if min_spacing > 2.0 else "ok"
                        rows.append({
                            "case": case,
                            "phase": phase,
                            "dir": ct_dir,
                            "file": fname,
                            "spacing_x": spacing[0],
                            "spacing_y": spacing[1],
                            "spacing_z": spacing[2],
                            "size_x": size[0],
                            "size_y": size[1],
                            "size_z": size[2],
                            "min_spacing": min_spacing,
                            "status": status,
                        })
                        if status == "fail":
                            failed.append({
                                "case": case,
                                "phase": phase,
                                "dir": ct_dir,
                                "file": fname,
                                "spacing": spacing,
                                "size": size,
                            })
                    except Exception as e:
                        print(f"[Error] Read failed: {fpath}, err={e}")
                        failed.append({
                            "case": case,
                            "phase": phase,
                            "dir": ct_dir,
                            "file": fname,
                            "spacing": ("error",),
                            "size": ("error",),
                            "err": str(e),
                        })
    return rows, failed


def save_report(rows, failed):
    # CSV 报告
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case", "phase", "dir", "file",
                "spacing_x", "spacing_y", "spacing_z",
                "size_x", "size_y", "size_z",
                "min_spacing", "status",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # 失败列表
    with open(FAILED_TXT, "w", encoding="utf-8") as f:
        f.write("# Cases/CT with min_spacing > 2.0 or read error\n")
        for item in failed:
            f.write(
                f"{item.get('case','?')}\t{item.get('phase','?')}\t{item.get('dir','?')}\t"
                f"{item.get('file','?')}\t{item.get('spacing')}\t{item.get('size')}\t"
                f"{item.get('err','')}\n"
            )

    print(f"[Info] Report saved: {REPORT_CSV}")
    print(f"[Info] Failed list saved: {FAILED_TXT} (count={len(failed)})")


if __name__ == "__main__":
    rows, failed = scan_ct_files(CT_ROOT)
    save_report(rows, failed)
