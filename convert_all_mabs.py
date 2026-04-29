import os
import pandas as pd
from convert_residue_level import analyze_fragmap_for_domain

# === CONFIG ===
mab_root = "/projects/bdtk/mcabreza/mAbs"  # folder containing all mAbs
output_root = "/projects/bdtk/mcabreza/ResidueLevel_Outputs"  # output lives here
os.makedirs(output_root, exist_ok=True)

summary = []  # will track each mAb status

# === LOOP THROUGH ALL MABS ===
for mab_name in sorted(os.listdir(mab_root)):
    mab_path = os.path.join(mab_root, mab_name)
    if not os.path.isdir(mab_path):
        continue

    # Search for silcs_fragmaps_* folder
    fragmap_folder = None
    for root, dirs, _ in os.walk(mab_path):
        for d in dirs:
            if d.startswith("silcs_fragmaps_"):
                fragmap_folder = os.path.join(root, d)
                break
        if fragmap_folder:
            break

    if not fragmap_folder:
        print(f"[SKIP] No silcs_fragmaps_ folder found for {mab_name}")
        summary.append((mab_name, "no_fragmaps"))
        continue

    output_file = os.path.join(output_root, f"{mab_name}_ResidueLevel.csv")

    if os.path.exists(output_file):
        print(f"[SKIP] {mab_name} already processed.")
        summary.append((mab_name, "skipped"))
        continue

    try:
        print(f"[RUN] Processing: {mab_name}")
        df = analyze_fragmap_for_domain(fragmap_folder)
        df.to_csv(output_file, index=False)
        print(f"✅ Saved CSV: {output_file}")
        summary.append((mab_name, "success"))
    except Exception as e:
        print(f"[ERROR] {mab_name} failed: {e}")
        summary.append((mab_name, f"error: {e}"))

# === SAVE SUMMARY CSV ===
summary_df = pd.DataFrame(summary, columns=["mAb", "status"])
summary_path = os.path.join(output_root, "batch_summary.csv")
summary_df.to_csv(summary_path, index=False)

print(f"\n🎯 Batch run complete! Summary saved at:\n{summary_path}")
