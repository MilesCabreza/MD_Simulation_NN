import os
import subprocess

# Path to mAbs folder
BASE_DIR = r"/projects/bdtk/mcabreza/mAbs"

# Path to your map2dx converter
MAP2DX_SCRIPT = r"/projects/bdtk/mcabreza/map2dx.py"

converted_count = 0
skipped_count = 0
failed_count = 0

for mab_name in sorted(os.listdir(BASE_DIR)):
    mab_path = os.path.join(BASE_DIR, mab_name)
    if not os.path.isdir(mab_path):
        continue

    print(f"\n🧬 Processing antibody: {mab_name}")

    for root, _, files in os.walk(mab_path):
        for file in files:
            if not file.endswith(".map"):
                continue

            map_path = os.path.join(root, file)
            dx_path = os.path.splitext(map_path)[0] + ".dx"

            if os.path.exists(dx_path):
                print(f"⏭️  Skipping (already exists): {os.path.basename(dx_path)}")
                skipped_count += 1
                continue

            print(f"➡️  Converting: {map_path}")
            try:
                result = subprocess.run(
                    ["python3", MAP2DX_SCRIPT, map_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                if os.path.exists(dx_path):
                    print(f"✅ Created: {dx_path}")
                    converted_count += 1
                else:
                    print(f"⚠️  Conversion ran but no .dx file created: {file}")
                    failed_count += 1
            except subprocess.CalledProcessError as e:
                print(f"❌ Error converting {file}:")
                print(e.stderr or e.stdout)
                failed_count += 1

print("\n==============================")
print(f"✅ Conversion complete")
print(f"   ➕ {converted_count} new .dx files")
print(f"   ⏭️  {skipped_count} skipped (already existed)")
print(f"   ❌ {failed_count} failed conversions")
print("==============================")
