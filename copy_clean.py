# COMMAND TO RUN: python3 copy_clean.py sheet.csv

#!/usr/bin/env python3
import argparse, csv, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Downloaded sheet as CSV")
    ap.add_argument("--src", default="/work/hdd/bdtk/iwu/DONE_mAb_batch5")
    ap.add_argument("--dst", default="/projects/bdtk/mcabreza/mAbs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # assume header exists
        # Expect A=Name, E=clean
        for row in reader:
            if not row:
                continue
            name = (row[0] if len(row) >= 1 else "").strip().strip('"')
            clean = (row[4] if len(row) >= 5 else "").strip().upper()
            if not name or clean != "V":
                continue

            src = src_root / name
            dst = dst_root / name
            if not src.is_dir():
                print(f"SKIP (missing): {src}", file=sys.stderr)
                continue

            cmd = ["rsync", "-aHAX", "--info=progress2", "--partial", "--inplace", f"{src}/", str(dst)]
            if args.dry_run:
                cmd.insert(1, "-n")
            print(" ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"FAIL ({e.returncode}): {src} -> {dst}", file=sys.stderr)

if __name__ == "__main__":
    main()
