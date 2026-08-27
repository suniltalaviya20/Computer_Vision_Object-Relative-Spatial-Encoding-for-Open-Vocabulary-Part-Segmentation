from pathlib import Path
import shutil

import gdown


FILE_ID = "1f5kqrM2_iK_bWmQBW3rdSnGrnke4PUbX"

project_root = Path(__file__).resolve().parents[1]
raw_dir = project_root / "data" / "raw"
archive = raw_dir / "PascalPart116.tar.gz"

raw_dir.mkdir(parents=True, exist_ok=True)

print("Downloading Pascal-Part-116...")
gdown.download(id=FILE_ID, output=str(archive), quiet=False)

print("Extracting dataset...")
shutil.unpack_archive(archive, raw_dir)
archive.unlink()

print(f"Dataset extracted under: {raw_dir}")