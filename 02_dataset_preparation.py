
# ============================================================
# RWF-2000 DOWNLOAD + SAFE UTF-8 EXTRACTION
#
# Download only if archive does not exist.
# Extract directly into /content/RWF-2000.
#
# Final structure:
# /content/RWF-2000/
# ├── train/
# │   ├── Fight/
# │   └── NonFight/
# └── val/
#     ├── Fight/
#     └── NonFight/
# ============================================================

from pathlib import Path
from zipfile import ZipFile
import importlib.util
import subprocess
import sys
import shutil

# ------------------------------------------------------------
# 1. INSTALL KAGGLEHUB IF REQUIRED
# ------------------------------------------------------------

if importlib.util.find_spec("kagglehub") is None:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "kagglehub"
    ])

import kagglehub

# ------------------------------------------------------------
# 2. DEFINE DATASET PATHS
# ------------------------------------------------------------

ARCHIVE = Path(
    "/root/.cache/kagglehub/datasets/"
    "vulamnguyen/rwf2000/1.archive"
)

OUTPUT = Path("/content/RWF-2000")

EXPECTED_TOTAL = 2000

# ------------------------------------------------------------
# 3. DOWNLOAD ONLY IF ARCHIVE DOES NOT EXIST
# ------------------------------------------------------------

if ARCHIVE.is_file() and ARCHIVE.stat().st_size > 0:

    print("Existing KaggleHub archive found.")
    print("Skipping download:", ARCHIVE)

else:

    print("Dataset archive not found.")
    print("Downloading RWF-2000 using KaggleHub...")

    try:
        kagglehub.dataset_download("vulamnguyen/rwf2000")

    except OSError as error:

        # KaggleHub may download the archive successfully
        # but fail during automatic extraction.

        print("\nKaggleHub automatic extraction failed.")
        print("Reason:", error)

    assert (
        ARCHIVE.is_file()
        and ARCHIVE.stat().st_size > 0
    ), (
        "Dataset archive download failed or archive "
        "is unavailable."
    )

    print("\nDataset archive downloaded successfully.")

# ------------------------------------------------------------
# 4. CHECK WHETHER DATASET IS ALREADY EXTRACTED
# ------------------------------------------------------------

def dataset_is_ready(root):

    expected_folders = [
        root / "train" / "Fight",
        root / "train" / "NonFight",
        root / "val" / "Fight",
        root / "val" / "NonFight",
    ]

    # All four class directories must exist.
    if not all(folder.is_dir() for folder in expected_folders):
        return False

    # Count files in all four directories.
    total = sum(
        1
        for folder in expected_folders
        for file in folder.iterdir()
        if file.is_file()
    )

    return total == EXPECTED_TOTAL


if dataset_is_ready(OUTPUT):

    print("\nDataset already exists and is complete.")
    print("Skipping extraction:", OUTPUT)

else:

    # --------------------------------------------------------
    # 5. PREPARE OUTPUT DIRECTORY
    # --------------------------------------------------------

    print("\nPreparing output directory...")

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Expected video counts.
    counts = {
        "train/Fight": 0,
        "train/NonFight": 0,
        "val/Fight": 0,
        "val/NonFight": 0,
    }

    # --------------------------------------------------------
    # 6. EXTRACT VIDEOS WITH SAFE FILENAMES
    # --------------------------------------------------------

    print("\nExtracting RWF-2000 archive...")
    print("Using UTF-8 filename decoding.")

    with ZipFile(
        ARCHIVE,
        mode="r",
        metadata_encoding="utf-8"
    ) as zf:

        for info in zf.infolist():

            if info.is_dir():
                continue

            # Expected archive structure:
            # RWF-2000/train/Fight/video.avi
            # RWF-2000/train/NonFight/video.avi
            # RWF-2000/val/Fight/video.avi
            # RWF-2000/val/NonFight/video.avi

            parts = info.filename.split("/")

            if (
                len(parts) != 4
                or parts[0] != "RWF-2000"
            ):
                continue

            split = parts[1]
            category = parts[2]

            key = f"{split}/{category}"

            if key not in counts:
                continue

            # Preserve video extension.
            extension = (
                Path(parts[3]).suffix.lower() or ".avi"
            )

            # Generate short, unique filenames.
            counts[key] += 1

            safe_name = (
                f"{category.lower()}_"
                f"{counts[key]:04d}{extension}"
            )

            destination_dir = OUTPUT / split / category

            destination_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            destination = destination_dir / safe_name

            # Stream video contents directly from ZIP.
            # Original long filename is never created on disk.
            with zf.open(info, "r") as source:

                with destination.open("wb") as target:

                    shutil.copyfileobj(
                        source,
                        target
                    )

            total_extracted = sum(counts.values())

            if total_extracted % 100 == 0:
                print(
                    f"Extracted "
                    f"{total_extracted}/{EXPECTED_TOTAL} videos"
                )

    # --------------------------------------------------------
    # 7. VERIFY EXTRACTION COUNTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("RWF-2000 EXTRACTION REPORT")
    print("=" * 65)

    for key, count in counts.items():
        print(f"{key}: {count} videos")

    total = sum(counts.values())

    print("\nTotal extracted videos:", total)

    assert all(count > 0 for count in counts.values()), (
        "One or more dataset folders are empty."
    )

    assert total == EXPECTED_TOTAL, (
        f"Expected {EXPECTED_TOTAL} videos, "
        f"but extracted {total}."
    )

# ------------------------------------------------------------
# 8. FINAL DIRECTORY AND VIDEO VERIFICATION
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FINAL DATASET VERIFICATION")
print("=" * 65)

final_total = 0

for split in ("train", "val"):

    for category in ("Fight", "NonFight"):

        folder = OUTPUT / split / category

        assert folder.is_dir(), (
            f"Missing directory: {folder}"
        )

        videos = [
            file
            for file in folder.iterdir()
            if file.is_file()
        ]

        video_count = len(videos)

        final_total += video_count

        print(
            f"{split}/{category}: "
            f"{video_count} videos"
        )

assert final_total == EXPECTED_TOTAL, (
    f"Expected {EXPECTED_TOTAL} videos, "
    f"found {final_total}."
)

# ------------------------------------------------------------
# 9. FINAL SUCCESS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("RWF-2000 DATASET READY")
print("=" * 65)

print("Dataset location:", OUTPUT)
print("Total videos:", final_total)
print("Storage: Colab local storage")

print("""
Final directory:

/content/RWF-2000/
├── train/
│   ├── Fight/
│   └── NonFight/
└── val/
    ├── Fight/
    └── NonFight/
""")

print("SUCCESS: Dataset download/extraction and verification completed.")
