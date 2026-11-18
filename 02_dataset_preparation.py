
#!/usr/bin/env python3

"""
RWF-2000 Dataset Preparation

Expected dataset structure:

/content/RWF-2000/
    train/
        Fight/       800 videos
        NonFight/    800 videos
    val/
        Fight/       200 videos
        NonFight/    200 videos

The script:
1. Checks whether the final dataset is already complete.
2. Reuses the KaggleHub archive if it exists.
3. Downloads only when the archive is missing.
4. Extracts videos safely using sequential filenames.
5. Verifies the final dataset.
"""

import os
import shutil
import zipfile
from pathlib import Path
from collections import Counter
from zipfile import ZipFile

# ============================================================
# 1. CONFIGURATION
# ============================================================

REPO_DIR = Path("/content/Vi-SAFE-3B42-Implementation")

OUTPUT_DIR = Path("/content/RWF-2000")

TEMP_DIR = Path("/content/rwf2000_extraction")

KAGGLE_CACHE_DIR = Path(
    "/root/.cache/kagglehub/datasets/vulamnguyen/rwf2000"
)

ARCHIVE_PATH = KAGGLE_CACHE_DIR / "1.archive"

DATASET_NAME = "vulamnguyen/rwf2000"

VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv"
}

EXPECTED_COUNTS = {
    ("train", "Fight"): 800,
    ("train", "NonFight"): 800,
    ("val", "Fight"): 200,
    ("val", "NonFight"): 200,
}

TOTAL_EXPECTED = 2000


# ============================================================
# 2. VERIFY EXISTING DATASET
# ============================================================

def get_video_files(folder):
    """Return video files inside a folder."""

    if not folder.exists():
        return []

    return [
        file
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in VIDEO_EXTENSIONS
    ]


def verify_dataset(dataset_dir):
    """
    Verify the expected RWF-2000 folder structure
    and exact video counts.
    """

    print("\nChecking dataset...")

    actual_counts = {}

    for (split, category), expected in EXPECTED_COUNTS.items():

        folder = dataset_dir / split / category

        videos = get_video_files(folder)

        actual_counts[(split, category)] = len(videos)

        print(
            f"{split}/{category}: "
            f"{len(videos)} videos "
            f"(expected: {expected})"
        )

    total = sum(actual_counts.values())

    is_valid = (
        actual_counts == EXPECTED_COUNTS
        and total == TOTAL_EXPECTED
    )

    print(f"\nTotal videos: {total}/{TOTAL_EXPECTED}")

    if is_valid:
        print("Dataset verification successful.")

    else:
        print("Dataset verification failed.")

    return is_valid


# ============================================================
# 3. SKIP IF DATASET IS ALREADY COMPLETE
# ============================================================

if verify_dataset(OUTPUT_DIR):

    print("\nDataset already exists and is complete.")
    print("Skipping download and extraction.")

else:

    print("\nDataset is missing or incomplete.")
    print("Preparing dataset using the cached archive...")


    # ========================================================
    # 4. CHECK ARCHIVE
    # ========================================================

    if ARCHIVE_PATH.exists():

        print("\nExisting KaggleHub archive found:")
        print(ARCHIVE_PATH)

        print(
            f"Archive size: "
            f"{ARCHIVE_PATH.stat().st_size / (1024**3):.2f} GB"
        )

    else:

        print("\nArchive not found.")
        print("Downloading dataset through KaggleHub...")

        try:
            import kagglehub

            try:
                kagglehub.dataset_download(DATASET_NAME)

            except Exception as error:
                print(
                    "\nKaggleHub automatic extraction failed:"
                )
                print(error)

                print(
                    "\nChecking whether the archive "
                    "was downloaded successfully..."
                )

        except ImportError:
            raise RuntimeError(
                "KaggleHub is not installed. "
                "Install it using: pip install kagglehub"
            )

        if not ARCHIVE_PATH.exists():

            raise FileNotFoundError(
                "Dataset archive was not found after download.\n"
                f"Expected location: {ARCHIVE_PATH}"
            )

        print("\nArchive downloaded successfully.")


    # ========================================================
    # 5. VALIDATE ARCHIVE
    # ========================================================

    print("\nValidating archive...")

    # IMPORTANT:
    # is_zipfile() belongs to the zipfile module,
    # not the ZipFile class.

    assert zipfile.is_zipfile(ARCHIVE_PATH), (
        "Downloaded archive is not a valid ZIP file."
    )

    print("Archive is a valid ZIP file.")


    # ========================================================
    # 6. INSPECT ARCHIVE CONTENTS
    # ========================================================

    print("\nInspecting archive contents...")

    with ZipFile(ARCHIVE_PATH, "r") as archive:

        video_entries = []

        for info in archive.infolist():

            if info.is_dir():
                continue

            path = Path(info.filename)

            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            parts = path.parts

            # Expected:
            # RWF-2000/train/Fight/video.avi

            if len(parts) < 4:
                continue

            if parts[0] != "RWF-2000":
                continue

            split = parts[1]
            category = parts[2]

            if (split, category) not in EXPECTED_COUNTS:
                continue

            video_entries.append(
                (info, split, category)
            )

        print(f"Video entries found: {len(video_entries)}")

        assert len(video_entries) == TOTAL_EXPECTED, (
            f"Expected {TOTAL_EXPECTED} videos, "
            f"but found {len(video_entries)}."
        )

        archive_counts = Counter(
            (split, category)
            for _, split, category in video_entries
        )

        print("\nArchive distribution:")

        for key, expected in EXPECTED_COUNTS.items():

            actual = archive_counts[key]

            print(
                f"{key[0]}/{key[1]}: "
                f"{actual}/{expected}"
            )

            assert actual == expected, (
                f"Incorrect archive count for {key}: "
                f"{actual}, expected {expected}"
            )


    # ========================================================
    # 7. EXTRACT VIDEOS SAFELY TO TEMP DIRECTORY
    # ========================================================

    print("\nStarting safe extraction...")

    # Remove only the temporary extraction directory.
    # The KaggleHub archive is never deleted.

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    extracted_counts = Counter()

    with ZipFile(ARCHIVE_PATH, "r") as archive:

        video_entries = []

        for info in archive.infolist():

            if info.is_dir():
                continue

            path = Path(info.filename)

            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            parts = path.parts

            if len(parts) < 4:
                continue

            if parts[0] != "RWF-2000":
                continue

            split = parts[1]
            category = parts[2]

            if (split, category) not in EXPECTED_COUNTS:
                continue

            video_entries.append(
                (info, split, category)
            )

        for index, (info, split, category) in enumerate(
            video_entries,
            start=1
        ):

            extracted_counts[(split, category)] += 1

            sequence = extracted_counts[(split, category)]

            # Generate a safe filename instead of using
            # the original potentially malformed filename.

            if category == "Fight":
                filename = f"fight_{sequence:04d}.avi"

            else:
                filename = f"nonfight_{sequence:04d}.avi"

            output_folder = (
                TEMP_DIR / split / category
            )

            output_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            output_file = output_folder / filename

            # Stream extraction to avoid loading large videos
            # into memory.

            with archive.open(info, "r") as source:
                with open(output_file, "wb") as destination:

                    shutil.copyfileobj(
                        source,
                        destination,
                        length=1024 * 1024
                    )

            if index % 100 == 0:
                print(
                    f"Extracted {index}/"
                    f"{TOTAL_EXPECTED} videos"
                )

    print("\nExtraction completed.")

    # ========================================================
    # 8. VERIFY TEMPORARY EXTRACTION
    # ========================================================

    print("\nVerifying temporary dataset...")

    if not verify_dataset(TEMP_DIR):

        raise RuntimeError(
            "Temporary extraction verification failed. "
            "The existing final dataset has not been replaced."
        )


    # ========================================================
    # 9. REPLACE FINAL DATASET
    # ========================================================

    print("\nMoving verified dataset to final location...")

    BACKUP_DIR = Path("/content/RWF-2000_incomplete_backup")

    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)

    if OUTPUT_DIR.exists():
        OUTPUT_DIR.rename(BACKUP_DIR)

    try:

        TEMP_DIR.rename(OUTPUT_DIR)

    except Exception:

        # Restore the old dataset if replacement fails.

        if BACKUP_DIR.exists() and not OUTPUT_DIR.exists():
            BACKUP_DIR.rename(OUTPUT_DIR)

        raise

    # Delete the backup only after successful replacement.
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)


    # ========================================================
    # 10. FINAL VERIFICATION
    # ========================================================

    print("\nFinal dataset verification:")

    if not verify_dataset(OUTPUT_DIR):

        raise RuntimeError(
            "Final dataset verification failed."
        )

    print("\n" + "=" * 55)
    print("RWF-2000 DATASET PREPARATION COMPLETED")
    print("=" * 55)

    print(f"\nDataset location: {OUTPUT_DIR}")

    print("\nFinal folder structure:")

    for split in ["train", "val"]:

        for category in ["Fight", "NonFight"]:

            folder = OUTPUT_DIR / split / category

            print(
                f"{folder}: "
                f"{len(get_video_files(folder))} videos"
            )

    print("\nTotal videos: 2000")
    print("Dataset is ready for the next stage.")
