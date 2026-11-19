
#!/usr/bin/env python3
"""
RWF-2000 Dataset Preparation

Expected final structure:

/content/RWF-2000/
    train/
        Fight/       800 videos
        NonFight/    800 videos
    val/
        Fight/       200 videos
        NonFight/    200 videos

Features:
1. Checks whether the dataset is already complete.
2. Downloads through KaggleHub when required.
3. Uses the directory returned by KaggleHub.
4. Supports an extracted dataset or ZIP archive.
5. Copies videos safely with sequential filenames.
6. Verifies all expected video counts.
7. Preserves the existing dataset until the replacement is verified.
"""

import shutil
import zipfile
from pathlib import Path, PurePosixPath
from collections import Counter

# ============================================================
# 1. CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("/content/RWF-2000")
TEMP_DIR = Path("/content/rwf2000_extraction")
BACKUP_DIR = Path("/content/RWF-2000_incomplete_backup")

DATASET_NAME = "vulamnguyen/rwf2000"

KAGGLE_CACHE_DIR = Path(
    "/root/.cache/kagglehub/datasets/vulamnguyen/rwf2000"
)

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}

EXPECTED_COUNTS = {
    ("train", "Fight"): 800,
    ("train", "NonFight"): 800,
    ("val", "Fight"): 200,
    ("val", "NonFight"): 200,
}

TOTAL_EXPECTED = 2000


# ============================================================
# 2. DATASET VERIFICATION
# ============================================================

def get_video_files(folder):
    """Return video files directly inside a folder."""

    if not folder.is_dir():
        return []

    return sorted(
        file for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in VIDEO_EXTENSIONS
    )


def verify_dataset(dataset_dir, verbose=True):
    """Check the exact expected dataset structure and counts."""

    if verbose:
        print("\nChecking dataset...")

    actual_counts = {}

    for (split, category), expected in EXPECTED_COUNTS.items():

        folder = dataset_dir / split / category
        count = len(get_video_files(folder))

        actual_counts[(split, category)] = count

        if verbose:
            print(
                f"{split}/{category}: "
                f"{count} videos (expected: {expected})"
            )

    total = sum(actual_counts.values())

    valid = (
        actual_counts == EXPECTED_COUNTS
        and total == TOTAL_EXPECTED
    )

    if verbose:
        print(f"\nTotal videos: {total}/{TOTAL_EXPECTED}")
        print(
            "Dataset verification successful."
            if valid else
            "Dataset verification failed."
        )

    return valid


# ============================================================
# 3. FIND THE ACTUAL DATASET DIRECTORY
# ============================================================

def is_dataset_root(folder):
    """Check whether a directory contains all four dataset folders."""

    return all(
        (folder / split / category).is_dir()
        for split, category in EXPECTED_COUNTS
    )


def find_dataset_root(search_path):
    """
    Find the directory containing:
    train/Fight, train/NonFight, val/Fight, val/NonFight.

    KaggleHub may return a directory with additional
    parent folders, so search recursively.
    """

    search_path = Path(search_path)

    if not search_path.exists():
        return None

    if is_dataset_root(search_path):
        return search_path

    candidates = []

    for train_folder in search_path.rglob("train"):

        if not train_folder.is_dir():
            continue

        candidate = train_folder.parent

        if is_dataset_root(candidate):
            candidates.append(candidate)

    if not candidates:
        return None

    # Prefer the shallowest matching dataset root.
    candidates.sort(key=lambda p: len(p.parts))

    return candidates[0]


# ============================================================
# 4. COPY EXTRACTED DATASET
# ============================================================

def copy_dataset(source_root, destination_root):
    """
    Copy videos into a clean destination directory.

    Original filenames are replaced with sequential filenames.
    Original video extensions are preserved.
    """

    if destination_root.exists():
        shutil.rmtree(destination_root)

    destination_root.mkdir(parents=True, exist_ok=True)

    for split, category in EXPECTED_COUNTS:

        source_folder = source_root / split / category
        destination_folder = destination_root / split / category

        destination_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        videos = get_video_files(source_folder)

        for index, source_file in enumerate(videos, start=1):

            if category == "Fight":
                prefix = "fight"
            else:
                prefix = "nonfight"

            filename = (
                f"{prefix}_{index:04d}"
                f"{source_file.suffix.lower()}"
            )

            shutil.copy2(
                source_file,
                destination_folder / filename
            )

        print(
            f"Copied {split}/{category}: "
            f"{len(videos)} videos"
        )


# ============================================================
# 5. EXTRACT DATASET FROM ZIP ARCHIVE
# ============================================================

def extract_zip_dataset(archive_path, destination_root):
    """
    Extract only the expected RWF-2000 video files.

    This avoids extracting unrelated files and avoids
    trusting original archive filenames.
    """

    print("\nInspecting ZIP archive...")

    with zipfile.ZipFile(archive_path, "r") as archive:

        entries = []

        for info in archive.infolist():

            if info.is_dir():
                continue

            # ZIP paths use forward slashes.
            path = PurePosixPath(info.filename)

            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            parts = path.parts

            # Find the RWF-2000 directory wherever it occurs.
            try:
                root_index = parts.index("RWF-2000")
            except ValueError:
                continue

            remaining = parts[root_index + 1:]

            if len(remaining) != 3:
                continue

            split, category, filename = remaining

            if (split, category) not in EXPECTED_COUNTS:
                continue

            entries.append(
                (info, split, category)
            )

        print(f"Video entries found: {len(entries)}")

        archive_counts = Counter(
            (split, category)
            for _, split, category in entries
        )

        for key, expected in EXPECTED_COUNTS.items():

            actual = archive_counts[key]

            print(
                f"{key[0]}/{key[1]}: "
                f"{actual}/{expected}"
            )

            if actual != expected:
                raise RuntimeError(
                    f"Incorrect archive count for {key}: "
                    f"{actual}, expected {expected}"
                )

        if len(entries) != TOTAL_EXPECTED:
            raise RuntimeError(
                f"Expected {TOTAL_EXPECTED} videos, "
                f"found {len(entries)}."
            )

        if destination_root.exists():
            shutil.rmtree(destination_root)

        destination_root.mkdir(
            parents=True,
            exist_ok=True
        )

        extracted_counts = Counter()

        for index, (info, split, category) in enumerate(
            entries, start=1
        ):

            extracted_counts[(split, category)] += 1
            sequence = extracted_counts[(split, category)]

            prefix = (
                "fight" if category == "Fight"
                else "nonfight"
            )

            extension = (
                PurePosixPath(info.filename).suffix.lower()
            )

            filename = f"{prefix}_{sequence:04d}{extension}"

            output_folder = (
                destination_root / split / category
            )

            output_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            output_file = output_folder / filename

            with archive.open(info, "r") as source:
                with open(output_file, "wb") as destination:
                    shutil.copyfileobj(
                        source,
                        destination,
                        length=1024 * 1024
                    )

            if index % 100 == 0:
                print(
                    f"Extracted {index}/{TOTAL_EXPECTED} videos"
                )

    print("ZIP extraction completed.")


# ============================================================
# 6. DOWNLOAD DATASET THROUGH KAGGLEHUB
# ============================================================

def download_dataset():
    """Download dataset and return KaggleHub's actual path."""

    try:
        import kagglehub
    except ImportError:
        raise RuntimeError(
            "KaggleHub is not installed. Run:\n"
            "!pip install -q kagglehub"
        )

    print("\nDownloading/accessing dataset through KaggleHub...")

    downloaded_path = kagglehub.dataset_download(
        DATASET_NAME
    )

    downloaded_path = Path(downloaded_path)

    print("\nKaggleHub returned this path:")
    print(downloaded_path)

    if not downloaded_path.exists():
        raise FileNotFoundError(
            "KaggleHub returned a path that does not exist:\n"
            f"{downloaded_path}"
        )

    return downloaded_path


# ============================================================
# 7. PREPARE DATASET
# ============================================================

def prepare_dataset():

    # First, check the final dataset.
    if verify_dataset(OUTPUT_DIR):

        print("\nDataset already exists and is complete.")
        print("Skipping download and extraction.")

        return

    print("\nDataset is missing or incomplete.")

    # Obtain the dataset path from KaggleHub.
    downloaded_path = download_dataset()

    # First try the extracted directory.
    source_root = find_dataset_root(downloaded_path)

    if source_root is not None:

        print("\nExtracted dataset directory found:")
        print(source_root)

        if not verify_dataset(source_root):
            raise RuntimeError(
                "The downloaded dataset directory has "
                "incorrect video counts."
            )

        print("\nCopying dataset to temporary directory...")

        copy_dataset(
            source_root,
            TEMP_DIR
        )

    else:

        print(
            "\nNo extracted dataset directory found. "
            "Searching for a ZIP archive..."
        )

        zip_files = list(downloaded_path.rglob("*.zip"))

        # Also search the KaggleHub cache.
        if KAGGLE_CACHE_DIR.exists():
            zip_files.extend(
                KAGGLE_CACHE_DIR.rglob("*.zip")
            )

        zip_files = list(dict.fromkeys(zip_files))

        if not zip_files:
            raise FileNotFoundError(
                "KaggleHub download completed, but no "
                "valid dataset directory or ZIP archive "
                "was found.\n"
                f"Downloaded path: {downloaded_path}"
            )

        archive_path = max(
            zip_files,
            key=lambda p: p.stat().st_size
        )

        print("\nZIP archive found:")
        print(archive_path)

        print(
            f"Archive size: "
            f"{archive_path.stat().st_size / (1024**3):.2f} GB"
        )

        if not zipfile.is_zipfile(archive_path):
            raise RuntimeError(
                "The downloaded file is not a valid ZIP archive."
            )

        extract_zip_dataset(
            archive_path,
            TEMP_DIR
        )

    # ========================================================
    # 8. VERIFY TEMPORARY DATASET
    # ========================================================

    print("\nVerifying temporary dataset...")

    if not verify_dataset(TEMP_DIR):
        raise RuntimeError(
            "Temporary dataset verification failed. "
            "The existing final dataset has not been replaced."
        )

    # ========================================================
    # 9. SAFELY REPLACE FINAL DATASET
    # ========================================================

    print("\nPreparing final dataset replacement...")

    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)

    if OUTPUT_DIR.exists():
        OUTPUT_DIR.rename(BACKUP_DIR)

    try:
        TEMP_DIR.rename(OUTPUT_DIR)

    except Exception:

        if BACKUP_DIR.exists() and not OUTPUT_DIR.exists():
            BACKUP_DIR.rename(OUTPUT_DIR)

        raise

    # ========================================================
    # 10. FINAL VERIFICATION
    # ========================================================

    print("\nFinal dataset verification:")

    if not verify_dataset(OUTPUT_DIR):
        raise RuntimeError(
            "Final dataset verification failed."
        )

    # Delete backup only after successful verification.
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)

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


# ============================================================
# 11. EXECUTION
# ============================================================

if __name__ == "__main__":
    prepare_dataset()
