
# ============================================================
# RWF-2000 DOWNLOAD + SAFE EXTRACTION
#
# Download using KaggleHub only when needed.
# Use the actual path returned by KaggleHub.
# Support extracted directories and ZIP archives.
# Create final dataset at /content/RWF-2000.
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


# ============================================================
# 1. INSTALL KAGGLEHUB IF REQUIRED
# ============================================================

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


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATASET_ID = "vulamnguyen/rwf2000"

OUTPUT = Path("/content/RWF-2000")

EXPECTED_TOTAL = 2000

VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv"
}

EXPECTED_FOLDERS = [
    ("train", "Fight"),
    ("train", "NonFight"),
    ("val", "Fight"),
    ("val", "NonFight"),
]


# ============================================================
# 3. CHECK EXISTING DATASET
# ============================================================

def dataset_is_ready(root):

    total = 0

    for split, category in EXPECTED_FOLDERS:

        folder = root / split / category

        if not folder.is_dir():
            return False

        videos = [
            file for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in VIDEO_EXTENSIONS
        ]

        if len(videos) == 0:
            return False

        total += len(videos)

    return total == EXPECTED_TOTAL


if dataset_is_ready(OUTPUT):

    print("Dataset already exists and is complete.")
    print("Skipping download and extraction.")
    print("Location:", OUTPUT)

else:

    # ========================================================
    # 4. DOWNLOAD USING KAGGLEHUB
    # ========================================================

    print("=" * 65)
    print("DOWNLOADING RWF-2000")
    print("=" * 65)

    print("Checking KaggleHub dataset cache...")

    try:

        downloaded_path = kagglehub.dataset_download(
            DATASET_ID
        )

        downloaded_path = Path(downloaded_path)

        print("\nKaggleHub returned:")
        print(downloaded_path)

    except Exception as error:

        raise RuntimeError(
            "KaggleHub dataset download failed. "
            "Check internet access and Kaggle permissions."
        ) from error

    if not downloaded_path.exists():

        raise FileNotFoundError(
            f"KaggleHub returned a path that does not exist: "
            f"{downloaded_path}"
        )


    # ========================================================
    # 5. LOCATE THE DATASET ROOT
    # ========================================================

    def has_expected_structure(root):

        return all(
            (root / split / category).is_dir()
            for split, category in EXPECTED_FOLDERS
        )


    def find_dataset_root(search_path):

        search_path = Path(search_path)

        # First check the returned directory itself.
        if search_path.is_dir() and has_expected_structure(search_path):
            return search_path

        # Check a common nested folder.
        nested = search_path / "RWF-2000"

        if nested.is_dir() and has_expected_structure(nested):
            return nested

        # Search nested directories for the expected structure.
        for train_folder in search_path.rglob("train"):

            if not train_folder.is_dir():
                continue

            candidate = train_folder.parent

            if has_expected_structure(candidate):
                return candidate

        return None


    dataset_root = find_dataset_root(downloaded_path)


    # ========================================================
    # 6. HANDLE ZIP ARCHIVE IF REQUIRED
    # ========================================================

    if dataset_root is None:

        print("\nExtracted directory not found.")
        print("Searching for ZIP archive...")

        zip_files = list(
            downloaded_path.rglob("*.zip")
        ) if downloaded_path.is_dir() else []

        if downloaded_path.is_file() and downloaded_path.suffix.lower() == ".zip":
            zip_files.append(downloaded_path)

        if not zip_files:

            raise FileNotFoundError(
                "Could not locate the RWF-2000 folder structure "
                "or a ZIP archive inside the KaggleHub result."
            )

        archive = zip_files[0]

        print("ZIP archive found:", archive)

        temporary_root = Path("/content/rwf2000_extraction")

        if temporary_root.exists():
            shutil.rmtree(temporary_root)

        temporary_root.mkdir(
            parents=True,
            exist_ok=True
        )

        print("\nExtracting ZIP archive...")

        with ZipFile(
            archive,
            mode="r",
            metadata_encoding="utf-8"
        ) as zf:

            for info in zf.infolist():

                if info.is_dir():
                    continue

                parts = info.filename.split("/")

                # Prevent unsafe paths and ignore unrelated files.
                if (
                    len(parts) < 4
                    or parts[-3] not in ("train", "val")
                    or parts[-2] not in ("Fight", "NonFight")
                ):
                    continue

                split = parts[-3]
                category = parts[-2]

                original_name = Path(parts[-1])

                if original_name.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue

                destination_dir = (
                    temporary_root / split / category
                )

                destination_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                destination = (
                    destination_dir / original_name.name
                )

                with zf.open(info, "r") as source:
                    with destination.open("wb") as target:
                        shutil.copyfileobj(source, target)

        dataset_root = find_dataset_root(temporary_root)

        if dataset_root is None:

            # The temporary extraction already has train/val at root.
            if has_expected_structure(temporary_root):
                dataset_root = temporary_root

            else:
                raise RuntimeError(
                    "ZIP extraction completed, but the expected "
                    "RWF-2000 folder structure was not found."
                )


    # ========================================================
    # 7. COPY INTO FINAL OUTPUT WITH SAFE FILENAMES
    # ========================================================

    print("\nDataset source:", dataset_root)
    print("Preparing final dataset directory...")

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    counts = {
        "train/Fight": 0,
        "train/NonFight": 0,
        "val/Fight": 0,
        "val/NonFight": 0,
    }

    print("\nCopying videos into final directory...")

    for split, category in EXPECTED_FOLDERS:

        source_dir = dataset_root / split / category

        destination_dir = OUTPUT / split / category

        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        videos = sorted([
            file for file in source_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in VIDEO_EXTENSIONS
        ])

        for index, source_file in enumerate(videos, start=1):

            extension = source_file.suffix.lower()

            safe_name = (
                f"{category.lower()}_{index:04d}{extension}"
            )

            destination_file = destination_dir / safe_name

            shutil.copy2(
                source_file,
                destination_file
            )

            counts[f"{split}/{category}"] += 1

        print(
            f"{split}/{category}: "
            f"{counts[f'{split}/{category}']} videos copied"
        )


    # ========================================================
    # 8. VERIFY EXTRACTION
    # ========================================================

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
        f"but found {total}."
    )


# ============================================================
# 9. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 65)
print("FINAL DATASET VERIFICATION")
print("=" * 65)

final_total = 0

for split, category in EXPECTED_FOLDERS:

    folder = OUTPUT / split / category

    assert folder.is_dir(), (
        f"Missing directory: {folder}"
    )

    videos = [
        file for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in VIDEO_EXTENSIONS
    ]

    count = len(videos)

    final_total += count

    print(f"{split}/{category}: {count} videos")


assert final_total == EXPECTED_TOTAL, (
    f"Expected {EXPECTED_TOTAL}, found {final_total}."
)

assert dataset_is_ready(OUTPUT), (
    "Final dataset verification failed."
)


# ============================================================
# 10. SUCCESS
# ============================================================

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
