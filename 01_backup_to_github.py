
# =================================================================
# BACKUP_TO_GITHUB.PY
# UPLOAD EVERYTHING TO EXISTING GITHUB REPOSITORY
# =================================================================

from pathlib import Path
import shutil
import subprocess
import tempfile
import os


# ------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------

PROJECT_DIR = Path("/content/Vi-SAFE-3B42-Implementation")

REPO_OWNER = "merajsiddieque"
REPO_NAME = "Vi-SAFE-3B42-Implementation"
BRANCH = "main"

REPO_URL = (
    f"https://github.com/{REPO_OWNER}/"
    f"{REPO_NAME}.git"
)

COMMIT_MESSAGE = "Update complete Vi-SAFE project from Colab"

GITHUB_MAX_FILE_SIZE = 100 * 1024 * 1024


# ------------------------------------------------
# 2. VERIFY LOCAL PROJECT
# ------------------------------------------------

if not PROJECT_DIR.is_dir():
    raise FileNotFoundError(
        f"Project directory not found: {PROJECT_DIR}"
    )


# ------------------------------------------------
# 3. LOAD GITHUB CREDENTIALS
# ------------------------------------------------

GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

if not GITHUB_USERNAME or not GITHUB_TOKEN:
    raise RuntimeError(
        "GitHub credentials are missing.\n"
        "Run the Colab credentials cell first."
    )

if GITHUB_USERNAME != REPO_OWNER:
    raise RuntimeError(
        "GitHub username does not match repository owner."
    )


# ------------------------------------------------
# 4. IGNORE ONLY GIT METADATA
# ------------------------------------------------

def ignore_git_metadata(directory, names):
    """
    Ignore only .git metadata directories/files.
    This applies recursively, including inside mmaction2.
    """
    return {
        name for name in names
        if name == ".git"
    }


# ------------------------------------------------
# 5. INVENTORY ALL LOCAL FILES
# ------------------------------------------------

all_files = [
    path
    for path in PROJECT_DIR.rglob("*")
    if path.is_file()
    and ".git" not in path.relative_to(PROJECT_DIR).parts
]

print("=" * 75)
print("COMPLETE PROJECT UPLOAD INVENTORY")
print("=" * 75)

print("Project:", PROJECT_DIR)
print("Total files:", len(all_files))

total_bytes = sum(path.stat().st_size for path in all_files)

print(f"Total size: {total_bytes / (1024 ** 3):.3f} GB")

print("\nFiles to upload:")

for path in sorted(all_files):
    print(path.relative_to(PROJECT_DIR))

print("=" * 75)


# ------------------------------------------------
# 6. CHECK GITHUB FILE-SIZE LIMIT
# ------------------------------------------------

large_files = [
    (path, path.stat().st_size)
    for path in all_files
    if path.stat().st_size >= GITHUB_MAX_FILE_SIZE
]

if large_files:

    print("\nFiles exceeding GitHub's 100 MB limit:")

    for path, size in large_files:
        print(
            f"{path.relative_to(PROJECT_DIR)}: "
            f"{size / (1024 ** 2):.2f} MB"
        )

    raise RuntimeError(
        "Upload stopped. Files exceeding 100 MB "
        "require Git LFS or another storage method."
    )

print("\nFile-size check passed.")


# ------------------------------------------------
# 7. GIT HELPER
# ------------------------------------------------

def run_git(args, cwd, env=None):

    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Git command failed:\n"
            f"git {' '.join(args)}\n\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout.strip()


# ------------------------------------------------
# 8. TEMPORARY AUTHENTICATION AND WORKSPACE
# ------------------------------------------------

try:

    with tempfile.TemporaryDirectory() as temp:

        TEMP_DIR = Path(temp)

        ASKPASS = TEMP_DIR / "askpass.sh"

        ASKPASS.write_text(
            '#!/bin/sh\n'
            'case "$1" in\n'
            '  *Username*) echo "$GITHUB_USERNAME" ;;\n'
            '  *Password*) echo "$GITHUB_TOKEN" ;;\n'
            '  *) echo "" ;;\n'
            'esac\n'
        )

        ASKPASS.chmod(0o700)

        GIT_ENV = os.environ.copy()

        GIT_ENV["GIT_ASKPASS"] = str(ASKPASS)
        GIT_ENV["GIT_TERMINAL_PROMPT"] = "0"
        GIT_ENV["GITHUB_USERNAME"] = GITHUB_USERNAME
        GIT_ENV["GITHUB_TOKEN"] = GITHUB_TOKEN

        # ------------------------------------------------
        # 9. CLONE EXISTING GITHUB REPOSITORY
        # ------------------------------------------------

        WORK_DIR = TEMP_DIR / "github-project"

        print("\n[1/8] Cloning existing GitHub repository...")

        run_git(
            [
                "clone",
                "--branch", BRANCH,
                REPO_URL,
                str(WORK_DIR)
            ],
            cwd=TEMP_DIR,
            env=GIT_ENV
        )

        print("Repository cloned successfully.")

        # ------------------------------------------------
        # 10. MIRROR LOCAL PROJECT INTO CLONE
        # ------------------------------------------------

        print("\n[2/8] Preparing GitHub workspace...")

        # Remove existing repository contents except root .git.

        for item in WORK_DIR.iterdir():

            if item.name == ".git":
                continue

            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        # Copy all local project contents.
        # Only nested .git metadata is excluded.

        for item in PROJECT_DIR.iterdir():

            if item.name == ".git":
                continue

            destination = WORK_DIR / item.name

            if item.is_dir():

                shutil.copytree(
                    item,
                    destination,
                    ignore=ignore_git_metadata
                )

            else:

                shutil.copy2(
                    item,
                    destination
                )

        print("All project contents copied successfully.")

        # ------------------------------------------------
        # 11. VERIFY COPIED FILE INVENTORY
        # ------------------------------------------------

        print("\n[3/8] Verifying copied project files...")

        copied_files = [
            path
            for path in WORK_DIR.rglob("*")
            if path.is_file()
            and ".git" not in path.relative_to(WORK_DIR).parts
        ]

        print("Local files:", len(all_files))
        print("Copied files:", len(copied_files))

        if len(all_files) != len(copied_files):
            raise RuntimeError(
                "File inventory mismatch. "
                "Some files were not copied."
            )

        # Verify every copied file and its size.

        for source_file in all_files:

            relative_path = source_file.relative_to(PROJECT_DIR)
            copied_file = WORK_DIR / relative_path

            if not copied_file.is_file():
                raise RuntimeError(
                    f"Missing copied file: {relative_path}"
                )

            if (
                source_file.stat().st_size
                != copied_file.stat().st_size
            ):
                raise RuntimeError(
                    f"File size mismatch: {relative_path}"
                )

        print("Copied file verification successful.")

        # ------------------------------------------------
        # 12. FORCE-STAGE AND VERIFY EVERY FILE
        # ------------------------------------------------

        print("\n[4/8] Force-staging every project file...")

        relative_files = [
            path.relative_to(PROJECT_DIR).as_posix()
            for path in all_files
        ]

        BATCH_SIZE = 100

        for start in range(0, len(relative_files), BATCH_SIZE):

            batch = relative_files[
                start:start + BATCH_SIZE
            ]

            run_git(
                ["add", "-f", "--"] + batch,
                cwd=WORK_DIR
            )

            print(
                f"Processed "
                f"{min(start + BATCH_SIZE, len(relative_files))}"
                f"/{len(relative_files)} files"
            )

        # Stage deletions of previously tracked files.

        run_git(
            ["add", "-u"],
            cwd=WORK_DIR
        )

        # IMPORTANT:
        # git diff --cached lists only changed files.
        # git ls-files lists files in the Git index.

        indexed_files_output = run_git(
            ["ls-files"],
            cwd=WORK_DIR
        )

        indexed_files = (
            indexed_files_output.splitlines()
            if indexed_files_output
            else []
        )

        indexed_set = set(indexed_files)

        print("\nExpected local files:", len(relative_files))
        print("Files in Git index:", len(indexed_files))

        # Verify all local files exist in the index.

        missing_from_index = [
            name for name in relative_files
            if name not in indexed_set
        ]

        if missing_from_index:

            print("\nFiles missing from Git index:")

            for name in missing_from_index[:30]:
                print(name)

            raise RuntimeError(
                f"{len(missing_from_index)} files "
                "are missing from the Git index."
            )

        print(
            "\nSUCCESS: Every inventoried local file "
            "is present in the Git index."
        )

        # Count files that actually have pending changes.

        changed_files_output = run_git(
            ["diff", "--cached", "--name-only"],
            cwd=WORK_DIR
        )

        changed_files = (
            changed_files_output.splitlines()
            if changed_files_output
            else []
        )

        print(
            "Files with pending changes:",
            len(changed_files)
        )

        # ------------------------------------------------
        # 13. VERIFY MMACTION2 AND YOLOv8
        # ------------------------------------------------

        print("\n[5/8] Verifying important project files...")

        mmaction2_indexed = [
            name for name in indexed_files
            if name.startswith("mmaction2/")
        ]

        print(
            "MMAction2 files in Git index:",
            len(mmaction2_indexed)
        )

        if (PROJECT_DIR / "mmaction2").is_dir():

            if not mmaction2_indexed:
                raise RuntimeError(
                    "MMAction2 source files are missing "
                    "from the Git index."
                )

            print(
                "Verified: MMAction2 source files are indexed."
            )

        MODEL_PATH = "yolov8n.pt"

        if (PROJECT_DIR / MODEL_PATH).exists():

            if MODEL_PATH not in indexed_set:
                raise RuntimeError(
                    "yolov8n.pt exists locally but is missing "
                    "from the Git index."
                )

            print("Verified: yolov8n.pt is indexed.")

        else:

            print("YOLOv8 checkpoint not found in local project.")

        # Final verification of all local files.

        missing_from_index = [
            path.relative_to(PROJECT_DIR).as_posix()
            for path in all_files
            if path.relative_to(PROJECT_DIR).as_posix()
            not in indexed_set
        ]

        if missing_from_index:
            raise RuntimeError(
                f"{len(missing_from_index)} local files "
                "are missing from the Git index."
            )

        print("All inventoried local files are indexed.")

        # ------------------------------------------------
        # 14. COMMIT CHANGES
        # ------------------------------------------------

        print("\n[6/8] Creating Git commit...")

        has_changes = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=WORK_DIR
        ).returncode != 0

        if has_changes:

            run_git(
                ["config", "user.name", GITHUB_USERNAME],
                cwd=WORK_DIR
            )

            run_git(
                [
                    "config",
                    "user.email",
                    f"{GITHUB_USERNAME}@users.noreply.github.com"
                ],
                cwd=WORK_DIR
            )

            run_git(
                ["commit", "-m", COMMIT_MESSAGE],
                cwd=WORK_DIR
            )

            print("Commit created successfully.")

        else:

            print(
                "No changes detected. "
                "Repository is already up to date."
            )

        # ------------------------------------------------
        # 15. PUSH TO EXISTING MAIN BRANCH
        # ------------------------------------------------

        print(
            "\n[7/8] Pushing complete project to GitHub..."
        )

        run_git(
            ["push", "origin", BRANCH],
            cwd=WORK_DIR,
            env=GIT_ENV
        )

        print("Push completed successfully.")

        # ------------------------------------------------
        # 16. VERIFY REMOTE COMMIT
        # ------------------------------------------------

        print("\n[8/8] Verifying remote commit...")

        run_git(
            ["fetch", "origin", BRANCH],
            cwd=WORK_DIR,
            env=GIT_ENV
        )

        LOCAL_COMMIT = run_git(
            ["rev-parse", "HEAD"],
            cwd=WORK_DIR
        )

        REMOTE_COMMIT = run_git(
            ["rev-parse", f"origin/{BRANCH}"],
            cwd=WORK_DIR
        )

        if LOCAL_COMMIT != REMOTE_COMMIT:
            raise RuntimeError(
                "GitHub verification failed: commit mismatch."
            )

        # Save report values before temporary directory closes.

        FINAL_COPIED_COUNT = len(copied_files)
        FINAL_INDEX_COUNT = len(indexed_files)
        FINAL_MMACTION_COUNT = len(mmaction2_indexed)
        FINAL_MODEL_INCLUDED = MODEL_PATH in indexed_set
        FINAL_CHANGED_COUNT = len(changed_files)

finally:

    # Clear credentials from the environment.

    GITHUB_TOKEN = None
    GITHUB_USERNAME = None

    os.environ.pop("GITHUB_TOKEN", None)
    os.environ.pop("GITHUB_USERNAME", None)

    if "GIT_ENV" in globals():
        GIT_ENV["GITHUB_TOKEN"] = ""
        GIT_ENV["GITHUB_USERNAME"] = ""


# ------------------------------------------------
# 17. FINAL REPORT
# ------------------------------------------------

print("\n" + "=" * 75)
print("GITHUB BACKUP COMPLETED SUCCESSFULLY")
print("=" * 75)

print("Repository:", REPO_URL)
print("Branch:", BRANCH)
print("Commit:", LOCAL_COMMIT[:12])

print("Local files inventoried:", len(all_files))
print("Copied files:", FINAL_COPIED_COUNT)
print("Files in Git index:", FINAL_INDEX_COUNT)
print("Files with pending changes:", FINAL_CHANGED_COUNT)

print("MMAction2 files:", FINAL_MMACTION_COUNT)

print(
    "YOLOv8 checkpoint:",
    "Included" if FINAL_MODEL_INCLUDED else "Not present"
)

print("\nLocal and remote commits match.")

print(
    f"Repository URL: "
    f"https://github.com/{REPO_OWNER}/{REPO_NAME}/tree/{BRANCH}"
)

print("=" * 75)
print("\nBackup script finished.")
