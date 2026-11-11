
# =================================================================
# CELL 2 — UPLOAD EVERYTHING TO EXISTING GITHUB REPOSITORY
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

REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"

COMMIT_MESSAGE = "Update complete Vi-SAFE project from Colab"

# ------------------------------------------------
# 2. VERIFY LOCAL PROJECT
# ------------------------------------------------

if not PROJECT_DIR.is_dir():
    raise FileNotFoundError(f"Project directory not found: {PROJECT_DIR}")

# ------------------------------------------------
# 3. LOAD GITHUB SECRETS
# ------------------------------------------------

from google.colab import userdata

GITHUB_USERNAME = userdata.get("GITHUB_USERNAME")
GITHUB_TOKEN = userdata.get("GITHUB_TOKEN")

if not GITHUB_USERNAME or not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_USERNAME or GITHUB_TOKEN missing from Colab Secrets.")

if GITHUB_USERNAME != REPO_OWNER:
    raise RuntimeError("GitHub username does not match repository owner.")

# ------------------------------------------------
# 4. DISPLAY COMPLETE LOCAL FILE INVENTORY
# ------------------------------------------------

all_files = [
    p for p in PROJECT_DIR.rglob("*")
    if p.is_file()
]

print("=" * 70)
print("COMPLETE PROJECT UPLOAD INVENTORY")
print("=" * 70)

print("Total files:", len(all_files))

total_bytes = sum(p.stat().st_size for p in all_files)
print(f"Total size: {total_bytes / (1024 ** 3):.3f} GB")

for path in sorted(all_files):
    print(path.relative_to(PROJECT_DIR))

print("=" * 70)

# ------------------------------------------------
# 5. GIT HELPER
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
            f"git {' '.join(args)}\n"
            f"{result.stderr}"
        )

    return result.stdout.strip()

# ------------------------------------------------
# 6. TEMPORARY AUTHENTICATION
# ------------------------------------------------

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
    # 7. CLONE EXISTING REPOSITORY
    # ------------------------------------------------

    WORK_DIR = TEMP_DIR / "github-project"

    print("\nCloning existing GitHub repository...")

    run_git(
        ["clone", "--branch", BRANCH, REPO_URL, str(WORK_DIR)],
        cwd=TEMP_DIR,
        env=GIT_ENV
    )

    # ------------------------------------------------
    # 8. MIRROR LOCAL PROJECT INTO CLONED REPOSITORY
    # ------------------------------------------------

    print("Preparing GitHub workspace...")

    # Remove existing tracked/untracked project contents,
    # while preserving Git metadata.
    for item in WORK_DIR.iterdir():
        if item.name == ".git":
            continue

        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    # Copy EVERYTHING, including hidden files and directories.
    # Do not create or apply any ignore rules.
    for item in PROJECT_DIR.iterdir():

        destination = WORK_DIR / item.name

        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)

    # ------------------------------------------------
    # 9. FORCE-STAGE EVERYTHING
    # ------------------------------------------------

    print("Force-staging all local project files...")

    # -f overrides existing Git ignore rules.
    # --all stages additions, modifications and deletions.
    run_git(
        ["add", "-f", "--all"],
        cwd=WORK_DIR
    )

    staged_files = run_git(
        ["diff", "--cached", "--name-only"],
        cwd=WORK_DIR
    )

    staged_list = staged_files.splitlines() if staged_files else []

    print("\nStaged files:", len(staged_list))

    print("\nStaged file list:")
    for filename in staged_list:
        print(filename)

    # ------------------------------------------------
    # 10. VERIFY YOLOv8 CHECKPOINT IS STAGED
    # ------------------------------------------------

    MODEL_PATH = "yolov8n.pt"

    if (PROJECT_DIR / MODEL_PATH).exists():

        if MODEL_PATH not in staged_list:
            raise RuntimeError(
                "yolov8n.pt exists locally but is not staged."
            )

        print("\nVerified: yolov8n.pt is staged.")

    # ------------------------------------------------
    # 11. COMMIT
    # ------------------------------------------------

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

        print("\nCommit created successfully.")

    else:
        print("\nNo changes detected.")

    # ------------------------------------------------
    # 12. PUSH TO EXISTING MAIN BRANCH
    # ------------------------------------------------

    print("\nPushing complete project to GitHub...")

    run_git(
        ["push", "origin", BRANCH],
        cwd=WORK_DIR,
        env=GIT_ENV
    )

    # ------------------------------------------------
    # 13. VERIFY REMOTE COMMIT
    # ------------------------------------------------

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
        raise RuntimeError("GitHub verification failed: commit mismatch.")

# ------------------------------------------------
# 14. FINAL REPORT
# ------------------------------------------------

print("\n" + "=" * 70)
print("GITHUB UPLOAD SUCCESSFUL")
print("=" * 70)

print("Repository:", REPO_URL)
print("Branch:", BRANCH)
print("Commit:", LOCAL_COMMIT[:12])
print("Local files inventoried:", len(all_files))
print("Files staged:", len(staged_list))
print("YOLOv8 checkpoint:", "Included" if (PROJECT_DIR / "yolov8n.pt").exists() else "Not present")

print("\nVerification: Local and remote commits match.")
print(f"https://github.com/{REPO_OWNER}/{REPO_NAME}/tree/{BRANCH}")

print("=" * 70)

# Clear token variables
GITHUB_TOKEN = None
GIT_ENV["GITHUB_TOKEN"] = ""
del GIT_ENV
