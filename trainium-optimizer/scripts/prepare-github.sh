#!/usr/bin/env bash
# prepare-github.sh — stage a CLEAN, SECRET-FREE copy of the framework for
# publishing, and refuse to proceed if anything secret-shaped is found.
#
# Why this exists: the parent workspace contains AWS credentials (creds.json),
# an HF token, and other secrets. A naive `git init && git add .` from the
# wrong directory would leak them publicly. This script:
#   1. copies ONLY the framework subtree into a clean staging dir
#   2. scans that copy for secret patterns and known secret filenames
#   3. hard-fails if anything is found
#   4. inits git in the staging dir (never in the working tree)
#
# Usage:
#   scripts/prepare-github.sh [--name REPO_NAME] [--out DIR]
#
# Defaults: name=trainium-optimizer, out=.tmp/github-staging/<name>

set -euo pipefail

REPO_NAME="trainium-optimizer"
OUT_BASE=""
while (( $# )); do
  case "$1" in
    --name) REPO_NAME="$2"; shift 2;;
    --out)  OUT_BASE="$2"; shift 2;;
    -h|--help)
      echo "usage: $0 [--name REPO_NAME] [--out DIR]"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# This script lives in <framework>/scripts/ ; framework root is its parent.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Refuse to run if the framework root looks like the parent workspace (which
# holds secrets). The framework must be a self-contained subtree.
if [[ -f "$SRC_ROOT/creds.json" || -f "$SRC_ROOT/.asana_token" ]]; then
  echo "REFUSING: $SRC_ROOT looks like the credentialed workspace root." >&2
  echo "Run this from the framework subtree, not the workspace." >&2
  exit 1
fi

OUT_BASE="${OUT_BASE:-$SRC_ROOT/.tmp/github-staging}"
STAGING="$OUT_BASE/$REPO_NAME"

echo "==> staging a clean copy at $STAGING"
rm -rf "$STAGING"
mkdir -p "$STAGING"

# Copy the framework, excluding run artifacts, caches, venvs, and the staging
# dir itself. rsync honors a small exclude list mirroring .gitignore.
rsync -a \
  --exclude '.git/' \
  --exclude '.tmp/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.venv/' --exclude 'venv/' \
  --exclude '.pytest_cache/' \
  --exclude 'implementation/artifacts/' \
  --exclude 'artifacts/' \
  --exclude 'optimization_runs/' \
  --exclude 'optimized_models/' \
  --exclude '*.neff' --exclude '*.ntff' \
  --exclude '*.safetensors' --exclude '*.bin' --exclude '*.pt' --exclude '*.pth' \
  "$SRC_ROOT/" "$STAGING/"

echo "==> scanning the staged copy for secrets"
FAIL=0

# 1. secret-shaped content. Exclude this scanner and the notices file, which
#    legitimately contain the patterns themselves (for scanning/documentation).
if grep -rilE \
    "AKIA[0-9A-Z]{16}|aws_secret_access_key|BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY|xoxb-[0-9]|ghp_[0-9a-zA-Z]{36}|hf_[0-9a-zA-Z]{34}|-----BEGIN PRIVATE KEY-----" \
    "$STAGING" 2>/dev/null \
    | grep -vE "prepare-github\.sh|THIRD_PARTY_NOTICES\.md|\.gitignore|RUN\.md"; then
  echo "  !! secret-shaped content found above" >&2
  FAIL=1
fi

# 2. known secret filenames
if find "$STAGING" -type f \( \
      -name "*.pem" -o -name "creds*.json" -o -name ".env" \
      -o -name "*token*" -o -name "*secret*" -o -name "credentials" \
    \) 2>/dev/null | grep -q .; then
  echo "  !! secret-named files found:" >&2
  find "$STAGING" -type f \( \
      -name "*.pem" -o -name "creds*.json" -o -name ".env" \
      -o -name "*token*" -o -name "*secret*" -o -name "credentials" \
    \) 2>/dev/null | sed 's/^/     /' >&2
  FAIL=1
fi

if (( FAIL )); then
  echo "" >&2
  echo "ABORTING: remove the above before publishing. Nothing was committed." >&2
  exit 1
fi
echo "  clean — no secrets found"

echo "==> initializing git in the staged copy"
cd "$STAGING"
git init -q
git add -A
git -c user.email="none@example.com" -c user.name="prepare-github" \
    commit -q -m "Initial public release: autonomous Trainium model optimizer + knowledge bank"

echo ""
echo "============================================================"
echo "Clean repo staged at: $STAGING"
echo ""
echo "Review it, then publish with your own account:"
echo "  cd $STAGING"
echo "  gh repo create $REPO_NAME --public --source=. --push"
echo ""
echo "Files staged: $(git ls-files | wc -l | tr -d ' ')"
echo "============================================================"
