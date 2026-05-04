#!/usr/bin/env bash
# Deploy the Gradio app to Hugging Face Spaces.
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login
#
# Usage:
#   bash scripts/deploy_hf.sh <hf_username>
#   bash scripts/deploy_hf.sh Deepanshu-Mehta
#
# This pushes the app/ directory contents to:
#   https://huggingface.co/spaces/<hf_username>/isic2024-demo

set -euo pipefail

HF_USER="${1:?Usage: $0 <hf_username>}"
SPACE_NAME="isic2024-demo"
SPACE_REPO="$HF_USER/$SPACE_NAME"
SPACE_URL="https://huggingface.co/spaces/$SPACE_REPO"
APP_DIR="$(cd "$(dirname "$0")/../app" && pwd)"

echo "=== Deploying to HF Spaces: $SPACE_URL ==="
echo "Source: $APP_DIR"

# Create the Space if it doesn't exist
if ! huggingface-cli repo info "spaces/$SPACE_REPO" &>/dev/null; then
    echo "Creating Space: $SPACE_REPO ..."
    huggingface-cli repo create "$SPACE_NAME" --type space --space-sdk gradio -y
fi

# Clone (or update) the Space repo into a temp directory
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "Cloning Space repo..."
git clone "https://huggingface.co/spaces/$SPACE_REPO" "$TMPDIR/space" 2>/dev/null || {
    mkdir -p "$TMPDIR/space"
    cd "$TMPDIR/space"
    git init
    git remote add origin "https://huggingface.co/spaces/$SPACE_REPO"
}

# Sync app/ contents into the Space
echo "Syncing files..."
rsync -av --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    "$APP_DIR/" "$TMPDIR/space/"

# Commit and push
cd "$TMPDIR/space"
git add -A
if git diff --cached --quiet; then
    echo "No changes to deploy."
else
    git commit -m "Deploy ISIC 2024 demo app"
    echo "Pushing to $SPACE_URL ..."
    git push origin main
    echo ""
    echo "=== Deployed successfully ==="
    echo "Live at: $SPACE_URL"
fi
