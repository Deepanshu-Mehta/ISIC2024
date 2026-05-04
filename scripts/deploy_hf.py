"""Deploy the Gradio app to Hugging Face Spaces.

Prerequisites:
    pip install huggingface_hub
    hf auth login

Usage:
    python scripts/deploy_hf.py --username Deepanshu-Mehta
    python scripts/deploy_hf.py --username Deepanshu-Mehta --space-name isic2024-demo
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy to HF Spaces")
    parser.add_argument("--username", required=True, help="HF username")
    parser.add_argument("--space-name", default="isic2024-demo", help="Space name")
    args = parser.parse_args()

    repo_id = f"{args.username}/{args.space_name}"
    app_dir = Path(__file__).resolve().parent.parent / "app"

    print(f"Deploying {app_dir} -> https://huggingface.co/spaces/{repo_id}")

    # Create the Space (no-op if it already exists)
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )
    print(f"Space ready: {repo_id}")

    # Upload the entire app/ directory
    api = HfApi()
    api.upload_folder(
        folder_path=str(app_dir),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=["__pycache__", "*.pyc"],
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nDeployed successfully!")
    print(f"Live at: {url}")


if __name__ == "__main__":
    main()
