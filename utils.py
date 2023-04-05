import json
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent


def load_secrets() -> dict:
    secrets_fp = get_project_root() / "src" / "secrets.json"
    with open(secrets_fp, "r") as f:
        return json.load(f)
