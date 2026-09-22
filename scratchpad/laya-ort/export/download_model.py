"""Download the laya english checkpoint (root of convaiinnovations/laya) locally.

Read-only, public HF repo. Same allow-list laya itself uses (laya/agent.py:133-135),
so nothing outside the root checkpoint is pulled. Env proxy/SSL_CERT_FILE are
honored by huggingface_hub (httpx loads system trust paths).
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "cli" / "model" / "laya-en"

ALLOW = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]


def main() -> None:
    print(f"downloading convaiinnovations/laya -> {DEST}")
    path = snapshot_download(
        "convaiinnovations/laya",
        allow_patterns=ALLOW,
        local_dir=str(DEST),
    )
    for f in ["model.safetensors", "rl_agent_config.json"]:
        p = Path(path) / f
        if not p.exists():
            raise FileNotFoundError(f"missing {f} after download")
    size_mb = (Path(path) / "model.safetensors").stat().st_size / 2**20
    print(f"done: model.safetensors {size_mb:.0f} MiB at {path}")


if __name__ == "__main__":
    main()