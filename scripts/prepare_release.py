"""Build the AstrBot plugin registry from the checked-in plugin metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def main() -> None:
    """Write a versioned archive locator and matching registry checksum.

    Raises:
        ValueError: The metadata does not describe the expected GitHub repository.
    """
    root = Path(__file__).resolve().parents[1]
    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8"))
    repo = str(metadata["repo"]).rstrip("/")
    if repo != "https://github.com/littleApricot/astrbot_plugin_reply_gate":
        raise ValueError("Unexpected plugin repository")
    version = str(metadata["version"])
    tag = version if version.startswith("v") else f"v{version}"
    registry_path = root / "plugins.json"
    registry = (
        json.loads(registry_path.read_text(encoding="utf-8"))
        if registry_path.exists()
        else {}
    )
    registry.update(
        {
            "$meta": {
                "schema_version": 1,
                "name": "littleApricot AstrBot Plugins",
                "version": version,
                "description": "回复判断与群聊表情图库发布源",
                "repository": repo,
            },
            "littleApricot/astrbot_plugin_reply_gate": {
                "author": metadata["author"],
                "name": metadata["name"],
                "display_name": metadata["display_name"],
                "desc": metadata["desc"],
                "version": version,
                "repo": repo,
                "download_url": f"{repo}/archive/refs/tags/{tag}.zip",
                "astrbot_version": metadata["astrbot_version"],
            },
        }
    )
    content = json.dumps(registry, ensure_ascii=False, indent=2) + "\n"
    (root / "plugins.json").write_text(content, encoding="utf-8")
    checksum = {"md5": hashlib.md5(content.encode("utf-8")).hexdigest()}
    (root / "plugins-md5.json").write_text(
        json.dumps(checksum, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared registry for {tag}. Push the commit and matching tag together.")


if __name__ == "__main__":
    main()
