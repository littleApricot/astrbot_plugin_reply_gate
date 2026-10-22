"""Verify registry loading and archive updates against an isolated local server."""

import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import yaml
from aiohttp import web
from astrbot.core.star.star import StarMetadata
from astrbot.core.star.updater import _PluginUpdater
from astrbot.dashboard.services.plugin_service import PluginService


class UpdateSourceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="reply-gate-update-")
        self.workspace = Path(self.temp.name)
        self.root = Path(__file__).resolve().parents[1]
        self.registry = json.loads((self.root / "plugins.json").read_text())
        self.version = yaml.safe_load((self.root / "metadata.yaml").read_text())[
            "version"
        ]
        buffer = io.BytesIO()
        with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
            for name in [
                "main.py",
                "shared_context.py",
                "vision_gate.py",
                "__init__.py",
                "metadata.yaml",
                "_conf_schema.json",
            ]:
                archive.write(self.root / name, f"release/{name}")
        self.archive_bytes = buffer.getvalue()
        app = web.Application()
        app.router.add_get("/plugins.json", self.serve_registry)
        app.router.add_get("/plugins-md5.json", self.serve_checksum)
        app.router.add_get("/release.zip", self.serve_archive)
        self.server = web.AppRunner(app)
        await self.server.setup()
        site = web.TCPSite(self.server, "127.0.0.1", 0)
        await site.start()
        self.base = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        self.registry["littleApricot/astrbot_plugin_reply_gate"]["download_url"] = (
            self.base + "/release.zip"
        )
        self.service = PluginService(None, None)

    async def asyncTearDown(self):
        await self.server.cleanup()
        self.temp.cleanup()

    async def serve_registry(self, request):
        return web.json_response(self.registry)

    async def serve_checksum(self, request):
        content = json.dumps(self.registry, ensure_ascii=False).encode()
        return web.json_response({"md5": hashlib.md5(content).hexdigest()})

    async def serve_archive(self, request):
        return web.Response(body=self.archive_bytes, content_type="application/zip")

    async def test_existing_plugin_updates_without_changing_local_configuration(self):
        data, _ = await self.service.get_online_plugins(
            custom_registry=self.base + "/plugins.json", force_refresh=True
        )
        entry = self.service.resolve_market_plugin_entry_by_id(
            data, "littleApricot/astrbot_plugin_reply_gate"
        )
        self.assertIsNotNone(entry)
        self.assertEqual(entry["version"], self.version)
        self.assertEqual(len(list(self.service.iter_market_plugin_entries(data))), 2)
        updater = _PluginUpdater()
        updater.plugin_store_path = str(self.workspace / "plugins")
        target = Path(updater.plugin_store_path) / "astrbot_plugin_reply_gate"
        target.mkdir(parents=True)
        (target / "old_file.py").write_text("old plugin")
        local_config = self.workspace / "config/astrbot_plugin_reply_gate_config.json"
        local_config.parent.mkdir()
        saved = {"bailian_api_key": "test-only-fake-key", "reply_threshold": 0.05}
        local_config.write_text(json.dumps(saved))
        plugin = StarMetadata(
            name="astrbot_plugin_reply_gate",
            root_dir_name="astrbot_plugin_reply_gate",
            repo=entry["repo"],
        )
        await updater.update(plugin, download_url=entry["download_url"])
        self.assertFalse((target / "old_file.py").exists())
        self.assertEqual(
            (target / "main.py").read_bytes(), (self.root / "main.py").read_bytes()
        )
        self.assertEqual(
            yaml.safe_load((target / "metadata.yaml").read_text())["version"],
            self.version,
        )
        self.assertEqual(json.loads(local_config.read_text()), saved)

    async def test_changed_registry_checksum_refreshes_the_version(self):
        url = self.base + "/plugins.json"
        await self.service.get_online_plugins(custom_registry=url, force_refresh=True)
        self.registry = copy.deepcopy(self.registry)
        self.registry["littleApricot/astrbot_plugin_reply_gate"]["version"] = "v9.9.9"
        data, _ = await self.service.get_online_plugins(
            custom_registry=url, force_refresh=False
        )
        entry = self.service.resolve_market_plugin_entry_by_id(
            data, "littleApricot/astrbot_plugin_reply_gate"
        )
        self.assertEqual(entry["version"], "v9.9.9")
