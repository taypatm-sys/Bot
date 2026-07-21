import logging
import os
from typing import Optional

from aiohttp import web


logger = logging.getLogger(__name__)


async def health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "service": "taypa-bot"})


def build_health_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    return app


async def start_health_server() -> Optional[web.AppRunner]:
    port_value = os.getenv("PORT", "").strip()
    if not port_value:
        return None

    try:
        port = int(port_value)
    except ValueError as error:
        raise RuntimeError("PORT must be a number") from error

    runner = web.AppRunner(build_health_app())
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=port)
    await site.start()
    logger.info("Health server is listening on port %s", port)
    return runner
