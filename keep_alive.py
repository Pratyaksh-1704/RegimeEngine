"""
keep_alive.py — Prevent Render free-tier spin-down.

Spawns a daemon thread that pings the app's own Streamlit health
endpoint every 13 minutes. Safe to import unconditionally; the
thread only starts if the RENDER environment variable is set.
"""
import os
import threading
import time
import urllib.request
import logging

logger = logging.getLogger(__name__)

_INTERVAL_SECONDS = 13 * 60  # 13 minutes (Render sleeps after 15)


def _ping_loop():
    """Background loop that hits the local health endpoint."""
    port = os.environ.get("PORT", "8501")
    url = f"http://localhost:{port}/_stcore/health"
    while True:
        time.sleep(_INTERVAL_SECONDS)
        try:
            urllib.request.urlopen(url, timeout=10)
            logger.debug("keep_alive: pinged %s", url)
        except Exception as exc:
            logger.debug("keep_alive: ping failed (%s)", exc)


def start():
    """Start the keep-alive thread (idempotent, runs at most once)."""
    if not os.environ.get("RENDER"):
        return  # skip locally
    t = threading.Thread(target=_ping_loop, daemon=True, name="keep-alive")
    t.start()
    logger.info("keep_alive: daemon thread started (interval=%ds)", _INTERVAL_SECONDS)


# Auto-start on import
start()
