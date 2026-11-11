"""Polite HTTP client shared by the enrichment sources.

* one minimum interval per host (APIs ask for ~1 request/second)
* retries with exponential backoff on 429 / 5xx / network errors, honouring Retry-After
* a run-level time budget so a slow or rate-limited API never blocks the daily update
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .. import __version__

log = logging.getLogger(__name__)
USER_AGENT = f"cs-paper-radar/{__version__} (+https://github.com/ifatwme/cs-paper-radar)"


class BudgetExceeded(RuntimeError):
    pass


class HttpError(RuntimeError):
    def __init__(self, status: int, url: str, body: str = ""):
        super().__init__(f"HTTP {status} for {url}: {body[:200]}")
        self.status = status


class PoliteClient:
    def __init__(self, min_interval: float = 1.0, timeout: float = 60, retries: int = 5,
                 budget_s: float | None = None, headers: dict[str, str] | None = None):
        self.min_interval = min_interval
        self.timeout = timeout
        self.retries = retries
        self.deadline = time.monotonic() + budget_s if budget_s else None
        self.headers = {"User-Agent": USER_AGENT, **(headers or {})}
        self._last: dict[str, float] = {}
        self.requests = 0

    def _wait(self, host: str) -> None:
        if self.deadline and time.monotonic() > self.deadline:
            raise BudgetExceeded("time budget exhausted")
        gap = self.min_interval - (time.monotonic() - self._last.get(host, 0.0))
        if gap > 0:
            time.sleep(gap)

    def request(self, method: str, url: str, params: dict | None = None, body: Any = None,
                headers: dict[str, str] | None = None) -> Any:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        host = urllib.parse.urlsplit(url).netloc
        data = json.dumps(body).encode() if body is not None else None
        hdrs = {**self.headers, **(headers or {})}
        if data is not None:
            hdrs["Content-Type"] = "application/json"
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            self._wait(host)
            req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
            try:
                self.requests += 1
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = resp.read()
                    self._last[host] = time.monotonic()
                    return json.loads(payload) if payload else None
            except urllib.error.HTTPError as exc:
                self._last[host] = time.monotonic()
                status = exc.code
                text = exc.read().decode("utf-8", "replace") if exc.fp else ""
                if status in (400, 401, 404, 422) or (status == 403 and "rate limit" not in text.lower()):
                    raise HttpError(status, url, text) from None
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                wait = float(retry_after) if retry_after and retry_after.isdigit() else min(120, 3 * 2**attempt)
                last = HttpError(status, url, text)
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                self._last[host] = time.monotonic()
                wait = min(60, 3 * 2**attempt)
                last = exc
            if self.deadline and time.monotonic() + wait > self.deadline:
                raise BudgetExceeded(f"time budget exhausted after {last}")
            log.warning("%s %s failed (%s); retry in %.0fs", method, host, last, wait)
            time.sleep(wait)
        raise last if last else RuntimeError("request failed")

    def get(self, url: str, params: dict | None = None, **kw) -> Any:
        return self.request("GET", url, params=params, **kw)

    def post(self, url: str, body: Any, params: dict | None = None, **kw) -> Any:
        return self.request("POST", url, params=params, body=body, **kw)
