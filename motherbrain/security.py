"""Hardening for a MotherBrain server that faces a network.

Nothing here makes the system "100% secure" — no honest component claims that.
What it does is close the specific holes a server like this actually has:

* An ingestion endpoint that accepts a filesystem path is an arbitrary-file-read
  primitive unless the path is confined to an allowlist. Anything read into the
  corpus can later be extracted back out through generation, so this is the
  sharpest edge in the whole system.
* An API key compared with `==` leaks its contents through timing.
* A server bound to a public interface with no key configured is simply open.
* Unbounded request bodies and unlimited request rates are a denial-of-service
  and a training-data-poisoning vector.
"""

from __future__ import annotations

import secrets
import threading
import time
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

# Anything under these names is refused outright, even inside an allowed root.
SENSITIVE_NAMES = {
    ".ssh", ".aws", ".gnupg", ".git-credentials", ".netrc", ".env",
    "id_rsa", "id_ed25519", "shadow", "authorized_keys",
}
SENSITIVE_SUFFIXES = {".key", ".pem", ".pfx", ".p12"}


def constant_time_eq(a: str | None, b: str | None) -> bool:
    """Compare secrets without leaking their contents through timing."""
    if a is None or b is None:
        return False
    return secrets.compare_digest(a.encode(), b.encode())


def safe_resolve(candidate: str, allowed_roots: list[Path]) -> Path:
    """Resolve `candidate` and prove it lives inside an allowed root.

    Resolution happens first so that `..` traversal and symlinks pointing out
    of the root are both normalised away before the containment check, rather
    than being string-matched beforehand.
    """
    if not allowed_roots:
        raise HTTPException(
            403, "path ingestion is disabled; start the server with --allow-path")

    try:
        target = Path(candidate).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(404, "path not found")

    for root in allowed_roots:
        try:
            target.relative_to(root)
        except ValueError:
            continue
        _reject_sensitive(target)
        return target

    raise HTTPException(
        403, "path is outside every allowed root; pass --allow-path to widen it")


def _reject_sensitive(target: Path) -> None:
    """Refuse credential-shaped files even when they sit inside an allowed root."""
    for part in target.parts:
        if part in SENSITIVE_NAMES:
            raise HTTPException(403, f"refusing to read {part!r}: looks like a secret")
    if target.suffix.lower() in SENSITIVE_SUFFIXES:
        raise HTTPException(
            403, f"refusing to read {target.suffix} files: they hold private keys")


class RateLimiter:
    """A token bucket per client address.

    Generation and training are expensive enough that an unthrottled endpoint
    is a denial-of-service button.
    """

    def __init__(self, per_minute: int = 120, burst: int | None = None) -> None:
        self.rate = per_minute / 60.0
        self.burst = burst if burst is not None else max(per_minute // 4, 5)
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            if len(self._buckets) > 8192:  # bound the memory an attacker can use
                cutoff = now - 3600
                self._buckets = {k: v for k, v in self._buckets.items() if v[1] > cutoff}
            return True


def client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def install_middleware(app, rate_limiter: RateLimiter | None,
                       max_body_bytes: int = 8 * 1024 * 1024) -> None:
    """Body-size caps, rate limiting, and conservative response headers."""

    @app.middleware("http")
    async def _guard(request: Request, call_next):
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_body_bytes:
            return JSONResponse(
                {"detail": f"request body exceeds {max_body_bytes} bytes"},
                status_code=413,
            )

        if rate_limiter is not None and request.url.path not in ("/health",):
            if not rate_limiter.allow(client_key(request)):
                return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)

        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")
        return response


def check_exposure(host: str, api_key: str | None, tls: bool,
                   insecure: bool) -> list[str]:
    """Refuse to expose an unauthenticated server to a network by accident.

    Returns advisory warnings; raises for the combination that is simply unsafe.
    """
    public = host not in ("127.0.0.1", "::1", "localhost")
    warnings: list[str] = []

    if public and not api_key and not insecure:
        raise SystemExit(
            "refusing to bind a public interface with no API key.\n"
            "  anyone who can reach this port could read, train and rewrite the model.\n"
            "  fix: pass --api-key (or set MB_API_KEY), or --insecure to override."
        )
    if public and not tls:
        warnings.append(
            "serving plaintext on a public interface: the API key and everything "
            "fed or generated cross the network in the clear. run `mb cert`.")
    if api_key and len(api_key) < 16:
        warnings.append(
            "the API key is short; prefer at least 16 random characters "
            "(python -c \"import secrets;print(secrets.token_urlsafe(32))\").")
    return warnings
