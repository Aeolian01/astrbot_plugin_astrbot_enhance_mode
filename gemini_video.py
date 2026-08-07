from __future__ import annotations

import asyncio
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse, urlunparse

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

BAD_VIDEO_CAPTION_PATTERNS = (
    "please provide the video",
    "provide the video",
    "upload a video",
    "send me the video",
    "no video was provided",
    "video was not provided",
    "请提供视频",
    "请您提供视频",
    "请上传视频",
    "请提供您想让我描述的视频",
    "请提供您想描述的视频",
    "一旦您提供了视频",
)


class GeminiVideoError(RuntimeError):
    pass


RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}


def _retry_delay(attempt: int, retry_after: str = "") -> float:
    try:
        parsed = float(str(retry_after or "").strip())
        if parsed > 0:
            return min(parsed, 8.0)
    except (TypeError, ValueError):
        pass
    return min(2.0 * (2 ** max(0, attempt)), 8.0)


def is_bad_video_caption(text: str) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return False
    lowered = clean.lower()
    compact = "".join(lowered.split())
    if "请提供" in clean and "视频" in clean:
        return True
    if "请您提供" in clean and "视频" in clean:
        return True
    if "请上传" in clean and "视频" in clean:
        return True
    if "请您上传" in clean and "视频" in clean:
        return True
    if "一旦您提供" in clean and "视频" in clean:
        return True
    return any("".join(pattern.split()) in compact for pattern in BAD_VIDEO_CAPTION_PATTERNS)


def _provider_config(provider: Any) -> dict[str, Any]:
    raw = getattr(provider, "provider_config", {})
    return raw if isinstance(raw, dict) else {}


def provider_label(provider: Any) -> str:
    for attr in ("id", "provider_id", "name"):
        value = str(getattr(provider, attr, "") or "").strip()
        if value:
            return value
    try:
        meta = provider.meta()
        value = str(getattr(meta, "id", "") or "").strip()
        if value:
            return value
    except Exception:
        pass
    return type(provider).__name__


def provider_model(provider: Any) -> str:
    getter = getattr(provider, "get_model", None)
    if callable(getter):
        try:
            value = str(getter() or "").strip()
            if value:
                return value
        except Exception:
            pass
    return str(_provider_config(provider).get("model") or "").strip()


def provider_api_key(provider: Any) -> str:
    getter = getattr(provider, "get_current_key", None)
    if callable(getter):
        try:
            value = str(getter() or "").strip()
            if value:
                return value
        except Exception:
            pass

    keys: list[Any] = []
    getter = getattr(provider, "get_keys", None)
    if callable(getter):
        try:
            fetched = getter()
            if isinstance(fetched, list):
                keys = fetched
            elif isinstance(fetched, str):
                keys = [fetched]
        except Exception:
            keys = []
    if not keys:
        raw_keys = _provider_config(provider).get("key", [])
        if isinstance(raw_keys, list):
            keys = raw_keys
        elif isinstance(raw_keys, str):
            keys = [raw_keys]

    for item in keys:
        value = str(item or "").strip()
        if value:
            return value
    return ""


def normalize_gemini_base_url(raw_base_url: str) -> str:
    raw = str(raw_base_url or "").strip().rstrip("/")
    if not raw:
        raw = DEFAULT_GEMINI_BASE_URL

    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        parsed = urlparse(DEFAULT_GEMINI_BASE_URL)

    path = (parsed.path or "").rstrip("/")
    if path.endswith("/openai"):
        path = path[: -len("/openai")]
    if path.endswith("/v1beta") or path.endswith("/v1"):
        versioned_path = path
    elif path:
        versioned_path = f"{path}/v1beta"
    else:
        versioned_path = "/v1beta"

    return urlunparse(
        (parsed.scheme, parsed.netloc, versioned_path, "", "", "")
    ).rstrip("/")


def _api_url(base_url: str, path: str) -> str:
    parsed = urlparse(base_url)
    base_path = (parsed.path or "").rstrip("/")
    joined = f"{base_path}/{str(path or '').lstrip('/')}"
    return urlunparse((parsed.scheme, parsed.netloc, joined, "", "", ""))


def _upload_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    path = parsed.path or "/v1beta"
    version = "v1"
    if "v1beta" in path:
        version = "v1beta"
    return urlunparse(
        (parsed.scheme, parsed.netloc, f"/upload/{version}/files", "", "", "")
    )


def is_gemini_provider(provider: Any) -> bool:
    cfg = _provider_config(provider)
    tokens = [
        provider_label(provider),
        provider_model(provider),
        str(cfg.get("api_base") or ""),
        str(cfg.get("type") or ""),
        str(cfg.get("provider_source_id") or ""),
    ]
    return any("gemini" in token.lower() for token in tokens)


def _model_path(model: str) -> str:
    clean = str(model or "").strip().strip("/")
    if not clean:
        raise GeminiVideoError("Gemini provider has no configured model.")
    if clean.startswith("models/"):
        return clean
    return f"models/{clean}"


def _mime_type(path: str) -> str:
    mime_type = mimetypes.guess_type(path)[0] or "video/mp4"
    return mime_type if mime_type.startswith("video/") else "video/mp4"


def _file_uri_to_path(file_uri: str) -> str:
    parsed = urlparse(file_uri)
    if parsed.scheme != "file":
        return file_uri
    netloc = unquote(parsed.netloc or "")
    path = unquote(parsed.path or "")
    if netloc and netloc != "localhost":
        path = f"//{netloc}{path}"
    return path


async def _download_to_temp(url: str, temp_dir: Path, timeout_sec: float) -> str:
    try:
        import aiohttp
    except Exception as exc:  # pragma: no cover - environment guard
        raise GeminiVideoError("aiohttp is required to download video URLs.") from exc

    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix or ".mp4"
    temp_dir.mkdir(parents=True, exist_ok=True)
    local_path = temp_dir / f"gemini_video_{uuid.uuid4().hex}{suffix}"
    timeout = aiohttp.ClientTimeout(total=None if timeout_sec <= 0 else timeout_sec)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            text = await resp.text() if resp.status >= 400 else ""
            if resp.status >= 400:
                raise GeminiVideoError(
                    f"Failed to download video: HTTP {resp.status} {text[:200]}"
                )
            with local_path.open("wb") as f:
                async for chunk in resp.content.iter_chunked(1024 * 256):
                    if chunk:
                        f.write(chunk)
    if not local_path.exists() or local_path.stat().st_size <= 0:
        raise GeminiVideoError("Downloaded video file is empty.")
    return str(local_path)


async def resolve_video_ref_to_local_path(
    video_ref: str,
    *,
    temp_dir: str | Path | None = None,
    timeout_sec: float = 120.0,
) -> tuple[str, bool]:
    clean = str(video_ref or "").strip()
    if not clean:
        raise GeminiVideoError("Video reference is empty.")

    if clean.startswith("file://"):
        clean = _file_uri_to_path(clean)

    path = Path(clean)
    if path.exists() and path.is_file():
        return str(path), False

    if clean.startswith(("http://", "https://")):
        base_temp = Path(temp_dir) if temp_dir else Path("/tmp/astrbot_gemini_video")
        return await _download_to_temp(clean, base_temp, timeout_sec), True

    raise GeminiVideoError(f"Video file not found or unsupported reference: {clean}")


async def _upload_file(
    session: Any,
    base_url: str,
    api_key: str,
    path: str,
    *,
    proxy_url: str = "",
) -> dict[str, Any]:
    file_size = os.path.getsize(path)
    mime_type = _mime_type(path)
    display_name = Path(path).name
    start_headers = {
        "x-goog-api-key": api_key,
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "Content-Type": "application/json",
    }
    async with session.post(
        _upload_url(base_url),
        headers=start_headers,
        json={"file": {"display_name": display_name}},
        proxy=proxy_url or None,
    ) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise GeminiVideoError(f"Gemini file upload start failed: {resp.status} {text[:500]}")
        upload_uri = resp.headers.get("x-goog-upload-url")
        if not upload_uri:
            raise GeminiVideoError("Gemini file upload start did not return upload URL.")

    upload_headers = {
        "Content-Length": str(file_size),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize",
    }
    with open(path, "rb") as f:
        async with session.post(
            upload_uri,
            headers=upload_headers,
            data=f,
            proxy=proxy_url or None,
        ) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise GeminiVideoError(f"Gemini file upload failed: {resp.status} {text[:500]}")
            try:
                data = await resp.json(content_type=None)
            except Exception as exc:
                raise GeminiVideoError(f"Gemini file upload returned invalid JSON: {text[:500]}") from exc
    file_info = data.get("file") if isinstance(data, dict) else None
    if not isinstance(file_info, dict) or not file_info.get("uri"):
        raise GeminiVideoError("Gemini file upload did not return file.uri.")
    file_info.setdefault("mimeType", mime_type)
    return file_info


async def _wait_file_active(
    session: Any,
    base_url: str,
    api_key: str,
    file_info: dict[str, Any],
    timeout_sec: float,
    proxy_url: str = "",
) -> dict[str, Any]:
    name = str(file_info.get("name") or "").strip()
    if not name:
        return file_info
    deadline = time.monotonic() + (timeout_sec if timeout_sec > 0 else 300.0)
    current = dict(file_info)
    headers = {"x-goog-api-key": api_key}
    while True:
        state = str(current.get("state") or "").upper()
        if state in {"", "ACTIVE"}:
            return current
        if state == "FAILED":
            raise GeminiVideoError(f"Gemini file processing failed: {current}")
        if time.monotonic() >= deadline:
            raise GeminiVideoError(f"Timed out waiting for Gemini file to become ACTIVE: {state}")
        await asyncio.sleep(2.0)
        async with session.get(
            _api_url(base_url, name),
            headers=headers,
            proxy=proxy_url or None,
        ) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise GeminiVideoError(f"Gemini file status failed: {resp.status} {text[:500]}")
            current = await resp.json(content_type=None)
            if not isinstance(current, dict):
                raise GeminiVideoError(f"Gemini file status returned invalid JSON: {text[:500]}")


def _extract_gemini_text(data: dict[str, Any]) -> str:
    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        return ""
    parts: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        raw_parts = content.get("parts")
        if not isinstance(raw_parts, list):
            continue
        for part in raw_parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
    return "\n".join(part for part in parts if part.strip()).strip()


async def _generate_caption(
    session: Any,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    file_info: dict[str, Any],
    proxy_url: str = "",
    max_retries: int = 2,
) -> str:
    file_uri = str(file_info.get("uri") or "").strip()
    mime_type = str(file_info.get("mimeType") or file_info.get("mime_type") or "video/mp4")
    if not file_uri:
        raise GeminiVideoError("Gemini uploaded file is missing uri.")
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "file_data": {
                            "mime_type": mime_type,
                            "file_uri": file_uri,
                        }
                    },
                ],
            }
        ],
        "generationConfig": {"temperature": 0.2},
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    attempts = max(0, int(max_retries)) + 1
    for attempt in range(attempts):
        async with session.post(
            _api_url(base_url, f"{_model_path(model)}:generateContent"),
            headers=headers,
            json=body,
            proxy=proxy_url or None,
        ) as resp:
            text = await resp.text()
            if resp.status >= 400:
                if resp.status in RETRYABLE_HTTP_STATUSES and attempt < attempts - 1:
                    await asyncio.sleep(
                        _retry_delay(attempt, resp.headers.get("Retry-After", ""))
                    )
                    continue
                raise GeminiVideoError(
                    f"Gemini video caption failed: {resp.status} {text[:500]}"
                )
            data = await resp.json(content_type=None)
            if not isinstance(data, dict):
                raise GeminiVideoError(
                    f"Gemini video caption returned invalid JSON: {text[:500]}"
                )
            return _extract_gemini_text(data)
    return ""


async def _delete_file(
    session: Any,
    base_url: str,
    api_key: str,
    file_info: dict[str, Any],
    proxy_url: str = "",
) -> None:
    name = str(file_info.get("name") or "").strip()
    if not name:
        return
    try:
        async with session.delete(
            _api_url(base_url, name),
            headers={"x-goog-api-key": api_key},
            proxy=proxy_url or None,
        ):
            return
    except Exception:
        return


async def caption_video_with_gemini_provider(
    provider: Any,
    video_ref: str,
    prompt: str,
    *,
    timeout_sec: float = 120.0,
    temp_dir: str | Path | None = None,
    proxy_url: str = "",
    max_retries: int = 2,
) -> str:
    if not is_gemini_provider(provider):
        raise GeminiVideoError(
            f"Provider `{provider_label(provider)}` is not a Gemini provider; video upload is unsupported."
        )

    cfg = _provider_config(provider)
    api_key = provider_api_key(provider)
    if not api_key:
        raise GeminiVideoError(f"Provider `{provider_label(provider)}` has no Gemini API key.")
    model = provider_model(provider)
    if not model:
        raise GeminiVideoError(f"Provider `{provider_label(provider)}` has no Gemini model.")
    base_url = normalize_gemini_base_url(str(cfg.get("api_base") or ""))

    try:
        import aiohttp
    except Exception as exc:  # pragma: no cover - environment guard
        raise GeminiVideoError("aiohttp is required for Gemini video upload.") from exc

    local_path = ""
    cleanup_local = False
    file_info: dict[str, Any] = {}
    effective_timeout = float(timeout_sec or 0)
    timeout = aiohttp.ClientTimeout(total=None if effective_timeout <= 0 else effective_timeout)
    try:
        local_path, cleanup_local = await resolve_video_ref_to_local_path(
            video_ref,
            temp_dir=temp_dir,
            timeout_sec=effective_timeout,
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            file_info = await _upload_file(
                session,
                base_url,
                api_key,
                local_path,
                proxy_url=proxy_url,
            )
            file_info = await _wait_file_active(
                session,
                base_url,
                api_key,
                file_info,
                effective_timeout,
                proxy_url=proxy_url,
            )
            return await _generate_caption(
                session,
                base_url,
                api_key,
                model,
                prompt,
                file_info,
                proxy_url=proxy_url,
                max_retries=max_retries,
            )
    finally:
        if file_info:
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    await _delete_file(
                        session,
                        base_url,
                        api_key,
                        file_info,
                        proxy_url=proxy_url,
                    )
            except Exception:
                pass
        if cleanup_local and local_path:
            try:
                Path(local_path).unlink(missing_ok=True)
            except Exception:
                pass
