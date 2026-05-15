"""OneDrive / Microsoft Graph sync layer.

Authenticates with an Azure AD app registration (client credentials flow) and
exposes two operations the app needs:

1. ``sync_metadata()`` — downloads all the small files (SciSports JSON, XML,
   videos.json) from the configured OneDrive folder into a local cache
   directory, then rewrites ``videos.json`` so each camera entry becomes a
   ``graph:<filename>`` placeholder. After this runs, ``DATA_DIR`` points at
   the cache and the rest of the app can read files normally.

2. ``resolve_video_url(url)`` — turns a ``graph:<filename>`` placeholder back
   into a short-lived signed download URL that ffmpeg can stream from. Called
   lazily right before each clip extraction so the URL is always fresh
   (Graph download URLs expire after ~1 hour).

The MP4s are NEVER fully downloaded — ffmpeg seeks into them over HTTP using
range requests against the Azure Blob URL Graph hands back. Only the JSON/XML
metadata (~60 MB per match) lives in the cache.

Required env vars (all five must be set, otherwise the module bails out and
the app falls back to local-disk mode):
    ONEDRIVE_TENANT_ID
    ONEDRIVE_CLIENT_ID
    ONEDRIVE_CLIENT_SECRET
    ONEDRIVE_USER_EMAIL       # mailbox whose OneDrive holds the files
    ONEDRIVE_BASE_FOLDER      # e.g. "Video Database"
Optional:
    ONEDRIVE_CACHE_DIR        # default: /data/onedrive_cache (Railway Volume)
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import quote

import msal
import requests


GRAPH_BASE = "https://graph.microsoft.com/v1.0"
GRAPH_SCOPES = ["https://graph.microsoft.com/.default"]
_DEFAULT_CACHE_DIR = Path(os.environ.get("ONEDRIVE_CACHE_DIR", "/data/onedrive_cache"))


class OneDriveError(RuntimeError):
    pass


class OneDriveSync:
    def __init__(self):
        # Pull every required env var up-front so we fail loudly at startup
        # rather than mid-rerun.
        missing = [k for k in (
            "ONEDRIVE_TENANT_ID",
            "ONEDRIVE_CLIENT_ID",
            "ONEDRIVE_CLIENT_SECRET",
            "ONEDRIVE_USER_EMAIL",
            "ONEDRIVE_BASE_FOLDER",
        ) if not os.environ.get(k)]
        if missing:
            raise OneDriveError(
                "OneDrive sync enabled but these env vars are missing: "
                + ", ".join(missing)
            )
        self.tenant_id     = os.environ["ONEDRIVE_TENANT_ID"]
        self.client_id     = os.environ["ONEDRIVE_CLIENT_ID"]
        self.client_secret = os.environ["ONEDRIVE_CLIENT_SECRET"]
        self.user_email    = os.environ["ONEDRIVE_USER_EMAIL"]
        self.base_folder   = os.environ["ONEDRIVE_BASE_FOLDER"].strip("/")
        self.cache_dir     = _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._app = msal.ConfidentialClientApplication(
            self.client_id,
            authority=f"https://login.microsoftonline.com/{self.tenant_id}",
            client_credential=self.client_secret,
        )
        self._token: str | None = None
        self._token_expiry: float = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    def _access_token(self) -> str:
        """Return a valid app-only access token, refreshing if within 60s of
        expiry. MSAL handles its own in-memory token cache so a refresh is
        only an HTTPS roundtrip if the previous token is stale."""
        with self._lock:
            if self._token and time.time() < self._token_expiry - 60:
                return self._token
            res = self._app.acquire_token_for_client(scopes=GRAPH_SCOPES)
            if "access_token" not in res:
                raise OneDriveError(
                    f"Graph token request failed: "
                    f"{res.get('error')}: {res.get('error_description')}"
                )
            self._token       = res["access_token"]
            self._token_expiry = time.time() + int(res.get("expires_in", 3600))
            return self._token

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._access_token()}"}

    def _drive_root(self) -> str:
        # /users/<email>/drive resolves the user's personal OneDrive. For
        # SharePoint Online sites the URL would be /sites/<site-id>/drive.
        return f"{GRAPH_BASE}/users/{quote(self.user_email)}/drive"

    def _item_url(self, *, relative_path: str) -> str:
        """Build a /root:/<path> URL — Graph's by-path addressing.

        Path segments are individually URL-encoded so spaces, parentheses and
        unicode in folder/filenames don't break the request."""
        full = f"{self.base_folder}/{relative_path}".strip("/")
        encoded = "/".join(quote(seg, safe="") for seg in full.split("/"))
        return f"{self._drive_root()}/root:/{encoded}"

    # ------------------------------------------------------------------
    # Listing & downloading
    # ------------------------------------------------------------------
    def list_files(self) -> list[dict]:
        """Return all DriveItem dicts under the base folder (one level deep).
        Includes ``@microsoft.graph.downloadUrl`` so we can grab signed URLs
        without a second roundtrip."""
        url = (self._item_url(relative_path="") + ":/children"
               "?$select=name,size,file,folder,@microsoft.graph.downloadUrl"
               "&$top=200")
        out = []
        while url:
            r = requests.get(url, headers=self._headers(), timeout=30)
            if r.status_code == 401:
                # Token might have raced expiry — force-refresh once.
                self._token = None
                r = requests.get(url, headers=self._headers(), timeout=30)
            r.raise_for_status()
            data = r.json()
            out.extend(data.get("value", []))
            url = data.get("@odata.nextLink")
        return out

    def get_download_url(self, filename: str) -> str:
        """Return a short-lived signed URL for streaming `filename`.
        Used for the MP4s — ffmpeg reads from this URL with HTTP range
        requests, so we don't pay the cost of downloading multi-GB files."""
        url = (self._item_url(relative_path=filename)
               + "?$select=@microsoft.graph.downloadUrl")
        r = requests.get(url, headers=self._headers(), timeout=30)
        r.raise_for_status()
        body = r.json()
        link = body.get("@microsoft.graph.downloadUrl")
        if not link:
            raise OneDriveError(f"No downloadUrl returned for {filename}")
        return link

    def download_to_cache(self, filename: str, *, overwrite: bool = False) -> Path:
        """Download `filename` from OneDrive into the local cache. Streams to
        disk so even 100 MB tracking files don't balloon memory."""
        local = self.cache_dir / filename
        if local.exists() and not overwrite:
            return local
        local.parent.mkdir(parents=True, exist_ok=True)
        link = self.get_download_url(filename)
        with requests.get(link, stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = local.with_suffix(local.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp.replace(local)
        return local

    # ------------------------------------------------------------------
    # Bootstrap: sync metadata and rewrite videos.json
    # ------------------------------------------------------------------
    def sync_metadata(self) -> Path:
        """Pull every JSON/XML in the base folder down to the cache, and
        rewrite videos.json so MP4 entries are ``graph:<filename>`` placeholders
        that ``resolve_video_url`` can turn into fresh URLs on demand.

        Returns the cache directory (caller assigns it to DATA_DIR).
        """
        items = self.list_files()
        for item in items:
            if "file" not in item:
                continue
            name = item.get("name", "")
            lower = name.lower()
            if lower.endswith(".json") or lower.endswith(".xml"):
                # Re-download videos.json every boot so config changes pick up
                # without manual cache busting. Other files only on first miss.
                overwrite = (name == "videos.json")
                self.download_to_cache(name, overwrite=overwrite)

        # Rewrite videos.json so the camera entries don't hold Windows paths
        # (which won't resolve on Railway) — replace them with graph:<filename>
        # placeholders. Bare filenames inside the cache dir would also work,
        # but graph: is explicit and survives the case where Railway's volume
        # doesn't persist the .mp4 itself.
        videos_path = self.cache_dir / "videos.json"
        if videos_path.exists():
            try:
                cfg = json.loads(videos_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cfg = {}
            changed = False
            for match_prefix, m in cfg.items():
                if not isinstance(m, dict):
                    continue
                for k, v in list(m.items()):
                    if k == "offset" or not isinstance(v, str):
                        continue
                    name = Path(v).name  # strips any Windows/Posix path prefix
                    placeholder = f"graph:{name}"
                    if v != placeholder:
                        m[k] = placeholder
                        changed = True
            if changed:
                videos_path.write_text(
                    json.dumps(cfg, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        return self.cache_dir


# ----------------------------------------------------------------------
# Module-level singleton + helpers
# ----------------------------------------------------------------------
_singleton: OneDriveSync | None = None
_singleton_lock = threading.Lock()


def is_enabled() -> bool:
    """True if all five OneDrive env vars are set. The app falls back to
    local-disk mode when this is False (handy for local dev)."""
    return all(os.environ.get(k) for k in (
        "ONEDRIVE_TENANT_ID",
        "ONEDRIVE_CLIENT_ID",
        "ONEDRIVE_CLIENT_SECRET",
        "ONEDRIVE_USER_EMAIL",
        "ONEDRIVE_BASE_FOLDER",
    ))


def get_sync() -> OneDriveSync:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = OneDriveSync()
        return _singleton


def resolve_video_url(video_url: str) -> str:
    """If `video_url` starts with ``graph:``, swap the placeholder for a
    fresh signed OneDrive download URL (good for ~1 hour). Otherwise return
    unchanged — so local development with file paths keeps working."""
    if not isinstance(video_url, str) or not video_url.startswith("graph:"):
        return video_url
    filename = video_url[len("graph:"):]
    return get_sync().get_download_url(filename)
