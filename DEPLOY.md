# Deploying D&V Team Analysis to Railway

This guide walks through pushing the current folder to GitHub, deploying it to
Railway, and pointing it at match data hosted on SharePoint via rclone.

---

## 1. Push to GitHub

From this folder (`video_base/`):

```powershell
# If you haven't initialised yet:
git init
git branch -M main

# Stage everything not excluded by .gitignore.
git add .
git status        # sanity-check: NO .mp4, NO SciSports JSON, NO rclone.exe
git commit -m "Initial commit: Railway-ready D&V Team Analysis"

# Create a GitHub repo (web UI or `gh repo create`), then:
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

What gets pushed:
- All `.py` files (`app.py`, `event_parser.py`, `fcdb_corner_inference.py`,
  `magnet_board.py`, `video_utils.py`)
- All PNG pitch images (`16m area.png`, `no_names_*.png`, `left_side_corner.png`,
  `right_side_corner.png`, `PA (3).png`, `example_corner_zones.png`)
- The model artefacts (`defender_role_rf.joblib`, `attacker_role_rf.joblib`,
  `feature_columns.json`)  ≈ 16 MB total — fine for git
- Config: `requirements.txt`, `packages.txt`, `runtime.txt`, `Procfile`,
  `railway.toml`, `.streamlit/config.toml`
- An **empty-ish** `videos.json` (you may want to commit a placeholder; see §3)

What's excluded (lives only on SharePoint):
- `*.mp4` videos
- `*SciSportsEvents*.json`, `*SciSportsPositions*.json`, `*.xml`
- `rclone.exe`
- Scratch dirs (`example_code_to_be_deleted/`, `claude-swap/`, `video_c_base/`)

---

## 2. Create the Railway project

1. **Railway dashboard → New Project → Deploy from GitHub repo**, pick this repo.
2. Railway auto-detects the `Procfile` / `railway.toml` and uses Nixpacks to
   build (the `packages.txt` file ensures `ffmpeg` is installed at build time).
3. First deploy will FAIL because there's no data yet — that's expected.
4. Open the service's **Settings → Public Networking** and **Generate Domain**
   so you have a public URL.

---

## 3. Mount SharePoint via rclone (same flow you used before)

Because you already connected another Railway app to this SharePoint, you have
the credentials. Steps for this app:

1. **Service → Variables**, add:
   - `DATA_DIR` = `/data/match-data` *(or wherever you'll mount SharePoint)*
   - `RCLONE_CONFIG_<REMOTE>_<KEYS>` — copy these from the other Railway app's
     variable list. They're the auth tokens for your SharePoint remote.

2. **Service → Settings → Volumes**, attach a Railway Volume mounted at
   `/data`. (Or skip and let rclone create the mount in-memory; the volume is
   only needed if you want clip extraction to persist between restarts.)

3. **Replace the start command** so rclone mounts SharePoint before Streamlit
   starts. In **Service → Settings → Custom Start Command**, paste:

   ```bash
   # Install rclone (Nixpacks doesn't have it by default)
   curl https://rclone.org/install.sh | bash && \
   mkdir -p /data/match-data && \
   rclone mount <REMOTE>:<SHAREPOINT_PATH> /data/match-data --daemon --vfs-cache-mode full && \
   sleep 3 && \
   streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false
   ```

   Replace `<REMOTE>` with your rclone remote name (e.g. `sp` or `sharepoint`)
   and `<SHAREPOINT_PATH>` with the folder path on SharePoint that holds the
   match data. This should match the layout from the other working Railway app.

4. Hit **Deploy**. Railway rebuilds, mounts SharePoint, then launches Streamlit
   reading from that mount.

---

## 4. SharePoint folder layout

The app expects the same flat layout that works locally:

```
<SharePoint root>/
  videos.json
  <prefix> SciSportsEvents - <match_id>.json
  <prefix> SciSportsPositions - <match_id>.json
  <prefix> <camera tag>.mp4          (e.g. " Broadcast 1234.mp4")
  ...
```

In `videos.json`, **use bare filenames** instead of absolute Windows paths
— the app resolves any non-absolute path against `DATA_DIR`. Example:

```json
{
  "20260328 FC Eindhoven vs FC Emmen": {
    "offset": 22,
    "Broadcast":  "20260328 FC Eindhoven vs FC Emmen 69c5c880c845b6ec6116d080.mp4",
    "Goal Right": "20260328 FC Eindhoven vs FC Emmen Goal Right 69c5c880c845b6ec6116d080.mp4",
    "Goal Left":  "20260328 FC Eindhoven vs FC Emmen Goal Left 69c5c880c845b6ec6116d080.mp4"
  }
}
```

This file should live **inside** the SharePoint folder (so it's part of the
mount), not in the repo. Add a placeholder `videos.json` to the repo if you
want — but the mounted one will take precedence because the app reads from
`DATA_DIR`.

---

## 5. First-load checklist

When you hit the Railway URL:

- Sidebar populates with matches → ✅ rclone mount is reading correctly.
- "No match files found" → rclone hasn't mounted yet, or `DATA_DIR` env var is
  wrong. SSH in (`railway shell`) and run `ls $DATA_DIR` — should list the
  SciSports JSONs.
- Sidebar populates but **clip fails** → `videos.json` paths are wrong, OR the
  SharePoint folder doesn't contain the .mp4 yet. Check `ls $DATA_DIR/*.mp4`.
- Yellow "Role predictions are disabled" warning → `defender_role_rf.joblib` /
  `attacker_role_rf.joblib` / `feature_columns.json` are missing from the
  repo. They should be in git (they're <16 MB total).

---

## 6. Local dev still works

No `DATA_DIR` env var → the app falls back to `Path(__file__).parent`, exactly
as before. Your existing local workflow is unchanged.

If you want to point your local app at the SharePoint mount on Windows, set
the env var in your shell:

```powershell
$env:DATA_DIR = "Z:\path\to\sharepoint\mount"
streamlit run app.py
```

---

## 7. Performance notes for the Railway deploy

- **`ffmpeg`** is installed automatically via `packages.txt`.
- **Plan size**: at least `Starter` (512 MB). Positions JSONs are loaded into
  memory (~60 MB each), so under heavy use 1 GB headroom is comfortable.
- **rclone vfs-cache-mode**: `full` is best for video streaming (caches read
  ranges). Set `--vfs-cache-max-size 10G` if your volume is sized accordingly.
- **Clip extraction** uses ffmpeg's stream-copy (no re-encode), so even over a
  network mount each clip takes 200–800 ms after the first hit.
- Streamlit caches `_cached_positions`, `_cached_extract_clip`, and the corner
  analyser results across reruns within a session, so repeat clicks are
  essentially free.
