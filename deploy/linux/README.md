# Routelix on Ubuntu 24.04 (aarch64 or amd64)

The public site is [https://routelix.induswebagency.com](https://routelix.induswebagency.com). nginx terminates TLS and proxies to the app on `127.0.0.1:8787`. Ports 1080, 5001, 9222, 12525, and 27017 are left alone.

Google Chrome has no Linux arm64 build. The installer uses Chromium from the xtradeb PPA or Playwright's Chromium, plus chromedriver. It does not use the snap.

## 1. DNS and the checkout

Point `routelix.induswebagency.com` at the VPS, then:

```bash
sudo mkdir -p /opt/routelix
sudo chown "$USER" /opt/routelix
git clone <this-repo> /opt/routelix
cd /opt/routelix
sudo bash deploy/linux/install.sh
```

`install.sh` installs apt packages, creates the `routelix` user, builds `.venv` from `requirements-linux.txt`, downloads the Piper lessac voice, the faster-whisper `base` model (int8 at runtime), and a Qwen2.5 1.5B Q4 GGUF. Skip the GGUF with `sudo SKIP_LLM_DOWNLOAD=1 bash deploy/linux/install.sh` if you will use Groq for answers.

Kokoro is supported (`TTS_ENGINE=kokoro`) but the ONNX files are not downloaded automatically. Place them at the paths in `.env`.

## 2. Environment

```bash
sudoedit /opt/routelix/.env
```

Set at least:

- `ROUTELIX_SECRET_KEY` (install.sh fills a random value when it creates the file)
- `ROUTELIX_ADMIN_PASSWORD` (10 characters or more)
- `CALLBACK_NUMBER`
- `GROQ_API_KEY` only if you want cloud speech or answers. Then set `STT_BACKEND=groq` and/or `LLM_BACKEND=groq`. With the key empty, keep `STT_BACKEND=local` and `LLM_BACKEND=local`.

Every variable is listed in `deploy/linux/.env.example`.

## 3. Services

```bash
sudo systemctl restart routelix-audio routelix
sudo systemctl status routelix --no-pager
curl -fsS http://127.0.0.1:8787/healthz
```

`routelix.service` sets `MemoryMax=8G`, `Restart=always`, and listens on `127.0.0.1:8787`. `routelix-audio.service` runs PipeWire (or PulseAudio) as the `routelix` user so Chromium and the dialer share `/run/routelix`.

## 4. nginx and the certificate

```bash
sudo cp /opt/routelix/deploy/linux/nginx/routelix-upgrade.conf /etc/nginx/conf.d/routelix-upgrade.conf
sudo cp /opt/routelix/deploy/linux/nginx/routelix.conf /etc/nginx/sites-available/routelix
sudo ln -sfn /etc/nginx/sites-available/routelix /etc/nginx/sites-enabled/routelix
sudo nginx -t
sudo systemctl reload nginx
sudo certbot --nginx -d routelix.induswebagency.com
```

Certbot adds the TLS server block. The upgrade map is there for WebSockets. The live run uses server-sent events (`/api/run/stream`); the site config turns proxy buffering off for that stream.

Open `https://routelix.induswebagency.com/`, then log in at `/login`.

## 5. Google Voice sign-in (manual, once per profile)

Headless Chromium cannot finish a Google password, 2FA, or robot check. Two supported options:

**Interactive noVNC, localhost only**

```bash
sudo systemctl start routelix-audio
sudo -u routelix bash /opt/routelix/deploy/linux/gv-login.sh sales_profile
```

From your laptop:

```bash
ssh -L 6080:127.0.0.1:6080 <user>@<server>
```

Open `http://127.0.0.1:6080/vnc.html`, sign in at voice.google.com, and confirm the dial pad is visible. Stop the headed Chromium. Later live runs reuse `chrome_profiles/sales_profile` with `CHROME_HEADLESS=true`.

**Copy a profile that is already signed in**

On a machine where Chromium is already logged into Google Voice, copy that user-data directory to the server and:

```bash
sudo -u routelix /opt/routelix/.venv/bin/python -m src.gv_session import-profile \
  --from /path/to/chromium-user-data --name sales_profile
```

A JSON cookie export is only a best-effort staging file:

```bash
sudo -u routelix /opt/routelix/.venv/bin/python -m src.gv_session import-cookies \
  --from cookies.json --name sales_profile
```

Google often rejects imported session cookies. If the dial pad does not appear, use noVNC.

## 6. Call audio

`sounddevice` only lists `pulse` and `default` on this stack, so TTS does not look up `routelix_to_gv` as a PortAudio device. It plays with `paplay`/`pacat --device=routelix_to_gv` and `PULSE_SINK`. Speech-to-text records with `parec` and `PULSE_SOURCE=routelix_from_gv.monitor`.

Two PipeWire issues block that playback until the deploy files are installed:

- A system service has no interactive logind seat. WirePlumber's default `with-logind=true` makes it disconnect, the null sink stays `SUSPENDED`, and `paplay` waits forever. `bluetooth.lua.d/51-no-logind.lua` turns that off.
- Idle sinks are suspended after a few seconds. The null sinks are created with `session.suspend-timeout-seconds=0` (space-separated `sink_properties`; commas are ignored). WirePlumber 0.4 and 0.5 drop-ins repeat that rule.

`routelix-audio.service` starts PipeWire, WirePlumber, and pipewire-pulse, then loads the sinks.

After pulling this update on a box that already ran the audio service:

```bash
sudo bash deploy/linux/install.sh
sudo systemctl restart routelix-audio routelix
sudo -u routelix env XDG_RUNTIME_DIR=/run/routelix PULSE_SERVER=unix:/run/routelix/pulse/native \
  /opt/routelix/.venv/bin/python -m src.main --audio-route-test
```

Preflight fails if a short `pacat` write to `routelix_to_gv` times out, which is the suspended-sink failure.

Before the first live run the process creates two null sinks:

- `routelix_to_gv` — Piper/Kokoro/Edge playback. Chromium's microphone is `routelix_to_gv.monitor`.
- `routelix_from_gv` — Chromium's speaker. Speech-to-text records `routelix_from_gv.monitor`.

Run Preflight, then Audio → test TTS and test capture, before a campaign. A VPS has no room speakers. Listen-in cannot play into a physical speaker unless the machine has one; use the Recordings tab. If Chromium does not attach to the sinks, the call will be one-way or silent even though the console is up. That is a browser/Pulse limitation, not something the installer can guarantee.

## 7. Concurrent agents

One Google Voice account is one live call. Add more accounts in Settings (Connect uses `ensure_logged_in`) to add agents. `MAX_AGENTS=auto` admits only as many as the CPU budget allows after `ROUTELIX_RESERVED_CORES` (default 1) and `ROUTELIX_RESERVED_RAM_GB` (default 8, for MailForge/MongoDB and Ollama). On a 4-core / 24 GB host that cap is 4. RAM would allow more. The dashboard shows the cap, the active slots, and the queue.

Each admitted account gets `routelix_to_gv_<n>` and `routelix_from_gv_<n>`. Speech-to-text uses `STT_LOCAL_MODEL=tiny` and `STT_THREADS=1` (measured 26x realtime for an 8 second clip on one x86 thread; about 16x on ARM at the default 0.6 scale). The local LLM is one shared queue (`LLM_PARALLEL=1`, `LLM_THREADS=2`), or `LLAMA_SERVER_URL` if you run `llama-server --parallel 1`.

Optional per-slot process, restarted by systemd:

```bash
sudo systemctl enable --now routelix-agent@1
```

The console's parallel dialer also restarts a crashed slot browser up to 3 times. Per account, calls wait `ACCOUNT_MIN_GAP_SECONDS` (45) and stop after `ACCOUNT_MAX_CALLS_PER_HOUR` (30).

## 8. Operators

The admin created from `ROUTELIX_ADMIN_PASSWORD` opens **Users** and adds operator accounts. Passwords are PBKDF2-SHA256 hashes in `data/users.json`. Login attempts are limited to 8 failures per 15 minutes per IP in this process.

## 9. Scripts and extra voices

The call script is edited at `/scripts`. Built-in templates cover a generic B2B appointment and INDUS Web Agency work: web development, SEO, AI bots and automation, and MailForge. Set `ACTIVE_SCRIPT_ID` (for example `indus-web`) so a live call follows that script. The page can type or hold-to-speak as the prospect. Nothing on that page dials.

`install.sh` fetches Piper lessac medium. More voices:

```bash
bash deploy/linux/download_voices.sh en_US-lessac-medium en_GB-alba-medium es_MX-ald-medium fr_FR-siwis-medium
```

Files land in `models/piper` and are not committed. Piper has no Urdu, Punjabi, Sindhi, or Haitian Creole voice. Kokoro needs its own model download and was not timed on this host. Lessac medium measured 47.6x realtime on a 4-vCPU x86 box (12.3 seconds of speech in 0.26 seconds). `/voices` shows that factor and an estimate for the engines that were not timed.
