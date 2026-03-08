# Festoes — Setup & Deployment Guide

## Running on Your Phone (Pydroid 3)

### Step 1 — Install Pydroid 3
Download **Pydroid 3** from the Play Store.
The paid version (Pydroid 3 Premium) is not required.

### Step 2 — Copy files to your phone
Copy these files to:
`/sdcard/Android/data/ru.iiec.pydroid3/files/`

Or any folder you prefer — just remember where you put it.

Required file:
- `festoes_v9_1.py`

Optional but recommended:
- `skills/` folder (with the 3 example skills)

### Step 3 — Install pip packages
Open Pydroid 3 → tap **≡** (menu) → **Pip** → **Install package**

Install these one at a time:
```
numpy
plyer
chess
```

Optional for advanced math:
```
sympy
```

### Step 4 — Add your API keys
Open `festoes_v9_1.py` in Pydroid 3's editor.

Find these lines near the top and fill in your keys:
```python
GEMINI_API_KEY        = ""   # get free at: ai.google.dev
WEATHER_API_KEY       = ""   # get free at: openweathermap.org/api
GOOGLE_SEARCH_API_KEY = ""   # optional: console.cloud.google.com
GOOGLE_SEARCH_ENGINE_ID = "" # optional: cse.google.com
```

You can skip Google Search — DuckDuckGo works without any key.

### Step 5 — Run
Open `festoes_v9_1.py` → tap the **Run** button (▶).

**First run:** Festoes trains all 1,000 networks (~30 seconds). You'll see a progress message. After this, it loads in ~2 seconds every time.

---

## Pushing to GitHub

### Step 1 — Prepare the file
Before pushing, **remove your API keys** from `festoes_v9_1.py`:
```python
GEMINI_API_KEY        = ""   # removed for public release
WEATHER_API_KEY       = ""
GOOGLE_SEARCH_API_KEY = ""
GOOGLE_SEARCH_ENGINE_ID = ""
```

### Step 2 — Create a GitHub repo
1. Go to github.com → New repository
2. Name: `festoes` (or `festoes-ai`)
3. Description: `Local AI chatbot for Android — 1,000 neural networks + Gemini`
4. Set to Public
5. Do NOT add a README (we have our own)

### Step 3 — Upload files
Upload these files to the repo:
```
festoes_v9_1.py       — main app
README.md             — documentation
requirements.txt      — dependencies
.gitignore            — ignore rules
skills/               — example skills folder
  weather_skill.py
  sympy_skill.py
  chess_openings_skill.py
```

Do NOT upload:
```
*.npz                 — too large (binary weights)
*.json                — personal user data
*.db                  — user database
```

### Step 4 — Tag a release
In GitHub → Releases → Create new release
- Tag: `v9.1`
- Title: `Festoes v9.1 — Production Release`
- Description: paste from README version history

---

## Sharing on itch.io (optional)

1. Go to itch.io → Create new project
2. Kind: **Tool**
3. Title: `Festoes — Local AI for Android`
4. Upload a `.zip` of the release files
5. Set price to free (or pay-what-you-want)
6. Add tags: `android`, `python`, `ai`, `chatbot`, `pydroid`

---

## Running on PC / Mac / Linux (for development)

Festoes also runs on desktop Python:
```bash
pip install numpy plyer chess sympy
python festoes_v9_1.py
```

Note: Voice input (`droid.recognizeSpeech`) only works on Android. On desktop the MIC button will fail silently — type instead.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: numpy` | Open Pip in Pydroid 3, install numpy |
| `Voice not available` | This is normal on some setups — type instead |
| `Gemini quota reached` | Free tier resets daily at ~10am Nairobi time |
| `Google search 403` | Enable Custom Search API at console.cloud.google.com |
| App crashes on start | Delete `*.npz` files — Festoes will retrain |
| Weights mismatch error | Delete `*.npz` and `festoes_v9_vocab.json` — retrain |

---

## Resetting Festoes

To do a clean reset (retrain from scratch):
1. Delete: `festoes_v9_weights.npz`, `festoes_v9_micro_weights.npz`, `festoes_v9_vocab.json`
2. Optionally delete: `pybot_history.json`, `festoes_v9_1.db`
3. Run Festoes — it will retrain automatically

Your settings (`pybot_settings.json`) and goals are preserved.
