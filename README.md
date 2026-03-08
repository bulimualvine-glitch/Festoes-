# Festoes v9.1
### A production-ready local AI chatbot for Android — built entirely in Python

![Version](https://img.shields.io/badge/version-9.1-green)
![Platform](https://img.shields.io/badge/platform-Android%20%2F%20Pydroid%203-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-yellow)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What is Festoes?

Festoes is a fully offline-capable AI chatbot that runs on Android via **Pydroid 3**. It combines:

- **1,000 local micro-neural-networks** (Int8 quantized, trained on 2,268 patterns)
- **Gemini 2.0 Flash** for complex questions (when online)
- **Google Custom Search + DuckDuckGo** fallback for web queries
- **SQLite3** persistence for history, settings, and learned pairs
- **Auto-backup** to `/sdcard/Festoes/backups/`
- **Skills plugin system** — drop `.py` files into `/skills/` to extend

No cloud subscription required. No heavy models. Works on a budget Android phone.

---

## Quick Start

### 1. Install Pydroid 3
Download from the Play Store: **Pydroid 3 — IDE for Python 3**

### 2. Install dependencies
Open Pydroid 3 → tap **≡** → **Pip** → install:
```
numpy
plyer
chess        (optional — for chess analysis)
sympy        (optional — for symbolic math)
```

### 3. Get API keys (free)
| Key | Where to get it | Required? |
|-----|----------------|-----------|
| Gemini API | [ai.google.dev](https://ai.google.dev) | Recommended |
| OpenWeatherMap | [openweathermap.org/api](https://openweathermap.org/api) | For weather |
| Google Custom Search | [console.cloud.google.com](https://console.cloud.google.com) | Optional |

### 4. Configure `festoes_v9_1.py`
Open the file and set your keys at the top:
```python
GEMINI_API_KEY        = "your-gemini-key-here"
WEATHER_API_KEY       = "your-weather-key-here"
GOOGLE_SEARCH_API_KEY = "your-google-key-here"   # optional
GOOGLE_SEARCH_ENGINE_ID = "your-engine-id-here"  # optional
```

### 5. Run
Open `festoes_v9_1.py` in Pydroid 3 and tap **Run**.

On first run, Festoes trains all 1,000 networks (takes ~30 seconds). After that it loads in under 2 seconds from saved weights.

---

## Features

### AI Brain
| Feature | Description |
|---------|-------------|
| 1,000 micro-networks | 64×32 architecture, expert-routed by intent |
| Int8 quantization | 4x faster inference on mobile CPUs |
| Confidence threshold | Low-confidence queries go straight to Gemini |
| Online learning | Trains on every Gemini reply |
| Vector memory | Recalls similar past conversations |

### v9 Enhancements
| Feature | Inspired by |
|---------|-------------|
| Word groups (semantic expansion) | Word2Vec |
| Positional weighting | BERT |
| Subword matching | BLOOM |
| RMS normalisation | LLaMA |
| Temperature sampling | GPT-2 |
| Expert network routing | Mixtral |
| Confidence threshold gating | Whisper |

### Built-in Skills
- **Weather** — live + offline cache fallback
- **Chess** — PGN/FEN analysis via python-chess + Gemini
- **Math** — AST-safe expression solver + sympy
- **Currency** — live exchange rates
- **Dictionary** — offline word definitions
- **News** — live headlines
- **Wikipedia** — instant summaries
- **Web Search** — Google Custom Search + DuckDuckGo fallback
- **Jokes, Riddles, Quiz, Stories** — local response engine
- **Timer, Stopwatch, Alarm, Reminders** — phone integration
- **Notes, Shopping list** — persistent local storage
- **Voice input** — SL4A `droid.recognizeSpeech()`
- **Image analysis** — Gemini Vision API

### Agent Mode
Proactive background engine that:
- Fires a morning briefing at 07:00 with weather + goals
- Reminds you of active goals at noon
- Runs silently — toggle with `agent on` / `agent off`

---

## Skills Plugin System

Drop any `.py` file into the `/skills/` folder. Festoes auto-discovers it on startup.

### Minimal skill template
```python
# skills/my_skill.py

SKILL_NAME = "my_skill"
TRIGGERS   = ["trigger word", "another trigger"]

def handle(text):
    """Return a response string, or None to pass to next handler."""
    tl = text.lower()
    if "trigger word" in tl:
        return "My skill response!"
    return None
```

### Example: Weather skill with cache
```python
# skills/weather_extended.py
def handle(text):
    if "forecast" in text.lower():
        # Use Festoes' built-in weather cache
        from festoes_v9_1 import _load_weather_cache
        cached = _load_weather_cache("Nairobi")
        if cached:
            return "Extended forecast based on: " + cached
    return None
```

### Example: Sympy math skill
```python
# skills/sympy_math.py
def handle(text):
    if "solve" in text.lower() or "integrate" in text.lower():
        try:
            import sympy as sp
            x = sp.Symbol('x')
            expr = text.lower().replace("solve", "").replace("integrate", "").strip()
            result = sp.solve(expr, x)
            return "Sympy solution: " + str(result)
        except Exception as e:
            return "Sympy error: " + str(e)
    return None
```

---

## File Structure

```
festoes_v9_1.py              — Main application
requirements.txt             — Dependencies

skills/                      — Drop plugin .py files here
  my_skill.py                — Example skill

festoes_v9_1.db              — SQLite database (auto-created)
festoes_v9_weights.npz       — Main network weights (auto-created)
festoes_v9_micro_weights.npz — Micro-network weights (auto-created)
festoes_v9_vocab.json        — Vocabulary (auto-created)
pybot_history.json           — Chat history (auto-created)
pybot_settings.json          — User settings (auto-created)
pybot_learned.json           — Online-learned pairs (auto-created)
festoes_weather_cache.json   — Offline weather cache (auto-created)
festoes_agent.json           — Agent mode state (auto-created)
festoes_goals.json           — User goals (auto-created)

/sdcard/Festoes/backups/     — Auto-backups (tap Backup button)
```

---

## Voice Input

Voice uses Pydroid 3's built-in SL4A bridge — **no extra install needed**.

Tap **MIC** → speak → Festoes transcribes and sends automatically.

> SL4A launches automatically when Pydroid 3 starts. You'll see `WARNING:root:launch SL4A with ('127.0.0.1', '8888')` in the console — this is normal and expected.

---

## Backup & Restore

**Create backup:**
Type `backup` or tap the **Backup** quick button.

Saves to: `/sdcard/Festoes/backups/festoes_YYYYMMDD_HHMM/`

Keeps the 5 most recent backups automatically.

**Restore:**
Copy the `.npz`, `.json`, and `.db` files from a backup folder back to the app directory.

---

## Quick Command Reference

| Command | What it does |
|---------|-------------|
| `search [topic]` | Google → DuckDuckGo → Gemini search |
| `weather in [city]` | Live weather (cached offline) |
| `define [word]` | Dictionary definition |
| `[math expression]` | Safe math solver |
| `100 usd to kes` | Currency conversion |
| `set timer 5 minutes` | Countdown timer |
| `wake me at 7:00am` | Alarm |
| `note [text]` | Save a note |
| `my goal is to [goal]` | Add a goal |
| `show goals` | List your goals |
| `backup now` | Backup all data |
| `agent status` | Check agent mode |
| `export chat` | Export chat to file |
| `android status` | Debug phone features |

---

## Health Check

On every startup Festoes runs a health check and displays:
```
=== v9.1 Health Check ===
  + python-chess: installed
  + Backup dir: /sdcard/Festoes/backups
  + SQLite DB: ready
  + Network: online
  + Int8 quantization: complete (1000 networks)
```

---

## Version History

| Version | Highlights |
|---------|-----------|
| v9.1 | Int8 quantization, SQLite3, auto-backup, skills plugin, python-chess, markdown rendering, weather cache, startup health check |
| v9.0 | Word2Vec groups, BERT positional weights, BLOOM subword, LLaMA RMS norm, GPT-2 temperature, Mixtral expert routing, Whisper confidence threshold, 866 training pairs |
| v8.x | 1,000 micro-networks, CSV augmentation, online learning, agent mode, voice, vision |

---

## Built with

- **Python 3.10+** — core language
- **NumPy** — neural network math
- **Tkinter** — UI (bundled with Python, no install)
- **SQLite3** — persistence (bundled with Python)
- **Gemini 2.0 Flash** — cloud AI backbone
- **Pydroid 3** — Android Python runtime

---

## License

MIT — free to use, modify, and distribute.

---

*Built in Nairobi. Runs on any Android phone.*
