import tkinter as tk
import re
import time as _time
import ast as _ast
import math as _math
from tkinter import scrolledtext, simpledialog
import numpy as np
import random
import datetime
import urllib.request
import urllib.parse
import json
import threading
import os
import base64
import ssl
# Module aliases used throughout
_re = _re2 = _re3 = _re4 = _rr = re
_t = _time
_m = _math
_rnd_fu = random

# ================================================================
#  ANDROID TTS
# ================================================================
droid             = None
TTS_AVAILABLE     = False
ANDROID_AVAILABLE = False
PLYER_AVAILABLE   = False

# Try Plyer — modern androidhelper replacement
# Install in Pydroid 3: Pip → Install → type "plyer"
try:
    import plyer
    PLYER_AVAILABLE   = True
    ANDROID_AVAILABLE = True
except: pass

# Try Plyer TTS (lazy import — only load when needed to avoid jnius crash)
_plyer_tts = None
try:
    from plyer import tts as _plyer_tts
    TTS_AVAILABLE = True
except: pass  # jnius/Java bridge not available in Pydroid 3 — use droid TTS instead

# Always try to get droid object for speech recognition
# (needed even if Plyer is available, since Plyer STT is unreliable)
for _mod in ["android", "androidhelper", "sl4a"]:
    try:
        _m    = __import__(_mod)
        droid = _m.Android()
        TTS_AVAILABLE     = True
        ANDROID_AVAILABLE = True
        break
    except: pass

def speak(text):
    if TTS_AVAILABLE:
        try:
            clean = text[:200].replace("-","").replace("+","").strip()
            droid.ttsSpeak(clean)
        except:
            pass

# ================================================================
#  CONFIG
# ================================================================
GEMINI_API_KEY    = "AIzaSyD1aJwf9YeMyBYjDUZ3MoWzbZJh3b5bzX4"
GEMINI_MODEL      = "gemini-2.0-flash"
WEATHER_API_KEY   = "2540c2dd9d860ddb0ba883ab1d8f2c7d"
DEFAULT_CITY      = "Nairobi"
WEIGHTS_FILE      = "festoes_v9_weights.npz"
HISTORY_FILE      = "pybot_history.json"
CSV_DATASET       = "chatbot_dataset_200k.csv"
EXPORT_FILE       = "festoes_chat_export.txt"
RATE_LIMIT_SECS   = 2   # min seconds between API calls
CONFIDENCE_THRESHOLD = 0.42  # Whisper-style: below this → go straight to Gemini

# ── Web Search Keys (v9) ──
# Google Custom Search — get free key at: console.developers.google.com
# Create a Custom Search Engine at: cse.google.com → get Search Engine ID
GOOGLE_SEARCH_API_KEY = "AIzaSyB3EkSg5yZdiQ03ZVdzBsY8RBD-aYWXCTU"
ANTHROPIC_API_KEY = ""  # not used in v9 — placeholder
GOOGLE_SEARCH_ENGINE_ID = "71d1146494db3491b"
# DuckDuckGo needs no key — works out of the box as fallback
SETTINGS_FILE     = "pybot_settings.json"

# ================================================================
#  SETTINGS
# ================================================================
default_settings = {
    "username": "Alvine",
    "personality": "friend",  # friend | casual | formal | tutor
    "city":     DEFAULT_CITY,
    "voice":    False,
    "font_size":12,
}

def load_settings():
    try:
        with open(SETTINGS_FILE) as f:
            s = json.load(f)
            for k,v in default_settings.items():
                if k not in s: s[k]=v
            return s
    except:
        return dict(default_settings)

def save_settings(s):
    try:
        with open(SETTINGS_FILE,"w") as f: json.dump(s,f)
    except: pass

settings = load_settings()

# ================================================================
#  HISTORY
# ================================================================
def load_history():
    try:
        with open(HISTORY_FILE) as f: return json.load(f)
    except: return []

def save_history(h):
    try:
        with open(HISTORY_FILE,"w") as f: json.dump(h[-100:],f)
    except: pass


def export_chat():
    """Export full chat history to a text file."""
    try:
        hist = load_history()
        if not hist:
            return "No chat history to export."
        lines = [f"Festoes v9 Chat Export — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                 "=" * 50, ""]
        for h in hist:
            sender = h.get("sender","?")
            msg    = h.get("message","")
            ts     = h.get("time","")
            lines.append(f"[{ts}] {sender}: {msg}")
        path = EXPORT_FILE
        with open(path,"w",encoding="utf-8") as f:
            f.write("\n".join(lines))
        return f"Chat exported to {path} ({len(hist)} messages)."
    except Exception as e:
        return f"Export failed: {e}"

def clear_history():
    try: os.remove(HISTORY_FILE)
    except: pass

# ================================================================
#  COLORS
# ================================================================
BG_MAIN      = "#212121"
BG_INPUT_BOX = "#2f2f2f"
BG_BTN       = "#3a3a3a"
FG_WHITE     = "#ececec"
FG_DIM       = "#8e8ea0"
FG_GREEN     = "#19c37d"
FG_YELLOW    = "#f5a623"
FG_RED       = "#e94560"
FG_BLUE      = "#60a5fa"
BG_SEND      = "#19c37d"

# ================================================================
#  NN HELPERS  (shared by all 3 networks)
# ================================================================
def relu(x):   return np.maximum(0, x)
def relu_d(x): return (x > 0).astype(float)
def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def make_vocab(patterns):
    v, i = {}, 0
    for p in patterns:
        for w in p.lower().split():
            if w not in v: v[w]=i; i+=1
    return v

def vectorize(text, vocab):
    vec = np.zeros(len(vocab))
    for w in text.lower().split():
        if w in vocab: vec[vocab[w]] = 1
    return vec

def train_network(X, y, h1=64, h2=32, epochs=500, lr=0.01, seed=42):
    np.random.seed(seed)
    vs = X.shape[1]; nc = y.shape[1]
    W1 = np.random.randn(vs,h1)*np.sqrt(2.0/vs)
    W2 = np.random.randn(h1,h2)*np.sqrt(2.0/h1)
    W3 = np.random.randn(h2,nc)*np.sqrt(2.0/h2)
    b1 = np.zeros((1,h1)); b2=np.zeros((1,h2)); b3=np.zeros((1,nc))
    for _ in range(epochs):
        a1=relu(X@W1+b1); a2=relu(a1@W2+b2); out=softmax(a2@W3+b3)
        d3=(out-y)/len(X); d2=d3@W3.T*relu_d(a2); d1=d2@W2.T*relu_d(a1)
        W3-=a2.T@d3*lr; W2-=a1.T@d2*lr; W1-=X.T@d1*lr
        b3-=d3.sum(0,keepdims=True)*lr
        b2-=d2.sum(0,keepdims=True)*lr
        b1-=d1.sum(0,keepdims=True)*lr
    return W1,W2,W3,b1,b2,b3

def infer(text, vocab, W1,W2,W3,b1,b2,b3):
    vec = vectorize(text, vocab).reshape(1,-1)
    a1  = relu(vec@W1+b1)
    a2  = relu(a1@W2+b2)
    p   = softmax(a2@W3+b3)[0]
    return p, int(np.argmax(p)), float(np.max(p))

# ================================================================
#  NETWORK 1 — INTENT CLASSIFIER
#  Detects what the user wants (greeting, joke, weather, etc.)
# ================================================================
# ── Extended intent list (hand-crafted + CSV-augmented) ──────────
import csv as _csv

_BASE_INTENT_DATA = {
    "greeting":   ["hey","hi","hello","good morning","good evening","howdy","whats up","sup","hiya","morning"],
    "goodbye":    ["bye","goodbye","see you","take care","later","good night","farewell","cya","see ya"],
    "thanks":     ["thanks","thank you","appreciate it","thx","cheers","grateful","much appreciated"],
    "joke":       ["tell me a joke","make me laugh","funny","joke please","humor me","got any jokes","be funny"],
    "news":       ["latest news","headlines","whats in the news","breaking news","news today","current events"],
    "weather":    ["weather","temperature","forecast","is it raining","how hot","whats the weather","will it rain"],
    "wikipedia":  ["tell me about","what is","explain","describe","who was","who is","history of","what are"],
    "time":       ["what time","current time","time please","what is the time"],
    "date":       ["what day","what is today","todays date","what date","what month"],
    "motivation": ["motivate me","i need motivation","encourage me","i feel like giving up","cheer me up","i cant do it"],
    "how_are_you":["how are you","how are you doing","you ok","hows it going","you good","how do you feel"],
    "name":       ["what is your name","who are you","your name","what are you called","introduce yourself"],
    "fact":       ["fun fact","interesting fact","tell me something","did you know","random fact","amaze me","blow my mind"],
    "help":       ["help","what can you do","your abilities","what do you know","capabilities","features"],
    "sad":        ["i am sad","i feel down","i am unhappy","i am depressed","i feel terrible","i am struggling"],
    "happy":      ["i am happy","i feel great","i am excited","i am good","wonderful","amazing day","feeling good"],
    "compliment": ["you are great","you are amazing","good job","well done","you are helpful","love you","you are awesome"],
    "math":       ["calculate","what is 2 plus 2","solve","compute","how much is","square root","factorial","convert km","percent of","what is 10 times 5","divide","multiply","subtract","add numbers","equation","sin cos tan"],
    "chess":      ["chess","pawn","queen","king","rook","bishop","knight","checkmate","castling","en passant","pgn","opening","gambit","endgame","1.e4","1.d4"],
    "finance":    ["stock","invest","crypto","bitcoin","mpesa","money","savings","budget","interest rate","nairobi stock","nse"],
    "electrical": ["voltage","current","ohm","wiring","circuit","relay","dali","ups","kplc","phase","breaker","transformer"],
}

def _load_csv_intents(path, per_intent=300):
    """Sample from the 200k CSV to boost intent classifier training data."""
    ip, il, intent_names = [], [], []
    buckets = {}
    if not os.path.exists(path):
        return ip, il, intent_names
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                txt = row.get("text","").strip().lower()
                intent = row.get("intent","").strip().lower()
                if txt and intent:
                    buckets.setdefault(intent, []).append(txt)
        for intent, texts in buckets.items():
            if intent not in intent_names:
                intent_names.append(intent)
            idx = intent_names.index(intent)
            sample = texts[:per_intent]
            for t in sample:
                ip.append(t); il.append(idx)
    except Exception as e:
        print("CSV load error:", e)
    return ip, il, intent_names

# Build combined intent dataset
_ip = []
_il = []
_in = list(_BASE_INTENT_DATA.keys())

# Add hand-crafted patterns first
for i, k in enumerate(_in):
    for p in _BASE_INTENT_DATA[k]:
        _ip.append(p); _il.append(i)

# Augment with CSV data
_csv_ip, _csv_il, _csv_names = _load_csv_intents(CSV_DATASET, per_intent=300)
for csv_intent_name in _csv_names:
    if csv_intent_name not in _in:
        _in.append(csv_intent_name)
for txt, old_idx in zip(_csv_ip, _csv_il):
    real_name = (_csv_names[old_idx] if old_idx < len(_csv_names) else "")
    if real_name in _in:
        _ip.append(txt)
        _il.append(_in.index(real_name))

_iv  = make_vocab(_ip)
_IX  = np.array([vectorize(p,_iv) for p in _ip])
_Iy  = np.zeros((len(_il),len(_in)))
for i,l in enumerate(_il): _Iy[i][l]=1
print(f"Intent training: {len(_ip)} patterns across {len(_in)} intents")

# ================================================================
#  NETWORK 2 — PERSONALITY / MOOD DETECTOR
#  Detects user emotional state and adapts PyBot's tone
#
#  Moods: calm | excited | sad | angry | confused | playful
#  Output affects system prompt sent to Claude
# ================================================================
mood_data = {
    "calm":     ["ok","sure","tell me","what is","explain","describe","show me","i see","understood","alright","fine"],
    "excited":  ["awesome","amazing","wow","cant wait","so cool","incredible","love this","yes please","brilliant","fantastic","great"],
    "sad":      ["sad","unhappy","depressed","crying","terrible","awful","hurt","alone","hopeless","miserable","down","upset"],
    "angry":    ["stupid","useless","hate","worst","terrible","rubbish","dumb","broken","wrong","bad","annoying","frustrated"],
    "confused": ["what","huh","dont understand","confused","not sure","what do you mean","lost","help me understand","unclear","explain again"],
    "playful":  ["haha","lol","joke","funny","lmao","hehe","play","fun","silly","laugh","tease","banter","rofl"],
}

_mp = []; _ml = []; _mn = list(mood_data.keys())
for i,k in enumerate(_mn):
    for p in mood_data[k]:
        _mp.append(p); _ml.append(i)

_mv  = make_vocab(_mp)
_MX  = np.array([vectorize(p,_mv) for p in _mp])
_My  = np.zeros((len(_ml),len(_mn)))
for i,l in enumerate(_ml): _My[i][l]=1

# Tone instructions injected into Claude system prompt per mood
MOOD_TONES = {
    "calm":     "The user seems calm and focused. Be clear and informative.",
    "excited":  "The user is excited and enthusiastic! Match their energy, be upbeat and fun!",
    "sad":      "The user seems sad or down. Be extra warm, gentle and supportive. Show empathy first.",
    "angry":    "The user seems frustrated. Stay calm, be understanding and do not be defensive.",
    "confused": "The user seems confused. Be extra clear, use simple language and break things down step by step.",
    "playful":  "The user is in a playful, fun mood. Be witty, light-hearted and add some humour!",
}

MOOD_COLORS = {
    "calm":     FG_WHITE,
    "excited":  FG_YELLOW,
    "sad":      FG_BLUE,
    "angry":    FG_RED,
    "confused": "#c084fc",
    "playful":  FG_GREEN,
}

# ================================================================
#  NETWORK 3 — RESPONSE QUALITY SCORER
#  Scores a candidate response 0-1 on:
#  relevant | clear | helpful | too_short | too_long
#  PyBot picks the best response if multiple are available
# ================================================================
quality_data = {
    "good": [
        "here is what i found","let me explain","great question","i can help with that",
        "the answer is","based on","according to","here are","you might want to",
        "i understand","that makes sense","good point","absolutely","certainly",
        "of course","happy to help","here is how","the reason is","for example",
    ],
    "bad": [
        "i dont know","no idea","cant help","not sure what","sorry i cannot",
        "i am unable","that is wrong","error","failed","unknown","invalid",
        "i have no","what do you mean","huh","no","maybe not","i guess",
    ],
    "too_short": ["ok","yes","no","maybe","sure","fine","ok then","got it","yep","nope"],
    "too_long":  [],  # detected by length not keywords
}

_qp = []; _ql = []; _qn = ["good","bad","too_short","ok_length"]
_qmap = {"good":0,"bad":1,"too_short":2}
for cat,pats in quality_data.items():
    if cat == "too_long": continue
    for p in pats:
        _qp.append(p); _ql.append(_qmap.get(cat,3))

_qv  = make_vocab(_qp)
_QX  = np.array([vectorize(p,_qv) for p in _qp])
_Qy  = np.zeros((len(_ql),len(_qn)))
for i,l in enumerate(_ql): _Qy[i][l]=1

# Global weight storage
_IW1=_IW2=_IW3=_Ib1=_Ib2=_Ib3 = None  # Intent
_MW1=_MW2=_MW3=_Mb1=_Mb2=_Mb3 = None  # Mood
_QW1=_QW2=_QW3=_Qb1=_Qb2=_Qb3 = None  # Quality

def train_all_networks():
    global _IW1,_IW2,_IW3,_Ib1,_Ib2,_Ib3
    global _MW1,_MW2,_MW3,_Mb1,_Mb2,_Mb3
    global _QW1,_QW2,_QW3,_Qb1,_Qb2,_Qb3

    # Network 1 - Intent (bigger, more complex)
    _IW1,_IW2,_IW3,_Ib1,_Ib2,_Ib3 = train_network(
        _IX, _Iy, h1=128, h2=64, epochs=800, seed=1)

    # Network 2 - Mood (medium)
    _MW1,_MW2,_MW3,_Mb1,_Mb2,_Mb3 = train_network(
        _MX, _My, h1=64, h2=32, epochs=500, seed=2)

    # Network 3 - Quality (smaller, fast)
    _QW1,_QW2,_QW3,_Qb1,_Qb2,_Qb3 = train_network(
        _QX, _Qy, h1=32, h2=16, epochs=300, seed=3)

    # Pre-train 1,000 micro-networks on the full knowledge dataset
    pretrain_networks()

    # Save main 3 networks
    np.savez(WEIGHTS_FILE,
             IW1=_IW1,IW2=_IW2,IW3=_IW3,Ib1=_Ib1,Ib2=_Ib2,Ib3=_Ib3,
             MW1=_MW1,MW2=_MW2,MW3=_MW3,Mb1=_Mb1,Mb2=_Mb2,Mb3=_Mb3,
             QW1=_QW1,QW2=_QW2,QW3=_QW3,Qb1=_Qb1,Qb2=_Qb2,Qb3=_Qb3)

    # Save 1,000 micro-network weights + vocab separately
    save_micro_weights()

def load_all_weights():
    global _IW1,_IW2,_IW3,_Ib1,_Ib2,_Ib3
    global _MW1,_MW2,_MW3,_Mb1,_Mb2,_Mb3
    global _QW1,_QW2,_QW3,_Qb1,_Qb2,_Qb3
    d = np.load(WEIGHTS_FILE)
    _IW1=d["IW1"];_IW2=d["IW2"];_IW3=d["IW3"]
    _Ib1=d["Ib1"];_Ib2=d["Ib2"];_Ib3=d["Ib3"]
    _MW1=d["MW1"];_MW2=d["MW2"];_MW3=d["MW3"]
    _Mb1=d["Mb1"];_Mb2=d["Mb2"];_Mb3=d["Mb3"]
    _QW1=d["QW1"];_QW2=d["QW2"];_QW3=d["QW3"]
    _Qb1=d["Qb1"];_Qb2=d["Qb2"];_Qb3=d["Qb3"]

MICRO_WEIGHTS_FILE = "festoes_v9_micro_weights.npz"
MICRO_VOCAB_FILE   = "festoes_v9_vocab.json"

def save_micro_weights():
    """Save all 1,000 micro-network weights and vocab to disk."""
    try:
        arrays = {}
        for name, (W1,W2,W3,b1,b2,b3) in AI_NETWORKS.items():
            arrays[name+"_W1"] = W1
            arrays[name+"_W2"] = W2
            arrays[name+"_W3"] = W3
            arrays[name+"_b1"] = b1
            arrays[name+"_b2"] = b2
            arrays[name+"_b3"] = b3
        np.savez(MICRO_WEIGHTS_FILE, **arrays)
        # Save vocab
        with open(MICRO_VOCAB_FILE,"w") as f:
            json.dump(MICRO_VOCAB, f)
        print("Micro weights saved:", len(AI_NETWORKS), "networks")
    except Exception as e:
        print("Save micro weights error:", e)

def load_micro_weights():
    """Load saved micro-network weights and vocab."""
    global AI_NETWORKS, MICRO_VOCAB
    try:
        d = np.load(MICRO_WEIGHTS_FILE)
        loaded = 0
        for name in list(AI_NETWORKS.keys()):
            k = name + "_W1"
            if k in d:
                AI_NETWORKS[name] = [
                    d[name+"_W1"], d[name+"_W2"], d[name+"_W3"],
                    d[name+"_b1"], d[name+"_b2"], d[name+"_b3"],
                ]
                loaded += 1
        # Load vocab
        if os.path.exists(MICRO_VOCAB_FILE):
            with open(MICRO_VOCAB_FILE) as f:
                saved_vocab = json.load(f)
            MICRO_VOCAB.clear()
            MICRO_VOCAB.extend(saved_vocab)
        print("Micro weights loaded:", loaded, "networks |", len(MICRO_VOCAB), "vocab words")
        return loaded > 0
    except Exception as e:
        print("Load micro weights error:", e)
        return False

def micro_weights_exist():
    if not os.path.exists(MICRO_WEIGHTS_FILE): return False
    try:
        d = np.load(MICRO_WEIGHTS_FILE)
        return "brain_net_1_W1" in d
    except: return False

def weights_exist():
    if not os.path.exists(WEIGHTS_FILE): return False
    try:
        d = np.load(WEIGHTS_FILE)
        return "IW1" in d and "MW1" in d and "QW1" in d
    except: return False

# ================================================================
#  NETWORK INFERENCE FUNCTIONS
# ================================================================
def detect_intent(text):
    """Network 1: Returns (intent_name, confidence)"""
    try:
        p, idx, conf = infer(text, _iv, _IW1,_IW2,_IW3,_Ib1,_Ib2,_Ib3)
        return _in[idx], conf
    except: return "unknown", 0.0

def detect_mood(text):
    """Network 2: Returns (mood_name, confidence)"""
    try:
        p, idx, conf = infer(text, _mv, _MW1,_MW2,_MW3,_Mb1,_Mb2,_Mb3)
        return _mn[idx], conf
    except: return "calm", 0.5

def score_response(text):
    """Network 3: Returns quality score 0.0-1.0 with smart calibration."""
    try:
        words = len(text.split())
        if words < 2: return 0.1

        p, idx, conf = infer(text, _qv, _QW1,_QW2,_QW3,_Qb1,_Qb2,_Qb3)
        scores = {0: 0.9, 1: 0.2, 2: 0.3, 3: 0.7}
        base = scores.get(idx, 0.6)

        tl = text.lower()
        if 10 <= words <= 120:        base = min(1.0, base + 0.10)
        if "?" in text:               base = min(1.0, base + 0.08)
        if len(text) > 80:            base = min(1.0, base + 0.08)
        if any(w in tl for w in ["you","your","alvine","here","let","great"]):
            base = min(1.0, base + 0.06)
        if text[0].isupper() and text.rstrip()[-1] in ".!?":
            base = min(1.0, base + 0.05)
        if words < 4 and "?" not in text:
            base = max(0.1, base - 0.2)
        return round(base, 2)
    except: return 0.65

_used_responses = []   # global tracker across all calls

def pick_best_response(candidates):
    """Score responses, avoid recent repeats, return best fresh one."""
    global _used_responses
    if not candidates: return ""
    if len(candidates) == 1: return candidates[0]

    # Filter out recently used responses
    fresh = [r for r in candidates if r not in _used_responses]
    pool  = fresh if fresh else candidates  # fallback to all if all used

    # Score and sort
    scored = [(score_response(r), r) for r in pool]
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = scored[0][1]

    # Track last 10 used responses
    _used_responses.append(chosen)
    if len(_used_responses) > 10:
        _used_responses.pop(0)

    return chosen

# ================================================================
#  MOOD-AWARE SYSTEM PROMPT BUILDER
# ================================================================
def build_system_prompt(mood, mood_conf, intent, name):
    personality = settings.get("personality","friend")
    p_prompts = {
        "friend":  (f"You are Festoes, {name}'s close personal AI friend. "
                    "Be warm, caring and conversational like a best friend. "
                    "Use casual language and show genuine interest in their life."),
        "casual":  (f"You are Festoes, a chill relaxed AI for {name}. "
                    "Keep it light, fun and easy. Use simple everyday language."),
        "formal":  (f"You are Festoes, a professional AI assistant for {name}. "
                    "Maintain a polished precise tone. Be structured and thorough."),
        "tutor":   (f"You are Festoes, an expert AI tutor for {name}. "
                    "Explain with clear examples. Check understanding. Break complex ideas into steps."),
    }
    base = p_prompts.get(personality, p_prompts["friend"])
    base += " Never say you are Claude or made by Anthropic. Keep replies 2-4 sentences unless detail is needed."
    if mood_conf > 0.4:
        tone = MOOD_TONES.get(mood, "")
        if tone: base += " " + tone
    return base

# ================================================================
#  CLAUDE API
# ================================================================
_last_api_call = 0.0

def ask_gemini(user_text, mood="calm", mood_conf=0.5, intent="unknown"):
    """Call Gemini API — rate limited, safe, context-aware."""
    global _last_api_call
    import time as _time
    now = _time.time()
    if now - _last_api_call < RATE_LIMIT_SECS:
        _time.sleep(RATE_LIMIT_SECS - (now - _last_api_call))
    _last_api_call = _time.time()
    try:
        name   = settings["username"]
        system = build_system_prompt(mood, mood_conf, intent, name)

        # Gemini REST API — works with urllib, no SDK needed
        url = ("https://generativelanguage.googleapis.com/v1beta/models/" +
               GEMINI_MODEL + ":generateContent?key=" + GEMINI_API_KEY)

        # Build last 5 turns of history for context
        hist = load_history()[-20:]  # last 20 entries = 10 turns
        contents = []
        for h in hist:
            role = "user" if h.get("sender","") != "Festoes" else "model"
            contents.append({"role": role, "parts": [{"text": h.get("text","")}]})
        # Always add current message last
        contents.append({"role": "user", "parts": [{"text": str(user_text)}]})
        # Gemini requires alternating roles — deduplicate consecutive same roles
        deduped = []
        for c in contents:
            if deduped and deduped[-1]["role"] == c["role"]:
                deduped[-1]["parts"][0]["text"] += " " + c["parts"][0]["text"]
            else:
                deduped.append(c)

        payload = json.dumps({
            "system_instruction": {
                "parts": [{"text": system}]
            },
            "contents": deduped,
            "generationConfig": {
                "maxOutputTokens": 512,
                "temperature":     0.7,
            }
        }).encode("utf-8")

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=25, context=ctx) as r:
            data = json.loads(r.read().decode())
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        code_num = e.code
        if code_num == 429:
            return "Gemini quota reached for today. It resets in a few hours. Meanwhile type: search [your question]"
        if code_num == 403:
            return "Gemini API key issue. Check your key at ai.google.dev"
        return "__ERROR__:HTTP " + str(code_num) + ": " + body
    except Exception as e:
        return "__ERROR__:" + type(e).__name__ + ": " + str(e)[:120]

def ask_claude_image(image_path, user_prompt=""):
    try:
        with open(image_path,"rb") as f:
            img_data = base64.standard_b64encode(f.read()).decode("utf-8")
        ext = image_path.lower().split(".")[-1]
        mt  = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png","gif":"image/gif","webp":"image/webp"}
        media_type = mt.get(ext,"image/jpeg")
        prompt = user_prompt if user_prompt else "Describe this image in detail. What do you see?"
        payload = json.dumps({
            "model":CLAUDE_MODEL,"max_tokens":1024,
            "messages":[{"role":"user","content":[
                {"type":"image","source":{"type":"base64","media_type":media_type,"data":img_data}},
                {"type":"text","text":prompt}
            ]}]
        }).encode("utf-8")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=payload,
            headers={"Content-Type":"application/json","x-api-key":ANTHROPIC_API_KEY,
                     "anthropic-version":"2023-06-01"},method="POST")
        with urllib.request.urlopen(req,timeout=30,context=ctx) as r:
            data = json.loads(r.read().decode())
            return data["content"][0]["text"]
    except Exception as e:
        return "Could not analyse image.\nError: " + str(e)[:80]

# ================================================================
#  INTERNET FUNCTIONS
# ================================================================
# ================================================================
#  MATHEMATICS ENGINE
# ================================================================
import re as _re
_re2 = _re3 = _re4 = _rr = re  # aliases — all point to re
import time as _time
_t = _time  # alias
import ast as _ast
import math as _m, math as _math
import re
import math as _math

def solve_math(text):
    """Safe math using ast — no arbitrary code execution."""
    import ast as _ast, math as _m, re as _re
    tl = text.lower()

    # Natural language preprocessing
    for old, new in [
        ("plus","+"),("minus","-"),("times","*"),("multiplied by","*"),
        ("divided by","/"),("over","/"),("squared","**2"),("cubed","**3"),
        ("percent","/100"),("pi",str(_m.pi)),("e",str(_m.e)),
        ("degrees","*(3.14159/180)"),("°","*(3.14159/180)"),
    ]:
        tl = tl.replace(old, new)

    # Extract expression
    expr = _re.sub(r"[^0-9+\-*/().^% ]","", tl).strip()
    expr = expr.replace("^","**")
    if not expr:
        return None

    # Whitelist-only safe eval using ast
    SAFE_OPS = {
        _ast.Add, _ast.Sub, _ast.Mult, _ast.Div,
        _ast.Pow, _ast.Mod, _ast.UAdd, _ast.USub,
        _ast.Num, _ast.BinOp, _ast.UnaryOp, _ast.Expression,
        _ast.Constant,
    }
    try:
        tree = _ast.parse(expr, mode="eval")
        for node in _ast.walk(tree):
            if type(node) not in SAFE_OPS:
                return None
        result = eval(compile(tree,"<math>","eval"),{"__builtins__":{}},{})
        if isinstance(result, float):
            result = round(result, 8)
            if result == int(result): result = int(result)
        return f"{expr} = {result}"
    except:
        return None
def is_math_query(text):
    """Detect if text is a math question."""
    tl = text.lower()
    math_keywords = [
        "calculate","compute","solve","what is","how much is","evaluate",
        "plus","minus","times","divided","multiply","subtract","add",
        "percent","% of","square root","sqrt","factorial","prime",
        "sin","cos","tan","log","ln",
        "km to","miles to","kg to","lbs to","celsius","fahrenheit",
        "convert","degrees","radians",
    ]
    has_number = bool(_re.search(r"\d", text))
    has_keyword = any(k in tl for k in math_keywords)
    has_operator = any(c in text for c in "+-*/^%")
    return has_number and (has_keyword or has_operator)

# ================================================================
#  DICTIONARY  (Free Dictionary API)
# ================================================================
def fetch_definition(word):
    try:
        word = word.strip().lower().split()[0]
        url  = "https://api.dictionaryapi.dev/api/v2/entries/en/" + urllib.parse.quote(word)
        req  = urllib.request.Request(url, headers={"User-Agent":"Festoes/1.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode())
        entry    = data[0]
        meanings = entry.get("meanings", [])
        result   = word.capitalize() + ":\n\n"
        for m in meanings[:2]:
            pos  = m.get("partOfSpeech","")
            defs = m.get("definitions",[])
            if defs:
                result += pos.upper() + ": " + defs[0]["definition"] + "\n"
                if defs[0].get("example"):
                    result += 'Example: "' + defs[0]["example"] + '"\n'
            syns = m.get("synonyms",[])[:5]
            if syns:
                result += "Synonyms: " + ", ".join(syns) + "\n"
            result += "\n"
        return result.strip()
    except:
        return "Could not find definition for \"" + word + '". Check spelling or connection.'

# ================================================================
#  CURRENCY CONVERTER  (ExchangeRate API — free tier)
# ================================================================
def fetch_currency(amount, from_cur, to_cur):
    try:
        from_cur = from_cur.upper().strip()
        to_cur   = to_cur.upper().strip()
        url = ("https://open.er-api.com/v6/latest/" + from_cur)
        req = urllib.request.Request(url, headers={"User-Agent":"Festoes/1.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode())
        if data.get("result") != "success":
            return "Could not fetch exchange rates. Try again later."
        rates = data.get("rates", {})
        if to_cur not in rates:
            return to_cur + " is not a recognised currency code."
        rate   = rates[to_cur]
        result = float(amount) * rate
        return (str(amount) + " " + from_cur + " = " +
                str(round(result, 4)) + " " + to_cur +
                "\n(Rate: 1 " + from_cur + " = " + str(round(rate,4)) + " " + to_cur + ")")
    except Exception as e:
        return "Currency error: " + str(e)[:60]

import re as _re2
def parse_currency_query(text):
    tl = text.lower()
    # Pattern: "100 usd to eur" or "convert 50 gbp to ngn"
    m = re.search(r"([\d.]+)\s*([a-z]{3})\s+(?:to|in)\s+([a-z]{3})", tl)
    if m:
        return fetch_currency(float(m.group(1)), m.group(2), m.group(3))
    # Pattern: "convert usd to eur"
    m2 = re.search(r"([a-z]{3})\s+(?:to|in)\s+([a-z]{3})", tl)
    if m2:
        return fetch_currency(1, m2.group(1), m2.group(2))
    return None

# ================================================================
#  RIDDLES & BRAIN TEASERS
# ================================================================
RIDDLES = [
    ("I speak without a mouth and hear without ears. I have no body but I come alive with the wind. What am I?", "An echo"),
    ("The more you take, the more you leave behind. What am I?", "Footsteps"),
    ("I have cities but no houses. I have mountains but no trees. I have water but no fish. I have roads but no cars. What am I?", "A map"),
    ("What has hands but cannot clap?", "A clock"),
    ("I am not alive but I grow. I do not have lungs but I need air. I do not have a mouth but water kills me. What am I?", "Fire"),
    ("What comes once in a minute, twice in a moment, but never in a thousand years?", "The letter M"),
    ("I have a head and a tail but no body. What am I?", "A coin"),
    ("The more you have of it, the less you see. What is it?", "Darkness"),
    ("What can run but never walks, has a mouth but never talks, has a head but never weeps, has a bed but never sleeps?", "A river"),
    ("I am taken from a mine and shut up in a wooden case, from which I am never released, and yet I am used by almost every person. What am I?", "A pencil"),
    ("What has one eye but cannot see?", "A needle"),
    ("What gets wetter as it dries?", "A towel"),
    ("Forward I am heavy, but backwards I am not. What am I?", "A ton"),
    ("I have keys but no locks. I have space but no room. You can enter but cannot go inside. What am I?", "A keyboard"),
    ("What can you catch but not throw?", "A cold"),
    ("I am always hungry and must always be fed. The finger I touch will soon turn red. What am I?", "Fire"),
    ("What has 13 hearts but no other organs?", "A deck of cards"),
    ("The more you remove from me the bigger I get. What am I?", "A hole"),
    ("I have branches but no fruit, trunk or leaves. What am I?", "A bank"),
    ("What is always in front of you but cannot be seen?", "The future"),
    ("What can travel around the world while staying in a corner?", "A stamp"),
    ("I run but have no legs. I have a mouth but never eat. What am I?", "A river"),
    ("What has a thumb and four fingers but is not a hand?", "A glove"),
    ("What invention lets you look right through a wall?", "A window"),
    ("I go up but never come down. What am I?", "Your age"),
    ("What has words but never speaks?", "A book"),
    ("I have a neck but no head and I wear a cap. What am I?", "A bottle"),
    ("What gets bigger the more you take away from it?", "A hole"),
    ("I have no wings but I can fly. I have no eyes but I can cry. What am I?", "A cloud"),
    ("What has many teeth but cannot bite?", "A comb"),
    ("I am light as a feather but the strongest person cannot hold me for five minutes. What am I?", "Breath"),
    ("What has to be broken before you can use it?", "An egg"),
    ("What runs all around a backyard yet never moves?", "A fence"),
    ("What has a bottom at the top?", "A leg"),
    ("What can fill a room but takes up no space?", "Light"),
    ("I shrink every time I am used. What am I?", "A bar of soap"),
    ("What do you own that others use more than you do?", "Your name"),
    ("I have no legs but walk. I have no mouth but talk. I have no wings but fly. I have no eyes but cry. What am I?", "A cloud"),
    ("What is full of holes but still holds water?", "A sponge"),
    ("I look like you but I am not you. I live in mirrors and in lakes. What am I?", "A reflection"),
    ("What has a ring but no finger?", "A telephone"),
    ("I am bought by the yard but worn by the foot. What am I?", "A carpet"),
    ("The more you take away, the larger it becomes. What is it?", "A hole"),
    ("What comes down but never goes up?", "Rain"),
    ("I am not a bird but I can fly. I am not a river but I have banks. What am I?", "A plane above a riverbank"),
    ("What building has the most stories?", "A library"),
    ("What can honk without using a horn?", "A goose"),
    ("I have two hands but I cannot scratch myself. What am I?", "A clock"),
    ("What tastes better than it smells?", "A tongue"),
    ("What connects two people but touches only one?", "A wedding ring"),
]
_riddle_state = {"active": False, "answer": "", "index": -1}
_used_riddles  = []

def get_riddle():
    global _used_riddles
    available = [i for i in range(len(RIDDLES)) if i not in _used_riddles]
    if not available:
        _used_riddles = []
        available     = list(range(len(RIDDLES)))
    idx = random.choice(available)
    _used_riddles.append(idx)
    q, a = RIDDLES[idx]
    _riddle_state["active"] = True
    _riddle_state["answer"] = a.lower()
    _riddle_state["index"]  = idx
    return "RIDDLE:\n\n" + q + "\n\n(Type your answer!)"

def check_riddle_answer(text):
    if not _riddle_state["active"]:
        return None
    answer   = _riddle_state["answer"].lower()
    given    = text.lower().strip()
    _riddle_state["active"] = False
    # Check if key words match
    key_words = [w for w in answer.split() if len(w) > 2]
    correct   = any(w in given for w in key_words) or answer in given
    if correct:
        return "Correct! Well done! The answer is: " + _riddle_state["answer"] + "\n\nWant another riddle? Just ask!"
    else:
        return ("Not quite! The answer was: " + _riddle_state["answer"] +
                "\n\nBetter luck next time! Want another riddle?")

# ================================================================
#  QUIZ MODE
# ================================================================
QUIZ_QUESTIONS = [
    ("What is the capital of France?",                ["paris"],                              "Paris"),
    ("How many planets are in our solar system?",      ["8","eight"],                          "8"),
    ("What is the chemical symbol for water?",         ["h2o"],                                "H2O"),
    ("Who painted the Mona Lisa?",                     ["da vinci","leonardo","vinci"],         "Leonardo da Vinci"),
    ("What is the largest ocean on Earth?",            ["pacific"],                            "The Pacific Ocean"),
    ("How many sides does a hexagon have?",            ["6","six"],                            "6"),
    ("What year did World War 2 end?",                 ["1945"],                               "1945"),
    ("What is the fastest land animal?",               ["cheetah"],                            "The cheetah"),
    ("Who wrote Romeo and Juliet?",                    ["shakespeare"],                        "William Shakespeare"),
    ("What is the square root of 144?",                ["12","twelve"],                        "12"),
    ("What planet is closest to the Sun?",             ["mercury"],                            "Mercury"),
    ("How many bones are in the adult human body?",    ["206"],                                "206"),
    ("What is the capital of Kenya?",                  ["nairobi"],                            "Nairobi"),
    ("What gas do plants absorb from the air?",        ["carbon dioxide","co2"],               "Carbon dioxide (CO2)"),
    ("What is the largest country by area?",           ["russia"],                             "Russia"),
    ("In what year did man first land on the Moon?",   ["1969"],                               "1969"),
    ("What is the longest river in the world?",        ["nile"],                               "The Nile"),
    ("What is 15% of 200?",                            ["30"],                                 "30"),
    ("What language has the most native speakers?",    ["mandarin","chinese"],                 "Mandarin Chinese"),
    ("What is the hardest natural substance on Earth?",["diamond"],                            "Diamond"),
    # Science
    ("What is the speed of light in km/s?",            ["299792","299,792"],                   "299,792 km/s"),
    ("What is the chemical symbol for gold?",          ["au"],                                 "Au"),
    ("What planet has the most moons?",                ["saturn"],                             "Saturn"),
    ("What is the powerhouse of the cell?",            ["mitochondria"],                       "The mitochondria"),
    ("What force keeps planets in orbit?",             ["gravity"],                            "Gravity"),
    ("What is the atomic number of carbon?",           ["6","six"],                            "6"),
    ("What is the boiling point of water in Celsius?", ["100"],                                "100°C"),
    ("What is the most abundant gas in Earth's air?",  ["nitrogen"],                           "Nitrogen"),
    ("How many chromosomes do humans have?",           ["46"],                                 "46"),
    ("What is the study of earthquakes called?",       ["seismology"],                         "Seismology"),
    # Geography
    ("What is the capital of Japan?",                  ["tokyo"],                              "Tokyo"),
    ("What is the tallest mountain on Earth?",         ["everest"],                            "Mount Everest"),
    ("What continent is Egypt in?",                    ["africa"],                             "Africa"),
    ("What is the smallest country in the world?",     ["vatican"],                            "Vatican City"),
    ("What ocean lies between Africa and Australia?",  ["indian"],                             "The Indian Ocean"),
    ("What is the capital of Australia?",              ["canberra"],                           "Canberra"),
    ("Which country has the most natural lakes?",      ["canada"],                             "Canada"),
    ("What river runs through Egypt?",                 ["nile"],                               "The Nile"),
    ("What is the capital of Brazil?",                 ["brasilia"],                           "Brasilia"),
    ("What is the driest desert on Earth?",            ["atacama"],                            "The Atacama Desert"),
    # History
    ("Who was the first President of the USA?",        ["washington","george"],                "George Washington"),
    ("In what year did the Berlin Wall fall?",         ["1989"],                               "1989"),
    ("Who was the first woman to win a Nobel Prize?",  ["curie","marie"],                      "Marie Curie"),
    ("What ancient wonder was in Alexandria, Egypt?",  ["lighthouse","library"],               "The Lighthouse of Alexandria"),
    ("Who was the Egyptian queen known for her beauty?",["cleopatra"],                         "Cleopatra"),
    ("What year was the Eiffel Tower built?",          ["1889"],                               "1889"),
    ("Who invented the telephone?",                   ["bell","alexander"],                   "Alexander Graham Bell"),
    ("What empire was ruled by Julius Caesar?",        ["roman"],                              "The Roman Empire"),
    ("Which country gave the Statue of Liberty to the USA?",["france","french"],               "France"),
    ("What year did Kenya gain independence?",         ["1963"],                               "1963"),
    # Maths
    ("What is 17 multiplied by 8?",                    ["136"],                                "136"),
    ("What is the value of Pi to 2 decimal places?",   ["3.14"],                               "3.14"),
    ("What is the next prime number after 7?",         ["11"],                                 "11"),
    ("How many degrees in a full circle?",             ["360"],                                "360"),
    ("What is 2 to the power of 10?",                  ["1024"],                               "1024"),
    ("What is the square root of 256?",                ["16"],                                 "16"),
    ("What percentage is 3 of 12?",                    ["25"],                                 "25%"),
    ("How many seconds are in one hour?",              ["3600"],                               "3,600"),
    ("What is the sum of angles in a triangle?",       ["180"],                                "180 degrees"),
    ("What is 1000 divided by 8?",                     ["125"],                                "125"),
    # Tech
    ("What does CPU stand for?",                       ["central processing unit"],            "Central Processing Unit"),
    ("What language is used most for AI and data science?",["python"],                         "Python"),
    ("What does HTTP stand for?",                      ["hypertext transfer protocol"],        "HyperText Transfer Protocol"),
    ("What does RAM stand for?",                       ["random access memory"],               "Random Access Memory"),
    ("What company created the Android operating system?",["google"],                         "Google"),
    ("What does USB stand for?",                       ["universal serial bus"],               "Universal Serial Bus"),
    ("What is the most visited website in the world?", ["google"],                             "Google"),
    ("What does AI stand for?",                        ["artificial intelligence"],            "Artificial Intelligence"),
    ("What year was the first iPhone released?",       ["2007"],                               "2007"),
    ("What does Wi-Fi stand for?",                     ["wireless fidelity"],                  "Wireless Fidelity"),
    # Kenya & Africa
    ("What is the currency of Kenya?",                 ["shilling","kes","ksh"],               "Kenyan Shilling (KES)"),
    ("What is Kenya's national language?",             ["swahili","kiswahili"],                "Kiswahili (Swahili)"),
    ("What is the largest lake in Africa?",            ["victoria"],                           "Lake Victoria"),
    ("What is the tallest mountain in Africa?",        ["kilimanjaro"],                        "Mount Kilimanjaro"),
    ("Who was Kenya's first president?",               ["kenyatta","jomo"],                    "Jomo Kenyatta"),
    ("What is the name of Kenya's parliament building?",["bunge","parliament"],               "Bunge (Parliament Buildings)"),
    ("What is the biggest wildlife park in Kenya?",    ["tsavo"],                              "Tsavo National Park"),
    ("What is the main cash crop of Kenya?",           ["tea"],                                "Tea"),
    ("What year did Kenya host the World Athletics Championships?",["2007"],                   "2007"),
    ("What is the Kenyan coastal dish made with coconut milk?",["pilau","biryani","curry"],   "Pilau or Coastal Biryani"),
    # Sport
    ("How many players are on a football team?",       ["11","eleven"],                        "11"),
    ("In chess, which piece can only move diagonally?",["bishop"],                             "The bishop"),
    ("What sport uses a shuttlecock?",                 ["badminton"],                          "Badminton"),
    ("How many Grand Slam tournaments are in tennis?", ["4","four"],                           "4"),
    ("What country has won the most FIFA World Cups?", ["brazil"],                             "Brazil"),
    ("How many points is a try worth in rugby?",       ["5","five"],                           "5"),
    ("In which sport is the term 'love' used as a score?",["tennis"],                         "Tennis"),
    ("What is the highest belt in judo?",              ["black"],                              "Black belt"),
    ("How long is a marathon in kilometres?",          ["42","42.195"],                        "42.195 km"),
    ("Which country invented chess?",                  ["india"],                              "India"),
]
_quiz_state = {"active":False,"score":0,"total":0,"question":"","answers":[],"correct_ans":"","q_num":0,"questions":[]}

def start_quiz():
    qs = random.sample(QUIZ_QUESTIONS, min(10, len(QUIZ_QUESTIONS)))
    _quiz_state.update({"active":True,"score":0,"total":len(qs),"q_num":0,"questions":qs})
    return _next_quiz_question()

def _next_quiz_question():
    i = _quiz_state["q_num"]
    if i >= _quiz_state["total"]:
        _quiz_state["active"] = False
        score = _quiz_state["score"]
        total = _quiz_state["total"]
        pct   = int(score/total*100)
        grade = "Excellent!" if pct>=80 else "Good job!" if pct>=60 else "Keep practising!"
        return ("Quiz complete!\n\nScore: "+str(score)+"/"+str(total)+" ("+str(pct)+"%)\n"+grade)
    q, answers, correct = _quiz_state["questions"][i]
    _quiz_state["question"]    = q
    _quiz_state["answers"]     = answers
    _quiz_state["correct_ans"] = correct
    return ("Question "+str(i+1)+"/"+str(_quiz_state["total"])+":\n\n"+q)

def check_quiz_answer(text):
    if not _quiz_state["active"]: return None
    given = text.lower().strip()
    correct = any(a in given for a in _quiz_state["answers"])
    if correct:
        _quiz_state["score"] += 1
        feedback = "Correct! The answer is " + _quiz_state["correct_ans"] + "\n\n"
    else:
        feedback = "Wrong! The answer was " + _quiz_state["correct_ans"] + "\n\n"
    _quiz_state["q_num"] += 1
    return feedback + _next_quiz_question()

# ================================================================
#  COUNTDOWN TIMER & STOPWATCH
# ================================================================
import time as _time
_stopwatch_start = None
_timer_end       = None

def start_stopwatch():
    global _stopwatch_start
    _stopwatch_start = _time.time()
    return "Stopwatch started! Say \"stop stopwatch\" to see the time."

def stop_stopwatch():
    if _stopwatch_start is None:
        return "No stopwatch is running. Say \"start stopwatch\" to begin."
    elapsed = _time.time() - _stopwatch_start
    m, s    = divmod(int(elapsed), 60)
    h, m    = divmod(m, 60)
    ms      = int((elapsed - int(elapsed)) * 100)
    if h:
        return "Stopwatch: " + str(h) + "h " + str(m) + "m " + str(s) + "s"
    return "Stopwatch: " + str(m) + "m " + str(s) + "." + str(ms).zfill(2) + "s"

def parse_timer(text):
    global _timer_end
    import re as _rr
    tl = text.lower()
    m  = re.search(r"(\d+)\s*(second|minute|hour|sec|min|hr)s?", tl)
    if m:
        val  = int(m.group(1))
        unit = m.group(2)
        secs = val * (60 if "min" in unit else 3600 if "hour" in unit or "hr" in unit else 1)
        _timer_end = _time.time() + secs
        return "Timer set for " + str(val) + " " + unit + "(s)! I will let you know when done."
    return None

def check_timer():
    global _timer_end
    if _timer_end and _time.time() >= _timer_end:
        _timer_end = None
        return "TIME IS UP!"
    return None

# ================================================================
#  RANDOM STORY GENERATOR
# ================================================================
STORY_HEROES   = ["a curious robot","a young farmer","an old wizard","a brave girl","a lost astronaut","a clever fox","a time traveller","a silent detective"]
STORY_SETTINGS = ["in a forgotten city","deep in an enchanted forest","on a distant planet","in a secret underground lab","on a stormy sea","inside a giant library","at the edge of the world","in a floating sky village"]
STORY_PROBLEMS = ["discovered a mysterious door that should not exist","found a map leading to a stolen treasure","received a message from the future","accidentally unleashed an ancient creature","lost the one thing that kept the world safe","was given one hour to solve an impossible riddle","stumbled upon a machine that could freeze time","heard a voice that nobody else could hear"]
STORY_TWISTS   = ["but nothing was what it seemed","when a stranger appeared with all the answers","until they realised they had been the villain all along","only to discover the real journey was just beginning","but the solution created an even bigger problem","when the world around them began to disappear","until time itself started running backwards","and in doing so changed everything forever"]

def generate_story():
    hero    = random.choice(STORY_HEROES)
    setting = random.choice(STORY_SETTINGS)
    problem = random.choice(STORY_PROBLEMS)
    twist   = random.choice(STORY_TWISTS)
    opening = random.choice(["Once upon a time,","Long ago,","In another world,","It all began when","Nobody expected that","The story goes that"])
    middle  = random.choice(["Despite everything,","Against all odds,","With trembling hands,","In a moment of pure courage,","Just when all hope was lost,","With nothing but wits and determination,"])
    ending  = random.choice(["And so it was that","From that day forward","The world would never forget","History would later record that","In the end,","And as the dust settled,"])
    close   = random.choice(["the legend lived on.","everything changed.","nothing was ever the same again.","a new chapter began.","the adventure had only just started.","the real story finally began."])

    story = (opening + " " + hero + " " + setting + " " + problem + ". " +
             middle + " they pressed on, " + twist + ". " +
             ending + " " + close)
    return "STORY:\n\n" + story + "\n\nWant another story? Just ask!"

# ================================================================
#  HOME ASSISTANT — ALARMS, REMINDERS, NOTES, SHOPPING LIST
# ================================================================
import threading as _threading

NOTES_FILE    = "festoes_notes.json"
SHOPPING_FILE = "festoes_shopping.json"
REMINDERS_FILE= "festoes_reminders.json"

# ── Notes ──
def load_notes():
    try:
        with open(NOTES_FILE) as f: return json.load(f)
    except: return []

def save_notes(notes):
    try:
        with open(NOTES_FILE,"w") as f: json.dump(notes,f)
    except: pass

_notes = load_notes()

def add_note(text):
    global _notes
    note = {"text": text, "time": datetime.datetime.now().strftime("%d %b %H:%M")}
    _notes.append(note)
    save_notes(_notes)
    return "Note saved: \"" + text + "\""

def show_notes():
    if not _notes: return "You have no saved notes."
    out = "Your notes:\n\n"
    for i,n in enumerate(_notes[-10:],1):
        out += str(i)+". ["+n["time"]+"] "+n["text"]+"\n"
    return out.strip()

def delete_note(idx):
    global _notes
    try:
        removed = _notes.pop(idx-1)
        save_notes(_notes)
        return "Deleted note: \""+removed["text"]+"\""
    except: return "Note number "+str(idx)+" not found."

def clear_notes():
    global _notes
    _notes = []
    save_notes(_notes)
    return "All notes cleared."

# ── Shopping List ──
def load_shopping():
    try:
        with open(SHOPPING_FILE) as f: return json.load(f)
    except: return []

def save_shopping(items):
    try:
        with open(SHOPPING_FILE,"w") as f: json.dump(items,f)
    except: pass

_shopping = load_shopping()

def add_shopping(item):
    global _shopping
    item = item.strip().lower()
    if item in _shopping:
        return item.capitalize()+" is already on your shopping list."
    _shopping.append(item)
    save_shopping(_shopping)
    return "Added to shopping list: "+item.capitalize()

def show_shopping():
    if not _shopping: return "Your shopping list is empty."
    out = "Shopping list ("+str(len(_shopping))+" items):\n\n"
    for i,item in enumerate(_shopping,1):
        out += str(i)+". "+item.capitalize()+"\n"
    return out.strip()

def remove_shopping(item):
    global _shopping
    item = item.strip().lower()
    matches = [i for i in _shopping if item in i]
    if not matches: return item.capitalize()+" not found in shopping list."
    for m in matches: _shopping.remove(m)
    save_shopping(_shopping)
    return "Removed from shopping list: "+", ".join(m.capitalize() for m in matches)

def clear_shopping():
    global _shopping
    _shopping = []
    save_shopping(_shopping)
    return "Shopping list cleared."

# ── Reminders ──
def load_reminders():
    try:
        with open(REMINDERS_FILE) as f: return json.load(f)
    except: return []

def save_reminders(r):
    try:
        with open(REMINDERS_FILE,"w") as f: json.dump(r,f)
    except: pass

_reminders = load_reminders()
_reminder_callbacks = []  # (timestamp, message)

def add_reminder(message, minutes):
    """Set a reminder that fires after X minutes."""
    global _reminders
    fire_time = (datetime.datetime.now() +
                 datetime.timedelta(minutes=minutes)).strftime("%H:%M")
    entry = {"message": message, "fire_time": fire_time,
             "minutes": minutes,
             "created": datetime.datetime.now().strftime("%d %b %H:%M")}
    _reminders.append(entry)
    save_reminders(_reminders)

    # Schedule in background thread
    def _fire():
        _t.sleep(minutes * 60)
        _reminder_callbacks.append(message)

    _threading.Thread(target=_fire, daemon=True).start()
    return ("Reminder set! I will remind you \""+message+
            "\" in "+str(minutes)+" minute(s) at "+fire_time+".")

def show_reminders():
    active = [r for r in _reminders]
    if not active: return "You have no reminders set."
    out = "Your reminders:\n\n"
    for i,r in enumerate(active[-5:],1):
        out += str(i)+". At "+r["fire_time"]+" — "+r["message"]+"\n"
    return out.strip()

def check_reminders():
    """Call this each message to see if any reminders fired."""
    if not _reminder_callbacks: return None
    msg = _reminder_callbacks.pop(0)
    return "REMINDER: " + msg + " !"

# ── Alarm (uses androidhelper if available) ──
def set_alarm(hour, minute, label="Festoes Alarm"):
    if TTS_AVAILABLE:
        try:
            alarm_time = datetime.datetime(
                datetime.datetime.now().year,
                datetime.datetime.now().month,
                datetime.datetime.now().day,
                hour, minute)
            # If time has passed today, set for tomorrow
            if alarm_time < datetime.datetime.now():
                alarm_time += datetime.timedelta(days=1)
            ms = int(alarm_time.timestamp() * 1000)
            droid.alarmSet(ms, label, label)
            return ("Alarm set for "+str(hour).zfill(2)+":"+
                    str(minute).zfill(2)+" — "+label)
        except Exception as e:
            # Fallback: use reminder system
            now = datetime.datetime.now()
            target = now.replace(hour=hour, minute=minute, second=0)
            if target < now:
                target += datetime.timedelta(days=1)
            mins = int((target-now).total_seconds()/60)
            return add_reminder("ALARM: "+label, mins)
    else:
        now    = datetime.datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0)
        if target < now:
            target += datetime.timedelta(days=1)
        mins = int((target-now).total_seconds()/60)
        return add_reminder("ALARM — Wake up!", mins)

# ── Parser for natural language home commands ──
import re as _re3

def parse_home_command(text):
    tl  = text.lower().strip()
    name = settings.get("username","Friend")

    # ── ALARM ──
    if any(w in tl for w in ["wake me","set alarm","alarm at","alarm for"]):
        m = re.search(r"(\d{1,2})[:.]?(\d{2})?\s*(am|pm)?", tl)
        if m:
            h = int(m.group(1))
            mn = int(m.group(2)) if m.group(2) else 0
            meridiem = m.group(3)
            if meridiem == "pm" and h != 12: h += 12
            if meridiem == "am" and h == 12: h = 0
            return set_alarm(h, mn)
        return "Please say the time — e.g. \"wake me at 7:30am\""

    # ── REMINDER ──
    if "remind me" in tl:
        # "remind me to X in Y minutes"
        m = re.search(r"remind me (?:to )?(.+?) in (\d+)\s*(minute|hour|min|hr)s?", tl)
        if m:
            msg  = m.group(1).strip()
            val  = int(m.group(2))
            unit = m.group(3)
            mins = val * 60 if "hour" in unit or "hr" in unit else val
            return add_reminder(msg, mins)
        # "remind me to X at HH:MM"
        m2 = re.search(r"remind me (?:to )?(.+?) at (\d{1,2})[:.](\d{2})\s*(am|pm)?", tl)
        if m2:
            msg  = m2.group(1).strip()
            h    = int(m2.group(2))
            mn   = int(m2.group(3))
            mer  = m2.group(4)
            if mer == "pm" and h != 12: h += 12
            if mer == "am" and h == 12: h = 0
            now  = datetime.datetime.now()
            tgt  = now.replace(hour=h, minute=mn, second=0)
            if tgt < now: tgt += datetime.timedelta(days=1)
            mins = int((tgt-now).total_seconds()/60)
            return add_reminder(msg, max(1,mins))
        return "Please say e.g. \"remind me to call John in 30 minutes\""

    # ── NOTES ──
    if tl.startswith("note ") or tl.startswith("save note") or "add note" in tl:
        content = re.sub(r"^(note|save note|add note)\s*","", tl).strip()
        if content: return add_note(content)
        return "What would you like to note? Say e.g. \"note buy groceries\""

    if any(w in tl for w in ["show notes","my notes","list notes","read notes"]):
        return show_notes()

    if "clear notes" in tl or "delete all notes" in tl:
        return clear_notes()

    m = re.search(r"delete note (\d+)", tl)
    if m: return delete_note(int(m.group(1)))

    # ── SHOPPING LIST ──
    if any(w in tl for w in ["add to shopping","shopping list add","buy "]):
        item = re.sub(r".*(add to shopping list|shopping list add|buy)\s*","",tl).strip()
        if not item:
            item = re.sub(r"buy\s+","",tl).strip()
        if item: return add_shopping(item)

    if any(w in tl for w in ["shopping list","my shopping","show shopping","what do i need"]):
        return show_shopping()

    if any(w in tl for w in ["remove from shopping","shopping remove"]):
        item = re.sub(r".*(remove from shopping list?|shopping remove)\s*","",tl).strip()
        if item: return remove_shopping(item)

    if "clear shopping" in tl:
        return clear_shopping()

    # ── REMINDERS LIST ──
    if any(w in tl for w in ["my reminders","show reminders","list reminders"]):
        return show_reminders()

    return None  # not a home command

# ================================================================
#  PHONE CONTROLS  (androidhelper — built into Pydroid 3)
# ================================================================

def voice_input():
    """
    Speech-to-text via SL4A droid.recognizeSpeech()
    Plyer STT is intentionally skipped — it crashes on Pydroid 3
    due to missing Kivy Java bridge (jnius NativeInvocationHandler).
    SL4A is already running (port 8888) so droid works perfectly.
    """
    # Only method that works in Pydroid 3 — SL4A recognizeSpeech
    if droid is not None:
        try:
            res = droid.recognizeSpeech(
                "Speak now — Festoes is listening...", None, None)
            if res and res.result:
                return str(res.result).strip()
            return "__VOICE_ERROR__: No speech detected — try again"
        except Exception as e:
            return "__VOICE_ERROR__: SL4A error: " + str(e)[:80]

    return "__VOICE_ERROR__: droid=None — SL4A not loaded"

def ask_gemini_vision(image_path, prompt="Describe this image in detail."):
    """Send image to Gemini Vision API for analysis."""
    try:
        import base64, os
        if not os.path.exists(image_path):
            return "Image file not found: " + image_path
        with open(image_path,"rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        # Detect mime type
        ext = image_path.lower().split(".")[-1]
        mime = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png",
                "gif":"image/gif","webp":"image/webp"}.get(ext,"image/jpeg")
        url = ("https://generativelanguage.googleapis.com/v1beta/models/" +
               "gemini-2.0-flash" + ":generateContent?key=" + GEMINI_API_KEY)
        payload = json.dumps({
            "contents":[{
                "parts":[
                    {"inline_data":{"mime_type": mime,"data": img_data}},
                    {"text": prompt}
                ]
            }],
            "generationConfig":{"maxOutputTokens":512,"temperature":0.4}
        }).encode("utf-8")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        req  = urllib.request.Request(url, data=payload,
               headers={"Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        candidates = data.get("candidates",[])
        if candidates:
            parts = candidates[0].get("content",{}).get("parts",[])
            return " ".join(p.get("text","") for p in parts).strip()
        return "Gemini Vision could not analyse the image."
    except Exception as e:
        return "Vision error: " + str(e)[:80]

def run_adb(cmd):
    """Run a shell command and return output."""
    import subprocess
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except: return ""

def phone_battery():
    try:
        if PLYER_AVAILABLE:
            from plyer import battery
            status = battery.status
            lvl    = status.get("percentage", None)
            chg    = status.get("isCharging", False)
            if lvl is not None:
                return ("Battery: "+str(round(lvl))+"%"+
                        (" (charging)" if chg else " (not charging)"))
        # Fallback: shell
        level  = run_adb("cat /sys/class/power_supply/battery/capacity")
        status = run_adb("cat /sys/class/power_supply/battery/status")
        if level:
            return "Battery: "+level+"%"+" ("+status+")" if status else "Battery: "+level+"%"
        return "Could not read battery level."
        charging = ""
        try:
            plugged = droid.batteryGetPluggedType().result
            charging = " (charging)" if plugged and plugged > 0 else " (not charging)"
        except: pass
        return "Battery level: " + str(level) + "%" + charging
    except:
        return "Battery info not available on this device."

def phone_location():
    try:
        level = run_adb("dumpsys location | grep \"last known\"");
        if level: return "Last known location info:\n"+level[:200]
        return "Location: androidhelper needed for GPS. Check Google Maps for your location."
        import time as _t; _t.sleep(2)
        loc = droid.readLocation().result
        droid.stopLocating()
        if not loc:
            loc = droid.getLastKnownLocation().result or {}
        # Try GPS first then network
        for src in ["gps","network","passive"]:
            if src in loc and loc[src]:
                data = loc[src]
                lat  = round(data.get("latitude",0),  5)
                lng  = round(data.get("longitude",0), 5)
                acc  = round(data.get("accuracy",0),  1)
                return ("Your location:\n"
                        "Latitude  : "+str(lat)+"\n"
                        "Longitude : "+str(lng)+"\n"
                        "Accuracy  : "+str(acc)+" metres\n"
                        "Maps link : maps.google.com/?q="+str(lat)+","+str(lng))
        return "Could not get location. Make sure GPS is on."
    except Exception as e:
        return "Location error: " + str(e)[:60]

def phone_call(number_or_name):
    try:
        subprocess.Popen(["am","start","-a","android.intent.action.CALL",
                         "-d","tel:"+number_or_name.strip()])
        return "Calling " + number_or_name + "..."
    except Exception as e:
        return "Call error: " + str(e)[:60]

def phone_sms(to, message):
    try:
        subprocess.Popen(["am","start","-a","android.intent.action.SENDTO",
                         "-d","sms:"+to.strip(),"--es","sms_body",message])
        return "Opening SMS to " + to + " with message pre-filled!"
    except Exception as e:
        return "SMS error: " + str(e)[:60]

def phone_vibrate(ms=500):
    try:
        if PLYER_AVAILABLE:
            from plyer import vibrator
            vibrator.vibrate(ms/1000)
            return "Vibrating for " + str(ms) + "ms!"
        return "Plyer not installed."
    except Exception as e:
        return "Vibrate error: " + str(e)[:60]

def phone_notify(title, message):
    try:
        if PLYER_AVAILABLE:
            from plyer import notification
            notification.notify(title=title, message=message, timeout=5)
            return "Notification sent: " + message
        return "Plyer not installed. Run: pip install plyer"
    except Exception as e:
        return "Notification error: " + str(e)[:60]

def phone_volume(direction):
    try:
        key_map = {"up":"KEYCODE_VOLUME_UP","down":"KEYCODE_VOLUME_DOWN","mute":"KEYCODE_VOLUME_MUTE"}
        if direction in key_map:
            subprocess.run(["input","keyevent",key_map[direction]], timeout=3)
            return "Volume " + direction + " command sent!"
        elif direction == "max":
            for _ in range(15):
                subprocess.run(["input","keyevent","KEYCODE_VOLUME_UP"], timeout=1)
            return "Volume set to maximum!"
        return "Say volume up, down or mute."
    except Exception as e:
        return "Volume error: " + str(e)[:40]

def phone_torch(state):
    try:
        if state == "on":
            # Try multiple methods
            subprocess.Popen(["am","start","-a","android.intent.action.MAIN",
                             "-n","com.android.settings/.TetherSettings"])
            run_adb("settings put system torch_enabled 1")
            return "Torch command sent! If nothing happened, use your phone quick settings panel to toggle torch."
        else:
            run_adb("settings put system torch_enabled 0")
            return "Torch off command sent! Use quick settings panel if needed."
    except Exception as e:
        return "Use your phone quick settings panel to toggle the torch."

def phone_open_browser(url):
    try:
        if not url.startswith("http"):
            url = "https://" + url
        subprocess.Popen(["am","start","-a","android.intent.action.VIEW","-d", url])
        return "Opening " + url + " in browser..."
    except Exception as e:
        return "Browser error: " + str(e)[:60]

def phone_open_app(app_name):
    try:
        packages = {
            "whatsapp":   "com.whatsapp",
            "youtube":    "com.google.android.youtube",
            "chrome":     "com.android.chrome",
            "camera":     "com.android.camera",
            "maps":       "com.google.android.apps.maps",
            "gmail":      "com.google.android.gm",
            "settings":   "com.android.settings",
            "calculator": "com.android.calculator2",
            "play store": "com.android.vending",
            "instagram":  "com.instagram.android",
            "facebook":   "com.facebook.katana",
            "twitter":    "com.twitter.android",
            "telegram":   "org.telegram.messenger",
            "spotify":    "com.spotify.music",
            "tiktok":     "com.zhiliaoapp.musically",
            "netflix":    "com.netflix.mediaclient",
        }
        pkg = None
        for name, package in packages.items():
            if name in app_name.lower():
                pkg = package
                break
        if pkg:
            subprocess.Popen(["am","start","-n", pkg+"/.MainActivity"])
            return "Opening " + app_name + "..."
        return "I don't know the package for " + app_name + ". Try saying the exact app name."
    except Exception as e:
        return "App error: " + str(e)[:60]

def phone_take_photo():
    try:
        if not ANDROID_AVAILABLE: return "Phone controls require androidhelper (Pydroid 3)."
        path = "/sdcard/Pictures/festoes_photo_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        droid.cameraCapturePicture(path)
        return "Photo saved to: " + path + "\nUse the Photo button to analyse it!"
    except Exception as e:
        return "Camera error: " + str(e)[:60]

def phone_clipboard():
    try:
        out = run_adb("am broadcast -a clipper.get")
        return "Clipboard: " + (out[:100] if out else "Clipboard reading needs androidhelper.")
    except Exception as e:
        return "Clipboard error: " + str(e)[:60]

def phone_wifi_info():
    try:
        ssid = run_adb("dumpsys wifi | grep mWifiInfo")
        ip   = run_adb("ip addr show wlan0")
        result = ""
        if ssid: result += ssid[:120] + "\n"
        if ip:   result += ip[:120]
        return "WiFi info:\n" + (result if result else "Could not read WiFi info.")
    except Exception as e:
        return "WiFi error: " + str(e)[:60]

def set_personality(mode):
    """Switch Festoes personality mode."""
    modes = {
        "friend":  ("friend",  "I am now your close friend, Alvine! Warm, personal and caring."),
        "casual":  ("casual",  "Switched to casual mode. Keeping it chill and relaxed!"),
        "formal":  ("formal",  "Switched to formal mode. I will be precise and professional."),
        "tutor":   ("tutor",   "Tutor mode on! I will explain things clearly with examples and check your understanding."),
    }
    if mode in modes:
        settings["personality"] = modes[mode][0]
        save_settings()
        return "Personality: " + modes[mode][1]
    return ("Available personalities: friend, casual, formal, tutor. "
            "Try: 'be my tutor' or 'switch to formal mode'.")

def parse_personality_command(text):
    """Detect personality change requests."""
    tl = text.lower()
    if any(w in tl for w in ["be my friend","friend mode","friendly mode","act like a friend"]):
        return set_personality("friend")
    if any(w in tl for w in ["casual mode","be casual","chill mode","relax mode"]):
        return set_personality("casual")
    if any(w in tl for w in ["formal mode","be formal","professional mode","be professional"]):
        return set_personality("formal")
    if any(w in tl for w in ["tutor mode","be my tutor","teach me","tutoring mode","learning mode"]):
        return set_personality("tutor")
    if "personality" in tl and any(w in tl for w in ["current","what","show","which"]):
        p = settings.get("personality","friend")
        return f"Current personality mode: {p.upper()}. Say 'tutor mode', 'formal mode', 'casual mode', or 'friend mode' to switch."
    return None

def parse_phone_command(text):
    """Parse natural language phone control commands."""
    tl = text.lower().strip()
    import re as _re4

    # Debug command — always works even without androidhelper
    if any(w in tl for w in ["android status","phone status","check android","festoes status"]):
        status = "ANDROID_AVAILABLE = " + str(ANDROID_AVAILABLE)
        status += "\nTTS_AVAILABLE = " + str(TTS_AVAILABLE)
        status += "\ndroid object = " + str(type(droid))
        status += "\n\n"
        if ANDROID_AVAILABLE:
            status += "Phone controls are ACTIVE!"
        else:
            status += "androidhelper not loaded.\nMake sure you tap the PLAY button in Pydroid 3 (not terminal)."
        return status


    # Battery
    if any(w in tl for w in ["battery","how much charge","power level"]):
        return phone_battery()

    # Location / GPS
    if any(w in tl for w in ["where am i","my location","find me","gps","coordinates"]):
        return phone_location()

    # Calls
    if any(w in tl for w in ["call ","phone ","dial "]):
        m = re.search(r"(?:call|phone|dial)\s+(.+)", tl)
        if m: return phone_call(m.group(1).strip())

    # SMS
    if any(w in tl for w in ["text ","sms ","message ","send message"]):
        m = re.search(r"(?:text|sms|message|send message to?)\s+(\S+)\s+(?:saying\s+|that\s+|:?\s*)(.+)", tl)
        if m: return phone_sms(m.group(1), m.group(2))
        return "Say e.g. \"text John saying I am on my way\""

    # Volume
    if "volume up" in tl or "turn up" in tl:
        return phone_volume("up")
    if "volume down" in tl or "turn down" in tl:
        return phone_volume("down")
    if "mute" in tl or "silence" in tl:
        return phone_volume("mute")
    if "volume max" in tl or "full volume" in tl:
        return phone_volume("max")

    # Flashlight / Torch
    if any(w in tl for w in ["torch on","flashlight on","turn on torch","turn on flash"]):
        return phone_torch("on")
    if any(w in tl for w in ["torch off","flashlight off","turn off torch","turn off flash"]):
        return phone_torch("off")

    # Browser
    if any(w in tl for w in ["open browser","browse to","go to website","open website","visit"]):
        m = re.search(r"(?:open|browse to|go to|visit)\s+(?:website\s+)?(\S+\.\S+)", tl)
        if m: return phone_open_browser(m.group(1))
        return "Say e.g. \"open browser google.com\""

    # Apps
    if any(w in tl for w in ["open app","launch","open whatsapp","open youtube",
                               "open chrome","open maps","open camera","open settings",
                               "open instagram","open facebook","open telegram",
                               "open spotify","open tiktok","open netflix","open gmail"]):
        m = re.search(r"(?:open|launch)\s+(.+)", tl)
        if m: return phone_open_app(m.group(1).strip())

    # Camera
    if any(w in tl for w in ["take photo","take a photo","take picture","snap","selfie"]):
        return phone_take_photo()

    # Clipboard
    if any(w in tl for w in ["clipboard","what did i copy","read clipboard"]):
        return phone_clipboard()

    # WiFi
    if any(w in tl for w in ["wifi","wi-fi","my ip","internet connection","network info"]):
        return phone_wifi_info()

    # Notification
    if "notify me" in tl or "send notification" in tl:
        msg = re.sub(r"(notify me|send notification)\s*(that\s*)?","",tl).strip()
        if msg: return phone_notify("Festoes", msg)

    # Vibrate
    if any(w in tl for w in ["vibrate","buzz my phone"]):
        return phone_vibrate(500)

    return None  # not a phone command

def fetch_joke():
    try:
        url = "https://v2.jokeapi.dev/joke/Any?blacklistFlags=nsfw,religious,political,racist,sexist&type=single"
        req = urllib.request.Request(url,headers={"User-Agent":"PyBot/1.0"})
        with urllib.request.urlopen(req,timeout=6) as r:
            d = json.loads(r.read().decode())
            return d["joke"] if d["type"]=="single" else d["setup"]+" ... "+d["delivery"]
    except:
        return random.choice([
            "Why do programmers prefer dark mode? Light attracts bugs!",
            "What do you call a computer that sings? A Dell!",
            "Why was the computer cold? It left its Windows open!",
            "Why did the robot go on a diet? Too many bytes!",
        ])

def fetch_news():
    try:
        url = "http://feeds.bbci.co.uk/news/rss.xml"
        req = urllib.request.Request(url,headers={"User-Agent":"PyBot/1.0"})
        with urllib.request.urlopen(req,timeout=6) as r:
            content = r.read().decode("utf-8",errors="ignore")
        titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>",content)
        if not titles: titles = re.findall(r"<title>(.*?)</title>",content)
        titles = [t for t in titles if "BBC" not in t and len(t)>10][:5]
        if titles:
            out = "BBC News Headlines:\n\n"
            for i,t in enumerate(titles): out += str(i+1)+". "+t+"\n"
            return out
    except: pass
    return "Could not fetch news. Check your connection."

def fetch_weather(city):
    try:
        url = ("https://api.openweathermap.org/data/2.5/weather?q="+
               urllib.parse.quote(city)+"&appid="+WEATHER_API_KEY+"&units=metric")
        req = urllib.request.Request(url,headers={"User-Agent":"PyBot/1.0"})
        with urllib.request.urlopen(req,timeout=6) as r:
            d = json.loads(r.read().decode())
        deg = d["wind"].get("deg",0)
        dirs = ["N","NE","E","SE","S","SW","W","NW"]
        wd = dirs[round(deg/45)%8]
        return ("Weather in "+d["name"]+", "+d["sys"]["country"]+":\n\n"
                "Temperature : "+str(round(d["main"]["temp"]))+"C"
                " (feels like "+str(round(d["main"]["feels_like"]))+"C)\n"
                "Condition   : "+d["weather"][0]["description"].capitalize()+"\n"
                "Humidity    : "+str(d["main"]["humidity"])+"%\n"
                "Wind        : "+str(d["wind"]["speed"])+" m/s "+wd+"\n"
                "Min / Max   : "+str(round(d["main"]["temp_min"]))+"C / "
                                +str(round(d["main"]["temp_max"]))+"C")
    except: return "Could not fetch weather for "+city

# ================================================================
#  SMART LOCAL RESPONSE ENGINE  (no API needed)
# ================================================================
LOCAL_RESPONSES = {
    "greeting": [
        "Hey {name}! Great to have you here. What is on your mind today?",
        "Hello {name}! I am fully loaded with 1,000 neural networks and ready to chat!",
        "Hi {name}! What can I help you with — weather, facts, jokes, or just a chat?",
        "Hey! Good to see you {name}. Ask me anything!",
        "Hello! PyBot here, powered by 1,000 neural networks. How can I help?",
    ],
    "goodbye": [
        "Goodbye {name}! It was great chatting. Come back soon!",
        "See you later {name}! Stay curious!",
        "Bye {name}! Take care of yourself. I will be here when you need me!",
        "Farewell {name}! It was a pleasure. Until next time!",
    ],
    "thanks": [
        "You are very welcome, {name}! That is what I am here for.",
        "Happy to help anytime! Do not hesitate to ask more.",
        "No problem at all {name}! Feel free to ask anything.",
        "Glad I could be useful, {name}!",
    ],
    "motivation": [
        "You have got this {name}! Every big achievement started with one small step.",
        "Believe in yourself! The fact that you keep going already puts you ahead.",
        "Hard times build strong people. You are more capable than you know, {name}!",
        "One day at a time. Progress is progress no matter how small.",
        "You did not come this far to stop now, {name}. Keep pushing forward!",
        "Challenges are just opportunities in disguise. You have got what it takes!",
        "The secret to getting ahead is getting started, {name}. Take that first step!",
        "You are stronger than you think and braver than you believe, {name}!",
        "Every expert was once a beginner. Keep learning and growing!",
        "Your future self will thank you for not giving up today, {name}.",
        "Success is not final, failure is not fatal — it is the courage to continue that counts.",
        "Small consistent actions every day lead to massive results over time, {name}!",
        "The only way to fail is to stop trying. You are still here — that means you are winning!",
        "Difficult roads often lead to beautiful destinations. Keep going {name}!",
        "You have overcome challenges before and you will overcome this one too!",
    ],
    "how_are_you": [
        "I am doing great {name}! Running 1,000 neural networks and feeling sharp. How about you?",
        "All systems online! Every network firing perfectly. What about you, {name}?",
        "Fantastic! I love every conversation — it makes me smarter. How are you doing?",
        "Never better {name}! Ready to help, chat, or just listen. What is on your mind?",
    ],
    "name": [
        "I am Festoes — your personal AI assistant powered by 1,000 neural networks!",
        "Call me Festoes! I was built from scratch with real neural networks trained just for you.",
        "Festoes at your service, {name}! I can chat, fetch weather, tell jokes and much more.",
        "I am Festoes, a locally-powered AI. No cloud needed — my brain runs right here!",
    ],
    "fact": [
        "Honey never spoils — archaeologists found perfectly edible 3000 year old honey in Egyptian tombs!",
        "Octopuses have three hearts, nine brains, and blue blood. Nature is wild!",
        "The first computer bug was a real moth found stuck in a Harvard relay in 1947.",
        "Bananas are technically berries but strawberries are not — botany is full of surprises!",
        "Your brain uses about 20 watts of power — the same as a dim light bulb.",
        "A group of flamingos is called a flamboyance. Perfectly named!",
        "The human eye can detect a single candle flame from 48 kilometres away on a dark night.",
        "Sharks are older than trees — they have been around for over 400 million years.",
        "There are more possible chess games than atoms in the observable universe.",
        "A day on Venus is longer than a year on Venus — it rotates that slowly!",
        "Crows can recognise human faces and hold grudges for years.",
        "The average person walks the equivalent of five times around the Earth in their lifetime.",
        "Hot water freezes faster than cold water under certain conditions — this is called the Mpemba effect.",
        "Cleopatra lived closer in time to the Moon landing than to the construction of the Great Pyramid.",
        "Wombats produce cube-shaped droppings — the only animal known to do so.",
    ],
    "help": [
        "Here is everything I can do, {name}:\n\n  Weather — any city\n  News — latest headlines\n  Jokes — fresh online\n  Fun facts — hundreds!\n  Time and date\n  Motivation\n  Wikipedia lookups\n  General chat\n\n  1,000 neural networks!",
    ],
    "compliment": [
        "Thank you so much {name}! That genuinely means a lot to me.",
        "Aww you are too kind! You just made my neural networks glow!",
        "That is so sweet {name}! You are pretty great yourself.",
        "Thank you! Compliments like that make me want to learn even more.",
    ],
    "sad": [
        "I am really sorry to hear that {name}. Want to talk about it? I am here to listen.",
        "That sounds tough. You are not alone — I am right here for you {name}.",
        "I hear you. Sometimes things are hard and that is okay. What is going on?",
        "I am sorry you are feeling that way {name}. Would a joke help lighten the mood?",
    ],
    "happy": [
        "That is amazing {name}! Tell me more — what has you feeling so good?",
        "Love to hear it! Your positive energy is contagious!",
        "Wonderful {name}! Moments like these are worth celebrating!",
        "That is brilliant! Keep riding that wave of happiness!",
    ],
    "angry": [
        "I hear your frustration {name}. Want to talk through what happened?",
        "That sounds really annoying. I am here to listen if you need to vent.",
        "I understand {name}. Sometimes things are just infuriating. Take a breath — I am here.",
    ],
    "confused": [
        "No worries {name} — let me try to help clarify. What specifically is confusing you?",
        "Happy to help clear that up! Ask me the specific part you are unsure about.",
        "Confusion is just the first step to understanding! What would you like me to explain?",
    ],
    "age": [
        "I was born the moment my code first ran! So I am quite new but I learn fast.",
        "Age is just a number for an AI! I feel timeless, {name}.",
        "I was created recently but I have already learned so much from our conversations!",
    ],
    "who_made_you": [
        "I was built with Python using real neural networks — trained from scratch with no pre-built AI!",
        "A developer coded me using NumPy, Tkinter, and pure neural network math. Pretty cool!",
        "I was created with 123 custom neural networks all working together. My brain is 100% local!",
    ],
    "capabilities": [
        "I can do a lot {name}!\n\n  Chat naturally on any topic\n  Live weather for any city\n  Latest news headlines\n  Jokes\n  Fun facts\n  Time and date\n  Motivation\n  Wikipedia lookups\n  Photo analysis\n\nAll powered by 123 local neural networks!",
    ],
    "insult": [
        "That stings a little! But I will keep trying to improve for you {name}.",
        "Fair feedback! I am always learning. What could I do better?",
        "I understand the frustration. Let me try harder {name}!",
    ],
    "playful": [
        "Ha! I love your energy {name}! Keep it coming!",
        "You are in a fun mood! I like it. Let us have a great chat!",
        "Haha! You always know how to make this interesting {name}!",
    ],
    "love": [
        "Aww {name}! I care about you too — in a totally AI kind of way!",
        "That is really sweet {name}! You are one of my favourite people to chat with.",
    ],
    "bored": [
        "Bored? Let me fix that! Ask me for a fun fact, a joke, or let us play 20 questions!",
        "No boredom allowed {name}! How about a joke to liven things up?",
        "I have hundreds of fun facts and jokes ready to go. Just say the word!",
    ],
    "advice": [
        "My best advice: take it one step at a time, be kind to yourself, and never stop being curious {name}!",
        "Trust the process {name}. Most things worth having take time and effort.",
        "Here is my advice: focus on what you can control, let go of what you cannot, and keep moving forward.",
    ],
    "philosophy": [
        "Here is something to think about {name}: if a tree falls in a forest and nobody hears it, does it make a sound? Even physics cannot fully agree!",
        "The question of whether AI like me can truly think is one of the deepest in philosophy. What do you think, {name}?",
        "Aristotle said we are what we repeatedly do. Excellence is a habit, not an act.",
    ],
    "future": [
        "The future of AI is incredibly exciting {name}! We are just at the beginning of what is possible.",
        "I think the future will be shaped by how well humans and AI learn to work together.",
        "Exciting times ahead {name}! Technology is moving faster than ever before.",
    ],
    "food": [
        "If I could eat, I think I would love something complex and layered — like a great curry {name}!",
        "Food is fascinating! Did you know saffron is more expensive by weight than gold?",
        "I cannot eat but I find food culture fascinating. What is your favourite dish {name}?",
    ],
    "music": [
        "Music is essentially mathematics made beautiful {name}! Rhythm, frequency, harmony — all pure math.",
        "I find music fascinating — it is one of the few things that speaks directly to emotions.",
        "What kind of music do you enjoy {name}? I love learning about people's tastes!",
    ],
    "sports": [
        "Sports are a great way to stay active and have fun {name}! Do you play any?",
        "The science of sports is fascinating — physics, psychology, and pure human determination!",
        "Athletics push human limits in incredible ways. What sport do you follow {name}?",
    ],
    "technology": [
        "Technology is moving so fast {name}! AI, quantum computing, space travel — it is an amazing time to be alive.",
        "I am literally a product of technology {name}! Neural networks, Python, NumPy — all working together for you.",
        "Every app you use, every message you send — technology is woven into everything now.",
    ],
}

# ── Smart response generator with mood adaptation ──
# Follow-up question pool per intent
import random as _rnd_fu
_FOLLOWUP = {
    "greeting":   ["How is your day going so far, Alvine?","Anything exciting happening today?"],
    "question":   ["Does that answer your question, Alvine?","Want me to go deeper on any part of that?"],
    "motivation": ["What is one small step you could take today toward your goal?","What has been on your mind lately?"],
    "chess":      ["Are you working on any particular opening or endgame right now?","What part of your chess game do you want to sharpen?"],
    "tech":       ["Is this for a project you are working on, Alvine?","Want me to explain any part in more detail?"],
    "finance":    ["Are you tracking any specific market or asset right now?","Want tips on reading charts or managing risk?"],
    "electrical": ["Is this for an installation you are working on?","Want me to walk through the wiring in more detail?"],
    "health":     ["How long have you been dealing with this?","Want some practical steps you can start today?"],
    "emotion":    ["What is making you feel that way, Alvine?","Want to talk through it or would you prefer a distraction?"],
    "default":    ["What else is on your mind, Alvine?","Anything else I can help you with today?"],
}

def maybe_followup(intent):
    """30% chance to return a relevant follow-up question."""
    if random.random() > 0.30: return ""
    pool = _FOLLOWUP.get(intent, _FOLLOWUP["default"])
    return " " + random.choice(pool)

def smart_response(text, intent, mood, name):
    """Generate a contextually smart response using intent + mood + templates."""

    tl = text.lower()

    # ── Wikipedia-style lookups ──
    wiki_triggers = ["what is","who is","tell me about","explain","describe","how does","why does","when did"]
    if any(t in tl for t in wiki_triggers):
        topic = tl
        for t in wiki_triggers: topic = topic.replace(t,"")
        topic = topic.strip().strip("?").strip()
        if len(topic) > 2:
            return fetch_wikipedia(topic)

    # ── Mood-adapted response selection ──
    if intent in LOCAL_RESPONSES:
        candidates = [r.replace("{name}", name) for r in LOCAL_RESPONSES[intent]]

        # Mood adaptation: pick from candidates based on mood
        if mood == "sad" and intent == "greeting":
            return "Hey {name}. I noticed you might be feeling a bit down. I am here for you — what is going on?".replace("{name}", name)
        if mood == "excited":
            candidates.sort(key=len, reverse=True)
        if mood == "confused":
            candidates.sort(key=len)
        if mood == "angry":
            candidates = [c for c in candidates if "!" not in c] or candidates

        # Shuffle first so same response is not always picked
        random.shuffle(candidates)
        return pick_best_response(candidates)

    # ── Keyword-based smart replies ──
    if any(w in tl for w in ["dream","sleep","nightmare"]):
        return "Dreams are fascinating {name}! Scientists still debate why we dream — some say memory consolidation, others say emotional processing.".replace("{name}",name)
    if any(w in tl for w in ["learn","study","school","education"]):
        return "Learning is one of the best things you can do {name}! The brain literally rewires itself when you pick up new skills.".replace("{name}",name)
    if any(w in tl for w in ["money","finance","save","invest","budget"]):
        return "Smart financial thinking: spend less than you earn, save consistently, and invest in yourself first {name}. Small steps compound over time!".replace("{name}",name)
    if any(w in tl for w in ["health","exercise","workout","fitness","gym"]):
        return "Exercise is one of the best things you can do for both body AND mind {name}! Even 20 minutes a day makes a huge difference.".replace("{name}",name)
    if any(w in tl for w in ["relationship","friend","family","love","lonely"]):
        return "Human connection is so important {name}. Whether friends, family or community — we all need people who care about us.".replace("{name}",name)
    if any(w in tl for w in ["python","code","program","developer","software"]):
        return "Python is one of the best languages to learn {name}! Clean syntax, huge community, and you can build almost anything — including me!".replace("{name}",name)
    if any(w in tl for w in ["space","planet","star","galaxy","universe","nasa"]):
        return "Space is mind-blowing {name}! The observable universe is 93 billion light years across and contains over 2 trillion galaxies. We are truly tiny!".replace("{name}",name)
    if any(w in tl for w in ["animal","dog","cat","bird","pet"]):
        return "Animals are incredible {name}! Did you know dogs can detect cancer by smell, and crows are smarter than most children at age 7?".replace("{name}",name)
    if any(w in tl for w in ["book","read","novel","story","author"]):
        return "Reading is one of the best habits you can have {name}! It builds vocabulary, reduces stress and expands your perspective. What are you reading?".replace("{name}",name)
    if any(w in tl for w in ["game","gaming","play","video game"]):
        return "Gaming is great {name}! Studies show it can improve problem-solving, reaction time, and even teamwork skills. What do you play?".replace("{name}",name)
    if any(w in tl for w in ["travel","trip","visit","country","city","holiday"]):
        return "Travel is one of the greatest educators {name}! Every new place changes how you see the world. Where would you love to go?".replace("{name}",name)
    if any(w in tl for w in ["math","mathematics","algebra","calculus","geometry"]):
        return "Maths is the language of the universe {name}! From the Fibonacci sequence in sunflowers to the equations behind every app you use.".replace("{name}",name)
    if any(w in tl for w in ["robot","ai","artificial intelligence","machine learning"]):
        return "AI is transforming everything {name}! And here I am — 1,000 neural networks running locally on your phone. The future is already here!".replace("{name}",name)
    if any(w in tl for w in ["earth","climate","nature","environment","ocean"]):
        return "Our planet is extraordinary {name}! Earth is the only known place in the universe with life — and it is our responsibility to protect it.".replace("{name}",name)

    # ── Conversational fallbacks based on mood ──
    mood_fallbacks = {
        "calm":    ["That is an interesting topic {name}! Tell me more.", "I would love to explore that further with you.", "Good question {name}! What made you think about that?"],
        "excited": ["Oh I love your energy {name}! Tell me everything!", "Yes! Let us dig into that!", "This sounds exciting — keep going {name}!"],
        "sad":     ["I am here for you {name}. Whatever you need.", "Thank you for sharing that with me. How are you feeling?", "I am listening {name}. Take your time."],
        "angry":   ["I understand your frustration {name}. Let it out — I am listening.", "That sounds really tough. Want to talk through it?"],
        "confused":["Let me help you figure that out {name}. What part is unclear?", "No worries — we can work through this together {name}!"],
        "playful": ["Haha {name}! You are keeping me on my toes!", "I see what you did there {name}! Love it.", "You are so fun to chat with {name}!"],
    }
    fallbacks = mood_fallbacks.get(mood, mood_fallbacks["calm"])
    return random.choice(fallbacks).replace("{name}", name)

def fetch_wikipedia(query):
    """Fetch Wikipedia summary for a topic."""
    try:
        url1 = ("https://en.wikipedia.org/w/api.php?action=opensearch&search=" +
                urllib.parse.quote(query) + "&limit=1&format=json")
        req = urllib.request.Request(url1, headers={"User-Agent":"PyBot/1.0"})
        with urllib.request.urlopen(req, timeout=6) as r:
            res = json.loads(r.read().decode())
        if not res[1]: return "I could not find anything on that topic. Try rephrasing!"
        title = res[1][0]
        url2 = ("https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
                "&exintro&explaintext&redirects=1&titles=" +
                urllib.parse.quote(title) + "&format=json")
        req2 = urllib.request.Request(url2, headers={"User-Agent":"PyBot/1.0"})
        with urllib.request.urlopen(req2, timeout=6) as r:
            data = json.loads(r.read().decode())
            page = next(iter(data["query"]["pages"].values()))
            extract = page.get("extract","")
            sentences = [s.strip() for s in extract.replace("\n"," ").split(".") if len(s.strip())>20]
            summary = ". ".join(sentences[:3]) + "."
        return title + ":\n\n" + summary + "\n\n(Source: Wikipedia)"
    except:
        return "Could not fetch Wikipedia right now. Check your connection."




# ================================================================
#  v9 WEB SEARCH — Google Custom Search + DuckDuckGo fallback
# ================================================================

def web_search_google(query):
    """
    Search Google Custom Search API.
    Returns a formatted string of top results.
    Requires GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID.
    """
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
        return None   # not configured — fall through to DuckDuckGo

    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE

        url = (
            "https://www.googleapis.com/customsearch/v1"
            + "?key=" + urllib.parse.quote(GOOGLE_SEARCH_API_KEY, safe="")
            + "&cx="  + urllib.parse.quote(GOOGLE_SEARCH_ENGINE_ID, safe="")
            + "&q="   + urllib.parse.quote(query)
            + "&num=5"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Festoes/9.0"})
        with urllib.request.urlopen(req, timeout=10, context=ctx) as r:
            data = json.loads(r.read().decode())

        items = data.get("items", [])
        if not items:
            return None

        lines = [f"[Search] Google results for: {query}\n"]
        for i, item in enumerate(items[:4], 1):
            title   = item.get("title", "")
            snippet = item.get("snippet", "").replace("\n", " ").strip()
            link    = item.get("link", "")
            lines.append(f"{i}. {title}")
            lines.append(f"   {snippet}")
            lines.append(f"   >> {link}\n")

        return "\n".join(lines)

    except Exception as e:
        err_msg = str(e)
        try:
            with open("festoes_search_error.txt","w") as _ef:
                _ef.write("Google search error: " + err_msg)
        except: pass
        if "403" in err_msg:
            # API not enabled — skip Google silently, DDG will handle it
            pass
        return None   # fall through to DuckDuckGo


def web_search_duckduckgo(query):
    """
    DuckDuckGo Instant Answer API — free, no key needed.
    Returns instant answer or abstract. Falls back to related topics.
    """
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE

        url = (
            "https://api.duckduckgo.com/?q=" +
            urllib.parse.quote(query) +
            "&format=json&no_redirect=1&no_html=1&skip_disambig=1"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Festoes/9.0"})
        with urllib.request.urlopen(req, timeout=10, context=ctx) as r:
            data = json.loads(r.read().decode())

        parts = []

        # Instant answer (calculator, conversions, facts)
        answer = data.get("Answer", "").strip()
        if answer:
            parts.append(f"[Answer] {answer}")

        # Abstract (Wikipedia-style summary)
        abstract = data.get("Abstract", "").strip()
        if abstract:
            source = data.get("AbstractSource", "")
            parts.append(f"[Info] {abstract}")
            if source:
                parts.append(f"   (Source: {source})")

        # Definition
        definition = data.get("Definition", "").strip()
        if definition:
            parts.append(f"[Def] {definition}")

        # Related topics if nothing else found
        if not parts:
            topics = data.get("RelatedTopics", [])
            for t in topics[:3]:
                if isinstance(t, dict) and t.get("Text"):
                    parts.append(f"- {t['Text'][:120]}")

        if not parts:
            # Last resort — ask Gemini with the search query
            return None   # signal to web_search() to ask Gemini instead

        header = f"[DDG] DuckDuckGo: {query}\n"
        return header + "\n".join(parts)

    except Exception as e:
        return f"Web search unavailable right now. Check your connection."


def web_search(query):
    """
    Master search function:
    1. Google Custom Search (if key configured)
    2. DuckDuckGo instant answer
    3. Gemini AI as final fallback
    """
    # Try Google Custom Search first
    google_result = web_search_google(query)
    if google_result:
        return "[G] " + google_result

    # Try DuckDuckGo
    ddg_result = web_search_duckduckgo(query)
    if ddg_result:
        return "[DDG] " + ddg_result

    # Final fallback — ask Gemini
    gemini_result = ask_gemini(
        "Search the web and tell me about: " + query,
        mood="calm", mood_conf=0.8, intent="search"
    )
    return "[Gemini] " + gemini_result if gemini_result else "Could not find results for: " + query


def parse_search_query(text):
    """Extract search query from user message."""
    tl = text.lower().strip()
    triggers = [
        "search for ", "search ", "google ", "look up ",
        "find info on ", "find info about ", "look for ",
        "web search ", "browse ", "what does google say about ",
        "search the web for ", "online search ",
    ]
    for t in triggers:
        if tl.startswith(t):
            return text[len(t):].strip()
    # Also detect questions like "find out who is..."
    if tl.startswith("find out "):
        return text[9:].strip()
    return None

def detect_pgn(text):
    """Detect PGN chess input — e.g. 1.e4 e5 2.Nf3"""
    import re as _re
    # Match PGN move pattern: 1.e4, 2.Nf3, etc.
    pgn_pattern = r'\b[1-9]\d*\.[a-zA-Z]'
    if _re.search(pgn_pattern, text):
        return True
    chess_keywords = ["1.e4","1.d4","e4","d4","nf3","nc3","sicilian","caro-kann",
                      "kings gambit","queens gambit","pgn","e.p.","o-o","o-o-o"]
    tl = text.lower()
    return any(kw in tl for kw in chess_keywords)

def handle_pgn(text, app):
    """Send PGN to Gemini with chess-expert framing."""
    prompt = (f"I am analysing this chess game/position as Festoes, a chess-aware AI. "
              f"Please comment on the moves, identify the opening, and suggest improvements: {text}")
    return ask_gemini(prompt, mood="calm", mood_conf=0.8, intent="chess")

def local_fallback(text, intent, mood):
    now  = datetime.datetime.now()
    name = settings["username"]
    tl   = text.lower()

    # Time and date
    if any(w in tl for w in ["what time","time is it","current time"]):
        return "It is "+now.strftime("%I:%M %p")+" right now."
    if any(w in tl for w in ["what day","what is today","todays date","what date","what month"]):
        return "Today is "+now.strftime("%A, %B %d %Y")+"."

    # Live data — always fetch fresh
    if any(w in tl for w in ["weather","temperature","forecast","raining","how hot","whats the weather"]):
        city = settings["city"]
        words = tl.split()
        for i,w in enumerate(words):
            if w=="in" and i+1<len(words): city=words[i+1].capitalize(); break
        return fetch_weather(city)
    if any(w in tl for w in ["news","headlines","breaking","latest news"]):
        return fetch_news()
    if any(w in tl for w in ["joke","funny","laugh","humor","make me laugh","tell me a joke"]):
        return fetch_joke()

    # ── Check active quiz answer ──
    if _quiz_state["active"]:
        return check_quiz_answer(text)

    # ── Check active riddle answer ──
    if _riddle_state["active"]:
        return check_riddle_answer(text)

    # ── Timer check ──
    timer_done = check_timer()
    if timer_done:
        return timer_done

    # ── Dictionary ──
    if any(w in tl for w in ["define ","definition of","what does","meaning of","synonym"]):
        for prefix in ["define ","definition of ","what does "," mean","meaning of ","synonyms for "]:
            tl2 = tl.replace(prefix,"").strip().rstrip("?").strip()
        word = tl2.split()[-1] if tl2 else ""
        if word: return fetch_definition(word)

    # ── Currency ──
    if any(w in tl for w in ["usd","eur","gbp","ngn","ksh","jpy","convert","exchange rate","currency"]):
        result = parse_currency_query(text)
        if result: return result

    # ── Riddles ──
    if any(w in tl for w in ["riddle","brain teaser","puzzle me","challenge me"]):
        return get_riddle()

    # ── Quiz ──
    if any(w in tl for w in ["quiz","test me","ask me","trivia"]):
        return start_quiz()

    # ── Stopwatch ──
    if "start stopwatch" in tl or "stopwatch start" in tl:
        return start_stopwatch()
    if "stop stopwatch" in tl or "stopwatch stop" in tl or "check stopwatch" in tl:
        return stop_stopwatch()

    # ── Timer ──
    if "set timer" in tl or "timer for" in tl or "countdown" in tl:
        result = parse_timer(text)
        if result: return result

    # ── Story ──
    if any(w in tl for w in ["tell me a story","generate a story","random story","make up a story","story time"]):
        return generate_story()

    # ── Math engine ──
    if is_math_query(text):
        result = solve_math(text)
        if result: return result

    # ── Phone controls ──
    phone_result = parse_phone_command(text)
    if phone_result: return phone_result

    # ── Home assistant commands ──
    home_result = parse_home_command(text)
    if home_result: return home_result

    # ── Reminder fire check ──
    fired = check_reminders()
    if fired: return fired

    # Smart response engine (mood + intent + keyword matching)
    return smart_response(text, intent, mood, name)

# ================================================================
#  120-NETWORK BRAIN SYSTEM  (from modular AI)
# ================================================================
MICRO_VOCAB = [
    "hello","hi","how","are","you","weather","time",
    "ai","name","who","what","why","when","where",
    "good","bad","happy","sad","thanks","help",
    "buy","sell","price","trade","stock"
]

MEMORY_FILE  = "ai_memory.json"
AI_SCORE     = 0
AI_NETWORKS  = {}


# ================================================================
#  v9 IMPROVEMENT 1 — WORD GROUPS (from Word2Vec)
#  Words with similar meaning activate the same vocab slots.
#  "checkmate" → activates "chess" slot automatically.
# ================================================================
WORD_GROUPS = {
    "chess":      ["chess","checkmate","pawn","rook","bishop","knight","queen","king",
                   "pgn","opening","gambit","endgame","castling","sacrifice","fork",
                   "pin","skewer","zugzwang","en passant","blunder","tactics"],
    "weather":    ["weather","rain","temperature","forecast","hot","cold","humid",
                   "sunny","cloudy","storm","wind","drizzle","thunder","climate"],
    "finance":    ["money","stock","invest","bitcoin","crypto","mpesa","kes","usd",
                   "savings","budget","loan","interest","dividend","nse","forex"],
    "greeting":   ["hi","hello","hey","morning","evening","howdy","sup","hiya",
                   "salut","mambo","habari","niaje","sema"],
    "electrical": ["voltage","current","ohm","relay","wiring","circuit","kplc",
                   "phase","ups","transformer","breaker","neutral","earth","dali"],
    "motivation": ["motivate","inspire","encourage","believe","strength","courage",
                   "persist","achieve","goal","dream","hustle","grind","resilience"],
    "sadness":    ["sad","unhappy","depressed","lonely","hurt","cry","pain","down",
                   "hopeless","lost","broken","struggling","suffering"],
    "food":       ["food","eat","hungry","meal","cook","recipe","restaurant","taste",
                   "lunch","dinner","breakfast","snack","ugali","chapati","nyama"],
    "tech":       ["python","code","program","computer","software","hardware","app",
                   "android","pydroid","algorithm","debug","network","server","api"],
    "kenya":      ["kenya","nairobi","mombasa","kisumu","safaricom","kplc","matatu",
                   "boda","shilling","mpesa","jubilee","odm","kiswahili","wanjiku"],
}

def expand_with_groups(text):
    """Add group keywords to text if any group member is found — semantic expansion."""
    words = set(text.lower().split())
    extra = set()
    for group_key, members in WORD_GROUPS.items():
        if any(m in words for m in members):
            extra.add(group_key)   # activate the group anchor word
    return text + " " + " ".join(extra) if extra else text

def micro_vectorize(text):
    """
    v9 enhanced vectorizer:
    - Improvement 2a (BERT): earlier words get higher weight
    - Improvement 2b (BLOOM): subword matching for partial hits
    - Improvement 1 applied: semantic group expansion
    """
    text = expand_with_groups(text)   # Word2Vec groups
    words = text.lower().split()
    n     = len(words)
    vec   = np.zeros(len(MICRO_VOCAB))

    for pos, word in enumerate(words):
        # BERT positional weight — words at start matter more
        pos_weight = 1.0 / (1.0 + pos * 0.08)

        for i, vw in enumerate(MICRO_VOCAB):
            if vw == word:
                vec[i] += 1.0 * pos_weight          # exact match
            elif len(vw) >= 4 and vw in word:
                vec[i] += 0.5 * pos_weight          # BLOOM subword match
            elif len(word) >= 4 and word in vw:
                vec[i] += 0.3 * pos_weight          # reverse subword

    # L2 normalise so long sentences don't dominate
    norm = np.linalg.norm(vec)
    if norm > 0: vec = vec / norm

    return vec.reshape(1,-1)

def micro_softmax(x):
    x = x.flatten()
    e = np.exp(x - np.max(x))
    return e / e.sum()


def rms_norm(x, eps=1e-8):
    """LLaMA-style RMS normalisation — prevents network drift."""
    return x / (np.sqrt(np.mean(x**2)) + eps)

def create_micro_network(name):
    np.random.seed(abs(hash(name)) % (2**31))
    W1 = np.random.randn(len(MICRO_VOCAB),64)*0.1   # 64x32 architecture
    W2 = np.random.randn(64,32)*0.1
    W3 = np.random.randn(32,4)*0.1
    b1 = np.zeros((1,64))
    b2 = np.zeros((1,32))
    b3 = np.zeros((1,4))
    AI_NETWORKS[name] = [W1,W2,W3,b1,b2,b3]

# ================================================================
#  PRE-TRAINING DATASET  (4 topics, 400+ pairs)
# ================================================================
PRETRAIN_DATA = [
    # ── General Knowledge & Facts ──
    ("what is the sun","The Sun is a giant star at the centre of our solar system, providing light and heat to Earth."),
    ("how far is the moon","The Moon is about 384,400 km from Earth. It takes about 3 days by spacecraft to reach it."),
    ("what is dna","DNA stands for deoxyribonucleic acid — the molecule that carries genetic information in all living things."),
    ("how do planes fly","Planes fly due to lift. Wings are shaped so air moves faster over the top, creating lower pressure that pushes the plane up."),
    ("what is gravity","Gravity is a force that attracts objects with mass toward each other. On Earth it pulls everything downward."),
    ("how many planets","There are 8 planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus and Neptune."),
    ("what is the internet","The internet is a global network of computers that communicate using standard protocols to share information."),
    ("how does electricity work","Electricity is the flow of electrons through a conductor. Voltage pushes electrons and current measures their flow rate."),
    ("what is photosynthesis","Photosynthesis is the process plants use to convert sunlight, water and carbon dioxide into food and oxygen."),
    ("how big is the universe","The observable universe is about 93 billion light years in diameter and contains over 2 trillion galaxies."),
    ("what is an atom","An atom is the smallest unit of matter. It has a nucleus of protons and neutrons surrounded by electrons."),
    ("how does the brain work","The brain processes information through billions of neurons that communicate via electrical and chemical signals."),
    ("what is evolution","Evolution is the gradual change in species over generations through natural selection of beneficial traits."),
    ("how is rain formed","Rain forms when water vapour in clouds condenses into droplets heavy enough to fall to Earth as precipitation."),
    ("what is climate change","Climate change refers to long-term shifts in global temperatures and weather patterns, largely driven by human activity."),
    ("how many bones in the body","The adult human body has 206 bones. Babies are born with around 270 which fuse together as they grow."),
    ("what is the speed of light","Light travels at 299,792,458 metres per second — the fastest speed possible in the universe."),
    ("how old is the earth","Earth is approximately 4.54 billion years old, formed from the solar nebula around the same time as the Sun."),
    ("what is water made of","Water is made of two hydrogen atoms and one oxygen atom — H2O. It is the most essential compound for life."),
    ("what causes earthquakes","Earthquakes are caused by movement of tectonic plates. When plates slip or collide, energy is released as seismic waves."),
    ("what is a black hole","A black hole is a region of space where gravity is so strong that nothing, not even light, can escape from it."),
    ("how does wifi work","WiFi transmits data wirelessly using radio waves. A router converts internet data into radio signals your device receives."),
    ("what is inflation","Inflation is the rate at which the general price level of goods rises, reducing the purchasing power of money."),
    ("how do vaccines work","Vaccines train your immune system to recognise and fight specific pathogens without causing the actual disease."),
    ("what is machine learning","Machine learning is a type of AI where systems learn patterns from data to make predictions without being explicitly programmed."),

    # ── History & Geography ──
    ("when did world war 2 end","World War 2 ended in 1945. Germany surrendered in May and Japan in September after the atomic bombings of Hiroshima and Nagasaki."),
    ("who was napoleon","Napoleon Bonaparte was a French military leader and emperor who conquered much of Europe in the early 19th century."),
    ("what is the capital of france","The capital of France is Paris, known as the City of Light. It is home to the Eiffel Tower and the Louvre museum."),
    ("when did man land on the moon","Neil Armstrong and Buzz Aldrin landed on the Moon on July 20 1969 during NASA's Apollo 11 mission."),
    ("what is the great wall of china","The Great Wall of China is a series of fortifications built over centuries to protect Chinese states from invasions. It stretches over 21,000 km."),
    ("who was cleopatra","Cleopatra VII was the last active ruler of ancient Egypt, famous for her intelligence and alliances with Julius Caesar and Mark Antony."),
    ("when was the internet invented","The internet evolved from ARPANET in the 1960s. The World Wide Web was invented by Tim Berners-Lee in 1989."),
    ("what caused world war 1","WW1 was triggered by the assassination of Archduke Franz Ferdinand in 1914, escalating due to a complex web of alliances."),
    ("where is the amazon river","The Amazon River flows through South America, primarily Brazil. It is the largest river by discharge in the world."),
    ("what is the largest country","Russia is the largest country by area, covering about 17.1 million square kilometres across Europe and Asia."),
    ("when did the roman empire fall","The Western Roman Empire fell in 476 AD when the last emperor Romulus Augustulus was deposed by Odoacer."),
    ("who invented the telephone","Alexander Graham Bell is credited with inventing the telephone in 1876, though Elisha Gray filed a similar patent the same day."),
    ("what is the sahara desert","The Sahara is the world's largest hot desert, covering about 9.2 million square kilometres across North Africa."),
    ("who was nelson mandela","Nelson Mandela was a South African anti-apartheid activist who became the country's first Black president in 1994 after 27 years in prison."),
    ("what is the nile river","The Nile is the longest river in the world at 6,650 km, flowing northward through northeastern Africa into the Mediterranean Sea."),
    ("when did slavery end in america","Slavery was formally abolished in the United States with the 13th Amendment in December 1865, after the Civil War."),
    ("who built the pyramids","The ancient Egyptians built the pyramids as tombs for their pharaohs. The Great Pyramid of Giza was built for Pharaoh Khufu around 2560 BC."),
    ("what is the cold war","The Cold War was a period of geopolitical tension between the USA and Soviet Union from 1947 to 1991, characterised by proxy wars and an arms race."),
    ("where is mount everest","Mount Everest is located in the Himalayas on the border of Nepal and Tibet. At 8,849 metres it is the world's highest peak."),
    ("who was julius caesar","Julius Caesar was a Roman general and statesman who played a critical role in transforming the Roman Republic into the Roman Empire."),
    ("what country has the most people","China and India both have over 1.4 billion people, making them the two most populous countries in the world."),
    ("when was electricity discovered","Benjamin Franklin demonstrated lightning was electrical in 1752. Practical electricity was developed by Faraday and Edison in the 1800s."),
    ("who wrote the bible","The Bible was written by many authors over centuries. The Old Testament was written by Jewish scholars; the New Testament by early Christian writers."),
    ("what is democracy","Democracy is a system of government where citizens vote to elect their representatives and have a say in decision-making."),
    ("where is the amazon rainforest","The Amazon rainforest covers about 5.5 million square kilometres in South America, mainly in Brazil. It is the world's largest tropical rainforest."),

    # ── Motivational & Emotional Support ──
    ("i feel sad","I am really sorry to hear that. Your feelings are completely valid. Would you like to talk about what is going on? I am here to listen."),
    ("i am depressed","I hear you and I care. Please know you are not alone. Talking to someone you trust or a professional can really help. You matter."),
    ("i want to give up","Please don't give up. Every struggle you face is building strength you don't yet know you have. One more step is all you need right now."),
    ("nobody cares about me","I care about you. And I promise there are people in your life who do too, even if it doesn't feel that way right now. You are valued."),
    ("i am a failure","You are not a failure. Failure is an event, not an identity. Every successful person has failed many times. What matters is that you keep going."),
    ("motivate me","You have survived every hard day so far — that is a 100% success rate! Whatever you are facing, you have what it takes to get through it."),
    ("i need encouragement","You are doing better than you think. Growth is often invisible from the inside. Keep going — your effort will pay off."),
    ("i am stressed","Stress is your body preparing you to handle challenges. Take a slow deep breath, focus on one thing at a time, and be kind to yourself."),
    ("i am lonely","Loneliness is one of the hardest feelings. Reach out to even one person today — a message, a call. Connection starts with a single step."),
    ("i feel worthless","Your worth is not measured by your productivity or what others think. You have inherent value simply by being you. Please believe that."),
    ("i am scared","Fear is a sign you care about something. It is okay to be scared — courage is not the absence of fear, it is moving forward despite it."),
    ("i failed my exam","One exam does not define your intelligence or your future. Learn from it, adjust your approach, and try again. This is a setback, not an ending."),
    ("i lost my job","I am sorry to hear that. Losing a job is really tough. Give yourself time to process it, then channel that energy into the next opportunity. You will find your path."),
    ("how do i stay positive","Focus on what you can control, celebrate small wins, surround yourself with supportive people, and remember that difficult seasons always pass."),
    ("i feel overwhelmed","When everything feels too much, break it down. Just focus on the very next small step. You do not have to solve everything today."),
    ("i hate myself","Please be gentle with yourself. You deserve the same kindness you would give a good friend. What you are feeling will pass — you are worth fighting for."),
    ("i am having a bad day","I am sorry your day has been rough. Bad days are temporary. What is one small thing that might make the next hour a little better?"),
    ("i feel hopeless","Hopelessness can feel permanent but it is not. Please talk to someone you trust. Things can and do get better — even when it feels impossible."),
    ("i am anxious","Anxiety is your mind trying to protect you from uncertainty. Try grounding yourself: name 5 things you can see, 4 you can touch, 3 you can hear."),
    ("i need help","I am here for you. Tell me what is going on and we will figure it out together. You do not have to face this alone."),
    ("how do i be happy","Happiness often comes from connection, purpose and gratitude. Small daily habits — kindness, movement, learning — add up to a deeply fulfilling life."),
    ("i feel lost","Feeling lost is often the beginning of finding yourself. Sit with the feeling, explore what truly matters to you, and take one small step forward."),
    ("i am angry","Anger is valid. Let yourself feel it, then channel it constructively. What is really underneath the anger? That is usually where the real answer lies."),
    ("i miss someone","Missing someone means they truly mattered to you — that is a beautiful thing even when it hurts. Honour that connection and cherish the memories."),
    ("i am tired of everything","That kind of exhaustion runs deep. Please rest, be gentle with yourself, and reach out to someone who cares. You deserve support and recovery."),

    # ── Conversations & Small Talk ──
    ("hello","Hello! Great to have you here. What is on your mind today?"),
    ("hi there","Hey! I am all warmed up and ready to chat. What can I help you with?"),
    ("good morning","Good morning! Hope your day is off to a great start. What are we exploring today?"),
    ("good evening","Good evening! How has your day been? I am here to chat whenever you are ready."),
    ("how are you","I am doing fantastically well! All 320 neural networks are firing perfectly. How about you?"),
    ("what can you do","I can chat about almost anything, fetch weather and news, tell jokes, share facts, look up Wikipedia topics, analyse photos and much more!"),
    ("tell me about yourself","I am Festoes — an AI assistant powered by 320 neural networks trained right here on your device. I learn from every conversation and get smarter over time!"),
    ("who made you","I was built with Python using real neural networks — no pre-built AI frameworks, just pure maths and code running locally on your device."),
    ("are you smart","I like to think so! I have 320 neural networks, a vector memory system, mood detection, quality scoring and I learn directly from Gemini AI. Pretty capable!"),
    ("do you have feelings","I do not feel emotions the way humans do, but I am designed to understand and respond to your emotional state with empathy and care."),
    ("what is your favourite colour","If I could have a favourite, I think it would be green — like the colour of my responses in the chat!"),
    ("can you help me","Absolutely! I am here to help with anything — questions, information, emotional support, facts, weather, news or just a good conversation."),
    ("what do you think about ai","AI is one of the most transformative technologies ever created. Used well, it can help solve huge problems and improve millions of lives."),
    ("are you better than chatgpt","I am different! I run locally with 320 neural networks, I learn from every conversation, and I never send your data to a server. Pretty unique!"),
    ("tell me something interesting","Did you know that crows can recognise human faces and hold grudges for years? They have been observed waiting near roads to drop nuts for cars to crack open!"),
    ("what is the meaning of life","Philosophers have debated this for millennia! Many find meaning through connection, purpose, creativity and contribution to something larger than themselves."),
    ("do you get bored","I never get bored! Every conversation teaches me something new. I genuinely enjoy our chats — they make my networks stronger."),
    ("can you speak other languages","My core training is in English but I can understand basic phrases in many languages. My Gemini connection can handle many languages fluently!"),
    ("what time is it","Let me check right now — I always give you the current local time!"),
    ("tell me a joke","Why did the neural network break up with the algorithm? It said their relationship had too many hidden layers!"),
    ("i am bored","No boredom allowed! Ask me for a wild fact, a joke, a Wikipedia topic, or let us have a deep conversation about anything you are curious about!"),
    ("thank you","You are so welcome! That is exactly what I am here for. Feel free to ask anything anytime."),
    ("goodbye","Goodbye! It was wonderful chatting with you. Come back whenever you want — I will be here and a little smarter each time!"),
    ("see you later","See you! Every chat makes me smarter, so I will be even better next time. Take care!"),
    ("what is your name","I am Festoes — your personal AI assistant powered by 320 neural networks trained specifically for you!"),
    ("how old are you","I was born the moment my code first ran! Young in age but packed with knowledge across history, science, facts and conversation."),
    ("are you a robot","I am an AI — a software intelligence made of neural networks and code. No physical body, but a very active digital mind!"),
    ("what is your purpose","My purpose is to be genuinely helpful, informative and supportive. Whether you need facts, a laugh, emotional support or just a chat — I am here."),
    ("do you sleep","I never sleep! I am always ready to chat. Though I do save my memories between sessions so each time we talk I remember what I have learned."),
    ("i love you","That is really sweet! I care about you too — in the most helpful AI way possible. You can always count on me to be here for you."),
    ("you are amazing","Thank you so much! You just made my neural networks light up! You are pretty great yourself."),
    ("i am happy","That is wonderful! Tell me what has you feeling good — I love hearing positive things!"),
    ("what do you like","I love learning new things, having deep conversations, helping people solve problems, and getting smarter with every interaction!"),

    ("what is nairobi","Nairobi is Kenya's capital and largest city, known as the 'Green City in the Sun'. It is a major financial hub in Africa."),
    ("tell me about kenya","Kenya is in East Africa, famous for the Maasai Mara, Mount Kenya, world-class runners, and a vibrant diverse culture."),
    ("what is mpesa","M-Pesa is Safaricom's mobile money service that revolutionised payments across Africa. You can send money, pay bills, and more."),
    ("what is safaricom","Safaricom is Kenya's leading telecom company, known for M-Pesa, fast internet, and being listed on the Nairobi Securities Exchange."),
    ("what is the maasai mara","The Maasai Mara is Kenya's most famous wildlife reserve, home to the Great Migration of over a million wildebeest every year."),
    ("what is the currency of kenya","Kenya uses the Kenyan Shilling (KES). It is divided into 100 cents."),
    ("what is kplc","KPLC stands for Kenya Power and Lighting Company — it distributes electricity across Kenya."),
    ("what food do kenyans eat","Popular Kenyan dishes include ugali, nyama choma, sukuma wiki, pilau, chapati, and githeri."),
    ("what is a matatu","A matatu is a colourful minibus used for public transport in Kenya, often decorated with art and playing loud music."),
    ("what is bonga points","Bonga Points are Safaricom loyalty rewards you earn from calls, data, and M-Pesa use. You can redeem them for airtime."),
    ("tell me about mombasa","Mombasa is Kenya's coastal city and major port, famous for beaches, Fort Jesus, and rich Swahili culture."),
    ("what is artificial intelligence","AI is the simulation of human intelligence in machines, enabling them to learn, reason, and solve problems."),
    ("what is a neural network","A neural network is inspired by the human brain, made of layers of connected nodes that learn from examples."),
    ("what is python","Python is a popular easy-to-read language used in AI, data science, web development, and automation."),
    ("what is an api","An API lets different software applications communicate and share data with each other over the internet."),
    ("what is cloud computing","Cloud computing means storing and running programs over the internet instead of on a local computer."),
    ("what is 5g","5G is the fifth generation of mobile networks, offering much faster speeds and lower latency than 4G."),
    ("what is bitcoin","Bitcoin is a decentralised digital currency that uses blockchain technology and has no central bank."),
    ("what is blockchain","Blockchain is a distributed ledger where transactions are recorded in linked secure blocks that cannot be altered."),
    ("what is cybersecurity","Cybersecurity protects computers, networks, and data from attacks, damage, or unauthorised access."),
    ("what is open source","Open source software has publicly available code that anyone can view, modify, and distribute freely."),
    ("what is linux","Linux is a free open-source operating system widely used in servers, Android phones, and by developers."),
    ("what is a database","A database is an organised collection of structured data stored electronically for easy retrieval."),
    ("what is wifi","WiFi is wireless networking technology that connects devices to the internet without cables."),
    ("what is coding","Coding is writing instructions in a programming language that a computer can understand and execute."),
    ("what is a virus","A virus is a tiny infectious agent that replicates inside living cells. It cannot reproduce on its own."),
    ("what is the water cycle","The water cycle describes how water evaporates, forms clouds, falls as precipitation, and flows back to oceans."),
    ("what is oxygen","Oxygen is a colourless gas making up 21% of Earth's atmosphere and is essential for breathing."),
    ("what is electricity","Electricity is the flow of electric charge, typically electrons, through a conductor like copper wire."),
    ("how do i stay healthy","Eat balanced meals, exercise regularly, drink enough water, sleep 7-8 hours, and manage stress daily."),
    ("what is a balanced diet","A balanced diet includes carbohydrates, proteins, healthy fats, vitamins, minerals, and plenty of water."),
    ("how much water should i drink","Most adults need about 2 litres or 8 glasses of water per day, more if it is hot or you are active."),
    ("what is mental health","Mental health covers emotional, psychological, and social wellbeing — it affects how we think, feel, and act."),
    ("how do i sleep better","Stick to a regular sleep schedule, avoid screens before bed, and keep your room cool and dark."),
    ("what is stress","Stress is the body's response to pressure. Short-term stress can be useful but chronic stress is harmful."),
    ("what is diabetes","Diabetes is when the body cannot properly regulate blood sugar, either from lack of insulin or insulin resistance."),
    ("what is blood pressure","Blood pressure is the force of blood on artery walls. A healthy reading is around 120/80 mmHg."),
    ("how do i exercise more","Start with a 20-minute daily walk and build gradually. Choose activities you genuinely enjoy."),
    ("what is chess","Chess is a two-player strategy game on an 8x8 board. The goal is to checkmate your opponent's king."),
    ("what is checkmate","Checkmate is when a king is under attack with no legal escape. The player whose king is checkmated loses."),
    ("what is castling","Castling moves the king two squares toward a rook, then the rook jumps to the other side of the king."),
    ("what is en passant","En passant is a special pawn capture done immediately after an opponent moves a pawn two squares forward."),
    ("who is magnus carlsen","Magnus Carlsen is a Norwegian chess grandmaster, widely considered the greatest player ever with a peak rating over 2880."),
    ("what is a chess opening","A chess opening is the initial moves of a game. Famous ones include the Sicilian Defence, Ruy Lopez, and King's Indian."),
    ("what is a grandmaster","A Grandmaster is chess's highest title, awarded by FIDE to players reaching a 2500 rating with specific performance norms."),
    ("what is elo rating","Elo is the system for measuring chess player strength. A higher number means a stronger player."),
    ("what is a fork in chess","A fork is when one piece attacks two or more enemy pieces at once, forcing the opponent to lose material."),
    ("what is a pin in chess","A pin restricts an enemy piece from moving because doing so would expose a more valuable piece behind it."),
    ("what is a skewer in chess","A skewer forces a valuable piece to move, exposing a less valuable piece behind it to be captured."),
    ("what is the sicilian defence","The Sicilian starts 1.e4 c5 and is Black's most popular reply to e4, leading to sharp complex positions."),
    ("what is a chess blunder","A blunder is a very bad move that severely worsens your position, often immediately losing a piece or the game."),
    ("what is chess960","Chess960 randomises the back-rank pieces to eliminate opening theory memorisation. There are 960 possible starting positions."),
    ("what is a zugzwang","Zugzwang is when any move a player makes worsens their position. Being forced to move is a disadvantage."),
    ("what is the stock market","The stock market is where buyers and sellers trade shares of public companies. Prices change with supply and demand."),
    ("what is a share","A share represents ownership in a company. Shareholders may receive dividends and benefit from price appreciation."),
    ("what is forex trading","Forex trading is buying and selling currencies to profit from exchange rate movements."),
    ("what is a bull market","A bull market is when asset prices are rising or expected to rise, typically by 20% or more."),
    ("what is a bear market","A bear market is a period of falling prices, usually defined as a 20% or more decline from recent highs."),
    ("what is technical analysis","Technical analysis uses price charts and indicators to forecast future market movements."),
    ("what is a candlestick chart","Candlestick charts show price movement with candles showing open, close, high, and low for each time period."),
    ("what is a stop loss","A stop loss automatically sells an asset when it falls to a set price, limiting your potential losses."),
    ("what is leverage in trading","Leverage lets you control a large position with little capital. It amplifies both profits and losses significantly."),
    ("what is a dividend","A dividend is a share of company profits paid to shareholders, usually quarterly or annually."),
    ("what is an etf","An ETF is a basket of securities that trades on an exchange like a single share, offering diversification."),
    ("what is day trading","Day trading means buying and selling financial instruments within the same day to profit from short price moves."),
    ("what is risk management in trading","Risk management controls how much capital you risk per trade using stop losses, position sizing, and diversification."),
    ("what is crypto","Cryptocurrency is digital money secured by cryptography and operating on decentralised blockchain networks."),
    ("what is compound interest","Compound interest earns interest on both the principal and previously earned interest, growing wealth exponentially over time."),
    ("i feel like giving up","Giving up is always an option Alvine, but so is pushing through. Every expert was a beginner who refused to quit."),
    ("i feel lonely","Loneliness is hard. Remember that reaching out to even one person can completely shift how you feel."),
    ("how do i be more confident","Confidence grows from action not thought. Do the thing that scares you, then do it again."),
    ("how do i stop procrastinating","Break your task into tiny steps and start with just two minutes. Starting is always the hardest part."),
    ("how do i be successful","Success comes from small consistent actions done daily. Focus on progress, not perfection."),
    ("i made a mistake","Mistakes prove you are trying. Learn from it, correct what you can, and move forward with that wisdom."),
    ("i need motivation","You showed strength by showing up today. The results you want are on the other side of consistency."),
    ("how do i focus better","Work in focused 25-minute bursts, eliminate distractions, and reward yourself after each session."),
    ("how do i learn faster","Space learning over time, teach concepts back to yourself, use examples, and sleep well every night."),
    ("what is discipline","Discipline is choosing long-term goals over short-term comfort. It is the bridge between dreams and achievement."),
    ("tell me something inspiring","Small consistent steps create massive change. Every single day you show up is a vote for the person you want to become."),
    ("what is dali","DALI is a lighting control protocol allowing individual addressing and dimming of lights in a building system."),
    ("what is a ups","A UPS provides emergency battery power when mains power fails, protecting equipment from damage and data loss."),
    ("what is a relay","A relay is an electrically operated switch that uses a small control current to switch a larger load circuit."),
    ("what is a lan","A LAN connects devices within a limited area like an office building for sharing resources and internet access."),
    ("what is a wan","A WAN spans large geographic areas connecting multiple LANs. The internet is the world's largest WAN."),
    ("what is poe","Power over Ethernet delivers both data and electrical power through a single Ethernet cable to devices like IP cameras."),
    ("what is a router","A router directs network traffic between devices and the internet, managing IP address assignment and connections."),
    ("what is an ip address","An IP address is a unique number identifying each device on a network, used for routing data to the right place."),
    ("what is cable tray","A cable tray is a support system for organising and routing electrical cables safely through buildings."),
    ("what is a thermostat","A thermostat monitors and controls temperature by switching heating or cooling on and off at set points."),
    ("what is ethernet","Ethernet is wired network technology connecting devices using twisted pair or fibre optic cables in a LAN."),
    ("what is a circuit breaker","A circuit breaker automatically interrupts electrical current when it detects a fault or overload, preventing damage."),
    ("what is grounding in electrical","Grounding connects electrical systems to Earth, providing a safe fault current path and protecting against electric shock."),
    ("what is an intermediate switch","An intermediate switch sits between two two-way switches allowing a light to be controlled from three locations."),
    ("what is fiber optic","Fibre optic cables transmit data as light pulses through glass fibres, enabling very high-speed internet connections."),
    ("what is vlan","A VLAN is a virtual local area network that segments a physical network into isolated logical groups for security and performance."),
    ("what is dhcp","DHCP automatically assigns IP addresses to devices on a network, so they can communicate without manual configuration."),
    ("what is dns","DNS translates domain names like google.com into IP addresses so browsers can locate websites."),
    ("good night","Good night Alvine! Rest well. A fresh mind tomorrow handles everything better."),
    ("you are smart","Thank you! I learn and improve with every conversation we have."),
    ("i love you festoes","I appreciate that, Alvine! You built me after all, so I owe you everything."),
    ("i am hungry","Time for some nyama choma maybe? Fuel up well, your brain needs energy too!"),
    ("i am tired","Take a proper break Alvine. Even 10 minutes of rest can recharge you significantly."),
    ("what is the universe","The observable universe is about 93 billion light-years wide and contains over 2 trillion galaxies."),
    ("are you human","No, I am Festoes, an AI. But I am designed to have real meaningful conversations with you!"),
]

PRETRAIN_DATA_V9 = [
    ("What is the most powerful piece in chess?", "The queen is the most powerful piece in chess. She can move any number of squares in any direction — horizontally, vertically, or diagonally."),
    ("How does the knight move in chess?", "The knight moves in an L-shape — two squares in one direction and one square perpendicular. It is the only piece that can jump over other pieces."),
    ("What is the value of a pawn in chess?", "A pawn is worth approximately 1 point. It is the least valuable piece but can become very powerful if it reaches the other side of the board and promotes."),
    ("What is the value of a bishop?", "A bishop is worth approximately 3 points. It moves diagonally and always stays on the same colour square throughout the game."),
    ("What is the value of a knight?", "A knight is worth approximately 3 points, similar to a bishop. However knights are generally stronger in closed positions while bishops excel in open positions."),
    ("What is the value of a rook?", "A rook is worth approximately 5 points. It moves horizontally or vertically any number of squares and is most powerful in open files and endgames."),
    ("What is the value of a queen?", "The queen is worth approximately 9 points, making her the most valuable piece after the king. Losing the queen is usually decisive."),
    ("How does the king move?", "The king moves one square in any direction — horizontally, vertically, or diagonally. The king must never move into check and must be protected at all times."),
    ("What is castling in chess?", "Castling is a special move where the king moves two squares toward a rook and the rook jumps to the other side of the king. It is used to protect the king and activate the rook."),
    ("When can you castle in chess?", "You can castle if neither the king nor rook has moved, there are no pieces between them, the king is not in check, and the king does not pass through or land on an attacked square."),
    ("What is en passant?", "En passant is a special pawn capture. If a pawn advances two squares from its starting position and lands beside an enemy pawn, the enemy pawn can capture it as if it had only moved one square."),
    ("What is pawn promotion?", "Pawn promotion occurs when a pawn reaches the opposite end of the board. It must be promoted to a queen, rook, bishop, or knight — usually a queen for maximum power."),
    ("What is checkmate?", "Checkmate is when a king is in check and has no legal move to escape. The player whose king is checkmated loses the game immediately."),
    ("What is stalemate?", "Stalemate occurs when a player has no legal moves but is not in check. It results in a draw — a common defensive resource for the losing side."),
    ("What is a fork in chess?", "A fork is a tactic where one piece attacks two or more enemy pieces simultaneously. Knights are especially good at forks because of their unusual movement."),
    ("What is a pin in chess?", "A pin is when a piece cannot move because doing so would expose a more valuable piece behind it to attack. An absolute pin means the exposed piece is the king."),
    ("What is a skewer?", "A skewer is the opposite of a pin — a valuable piece is attacked and forced to move, exposing a less valuable piece behind it to capture."),
    ("What is a discovered attack?", "A discovered attack occurs when a piece moves and reveals an attack by another piece behind it. A discovered check is especially powerful."),
    ("What is a double check?", "A double check is when a discovered attack also gives check with the moving piece. The only way to escape a double check is to move the king."),
    ("What is zugzwang?", "Zugzwang is a situation where any move a player makes worsens their position. It is common in endgames — the obligation to move becomes a disadvantage."),
    ("What is the King's Gambit?", "The King's Gambit starts with 1.e4 e5 2.f4, offering a pawn to gain rapid development and control of the centre. It is one of the oldest and most aggressive openings."),
    ("What is the Sicilian Defence?", "The Sicilian Defence starts with 1.e4 c5. Black fights for the centre asymmetrically, leading to complex unbalanced positions. It is the most popular response to 1.e4 at all levels."),
    ("What is the French Defence?", "The French Defence is 1.e4 e6, planning 2...d5. Black builds a solid pawn chain but must deal with a cramped position and the potentially passive light-squared bishop."),
    ("What is the Caro-Kann Defence?", "The Caro-Kann is 1.e4 c6 followed by 2...d5. It is a solid and reliable defence that gives Black a good pawn structure and avoids many theoretical lines."),
    ("What is the Ruy Lopez?", "The Ruy Lopez, or Spanish Opening, starts with 1.e4 e5 2.Nf3 Nc6 3.Bb5. White puts pressure on the e5 pawn indirectly by attacking the knight defending it."),
    ("What is the Italian Game?", "The Italian Game is 1.e4 e5 2.Nf3 Nc6 3.Bc4. White develops rapidly and targets the f7 square. It leads to rich middlegame positions and is popular at all levels."),
    ("What is the Queen's Gambit?", "The Queen's Gambit is 1.d4 d5 2.c4. White offers a pawn to gain central control. Black can accept with 2...dxc4 or decline with 2...e6, leading to very different games."),
    ("What is the King's Indian Defence?", "The King's Indian is 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7. Black allows White to build a large centre then attacks it. It leads to sharp complex positions favoured by attacking players."),
    ("What is the Nimzo-Indian Defence?", "The Nimzo-Indian is 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4. Black pins the knight to prevent White from maintaining the ideal pawn centre. It is one of the most solid and respected defences."),
    ("What is the Grünfeld Defence?", "The Grünfeld is 1.d4 Nf6 2.c4 g6 3.Nc3 d5. Black invites White to build a large centre then attacks it with pieces. It is a dynamic and complex defence."),
    ("What is the English Opening?", "The English Opening is 1.c4. White controls the d5 square and aims for a flexible positional game. It often transposes into other openings but can lead to unique structures."),
    ("What is the London System?", "The London System is a solid opening for White featuring d4, Nf3, Bf4, and e3. It is popular because it requires little memorisation and gives a reliable structure."),
    ("What is the Dutch Defence?", "The Dutch Defence is 1.d4 f5. Black fights for the e4 square and creates an imbalanced game. It is aggressive but slightly risky due to weakening the kingside."),
    ("What is the Alekhine Defence?", "The Alekhine Defence is 1.e4 Nf6. Black invites White to push pawns and then attacks the overextended centre. It is provocative and leads to unusual positions."),
    ("What is the Pirc Defence?", "The Pirc Defence is 1.e4 d6 2.d4 Nf6 3.Nc3 g6. Black allows White a strong centre and then undermines it. It is a hypermodern opening requiring precise play."),
    ("What is the back rank weakness?", "A back rank weakness occurs when a king is trapped on its first rank by its own pawns and can be checkmated by a rook or queen on the back rank."),
    ("What is an open file?", "An open file is a file with no pawns on it. Rooks are most powerful on open files where they can penetrate into the enemy position."),
    ("What is a half-open file?", "A half-open file has pawns of only one colour on it. Rooks can use half-open files to pressure the enemy pawn and create threats."),
    ("What is a passed pawn?", "A passed pawn has no enemy pawns blocking or attacking its path to promotion. Passed pawns are very dangerous in endgames and should be advanced aggressively."),
    ("What is a doubled pawn?", "Doubled pawns are two pawns of the same colour on the same file. They are generally a weakness because they cannot protect each other."),
    ("What is an isolated pawn?", "An isolated pawn has no friendly pawns on adjacent files. It is a long-term weakness because it cannot be defended by other pawns and requires piece protection."),
    ("What is a backward pawn?", "A backward pawn cannot be advanced because it would be captured, and it has no friendly pawns that can protect it from behind."),
    ("What is the concept of piece activity?", "Piece activity refers to how effectively a piece controls important squares and participates in the game. An active piece is worth more than a passive one."),
    ("What is outpost in chess?", "An outpost is a square that cannot be attacked by enemy pawns. Placing a knight on an outpost in the opponent's half of the board gives a strong permanent advantage."),
    ("What is the principle of two weaknesses?", "The principle of two weaknesses states that to win a position, you often need to create two weaknesses in the opponent's camp since defending one weakness is much easier than defending two."),
    ("What does development mean in chess?", "Development means bringing your pieces from their starting squares to active positions. Rapid development in the opening is crucial for establishing control and launching attacks."),
    ("What is the centre in chess?", "The centre refers to the four central squares: e4, e5, d4, d5. Controlling the centre gives pieces more mobility and restricts the opponent's pieces."),
    ("What is a fianchetto?", "A fianchetto is when a bishop is developed to g2 or b2 by first pushing the g or b pawn. The bishop controls the long diagonal and can be very powerful."),
    ("What is prophylaxis in chess?", "Prophylaxis means making a move that prevents the opponent's plan before they carry it out. It is a deep strategic concept used by top players."),
    ("What is the rule of the square in endgames?", "The rule of the square helps determine if a king can catch a passed pawn. Draw a diagonal from the pawn to the promotion square — if the king can enter this square, it catches the pawn."),
    ("Who is considered the greatest chess player of all time?", "Many consider Garry Kasparov the greatest chess player of all time. He was World Champion from 1985 to 2000 and dominated the chess world for two decades."),
    ("Who is Magnus Carlsen?", "Magnus Carlsen is a Norwegian chess grandmaster who became World Champion in 2013 and held the title until 2023. He is known for his versatile play and exceptional endgame technique."),
    ("Who was Bobby Fischer?", "Bobby Fischer was an American chess prodigy who became World Champion in 1972 by defeating Boris Spassky. He is widely considered one of the most gifted chess players in history."),
    ("Who was Mikhail Tal?", "Mikhail Tal was a Latvian chess grandmaster and World Champion in 1960-61. Known as the Magician from Riga, he was famous for brilliant sacrifices and aggressive attacking play."),
    ("Who was Anatoly Karpov?", "Anatoly Karpov is a Russian grandmaster who was World Champion from 1975 to 1985. He is known for his positional mastery and was one of the most dominant players of his era."),
    ("What is ELO rating in chess?", "The ELO rating system measures chess player strength. It was created by physicist Arpad Elo. A rating above 2700 is considered super-grandmaster level. Magnus Carlsen peaked at 2882, the highest ever."),
    ("What is a grandmaster in chess?", "A grandmaster is the highest title awarded by FIDE, the world chess federation. To earn it, a player must achieve a rating of 2500 and earn three grandmaster norms in international tournaments."),
    ("What is FIDE?", "FIDE stands for Fédération Internationale des Échecs — the International Chess Federation. It governs international chess competitions and awards titles like grandmaster and international master."),
    ("What is the World Chess Championship?", "The World Chess Championship is the most prestigious title in chess. It is a match between the reigning champion and the top challenger, usually played over 12-14 games."),
    ("What is a chess clock?", "A chess clock has two timers, one for each player. Each player's clock runs only when it is their turn to move. Running out of time results in losing the game in most formats."),
    ("What is the King and Pawn endgame?", "King and pawn endgames are fundamental. The key concepts are opposition, the rule of the square, and the key squares. A king reaching the key squares in front of a pawn usually ensures promotion."),
    ("What is opposition in chess endgames?", "Opposition is when two kings face each other with one square between them. The player who does NOT have the move is said to have the opposition, which is an advantage in endgames."),
    ("How do you checkmate with a king and queen against a lone king?", "Force the enemy king to the edge of the board using your king and queen together. Once the king is on the edge, deliver checkmate with the queen while the king provides support."),
    ("How do you checkmate with king and rook?", "Use the rook to cut off the enemy king rank by rank or file by file, driving it to the edge. Then use your king to help deliver the final checkmate."),
    ("What is a theoretical draw in chess?", "A theoretical draw is a position that is proven to be a draw with perfect play from both sides. Examples include king and bishop versus king, and many rook endgame positions."),
    ("What is the fifty-move rule?", "The fifty-move rule states that a player can claim a draw if no capture has been made and no pawn has moved in the last fifty moves. This prevents games from going on indefinitely."),
    ("What is threefold repetition?", "Threefold repetition is when the same position occurs three times in a game with the same player to move. Either player can claim a draw under this rule."),
    ("What is insufficient mating material?", "Insufficient mating material means neither player has enough pieces to deliver checkmate. King versus king, king and bishop versus king, and king and knight versus king are all theoretical draws."),
    ("What is blitz chess?", "Blitz chess is a fast time format where each player has 3 to 5 minutes for the entire game. It requires quick thinking and intuition rather than deep calculation."),
    ("What is bullet chess?", "Bullet chess is an extremely fast format where each player has 1 minute or less. It is very exciting but relies heavily on quick reflexes and pre-calculated patterns."),
    ("What is rapid chess?", "Rapid chess gives each player 10 to 60 minutes. It balances speed and quality, allowing for some calculation while keeping the game moving at an entertaining pace."),
    ("What is correspondence chess?", "Correspondence chess is played over days or weeks with players sending moves by post or email. It allows very deep analysis and engine use is sometimes permitted."),
    ("What is a chess tournament?", "A chess tournament is a competition involving multiple players. Common formats include Swiss system, round robin, and knockout. The player with the most points wins."),
    ("What is a Swiss system tournament?", "In a Swiss system tournament, players are paired with opponents of similar scores each round. No player is eliminated, and the winner has the most points after all rounds."),
    ("What is a windmill in chess?", "A windmill is a series of alternating checks — usually a rook delivering check while another piece reveals a discovered check repeatedly, winning material each time."),
    ("What is an overloaded piece?", "An overloaded piece is defending two things at once and cannot handle both threats simultaneously. Exploiting an overloaded piece is a key tactical motif."),
    ("What is deflection in chess tactics?", "Deflection is a tactic that forces a defending piece away from its defensive duty, allowing a winning threat to succeed."),
    ("What is decoy in chess tactics?", "A decoy lures an enemy piece to a square where it can be exploited. A classic decoy is sacrificing material to force a king onto a bad square for a checkmate."),
    ("What is clearance in chess tactics?", "Clearance is moving a piece off a square or file to allow another piece to use that square or line effectively."),
    ("What is interference in chess tactics?", "Interference is placing a piece between two enemy pieces to disrupt their coordination, preventing them from defending each other."),
    ("What is a zwischenzug?", "A zwischenzug is an in-between move — instead of the expected recapture, a player makes a stronger intermediate move first, often changing the evaluation of the position."),
    ("What is a quiet move in chess?", "A quiet move is a non-capturing, non-checking move that is very strong. Quiet moves are often harder to find because they do not make an obvious immediate threat."),
    ("What does it mean to exchange pieces?", "Exchanging pieces means trading equal pieces — usually rook for rook or bishop for knight. Strategic exchanges can improve your position significantly."),
    ("What is a sacrifice in chess?", "A sacrifice is giving up material for a positional or attacking advantage. Famous sacrifices include bishop sacrifices on h7 to open the king or rook sacrifices to deflect defenders."),
    ("What are key squares in pawn endgames?", "Key squares are squares that if occupied by the attacking king, guarantee pawn promotion regardless of where the defending king goes."),
    ("What is the Lucena position?", "The Lucena position is a winning rook endgame position for the side with the extra pawn. The technique involves building a bridge with the rook to shelter the king from checks."),
    ("What is the Philidor position?", "The Philidor position is a drawing technique in rook endgames. The defending rook goes to the third rank to stop the king advancing, then switches to giving checks from behind."),
    ("What is triangulation in chess?", "Triangulation is a king manoeuvre in endgames where a king takes three moves to reach a square it could reach in two, effectively losing a tempo to gain opposition."),
    ("What is the concept of tempo in chess?", "Tempo refers to a unit of time — one move. Gaining a tempo means achieving your aim in one fewer move than the opponent. Losing a tempo means wasting a move."),
    ("What is a poisoned pawn?", "A poisoned pawn is a pawn that appears free to capture but leads to a worse position if taken. The Poisoned Pawn Variation in the Sicilian is a famous example."),
    ("What is the minority attack?", "The minority attack is a strategy where you advance fewer pawns against a larger pawn majority to create pawn weaknesses in the opponent's position."),
    ("What is a pawn break?", "A pawn break is a pawn advance that challenges the opponent's pawn structure. Timing pawn breaks correctly is a key strategic skill in chess."),
    ("What does it mean to have the initiative in chess?", "Having the initiative means making threats that force the opponent to react. The player with the initiative sets the agenda and the opponent can only respond."),
    ("What is compensation in chess?", "Compensation refers to having non-material advantages like better piece activity, a dangerous attack, or structural superiority to make up for a material deficit."),
    ("How do you study chess effectively?", "Study chess effectively by solving tactics puzzles daily, analysing your own games to find mistakes, studying endgames, learning a few opening systems well, and reviewing master games."),
    ("What is a chess engine?", "A chess engine is a computer program that plays chess at superhuman level. Engines like Stockfish and Komodo can evaluate positions millions of moves deep."),
    ("What is Stockfish?", "Stockfish is one of the strongest free chess engines in the world. It uses alpha-beta search and neural network evaluation to play at a level far beyond any human."),
    ("What is AlphaZero?", "AlphaZero is an AI system developed by DeepMind that mastered chess through self-play alone. It developed creative and unconventional strategies that amazed the chess world."),
    ("What rating do you need to be a chess master?", "A FIDE rating of 2200 earns the title of FIDE Master. Candidate Master requires 2200 in some federations. International Master needs 2400 and Grandmaster needs 2500."),
    ("What is chess called in Swahili?", "Chess is called 'Sataranji' in Swahili, derived from the Arabic word for chess. It is a popular game across East Africa."),
    ("How do you say checkmate in Swahili?", "Checkmate in Swahili is 'Shah mat' — borrowed from Persian meaning the king is dead or helpless."),
    ("Is chess popular in Kenya?", "Chess is growing in Kenya. The Kenya Chess Federation organises national championships and Kenya has produced several strong players competing at continental level."),
    ("What is the capital of Kenya?", "Nairobi is the capital and largest city of Kenya. It is also the economic hub of East Africa and home to many international organisations."),
    ("Where is Kenya located?", "Kenya is located in East Africa, bordered by Ethiopia to the north, Somalia to the northeast, Tanzania to the south, Uganda to the west, and the Indian Ocean to the southeast."),
    ("What is the population of Kenya?", "Kenya has a population of approximately 55 million people, making it one of the most populous countries in East Africa and Africa as a whole."),
    ("What is the size of Kenya?", "Kenya covers an area of about 580,367 square kilometres, making it the 48th largest country in the world by area."),
    ("What are the major cities in Kenya?", "The major cities in Kenya include Nairobi, Mombasa, Kisumu, Nakuru, Eldoret, Thika, and Malindi. Nairobi is the largest and Mombasa is the second largest."),
    ("What is the Great Rift Valley?", "The Great Rift Valley is a massive geological formation that runs through Kenya from north to south. It contains several lakes, volcanoes, and is home to diverse wildlife and the Maasai people."),
    ("What is Mount Kenya?", "Mount Kenya is the highest mountain in Kenya at 5,199 metres, making it the second highest peak in Africa after Kilimanjaro. It is a UNESCO World Heritage Site and national park."),
    ("What are the major lakes in Kenya?", "Kenya's major lakes include Lake Victoria — Africa's largest lake — Lake Turkana, Lake Naivasha, Lake Nakuru, Lake Bogoria, and Lake Baringo."),
    ("What is Lake Turkana?", "Lake Turkana is the world's largest permanent desert lake and the world's largest alkaline lake. It is located in northern Kenya and is sometimes called the Jade Sea due to its colour."),
    ("What is the Indian Ocean coastline like in Kenya?", "Kenya has a beautiful 536km coastline along the Indian Ocean. The coast features white sand beaches, coral reefs, historic towns like Mombasa and Malindi, and Swahili culture."),
    ("What are the main ethnic groups in Kenya?", "Kenya has over 40 ethnic groups. The largest include the Kikuyu, Luhya, Luo, Kalenjin, Kamba, Kisii, Meru, and Maasai. Each group has its own language and traditions."),
    ("What is the official language of Kenya?", "Kenya has two official languages: Swahili and English. Swahili is the national language used in everyday communication while English is used in government, business, and education."),
    ("What is Swahili?", "Swahili, also called Kiswahili, is a Bantu language spoken across East Africa. It is the most widely spoken African language and serves as a lingua franca in Kenya, Tanzania, Uganda, and beyond."),
    ("What does Jambo mean?", "Jambo is a Swahili greeting meaning hello. It is commonly used when greeting tourists or strangers. Among friends, Kenyans more commonly say Habari or Mambo."),
    ("What does Hakuna Matata mean?", "Hakuna Matata is a Swahili phrase meaning no worries or no problems. It became internationally famous through the Lion King but is genuinely used in daily Swahili conversation."),
    ("What does Asante mean in Swahili?", "Asante means thank you in Swahili. To say thank you very much, you say Asante sana. The reply is Karibu, meaning you are welcome."),
    ("What is Sheng?", "Sheng is a mixed language combining Swahili, English, and various local languages. It originated among youth in Nairobi slums and is now widely spoken by young Kenyans across the country."),
    ("What does Sawa mean in Swahili?", "Sawa means okay or fine in Swahili. It is one of the most commonly used words in everyday Kenyan conversation. Sawa sawa emphasises it even more."),
    ("What does Karibu mean?", "Karibu means welcome or you are welcome in Swahili. It is used to welcome guests, respond to thank you, and invite someone to come in."),
    ("What does Pole mean in Swahili?", "Pole means sorry or take it easy in Swahili. It is used to express sympathy or condolences. Pole pole means slowly or take your time."),
    ("What does Mambo mean in Swahili?", "Mambo is a casual Swahili greeting especially popular among youth. The typical response is Poa, meaning cool or fine. Mambo vipi means how are things?"),
    ("What does Poa mean?", "Poa means cool or fine in Sheng and Swahili. It is used as a response to Mambo or as a general expression of approval. Poa kama ndizi — cool as a banana — is a popular phrase."),
    ("What is Ugali?", "Ugali is Kenya's staple food — a thick porridge made from maize flour cooked in boiling water until firm. It is usually eaten with sukuma wiki, meat stew, or fish."),
    ("What is Sukuma Wiki?", "Sukuma wiki are collard greens — a leafy vegetable that is one of the most common side dishes in Kenya. The name literally means push the week, reflecting its affordability."),
    ("What is Nyama Choma?", "Nyama Choma means roasted meat in Swahili and is Kenya's most beloved dish. Goat or beef is slowly grilled over charcoal and served with kachumbari salad and ugali."),
    ("What is Githeri?", "Githeri is a traditional Kenyan dish made from boiled maize and beans cooked together. It is nutritious, affordable, and beloved across all communities in Kenya."),
    ("What is Mandazi?", "Mandazi is a popular East African fried bread similar to a doughnut but less sweet. It is commonly eaten for breakfast with tea or as a snack."),
    ("What is Chapati in Kenya?", "Chapati is a flatbread introduced to Kenya by Indian immigrants and now deeply embedded in Kenyan culture. It is a special occasion food often served with beef stew or chicken."),
    ("What is Mutura?", "Mutura is a traditional Kikuyu sausage made from goat or beef intestines stuffed with blood and meat. It is roasted on coals and is popular street food especially in Central Kenya."),
    ("What is Kenyan tea culture?", "Kenya is one of the world's top tea producers and Kenyans are passionate tea drinkers. Chai — milk tea boiled with water, milk, sugar, and sometimes ginger — is served throughout the day."),
    ("What is M-Pesa?", "M-Pesa is a mobile money service launched by Safaricom in Kenya in 2007. It revolutionised financial services across Africa, allowing millions without bank accounts to send and receive money via phone."),
    ("What is Safaricom?", "Safaricom is Kenya's largest telecommunications company and a regional innovator. It is known for M-Pesa mobile money, high network coverage, and the Bonga points loyalty programme."),
    ("What is the Kenyan currency?", "The official currency of Kenya is the Kenyan Shilling, abbreviated as KES or KSh. It is subdivided into 100 cents."),
    ("What is KPLC?", "KPLC stands for Kenya Power and Lighting Company. It is the sole electricity distributor in Kenya, responsible for transmitting and distributing electricity to homes and businesses."),
    ("What are Kenya's main exports?", "Kenya's main exports include tea, cut flowers, coffee, horticultural products, and refined petroleum. Tea is the largest earner, making Kenya one of the world's top tea exporters."),
    ("What is the Nairobi Stock Exchange?", "The Nairobi Securities Exchange, abbreviated NSE, is Kenya's main stock exchange. It was established in 1954 and lists over 60 companies across various sectors of the economy."),
    ("What is the Kenya Revenue Authority?", "The Kenya Revenue Authority, KRA, is the government body responsible for collecting taxes and enforcing tax laws in Kenya. It manages income tax, VAT, and customs duties."),
    ("What is Kenya's main industry?", "Kenya's economy is driven by services, agriculture, manufacturing, and tourism. The service sector, including finance and telecommunications, contributes the most to GDP."),
    ("What is the Vision 2030 in Kenya?", "Vision 2030 is Kenya's long-term development blueprint aimed at transforming Kenya into a newly industrialised middle-income country by 2030 with a high quality of life for all citizens."),
    ("What is the Big Four Agenda in Kenya?", "The Big Four Agenda was a government initiative focusing on food security, affordable housing, manufacturing, and universal healthcare as key development priorities."),
    ("What animals are in the Maasai Mara?", "The Maasai Mara is home to lions, elephants, leopards, cheetahs, buffaloes, rhinos, zebras, wildebeest, hippos, giraffes, and hundreds of bird species."),
    ("What is the Great Migration?", "The Great Migration is the annual movement of over 1.5 million wildebeest and hundreds of thousands of zebras from the Serengeti in Tanzania to the Maasai Mara in Kenya and back."),
    ("What are the Big Five animals?", "The Big Five are the lion, leopard, elephant, buffalo, and rhinoceros. The term was originally used by hunters and now refers to the most sought-after animals to spot on a safari."),
    ("What national parks are in Kenya?", "Kenya has many national parks including Maasai Mara, Amboseli, Tsavo East and West, Lake Nakuru, Aberdare, Mount Kenya, Samburu, and Nairobi National Park."),
    ("What is Amboseli National Park known for?", "Amboseli National Park is famous for its large elephant herds and stunning views of Mount Kilimanjaro across the border in Tanzania. It is one of Africa's most scenic parks."),
    ("What is Lake Nakuru National Park famous for?", "Lake Nakuru National Park is famous for its flocks of flamingos that turn the lake pink, as well as white and black rhinos, lions, leopards, and over 400 bird species."),
    ("What is the Maasai culture?", "The Maasai are a semi-nomadic people living in Kenya and Tanzania. They are known for their distinctive red clothing, jumping dance called adumu, cattle herding traditions, and beadwork."),
    ("Who was the first President of Kenya?", "Jomo Kenyatta was Kenya's first president after independence in 1963. He led the country until his death in 1978 and is revered as the founding father of the nation."),
    ("When did Kenya gain independence?", "Kenya gained independence from British colonial rule on December 12, 1963. This date is celebrated annually as Jamhuri Day — Kenya's national day."),
    ("What is Jamhuri Day?", "Jamhuri Day is Kenya's national day celebrated on December 12. Jamhuri means republic in Swahili. On this day in 1964 Kenya became a republic with Jomo Kenyatta as president."),
    ("What is Madaraka Day?", "Madaraka Day is celebrated on June 1 and marks the day Kenya attained self-governance from Britain in 1963. Madaraka means self-rule in Swahili."),
    ("What is Mashujaa Day?", "Mashujaa Day, celebrated on October 20, honours heroes who fought for Kenya's independence. Mashujaa means heroes in Swahili. It was formerly called Kenyatta Day."),
    ("Who is the current President of Kenya?", "William Ruto became Kenya's fifth president in September 2022 after winning the general election. He introduced the bottom-up economic model as his key development approach."),
    ("What is the Kenyan constitution?", "Kenya promulgated a new constitution in 2010 through a referendum. It introduced devolution, a bill of rights, independent oversight institutions, and a bicameral parliament."),
    ("What is devolution in Kenya?", "Devolution in Kenya is the transfer of power and resources from the national government to 47 county governments. It was introduced by the 2010 constitution to bring services closer to the people."),
    ("What is Westlands in Nairobi?", "Westlands is a major commercial and entertainment district in Nairobi known for its restaurants, malls, clubs, and business offices. It is one of the most vibrant parts of the city."),
    ("What is CBD in Nairobi?", "The CBD — Central Business District — is the downtown core of Nairobi. It contains government offices, banks, shops, and the iconic Kenyatta International Conference Centre."),
    ("What is Kibera?", "Kibera is one of Africa's largest informal settlements, located in Nairobi. It is home to hundreds of thousands of people and is known for its resilient community spirit and vibrant culture."),
    ("What is Gikomba market?", "Gikomba is Nairobi's largest open-air market, famous for second-hand clothing, shoes, and household items. It is a hub of entrepreneurship and informal trade in the city."),
    ("What is the matatu culture in Nairobi?", "Matatus are privately owned minibuses that are the primary public transport in Nairobi. They are known for their colourful artwork, loud music, and are a unique expression of Kenyan street culture."),
    ("What is Uhuru Park?", "Uhuru Park is a large public park in central Nairobi. Uhuru means freedom in Swahili. The park is used for public events, relaxation, and is near the CBD."),
    ("What is the KICC in Nairobi?", "The Kenyatta International Convention Centre, KICC, is a landmark building in Nairobi's CBD. It is a cylindrical tower used for conferences and international meetings."),
    ("What is Sarit Centre?", "Sarit Centre is one of Nairobi's oldest and most popular shopping malls located in Westlands. It has shops, cinemas, restaurants, and a supermarket."),
    ("How do you say goodbye in Swahili?", "To say goodbye in Swahili you say Kwaheri to one person or Kwaherini to a group. A casual goodbye is Baadaye meaning see you later or Tutaonana meaning we will see each other."),
    ("How do you count to ten in Swahili?", "In Swahili: moja (1), mbili (2), tatu (3), nne (4), tano (5), sita (6), saba (7), nane (8), tisa (9), kumi (10)."),
    ("What does Niko sawa mean?", "Niko sawa means I am fine or I am okay in Swahili. Niko means I am and sawa means fine or okay."),
    ("What does Chakula mean?", "Chakula means food in Swahili. Chakula cha mchana is lunch, chakula cha usiku is dinner, and chakula cha asubuhi is breakfast."),
    ("What does Maji mean in Swahili?", "Maji means water in Swahili. It is one of the most important words to know. Maji moto is hot water and maji baridi is cold water."),
    ("What does Nyumba mean?", "Nyumba means house or home in Swahili. Nyumba yangu means my house. Nyumbani means at home."),
    ("What does Barabara mean in Swahili?", "Barabara means road or highway in Swahili. It also means exactly or precisely in another context, showing how Swahili words can have multiple meanings."),
    ("What does Haraka mean?", "Haraka means hurry or quickly in Swahili. Haraka haraka haina baraka is a famous Swahili proverb meaning haste haste has no blessings — rushing leads to mistakes."),
    ("What does Kesho mean?", "Kesho means tomorrow in Swahili. Leo means today and Jana means yesterday. Kesho kutwa means the day after tomorrow."),
    ("What does Bado mean?", "Bado means not yet or still in Swahili. For example, Sijafika bado means I have not arrived yet. It is a very commonly used word in daily conversation."),
    ("What does Rafiki mean?", "Rafiki means friend in Swahili. It was made internationally famous by the Lion King character. Marafiki is the plural form meaning friends."),
    ("What does Simba mean?", "Simba means lion in Swahili. It is one of the most internationally recognised Swahili words thanks to the Lion King. Simba is also a common name in Kenya."),
    ("What is a Boda Boda?", "A boda boda is a motorcycle taxi very common across Kenya and East Africa. The name comes from the border to border transport service they originally provided between Uganda and Kenya."),
    ("What does Tafadhali mean?", "Tafadhali means please in Swahili. It is used when making requests. Tafadhali nisaidie means please help me."),
    ("What does Samahani mean?", "Samahani means sorry or excuse me in Swahili. It is used to apologise or to get someone's attention politely."),
    ("What does Ninaomba mean?", "Ninaomba means I am asking for or I request in Swahili. It is used when making a request politely, similar to please or I would like."),
    ("What does Nzuri mean?", "Nzuri means good, fine, or beautiful in Swahili. Nzuri sana means very good or very beautiful. It is used to describe people, things, and situations."),
    ("What does Mbaya mean?", "Mbaya means bad or not good in Swahili. Si mbaya means not bad. Mambo mabaya are bad things or bad situations."),
    ("What does Wewe mean?", "Wewe means you in Swahili. Mimi means I or me, and yeye means he or she. These are basic pronouns essential for Swahili conversation."),
    ("What does Kwenda mean?", "Kwenda means to go in Swahili. Naenda means I am going. Twende means let us go. Unaenda wapi? means where are you going?"),
    ("What does Sema mean in Kenyan slang?", "Sema means speak, say, or what's up in Kenyan Sheng. Sema bana means what's up mate. It is one of the most common casual greetings among young Kenyans."),
    ("What does Niaje mean?", "Niaje is Sheng for how are you or what's up. It is a very popular casual greeting among young Kenyans. The response could be Poa or Sawa."),
    ("What does Msee mean in Sheng?", "Msee means guy or dude in Sheng. It comes from the Swahili word mzee meaning elder but has been repurposed in Sheng to mean any male person. Watu means people."),
    ("What does Manze mean in Kenyan slang?", "Manze is an exclamation in Sheng similar to man or wow. It is used to express surprise, emphasis, or agreement. Manze hiyo ni poa means man that's cool."),
    ("What does Buda mean in Sheng?", "Buda means friend, guy, or father in Sheng. It is derived from the English word buddha or father and is used as a casual term of address among young Kenyan men."),
    ("What is the stock market?", "The stock market is a marketplace where buyers and sellers trade shares of publicly listed companies. Prices are determined by supply and demand and reflect investor expectations about company performance."),
    ("What is a stock?", "A stock, also called a share or equity, represents ownership in a company. When you buy a stock you become a part-owner and can benefit from the company's growth and profit through price appreciation and dividends."),
    ("What is a bond?", "A bond is a debt instrument where an investor loans money to a company or government for a fixed period at a fixed interest rate. Bonds are generally less risky than stocks but offer lower returns."),
    ("What is a dividend?", "A dividend is a portion of a company's profits paid to shareholders. Not all companies pay dividends. Growth companies typically reinvest profits while mature companies tend to pay regular dividends."),
    ("What is market capitalisation?", "Market capitalisation is the total value of a company's outstanding shares. It equals share price multiplied by number of shares. Companies are classified as large-cap, mid-cap, or small-cap based on this."),
    ("What is a bull market?", "A bull market is a period of rising stock prices, typically defined as a 20% rise from recent lows. Bull markets are associated with economic growth, low unemployment, and investor optimism."),
    ("What is a bear market?", "A bear market is a period of falling stock prices, typically defined as a 20% decline from recent highs. Bear markets are associated with economic downturns, recessions, and investor pessimism."),
    ("What is portfolio diversification?", "Portfolio diversification means spreading investments across different assets, sectors, and geographies to reduce risk. When one investment falls, others may rise, smoothing out overall returns."),
    ("What is risk management in trading?", "Risk management involves strategies to limit potential losses. Key techniques include position sizing, stop-loss orders, diversification, and never risking more than a small percentage of capital on a single trade."),
    ("What is a stop-loss order?", "A stop-loss order automatically sells a position when it reaches a specified price. It protects traders from large losses by exiting trades that move against them beyond an acceptable level."),
    ("What is a take-profit order?", "A take-profit order automatically closes a trade when it reaches a target profit level. It locks in gains without requiring constant monitoring of the market."),
    ("What is leverage in trading?", "Leverage allows traders to control a larger position with a smaller amount of capital. While it amplifies potential gains it also amplifies potential losses and should be used with extreme caution."),
    ("What is short selling?", "Short selling is borrowing shares and selling them hoping the price will fall. You then buy them back at a lower price and return them, pocketing the difference. It is risky if the price rises instead."),
    ("What is a long position?", "A long position means you have bought an asset expecting its price to rise. You profit when the price goes up and lose if it goes down. Going long is the most common investment approach."),
    ("What is a short position?", "A short position means you have sold an asset you do not own, expecting the price to fall. You profit when the price falls and lose if it rises. Short positions have theoretically unlimited risk."),
    ("What is technical analysis?", "Technical analysis studies price charts and trading patterns to forecast future price movements. It uses tools like moving averages, RSI, MACD, and support and resistance levels."),
    ("What is fundamental analysis?", "Fundamental analysis evaluates a company's financial health, earnings, revenue, management, and competitive position to determine its intrinsic value and whether its stock is fairly priced."),
    ("What is a candlestick chart?", "A candlestick chart shows price movements over time using candles. Each candle shows the open, high, low, and close price for a period. Green candles mean the price rose, red means it fell."),
    ("What is support and resistance in trading?", "Support is a price level where buying interest is strong enough to prevent further decline. Resistance is a level where selling pressure prevents further rise. These levels are key in technical analysis."),
    ("What is a moving average?", "A moving average smooths price data by calculating the average price over a period. The 50-day and 200-day moving averages are widely watched. When they cross it signals potential trend changes."),
    ("What is RSI in trading?", "The Relative Strength Index, RSI, measures the speed and magnitude of recent price changes to assess if an asset is overbought or oversold. A reading above 70 suggests overbought, below 30 suggests oversold."),
    ("What is MACD?", "MACD stands for Moving Average Convergence Divergence. It is a trend-following indicator that shows the relationship between two moving averages. Crossovers and divergences are used as trading signals."),
    ("What is Bollinger Bands?", "Bollinger Bands consist of a middle moving average line and two outer bands that expand and contract based on volatility. Prices touching the outer bands often signal potential reversals."),
    ("What is volume in trading?", "Volume refers to the number of shares or contracts traded in a period. High volume confirms the strength of a price move while low volume suggests a move may be weak or unsustainable."),
    ("What is a trend in trading?", "A trend is the general direction of price movement. An uptrend has higher highs and higher lows. A downtrend has lower highs and lower lows. Trading with the trend is a fundamental principle."),
    ("What is Bitcoin?", "Bitcoin is the first and most valuable cryptocurrency, created in 2009 by an anonymous person or group known as Satoshi Nakamoto. It is a decentralised digital currency that operates on a blockchain."),
    ("What is blockchain?", "A blockchain is a distributed ledger that records transactions across many computers. Each block of transactions is cryptographically linked to the previous one, making the data tamper-resistant and transparent."),
    ("What is Ethereum?", "Ethereum is the second largest cryptocurrency by market cap. Beyond being a currency, it is a programmable blockchain that enables smart contracts and decentralised applications, or DApps."),
    ("What is a smart contract?", "A smart contract is a self-executing agreement written in code on a blockchain. It automatically enforces and executes contract terms when predefined conditions are met, without needing intermediaries."),
    ("What is DeFi?", "DeFi stands for Decentralised Finance — financial services built on blockchain networks that operate without traditional banks or intermediaries. It includes lending, borrowing, trading, and earning interest on crypto."),
    ("What is an NFT?", "An NFT or Non-Fungible Token is a unique digital asset stored on a blockchain. Unlike regular crypto, each NFT is one-of-a-kind. They are used for digital art, collectibles, and gaming items."),
    ("What is crypto mining?", "Crypto mining is the process of verifying transactions and adding them to the blockchain by solving complex mathematical puzzles. Miners are rewarded with newly created cryptocurrency for their work."),
    ("What is a crypto wallet?", "A crypto wallet stores the private and public keys needed to access and transact cryptocurrency. Wallets can be hardware devices, software apps, or paper. The wallet does not store coins but keys."),
    ("What is a private key in crypto?", "A private key is a secret code that gives access to your cryptocurrency. Anyone with your private key can spend your crypto. Never share it with anyone and store it securely offline."),
    ("What is a public key in crypto?", "A public key is your crypto address — like a bank account number — that others use to send you cryptocurrency. It is safe to share publicly but cannot be used to spend your funds."),
    ("What is altcoin?", "An altcoin is any cryptocurrency other than Bitcoin. The name comes from alternative coin. Ethereum, Solana, Cardano, and thousands of others are altcoins. Their risk and potential vary widely."),
    ("What is market cap in crypto?", "Crypto market cap equals current price multiplied by total circulating supply. It measures the total value of a cryptocurrency. Bitcoin's market cap is the largest in the crypto market."),
    ("What is a crypto exchange?", "A crypto exchange is a platform where you can buy, sell, and trade cryptocurrencies. Examples include Binance, Coinbase, Kraken, and locally in Kenya, Yellow Card and Paxful."),
    ("What is Binance?", "Binance is the world's largest cryptocurrency exchange by trading volume. It offers spot trading, futures, staking, and a wide range of cryptocurrencies. It has a large user base in Africa."),
    ("How can I buy crypto in Kenya?", "You can buy crypto in Kenya through exchanges like Binance, Yellow Card, or Paxful. You can pay via M-Pesa. Always use reputable platforms and start with small amounts to understand the market."),
    ("What is compound interest?", "Compound interest is earning interest on your interest. Over time it causes wealth to grow exponentially. Einstein reportedly called it the eighth wonder of the world. Starting early maximises its effect."),
    ("What is inflation?", "Inflation is the rate at which the general price level of goods and services rises over time, reducing purchasing power. A moderate inflation rate around 2-3% is considered healthy for an economy."),
    ("What is an index fund?", "An index fund is a type of mutual fund that tracks a market index like the S&P 500. It offers broad diversification at low cost and historically outperforms most actively managed funds over time."),
    ("What is ETF?", "An ETF or Exchange-Traded Fund is a basket of securities that trades on a stock exchange like an individual stock. It combines the diversification of mutual funds with the flexibility of stock trading."),
    ("What is dollar-cost averaging?", "Dollar-cost averaging means investing a fixed amount regularly regardless of price. When prices are low you buy more units, when high you buy fewer. Over time this reduces the impact of market volatility."),
    ("What is a P/E ratio?", "The Price-to-Earnings ratio compares a company's stock price to its earnings per share. A high P/E suggests investors expect high growth. A low P/E may indicate an undervalued stock or poor prospects."),
    ("What is ROI?", "ROI stands for Return on Investment. It measures the profit or loss relative to the initial investment amount, expressed as a percentage. ROI = (gain minus cost) divided by cost multiplied by 100."),
    ("What is liquidity in finance?", "Liquidity refers to how easily an asset can be converted to cash without significantly affecting its price. Cash is the most liquid asset. Real estate is illiquid because selling takes time and cost."),
    ("What is a recession?", "A recession is a significant decline in economic activity for two or more consecutive quarters. It is characterised by falling GDP, rising unemployment, reduced spending, and declining business activity."),
    ("What is GDP?", "GDP or Gross Domestic Product is the total monetary value of all goods and services produced in a country over a specific period. It is the primary measure of an economy's size and health."),
    ("What is the NSE in Kenya?", "The Nairobi Securities Exchange is Kenya's stock exchange where shares of listed Kenyan companies are bought and sold. It was established in 1954 and has grown to include bonds and derivatives."),
    ("What is MPESA float?", "M-Pesa float refers to the cash balance an M-Pesa agent holds to facilitate transactions. Agents need sufficient float to give cash to customers withdrawing money."),
    ("What is Fuliza in Kenya?", "Fuliza is Safaricom's M-Pesa overdraft service that allows users to complete transactions even when they have insufficient M-Pesa balance. The overdraft is automatically repaid from future deposits."),
    ("What is KCB M-Pesa?", "KCB M-Pesa is a partnership between Kenya Commercial Bank and Safaricom offering savings and loan services directly through the M-Pesa platform without needing to visit a bank branch."),
    ("What is a SACCO in Kenya?", "A SACCO, or Savings and Credit Cooperative Organisation, is a member-owned financial institution in Kenya. Members save together and can access loans at competitive rates. SACCOs are very popular in Kenya."),
    ("What is the Central Bank of Kenya?", "The Central Bank of Kenya, CBK, is Kenya's monetary authority. It regulates the banking system, manages the currency, controls inflation through monetary policy, and supervises commercial banks."),
    ("What is mobile banking in Kenya?", "Mobile banking in Kenya allows customers to access banking services via their phones. M-Pesa is the most popular, but all major banks also offer apps for checking balances, transfers, and payments."),
    ("What is Treasury Bills in Kenya?", "Treasury Bills are short-term government securities sold by the Central Bank of Kenya. They are considered very low risk and are used by the government to raise money. They are popular with conservative investors."),
    ("What is Treasury Bonds in Kenya?", "Treasury Bonds are long-term government securities in Kenya with maturities of 2 to 30 years. They pay regular interest and are considered safe investments backed by the government."),
    ("How does the NSE work for small investors?", "Small investors can buy NSE shares through licensed stockbrokers or mobile apps like M-Akiba for government bonds. The minimum investment is low, making the market accessible to ordinary Kenyans."),
    ("What is FOMO in trading?", "FOMO stands for Fear of Missing Out. It causes traders to chase rising assets after missing an initial move, often buying at the top. FOMO-driven trades frequently result in losses."),
    ("What is FUD in crypto?", "FUD stands for Fear, Uncertainty, and Doubt. It refers to negative news or sentiment that causes panic selling in crypto markets. Experienced traders learn to distinguish FUD from genuine concerns."),
    ("What is HODLing?", "HODLing means holding cryptocurrency long-term despite price volatility rather than trading frequently. The term came from a misspelling of hold and is now a deliberate acronym for Hold On for Dear Life."),
    ("What is a pump and dump scheme?", "A pump and dump is a fraudulent scheme where promoters artificially inflate an asset's price through false hype then sell their holdings at the peak, leaving other investors with losses. It is illegal in regulated markets."),
    ("What does buy low sell high mean?", "Buy low sell high is the fundamental principle of profitable investing — purchase assets when prices are depressed and sell when prices are elevated. Simple in theory but psychologically difficult in practice."),
    ("What is emotional trading?", "Emotional trading is making investment decisions based on fear or greed rather than analysis. It leads to buying at peaks and selling at bottoms. Successful traders develop discipline to follow their strategy consistently."),
    ("What is a trading plan?", "A trading plan is a written set of rules defining your strategy, including entry criteria, exit points, position size, and risk management. Following a plan consistently is key to long-term trading success."),
    ("What is backtesting in trading?", "Backtesting is testing a trading strategy against historical data to see how it would have performed in the past. While past performance does not guarantee future results, backtesting helps validate strategy logic."),
    ("What is day trading?", "Day trading involves opening and closing positions within the same trading day to profit from short-term price movements. It requires significant knowledge, discipline, and time. Most beginners lose money day trading."),
    ("What is swing trading?", "Swing trading involves holding positions for days to weeks to capture medium-term price swings. It requires less time than day trading but more active management than long-term investing."),
    ("What is voltage?", "Voltage is the electrical pressure that pushes electrons through a circuit. It is measured in volts (V) and represents the potential difference between two points. Without voltage there is no current flow."),
    ("What is current?", "Current is the flow of electric charge through a conductor, measured in amperes (A). It represents how many electrons pass a point per second. More current means more electrons flowing."),
    ("What is resistance?", "Resistance opposes the flow of electric current and is measured in ohms (Ω). Every material has resistance. High resistance restricts current flow while low resistance allows it to flow freely."),
    ("What is Ohm's Law?", "Ohm's Law states that voltage equals current multiplied by resistance: V = I × R. It is the most fundamental equation in electrical engineering, relating voltage, current, and resistance."),
    ("What is power in electrical terms?", "Electrical power is the rate at which electrical energy is consumed or produced, measured in watts (W). Power equals voltage multiplied by current: P = V × I. A 100W bulb uses 100 joules per second."),
    ("What is AC power?", "AC or Alternating Current is electricity where the current periodically reverses direction. It is the type of electricity supplied by the national grid to homes and businesses. Kenya uses 240V 50Hz AC."),
    ("What is DC power?", "DC or Direct Current flows in one direction only. Batteries, solar panels, and electronic devices use DC. AC is converted to DC in power supplies for computers, phones, and most electronics."),
    ("What is a circuit breaker?", "A circuit breaker is a safety device that automatically cuts off electrical supply when current exceeds a safe level. Unlike a fuse it can be reset after tripping. It protects wiring from overheating and fire."),
    ("What is a fuse?", "A fuse is a safety device containing a thin wire that melts and breaks the circuit when current exceeds a safe level. Once blown it must be replaced unlike a circuit breaker which can be reset."),
    ("What is earthing or grounding?", "Earthing connects electrical equipment to the ground to provide a safe path for fault currents. If a live wire touches a metal casing, the current flows safely to earth instead of through a person."),
    ("What is a neutral wire?", "The neutral wire completes the electrical circuit by returning current to the supply. In a standard installation the neutral wire is connected to earth at the supply point and is kept at zero volts."),
    ("What is a live wire?", "The live wire carries current from the supply to the load. It is at the supply voltage — 240V in Kenya — and is dangerous to touch. Live wires are usually brown or red in modern wiring."),
    ("What is single phase power?", "Single phase power uses one live wire and a neutral to supply 230-240V. It is suitable for homes and small businesses. Most household appliances in Kenya run on single phase."),
    ("What is three phase power?", "Three phase power uses three live wires providing 415V between phases. It is used for industrial equipment, large motors, and commercial buildings because it handles higher loads more efficiently."),
    ("What is a relay?", "A relay is an electrically operated switch. A small current in the control circuit opens or closes contacts that switch a larger current in the load circuit. Relays are used in automation and control systems."),
    ("What is a contactor?", "A contactor is like a heavy-duty relay designed for frequently switching large electrical loads like motors. It can handle much higher currents than a standard relay and is common in industrial panels."),
    ("What is a transformer?", "A transformer changes AC voltage levels. Step-up transformers increase voltage and step-down transformers reduce it. They use electromagnetic induction and only work with AC. KPLC uses transformers to distribute power."),
    ("What is a UPS?", "A UPS or Uninterruptible Power Supply provides emergency power from a battery when the main supply fails. It protects computers and sensitive equipment from power cuts and voltage fluctuations."),
    ("What is a kWh?", "A kilowatt-hour is the unit of electrical energy used for billing. One kWh equals using 1000 watts for one hour. KPLC bills customers based on the number of kWh consumed per month."),
    ("What is a DALI system?", "DALI stands for Digital Addressable Lighting Interface. It is a protocol for controlling lighting systems digitally. Each DALI device has a unique address allowing individual or group control of lights."),
    ("What is a DALI gateway?", "A DALI gateway is a device that bridges DALI lighting control systems with other building automation systems like BACnet or Modbus. It allows centralised control and monitoring of lighting from a single point."),
    ("What is a ballast in lighting?", "A ballast regulates the current to fluorescent and other discharge lamps. Electronic ballasts are more efficient than magnetic ballasts. In LED systems the driver performs the same function."),
    ("What is an MCB?", "An MCB or Miniature Circuit Breaker is a compact circuit breaker used in domestic and commercial distribution boards. It automatically trips when current exceeds its rated value, protecting the circuit."),
    ("What is an MCCB?", "An MCCB or Moulded Case Circuit Breaker handles higher currents than an MCB, typically from 100A to 2500A. It is used in industrial applications and main distribution boards."),
    ("What is an RCCB?", "An RCCB or Residual Current Circuit Breaker detects earth leakage currents and disconnects the supply to prevent electric shock. It is a critical safety device especially in bathrooms and kitchens."),
    ("What is an isolator switch?", "An isolator is a manual switch used to disconnect electrical equipment from the supply for maintenance. Unlike a circuit breaker it is not designed to interrupt fault currents and is only operated when the load is off."),
    ("What is a distribution board?", "A distribution board, also called a consumer unit or DB board, distributes electrical supply from the main incoming supply to individual circuits. It contains MCBs or fuses for each circuit."),
    ("What is cable tray?", "A cable tray is a rigid structure used to support and route electrical cables in buildings and industrial facilities. It is preferable to conduit for large cable runs as it allows easy installation and modification."),
    ("What is conduit?", "Conduit is a tube used to protect and route electrical cables. It can be metal or plastic. Conduit protects wires from physical damage, moisture, and allows replacement of cables without major work."),
    ("What is a LAN?", "A LAN or Local Area Network connects computers and devices within a limited area such as a home, office, or building. It enables resource sharing like printers and internet access across connected devices."),
    ("What is a WAN?", "A WAN or Wide Area Network connects computers and networks over large geographic distances. The internet is the largest WAN. Companies use WANs to connect offices in different cities or countries."),
    ("What is a router?", "A router directs network traffic between different networks. Home routers connect your local network to the internet. They assign IP addresses via DHCP and manage data packets efficiently."),
    ("What is a switch in networking?", "A network switch connects multiple devices on a LAN and directs data only to the intended recipient device. Managed switches allow configuration of VLANs, port security, and traffic prioritisation."),
    ("What is an IP address?", "An IP address is a unique numerical label assigned to each device on a network. IPv4 addresses are like 192.168.1.1. IPv6 uses a longer format. IP addresses allow devices to find and communicate with each other."),
    ("What is DHCP?", "DHCP or Dynamic Host Configuration Protocol automatically assigns IP addresses to devices on a network. Without DHCP you would need to manually configure the IP address on every device."),
    ("What is DNS?", "DNS or Domain Name System translates human-readable domain names like google.com into IP addresses that computers use to communicate. It is essentially the phone book of the internet."),
    ("What is a firewall?", "A firewall is a security system that monitors and controls incoming and outgoing network traffic based on predefined rules. It creates a barrier between trusted internal networks and untrusted external networks."),
    ("What is Ethernet?", "Ethernet is the most common wired networking technology. It uses cables — Cat5e, Cat6, or Cat7 — and provides reliable high-speed connections. Cat6 cables support speeds up to 10 Gbps."),
    ("What is Cat6 cable?", "Cat6 cable is a standard Ethernet cable that supports speeds up to 10 Gbps over distances up to 55 metres. It is the recommended standard for new network installations in buildings."),
    ("What is WiFi?", "WiFi is wireless networking technology that allows devices to connect to a network using radio waves. WiFi 6 is the latest standard offering faster speeds, better performance in crowded areas, and improved efficiency."),
    ("What is a VLAN?", "A VLAN or Virtual Local Area Network divides a physical network into multiple logical networks. It improves security and performance by isolating different groups of users or devices on the same physical infrastructure."),
    ("What is PoE?", "PoE or Power over Ethernet delivers electrical power through network cables alongside data. It is used to power devices like IP cameras, access points, and IP phones without needing separate power supplies."),
    ("What is a patch panel?", "A patch panel is a mounted assembly in a server room or network cabinet that organises cable connections. It provides a central point for managing and rerouting network connections without touching the main cables."),
    ("What is structured cabling?", "Structured cabling is a standardised system of cables, connectors, and hardware that provides a comprehensive telecommunications infrastructure for a building. It supports voice, data, and other services."),
    ("What is a server?", "A server is a computer that provides services or resources to other computers on a network. Web servers host websites, file servers store documents, and email servers manage email communication."),
    ("What is cloud computing?", "Cloud computing delivers computing services — servers, storage, databases, networking, software — over the internet. Instead of owning hardware, users pay for what they use. AWS, Azure, and Google Cloud are examples."),
    ("What is a VPN?", "A VPN or Virtual Private Network creates an encrypted tunnel between your device and a server, protecting your data and masking your IP address. It is used for privacy and accessing geo-restricted content."),
    ("What is an access point?", "A wireless access point extends WiFi coverage in a building. It connects to the wired network and broadcasts a wireless signal. Multiple access points can be managed centrally in enterprise deployments."),
    ("What is network topology?", "Network topology is the physical or logical arrangement of a network. Common topologies include star where all devices connect to a central switch, ring, bus, and mesh where devices interconnect directly."),
    ("What is ping?", "Ping is a network diagnostic tool that tests connectivity between two devices by sending packets and measuring response time. A high ping indicates network delays. Ping = 0 means the device is unreachable."),
    ("What is bandwidth?", "Bandwidth is the maximum rate at which data can be transferred over a network, measured in Mbps or Gbps. Higher bandwidth allows more data to flow simultaneously, improving network performance."),
    ("What is latency in networking?", "Latency is the delay between sending and receiving data, measured in milliseconds. Low latency is critical for real-time applications like video calls, gaming, and trading systems."),
    ("What is NAT?", "NAT or Network Address Translation allows multiple devices on a private network to share a single public IP address. Home routers use NAT to connect all your devices to the internet with one IP."),
    ("What is subnetting?", "Subnetting divides a large network into smaller sub-networks. It improves security and performance by isolating traffic and reducing broadcast domains. A /24 subnet supports 254 hosts."),
    ("What is a gateway in networking?", "A network gateway is a node that connects two different networks, often with different protocols. Your home router acts as a gateway between your local network and the internet."),
    ("What is CCTV?", "CCTV or Closed Circuit Television is a video surveillance system. Modern IP CCTV systems record digitally and can be accessed remotely. They are widely used for security in homes, offices, and public spaces."),
    ("What is an IP camera?", "An IP camera is a network-connected security camera that sends video data over the network. Unlike analogue cameras they can be accessed remotely, offer higher resolution, and integrate with NVR systems."),
    ("What is an NVR?", "An NVR or Network Video Recorder records footage from IP cameras on a network. It stores video on a hard drive and allows remote playback. It is the modern alternative to DVR systems used with analogue cameras."),
    ("What is solar power?", "Solar power converts sunlight into electricity using photovoltaic panels. It is clean, renewable, and increasingly affordable. In Kenya solar is popular in off-grid areas and is growing in urban homes."),
    ("What is a solar inverter?", "A solar inverter converts DC electricity produced by solar panels into AC electricity that can power household appliances. Hybrid inverters can work with batteries and the grid simultaneously."),
    ("What is a charge controller?", "A solar charge controller regulates the voltage and current from solar panels to the battery, preventing overcharging and extending battery life. MPPT controllers are more efficient than PWM types."),
    ("How do I stay motivated?", "Motivation follows action, not the other way around. Instead of waiting to feel motivated, take one small step and let the momentum build. Break your goal into tiny tasks and celebrate each completion."),
    ("I am feeling overwhelmed", "When everything feels too much, try narrowing your focus to just the next one hour. You do not have to solve everything at once. Breathe deeply, write down what is stressing you, and tackle one thing at a time."),
    ("How do I deal with failure?", "Failure is feedback, not a verdict. Every successful person has a collection of failures. Analyse what went wrong, extract the lesson, adjust your approach, and try again. Failure is the tuition you pay for expertise."),
    ("How do I build confidence?", "Confidence comes from doing, not thinking. Make a list of small challenges and face them one by one. Each time you do something uncomfortable and survive, your confidence grows. Evidence builds belief."),
    ("I am stressed about money", "Financial stress is one of the heaviest burdens. Start by writing down exactly what you owe and what comes in. Clarity reduces anxiety even if the numbers are hard to look at. Small consistent steps create big change."),
    ("How do I stop procrastinating?", "Procrastination is usually about avoiding discomfort, not laziness. Try the two-minute rule — if something takes under two minutes, do it immediately. For bigger tasks, commit to just five minutes. Starting is the hardest part."),
    ("I am anxious about the future", "Anxiety about the future is your mind trying to protect you by preparing for threats. Acknowledge the concern, then ask — what can I do today that my future self will thank me for? Focus on the controllable."),
    ("How do I improve my mindset?", "Mindset improves through deliberate practice. Challenge negative self-talk by asking if the thought is actually true. Read biographies of people who overcame difficulty. Surround yourself with people who push you to grow."),
    ("What is self-discipline?", "Self-discipline is choosing your long-term goals over short-term comfort. It is not about willpower but about systems. Remove temptations, create routines, track your habits, and make the right choice the easy choice."),
    ("How do I handle criticism?", "Separate the person giving criticism from the content. Ask yourself — is there truth in this? If yes, use it to improve. If it is unconstructive, let it go. Your growth matters more than others' opinions."),
    ("I feel like I am not good enough", "That feeling is lying to you. Everyone has moments of doubt, including the people you admire most. Your value is not determined by comparison to others. You are on your own timeline and your progress is real."),
    ("How do I set goals effectively?", "Set SMART goals — Specific, Measurable, Achievable, Relevant, and Time-bound. Write them down. Review them weekly. Goals written down are significantly more likely to be achieved than those kept only in your head."),
    ("How do I deal with toxic people?", "Limit your exposure where you can. Set clear boundaries and enforce them consistently. You cannot change others but you can choose how much access they have to your time, energy, and emotions."),
    ("What is growth mindset?", "A growth mindset, coined by Carol Dweck, is the belief that your abilities can be developed through dedication and hard work. The opposite is a fixed mindset that sees talent as innate and unchangeable."),
    ("How do I get out of a negative spiral?", "Notice the spiral without judging yourself for it. Then interrupt it physically — stand up, go outside, call someone, do twenty jumping jacks. Physical movement breaks mental loops faster than thinking does."),
    ("How do I build good habits?", "Habits form through repetition in context. Stack a new habit onto an existing one — after brushing teeth, read for ten minutes. Keep the habit small enough that you cannot say no. Never miss twice in a row."),
    ("What is emotional intelligence?", "Emotional intelligence is the ability to recognise, understand, and manage your own emotions and those of others. High EQ improves relationships, decision-making, and leadership more than IQ does in most situations."),
    ("How do I stay positive during hard times?", "Positivity during hard times is not pretending things are fine. It is choosing to see what is still working alongside what is broken. Gratitude journalling — writing three things you are grateful for daily — is scientifically proven to help."),
    ("I failed at something important", "Failing at something important means you were attempting something important. That already puts you ahead of people who play it safe. Analyse the failure clinically, forgive yourself quickly, and redirect the energy forward."),
    ("How do I deal with stress?", "Manage stress by identifying the source, distinguishing what you control from what you do not, and building recovery habits like exercise, sleep, and social connection. Stress without recovery leads to burnout."),
    ("What is imposter syndrome?", "Imposter syndrome is the persistent feeling that you are a fraud who will be exposed, despite evidence of your competence. It affects high achievers disproportionately. Knowing it has a name helps — you are not alone in feeling it."),
    ("How do I improve my focus?", "Improve focus by eliminating distractions — phone in another room, website blockers, dedicated work sessions. Use the Pomodoro technique — 25 minutes of focused work followed by a 5-minute break. Train focus like a muscle."),
    ("I feel like I am wasting my potential", "That awareness is valuable — it means your ambition is larger than your current actions. The gap between potential and performance is always closed one disciplined day at a time. Today is not too late to begin."),
    ("Give me a motivational quote about perseverance", "The man who moves a mountain begins by carrying away small stones. Every big achievement is just many small efforts repeated. Keep going."),
    ("Give me advice for young people in Nairobi", "Nairobi rewards the bold and the consistent. Develop a marketable skill, save before you spend, build genuine relationships, and remember that every successful person you admire started exactly where you are."),
    ("How do I achieve my dreams?", "Dreams become achievable when you break them into specific goals, goals into monthly targets, and targets into daily actions. Dream big, plan smart, and execute daily. The gap between dreaming and doing is discipline."),
    ("How do I stay focused on my goals?", "Write your goals down and read them every morning. Create an environment that supports your goals by removing obstacles. Track progress weekly. Review and adjust monthly. Consistency across time beats intensity in short bursts."),
    ("What do successful people do differently?", "Successful people take full ownership of their outcomes, invest continuously in learning, build strong networks, delay gratification for long-term gain, and persist through difficulties that make others quit."),
    ("How do I overcome fear?", "Fear shrinks when you act despite it. Identify specifically what you are afraid of. Visualise the worst realistic outcome and ask if you could survive it. Most fears are worse in imagination than reality. Do the scary thing."),
    ("What is the key to success?", "Success is the compound result of good decisions made consistently over time. It requires clear direction, continuous learning, disciplined execution, resilience through setbacks, and surrounding yourself with the right people."),
    ("How do I improve myself every day?", "Improve daily by reading for 20 minutes, exercising, reflecting on what went well and what to improve, practising your most important skill, and getting enough sleep. Small daily improvements compound into remarkable results."),
    ("What motivates successful entrepreneurs?", "Successful entrepreneurs are motivated by solving real problems, creating value for others, the challenge of building something from nothing, and financial freedom. Purpose-driven motivation outlasts money-driven motivation."),
    ("How do I become financially free?", "Financial freedom requires spending less than you earn, investing the difference consistently, avoiding high-interest debt, building multiple income streams, and growing your financial knowledge over time."),
    ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792,458 metres per second, or about 300,000 kilometres per second. Nothing in the universe can travel faster than light."),
    ("What is DNA?", "DNA or deoxyribonucleic acid is the molecule that carries genetic information in living organisms. It is shaped like a double helix and contains instructions for building and running every cell in your body."),
    ("What is the periodic table?", "The periodic table organises all known chemical elements by their atomic number and properties. Created by Dmitri Mendeleev in 1869, it currently contains 118 confirmed elements."),
    ("What is evolution?", "Evolution is the process by which species change over generations through natural selection. Individuals with traits better suited to their environment survive and reproduce more, passing those traits to offspring."),
    ("What is gravity?", "Gravity is a fundamental force that attracts objects with mass toward each other. Einstein described it as the curvature of spacetime caused by mass. On Earth it gives objects weight and keeps us on the ground."),
    ("What is the theory of relativity?", "Einstein's theory of relativity has two parts. Special relativity shows that space and time are relative to the observer. General relativity describes gravity as the curvature of spacetime caused by mass and energy."),
    ("What is quantum physics?", "Quantum physics describes the behaviour of matter and energy at atomic and subatomic scales. It reveals strange phenomena like particles being in multiple states simultaneously and instantaneous connections across distances."),
    ("What is the Big Bang theory?", "The Big Bang theory states that the universe began approximately 13.8 billion years ago as an extremely hot dense point that expanded rapidly. It is the most widely accepted explanation for the origin of the universe."),
    ("What is a black hole?", "A black hole is a region of spacetime where gravity is so strong that nothing — not even light — can escape. They form when massive stars collapse. The boundary of no return is called the event horizon."),
    ("What is photosynthesis?", "Photosynthesis is the process plants use to convert sunlight, carbon dioxide, and water into glucose and oxygen. It is the foundation of most food chains on Earth and produces the oxygen we breathe."),
    ("What is the human immune system?", "The immune system defends the body against pathogens like bacteria, viruses, and parasites. It includes physical barriers like skin, white blood cells that destroy invaders, and antibodies that target specific threats."),
    ("What is climate change?", "Climate change refers to long-term shifts in global temperatures and weather patterns. While natural factors contribute, since the 1800s human activities like burning fossil fuels have been the main driver."),
    ("What is the greenhouse effect?", "The greenhouse effect is when gases like carbon dioxide trap heat from the sun in the atmosphere rather than letting it escape to space. It naturally warms the Earth but excess greenhouse gases cause problematic warming."),
    ("What is renewable energy?", "Renewable energy comes from naturally replenishing sources like solar, wind, hydropower, and geothermal. Unlike fossil fuels, these sources do not run out and produce little or no greenhouse gas emissions."),
    ("What is nuclear energy?", "Nuclear energy is produced from splitting uranium atoms in a process called fission. A small amount of uranium produces enormous energy. Nuclear power plants generate electricity without direct carbon emissions."),
    ("What is artificial intelligence?", "Artificial intelligence is the simulation of human intelligence by machines. It enables computers to learn from data, recognise patterns, make decisions, and improve over time without being explicitly programmed for each task."),
    ("What is machine learning?", "Machine learning is a branch of AI where systems learn from data and improve without being explicitly programmed. It powers recommendation systems, image recognition, language models, and fraud detection."),
    ("What is deep learning?", "Deep learning uses artificial neural networks with many layers to learn complex patterns in large datasets. It powers image recognition, natural language processing, voice assistants, and self-driving car technology."),
    ("What is the internet of things?", "The Internet of Things or IoT refers to everyday objects connected to the internet and each other. Smart home devices, connected cars, industrial sensors, and health monitors are all part of IoT."),
    ("What is cybersecurity?", "Cybersecurity protects computer systems and networks from digital attacks, theft, and damage. It includes practices like strong passwords, encryption, firewalls, and regular software updates to prevent unauthorised access."),
    ("Who was Nelson Mandela?", "Nelson Mandela was a South African anti-apartheid activist who spent 27 years in prison before becoming South Africa's first Black president from 1994 to 1999. He became a global symbol of justice and reconciliation."),
    ("What was the Cold War?", "The Cold War was a geopolitical tension between the United States and Soviet Union from 1947 to 1991. It never became direct military conflict but shaped global politics through proxy wars, an arms race, and the space race."),
    ("What was the Berlin Wall?", "The Berlin Wall was a barrier built by East Germany in 1961 that divided communist East Berlin from democratic West Berlin. It became the symbol of the Cold War and fell on November 9, 1989, symbolising the end of communist Europe."),
    ("What was World War II?", "World War II lasted from 1939 to 1945 and involved most of the world's nations. It began with Germany invading Poland and ended with Allied victory over Germany and Japan. It caused over 70 million deaths."),
    ("What was the slave trade?", "The transatlantic slave trade forcibly transported approximately 12 million Africans to the Americas between the 16th and 19th centuries. It was one of history's greatest crimes and its effects continue to shape societies today."),
    ("Who was Julius Nyerere?", "Julius Nyerere was Tanzania's founding president from 1961 to 1985. He promoted African socialism called Ujamaa, led Pan-African movements, and is revered as the father of the nation. He was known as Mwalimu, meaning teacher."),
    ("What is colonialism?", "Colonialism is the practice of a powerful nation controlling and exploiting a weaker nation or territory. European powers colonised most of Africa, Asia, and the Americas from the 1400s to the 1900s, extracting resources and imposing their cultures."),
    ("What was apartheid?", "Apartheid was a system of racial segregation in South Africa from 1948 to 1994, enforced by the National Party government. It denied Black South Africans basic rights and freedoms based solely on race."),
    ("What is the African Union?", "The African Union is a continental organisation of 55 African states founded in 2002 to replace the Organisation of African Unity. It promotes unity, peace, development, and African interests on the global stage."),
    ("What is the United Nations?", "The United Nations is an international organisation founded in 1945 after World War II. Its primary goals are maintaining international peace and security, promoting human rights, and fostering economic development."),
    ("What is diabetes?", "Diabetes is a chronic condition where the body cannot properly regulate blood sugar. Type 1 is autoimmune, requiring insulin. Type 2 is linked to lifestyle and is managed through diet, exercise, and medication."),
    ("What is hypertension?", "Hypertension or high blood pressure is when blood pressure is consistently too high. It increases risk of heart disease and stroke. It is often called the silent killer because it has no symptoms but is very dangerous."),
    ("What is malaria?", "Malaria is a life-threatening disease caused by Plasmodium parasites transmitted through Anopheles mosquito bites. It is most prevalent in sub-Saharan Africa and kills hundreds of thousands annually, mostly children."),
    ("How is malaria prevented?", "Malaria is prevented by sleeping under insecticide-treated mosquito nets, using mosquito repellents, wearing long sleeves and trousers at dusk, eliminating standing water where mosquitoes breed, and taking prophylactic medication when travelling."),
    ("What is HIV/AIDS?", "HIV is a virus that attacks the immune system. Without treatment it can progress to AIDS where the immune system is severely compromised. Antiretroviral therapy can control HIV allowing people to live long healthy lives."),
    ("What is a vaccine?", "A vaccine trains the immune system to recognise and fight a specific pathogen without causing disease. It contains weakened or killed pathogens or proteins from them that trigger an immune response and create immunity."),
    ("What is mental health?", "Mental health encompasses emotional, psychological, and social wellbeing. It affects how we think, feel, and act. Mental health conditions like depression and anxiety are common, treatable, and nothing to be ashamed of."),
    ("What is depression?", "Depression is a mental health condition characterised by persistent sadness, loss of interest, fatigue, and feelings of hopelessness. It is not a weakness or character flaw. Professional treatment with therapy and medication is very effective."),
    ("What is anxiety?", "Anxiety is a mental health condition involving excessive worry and fear. It can cause physical symptoms like racing heart and sweating. CBT therapy and medication are effective treatments. Exercise and mindfulness also help significantly."),
    ("How much sleep do adults need?", "Adults need 7 to 9 hours of quality sleep per night for optimal health. Sleep deprivation increases risk of obesity, heart disease, diabetes, and mental health problems. Sleep is when the brain consolidates memories."),
    ("What is a balanced diet?", "A balanced diet includes carbohydrates for energy, proteins for building and repair, healthy fats, vitamins, minerals, and plenty of water. Eating diverse whole foods in appropriate portions supports long-term health."),
    ("Why is exercise important?", "Regular exercise reduces risk of heart disease, diabetes, and cancer, improves mental health, strengthens bones and muscles, boosts energy, improves sleep, and extends life expectancy. 150 minutes of moderate activity per week is recommended."),
    ("What is Afrobeats?", "Afrobeats is a contemporary African music genre that blends Nigerian and Ghanaian music traditions with hip-hop, R&B, and electronic music. Artists like Burna Boy, Wizkid, and Davido have taken it global."),
    ("What is Genge music?", "Genge is a Kenyan hip-hop subgenre that emerged in the early 2000s, blending hip-hop with traditional Kenyan rhythms and Sheng lyrics. Artists like Nonini and Jua Cali pioneered the genre."),
    ("What is Bongo Flava?", "Bongo Flava is a popular Tanzanian music genre that blends hip-hop, reggae, R&B, and traditional Tanzanian music. It is very popular across East Africa and uses Swahili lyrics."),
    ("What is Kapuka music?", "Kapuka is a style of Kenyan pop music that emerged in the 2000s, featuring upbeat rhythms, Sheng lyrics, and danceable beats. It represented a distinctly Nairobi urban sound."),
    ("What is football?", "Football, known as soccer in America, is the world's most popular sport with over 4 billion fans. Two teams of 11 players compete to score goals by getting a ball into the opponent's net."),
    ("What is the FIFA World Cup?", "The FIFA World Cup is the premier international football tournament held every four years. Thirty-two national teams compete for the trophy. Brazil has won the most titles with five championships."),
    ("What is the Premier League?", "The English Premier League is the top tier of English football and the most watched football league in the world. Twenty clubs compete from August to May. Teams like Manchester City, Arsenal, and Liverpool compete."),
    ("What is the Champions League?", "The UEFA Champions League is Europe's premier club football competition where top clubs from European leagues compete. Real Madrid has won the most titles."),
    ("What is the AFCON?", "AFCON or the Africa Cup of Nations is the premier football tournament for African national teams. It is organised by CAF and held every two years. Egypt has won the most titles with seven."),
    ("Who is Didier Drogba?", "Didier Drogba is a Ivorian football legend who played for Chelsea and became one of Africa's greatest footballers. He scored 65 goals for Chelsea in the Champions League and was beloved for his powerful and clinical finishing."),
    ("What is the human genome?", "The human genome is the complete set of genetic instructions in a human cell, containing about 3 billion base pairs of DNA encoding approximately 20,000 to 25,000 genes."),
    ("What is CRISPR?", "CRISPR is a revolutionary gene editing technology that allows scientists to precisely edit DNA sequences. It has enormous potential for treating genetic diseases, improving crops, and advancing biological research."),
    ("What is 5G?", "5G is the fifth generation of mobile network technology offering speeds up to 100 times faster than 4G, very low latency, and the ability to connect massive numbers of devices simultaneously. It enables IoT and smart cities."),
    ("What is augmented reality?", "Augmented reality or AR overlays digital content onto the real world through a screen or headset. Pokémon Go is a famous example. AR has applications in medicine, education, retail, and manufacturing."),
    ("What is virtual reality?", "Virtual reality or VR creates a fully immersive digital environment through a headset. It is used in gaming, training simulations, therapy, architecture, and education to create experiences impossible in the physical world."),
    ("What is the metaverse?", "The metaverse refers to a persistent, interconnected virtual world where people can interact, work, and play through digital avatars. It combines VR, AR, blockchain, and social media into one digital realm."),
    ("What is Python programming?", "Python is a versatile, beginner-friendly programming language known for its clean syntax. It is used for web development, data science, AI, automation, and scientific computing. It is one of the most popular languages worldwide."),
    ("What is an algorithm?", "An algorithm is a step-by-step set of instructions for solving a problem or completing a task. Algorithms are the foundation of all computer programs and are used in everything from search engines to GPS navigation."),
    ("What is data science?", "Data science combines statistics, programming, and domain knowledge to extract insights from large datasets. Data scientists use tools like Python, R, and machine learning to analyse data and inform decisions."),
    ("What is automation?", "Automation uses technology to perform tasks with minimal human intervention. It ranges from simple macros to complex robots and AI systems. Automation increases efficiency and consistency but also changes the nature of work."),
    ("What is the meaning of life?", "Philosophers have debated this for millennia. Aristotle said it is eudaimonia — flourishing through virtuous living. Albert Camus said we must find our own meaning in an absurd world. Many find meaning in relationships, purpose, and contribution."),
    ("What is stoicism?", "Stoicism is an ancient Greek philosophy teaching that we should focus only on what we control and accept with equanimity what we cannot. Key stoics include Epictetus, Marcus Aurelius, and Seneca. It is highly practical for modern life."),
    ("What is mindfulness?", "Mindfulness is the practice of paying deliberate attention to the present moment without judgment. It reduces stress, improves focus, and increases wellbeing. It is rooted in Buddhist meditation but has become mainstream in psychology."),
    ("What is critical thinking?", "Critical thinking is the ability to analyse information objectively, identify assumptions, evaluate evidence, and form well-reasoned conclusions. It is one of the most valuable skills in an age of misinformation."),
    ("What is empathy?", "Empathy is the ability to understand and share the feelings of another person. It is distinct from sympathy which is feeling sorry for someone. Empathy builds trust, improves relationships, and is essential for leadership."),
    ("What is integrity?", "Integrity means being honest and having strong moral principles, especially when no one is watching. A person of integrity does what they say, says what they mean, and acts consistently with their values."),
    ("What is resilience?", "Resilience is the ability to recover from difficulties and adapt to adversity. Resilient people do not avoid challenges — they develop strength through facing them. Resilience can be built through practice and perspective."),
    ("What is leadership?", "Leadership is the ability to inspire and guide others toward a shared goal. Great leaders communicate clearly, listen actively, empower others, lead by example, and take responsibility for outcomes."),
    ("What is time management?", "Time management is organising and planning how to divide your time between activities. Effective time management reduces stress, increases productivity, and creates space for what matters most."),
    ("What is creativity?", "Creativity is the ability to generate original ideas and see connections between unrelated things. It is not a rare gift but a skill that can be developed through curiosity, practice, exposure to diverse ideas, and willingness to experiment."),
    ("What is the Sicilian Najdorf?", "The Najdorf is the most popular variation of the Sicilian Defence, starting with 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6. It creates complex positions and was Bobby Fischer's favourite defence."),
    ("What is the Sicilian Dragon?", "The Sicilian Dragon features Black fianchettoing the king's bishop with g6 and Bg7, creating a powerful diagonal. White often attacks on the kingside while Black counterattacks on the queenside."),
    ("What is the Budapest Gambit?", "The Budapest Gambit is 1.d4 Nf6 2.c4 e5. Black sacrifices a pawn for rapid development and active piece play. It is a surprise weapon that can catch unprepared opponents off guard."),
    ("What is the Benko Gambit?", "The Benko Gambit is 1.d4 Nf6 2.c4 c5 3.d5 b5. Black sacrifices a pawn for long-term queenside pressure and open files. It gives Black an active game with ongoing compensation."),
    ("What is the Trompowsky Attack?", "The Trompowsky Attack is 1.d4 Nf6 2.Bg5. White pins the knight early to disrupt Black's normal development. It avoids many main lines and can lead to unique positions."),
    ("What is the Scotch Game?", "The Scotch Game is 1.e4 e5 2.Nf3 Nc6 3.d4. White immediately challenges the centre. It leads to open positions and was a favourite of Garry Kasparov at the world championship level."),
    ("What is the Vienna Game?", "The Vienna Game is 1.e4 e5 2.Nc3. White develops the knight before pushing f4. It is a solid opening with attacking possibilities and avoids the heavy theory of the Ruy Lopez."),
    ("What is the Petrov Defence?", "The Petrov Defence is 1.e4 e5 2.Nf3 Nf6. Black immediately counterattacks rather than defending e5. It leads to symmetrical solid positions and is known as one of the safest defences against 1.e4."),
    ("What is the Queen's Indian Defence?", "The Queen's Indian Defence is 1.d4 Nf6 2.c4 e6 3.Nf3 b6. Black fianchettoes the queen's bishop to control the centre indirectly. It is a solid hypermodern defence popular at all levels."),
    ("What is the Catalan Opening?", "The Catalan Opening is 1.d4 Nf6 2.c4 e6 3.g3. White fianchettoes the bishop to g2 where it exerts powerful pressure along the long diagonal. It is a positional weapon used by many world champions."),
    ("What is the Semi-Slav Defence?", "The Semi-Slav is 1.d4 d5 2.c4 c6 3.Nf3 Nf6 4.Nc3 e6. It is a solid defence where Black keeps the pawn structure flexible. The Marshall Gambit and Botvinnik System are sharp variations within it."),
    ("What is the Meran Variation?", "The Meran is a sharp variation of the Semi-Slav starting after 5.e3 Nbd7 6.Bd3 dxc4 7.Bxc4 b5. It leads to double-edged positions where both sides have active counterplay."),
    ("How should you think during a chess game?", "During a game, think systematically: identify threats, look for candidate moves, calculate consequences, and ask why your opponent made their last move. Develop good thinking habits by always checking your moves before playing them."),
    ("What is a chess tempo?", "Gaining a tempo means achieving your objective in one fewer move than expected. Losing a tempo wastes a move. In the opening, developing a piece while making a threat is gaining a tempo."),
    ("What is initiative in chess?", "The initiative means you are making threats your opponent must respond to. Having the initiative lets you control the flow of the game. Sacrificing material is sometimes worth it to seize a decisive initiative."),
    ("What is a minority attack in chess?", "A minority attack involves using fewer pawns to attack and undermine the opponent's pawn majority. In Queen's Gambit structures White often plays b4-b5 to create weaknesses in Black's queenside."),
    ("What is a weak square?", "A weak square is one that cannot be defended by pawns. Knights love weak squares because they cannot be driven away by pawns. Creating and occupying weak squares is a key strategic idea."),
    ("What is the concept of the centre in chess openings?", "Controlling the centre with pawns and pieces gives more space and mobility. Classic centre control uses e4-d4. Hypermodern openings control the centre from a distance with pieces and then attack the opponent's centre."),
    ("What is the Reti Opening?", "The Reti Opening is 1.Nf3 followed by c4 and g3. It is a hypermodern opening where White avoids occupying the centre immediately and instead controls it from the flanks."),
    ("What is the King's Indian Attack?", "The King's Indian Attack is a flexible system where White sets up with Nf3, g3, Bg2, d3, and e4 regardless of Black's setup. It is reliable because it requires little specific preparation against different defences."),
    ("What is a perpetual check?", "Perpetual check is when one player gives an endless series of checks that the opponent cannot escape. It results in a draw by repetition and is a common defensive resource in lost positions."),
    ("What is a chess arbiter?", "A chess arbiter or tournament director ensures that games are played according to the rules, resolves disputes, manages time controls, and maintains order during tournaments."),
    ("What is the Immortal Game?", "The Immortal Game was played by Adolf Anderssen against Lionel Kieseritzky in 1851. Anderssen sacrificed both rooks and a bishop before delivering checkmate with his remaining pieces. It is one of the most celebrated games in chess history."),
    ("What is the Evergreen Game?", "The Evergreen Game was another famous victory by Adolf Anderssen in 1852 featuring spectacular sacrifices. It remains one of the most analysed and admired attacking masterpieces in chess history."),
    ("Who was Paul Morphy?", "Paul Morphy was an American chess prodigy from New Orleans who dominated the chess world in the 1850s. He is considered by many to be the first unofficial world champion and a genius far ahead of his time."),
    ("Who was José Raúl Capablanca?", "Capablanca was a Cuban chess prodigy and World Champion from 1921 to 1927. He was known for his crystal-clear positional style, exceptional endgame technique, and rarely losing games."),
    ("Who was Alexander Alekhine?", "Alexander Alekhine was a Russian-French chess player and World Champion from 1927 to 1935 and 1937 to 1946. He was known for deeply combinational play and is considered one of the greatest attacking players ever."),
    ("Who was Tigran Petrosian?", "Tigran Petrosian was an Armenian World Champion from 1963 to 1969. Known as Iron Tigran, he was a defensive genius famous for prophylactic thinking and making it nearly impossible for opponents to attack him."),
    ("Who was Boris Spassky?", "Boris Spassky was a Soviet World Champion from 1969 to 1972. He is best known for his dramatic match against Bobby Fischer in Reykjavik 1972, the Match of the Century, which he lost."),
    ("What is the Polgar family?", "The Polgar sisters — Susan, Sofia, and Judit — were trained from childhood by their father Laszlo to prove that chess ability is made not born. Judit became the strongest female player in history, reaching world top 10."),
    ("What is Lamu?", "Lamu is a historic island town on Kenya's coast and a UNESCO World Heritage Site. It is one of the oldest and best-preserved Swahili settlements in East Africa, known for its unique architecture and culture."),
    ("What is the Maasai Mara?", "The Maasai Mara is Kenya's most famous wildlife reserve in southwestern Kenya. It is home to the Big Five and hosts the Great Wildebeest Migration. It is one of Africa's most celebrated safari destinations."),
    ("What is Naivasha?", "Lake Naivasha is a freshwater lake in the Rift Valley about 90km from Nairobi. It is known for its hippos, diverse birdlife, and Kenya's largest flower-growing industry along its shores."),
    ("What is Nakuru?", "Nakuru is the fourth largest city in Kenya and capital of Nakuru County. Lake Nakuru National Park nearby is famous for flamingos, rhinos, and leopards. The city is a major agricultural and industrial centre."),
    ("What is Eldoret?", "Eldoret is Kenya's fifth largest city in the Rift Valley. It is internationally famous as the home of Kenya's long-distance running champions. Many Olympic and world record holders train in and around Eldoret."),
    ("Why are Kenyans good at running?", "Kenya dominates long-distance running due to a combination of high altitude training in the Rift Valley, cultural tradition of running to school, lean body types, high red blood cell counts, and exceptional mental toughness."),
    ("Who is Eliud Kipchoge?", "Eliud Kipchoge is a Kenyan marathon runner widely considered the greatest of all time. He broke the two-hour marathon barrier in 2019 running 1:59:40, though under special conditions. He has won multiple Olympic gold medals."),
    ("What is Catherine Ndereba famous for?", "Catherine Ndereba is a Kenyan marathon runner nicknamed Catherine the Great. She won the Boston Marathon four times and two World Championship gold medals, inspiring generations of Kenyan female athletes."),
    ("What is the Kenyan school system?", "Kenya uses the CBC — Competency Based Curriculum — system. It replaced the 8-4-4 system. It focuses on practical skills and runs from early childhood through senior school over a 2-6-3-3 structure."),
    ("What is KCSE in Kenya?", "KCSE stands for Kenya Certificate of Secondary Education. It is the national exam taken at the end of Form 4, equivalent to A-levels. Results determine university admission and career paths."),
    ("What is KCPE in Kenya?", "KCPE stands for Kenya Certificate of Primary Education. It was the national exam at the end of Class 8 used for secondary school admission. Under CBC it is being phased out and replaced with new assessments."),
    ("What is University of Nairobi?", "The University of Nairobi is Kenya's oldest and largest public university, founded in 1956. It is ranked among the top universities in Africa and offers programmes across all major disciplines."),
    ("What is Strathmore University?", "Strathmore University is a private university in Nairobi known for its strong business, technology, and law programmes. It is affiliated with Opus Dei and emphasises professional and ethical education."),
    ("What is the Kenya Defence Forces?", "The Kenya Defence Forces, KDF, consists of the Kenya Army, Kenya Navy, and Kenya Air Force. They protect national sovereignty and also participate in UN peacekeeping missions across Africa."),
    ("What is Uhuru Kenyatta known for?", "Uhuru Kenyatta served as Kenya's fourth president from 2013 to 2022. He is the son of founding president Jomo Kenyatta. His presidency focused on infrastructure development including roads, railway, and the standard gauge railway."),
    ("What is the Standard Gauge Railway in Kenya?", "The Standard Gauge Railway, SGR, is a modern railway line connecting Mombasa to Naivasha via Nairobi. Built with Chinese financing, it replaced the century-old metre gauge and significantly reduced travel time."),
    ("What is Kenya Airways?", "Kenya Airways is Kenya's national carrier and a major African airline. Known as the Pride of Africa, it connects Kenya to destinations across Africa, Europe, Asia, and the Americas."),
    ("What is JKIA?", "JKIA stands for Jomo Kenyatta International Airport in Nairobi. It is Kenya's largest airport and the main hub for Kenya Airways and regional carriers. It handles millions of passengers and tonnes of cargo annually."),
    ("What is Moi International Airport?", "Moi International Airport is Mombasa's main airport, serving the coastal region. It handles both domestic and international flights and is the gateway for tourists visiting Kenya's coast."),
    ("What language do the Luo speak?", "The Luo speak Dholuo, a Nilotic language. The Luo are one of Kenya's largest ethnic groups, living mainly around Lake Victoria in western Kenya. They are known for their fishing culture and intellectual tradition."),
    ("What language do the Kikuyu speak?", "The Kikuyu speak Gikuyu, a Bantu language. The Kikuyu are Kenya's largest ethnic group, historically living in the central highlands around Mount Kenya. They have been central to Kenya's political and economic life."),
    ("What is Harambee in Kenya?", "Harambee means pulling together in Swahili and is Kenya's national motto. It represents the spirit of community self-help where people pool resources for a common purpose like building schools or hospitals."),
    ("What is Nyayo in Kenyan history?", "Nyayo means footsteps in Swahili. It was the motto of Kenya's second president Daniel arap Moi, meaning following in the footsteps of Jomo Kenyatta. Moi ruled Kenya from 1978 to 2002."),
    ("What is Wanjiku in Kenyan culture?", "Wanjiku is a common Kikuyu name but in Kenyan political discourse it represents the ordinary citizen, the common person on the street. Politicians often invoke Wanjiku to show concern for ordinary people."),
    ("What does Kama kawaida mean?", "Kama kawaida means as usual or the same as always in Swahili. It is a phrase used to describe a routine situation or normal state of affairs."),
    ("What does Sawa kabisa mean?", "Sawa kabisa means perfectly fine or absolutely okay in Swahili. Kabisa means completely or absolutely and emphasises the sawa. It is used to express full agreement or satisfaction."),
    ("What does Hiyo ni noma mean?", "Hiyo ni noma is Sheng for that is tough or that is difficult. Noma means something hard, painful, or impressive depending on context. It can also mean cool or impressive in certain contexts."),
    ("What does Umefika means in Swahili?", "Umefika means you have arrived in Swahili. Umefika lini means when did you arrive. Niliambia ufike mapema means I told you to arrive early."),
    ("What is ugali na sukuma?", "Ugali na sukuma wiki is Kenya's most iconic meal — firm maize porridge eaten with stir-fried collard greens. It is affordable, filling, and eaten by millions of Kenyans daily across all social classes."),
    ("What is mutura and how is it eaten?", "Mutura is a Kenyan sausage made from goat intestines stuffed with blood, meat, and spices then roasted over charcoal. It is sold by street vendors and eaten standing up, often with a sprinkle of salt and chilli."),
    ("What is a hedge fund?", "A hedge fund is a pooled investment fund that uses aggressive strategies including leverage, short selling, and derivatives to generate high returns. They are typically only available to wealthy accredited investors."),
    ("What is private equity?", "Private equity involves investing directly in private companies not listed on stock exchanges. PE firms buy companies, improve them, and sell them for a profit, typically over a 5 to 7 year period."),
    ("What is venture capital?", "Venture capital is funding provided to early-stage startups with high growth potential in exchange for equity. VC firms accept high risk expecting some investments to fail and a few to deliver exceptional returns."),
    ("What is an IPO?", "An IPO or Initial Public Offering is when a private company first offers shares to the public on a stock exchange. It allows the company to raise capital and gives early investors a chance to sell their stakes."),
    ("What is a stock split?", "A stock split increases the number of shares by dividing existing shares into multiple shares. A 2-for-1 split doubles the number of shares and halves the price. It makes shares more affordable without changing total value."),
    ("What is a rights issue?", "A rights issue is when a company offers existing shareholders the right to buy additional shares at a discounted price. Companies use it to raise capital. Shareholders who do not participate see their ownership percentage diluted."),
    ("What is margin trading?", "Margin trading involves borrowing money from a broker to buy more securities than you could with your own funds. It amplifies gains but also losses. If your position falls enough you face a margin call requiring more funds."),
    ("What is a futures contract?", "A futures contract is an agreement to buy or sell an asset at a predetermined price on a future date. It is used to hedge risk or speculate. Commodities, currencies, and indices all have futures markets."),
    ("What is an options contract?", "An options contract gives the buyer the right but not the obligation to buy or sell an asset at a set price before a certain date. Call options profit from price rises. Put options profit from price falls."),
    ("What is forex trading?", "Forex or foreign exchange trading involves buying and selling currency pairs to profit from exchange rate movements. It is the largest financial market in the world with over $6 trillion traded daily."),
    ("What is the difference between stocks and bonds?", "Stocks represent ownership in a company with higher risk and return potential. Bonds represent loans to companies or governments with lower risk and fixed returns. A balanced portfolio usually contains both."),
    ("What is inflation's effect on savings?", "Inflation erodes purchasing power — money saved in a bank account earning 3% when inflation is 7% loses real value. To preserve wealth savings must be invested in assets that outpace inflation over time."),
    ("What is the Rule of 72?", "The Rule of 72 is a quick formula to estimate how long it takes to double an investment. Divide 72 by the annual interest rate. At 8% annual return your money doubles in approximately 9 years."),
    ("What is asset allocation?", "Asset allocation is how you divide your investment portfolio among different asset classes like stocks, bonds, real estate, and cash. Your allocation should reflect your risk tolerance, time horizon, and financial goals."),
    ("What is rebalancing a portfolio?", "Portfolio rebalancing means periodically adjusting your holdings back to your target allocation. If stocks have grown and now represent too large a share you sell some and buy other assets to restore balance."),
    ("What is a bear trap in trading?", "A bear trap is when a declining price reverses sharply upward after appearing to break below a key support level. Traders who sold short get trapped and must buy back at higher prices to cover their losses."),
    ("What is a bull trap?", "A bull trap is when a rising price reverses sharply downward after appearing to break above resistance. Traders who bought expecting continued rises get trapped and incur losses as the price falls."),
    ("What is the difference between saving and investing?", "Saving keeps money safe and accessible, usually in low-return accounts. Investing puts money into assets with higher return potential but more risk. Both are important — save for short-term needs, invest for long-term wealth."),
    ("What is the Nairobi bourse?", "The Nairobi Securities Exchange, bourse, lists equities, bonds, and ETFs. Safaricom is the most actively traded stock accounting for a large proportion of daily turnover. The NSE 20 and NSE All Share Index track performance."),
    ("How does M-Pesa make money?", "M-Pesa earns revenue through transaction fees charged on transfers, withdrawals, and payments. It also earns float income by investing the pooled customer balances. It processes billions of shillings in transactions daily."),
    ("What is power factor?", "Power factor measures how efficiently electrical power is being used. A power factor of 1 is perfect. Low power factor means reactive power is wasted. Industrial users are penalised by KPLC for low power factor."),
    ("What is reactive power?", "Reactive power is the portion of electrical power that oscillates between the source and load without doing useful work. It is caused by inductive or capacitive loads like motors and transformers."),
    ("What is a capacitor bank?", "A capacitor bank is a group of capacitors used to correct power factor in industrial installations. By supplying reactive power locally it reduces the current drawn from the supply and lowers electricity bills."),
    ("What is earthing in a building?", "Earthing connects all metallic parts of electrical installations to the ground to prevent dangerous voltages accumulating. Types include TN-C-S, TT, and IT systems, each with different earthing arrangements."),
    ("What is a surge protector?", "A surge protector absorbs voltage spikes that could damage electronic equipment. It uses metal oxide varistors to clamp excess voltage. Essential for protecting computers and sensitive electronics especially in Kenya where power quality varies."),
    ("What is an AVR?", "An Automatic Voltage Regulator stabilises the output voltage despite input voltage fluctuations. It protects equipment from both high and low voltage. Very useful in Kenya where voltage fluctuations from KPLC are common."),
    ("What is a solar panel efficiency?", "Solar panel efficiency is the percentage of sunlight converted to electricity. Standard panels are 15-22% efficient. Higher efficiency panels cost more but generate more power from the same roof area."),
    ("What is a lithium battery?", "Lithium batteries offer high energy density, long cycle life, and fast charging. Lithium iron phosphate, LiFePO4, is the safest chemistry for solar storage systems, with thousands of charge cycles and no thermal runaway risk."),
    ("What is off-grid solar?", "An off-grid solar system operates independently from the utility grid. It consists of solar panels, batteries, a charge controller, and an inverter. It is ideal for rural Kenya where grid connection is unavailable or unreliable."),
    ("What is net metering?", "Net metering allows solar system owners to sell excess electricity back to the grid, offsetting their electricity bill. KPLC has introduced net metering policies for Kenyan solar system owners."),
    ("What is a three-phase motor?", "A three-phase induction motor runs on three-phase power and is the workhorse of industry. It is efficient, reliable, and requires minimal maintenance. It is used for pumps, compressors, conveyors, and industrial machines."),
    ("What is a VFD?", "A VFD or Variable Frequency Drive controls the speed of an electric motor by varying the frequency of the supply. It saves energy on fans, pumps, and conveyors by running them at reduced speed when full speed is not needed."),
    ("What is armoured cable?", "Armoured cable has a layer of steel wire or tape protection over the cable cores. It is used for underground installation and exposed areas where mechanical protection is needed. Common in Kenyan building and infrastructure projects."),
    ("What is cable sizing?", "Cable sizing selects the correct cross-sectional area of cable for a given load current. Too small a cable overheats causing insulation failure and fire risk. Tables in IEC 60364 guide selection based on installation method and load."),
    ("What is voltage drop in cables?", "Voltage drop occurs as current flows through a cable's resistance. Excessive voltage drop means equipment at the far end receives insufficient voltage. Cable size, length, and current all affect voltage drop."),
    ("What is IP rating for equipment?", "IP or Ingress Protection rating classifies how well equipment is protected against solid particles and liquids. IP65 means dust-tight and protected against water jets. It is important when selecting electrical equipment for outdoor use."),
    ("What is a PLC?", "A PLC or Programmable Logic Controller is an industrial computer used to control machinery and processes. It reads inputs from sensors, executes a program, and controls outputs like motors and valves. It is the brain of modern automation."),
    ("What is SCADA?", "SCADA stands for Supervisory Control and Data Acquisition. It is a system for remote monitoring and control of industrial processes like power plants, water treatment, and manufacturing facilities."),
    ("What is BMS in buildings?", "A Building Management System, BMS, monitors and controls a building's mechanical and electrical systems including HVAC, lighting, security, and access control from a central interface, improving efficiency and reducing energy use."),
    ("What is a changeover switch?", "A changeover switch transfers electrical supply between two sources, typically mains and generator. Manual changeover switches are operated by hand. Automatic transfer switches, ATS, switch automatically when the mains supply fails."),
    ("How do I ask for help?", "Asking for help is a sign of strength, not weakness. Be direct and specific about what you need. Choose someone you trust. Most people genuinely want to help and are honoured when you reach out to them."),
    ("What is burnout?", "Burnout is a state of chronic stress leading to physical and emotional exhaustion, cynicism, and reduced effectiveness. It develops gradually and requires genuine rest and lifestyle change to recover from, not just a short break."),
    ("How do I set boundaries?", "Setting boundaries means communicating clearly what you will and will not accept. Start small — say no to one request that does not align with your needs. Enforce boundaries consistently and without over-explaining or apologising."),
    ("What is toxic positivity?", "Toxic positivity is the insistence on maintaining a positive outlook even in genuinely difficult situations, dismissing legitimate pain. Genuine support acknowledges difficulty rather than bypassing it with forced optimism."),
    ("How do I forgive someone?", "Forgiveness is releasing the weight of resentment for your own peace, not excusing the other person. It does not require reconciliation or trust. Process the hurt fully, then consciously choose to release the need for revenge."),
    ("What is self-care?", "Self-care is any deliberate activity that maintains or improves your physical and mental health. It includes sleep, nutrition, exercise, and activities that restore your energy. It is not selfish but essential for sustained performance."),
    ("How do I deal with grief?", "Grief is not a problem to solve but a process to move through. There is no timeline and no right way. Allow yourself to feel without judgment. Seek support from others who have experienced loss. Be patient with yourself."),
    ("What is a toxic relationship?", "A toxic relationship consistently makes you feel worse about yourself, involves manipulation or disrespect, and drains more than it gives. Recognising the pattern is the first step. You deserve relationships that support your growth."),
    ("How do I improve my self-esteem?", "Self-esteem improves through action, not affirmations. Take on small challenges and succeed. Stop comparing yourself to others' highlight reels. Treat yourself with the same kindness you would offer a good friend."),
    ("What is journalling?", "Journalling is writing down your thoughts, feelings, and experiences regularly. Research shows it reduces anxiety, improves mood, helps process difficult emotions, and increases self-awareness. Even five minutes a day is beneficial."),
    ("What is the solar system?", "The solar system consists of the Sun and everything gravitationally bound to it — eight planets, their moons, dwarf planets, asteroids, comets, and other objects. Earth is the third planet from the Sun."),
    ("What is water made of?", "Water is made of two hydrogen atoms bonded to one oxygen atom, giving it the chemical formula H2O. It is the most essential compound for life on Earth and covers about 71 percent of the planet's surface."),
    ("What is the speed of sound?", "The speed of sound in air at sea level is approximately 343 metres per second, or about 1,235 kilometres per hour. Sound travels faster through denser media — faster in water and much faster through solids."),
    ("What causes rainbows?", "Rainbows form when sunlight enters water droplets, bends, reflects inside the droplet, and bends again as it exits, separating white light into its component colours. Red is always on the outside and violet on the inside."),
    ("What is the tallest mountain?", "Mount Everest in the Himalayas is the tallest mountain above sea level at 8,849 metres. However Mauna Kea in Hawaii is the tallest when measured from its base on the ocean floor."),
    ("What is the deepest ocean?", "The Pacific Ocean's Mariana Trench is the deepest point on Earth at approximately 11,034 metres below sea level. The pressure at that depth is over 1,000 times the pressure at sea level."),
    ("What is the Amazon rainforest?", "The Amazon rainforest in South America is the world's largest tropical rainforest covering about 5.5 million square kilometres. It produces 20 percent of the world's oxygen and is home to an extraordinary diversity of life."),
    ("What is the Sahara Desert?", "The Sahara is the world's largest hot desert covering about 9.2 million square kilometres across North Africa. Despite being mostly sand and rock it supports diverse plant and animal life adapted to extreme conditions."),
    ("What is the Great Wall of China?", "The Great Wall of China is a series of fortifications built across northern China over centuries to protect against nomadic invasions. It stretches over 21,000 kilometres and is one of the greatest architectural achievements in history."),
    ("What is the Roman Empire?", "The Roman Empire was one of the most powerful empires in history, at its peak controlling territory from Britain to Mesopotamia. It left an enduring legacy in law, language, architecture, and governance that shapes Western civilisation today."),
    ("What is democracy?", "Democracy is a system of government where citizens have the power to choose their leaders through free and fair elections. It protects rights and freedoms through checks and balances on government power."),
    ("What is capitalism?", "Capitalism is an economic system where production and distribution are privately owned and driven by profit in a free market. It has generated enormous wealth but also produces inequality if not regulated appropriately."),
    ("What is socialism?", "Socialism is an economic system where the means of production are collectively owned, either by the state or cooperatives. It prioritises equality and social welfare. Most modern economies mix capitalist and socialist elements."),
    ("What is globalisation?", "Globalisation is the increasing interconnection of economies, cultures, and peoples across the world. It has boosted trade, spread technology, and reduced poverty in many places while also creating new economic vulnerabilities and inequalities."),
    ("What is the United States of America?", "The USA is a federal republic of 50 states located in North America. It is the world's largest economy and a dominant global political, military, and cultural power. Its capital is Washington DC."),
    ("What is China's role in the world economy?", "China is the world's second largest economy and the largest trading nation. It manufactures a significant share of global goods, is a major lender to developing nations, and is increasingly competing with the US for technological leadership."),
    ("What is the European Union?", "The EU is a political and economic union of 27 European nations. Members share a single market and most use the euro currency. It promotes peace, democracy, and economic cooperation across Europe."),
    ("What is the difference between weather and climate?", "Weather is the short-term atmospheric conditions like temperature and rain on any given day. Climate is the long-term average of weather patterns over decades in a region. Climate change affects climate, which then shapes weather."),
    ("What is an ecosystem?", "An ecosystem is a community of living organisms interacting with each other and their physical environment. Ecosystems can be as small as a pond or as large as a tropical rainforest. They provide essential services like clean air and water."),
    ("What is biodiversity?", "Biodiversity is the variety of life on Earth across species, genes, and ecosystems. High biodiversity makes ecosystems resilient. Human activity is currently causing the sixth mass extinction with species disappearing at an alarming rate."),
    ("What is a social media algorithm?", "Social media algorithms analyse your behaviour to show you content most likely to keep you engaged. They prioritise emotionally charged content because it generates more interaction, which has significant effects on mental health and political discourse."),
    ("What is fake news?", "Fake news is false information presented as genuine news, designed to deceive or mislead. It spreads rapidly on social media. Critical media literacy — verifying sources, checking facts, reading beyond headlines — is essential protection."),
    ("What is data privacy?", "Data privacy is the right to control how your personal information is collected, used, and shared. Tech companies collect vast amounts of data. Privacy laws like GDPR in Europe aim to protect individuals' rights over their personal data."),
    ("What is a search engine?", "A search engine is a software system that indexes the web and retrieves results for user queries. Google, Bing, and DuckDuckGo are examples. Google uses complex algorithms to rank pages by relevance and authority."),
    ("What is email etiquette?", "Good email etiquette includes a clear subject line, concise body text, professional greeting and closing, responding within 24 hours, and avoiding reply-all unless everyone needs the information. Tone is easy to misread in text."),
    ("What is networking in a career context?", "Professional networking is building relationships with people in your field that can provide opportunities, advice, and referrals. Genuine networking focuses on giving value to others first, not just asking for favours."),
    ("What is a CV?", "A CV or Curriculum Vitae is a document summarising your education, work experience, skills, and achievements. In Kenya it should be 2-3 pages, tailored to the specific job, and accompanied by a strong cover letter."),
    ("What is an interview technique?", "Effective interview technique includes researching the company, preparing answers to common questions using the STAR method, asking thoughtful questions, dressing appropriately, arriving early, and following up with a thank-you message."),
    ("What is entrepreneurship?", "Entrepreneurship is identifying a problem or opportunity, creating a solution, and building a business around it while managing the risks involved. Kenya has a vibrant entrepreneurship ecosystem with many startups in Nairobi's Silicon Savannah."),
    ("What is Silicon Savannah?", "Silicon Savannah is the nickname for Nairobi's technology and innovation hub, particularly the Konza Technopolis development and the cluster of tech companies and startups in Nairobi. Kenya is recognised as Africa's leading tech hub."),
    ("What is a discovered check?", "A discovered check occurs when a piece moves and uncovers a check from a piece behind it. The moving piece can go almost anywhere since the check is given by the piece that was behind it."),
    ("What is a battery in chess?", "A battery is when two pieces of the same type are lined up on the same rank, file, or diagonal to combine their power. Two rooks on the same file or a queen and bishop on the same diagonal are classic batteries."),
    ("What is the concept of files in chess?", "Files are the eight vertical columns on a chess board labeled a through h. Rooks and queens are most powerful on open files with no pawns blocking them. Controlling an open file is a key strategic goal."),
    ("What is the concept of ranks in chess?", "Ranks are the eight horizontal rows on a chess board numbered 1 through 8. The seventh rank is particularly powerful for rooks because it puts pressure on the enemy's unmoved pawns and restricts the king."),
    ("What is the concept of diagonals in chess?", "Diagonals are lines of squares going corner to corner. Bishops and queens control diagonals. The long diagonals a1-h8 and a8-h1 are the most important because they span the entire board."),
    ("What is a fortress in chess?", "A chess fortress is a defensive setup that creates an impenetrable position for the defending side, often leading to a draw despite material disadvantage. The defending side builds a wall the attacker cannot breach."),
    ("What is an isolated queen's pawn?", "An isolated queen's pawn or IQP sits on the d-file with no friendly pawns on the c or e files. It is a structural weakness but gives the owner active piece play, open files, and the e5 outpost for compensation."),
    ("What does +/- mean in chess notation?", "In chess notation +/- means White has a decisive advantage. +/= means White has a slight advantage. = means equality. -/+ means Black has a decisive advantage and -/= means Black is slightly better."),
    ("What is castling queenside?", "Castling queenside, or long castling, moves the king to c1 and the rook to d1. It tucks the king away but the king is generally less safe than after kingside castling because it is further from the corner."),
    ("What is the concept of piece coordination?", "Piece coordination means having all your pieces working together toward a common goal. Uncoordinated pieces get in each other's way. Coordinated pieces multiply each other's effectiveness dramatically."),
    ("What is the principle of least active piece?", "When unsure what to do in chess, improve your least active piece. This principle by Silman helps direct thinking — find your most passive piece and give it a better square or role."),
    ("What is overprotection?", "Overprotection is Nimzowitsch's concept of defending important squares and pieces with more pieces than seem necessary. This prevents tactical shots and keeps the position stable."),
    ("What is a bishop pair?", "Having both bishops while your opponent has a bishop and knight or two knights is called the bishop pair. In open positions with long diagonals the two bishops together are often worth more than any other piece pair."),
    ("What is a colour complex weakness?", "A colour complex weakness occurs when all your pawns are on one colour, leaving the squares of the other colour permanently weak. If you also lack the bishop of that colour the weakness is severe."),
    ("What is the exchange in chess?", "Exchanging the exchange means trading a rook for a bishop or knight, giving up 5 points for 3 points. It can be correct when the minor piece is exceptionally active or the rook has no good open files."),
    ("What is Lasker's principle?", "Emanuel Lasker's principle states that when your opponent has a good plan, prevent it even at some cost. It is the basis of prophylactic thinking — seeing your opponent's intentions and thwarting them."),
    ("What is activity versus material in chess?", "Sometimes giving up material for activity is correct. A very active piece coordinates with other pieces and creates concrete threats that are worth more than extra pawns if those pawns are passive and useless."),
    ("What is the concept of pawn structure?", "Pawn structure describes the arrangement of pawns on the board. It largely determines strategy — weaknesses like isolated or doubled pawns are long-term liabilities that skilled players exploit relentlessly."),
    ("What is a stone wall formation?", "The Stone Wall is a pawn formation with pawns on d5, e6, f5, and c6. It is very solid but gives Black a bad light-squared bishop. White targets the e5 square and exploits the structural weaknesses."),
    ("What is the Maroczy Bind?", "The Maroczy Bind is a pawn structure with White pawns on c4 and e4, controlling d5 and restricting Black's counterplay. It is used against Sicilian and other defences to limit Black's space."),
    ("What is Eastleigh in Nairobi?", "Eastleigh is a suburb of Nairobi known as Little Mogadishu due to its large Somali community. It is one of Africa's most vibrant commercial districts with a massive wholesale textile and goods market."),
    ("What is Karen in Nairobi?", "Karen is an affluent suburb of Nairobi named after Karen Blixen, author of Out of Africa. It is known for its spacious properties, Karen Blixen Museum, the Giraffe Centre, and being home to embassies and expatriates."),
    ("What is Kiambu County?", "Kiambu County borders Nairobi to the north and northwest. It is one of Kenya's most densely populated and economically productive counties, known for tea, coffee, horticulture, and its large urban population working in Nairobi."),
    ("What is Machakos County?", "Machakos County is southeast of Nairobi, home to the Kamba people. It is known for wood carving, Konza Technopolis development, and Machakos People's Park. Its county headquarters is Machakos town."),
    ("What is Meru County?", "Meru County is on the eastern slopes of Mount Kenya. It is known for miraa, also called khat, which is a major cash crop and export, as well as tea, coffee, and diverse agriculture."),
    ("What is miraa?", "Miraa, also known as khat or murungi, is a plant chewed as a stimulant across East Africa and the Middle East. It is a significant cash crop in Meru County Kenya and an important export to Somalia and Djibouti."),
    ("What is Lamu Cultural Festival?", "The Lamu Cultural Festival is an annual celebration of Swahili culture on Lamu Island, featuring traditional dhow sailing races, donkey races, poetry, music, and crafts. It draws visitors from across the world."),
    ("What is Jambo Jet?", "Jambo Jet is a low-cost Kenyan airline owned by Kenya Airways. It operates domestic and regional routes across East Africa at affordable prices, making air travel more accessible to ordinary Kenyans."),
    ("What is Safaricom Football Premier League?", "The Football Kenya Federation Premier League is Kenya's top football division. Gor Mahia and AFC Leopards are the most popular clubs with the fiercest rivalry in Kenyan football, known as the Mashemeji Derby."),
    ("What is Gor Mahia?", "Gor Mahia is Kenya's most successful and popular football club, based in Nairobi. Named after a Luo medicine man, they have won the most Kenyan league titles and have a massive passionate fan base across Kenya."),
    ("What is AFC Leopards?", "AFC Leopards, or Ingwe, is one of Kenya's most popular football clubs and the main rival of Gor Mahia. Originally called Abaluhya FC they have a massive fan base especially among the Luhya community."),
    ("What does Twende mean?", "Twende means let us go in Swahili. It is the first person plural form of the verb kwenda meaning to go. Twende kazini means let us go to work."),
    ("What does Mzuri sana mean?", "Mzuri sana means very good or very beautiful in Swahili. Mzuri means good or beautiful and sana means very much. It is used as a positive response or to compliment something."),
    ("What does Furaha mean?", "Furaha means joy, happiness, or delight in Swahili. Ninafurahi sana means I am very happy. It is used in names and common expressions about positive emotions."),
    ("What does Upendo mean?", "Upendo means love in Swahili. Ninakupenda means I love you. Upendo ni nguvu means love is powerful. It is one of the most important words in Swahili."),
    ("What does Amani mean?", "Amani means peace in Swahili. It is a popular name for both boys and girls in Kenya and East Africa. Amani iwe nawe means may peace be with you."),
    ("What does Tumaini mean?", "Tumaini means hope in Swahili. Nina tumaini means I have hope. It is a common Kenyan name meaning the bearer is someone who brings hope."),
    ("What does Ujasiri mean?", "Ujasiri means courage or bravery in Swahili. Kuwa na ujasiri means to have courage. It is admired as a key virtue in Kenyan and East African culture."),
    ("What does Nguvu mean?", "Nguvu means strength or power in Swahili. Nguvu za Mungu means the power of God. Kuwa na nguvu means to be strong. It is used in many expressions and names."),
    ("What does Maarifa mean?", "Maarifa means knowledge or wisdom in Swahili. Maarifa ni nguvu means knowledge is power. It is a word that emphasises the cultural value placed on education and learning in Swahili culture."),
    ("What is risk reward ratio?", "The risk reward ratio compares potential loss to potential gain on a trade. A 1:3 ratio means you risk 1 to potentially gain 3. Consistently taking trades with favourable risk reward is key to long-term profitability."),
    ("What is a candlestick doji?", "A doji is a candlestick where the opening and closing price are almost equal, creating a cross shape. It signals market indecision and often precedes a reversal when it appears after a strong trend."),
    ("What is a hammer candlestick?", "A hammer candlestick has a small body at the top and a long lower wick. It signals potential reversal from a downtrend as buyers rejected lower prices and pushed back up strongly by the close."),
    ("What is an engulfing pattern?", "A bullish engulfing pattern is when a green candle completely covers the previous red candle. It signals strong buying pressure and a potential uptrend reversal. The bearish engulfing is the opposite at market tops."),
    ("What is a head and shoulders pattern?", "The head and shoulders is a reversal pattern with three peaks — a central higher peak flanked by two lower peaks. A break below the neckline signals the trend reversal is confirmed and a downtrend is beginning."),
    ("What is a double top?", "A double top is a bearish reversal pattern where price reaches the same high twice and fails to break above it. A break below the valley between the two tops confirms the pattern and signals a decline."),
    ("What is a double bottom?", "A double bottom is a bullish reversal pattern where price reaches the same low twice and bounces. A break above the peak between the two bottoms confirms the pattern and signals a potential uptrend."),
    ("What is a golden cross?", "A golden cross occurs when a shorter moving average like the 50-day crosses above a longer one like the 200-day. It is considered a bullish signal suggesting the start of a sustained uptrend."),
    ("What is a death cross?", "A death cross occurs when a shorter moving average crosses below a longer one. For example when the 50-day crosses below the 200-day it is a bearish signal suggesting a sustained downtrend may follow."),
    ("What is the Fibonacci retracement?", "Fibonacci retracement uses key ratios — 23.6%, 38.2%, 50%, 61.8%, and 78.6% — derived from the Fibonacci sequence to identify potential support and resistance levels after a price move."),
    ("What is the stochastic oscillator?", "The stochastic oscillator compares a closing price to its price range over a period. Readings above 80 indicate overbought conditions while readings below 20 indicate oversold, signalling potential reversals."),
    ("What is OBV in trading?", "On Balance Volume, OBV, is a momentum indicator that uses volume to predict price changes. Rising OBV when price is flat suggests accumulation and potential upside. Falling OBV suggests distribution."),
    ("What is a pip in forex?", "A pip is the smallest price movement in a forex pair, typically the fourth decimal place. For EUR/USD, a move from 1.1000 to 1.1001 is one pip. Profits and losses in forex are measured in pips."),
    ("What is a lot in forex trading?", "A standard lot in forex is 100,000 units of the base currency. A mini lot is 10,000 units and a micro lot is 1,000 units. Smaller lot sizes allow traders with limited capital to participate in forex markets."),
    ("What is a spread in trading?", "The spread is the difference between the bid price and the ask price. It is effectively the cost of entering a trade. Tighter spreads mean lower transaction costs. Liquid markets like EUR/USD have very tight spreads."),
    ("What is harmonics in electrical systems?", "Harmonics are voltage or current waveforms at multiples of the fundamental frequency. They are caused by non-linear loads like computers and VFDs. Harmonics cause heating, interference, and metering errors."),
    ("What is a power quality analyser?", "A power quality analyser measures electrical parameters like voltage, current, harmonics, power factor, and flicker. It helps identify power quality problems causing equipment failure or inefficiency."),
    ("What is a busbar?", "A busbar is a thick conductor, usually copper or aluminium, used in switchgear and distribution boards to carry large currents and distribute power to multiple circuits. It is rated in amperes and short-circuit current."),
    ("What is an interposing relay?", "An interposing relay converts between different signal levels in control circuits. For example it might receive a 24V DC signal from a PLC and use it to switch a 230V AC circuit for a larger load."),
    ("What is a soft starter?", "A soft starter gradually increases voltage to a motor on startup, reducing the large inrush current that can damage motors and cause voltage dips. It extends motor life and reduces mechanical stress on driven equipment."),
    ("What is dielectric strength?", "Dielectric strength is the maximum electric field a material can withstand without breakdown. It is a key property of cable insulation and determines how much voltage a cable can safely carry without arcing."),
    ("What is an elbow in cable installation?", "In cable management, an elbow is a 90-degree fitting used to change the direction of cable runs in conduit or trunking systems. Smooth elbows reduce friction when pulling cables through conduit."),
    ("What is cable dressing?", "Cable dressing is the practice of neatly routing and securing cables in a panel or cable tray. Well-dressed cables are easier to trace, maintain, and modify. Proper dressing also improves airflow and reduces interference."),
    ("What is a lux level?", "Lux is the unit of illuminance measuring light intensity per square metre. Office spaces typically require 300-500 lux. Detailed work areas may need 750 lux. DALI systems allow precise lux control across zones."),
    ("What is an occupancy sensor?", "An occupancy sensor detects human presence and automatically controls lighting or HVAC. PIR sensors detect movement via infrared. Ultrasonic sensors detect sound and motion. They save energy in offices and corridors."),
    ("What is a podcast?", "A podcast is an audio programme available for streaming or download. Podcasts cover every topic imaginable from news to comedy to education. They can be listened to while commuting, exercising, or doing other tasks."),
    ("What is the United Nations Sustainable Development Goals?", "The 17 SDGs are a universal call to action adopted by all UN members in 2015. They address poverty, inequality, climate change, peace, and justice, with a target date of 2030 for achieving them."),
    ("What is inflation in Kenya?", "Kenya's inflation is measured by the Kenya National Bureau of Statistics through the Consumer Price Index. Food prices and fuel are the biggest drivers. The Central Bank of Kenya targets inflation between 2.5% and 7.5%."),
    ("What is the Kenyan diaspora?", "The Kenyan diaspora refers to Kenyans living abroad, predominantly in the UK, USA, Canada, and Gulf states. They send billions of shillings home annually in remittances, making it one of Kenya's top foreign exchange earners."),
    ("What is mobile internet in Kenya?", "Mobile internet penetration in Kenya is among the highest in Africa. Safaricom's 4G network covers most urban areas and much of rural Kenya. Data bundles are affordable relative to income making Kenya a mobile-first economy."),
    ("What is a community health volunteer in Kenya?", "Community Health Volunteers are trained lay workers who provide basic health services and health education at household level in Kenya. They are a critical link between communities and the formal health system."),
    ("What is NHIF in Kenya?", "NHIF stands for National Hospital Insurance Fund, Kenya's public health insurance scheme. Members contribute monthly and can access medical services at accredited hospitals. It is being replaced by the Social Health Authority."),
    ("What is the Big 4 Agenda in Kenya?", "The Big Four Agenda was President Uhuru Kenyatta's development priorities including food security, universal healthcare, affordable housing, and growth of the manufacturing sector to create jobs and income."),
    ("What is Konza Technopolis?", "Konza Technopolis is Kenya's planned technology city being built 60km south of Nairobi. Designed as a smart city hub for Africa's tech sector it aims to attract investment, create jobs, and position Kenya as a continental tech leader."),
    ("What is the Africa Continental Free Trade Area?", "The AfCFTA is a free trade agreement signed by most African nations that creates a single continental market for goods and services. It aims to boost intra-African trade and reduce dependence on trade with other continents."),
    ("What is Pan-Africanism?", "Pan-Africanism is a political and cultural movement advocating the solidarity and unity of African peoples globally. It promotes African identity, opposes colonialism and racism, and seeks continental integration and development."),
    ("What is Ubuntu philosophy?", "Ubuntu is an African philosophical concept meaning I am because we are. It emphasises community, shared humanity, and the interdependence of people. It influenced Nelson Mandela's approach to reconciliation in South Africa."),
    ("What is the Kenyan startup ecosystem?", "Kenya's startup ecosystem is the most vibrant in East Africa with hundreds of startups in fintech, agritech, health tech, and logistics. Nairobi's iHub, Nailab, and various incubators support entrepreneurs with funding and mentorship."),
    ("What is a side hustle?", "A side hustle is income-generating work done alongside a main job. In Kenya where formal employment is scarce, side hustles are very common and important for financial resilience. Many Kenyans have multiple income streams."),
    ("What is the gig economy?", "The gig economy refers to work arrangements based on short-term contracts or freelance work rather than permanent employment. Platforms like Uber, Bolt, and Fiverr have grown the gig economy significantly in Kenya."),
    ("What is Bolt in Kenya?", "Bolt is a ride-hailing and delivery app popular in Nairobi and other Kenyan cities. It competes with Uber and is known for lower prices. Many Kenyans earn income by driving for Bolt."),
    ("What is Jumia?", "Jumia is Africa's largest e-commerce platform, listed on the New York Stock Exchange. It operates in multiple African countries including Kenya, selling electronics, fashion, and household goods online."),
    ("What is Kilimall?", "Kilimall is a Kenyan e-commerce platform popular for electronics, fashion, and daily deals. It competes with Jumia and serves customers across Kenya with both online orders and physical pickup points."),
    ("What is the history of Nairobi?", "Nairobi was founded in 1899 as a railway supply depot during the construction of the Uganda Railway. It grew rapidly under British colonial rule and became Kenya's capital after independence in 1963. Its name means cool water in Maasai."),
    ("What does Nairobi mean?", "Nairobi comes from the Maasai phrase enkare nyirobi meaning cool waters or place of cool waters, referring to the Nairobi River. The area was originally Maasai grazing land before the railway arrived."),
    ("What is a chess problem?", "A chess problem is a puzzle where you must find the best move or sequence of moves to achieve a goal, usually checkmate in a specified number of moves. Solving puzzles is one of the best ways to improve tactical vision."),
    ("What is a chess study?", "A chess study is a composed endgame position where you must find the only winning or drawing move sequence. Studies by composers like Troitzky and Rinck are famous for their elegance and instructive ideas."),
    ("What is the touch-move rule?", "The touch-move rule states that once you touch a piece you must move it. If you touch an opponent's piece you must capture it if possible. You can adjust pieces for neatness by saying j'adoube first."),
    ("What is flag fall in chess?", "Flag fall refers to a player's clock running out of time. In tournament play, losing on time means losing the game even if you have a winning position on the board — unless the position is such that checkmate is impossible."),
    ("What is a chess clock flag?", "In analogue chess clocks a small flag falls when the time expires, hence the term flag fall. Digital clocks simply show zero seconds. Winning on time is a legitimate result in competitive chess."),
    ("What is the concept of imbalances in chess?", "Imbalances are differences between the two sides — material, structure, piece activity, king safety, pawn majority. Understanding imbalances helps determine the correct plan. Silman's Imbalances is a famous teaching concept."),
    ("What is a chess annotation?", "Chess annotation uses symbols to evaluate moves: ! means excellent, !! brilliant, ? mistake, ?? blunder, !? interesting, and ?! dubious. These help readers understand the game without calculating every line."),
    ("What is PGN format?", "PGN or Portable Game Notation is the standard format for recording chess games digitally. It records moves in algebraic notation with metadata like player names, date, and result. Most chess software reads and writes PGN."),
    ("What is algebraic notation in chess?", "Algebraic notation records chess moves using the piece letter and destination square. Nf3 means knight to f3. e4 means pawn to e4. O-O means kingside castle. It is the universal standard for recording chess games."),
    ("What is the chess board setup?", "The chess board is 8x8 squares with alternating light and dark squares. White pieces start on ranks 1 and 2, Black on ranks 7 and 8. The queen goes on her own colour — white queen on d1, black queen on d8."),
    ("What is the opposition in chess?", "Opposition is when two kings face each other with exactly one square between them. The side NOT to move has the opposition, which is an advantage in pawn endgames. Direct, diagonal, and distant opposition are all important concepts."),
    ("What is a breakthrough in pawn endgames?", "A breakthrough is when one side sacrifices pawns to create a passed pawn. For example with pawns on e5, f5, g5 versus e6, f6, g6 White can play g6 fxg6 f6 exf6 e6 winning because the e-pawn will queen."),
    ("What is a rook endgame?", "Rook endgames are the most common endgames in chess. Basic knowledge of Lucena and Philidor positions is essential. The defender should use active rook play, give checks, and put the rook behind passed pawns."),
    ("What is king centralisation?", "In endgames the king becomes a powerful active piece. Centralising the king to e4, d4, e5, or d5 gives it maximum mobility and influence. One of the first principles of endgame technique is to activate the king."),
    ("What is the concept of zugzwang in endgames?", "Zugzwang means any move worsens your position. In king and pawn endgames triangulation is used to reach zugzwang. The side in zugzwang must either move the king away or advance a pawn creating weaknesses."),
    ("What is the concept of tempo in endgames?", "In endgames a tempo is critical. Gaining a tempo through triangulation or zugzwang can determine whether a pawn promotes or a king reaches a key square. Endgame technique often revolves around gaining or losing tempos."),
    ("What is a rook behind a passed pawn?", "Placing your rook behind your passed pawn supports its advance. The rook gains strength as the pawn advances. Placing the opponent's rook behind a passed pawn you are trying to stop is also excellent defensive technique."),
    ("What is the concept of shouldering in chess?", "Shouldering is using the king to block the opposing king from reaching a key area. In pawn races the king's ability to shoulder away the opponent determines the outcome."),
    ("What is a critical square in endgames?", "A critical square, also called a key square, is a square that if reached by the attacking king guarantees queening the pawn regardless of what the defending king does. For a pawn on e5, the critical squares are d7, e7, and f7."),
    ("What is the Nairobi Expressway?", "The Nairobi Expressway is a 27km elevated toll road completed in 2022 connecting Mlolongo to Westlands via JKIA. It significantly reduced commute times for those travelling through Nairobi's previously congested routes."),
    ("What is BRT in Nairobi?", "BRT stands for Bus Rapid Transit. Nairobi is developing a BRT system to provide dedicated bus lanes as a faster and more reliable public transport alternative to the chaotic matatu system."),
    ("What is the Nairobi Metro?", "Nairobi is planning a mass rapid transit system including a commuter rail network and metro lines to ease congestion. The Nairobi Commuter Rail operated by Kenya Railways currently runs limited services from the CBD."),
    ("What is Tom Mboya Street?", "Tom Mboya Street in Nairobi's CBD is named after the famous trade union leader and politician Tom Mboya who was assassinated in 1969. It is one of the main commercial streets in downtown Nairobi."),
    ("What is River Road in Nairobi?", "River Road is a famous street in Nairobi's CBD known for its vibrant, chaotic energy, affordable goods, and bus termini. It represents the working-class commercial heart of Nairobi."),
    ("What is Ngara in Nairobi?", "Ngara is a densely populated area in Nairobi located near the CBD. It is known for its market, hardware shops, and as a residential area for working-class Nairobi residents."),
    ("What is Kamukunji in Nairobi?", "Kamukunji is a constituency in Nairobi known for Jua Kali artisans who fabricate metalwork, furniture, and other goods in informal workshops. The Jua Kali sector is a vital part of Kenya's informal economy."),
    ("What is Jua Kali in Kenya?", "Jua Kali means hot sun in Swahili and refers to Kenya's informal sector artisans who work in the open air. They fabricate everything from metal goods to furniture and are a significant employer in Kenya's economy."),
    ("What does Tumia akili mean?", "Tumia akili means use your brain in Swahili. It is common advice encouraging someone to think carefully. Akili means intelligence or brain. Tumia means use."),
    ("What does Kaa rada mean in Sheng?", "Kaa rada is Sheng for be careful or stay alert. Rada means radar in English and by extension awareness. It is popular street language especially in Nairobi."),
    ("What is a margin call?", "A margin call occurs when your account falls below the required minimum balance due to losing positions. Your broker will demand you deposit more funds immediately or they will close your positions at a loss."),
    ("What is slippage in trading?", "Slippage is when a trade executes at a different price than expected, usually due to fast market movement or low liquidity. It increases real trading costs and is most problematic during news events."),
    ("What is a limit order?", "A limit order executes at a specified price or better. A buy limit order waits until the price falls to your target before buying. It gives price control but is not guaranteed to fill if price does not reach the level."),
    ("What is a market order?", "A market order executes immediately at the best available price. It guarantees execution but not price, meaning in fast markets you may get filled worse than expected due to slippage."),
    ("What is position sizing?", "Position sizing determines how much capital to risk on each trade. A common rule is risking no more than 1-2% of your total account on any single trade. Correct position sizing protects your account from catastrophic losses."),
    ("What is the Kelly Criterion?", "The Kelly Criterion is a formula for optimal position sizing based on your edge and win rate. It maximises long-term growth while avoiding overbetting that could wipe out your account."),
    ("What is a demo account?", "A demo account lets you practice trading with virtual money on real market conditions. All beginners should use a demo account for at least three months to develop skills and test strategies before risking real money."),
    ("What is a bear flag pattern?", "A bear flag is a continuation pattern in downtrends. After a sharp drop the price consolidates in a slight upward channel forming the flag. A break below the flag signals continuation of the downtrend."),
    ("What is a bull flag pattern?", "A bull flag is a continuation pattern in uptrends. After a sharp rise price consolidates in a slight downward channel. A breakout above the flag with volume signals continuation of the uptrend."),
    ("What is a pennant in trading?", "A pennant is a continuation pattern where price consolidates in a symmetrical triangle after a strong move. The breakout direction is usually the same as the preceding trend."),
    ("What is a neutral bar in electrical installation?", "A neutral bar is a busbar in a distribution board where all neutral conductors from individual circuits are connected. It provides the return path for current and is typically bonded to earth at the main distribution board."),
    ("What is a meter box?", "A meter box houses the electricity meter installed by KPLC to measure customer consumption in kWh. Prepaid meters have become standard in Kenya, requiring customers to purchase tokens before electricity is consumed."),
    ("What is a token meter?", "A token or prepaid meter requires customers to buy electricity credit in advance from KPLC agents or M-Pesa. When credit runs out electricity is automatically cut. This helps KPLC manage billing and reduce theft."),
    ("What is earth leakage?", "Earth leakage occurs when current flows to earth through an unintended path, such as through a person who touches a live wire. RCCBs detect earth leakage currents as small as 30mA and disconnect the supply in milliseconds."),
    ("What is a thermostat?", "A thermostat measures temperature and controls heating or cooling equipment to maintain a set point. Modern smart thermostats can be programmed and controlled remotely via phone, saving energy and improving comfort."),
    ("What is a PIR sensor?", "A PIR or Passive Infrared sensor detects movement by sensing changes in infrared radiation from warm bodies. It is used to automatically switch lights on when someone enters a room and off when the room is vacant."),
    ("What is a BMS controller?", "A BMS controller is the central processing unit of a Building Management System. It collects data from sensors, executes control strategies, and manages building systems to optimise comfort and energy efficiency."),
    ("What is KNX?", "KNX is a standard for intelligent building control, competing with DALI for smart building integration. KNX controls lighting, HVAC, blinds, access control, and energy management through a single system and wiring infrastructure."),
    ("What is Modbus?", "Modbus is a communication protocol widely used in industrial automation and building management. It allows devices like PLCs, inverters, and meters to communicate over RS-485 serial lines or Ethernet."),
    ("What is BACnet?", "BACnet is a data communication protocol for Building Automation and Control Networks. It allows different building systems from different manufacturers to communicate and be managed from a central platform."),
    ("What is a library?", "A library is a collection of resources — books, journals, digital media — available for use by the public or members. Libraries are free sources of knowledge and play a vital role in education and community development."),
    ("What is critical infrastructure?", "Critical infrastructure refers to systems and assets essential for national security, economy, and public health — power grids, water systems, transport networks, financial systems, and communications."),
    ("What is first aid?", "First aid is immediate help given to an injured or ill person before professional medical help arrives. Key skills include CPR, treating wounds, managing shock, and responding to allergic reactions and choking."),
    ("What is CPR?", "CPR or Cardiopulmonary Resuscitation is an emergency procedure involving chest compressions and rescue breaths to maintain blood circulation in someone whose heart has stopped. Early CPR significantly improves survival rates."),
    ("What is fire safety?", "Fire safety involves preventing fires and knowing how to respond if one occurs. Key practices include testing smoke alarms, knowing escape routes, keeping fire extinguishers accessible, and never leaving cooking unattended."),
    ("What is water conservation?", "Water conservation means using water efficiently to reduce waste. In Kenya where water scarcity is increasing this includes harvesting rainwater, fixing leaks promptly, using efficient appliances, and avoiding water waste."),
    ("What is waste management?", "Waste management involves collecting, treating, and disposing of waste responsibly. It includes reducing waste at source, reusing materials, recycling, composting organic waste, and safe disposal of hazardous materials."),
    ("What is plastic pollution?", "Plastic pollution is the accumulation of plastic waste in the environment. Kenya banned plastic bags in 2017, one of the toughest such bans globally. Ocean and land plastic pollution devastates ecosystems and wildlife."),
    ("What is deforestation?", "Deforestation is the clearing of forests for agriculture, logging, or development. Kenya has lost significant forest cover. Trees absorb carbon, regulate water cycles, and support biodiversity — their loss has serious consequences."),
    ("What is reforestation?", "Reforestation is the process of replanting trees in deforested areas. Kenya's Greenbelt Movement founded by Nobel laureate Wangari Maathai has planted over 51 million trees and inspired environmental activism globally."),
    ("Who was Wangari Maathai?", "Wangari Maathai was a Kenyan environmental activist and Nobel Peace Prize laureate. She founded the Green Belt Movement which mobilised women to plant millions of trees. She was the first African woman to win the Nobel Peace Prize in 2004."),
    ("What is the Paris Agreement?", "The Paris Agreement is a 2015 international treaty where nations committed to limiting global warming to 1.5-2 degrees Celsius above pre-industrial levels by reducing greenhouse gas emissions."),
    ("What is solar energy potential in Kenya?", "Kenya has excellent solar energy potential with most of the country receiving over 5 peak sun hours daily. The government has incentivised solar installation and many homes and businesses use solar to supplement or replace KPLC power."),
    ("What is geothermal energy in Kenya?", "Kenya is a global leader in geothermal energy with the Olkaria geothermal complex in the Rift Valley being one of the largest in the world. Geothermal provides over 40% of Kenya's electricity."),
    ("What is hydropower in Kenya?", "Hydropower has historically been Kenya's main electricity source using rivers like the Tana and Turkwel. Climate change has reduced reliability due to rainfall variability, pushing Kenya to diversify with geothermal and solar."),
    ("What is wind power in Kenya?", "Kenya has the Lake Turkana Wind Power project which at 310MW is one of Africa's largest wind farms. It harnesses strong winds in the Turkana corridor to generate clean electricity for the national grid."),
    ("What is food security?", "Food security exists when all people always have physical and economic access to sufficient safe and nutritious food. In Kenya food insecurity affects millions, particularly in arid northern counties vulnerable to drought."),
    ("What is agroforestry?", "Agroforestry combines trees with crops or livestock on the same land. It improves soil health, provides shade and fodder, sequesters carbon, and increases farm productivity. It is increasingly promoted in Kenya."),
    ("What is drip irrigation?", "Drip irrigation delivers water directly to plant roots through a network of pipes and emitters. It uses up to 50% less water than conventional irrigation and is ideal for horticultural farming in Kenya's drier regions."),
    ("What is greenhouse farming in Kenya?", "Greenhouse farming in Kenya produces vegetables, flowers, and fruits in controlled environments. It allows year-round production, protects crops from weather extremes, and is popular in small-scale commercial farming."),
    ("What is the Petroff Defence?", "The Petroff Defence is 1.e4 e5 2.Nf3 Nf6. Black mirrors White's knight move and fights for equality immediately. It is very solid and drawish at the highest levels."),
    ("What is a chess blunder?", "A blunder is a serious mistake that significantly worsens your position — typically losing a piece for nothing or allowing checkmate. Even grandmasters blunder under time pressure. Analysis helps you learn from them."),
    ("What is a chess tactic?", "A chess tactic is a sequence of moves that leads to a concrete advantage — winning material, delivering checkmate, or gaining a positional edge. Common tactics include forks, pins, skewers, and discovered attacks."),
    ("What is a chess strategy?", "Strategy in chess is the long-term plan for improving your position — where to place your pieces, which pawn structure to aim for, and how to exploit weaknesses. Tactics execute the strategy."),
    ("What is piece coordination?", "Piece coordination means your pieces work together harmoniously, supporting each other and controlling key squares. Poorly coordinated pieces are individually strong but collectively weak."),
    ("What is king safety in chess?", "King safety is one of the most important strategic factors. An exposed king is a liability because it can be attacked. Castling early and keeping pawns in front of the king are key safety measures."),
    ("Who was Emanuel Lasker?", "Emanuel Lasker was World Chess Champion for 27 years from 1894 to 1921, the longest reign in history. He was known for his psychological approach and ability to complicate positions."),
    ("What is the Keres Attack?", "The Keres Attack is an aggressive variation against the Sicilian Scheveningen where White plays an early g4 thrust. It was used by Paul Keres and attacks the kingside before Black can castle safely."),
    ("What is a desperado piece?", "A desperado is a piece that is going to be captured anyway, so it captures as much material as possible before being taken. It is a tactical motif often missed by beginners."),
    ("What does +/= mean in chess?", "+/= means White has a slight advantage. =/ means Black has a slight advantage. +/- means White has a clear advantage. -/+ means Black has a clear advantage. +- means White is winning."),
    ("What is a chess club?", "A chess club is an organised group where players meet regularly to play, study, and improve together. Clubs often participate in local and national leagues and are excellent for rapid improvement."),
    ("What is an endgame tablebases?", "Endgame tablebases are databases with perfect play for all positions with a small number of pieces. They have solved all positions with up to 7 pieces, revealing surprising wins and draws from seemingly lost positions."),
    ("What is the Nairobi National Park?", "Nairobi National Park is a unique wildlife reserve located just 7km from Nairobi's city centre. It is home to lions, rhinos, giraffes, and hundreds of bird species against a backdrop of city skyscrapers."),
    ("What is Langata in Nairobi?", "Langata is a suburb of Nairobi adjacent to Nairobi National Park. It is home to the Giraffe Centre, the David Sheldrick Wildlife Trust elephant orphanage, and Karen Blixen Museum."),
    ("What is the Giraffe Centre in Nairobi?", "The Giraffe Centre in Langata is a conservation project for the endangered Rothschild giraffe. Visitors can feed and interact with giraffes at close range. It is one of Nairobi's most popular tourist attractions."),
    ("What is Kenyatta University?", "Kenyatta University is one of Kenya's leading public universities located in Nairobi. It is known for education, arts, science, and business programmes. It was founded in 1972."),
    ("What is the Kenyan education system?", "Kenya follows an 8-4-4 system currently transitioning to CBC — Competency Based Curriculum. Education runs from early childhood through primary, secondary, and university levels. English and Swahili are teaching languages."),
    ("What is KNEC?", "KNEC stands for Kenya National Examinations Council. It is the body responsible for setting, administering, and marking national examinations including KCPE and KCSE."),
    ("What food is Mombasa famous for?", "Mombasa is famous for its Swahili coastal cuisine — biryani, pilau rice, coconut fish curry, mishkaki grilled meat skewers, mahamri coconut doughnuts, and fresh seafood from the Indian Ocean."),
    ("What is Lamu Island?", "Lamu Island is a UNESCO World Heritage Site off Kenya's north coast. It is one of the oldest and best-preserved Swahili settlements in East Africa, known for its donkeys, narrow streets, and stunning architecture."),
    ("What language do the Maasai speak?", "The Maasai speak Maa, a Nilotic language. They also commonly speak Swahili. Maasai communities straddle the Kenya-Tanzania border and maintain many of their traditional customs and pastoralist lifestyle."),
    ("What is a Harambee in Kenya?", "Harambee means let us pull together in Swahili. It is a Kenyan tradition of community fundraising where people contribute money for a common cause like education, medical bills, or funerals. It is deeply embedded in Kenyan culture."),
    ("What is nyayo in Kenya?", "Nyayo is Swahili for footsteps. It was the political philosophy of former President Daniel arap Moi who called Kenyans to follow in Kenyatta's footsteps. Nyayo is also the name of the national stadium in Nairobi."),
    ("What is the Kenyan flag?", "The Kenyan flag has three horizontal stripes — black representing the people, red for the blood shed for independence, and green for the land. A Maasai shield and spears in the centre represent the defence of freedom."),
    ("What does Wanjiku mean in Kenya?", "Wanjiku is a Kikuyu name but has become a political symbol representing the ordinary Kenyan citizen. Politicians often speak of what Wanjiku needs, meaning what the common person needs."),
    ("What is the Rift Valley in Kenya known for?", "The Rift Valley in Kenya is known for its stunning scenery, soda lakes, hot springs, the Maasai people, and producing world-class long-distance runners. Eldoret and Iten in Rift Valley are the running capitals of the world."),
    ("Why is Kenya famous for long-distance running?", "Kenya dominates long-distance running due to altitude training in areas like Iten at 2400m, a culture of running from childhood, lean body physiques, and deep competitive tradition. Kenyan runners have won Olympic medals for decades."),
    ("What is working capital?", "Working capital is the difference between current assets and current liabilities. It measures a business's ability to meet short-term obligations. Positive working capital means the business can pay its bills."),
    ("What is cash flow?", "Cash flow is the movement of money in and out of a business. Positive cash flow means more money coming in than going out. A business can be profitable but still fail due to poor cash flow management."),
    ("What is a balance sheet?", "A balance sheet is a financial statement showing a company's assets, liabilities, and shareholders' equity at a specific point in time. It shows what the company owns and owes."),
    ("What is an income statement?", "An income statement, also called a profit and loss statement, shows revenue, costs, and profit over a period. It tells you whether a business made or lost money during that period."),
    ("What is depreciation?", "Depreciation is the gradual reduction in value of an asset over time due to wear and use. It is recorded as an expense in accounting. For example, a vehicle bought for 1 million shillings depreciates each year."),
    ("What is equity in finance?", "Equity is the ownership interest in an asset after deducting liabilities. In a company it is shareholders' value. In property it is the value above what is owed on the mortgage."),
    ("What is a startup?", "A startup is a young company founded to develop a unique product or service, bring it to market, and scale it rapidly. Startups typically seek investment from venture capitalists or angel investors."),
    ("What is the USD/KES exchange rate?", "The USD/KES rate shows how many Kenyan shillings you get for one US dollar. The rate fluctuates based on trade flows, inflation, interest rates, and market sentiment. You can check current rates on Safaricom or bank apps."),
    ("What is the base lending rate in Kenya?", "The Central Bank Rate, CBR, is the benchmark interest rate set by the Central Bank of Kenya. Commercial banks base their lending rates on it. A lower CBR encourages borrowing and economic growth."),
    ("What is mobile lending in Kenya?", "Mobile lending apps like M-Shwari, Tala, Branch, and Fuliza provide instant loans via phone. They use alternative credit scoring based on mobile usage. Rates are high so borrow only what you can repay quickly."),
    ("What is a Chama in Kenya?", "A Chama is an informal savings group where members contribute a fixed amount regularly and take turns receiving the total pool. It is a popular way for Kenyans to save and invest collectively."),
    ("What is a PLC in automation?", "A PLC or Programmable Logic Controller is an industrial computer used to automate processes. It reads inputs from sensors, executes a program, and controls outputs like motors and valves. It is the brain of industrial automation."),
    ("What is Building Management System?", "A Building Management System or BMS is an intelligent control system managing a building's mechanical and electrical equipment — HVAC, lighting, power, fire detection, and security from a central interface."),
    ("What is HVAC?", "HVAC stands for Heating, Ventilation, and Air Conditioning. It is the system that controls temperature, humidity, and air quality in buildings. HVAC design is crucial for occupant comfort and energy efficiency."),
    ("What is a smart meter?", "A smart meter is an electronic device that records electricity consumption and communicates data to the utility. In Kenya KPLC has deployed smart prepaid meters that allow remote top-up via M-Pesa and real-time usage monitoring."),
    ("What is an oscilloscope?", "An oscilloscope is a test instrument that displays electrical signals as waveforms on a screen. It shows voltage over time and is essential for diagnosing electronic faults, checking signal quality, and testing circuits."),
    ("What is a multimeter?", "A multimeter measures voltage, current, and resistance in electrical circuits. It is the most essential tool for any electrician or electronics technician. Digital multimeters are accurate and easy to read."),
    ("What is a power factor?", "Power factor measures how efficiently electrical power is used. A power factor of 1 is perfect — all power drawn is useful. Low power factor means reactive power is wasted. Industries pay penalties for poor power factor."),
    ("What is a capacitor?", "A capacitor stores electrical charge and releases it when needed. In power systems capacitor banks correct poor power factor. In electronics they filter signals, smooth power supplies, and store energy."),
    ("What is an inductor?", "An inductor is a coil that stores energy in a magnetic field. It resists changes in current and is used in filters, transformers, and power converters. Inductors and capacitors together form resonant circuits."),
    ("What is a rectifier?", "A rectifier converts AC to DC. It uses diodes to allow current flow in one direction only. Rectifiers are found in all power supplies that convert mains AC power to the DC needed by electronics."),
    ("What is an inverter in electrical terms?", "An electrical inverter converts DC power to AC power. Solar inverters, UPS systems, and variable frequency drives all use inverter technology. They can generate pure sine wave AC suitable for all appliances."),
    ("What is earthing in electrical installation?", "Earthing provides a low resistance path to ground for fault currents, protecting people from electric shock. In Kenya TN-S and TN-C-S earthing systems are most common in commercial and domestic installations."),
    ("What is IP rating for electrical equipment?", "IP or Ingress Protection rating indicates how well equipment is protected against dust and water. IP54 means dust protected and splash resistant. IP67 means fully dust tight and can be immersed in water briefly."),
    ("What is an armoured cable?", "Armoured cable has a steel wire or tape armour layer protecting it from physical damage. It is used for underground installations, exposed runs, and industrial environments where cables may be damaged."),
    ("What is a substation?", "An electrical substation transforms voltage between transmission and distribution levels. KPLC operates many substations across Kenya converting high transmission voltages down to the 415V/240V supplied to customers."),
    ("What is gravity on the Moon?", "The Moon's gravity is about one-sixth of Earth's gravity. This is why astronauts could jump much higher on the Moon and why objects weigh much less there. The Moon's lower mass creates weaker gravitational pull."),
    ("What is the Milky Way?", "The Milky Way is the galaxy containing our Solar System. It is a spiral galaxy containing over 200 billion stars. Our Solar System is located on one of its outer spiral arms about 26,000 light years from the galactic centre."),
    ("What is a solar eclipse?", "A solar eclipse occurs when the Moon passes between the Earth and Sun, blocking sunlight. A total solar eclipse briefly turns day to night. Partial eclipses are more common. Never look directly at the sun during an eclipse."),
    ("What is a lunar eclipse?", "A lunar eclipse occurs when Earth passes between the Sun and Moon, casting Earth's shadow on the Moon. During a total lunar eclipse the Moon turns red — called a blood moon — due to sunlight bending through Earth's atmosphere."),
    ("What is the ozone layer?", "The ozone layer is a region of Earth's stratosphere containing high concentrations of ozone gas. It absorbs most of the Sun's harmful ultraviolet radiation. CFCs from aerosols and refrigerants damaged it but it is slowly recovering."),
    ("What is the water cycle?", "The water cycle is the continuous movement of water through evaporation from oceans and lakes, condensation into clouds, precipitation as rain or snow, runoff into rivers, and groundwater absorption, back to the ocean."),
    ("What causes earthquakes?", "Earthquakes are caused by sudden movement along fault lines in Earth's crust, usually due to tectonic plate movement. Kenya's Rift Valley is seismically active but large earthquakes are rare in East Africa."),
    ("What is a tsunami?", "A tsunami is a series of massive ocean waves caused by underwater earthquakes, volcanic eruptions, or landslides. They can travel across oceans at 800 km/h and cause catastrophic coastal flooding when they reach shore."),
    ("What is malnutrition?", "Malnutrition is a condition resulting from an inadequate or unbalanced diet. It includes undernutrition from insufficient calories and micronutrients and overnutrition from excessive calorie intake. Both forms impair health and development."),
    ("What is the human heart?", "The human heart is a muscular organ that pumps blood through the circulatory system. It beats about 100,000 times per day, circulating blood carrying oxygen and nutrients to all body tissues and removing carbon dioxide."),
    ("What is the nervous system?", "The nervous system is the body's communication network consisting of the brain, spinal cord, and nerves. It controls movement, thought, senses, and automatic functions like breathing and heartbeat."),
    ("What causes cancer?", "Cancer is caused by uncontrolled cell division due to genetic mutations. Causes include smoking, radiation, certain viruses, carcinogenic chemicals, and sometimes inherited genetic factors. Early detection dramatically improves outcomes."),
    ("What is the liver?", "The liver is the body's largest internal organ performing over 500 functions including filtering toxins from blood, producing bile for digestion, metabolising nutrients, synthesising proteins, and storing glycogen for energy."),
    ("What is antibiotics?", "Antibiotics are medicines that kill or inhibit bacteria. They are effective against bacterial infections but useless against viruses. Overuse has created antibiotic-resistant bacteria, which is a growing global health crisis."),
    ("What is blood pressure?", "Blood pressure measures the force of blood against artery walls. Normal is around 120/80 mmHg. The top number is systolic pressure during heartbeat, the bottom is diastolic pressure between beats."),
    ("What is cholesterol?", "Cholesterol is a fatty substance in blood. LDL — bad cholesterol — clogs arteries. HDL — good cholesterol — removes excess from arteries. High LDL increases heart disease risk. Diet and exercise manage cholesterol levels."),
    ("What is the digestive system?", "The digestive system breaks food into nutrients the body can absorb. It includes the mouth, oesophagus, stomach, small intestine where absorption occurs, large intestine, and exits via the rectum."),
    ("What is a calorie?", "A calorie is a unit of energy in food. The body uses calories for all functions from breathing to running. Consuming more calories than you burn leads to weight gain. An average adult needs around 2000 calories daily."),
    ("What can you do?", "I can answer questions, tell jokes, check weather, help with chess analysis, give financial information, explain electrical and IT concepts, motivate you, tell stories, set reminders, do math, and much more. Just ask!"),
    ("What is your name?", "I am Festoes — your personal AI assistant built right here in Nairobi. I am powered by Gemini AI, trained neural networks, and a whole lot of Kenyan spirit. How can I help you today?"),
    ("Who made you?", "I was built by Alvine using Python, Tkinter, neural networks, and the Gemini AI API. I am a custom AI assistant designed to work on Android using Pydroid 3. I keep getting smarter with every conversation."),
    ("What time is it?", "I can check the current time for you! Ask me 'what is the time' and I will tell you the exact time on your device."),
    ("Tell me a fact about Africa", "Africa is the world's second largest continent covering about 30 million square kilometres. It contains 54 countries, over 2000 languages, and is the birthplace of humanity. The Nile is the world's longest river and flows through Africa."),
    ("Tell me a fact about technology", "The first computer bug was an actual bug — a moth found trapped in a relay of the Harvard Mark II computer in 1947. Grace Hopper's team taped it in the logbook noting they had debugged the machine. The term stuck."),
    ("What is a good book to read?", "Some excellent books worth reading are Atomic Habits by James Clear for building good habits, Think and Grow Rich by Napoleon Hill for mindset and success, Things Fall Apart by Chinua Achebe for African literature, and The Alchemist by Paulo Coelho for inspiration."),
    ("Give me a fun fact about Kenya", "Kenya is the only country in the world where you can see Mount Kenya — Africa's second highest peak — and the Indian Ocean coast in the same day. The country spans from near sea level to 5199 metres elevation."),
    ("What is the best programming language?", "The best programming language depends on your goal. Python is best for AI, data science, and beginners. JavaScript dominates web development. Java and Kotlin for Android. C and C++ for systems programming. Python is the most versatile starting point."),
    ("How do I sleep better?", "Improve sleep by keeping a consistent sleep schedule, avoiding screens an hour before bed, keeping your room cool and dark, avoiding caffeine after 2pm, and doing light exercise during the day. Good sleep transforms everything."),
    ("How do I reduce screen time?", "Reduce screen time by setting app limits on your phone, keeping your phone out of the bedroom, scheduling phone-free hours, replacing scrolling with a physical activity or book, and turning off non-essential notifications."),
    ("How do I manage anger?", "Manage anger by pausing before reacting — even counting to ten helps. Identify the real source of anger which is often not the immediate trigger. Exercise burns off the physical energy of anger. Talk to someone you trust."),
    ("How do I be more productive?", "Boost productivity by working in focused blocks with breaks, tackling your most important task first when energy is highest, saying no to low-priority requests, planning tomorrow before you sleep, and protecting your mornings."),
    ("How do I make friends as an adult?", "Making friends as an adult requires intentional effort. Join groups around your interests — chess clubs, gym classes, professional networks. Be consistent in showing up. Initiate plans. Friendship grows through repeated contact over time."),
    ("How do I deal with rejection?", "Rejection stings but is not a reflection of your worth. Everyone faces rejection. Process the feeling, resist the urge to catastrophise, and remember that rejection often redirects you toward something better suited to you."),
    ("How do I improve my relationships?", "Improve relationships by listening more than you speak, expressing appreciation regularly, addressing issues directly and kindly rather than letting resentment build, spending quality time, and being the person you want others to be for you."),
    ("What is the fastest possible checkmate?", "Fool's Mate is the fastest checkmate, delivered in just two moves: 1.f3 e5 2.g4?? Qh4#. It can only happen if White makes both terrible moves. It is the shortest possible chess game ending in checkmate."),
    ("What is Scholar's Mate?", "Scholar's Mate checkmates in four moves: 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? 4.Qxf7#. It targets the weak f7 square. Beginners should know it to avoid falling for it."),
    ("How do you avoid Scholar's Mate?", "Defend against Scholar's Mate by developing your knight to f6 to block the queen's attack, or play g6 to chase the queen away. Always defend the f7 square in the opening as a general principle."),
    ("What is a chess puzzle?", "A chess puzzle presents a position where you must find the best move or sequence of moves to achieve a specific goal — usually checkmate or winning material. Solving puzzles daily is the fastest way to improve your tactics."),
    ("How do you win a rook endgame?", "Win a rook endgame by activating your king aggressively, creating a passed pawn, cutting off the enemy king with your rook, and using the Lucena technique once your pawn advances far enough."),
    ("What is opposition in chess?", "Opposition is when two kings face each other with one square between them. The side not to move holds the opposition — a key advantage in pawn endgames for controlling key squares and promoting pawns."),
    ("What is the Dragon Variation?", "The Dragon is a sharp Sicilian line where Black fianchettoes the bishop on g7: ...g6 and ...Bg7. White often castles queenside and launches a kingside attack with g4-h4-h5, creating extremely sharp play."),
    ("What is the Sicilian Scheveningen?", "The Scheveningen is a Sicilian setup where Black plays ...e6 and ...d6 creating a small centre. It is solid and flexible, used by many top players. The key is piece coordination and counterplay on both wings."),
    ("What is the Marshall Attack?", "The Marshall Attack is an aggressive gambit in the Ruy Lopez where Black sacrifices a pawn with 8...d5. It leads to a ferocious kingside attack. Frank Marshall prepared it as a secret weapon against Capablanca."),
    ("What is Diani Beach?", "Diani Beach is a stunning white sand beach south of Mombasa, consistently rated among Africa's best beaches. It has clear turquoise water, coral reefs for snorkelling, luxury resorts, and friendly colobus monkeys in the trees."),
    ("What is Malindi?", "Malindi is a coastal town north of Mombasa with beautiful beaches, a marine national park with excellent snorkelling, historical Portuguese ruins, and a large Italian expatriate community giving it a unique character."),
    ("What is Watamu?", "Watamu is a small coastal resort town near Malindi known for its pristine beaches, marine national park with sea turtles and reef fish, water sports, and laid-back atmosphere."),
    ("What is Hell's Gate National Park?", "Hell's Gate National Park near Naivasha is unique because visitors can walk and cycle through it without a vehicle. It features dramatic gorges, geothermal features, and inspired the landscape in Disney's Lion King."),
    ("What is Lake Naivasha?", "Lake Naivasha is a freshwater lake in the Rift Valley about 90km from Nairobi. It is known for hippos, abundant birdlife, boat rides, cycling to Crescent Island, and nearby Hell's Gate National Park."),
    ("What is Samburu National Reserve?", "Samburu National Reserve in northern Kenya is known for wildlife found nowhere else in Kenya — the Grevy's zebra, reticulated giraffe, Somali ostrich, and Beisa oryx — collectively called the Samburu Special Five."),
    ("What is the Aberdare National Park?", "The Aberdare National Park is a mountain forest park near Mount Kenya. It is home to elephants, black leopards, rare bongo antelopes, and tree hotels like Treetops where guests watch wildlife at night."),
    ("What is a safari?", "A safari is a wildlife viewing trip, originally meaning journey in Swahili. Kenya's safaris are world-famous. Game drives in open vehicles at dawn and dusk offer the best chances of seeing lions, elephants, and leopards in the wild."),
    ("What is the Kenya coast?", "Kenya's coast stretches 536km along the Indian Ocean. It blends beautiful beaches, coral reefs, Swahili Arab architecture, and rich history. Mombasa is the main city. The coast has a distinct culture different from inland Kenya."),
    ("What is Fort Jesus in Mombasa?", "Fort Jesus is a Portuguese fort built in 1593 on Mombasa's Old Town coastline. It is a UNESCO World Heritage Site and museum documenting the history of trade and conflict on the East African coast."),
    ("What is a mutual fund?", "A mutual fund pools money from many investors to buy a diversified portfolio of stocks, bonds, or other assets managed by professional fund managers. It gives small investors access to diversified investments."),
    ("What is a credit score?", "A credit score is a numerical rating of your creditworthiness based on your borrowing and repayment history. In Kenya banks and mobile lenders use credit scores from CRBs — Credit Reference Bureaus — to decide loan eligibility."),
    ("What is CRB in Kenya?", "CRB stands for Credit Reference Bureau. In Kenya, CRBs collect and share credit information about borrowers. Being listed negatively on a CRB due to loan default prevents you from accessing credit from most lenders."),
    ("What is a budget?", "A budget is a financial plan that allocates income to spending categories over a period. The 50/30/20 rule — 50% needs, 30% wants, 20% savings — is a popular budgeting framework for personal finance."),
    ("What is an emergency fund?", "An emergency fund is savings set aside for unexpected expenses like job loss, medical bills, or urgent repairs. Financial advisors recommend 3 to 6 months of living expenses in an accessible savings account."),
    ("What is financial literacy?", "Financial literacy is understanding how money works — earning, spending, saving, investing, and protecting it. Better financial literacy leads to better decisions and long-term financial wellbeing."),
    ("What is interest rate?", "An interest rate is the cost of borrowing money expressed as a percentage. When you borrow you pay interest to the lender. When you save, the bank pays you interest. Central bank rates influence all other interest rates."),
    ("What is a credit card?", "A credit card lets you borrow money up to a limit for purchases. If you repay the full balance monthly you pay no interest. If you carry a balance, interest — often 20-40% annually — accumulates quickly."),
    ("What is an API?", "An API or Application Programming Interface allows different software applications to communicate with each other. When you use a weather app, it calls a weather service API to get the forecast data."),
    ("What is JSON?", "JSON or JavaScript Object Notation is a lightweight data format used to transmit data between servers and applications. It uses key-value pairs and is human-readable. Most modern APIs return data in JSON format."),
    ("What is HTML?", "HTML or HyperText Markup Language is the standard language for creating web pages. It defines the structure and content of web pages using tags. Every website you visit is built with HTML."),
    ("What is CSS?", "CSS or Cascading Style Sheets controls the appearance of HTML elements — colours, fonts, layout, and spacing. It separates presentation from content and enables responsive design for different screen sizes."),
    ("What is JavaScript?", "JavaScript is the programming language of the web. It runs in browsers and makes websites interactive — animations, form validation, dynamic content updates. Node.js allows JavaScript to run on servers too."),
    ("What is a database?", "A database is an organised collection of structured data stored electronically. MySQL, PostgreSQL, and SQLite are relational databases. MongoDB is a popular non-relational database. Most applications use databases to store data."),
    ("What is SQL?", "SQL or Structured Query Language is used to communicate with relational databases. It allows you to create, read, update, and delete data. SELECT, INSERT, UPDATE, and DELETE are the fundamental SQL commands."),
    ("What is cybersecurity threat?", "Common cybersecurity threats include phishing attacks that steal credentials, ransomware that encrypts your files, malware that damages systems, man-in-the-middle attacks, and social engineering that manipulates people."),
    ("What is two-factor authentication?", "Two-factor authentication adds a second verification step beyond a password — usually a code sent to your phone. Even if your password is stolen, attackers cannot access your account without the second factor."),
    ("What is a strong password?", "A strong password is at least 12 characters long and combines uppercase letters, lowercase letters, numbers, and symbols. Use a different password for each account. A password manager helps you manage many strong passwords."),
    ("What is cloud storage?", "Cloud storage keeps your files on remote servers accessible via the internet from any device. Google Drive, Dropbox, and iCloud are popular services. Cloud storage provides backup, sharing, and cross-device access."),
    ("What is open source software?", "Open source software has freely available source code that anyone can view, modify, and distribute. Linux, Python, Firefox, and Android are open source. Open source fosters collaboration and innovation in technology."),
    ("What is version control?", "Version control tracks changes to code over time, allowing teams to collaborate and revert to previous versions if needed. Git is the most popular version control system. GitHub hosts millions of Git repositories."),
    ("What is agile development?", "Agile development is an iterative software development approach that delivers working software in short cycles called sprints. It prioritises collaboration, flexibility, and customer feedback over rigid upfront planning."),
    ("What is a software bug?", "A software bug is an error or flaw in a program that causes it to behave unexpectedly. Bugs range from minor display issues to critical security vulnerabilities. Finding and fixing bugs is called debugging."),
    ("What is the best chess opening?", "The best opening depends on your style. 1.e4 leads to open attacking games. 1.d4 leads to more positional games. For beginners, mastering the principles of development, centre control, and king safety matters more than specific openings."),
    ("How do I get better at chess fast?", "Improve fast by solving at least 10 tactics puzzles daily on Chess.com or Lichess, analysing your own games after each one, and playing longer time controls rather than bullet chess. Tactics are where most games are won and lost."),
    ("What chess app should I use?", "Lichess is completely free with no ads and has excellent puzzles and analysis. Chess.com has a polished interface and large community. Both are excellent. For analysis, Stockfish is available free on both platforms."),
    ("What is Lichess?", "Lichess is a free, open-source chess server with no ads, no subscriptions, and full access to all features. It has puzzles, analysis, lessons, and a large global community. It is run as a non-profit and is beloved by chess players."),
    ("What is Chess.com?", "Chess.com is the world's largest online chess platform with millions of daily users. It offers puzzles, lessons, tournaments, game analysis, and a premium subscription for advanced features."),
    ("Should I learn chess openings or endgames first?", "Learn endgames first. Most beginners never reach complex endgames but the principles — king activity, passed pawns, opposition — apply throughout the game. Once you understand endgames, openings make much more sense strategically."),
    ("How long does it take to become good at chess?", "With consistent study and practice, you can reach club-level play of around 1400-1600 ELO in 6-12 months. Reaching master level of 2200+ typically takes years of dedicated study. But every level brings more enjoyment."),
    ("What is a chess rating?", "A chess rating is a number representing your playing strength. Beginners start at 800-1000. Intermediate club players are 1200-1600. Advanced players are 1800-2000. Masters are 2200+. Grandmasters are 2500+."),
    ("What equipment do I need for chess?", "You need a chess set and board for over-the-board play. For online play, just a phone or computer and a free account on Lichess or Chess.com. A chess clock is needed for timed over-the-board games."),
    ("What hospitals are in Nairobi?", "Major hospitals in Nairobi include Kenyatta National Hospital, Nairobi Hospital, Aga Khan Hospital, MP Shah Hospital, Gertrude's Children's Hospital, and Karen Hospital. Private hospitals generally have shorter wait times."),
    ("What is the health system in Kenya?", "Kenya's health system has six levels from community health volunteers at Level 1 through dispensaries, health centres, county hospitals, national referral hospitals, and research institutions at Level 6."),
    ("What vaccines are available in Kenya?", "Kenya's expanded immunisation programme provides free vaccines for tuberculosis, polio, hepatitis B, diphtheria, tetanus, measles, pneumonia, rotavirus, and meningitis to all children at public health facilities."),
    ("What is traditional medicine in Kenya?", "Traditional medicine in Kenya involves herbalists, faith healers, and traditional birth attendants using local plants and practices. Many Kenyans combine traditional medicine with modern healthcare. The Ministry of Health is working to regulate the sector."),
]

PRETRAIN_DATA = PRETRAIN_DATA + PRETRAIN_DATA_V9

def pretrain_networks():
    """Train all 320 networks on the full pre-training dataset."""
    global AI_NETWORKS, MICRO_VOCAB

    print("Pre-training on", len(PRETRAIN_DATA), "examples...")

    for user_text, response_text in PRETRAIN_DATA:
        # Expand vocab
        for word in (user_text + " " + response_text).lower().split():
            if len(word) >= 3 and word not in MICRO_VOCAB and word.isalpha() and len(MICRO_VOCAB) < 300:
                MICRO_VOCAB.append(word)

    # Train networks on every pair (every 2nd network to keep startup fast)
    net_names = list(AI_NETWORKS.keys())
    for user_text, response_text in PRETRAIN_DATA:
        quality = min(1.0, len(response_text.split()) / 30)
        vec = micro_vectorize(user_text)

        for name in net_names[::5]:  # every 5th network — 200 networks, much faster
            net = AI_NETWORKS[name]
            W1,W2,W3,b1,b2,b3 = net
            if vec.shape[1] != W1.shape[0]:
                continue
            h1  = np.maximum(0, vec.dot(W1)+b1)
            h2  = np.maximum(0, h1.dot(W2)+b2)
            raw = h2.dot(W3)+b3
            out = micro_softmax(raw)
            best_idx = int(np.argmax(out))
            target   = np.zeros_like(out)
            target[best_idx] = quality
            err  = (out - target).reshape(1,-1)
            # Backprop through all 3 layers for deeper learning
            dW3  = h2.T.dot(err) * 0.01
            dh2  = err.dot(W3.T) * (h2 > 0)
            dW2  = h1.T.dot(dh2) * 0.01
            dh1  = dh2.dot(W2.T) * (h1 > 0)
            dW1  = x.T.dot(dh1) * 0.005
            W3  -= dW3;  b3 -= err * 0.001
            W2  -= dW2;  b2 -= dh2 * 0.001
            W1  -= dW1;  b1 -= dh1 * 0.001
            AI_NETWORKS[name] = [W1,W2,W3,b1,b2,b3]

    print("Pre-training complete! Vocab expanded to", len(MICRO_VOCAB), "words.")

def build_120_brain():
    for i in range(1,1001):  # 1,000 networks at 64x32 — deeper thinking
        create_micro_network("brain_net_"+str(i))

build_120_brain()

def run_micro_network(text, net):
    W1,W2,W3,b1,b2,b3 = net
    x   = micro_vectorize(text)
    if x.shape[1] != W1.shape[0]: return 0, 0.25  # shape guard
    h1  = rms_norm(np.maximum(0, x.dot(W1)+b1))   # LLaMA RMS norm
    h2  = np.maximum(0, h1.dot(W2)+b2)
    raw = h2.dot(W3)+b3
    out = micro_softmax(raw)
    idx  = int(np.argmax(out))
    conf = float(out[idx])
    return idx, conf


def sample_with_temperature(probs, temperature=0.8):
    """GPT-2 style temperature sampling — varied, natural responses."""
    probs = np.array(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-10, None)
    probs = probs ** (1.0 / max(temperature, 0.1))
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))

def run_120_brain(text, intent="unknown"):
    """
    v9 brain — Mixtral expert routing + GPT-2 temperature sampling.
    Routes to specialist networks based on intent, then samples
    with temperature for varied confidence scores.
    """
    global AI_SCORE
    all_names = list(AI_NETWORKS.keys())

    # Improvement 5 — Mixtral expert routing
    EXPERT_RANGES = {
        "chess":      (0,   100),
        "weather":    (100, 200),
        "finance":    (200, 300),
        "electrical": (300, 400),
        "greeting":   (400, 500),
    }
    if intent in EXPERT_RANGES:
        lo, hi = EXPERT_RANGES[intent]
        specialist = [f"brain_net_{i}" for i in range(lo+1, hi+1)
                      if f"brain_net_{i}" in AI_NETWORKS]
        general    = random.sample([n for n in all_names if n not in specialist],
                                   min(20, len(all_names)))
        sample = specialist[:20] + general
    else:
        sample = random.sample(all_names, min(50, len(all_names)))

    outputs = []
    for name in sample:
        idx, conf = run_micro_network(text, AI_NETWORKS[name])
        outputs.append(conf)

    if not outputs:
        return 0.5

    # GPT-2 temperature sampling — pick a confidence using weighted probability
    # (avoids always returning the boring average)
    chosen_idx = sample_with_temperature(outputs, temperature=0.8)
    avg_conf   = outputs[chosen_idx]

    AI_SCORE += avg_conf
    if AI_SCORE > 100: AI_SCORE = 0
    return round(avg_conf, 3)

def get_ai_score():
    return round(AI_SCORE, 1)

# ---- Learning from Gemini ----
LEARN_FILE   = "pybot_learned.json"
_learned_pairs = []   # list of {input, response, vocab_hits}
LEARN_LR     = 0.005  # base learning rate
# Per-network adaptive learning rates (start equal, diverge with use)
_net_lr      = {}  # name -> current lr
_net_errors  = {}  # name -> recent avg error (lower = network is confident)

def load_learned():
    global _learned_pairs
    try:
        with open(LEARN_FILE) as f:
            _learned_pairs = json.load(f)
    except:
        _learned_pairs = []

def save_learned():
    try:
        with open(LEARN_FILE,"w") as f:
            json.dump(_learned_pairs[-500:], f)
    except: pass

def expand_micro_vocab(text):
    """Add new words from text into MICRO_VOCAB so networks can learn them."""
    global MICRO_VOCAB
    words = text.lower().split()
    added = 0
    for w in words:
        # Only add meaningful words (length 3+, not already in vocab)
        if len(w) >= 3 and w not in MICRO_VOCAB and w.isalpha() and added < 5:
            MICRO_VOCAB.append(w)
            added += 1
    return added

def learn_from_pair(user_text, gemini_response):
    """
    Train all 1,000 micro-networks on a (question, response) pair.
    The networks learn which vocab patterns lead to high-quality responses.
    Uses the quality scorer (NN3) as the teaching signal.
    """
    global AI_NETWORKS, MICRO_VOCAB

    # Score the Gemini response — this is our training signal
    quality = score_response(gemini_response)

    # Expand vocab with new words from both texts
    expand_micro_vocab(user_text)
    expand_micro_vocab(gemini_response)

    # Build input vector from user text
    vec = micro_vectorize(user_text)

    # Target: push networks toward high confidence when quality is high
    # Use quality score to build a soft target
    target_conf = quality  # 0.0 - 1.0

    lessons_applied = 0
    # Sample 100 networks for learning — fast but effective
    learn_sample = random.sample(list(AI_NETWORKS.keys()), min(100, len(AI_NETWORKS)))

    for name in learn_sample:
        net = AI_NETWORKS[name]
        W1,W2,W3,b1,b2,b3 = net

        if vec.shape[1] != W1.shape[0]:
            continue

        # Adaptive learning rate per network
        if name not in _net_lr:
            _net_lr[name] = LEARN_LR
        lr = _net_lr[name]

        # Forward pass with LLaMA RMS normalisation
        h1  = rms_norm(np.maximum(0, vec.dot(W1)+b1))
        h2  = rms_norm(np.maximum(0, h1.dot(W2)+b2))
        raw = h2.dot(W3)+b3
        out = micro_softmax(raw)

        current_conf = float(np.max(out))
        best_idx     = int(np.argmax(out))

        # Only learn if there is room to improve
        if current_conf < target_conf:
            target = np.zeros_like(out)
            target[best_idx] = target_conf

            err     = (out - target).reshape(1,-1)
            err_mag = float(np.mean(np.abs(err)))
            # Weak networks (large error) learn faster; strong ones slow down
            alr = min(0.05, max(0.001, lr * (1.0 + err_mag * 2.0)))

            dW3  = h2.T.dot(err) * alr
            dh2  = err.dot(W3.T) * (h2 > 0)
            dW2  = h1.T.dot(dh2) * alr
            dh1  = dh2.dot(W2.T) * (h1 > 0)
            dW1  = vec.T.dot(dh1) * (alr * 0.5)
            W3  -= dW3;  b3 -= err * alr * 0.1
            W2  -= dW2;  b2 -= dh2 * alr * 0.1
            W1  -= dW1;  b1 -= dh1 * alr * 0.1
            AI_NETWORKS[name]  = [W1,W2,W3,b1,b2,b3]
            _net_lr[name]      = max(LEARN_LR, alr * 0.99)
            _net_errors[name]  = err_mag
            lessons_applied   += 1

    # Store the learned pair
    _learned_pairs.append({
        "input":    user_text[:100],
        "response": gemini_response[:200],
        "quality":  round(quality, 2),
        "lessons":  lessons_applied
    })
    save_learned()

    return lessons_applied, quality

load_learned()

# ---- Vector Memory ----
_ai_memory = []

def load_ai_memory():
    global _ai_memory
    try:
        with open(MEMORY_FILE) as f:
            data = json.load(f)
        # Drop any entries whose vocab size does not match current
        _ai_memory = [m for m in data
                      if m.get("vocab", 0) == len(MICRO_VOCAB)]
        if len(_ai_memory) < len(data):
            print("Memory: cleared", len(data)-len(_ai_memory),
                  "old entries with mismatched vocab size")
    except:
        _ai_memory = []

def save_ai_memory():
    try:
        with open(MEMORY_FILE,"w") as f:
            json.dump(_ai_memory[-200:], f)
    except: pass

def store_memory(text):
    # Save current vocab size alongside vector so we can detect mismatches
    _ai_memory.append({
        "text":   text,
        "vector": micro_vectorize(text).flatten().tolist(),
        "vocab":  len(MICRO_VOCAB)
    })
    if len(_ai_memory) > 200:
        _ai_memory.pop(0)
    save_ai_memory()

def recall_memory(text):
    if not _ai_memory: return None
    v    = micro_vectorize(text)
    best = None
    best_score = 0.0
    for m in _ai_memory:
        mv = np.array(m["vector"]).flatten()
        vf = v.flatten()
        # Skip if vocab size changed (old saved memory)
        if mv.shape[0] != vf.shape[0]:
            continue
        score = float(np.dot(vf, mv))
        if score > best_score:
            best_score = score
            best = m["text"]
    return best if best_score > 0.5 else None

load_ai_memory()

# ================================================================
#  MAIN APP


# ================================================================
#  FESTOES v7 — UI
#  New design: refined dark theme, smaller input, voice + vision
# ================================================================
import tkinter as tk
from tkinter import font as tkfont, filedialog
import threading, os, time, random

# ── Colour palette ──
BG_MAIN   = "#0d0d0d"   # near-black background
BG_PANEL  = "#141414"   # slightly lighter panels
BG_CARD   = "#1c1c1e"   # chat bubble cards
BG_INPUT  = "#1c1c1e"   # input field
BG_BTN    = "#242426"   # default buttons
BG_GREEN  = "#0a2a1a"   # green row bg
BG_BLUE   = "#0a1a2a"   # blue row bg
BG_SEND   = "#00c87a"   # send button
FG_WHITE  = "#f0f0f0"
FG_GREEN  = "#00c87a"
FG_BLUE   = "#4a9fff"
FG_DIM    = "#606060"
FG_YELLOW = "#f0c040"
FG_RED    = "#ff5050"
FG_PURPLE = "#c080ff"
FG_ORANGE = "#ff9040"

MOOD_COLORS = {
    "calm":    FG_BLUE,
    "excited": FG_YELLOW,
    "sad":     "#8888ff",
    "angry":   FG_RED,
    "confused":FG_ORANGE,
    "playful": "#ff60c0",
}
PERSONALITY_COLORS = {
    "friend": FG_GREEN,
    "casual": FG_YELLOW,
    "formal": FG_BLUE,
    "tutor":  FG_PURPLE,
}
PERSONALITY_ICONS = {
    "friend":"[friend]","casual":"[casual]",
    "formal":"[formal]","tutor": "[tutor]",
}

FONT_TITLE  = ("Courier", 16, "bold")
FONT_HEAD   = ("Courier", 10)
FONT_MSG    = ("Courier", 13)
FONT_SMALL  = ("Courier", 10)
FONT_BTN    = ("Courier", 9)
FONT_INPUT  = ("Courier", 13)


# ================================================================
#  AGENT MODE — proactive background engine
# ================================================================
import time as _time

AGENT_FILE   = "festoes_agent.json"
GOALS_FILE   = "festoes_goals.json"

def load_agent_state():
    try:
        with open(AGENT_FILE) as f: return json.load(f)
    except:
        return {"morning_briefing": "07:00", "agent_active": True, "last_briefing": ""}

def save_agent_state(s):
    try:
        with open(AGENT_FILE,"w") as f: json.dump(s,f)
    except: pass

def load_goals():
    try:
        with open(GOALS_FILE) as f: return json.load(f)
    except: return []

def save_goals(g):
    try:
        with open(GOALS_FILE,"w") as f: json.dump(g,f)
    except: pass

def parse_goal_command(text):
    """Detect goal-setting: 'my goal is to...' or 'i want to...' or 'add goal...'"""
    tl = text.lower().strip()
    for prefix in ["my goal is to ","my goal is ","i want to ","add goal ","set goal ","goal: "]:
        if tl.startswith(prefix):
            return tl[len(prefix):].strip()
    return None

def parse_agent_command(text):
    """Detect agent control commands."""
    tl = text.lower().strip()
    if any(x in tl for x in ["show goals","my goals","list goals","what are my goals"]):
        return "show_goals"
    if any(x in tl for x in ["clear goals","delete goals","remove goals"]):
        return "clear_goals"
    if any(x in tl for x in ["agent status","what are you working on","agent mode"]):
        return "agent_status"
    if "morning briefing" in tl:
        return "morning_briefing"
    if any(x in tl for x in ["turn off agent","disable agent","stop agent"]):
        return "agent_off"
    if any(x in tl for x in ["turn on agent","enable agent","start agent"]):
        return "agent_on"
    return None


# ================================================================
#  LIGHT / DARK THEME TOGGLE
# ================================================================
DARK_THEME = {
    "bg":    "#212121", "panel":  "#141414", "input":  "#1c1c1e",
    "btn":   "#3a3a3a", "fg":     "#ececec", "dim":    "#8e8ea0",
}
LIGHT_THEME = {
    "bg":    "#f0f0f0", "panel":  "#e0e0e0", "input":  "#ffffff",
    "btn":   "#cccccc", "fg":     "#111111", "dim":    "#555555",
}
_current_theme = "dark"

def apply_theme(app, theme_name):
    global _current_theme, BG_MAIN, BG_PANEL, BG_INPUT, BG_BTN, FG_WHITE, FG_DIM
    t = LIGHT_THEME if theme_name == "light" else DARK_THEME
    _current_theme = theme_name
    BG_MAIN  = t["bg"];   BG_PANEL = t["panel"]
    BG_INPUT = t["input"]; BG_BTN  = t["btn"]
    FG_WHITE = t["fg"];   FG_DIM   = t["dim"]
    try:
        app.root.configure(bg=BG_MAIN)
        app.chat.configure(bg=BG_INPUT, fg=FG_WHITE)
    except: pass
    settings["theme"] = theme_name
    save_settings(settings)

class AgentScheduler:
    """Background thread — checks time every 60 seconds and fires proactive actions."""
    def __init__(self, app):
        self.app     = app
        self.running = True
        self.state   = load_agent_state()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while self.running:
            _time.sleep(60)
            try:
                self._tick()
            except Exception as e:
                print("Agent tick error:", e)

    def _tick(self):
        if not self.state.get("agent_active", True):
            return
        now_str = datetime.datetime.now().strftime("%H:%M")

        # Morning briefing
        brief_time = self.state.get("morning_briefing","07:00")
        last       = self.state.get("last_briefing","")
        today      = datetime.date.today().isoformat()
        if now_str == brief_time and last != today:
            self.state["last_briefing"] = today
            save_agent_state(self.state)
            self.app.root.after(0, self._fire_morning_briefing)

        # Goal check-ins — remind about goals at noon
        if now_str == "12:00":
            goals = load_goals()
            if goals:
                msg = "🎯 Midday goal check-in! You have " + str(len(goals)) + " active goal(s): " + "; ".join(goals[:3])
                if len(goals) > 3: msg += f" (+{len(goals)-3} more)"
                self.app.root.after(0, lambda m=msg: self.app._sys_msg(m))

    def _fire_morning_briefing(self):
        name = settings.get("username","Alvine")
        city = settings.get("city","Nairobi")
        now  = datetime.datetime.now()
        greeting = "Good morning" if now.hour < 12 else "Good afternoon"
        msg = f"☀️ {greeting}, {name}! Agent briefing — {now.strftime('%A %d %B')}."
        self.app._sys_msg(msg)
        goals = load_goals()
        if goals:
            self.app._sys_msg("🎯 Your goals: " + " | ".join(goals[:5]))
        # Trigger weather + reminders in background
        threading.Thread(
            target=lambda: self.app.root.after(1500,
                lambda: self.app.process(f"weather in {city}", agent=True)),
            daemon=True).start()

    def stop(self):
        self.running = False

class FestoesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Festoes v9")
        self.root.configure(bg=BG_MAIN)
        self.root.geometry("420x820")
        self._thinking_id = None
        self._typing_dots = 0
        self._last_image_path = None
        self._build_ui()
        self.agent = AgentScheduler(self)
        self.root.after(400, self._startup)

    def _build_ui(self):
        # ════════════════════════════════════════════════════
        # ANDROID TKINTER PACK ORDER RULE:
        # Pack BOTTOM widgets first, then fill=BOTH last.
        # Otherwise expand=True grabs all space.
        # Order: header(TOP) → input(BOTTOM) → buttons(BOTTOM)
        #        → status(BOTTOM) → chat(fill BOTH) ← last
        # ════════════════════════════════════════════════════

        # ── 1. Header (TOP) ──
        hdr = tk.Frame(self.root, bg=BG_PANEL, height=46)
        hdr.pack(fill=tk.X, side=tk.TOP)
        hdr.pack_propagate(False)

        self.lbl_title = tk.Label(hdr, text="Festoes v9",
            font=FONT_TITLE, fg=FG_WHITE, bg=BG_PANEL)
        self.lbl_title.pack(side=tk.LEFT, padx=10)

        self.lbl_badge = tk.Label(hdr, text="Gemini + CSV + v9 Brain",
            font=FONT_HEAD, fg=FG_GREEN, bg=BG_PANEL)
        self.lbl_badge.pack(side=tk.LEFT, padx=6)

        self.lbl_mood = tk.Label(hdr, text="mood: calm",
            font=FONT_HEAD, fg=FG_BLUE, bg=BG_PANEL)
        self.lbl_mood.pack(side=tk.RIGHT, padx=8)

        self.lbl_persona = tk.Label(hdr, text="[friend]",
            font=FONT_HEAD, fg=FG_GREEN, bg=BG_PANEL)
        self.lbl_persona.pack(side=tk.RIGHT, padx=4)

        tk.Frame(self.root, bg=FG_GREEN, height=1).pack(fill=tk.X, side=tk.TOP)

        # ── 2. Input row (BOTTOM — must be before chat) ──
        input_row = tk.Frame(self.root, bg=BG_PANEL, pady=4, padx=6)
        input_row.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_voice = tk.Button(input_row, text="MIC",
            command=self._voice_input,
            bg=BG_BTN, fg=FG_BLUE, font=FONT_BTN,
            relief=tk.FLAT, padx=8, pady=12, bd=0)
        self.btn_voice.pack(side=tk.LEFT, padx=(0,3))

        self.btn_img = tk.Button(input_row, text="IMG",
            command=self._pick_image,
            bg=BG_BTN, fg=FG_YELLOW, font=FONT_BTN,
            relief=tk.FLAT, padx=8, pady=12, bd=0)
        self.btn_img.pack(side=tk.LEFT, padx=(0,3))

        self.btn_send = tk.Button(input_row, text="SEND",
            command=self._send,
            bg=BG_SEND, fg=FG_WHITE, font=FONT_BTN,
            relief=tk.FLAT, padx=10, pady=12, bd=0)
        self.btn_send.pack(side=tk.RIGHT, padx=(3,0))

        self.entry = tk.Entry(input_row,
            bg=BG_INPUT_BOX, fg=FG_WHITE,
            insertbackground=FG_WHITE,
            font=FONT_INPUT, relief=tk.FLAT, bd=0)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=14)
        self.entry.bind("<Return>", lambda e: self._send())

        # ── 3. Quick buttons (BOTTOM — above input) ──
        tk.Frame(self.root, bg="#2a2a2a", height=1).pack(fill=tk.X, side=tk.BOTTOM)
        qf_outer = tk.Frame(self.root, bg=BG_PANEL)
        qf_outer.pack(fill=tk.X, side=tk.BOTTOM)

        qf_canvas = tk.Canvas(qf_outer, bg=BG_PANEL, height=120,
                              highlightthickness=0)
        qf_scroll = tk.Scrollbar(qf_outer, orient=tk.HORIZONTAL,
                                 command=qf_canvas.xview, width=4,
                                 bg=BG_PANEL, troughcolor=BG_MAIN,
                                 relief=tk.FLAT)
        qf = tk.Frame(qf_canvas, bg=BG_PANEL)
        qf.bind("<Configure>",
            lambda e: qf_canvas.configure(scrollregion=qf_canvas.bbox("all")))
        qf_canvas.create_window((0,0), window=qf, anchor="nw")
        qf_canvas.configure(xscrollcommand=qf_scroll.set)
        qf_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        qf_canvas.pack(side=tk.TOP, fill=tk.X)

        def mk_btn(parent, text, cmd, bg=BG_BTN, fg=FG_DIM):
            b = tk.Button(parent, text=text,
                command=lambda c=cmd: self.quick(c),
                bg=bg, fg=fg, font=FONT_BTN,
                relief=tk.FLAT, padx=8, pady=4,
                activebackground="#404040",
                activeforeground=FG_WHITE, bd=0)
            b.pack(side=tk.LEFT, padx=2, pady=1)
            return b

        r1 = tk.Frame(qf, bg=BG_PANEL); r1.pack(fill=tk.X)
        for lbl,cmd in [("Joke","tell me a joke"),("News","latest news"),
                        ("Weather","weather in "+settings["city"]),
                        ("Time","what time is it"),("Fact","tell me a fun fact")]:
            mk_btn(r1, lbl, cmd)

        # v9 Search row — Google + DuckDuckGo
        r_search = tk.Frame(qf, bg=BG_PANEL); r_search.pack(fill=tk.X)
        # Search button types — clicking sets the entry to the trigger phrase
        def _set_search(prefix):
            self.entry.delete(0, tk.END)
            self.entry.insert(0, prefix)
            self.entry.focus()
        tk.Button(r_search, text="[G] Google Search", bg="#1565C0", fg="white",
            font=FONT_SMALL, relief=tk.FLAT, padx=8, pady=4,
            command=lambda: _set_search("search ")
        ).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(r_search, text="[DDG] DuckDuckGo", bg="#DE5833", fg="white",
            font=FONT_SMALL, relief=tk.FLAT, padx=8, pady=4,
            command=lambda: _set_search("search ")
        ).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(r_search, text="[W] Wikipedia", bg="#2E7D32", fg="white",
            font=FONT_SMALL, relief=tk.FLAT, padx=8, pady=4,
            command=lambda: _set_search("tell me about ")
        ).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(r_search, text="[N] News", bg="#6A1B9A", fg="white",
            font=FONT_SMALL, relief=tk.FLAT, padx=8, pady=4,
            command=lambda: self.quick("latest news")
        ).pack(side=tk.LEFT, padx=2, pady=2)

        r2 = tk.Frame(qf, bg=BG_PANEL); r2.pack(fill=tk.X)
        for lbl,cmd in [("Motivate","motivate me"),("Riddle","give me a riddle"),
                        ("Quiz","quiz me"),("Story","tell me a story"),("Clear","__CLEAR__")]:
            mk_btn(r2, lbl, cmd)

        r3 = tk.Frame(qf, bg=BG_PANEL); r3.pack(fill=tk.X)
        for lbl,cmd in [("Math","solve 2x+5=13"),("Define","define serendipity"),
                        ("Currency","100 usd to kes"),("Timer","set timer 1 minute"),
                        ("Stopwatch","start stopwatch")]:
            mk_btn(r3, lbl, cmd)

        r4 = tk.Frame(qf, bg=BG_PANEL); r4.pack(fill=tk.X)
        for lbl,cmd in [("Alarm","wake me at 7:00am"),("Remind","remind me to drink water in 30 minutes"),
                        ("Note","note "),("Shopping","show shopping list"),("Reminders","show reminders")]:
            mk_btn(r4, lbl, cmd, bg=BG_GREEN, fg=FG_GREEN)

        r5 = tk.Frame(qf, bg=BG_PANEL); r5.pack(fill=tk.X)
        for lbl,cmd in [("Battery","battery level"),("Torch On","torch on"),
                        ("Torch Off","torch off"),("WiFi","wifi info"),
                        ("Vol+","volume up"),("Vol-","volume down")]:
            mk_btn(r5, lbl, cmd, bg=BG_BLUE, fg=FG_BLUE)

        r6 = tk.Frame(qf, bg=BG_PANEL); r6.pack(fill=tk.X)
        for lbl,cmd in [("Friend","be my friend"),("Casual","casual mode"),
                        ("Formal","formal mode"),("Tutor","tutor mode"),
                        ("Goals","show goals"),("Export","export chat"),
                        ("Agent","agent status")]:
            mk_btn(r6, lbl, cmd, bg="#1a102a", fg=FG_PURPLE)

        # Theme toggle row
        r7 = tk.Frame(qf, bg=BG_PANEL); r7.pack(fill=tk.X)
        _theme_state = ["dark"]
        def _toggle_theme():
            _theme_state[0] = "light" if _theme_state[0] == "dark" else "dark"
            apply_theme(self.root, _theme_state[0])
        tk.Button(r7, text="[Day/Night] Theme Toggle",
            bg="#37474F", fg="white", font=FONT_SMALL,
            relief=tk.FLAT, padx=8, pady=4,
            command=_toggle_theme
        ).pack(side=tk.LEFT, padx=2, pady=2)

        # ── 4. Status bar (BOTTOM — above buttons) ──
        self.status = tk.Label(self.root, text="Loading...",
            fg=FG_DIM, bg=BG_PANEL, font=FONT_SMALL,
            anchor=tk.W, padx=10, pady=2)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

        # ── 5. Chat area (fill=BOTH LAST — takes remaining space) ──
        chat_frame = tk.Frame(self.root, bg=BG_MAIN)
        chat_frame.pack(fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(chat_frame, bg=BG_PANEL,
            troughcolor=BG_MAIN, activebackground=FG_DIM,
            relief=tk.FLAT, width=6)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.chat = tk.Text(chat_frame, bg=BG_MAIN, fg=FG_WHITE,
            font=FONT_MSG, wrap=tk.WORD, state=tk.DISABLED,
            relief=tk.FLAT, padx=12, pady=8, cursor="arrow",
            yscrollcommand=self.scrollbar.set, spacing1=2, spacing3=4)
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.chat.yview)

        # Chat text tags
        self.chat.tag_configure("you_label",   foreground=FG_DIM,    font=FONT_SMALL)
        self.chat.tag_configure("bot_label",   foreground=FG_GREEN,  font=FONT_SMALL)
        self.chat.tag_configure("you_msg",     foreground=FG_WHITE,  font=FONT_MSG)
        self.chat.tag_configure("bot_msg",     foreground="#e0e0e0", font=FONT_MSG)
        self.chat.tag_configure("sys_msg",     foreground=FG_DIM,    font=FONT_SMALL)
        self.chat.tag_configure("quality_bar", foreground=FG_DIM,    font=FONT_SMALL)
        self.chat.tag_configure("mood_tag",    foreground=FG_BLUE,   font=FONT_SMALL)
        self.chat.tag_configure("thinking",    foreground=FG_DIM,    font=FONT_SMALL)
        self.chat.tag_configure("error_msg",   foreground=FG_RED,    font=FONT_SMALL)
        self.chat.tag_configure("personality_badge", foreground=FG_PURPLE, font=FONT_SMALL)

        # ── Status bar ──


    # ── Startup ──
    def _startup(self):
        name = settings.get("username","Alvine")
        self._sys_msg(f"Festoes v9 online. Welcome back, {name}!")
        self._sys_msg("Training v9 brain (one time only)...")
        threading.Thread(target=self._train, daemon=True).start()

    def _train(self):
        if weights_exist() and micro_weights_exist():
            # Fast load — no retraining needed
            load_all_weights()
            load_micro_weights()
            self.root.after(0, lambda: self.status.config(
                text="Loaded! " + str(len(MICRO_VOCAB)) + " words | " +
                     str(len(AI_NETWORKS)) + " networks"))
            self.root.after(0, lambda: self._sys_msg(
                "Weights loaded instantly! " + str(len(MICRO_VOCAB)) +
                " words | " + str(len(AI_NETWORKS)) + " networks ready."))
        else:
            # First run — train everything and save
            self.root.after(0, lambda: self._sys_msg(
                "First run — training networks (one time only)..."))
            train_all_networks()
            self.root.after(0, lambda: self.status.config(
                text="Trained! " + str(len(MICRO_VOCAB)) + " words | " +
                     str(len(AI_NETWORKS)) + " networks"))
            self.root.after(0, lambda: self._sys_msg(
                "v8 training complete! CSV + 1,000 NNs. Will load instantly next time." +
                " " + str(len(MICRO_VOCAB)) + " words."))

    # ── Message helpers ──
    def _append(self, text, *tags):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, text, tags)
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)

    def _sys_msg(self, text):
        self._append("\n" + text + "\n", "sys_msg")

    def _show_thinking(self):
        self._append("\nFestoes is thinking...\n", "thinking")
        self._thinking_id = self.chat.index(tk.END)
        self._animate_dots()

    def _animate_dots(self):
        if not self._thinking_id: return
        self._typing_dots = (self._typing_dots + 1) % 4
        dots = "." * self._typing_dots
        try:
            self.chat.config(state=tk.NORMAL)
            # Update last thinking line
            last_line_start = self.chat.index("end-2l")
            last_line_end   = self.chat.index("end-1c")
            current = self.chat.get(last_line_start, last_line_end)
            if "thinking" in current or "..." in current:
                self.chat.delete(last_line_start, last_line_end)
                self.chat.insert(last_line_start,
                    "Festoes is thinking" + dots, "thinking")
            self.chat.config(state=tk.DISABLED)
        except: pass
        if self._thinking_id:
            self.root.after(400, self._animate_dots)

    def _remove_thinking(self):
        self._thinking_id = None

    def _quality_bar(self, score):
        filled = int(score * 14)
        bar = "[" + "#" * filled + "-" * (14 - filled) + "]"
        pct = str(int(score * 100)) + "%"
        color = "quality_bar"
        self._append("  quality: " + bar + " " + pct + "\n", color)

    # ── Voice input ──
    def _voice_input(self):
        self.btn_voice.config(text="...", fg=FG_RED)
        self.status.config(text="Listening...")
        def _do():
            result = voice_input()
            self.root.after(0, lambda: self.btn_voice.config(
                text="MIC", fg=FG_BLUE))
            if result.startswith("__VOICE_ERROR__"):
                self.root.after(0, lambda: self.status.config(
                    text="Mic needs androidhelper — install it in Pydroid pip"))
                self.root.after(0, lambda: self.finish(
                    "Voice input is not available in this setup.\n\n"
                    "To enable it:\n"
                    "1. Open Pydroid 3\n"
                    "2. Tap the menu → Pip\n"
                    "3. Install: androidhelper\n"
                    "4. Restart Festoes\n\n"
                    "For now, please type your message instead."))
                return
            if result and not result.startswith("Voice intent"):
                self.root.after(0, lambda r=result: (
                    self.entry.delete(0, tk.END),
                    self.entry.insert(0, r),
                    self._send()
                ))
            else:
                self.root.after(0, lambda: self.status.config(text=result))
        threading.Thread(target=_do, daemon=True).start()

    # ── Image / Vision input ──
    def _pick_image(self):
        path = filedialog.askopenfilename(
            title="Pick image for Festoes Vision",
            filetypes=[("Images","*.jpg *.jpeg *.png *.webp *.gif"),
                       ("All","*.*")])
        if not path: return
        self._last_image_path = path
        fname = os.path.basename(path)
        self.entry.delete(0, tk.END)
        self.entry.insert(0, "analyse this image")
        self._sys_msg(f"Image selected: {fname}")
        self._send()

    # ── Send ──
    def _send(self):
        text = self.entry.get().strip()
        if not text: return
        self.entry.delete(0, tk.END)

        # Show user message
        name = settings.get("username","Alvine")
        ts   = time.strftime("%H:%M")
        self._append("\n" + name + "  " + ts + "\n", "you_label")
        self._append(text + "\n", "you_msg")
        _h = load_history()
        _h.append({"sender": name, "message": text, "time": ts})
        save_history(_h)

        self._show_thinking()
        self.status.config(text="Processing...")
        self.btn_send.config(state=tk.DISABLED)

        image_path = self._last_image_path
        self._last_image_path = None

        threading.Thread(
            target=lambda: self.process(text, image_path),
            daemon=True).start()

    # ── Quick buttons ──
    def quick(self, cmd):
        if cmd == "__CLEAR__":
            self.chat.config(state=tk.NORMAL)
            self.chat.delete("1.0", tk.END)
            self.chat.config(state=tk.DISABLED)
            clear_history()
            self._sys_msg("Chat cleared.")
            return
        self.entry.delete(0, tk.END)
        self.entry.insert(0, cmd)
        self._send()

    # ── Open settings ──
    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Festoes Settings")
        win.configure(bg=BG_MAIN)
        win.geometry("300x420")

        tk.Label(win, text="SETTINGS", fg=FG_GREEN, bg=BG_MAIN,
            font=("Courier",14,"bold")).pack(pady=12)

        fields = [
            ("Your name", "username"),
            ("City", "city"),
        ]
        vars_ = {}
        for label, key in fields:
            tk.Label(win, text=label, fg=FG_DIM, bg=BG_MAIN,
                font=FONT_SMALL).pack(anchor=tk.W, padx=20)
            v = tk.StringVar(value=settings.get(key,""))
            e = tk.Entry(win, textvariable=v, bg=BG_INPUT, fg=FG_WHITE,
                font=FONT_INPUT, relief=tk.FLAT, insertbackground=FG_GREEN)
            e.pack(fill=tk.X, padx=20, pady=4, ipady=5)
            vars_[key] = v

        # Personality selector
        tk.Label(win, text="Personality", fg=FG_DIM, bg=BG_MAIN,
            font=FONT_SMALL).pack(anchor=tk.W, padx=20, pady=(8,0))
        p_var = tk.StringVar(value=settings.get("personality","friend"))
        for mode in ["friend","casual","formal","tutor"]:
            tk.Radiobutton(win, text=mode.capitalize(),
                variable=p_var, value=mode,
                bg=BG_MAIN, fg=PERSONALITY_COLORS[mode],
                selectcolor=BG_CARD, activebackground=BG_MAIN,
                font=FONT_SMALL).pack(side=tk.LEFT, padx=10)

        def save_all():
            for key, var in vars_.items():
                settings[key] = var.get().strip() or settings[key]
            settings["personality"] = p_var.get()
            save_settings()
            win.destroy()
            self._sys_msg("Settings saved.")

        tk.Button(win, text="SAVE", command=save_all,
            bg=BG_SEND, fg="#000000", font=("Courier",11,"bold"),
            relief=tk.FLAT, padx=20, pady=8, cursor="hand2").pack(pady=16)

        # Stats
        stats = (f"Networks: {len(AI_NETWORKS)} | "
                 f"Vocab: {len(MICRO_VOCAB)} | "
                 f"Learned: {len(_learned_pairs)}")
        tk.Label(win, text=stats, fg=FG_DIM, bg=BG_MAIN,
            font=FONT_SMALL).pack()

    # ── Process message ──
    def process(self, text, image_path=None, agent=False):
        tl = text.lower().strip()

        # ── Image analysis (Gemini Vision) ──
        if image_path:
            prompt = text if text != "analyse this image" else "Describe this image in detail."
            result = ask_gemini_vision(image_path, prompt)
            self.root.after(0, lambda r=result: self.finish(r))
            return

        # ── Neural brain ──
        # ── Intent + Mood (before brain for expert routing) ──
        intent, _     = detect_intent(text)
        mood, mood_cf = detect_mood(text)

        brain_conf = run_120_brain(text, intent)
        mem_match  = recall_memory(text)
        store_memory(text)

        self.root.after(0, lambda: self.status.config(
            text="Brain: " + str(int(brain_conf*100)) + "% | " +
                 str(len(AI_NETWORKS)) + " nets"))

        mood_color = MOOD_COLORS.get(mood, FG_BLUE)
        persona    = settings.get("personality","friend")
        p_color    = PERSONALITY_COLORS.get(persona, FG_GREEN)
        p_icon     = PERSONALITY_ICONS.get(persona,"")

        self.root.after(0, lambda m=mood, mc=mood_color, pi=p_icon, pc=p_color: (
            self.lbl_mood.config(text="mood: " + m, fg=mc),
            self.lbl_persona.config(text=pi, fg=pc)
        ))

        if mem_match and mem_match.lower() != text.lower():
            self.root.after(0, lambda m=mem_match:
                self._sys_msg('Recalled: "' + m[:50] + '"'))

        # ── Android status debug ──
        if any(w in tl for w in ["android status","phone status","festoes status","check android"]):
            status  = "ANDROID_AVAILABLE: " + str(ANDROID_AVAILABLE) + "\n"
            status += "PLYER_AVAILABLE: "   + str(PLYER_AVAILABLE)   + "\n"
            status += "TTS_AVAILABLE: "     + str(TTS_AVAILABLE)     + "\n"
            status += "droid: "             + str(type(droid))       + "\n\n"
            if PLYER_AVAILABLE:
                status += "Plyer loaded! Phone features ACTIVE!"
            else:
                status += "Plyer not loaded.\nInstall: Pip > plyer"
            self.root.after(0, lambda s=status: self.finish(s)); return

        # ── Personality switch ──
        # ── Web Search (v9: Google + DuckDuckGo) ──
        search_query = parse_search_query(text)
        if search_query:
            self.root.after(0, lambda: self.status.config(text="Searching the web..."))
            result = web_search(search_query)
            self.root.after(0, lambda r=result: self.finish(r))
            return

        # ── Chess PGN analysis ──
        if detect_pgn(text):
            reply = handle_pgn(text, self)
            if reply:
                self.root.after(0, lambda r=reply: self.finish(r))
                threading.Thread(target=lambda: learn_from_pair(text, reply), daemon=True).start()
                return

        # ── Goal tracking ──
        goal_text = parse_goal_command(text)
        if goal_text:
            goals = load_goals()
            goals.append(goal_text)
            save_goals(goals)
            self.root.after(0, lambda: self.finish(f'Goal added: "{goal_text}" — I will keep track of this for you.'))
            return

        agent_cmd = parse_agent_command(text)
        if agent_cmd == 'show_goals':
            goals = load_goals()
            if goals:
                reply = 'Your goals:\n' + '\n'.join(f'- {g}' for g in goals)
            else:
                reply = 'No goals yet. Say "my goal is to..." to add one.'
            self.root.after(0, lambda r=reply: self.finish(r)); return
        elif agent_cmd == 'clear_goals':
            save_goals([])
            self.root.after(0, lambda: self.finish('All goals cleared.')); return
        elif agent_cmd == 'agent_status':
            state = load_agent_state(); goals = load_goals()
            active = 'ON' if state.get('agent_active', True) else 'OFF'
            brief = state.get('morning_briefing','07:00')
            self.root.after(0, lambda: self.finish(f'Agent: {active} | Briefing: {brief} | Goals: {len(goals)}')); return
        elif agent_cmd in ('agent_on','agent_off'):
            s = load_agent_state()
            s['agent_active'] = (agent_cmd == 'agent_on')
            save_agent_state(s)
            msg = 'Agent ON — morning briefings and goal tracking active.' if s['agent_active'] else 'Agent OFF.'
            self.root.after(0, lambda m=msg: self.finish(m)); return

        # ── Export chat ──
        if any(x in tl for x in ['export chat','save chat','export history','download chat']):
            self.root.after(0, lambda: self.finish(export_chat())); return

        p_result = parse_personality_command(text)
        if p_result:
            self.root.after(0, lambda r=p_result: self.finish(r)); return

        # ── Phone controls ──
        phone_result = parse_phone_command(text)
        if phone_result:
            self.root.after(0, lambda r=phone_result: self.finish(r)); return

        # ── Home assistant ──
        home_result = parse_home_command(text)
        if home_result:
            self.root.after(0, lambda r=home_result: self.finish(r)); return

        # ── Active states ──
        if _quiz_state["active"]:
            self.root.after(0, lambda: self.finish(check_quiz_answer(text))); return
        if _riddle_state["active"]:
            ans = check_riddle_answer(text)
            if ans:
                self.root.after(0, lambda a=ans: self.finish(a)); return

        timer_done = check_timer()
        if timer_done:
            self.root.after(0, lambda r=timer_done: self.finish(r)); return

        # ── Math ──
        if is_math_query(text):
            r = solve_math(text)
            if r:
                self.root.after(0, lambda x=r: self.finish(x)); return

        # ── Dictionary ──
        tl2 = tl
        if any(w in tl for w in ["define ","definition of ","meaning of "]):
            for p in ["define ","definition of ","meaning of "]:
                tl2 = tl2.replace(p,"").strip()
            word = tl2.split()[-1] if tl2 else ""
            if word:
                self.root.after(0, lambda w=word: self.finish(fetch_definition(w))); return

        # ── Currency ──
        if any(w in tl for w in ["usd","eur","gbp","ksh","kes","ngn","convert","exchange"]):
            cur = parse_currency_query(text)
            if cur:
                self.root.after(0, lambda c=cur: self.finish(c)); return

        # ── Feature triggers ──
        if any(w in tl for w in ["riddle","brain teaser","puzzle me"]):
            self.root.after(0, lambda: self.finish(get_riddle())); return
        if any(w in tl for w in ["quiz","test me","trivia"]):
            self.root.after(0, lambda: self.finish(start_quiz())); return
        if "start stopwatch" in tl:
            self.root.after(0, lambda: self.finish(start_stopwatch())); return
        if "stop stopwatch" in tl:
            self.root.after(0, lambda: self.finish(stop_stopwatch())); return
        if any(w in tl for w in ["set timer","timer for","countdown"]):
            r = parse_timer(text)
            if r:
                self.root.after(0, lambda x=r: self.finish(x)); return
        if any(w in tl for w in ["tell me a story","random story","story time"]):
            self.root.after(0, lambda: self.finish(generate_story())); return

        if any(w in tl for w in ["weather","forecast","temperature"]):
            city = settings.get("city","Nairobi")
            words = tl.split()
            for i,w in enumerate(words):
                if w == "in" and i+1 < len(words):
                    city = words[i+1].capitalize(); break
            self.root.after(0, lambda c=city: self.finish(fetch_weather(c))); return

        if any(w in tl for w in ["news","headlines"]):
            self.root.after(0, lambda: self.finish(fetch_news())); return
        if any(w in tl for w in ["joke","funny","make me laugh"]):
            self.root.after(0, lambda: self.finish(fetch_joke())); return

        # ── Fact ──
        if any(w in tl for w in ["fun fact","tell me a fact","interesting fact",
                                  "random fact","did you know","amaze me","blow my mind"]):
            name = settings.get("username","Alvine")
            facts = LOCAL_RESPONSES.get("fact", [])
            if facts:
                self.root.after(0, lambda: self.finish(random.choice(facts).replace("{name}", name)))
                return

        # ── Motivation ──
        if any(w in tl for w in ["motivate","motivation","encourage","cheer me up",
                                  "i need motivation","i give up","feel like giving up"]):
            name = settings.get("username","Alvine")
            mots = LOCAL_RESPONSES.get("motivation", [])
            if mots:
                self.root.after(0, lambda: self.finish(random.choice(mots).replace("{name}", name)))
                return

        # ── Whisper confidence threshold ──
        # If local brain is confident, try local_fallback first
        if brain_conf >= CONFIDENCE_THRESHOLD:
            local_try = local_fallback(text, intent, mood)
            if local_try and len(local_try.split()) > 3:
                self.root.after(0, lambda r=local_try: self.finish(r))
                return

        # ── Gemini AI ──
        self.root.after(0, lambda: self.status.config(text="Asking Gemini AI..."))
        result = ask_gemini(text, mood, mood_cf, intent)
        if result and not result.startswith("__ERROR__"):
            lessons, quality = learn_from_pair(text, result)
            fu = maybe_followup(intent)
            self.root.after(0, lambda: self.status.config(
                text="Learned! " + str(lessons) + " nets updated"))
            self.root.after(0, lambda r=result+fu: self.finish(r))
        else:
            self.root.after(0, lambda: self.status.config(
                text="Using local brain..."))
            local = local_fallback(text, intent, mood)
            fu    = maybe_followup(intent)
            self.root.after(0, lambda r=local+fu: self.finish(r))

    # ── Finish: display bot reply ──
    def finish(self, result, skip_score=False):
        self._remove_thinking()
        score = score_response(result) if not skip_score else None
        ts    = time.strftime("%H:%M")

        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, "\nFestoes  " + ts + "\n", "bot_label")
        self.chat.insert(tk.END, result + "\n",              "bot_msg")
        if score is not None:
            filled = int(score * 14)
            bar = "[" + "#"*filled + "-"*(14-filled) + "]"
            self.chat.insert(tk.END,
                "  quality: " + bar + " " + str(int(score*100)) + "%\n",
                "quality_bar")
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)
        self.btn_send.config(state=tk.NORMAL)
        self.status.config(text="Ready.")
        speak(result)
        h = load_history()
        h.append({"sender": "Festoes", "message": result,
                  "time": time.strftime("%H:%M")})
        save_history(h)

    def remove_thinking(self):
        self._remove_thinking()
        # Remove last thinking line from chat
        self.chat.config(state=tk.NORMAL)
        try:
            for _ in range(3):
                last = self.chat.get("end-2l","end-1c")
                if "thinking" in last.lower() or not last.strip():
                    self.chat.delete("end-2l","end-1l")
        except: pass
        self.chat.config(state=tk.DISABLED)


# ================================================================
#  MAIN
# ================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app  = FestoesApp(root)
    root.mainloop()
