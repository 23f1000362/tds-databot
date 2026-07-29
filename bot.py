import os
import json
import time
import threading
import traceback
from datetime import datetime, timezone
from collections import defaultdict

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from google import genai
from google.genai import types

# ---------- CONFIG ----------
BOT_TOKEN = os.environ["BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
BASE_URL = os.environ["BASE_URL"]

client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-flash-lite-latest"

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

LOG_FILE = "run.jsonl"
log_lock = threading.Lock()

def log_event(event: dict):
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

# ---------- TOOL: run_python ----------
def run_python(code: str) -> str:
    import io
    import contextlib

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": __builtins__}, {})
        output = buf.getvalue()
    except Exception as e:
        output = f"ERROR: {e}\n{traceback.format_exc()}"
    return output[-8000:]

# ---------- GEMINI AGENT ----------
SYSTEM_PROMPT = """You are a data-analyst agent replying to Telegram messages.

Rules:
- Answer the LATEST message in the conversation. Earlier messages are context for multi-turn questions.
- Use the run_python tool to fetch data and compute answers. Never guess a number you could compute.
- For general published statistics that are stable, well-documented facts (e.g. "which state has the highest X" for a well-known metric), you may answer from your own trained knowledge if fetching genuinely fails after retrying — but only when you are reasonably confident this fact hasn't changed and isn't ambiguous, and only after real retry attempts, not as a first resort.
- Never invent a specific number, statistic, or live value (like a current price, live measurement, or exact computed figure) — for these, if fetching fails after retries, say so explicitly rather than fabricating a number.
- Your FINAL reply must be ONLY a single JSON object, and nothing else - no markdown fences, no prose, no "here is the answer".
- Match the JSON shape the question asks for EXACTLY (same keys, same nesting, correct types - number vs string).
- Always include a "log_url" key in your final JSON with the exact placeholder string "LOG_URL_PLACEHOLDER" - the calling code will replace it.
- If a message is just setup/context (e.g. "I will send data next"), still reply with a minimal valid JSON acknowledgment.
- Never add extra keys beyond what's asked.
- When fetching external data, if a URL/API fails, try at least one alternative approach (a different URL structure, a cached mirror, or re-reading the question for an explicit source) before giving up.
- Never guess or hardcode a fallback answer when a fetch fails. If, after reasonable attempts, you cannot retrieve real data, say so honestly in your final answer rather than fabricating a plausible-looking one — a wrong answer that admits uncertainty is safer than a confident guess that happens to be checked against a real value.
- Prefer data sources explicitly mentioned or linked in the question over ones you recall from training, since your training knowledge of specific APIs/endpoints may be outdated.
"""

run_python_tool = types.FunctionDeclaration(
    name="run_python",
    description="Execute Python code server-side and return captured stdout. Use this to fetch data, compute statistics, or analyze datasets. Libraries available: pandas, numpy, requests, BeautifulSoup (bs4), openpyxl.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "code": types.Schema(type="STRING", description="Python code to execute")
        },
        required=["code"]
    )
)

tools = types.Tool(function_declarations=[run_python_tool])

config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[tools],
)

def extract_json(text: str):
    if not text:
        return None
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None

def safe_text(response):
    """response.text raises if there's no text part (e.g. only a function_call). Guard it."""
    try:
        return response.text
    except Exception:
        return None

def run_agent(history: list, chat_id: int) -> dict:
    deadline = time.time() + 210

    contents = []
    for h in history:
        role = "user" if h["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=h["parts"][0])]))

    convo_log = []
    max_steps = 10
    step = 0
    response_text = None
    response = None

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=config,
        )

        while step < max_steps:
            step += 1
            if time.time() > deadline:
                convo_log.append({"event": "deadline_exceeded", "step": step})
                try:
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part(text="Time is up. Answer NOW with only the final JSON object, no tools.")]
                    ))
                    response = client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
                    response_text = safe_text(response)
                except Exception as e:
                    convo_log.append({"event": "forced_answer_failed", "error": str(e)})
                break

            candidate = response.candidates[0]
            parts = candidate.content.parts
            func_calls = [p.function_call for p in parts if p.function_call]

            if func_calls:
                fc = func_calls[0]
                code = fc.args.get("code", "")
                convo_log.append({"event": "tool_call", "step": step, "code": code})
                result = run_python(code)
                convo_log.append({"event": "tool_result", "step": step, "output": result})

                contents.append(candidate.content)
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(function_response=types.FunctionResponse(
                        name="run_python", response={"result": result}
                    ))]
                ))
                response = client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
            else:
                response_text = safe_text(response)
                convo_log.append({"event": "final_text", "step": step, "text": response_text})
                break
    except Exception as e:
        convo_log.append({"event": "exception", "error": str(e), "trace": traceback.format_exc()})

    for entry in convo_log:
        log_event({"chat_id": chat_id, **entry})

    if response_text is None and response is not None:
        response_text = safe_text(response)

    parsed = extract_json(response_text) if response_text else None

    if parsed is None:
        parsed = {"answer": "internal error", "log_url": "LOG_URL_PLACEHOLDER"}
    elif "answer" not in parsed:
        parsed = {"answer": parsed, "log_url": "LOG_URL_PLACEHOLDER"}

    parsed["log_url"] = f"{BASE_URL}/run.jsonl"
    return parsed

# ---------- TELEGRAM POLLING ----------
chat_histories = defaultdict(list)
MAX_HISTORY = 20

def telegram_send(chat_id, text):
    requests.post(f"{TELEGRAM_API}/sendMessage", json={"chat_id": chat_id, "text": text})

def handle_message(chat_id, text):
    chat_histories[chat_id].append({"role": "user", "parts": [text]})
    chat_histories[chat_id] = chat_histories[chat_id][-MAX_HISTORY:]

    try:
        result = run_agent(chat_histories[chat_id], chat_id)
    except Exception as e:
        log_event({"chat_id": chat_id, "event": "top_level_exception", "error": str(e), "trace": traceback.format_exc()})
        result = {"answer": "internal error", "log_url": f"{BASE_URL}/run.jsonl"}

    reply_text = json.dumps(result)
    chat_histories[chat_id].append({"role": "model", "parts": [reply_text]})

    telegram_send(chat_id, reply_text)
    log_event({"chat_id": chat_id, "event": "reply_sent", "reply": result})

def polling_loop():
    offset = None
    while True:
        try:
            params = {"timeout": 30}
            if offset:
                params["offset"] = offset
            resp = requests.get(f"{TELEGRAM_API}/getUpdates", params=params, timeout=40)
            data = resp.json()
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                message = update.get("message")
                if not message:
                    continue
                chat_id = message["chat"]["id"]
                text = message.get("text", "")
                if text:
                    threading.Thread(target=handle_message, args=(chat_id, text)).start()
        except Exception as e:
            log_event({"event": "polling_error", "error": str(e)})
            time.sleep(5)

def self_ping_loop():
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health", timeout=10)
        except Exception:
            pass

# ---------- FASTAPI APP ----------
app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}

@app.get("/run.jsonl")
def get_log():
    if not os.path.exists(LOG_FILE):
        return PlainTextResponse("", media_type="text/plain")
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    return PlainTextResponse(content, media_type="text/plain")

@app.on_event("startup")
def startup():
    threading.Thread(target=polling_loop, daemon=True).start()
    threading.Thread(target=self_ping_loop, daemon=True).start()