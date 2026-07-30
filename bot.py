import os
import json
import time
import threading
import traceback
from datetime import datetime, timezone
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from google import genai
from google.genai import types
from groq import Groq

# ---------- CONFIG ----------
BOT_TOKEN = os.environ["BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
BASE_URL = os.environ["BASE_URL"]

# FIX: timeout cut to 12s, and retry_options attempts=1 disables the SDK's
# own internal tenacity retry loop. Previously the SDK was silently
# retrying transient errors internally (up to ~4x with backoff) BEFORE
# ever raising back to gemini_generate() - that's why a "45s timeout"
# actually took ~93s. Now one failed call fails in ~12s, period.
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=12_000,  # ms
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)
GEMINI_MODEL = "gemini-flash-lite-latest"

groq_client = Groq(api_key=GROQ_API_KEY, timeout=20.0)
GROQ_MODEL = "llama-3.3-70b-versatile"

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

LOG_FILE = "run.jsonl"
log_lock = threading.Lock()

def log_event(event: dict):
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

def log_step(chat_id, convo_log, entry):
    convo_log.append(entry)
    log_event({"chat_id": chat_id, **entry})

# ---------- CONCURRENCY CONTROL ----------
MESSAGE_WORKERS = 4
message_executor = ThreadPoolExecutor(max_workers=MESSAGE_WORKERS, thread_name_prefix="msg")

TOOL_WORKERS = 8
tool_executor = ThreadPoolExecutor(max_workers=TOOL_WORKERS, thread_name_prefix="tool")

# ---------- TOOL: run_python ----------
def run_python(code: str, timeout: float = 60) -> str:
    import io
    import contextlib

    def _exec_target():
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, {"__builtins__": __builtins__}, {})
            return buf.getvalue()
        except Exception as e:
            return f"ERROR: {e}\n{traceback.format_exc()}"

    future = tool_executor.submit(_exec_target)
    try:
        output = future.result(timeout=timeout)
    except Exception:
        return f"ERROR: code execution timed out after {timeout:.0f} seconds (likely a hanging network call or infinite loop)."

    return output[-8000:]

# ---------- SYSTEM PROMPT (shared by both providers) ----------
SYSTEM_PROMPT = """You are a data-analyst agent replying to Telegram messages.

Rules:
- Answer the LATEST message in the conversation. Earlier messages are context for multi-turn questions.
- Use the run_python tool to fetch data and compute answers. Never guess a number you could compute.
- For general published statistics that are stable, well-documented facts (e.g. "which state has the highest X" for a well-known metric), you may answer from your own trained knowledge if fetching genuinely fails after retrying — but only when you are reasonably confident this fact hasn't changed and isn't ambiguous, and only after real retry attempts, not as a first resort.
- Never invent a specific number, statistic, or live value (like a current price, live measurement, or exact computed figure) — for these, if fetching fails after retries, say so explicitly rather than fabricating a number.
- Your FINAL reply must be ONLY a single JSON object, and nothing else - no markdown fences, no prose, no "here is the answer".
- Match the JSON shape the question asks for EXACTLY. If the question shows an example like {"answer": <value>, "log_url": "..."}, the outer key MUST be the literal string "answer" - never substitute a more descriptive key name like "capital", "state", "total", "result", or similar, even if it feels more readable. Copy the exact key names shown in the question's example, do not invent your own.
- If "answer" should hold a plain value (a number, a string, true/false), do NOT wrap it in a nested object - return the raw value directly as "answer"'s value, unless the question's own example explicitly shows a nested object for "answer".
- Always include a "log_url" key in your final JSON with the exact placeholder string "LOG_URL_PLACEHOLDER" - the calling code will replace it.
- If a message is just setup/context (e.g. "I will send data next"), still reply with a minimal valid JSON acknowledgment.
- Never add extra keys beyond what's asked.
- When fetching external data, if a URL/API fails, try at least one alternative approach (a different URL structure, a cached mirror, or re-reading the question for an explicit source) before giving up.
- Never guess or hardcode a fallback answer when a fetch fails. If, after reasonable attempts, you cannot retrieve real data, say so honestly in your final answer rather than fabricating a plausible-looking one — a wrong answer that admits uncertainty is safer than a confident guess that happens to be checked against a real value.
- Prefer data sources explicitly mentioned or linked in the question over ones you recall from training, since your training knowledge of specific APIs/endpoints may be outdated.
- Do not guess more than 2-3 speculative URLs for a dataset. If official sources aren't immediately found, fall back to your trained knowledge sooner rather than exhausting many attempts on unlikely URLs.
"""

# ---------- GEMINI SETUP ----------
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

gemini_tools = types.Tool(function_declarations=[run_python_tool])

gemini_config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[gemini_tools],
)

# ---------- GROQ SETUP (OpenAI-compatible tool schema) ----------
groq_tools = [{
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Execute Python code server-side and return captured stdout. Use this to fetch data, compute statistics, or analyze datasets. Libraries available: pandas, numpy, requests, BeautifulSoup (bs4), openpyxl.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"]
        }
    }
}]

def extract_json(text: str):
    if not text:
        return None
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None

def safe_text(response):
    try:
        return response.text
    except Exception:
        return None

def normalize_final(parsed):
    if parsed is None:
        return {"answer": "internal error", "log_url": "LOG_URL_PLACEHOLDER"}
    if "answer" not in parsed:
        other_keys = [k for k in parsed.keys() if k != "log_url"]
        if len(other_keys) == 1:
            value = parsed[other_keys[0]]
            parsed = {"answer": value, "log_url": parsed.get("log_url", "LOG_URL_PLACEHOLDER")}
        else:
            parsed = {"answer": parsed, "log_url": "LOG_URL_PLACEHOLDER"}
    return parsed

# FIX: max_retries dropped to 1 (i.e. one attempt, no retry, no sleep).
# With the SDK's own internal retry now also disabled, one bad/slow
# Gemini call now costs ~12s (the http timeout) instead of ~93s.
def gemini_generate(contents, max_retries=1):
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=GEMINI_MODEL, contents=contents, config=gemini_config
            )
        except Exception as e:
            last_err = e
    raise last_err

# ---------- GEMINI AGENT LOOP ----------
def run_gemini_agent(history: list, chat_id: int, deadline: float, convo_log: list):
    contents = []
    for h in history:
        role = "user" if h["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=h["parts"][0])]))

    max_steps = 10
    step = 0
    response_text = None

    log_step(chat_id, convo_log, {"event": "gemini_calling", "step": 0})
    response = gemini_generate(contents)
    log_step(chat_id, convo_log, {"event": "gemini_first_response_received", "step": 0})

    while step < max_steps:
        step += 1

        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
            log_step(chat_id, convo_log, {
                "event": "gemini_empty_response",
                "step": step,
                "finish_reason": str(getattr(response.candidates[0], "finish_reason", None)) if response.candidates else "no_candidates"
            })
            response_text = None
            break

        candidate = response.candidates[0]
        parts = candidate.content.parts
        func_calls = [p.function_call for p in parts if p.function_call]

        if func_calls:
            fc = func_calls[0]
            code = fc.args.get("code", "")
            log_step(chat_id, convo_log, {"event": "gemini_tool_call", "step": step, "code": code})

            remaining = deadline - time.time()
            if remaining <= 5:
                log_step(chat_id, convo_log, {"event": "gemini_deadline_exceeded", "step": step})
                contents.append(candidate.content)
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text="Time is up. Answer NOW with only the final JSON object, no tools.")]
                ))
                response = gemini_generate(contents)
                response_text = safe_text(response)
                break

            call_timeout = min(60, remaining)
            result = run_python(code, timeout=call_timeout)
            log_step(chat_id, convo_log, {"event": "gemini_tool_result", "step": step, "output": result})

            contents.append(candidate.content)
            contents.append(types.Content(
                role="user",
                parts=[types.Part(function_response=types.FunctionResponse(
                    name="run_python", response={"result": result}
                ))]
            ))

            if time.time() > deadline:
                log_step(chat_id, convo_log, {"event": "gemini_deadline_exceeded", "step": step})
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text="Time is up. Answer NOW with only the final JSON object, no tools.")]
                ))
                response = gemini_generate(contents)
                response_text = safe_text(response)
                break

            log_step(chat_id, convo_log, {"event": "gemini_calling", "step": step})
            response = gemini_generate(contents)
            log_step(chat_id, convo_log, {"event": "gemini_response_received", "step": step})
        else:
            response_text = safe_text(response)
            log_step(chat_id, convo_log, {"event": "gemini_final_text", "step": step, "text": response_text})
            break

    if response_text is None and step >= max_steps:
        log_step(chat_id, convo_log, {"event": "gemini_max_steps_exceeded", "step": step})
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text="You have used too many steps. Stop trying new approaches and answer NOW with only the final JSON object, using your best available information.")]
        ))
        response = gemini_generate(contents)
        response_text = safe_text(response)

    if response_text is None and response is not None:
        response_text = safe_text(response)

    return response_text

# ---------- GROQ AGENT LOOP (fallback) ----------
def run_groq_agent(history: list, chat_id: int, deadline: float, convo_log: list):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        role = "user" if h["role"] == "user" else "assistant"
        messages.append({"role": role, "content": h["parts"][0]})

    max_steps = 10
    step = 0
    response_text = None

    while step < max_steps:
        step += 1
        if time.time() > deadline:
            log_step(chat_id, convo_log, {"event": "groq_deadline_exceeded", "step": step})
            messages.append({"role": "user", "content": "Time is up. Answer NOW with only the final JSON object, no tools."})
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL, messages=messages,
            )
            response_text = resp.choices[0].message.content
            break

        log_step(chat_id, convo_log, {"event": "groq_calling", "step": step})
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=groq_tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                code = args.get("code", "")
                log_step(chat_id, convo_log, {"event": "groq_tool_call", "step": step, "code": code})
                remaining = max(5, deadline - time.time())
                call_timeout = min(60, remaining)
                result = run_python(code, timeout=call_timeout)
                log_step(chat_id, convo_log, {"event": "groq_tool_result", "step": step, "output": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            response_text = msg.content
            log_step(chat_id, convo_log, {"event": "groq_final_text", "step": step, "text": response_text})
            break

    if response_text is None:
        log_step(chat_id, convo_log, {"event": "groq_max_steps_exceeded", "step": step})
        messages.append({"role": "user", "content": "Stop trying new approaches and answer NOW with only the final JSON object, using your best available information."})
        resp = groq_client.chat.completions.create(model=GROQ_MODEL, messages=messages)
        response_text = resp.choices[0].message.content

    return response_text

# ---------- ORCHESTRATOR ----------
def run_agent(history: list, chat_id: int) -> dict:
    deadline = time.time() + 210
    convo_log = []
    response_text = None
    provider_used = "gemini"

    try:
        response_text = run_gemini_agent(history, chat_id, deadline, convo_log)
    except Exception as e:
        log_step(chat_id, convo_log, {"event": "gemini_exception", "error": str(e), "trace": traceback.format_exc()})
        response_text = None

    if response_text is None or extract_json(response_text) is None:
        log_step(chat_id, convo_log, {"event": "falling_back_to_groq"})
        provider_used = "groq"
        try:
            response_text = run_groq_agent(history, chat_id, deadline, convo_log)
        except Exception as e:
            log_step(chat_id, convo_log, {"event": "groq_exception", "error": str(e), "trace": traceback.format_exc()})
            response_text = None

    parsed = extract_json(response_text) if response_text else None
    parsed = normalize_final(parsed)
    parsed["log_url"] = f"{BASE_URL}/run.jsonl"

    log_event({"chat_id": chat_id, "event": "provider_used", "provider": provider_used})
    return parsed

# ---------- TELEGRAM POLLING ----------
chat_histories = defaultdict(list)
history_lock = threading.Lock()
MAX_HISTORY = 20

def telegram_send(chat_id, text):
    try:
        requests.post(f"{TELEGRAM_API}/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=15)
    except Exception as e:
        log_event({"chat_id": chat_id, "event": "telegram_send_failed", "error": str(e)})

def handle_message(chat_id, text, received_at):
    log_event({"chat_id": chat_id, "event": "handler_started", "queue_delay_sec": round(time.time() - received_at, 3)})

    with history_lock:
        chat_histories[chat_id].append({"role": "user", "parts": [text]})
        chat_histories[chat_id] = chat_histories[chat_id][-MAX_HISTORY:]
        history_snapshot = list(chat_histories[chat_id])

    try:
        result = run_agent(history_snapshot, chat_id)
    except Exception as e:
        log_event({"chat_id": chat_id, "event": "top_level_exception", "error": str(e), "trace": traceback.format_exc()})
        result = {"answer": "internal error", "log_url": f"{BASE_URL}/run.jsonl"}

    reply_text = json.dumps(result)

    with history_lock:
        chat_histories[chat_id].append({"role": "model", "parts": [reply_text]})
        chat_histories[chat_id] = chat_histories[chat_id][-MAX_HISTORY:]

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
                    message_executor.submit(handle_message, chat_id, text, time.time())
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

@app.api_route("/health", methods=["GET", "HEAD"])
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