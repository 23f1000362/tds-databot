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
from groq import Groq

# ---------- CONFIG ----------
BOT_TOKEN = os.environ["BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
BASE_URL = os.environ["BASE_URL"]

client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-flash-lite-latest"

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"

DAILY_GEMINI_LIMIT = 500

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

LOG_FILE = "run.jsonl"
log_lock = threading.Lock()


def log_event(event: dict):
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")


# ---------- GEMINI DAILY QUOTA ----------
usage_lock = threading.Lock()
usage_state = {"date": None, "count": 0}


def gemini_quota_available() -> bool:
    today = datetime.now(timezone.utc).date().isoformat()
    with usage_lock:
        if usage_state["date"] != today:
            usage_state["date"] = today
            usage_state["count"] = 0
        if usage_state["count"] >= DAILY_GEMINI_LIMIT:
            return False
        usage_state["count"] += 1
        return True


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


# ---------- SYSTEM PROMPT (shared by both providers) ----------
SYSTEM_PROMPT = f"""You are a data-analyst agent replying to Telegram messages.

Rules:
- Answer the LATEST message in the conversation. Earlier messages are context for multi-turn questions.
- Use the run_python tool to fetch data and compute answers. Never guess a number you could compute.
- For general published facts that are genuinely static and non-time-sensitive (e.g. "what is the capital of X", "who wrote X"), you may answer from your own trained knowledge if fetching genuinely fails after retrying. Do NOT use this for statistics, rankings, or rates that are periodically updated (census data, health/economic indicators, "highest/lowest X" rankings) — these must go through the official-source rules below, since your trained knowledge of these can be stale even when you're confident.
- Never invent a specific number, statistic, or live value (like a current price, live measurement, or exact computed figure) — for these, if fetching fails after retries, say so explicitly rather than fabricating a number.
- Your FINAL reply must be ONLY a single JSON object, and nothing else - no markdown fences, no prose, no "here is the answer".
- Match the JSON shape the question asks for EXACTLY (same keys, same nesting, correct types - number vs string).
- Always include a "log_url" key in your final JSON with the exact value "{BASE_URL}/run.jsonl".
- If a message is just setup/context (e.g. "I will send data next"), still reply with a minimal valid JSON acknowledgment.
- Never add extra keys beyond what's asked.
- When fetching external data, if a URL/API fails, try at least one alternative approach (a different URL structure, a cached mirror, or re-reading the question for an explicit source) before giving up.
- Never guess or hardcode a fallback answer when a fetch fails. If, after reasonable attempts, you cannot retrieve real data, say so honestly in your final answer rather than fabricating a plausible-looking one — a wrong answer that admits uncertainty is safer than a confident guess that happens to be checked against a real value.
- Prefer data sources explicitly mentioned or linked in the question over ones you recall from training, since your training knowledge of specific APIs/endpoints may be outdated.
- Do not guess more than 2-3 speculative URLs for a dataset. If official sources aren't findable that way, broaden to a regular web search rather than continuing to guess URLs — do not skip straight to trained knowledge for time-sensitive statistics (see rules on official sources below).
- For official/government statistics questions (census, health ministry data, economic indicators, rankings, etc. from any country), search for the specific official source or bulletin directly (e.g. "site:gov domain name of the report") rather than broad generic keyword searches — broad searches often blend results from unrelated countries/regions or multiple time periods into one messy result set.
- If your search results contain data from multiple time periods or reporting cycles, always use the most recently published one and explicitly discard older figures. Do not average conflicting numbers or default to an older "commonly known" ranking once you have newer data.
- If you already have clear, recent, sourced data that contradicts a fact you'd otherwise recall from training, trust the newer sourced data over your training knowledge.
- If after 3-4 focused search steps you still don't have one clean, single-source, unambiguous answer, say so explicitly in your final answer rather than picking one number out of a mix of conflicting snippets.
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

config = types.GenerateContentConfig(
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
                candidate = text[start:i + 1]
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


# ---------- GEMINI AGENT LOOP ----------
def run_gemini_agent(history: list, chat_id: int, convo_log: list):
    contents = []
    for h in history:
        role = "user" if h["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=h["parts"][0])]))

    max_steps = 10
    step = 0
    response_text = None
    response = None

    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
    except Exception as e:
        convo_log.append({"event": "gemini_exception", "error": str(e), "trace": traceback.format_exc()})
        return None

    while response is not None and step < max_steps:
        step += 1

        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
            convo_log.append({"event": "gemini_empty_response", "step": step})
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

            try:
                response = client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
            except Exception as e:
                convo_log.append({"event": "gemini_exception", "error": str(e), "trace": traceback.format_exc()})
                response = None
                break
        else:
            response_text = safe_text(response)
            convo_log.append({"event": "final_text", "step": step, "text": response_text})
            break

    if response_text is None and step >= max_steps:
        convo_log.append({"event": "max_steps_exceeded", "step": step})
        try:
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text="You have used too many steps. Stop trying new approaches and answer NOW with only the final JSON object, using your best available information.")]
            ))
            response = client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
            response_text = safe_text(response)
            convo_log.append({"event": "forced_final_after_max_steps", "text": response_text})
        except Exception as e:
            convo_log.append({"event": "forced_answer_failed", "error": str(e)})

    if response_text is None and response is not None:
        response_text = safe_text(response)

    return response_text


# ---------- GROQ AGENT LOOP (fallback) ----------
def run_groq_agent(history: list, chat_id: int, convo_log: list):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        role = "user" if h["role"] == "user" else "assistant"
        messages.append({"role": role, "content": h["parts"][0]})

    max_steps = 10
    step = 0
    response_text = None

    while step < max_steps:
        step += 1
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=groq_tools,
                tool_choice="auto",
            )
        except Exception as e:
            convo_log.append({"event": "groq_exception", "error": str(e), "trace": traceback.format_exc()})
            break

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
                convo_log.append({"event": "groq_tool_call", "step": step, "code": code})
                result = run_python(code)
                convo_log.append({"event": "groq_tool_result", "step": step, "output": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            response_text = msg.content
            convo_log.append({"event": "groq_final_text", "step": step, "text": response_text})
            break

    if response_text is None:
        convo_log.append({"event": "groq_max_steps_exceeded", "step": step})
        try:
            messages.append({"role": "user", "content": "Stop trying new approaches and answer NOW with only the final JSON object, using your best available information."})
            resp = groq_client.chat.completions.create(model=GROQ_MODEL, messages=messages)
            response_text = resp.choices[0].message.content
            convo_log.append({"event": "groq_forced_final", "text": response_text})
        except Exception as e:
            convo_log.append({"event": "groq_forced_answer_failed", "error": str(e)})

    return response_text


# ---------- ORCHESTRATOR ----------
def run_agent(history: list, chat_id: int) -> dict:
    convo_log = []
    response_text = None
    provider_used = "gemini"

    if gemini_quota_available():
        response_text = run_gemini_agent(history, chat_id, convo_log)
    else:
        convo_log.append({"event": "gemini_quota_exhausted_skipping"})

    if response_text is None or extract_json(response_text) is None:
        convo_log.append({"event": "falling_back_to_groq"})
        provider_used = "groq"
        response_text = run_groq_agent(history, chat_id, convo_log)

    for entry in convo_log:
        log_event({"chat_id": chat_id, **entry})

    parsed = extract_json(response_text) if response_text else None

    if parsed is None:
        parsed = {"answer": "internal error", "log_url": "LOG_URL_PLACEHOLDER"}
    elif "answer" not in parsed:
        parsed = {"answer": parsed, "log_url": "LOG_URL_PLACEHOLDER"}

    parsed["log_url"] = f"{BASE_URL}/run.jsonl"
    log_event({"chat_id": chat_id, "event": "provider_used", "provider": provider_used})
    return parsed


# ---------- TELEGRAM POLLING ----------
chat_histories = defaultdict(list)
MAX_HISTORY = 20


def telegram_send(chat_id, text):
    try:
        requests.post(f"{TELEGRAM_API}/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=15)
    except Exception as e:
        log_event({"chat_id": chat_id, "event": "telegram_send_failed", "error": str(e)})


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


# Dedup guard: prevents double-replies if two instances/pollers ever
# end up hitting getUpdates before the offset commits (this is what
# caused the earlier duplicate/incorrect-answer incident).
processed_updates = set()
processed_lock = threading.Lock()


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
                uid = update["update_id"]
                with processed_lock:
                    if uid in processed_updates:
                        continue
                    processed_updates.add(uid)
                    if len(processed_updates) > 5000:
                        processed_updates.clear()
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
