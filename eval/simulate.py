"""Multi-turn replay evaluator for the SHL Assessment Recommender.

Unlike eval/evaluate.py (which sends only the first user message), this
harness mirrors the REAL evaluator: an LLM-simulated user holds a multi-turn
conversation against POST /chat, answering the agent's questions from a
persona's facts and ending when the agent commits to a shortlist. It then
computes Recall@10 against each trace's labeled final shortlist.

Usage:
    # 1. start the service (with a real GEMINI_API_KEY):
    uvicorn app.main:app --port 8000
    # 2. run the simulator (needs GEMINI_API_KEY in env for the simulated user):
    python eval/simulate.py --base-url http://localhost:8000 --max-turns 8

If GEMINI_API_KEY is not set, the simulated user degrades to a scripted
replay that simply sends each of the trace's real user messages in order —
still multi-turn, just non-adaptive.
"""

import os
import re
import sys
import json
import argparse
import httpx

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACES_DIR = os.path.join(PROJECT_ROOT, "eval", "traces", "GenAI_SampleConversations")
CATALOG_PATH = os.path.join(PROJECT_ROOT, "data", "catalog.json")

try:
    import google.generativeai as genai
    _HAVE_GENAI = True
except Exception:
    _HAVE_GENAI = False

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
_USE_LLM_USER = _HAVE_GENAI and bool(GEMINI_KEY)
if _USE_LLM_USER:
    genai.configure(api_key=GEMINI_KEY)


# ── trace parsing ────────────────────────────────────────────────
def parse_trace(filepath: str) -> dict:
    content = open(filepath).read()
    user_blocks = re.findall(
        r"\*\*User\*\*\s*\n([\s\S]*?)(?=\n\*\*Agent\*\*|\Z)", content
    )
    user_messages = []
    for block in user_blocks:
        lines = [
            l.lstrip("> ").strip()
            for l in block.strip().split("\n")
            if l.strip().startswith(">")
        ]
        if lines:
            user_messages.append(" ".join(lines))

    tables = list(re.finditer(r"\| # \| Name.*?\n(?:\|.*\n)+", content))
    expected = []
    if tables:
        expected = re.findall(
            r"<(https://www\.shl\.com/products/product-catalog/view/[^>]+)>",
            tables[-1].group(),
        )
    return {
        "user_messages": user_messages,
        "expected_urls": list({u.rstrip("/").lower() for u in expected}),
    }


# ── simulated user ───────────────────────────────────────────────
_USER_SYS = (
    "You are simulating a hiring manager talking to an assessment recommender. "
    "Here are your FACTS (the things you know about your hiring need):\n{facts}\n\n"
    "Rules:\n"
    "- Answer the agent's questions truthfully and briefly using ONLY your facts.\n"
    "- If asked something not covered by your facts, say you have no preference.\n"
    "- Do NOT volunteer the whole brief at once; respond like a real person.\n"
    "- Reply with a single short user message, no quotes, no role labels."
)


def simulated_user_reply(facts: str, history: list[dict]) -> str:
    """Generate the next user turn given the agent's latest reply."""
    if not _USE_LLM_USER:
        return ""  # caller falls back to scripted messages
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=_USER_SYS.format(facts=facts),
        generation_config=genai.types.GenerationConfig(
            temperature=0.4, max_output_tokens=128
        ),
    )
    try:
        return model.generate_content(
            convo + "\n\nYour next message:"
        ).text.strip()
    except Exception:
        return "No strong preference — your recommendation is fine."


def run_conversation(base_url: str, trace: dict, max_turns: int) -> dict:
    """Drive a multi-turn conversation and return the final shortlist + metrics."""
    facts = " ".join(trace["user_messages"])
    scripted = list(trace["user_messages"])
    messages: list[dict] = [{"role": "user", "content": scripted[0]}]
    last_recs: list[dict] = []
    turns = 0
    schema_ok_all = True

    while turns < max_turns:
        turns += 1
        try:
            resp = httpx.post(
                f"{base_url}/chat", json={"messages": messages}, timeout=30
            )
            data = resp.json()
        except Exception as e:
            return {"error": str(e), "turns": turns, "recs": last_recs,
                    "schema_ok": False}

        for f in ("reply", "recommendations", "end_of_conversation"):
            if f not in data:
                schema_ok_all = False
        recs = data.get("recommendations", []) or []
        if recs:
            last_recs = recs

        messages.append({"role": "assistant", "content": data.get("reply", "")})

        # Stop when the agent commits to a shortlist or signals completion.
        if recs or data.get("end_of_conversation"):
            break

        # Otherwise the simulated user answers and the loop continues.
        if _USE_LLM_USER:
            nxt = simulated_user_reply(facts, messages)
        else:
            nxt = scripted[turns] if turns < len(scripted) else \
                "No strong preference — please recommend now."
        messages.append({"role": "user", "content": nxt})

    return {"turns": turns, "recs": last_recs, "schema_ok": schema_ok_all}


def recall_at_10(recs: list[dict], expected: list[str]) -> float:
    if not expected:
        return 1.0
    got = {r.get("url", "").rstrip("/").lower() for r in recs[:10]}
    return len(got & set(expected)) / len(expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--max-turns", type=int, default=8)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    catalog = json.load(open(CATALOG_PATH))
    catalog_urls = {c["url"].rstrip("/").lower() for c in catalog}

    mode = "LLM-simulated user" if _USE_LLM_USER else "scripted replay"
    print(f"\n{'='*60}\n  Multi-turn evaluation ({mode})\n  Target: {base}\n{'='*60}\n")

    total = schema_pass = url_pass = n = 0.0
    for i in range(1, 11):
        fp = os.path.join(TRACES_DIR, f"C{i}.md")
        if not os.path.exists(fp):
            continue
        trace = parse_trace(fp)
        out = run_conversation(base, trace, args.max_turns)
        recs = out.get("recs", [])
        r = recall_at_10(recs, trace["expected_urls"])
        in_cat = all(
            x.get("url", "").rstrip("/").lower() in catalog_urls for x in recs
        )
        total += r
        schema_pass += 1 if out.get("schema_ok") else 0
        url_pass += 1 if in_cat else 0
        n += 1
        bar = "█" * int(r * 20) + "░" * (20 - int(r * 20))
        print(f"  C{i:<2} {bar} {r:.2f}  recs={len(recs):<2} turns={out.get('turns')} "
              f"schema={'ok' if out.get('schema_ok') else 'BAD'} "
              f"catalog={'ok' if in_cat else 'BAD'}")

    if n:
        print(f"\n{'='*60}")
        print(f"  Mean Recall@10:    {total/n:.3f}")
        print(f"  Schema compliance: {int(schema_pass)}/{int(n)}")
        print(f"  URLs from catalog: {int(url_pass)}/{int(n)}")
        print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
