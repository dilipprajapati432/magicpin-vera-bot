"""
server.py — magicpin AI Challenge HTTP bot server
Exposes the 5 required endpoints with exact schema compliance.
Run: uvicorn server:app --host 0.0.0.0 --port 8080
Env: GROQ_API_KEY (optional — falls back to key embedded in bot.py)
"""

import os, time, uuid, logging
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any

import bot as Bot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vera-bot")

app = FastAPI(title="Vera — magicpin AI Challenge Bot", version="1.0.0")

START_TIME = time.time()

# ── State ─────────────────────────────────────────────────────────────────────

# (scope, context_id) -> { version, payload }
contexts: dict[tuple[str, str], dict] = {}

# conversation_id -> ConversationState
conversations: dict[str, Bot.ConversationState] = {}

# suppression set — already-sent suppression keys
sent_suppressions: set[str] = set()


# ── Utils ─────────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.") + f"{datetime.utcnow().microsecond // 1000:03d}Z"

def ctx_counts() -> dict:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        counts[scope] = counts.get(scope, 0) + 1
    return counts

def get_payload(scope: str, context_id: str) -> dict | None:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


# ── GET /v1/healthz ────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": ctx_counts(),
    }


# ── GET /v1/metadata ──────────────────────────────────────────────────────────

@app.get("/v1/metadata")
def metadata():
    return {
        "team_name": "Vera Engine",
        "team_members": ["Candidate"],
        "model": "llama-3.3-70b-versatile (Groq)",
        "approach": (
            "Claude-powered 4-context composer with per-trigger-kind prompt dispatch, "
            "auto-reply detection, intent transition routing, and anti-repetition guard."
        ),
        "contact_email": "candidate@example.com",
        "version": "1.0.0",
        "submitted_at": "2026-05-02T00:00:00Z",
    }


# ── POST /v1/context ──────────────────────────────────────────────────────────

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str = ""

@app.post("/v1/context")
async def push_context(body: ContextBody):
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return JSONResponse(status_code=400, content={
            "accepted": False, "reason": "invalid_scope",
            "details": f"scope must be one of {valid_scopes}"
        })

    key = (body.scope, body.context_id)
    existing = contexts.get(key)

    if existing:
        if existing["version"] > body.version:
            # Stale: we already have a newer version
            return JSONResponse(status_code=409, content={
                "accepted": False,
                "reason": "stale_version",
                "current_version": existing["version"],
            })
        if existing["version"] == body.version:
            # Idempotent re-push: no-op, return success
            return {
                "accepted": True,
                "ack_id": existing.get("ack_id", f"ack_{body.context_id}_v{body.version}"),
                "stored_at": existing.get("stored_at", now_iso()),
            }

    # Store/replace
    ack_id    = f"ack_{body.context_id}_v{body.version}"
    stored_at = now_iso()
    contexts[key] = {
        "version": body.version,
        "payload": body.payload,
        "ack_id": ack_id,
        "stored_at": stored_at,
    }

    logger.info(f"Context stored: scope={body.scope} id={body.context_id} v{body.version}")
    return {"accepted": True, "ack_id": ack_id, "stored_at": stored_at}


# ── POST /v1/tick ─────────────────────────────────────────────────────────────

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []

@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []

    for trg_id in body.available_triggers:
        # Reduced from 20 to 5 to avoid 30-second timeout and Groq 12k TPM limits
        if len(actions) >= 5:
            break

        trg_payload = get_payload("trigger", trg_id)
        if not trg_payload:
            logger.warning(f"Trigger {trg_id} not in context store — skipping")
            continue

        suppression_key = trg_payload.get("suppression_key", "")
        if suppression_key and suppression_key in sent_suppressions:
            logger.info(f"Suppressing duplicate: {suppression_key}")
            continue

        # Safely extract IDs which might be top-level, named target_*, or nested inside a "payload" object
        merchant_id = trg_payload.get("merchant_id") or trg_payload.get("target_merchant_id")
        if not merchant_id and isinstance(trg_payload.get("payload"), dict):
            merchant_id = trg_payload["payload"].get("merchant_id") or trg_payload["payload"].get("target_merchant_id")

        customer_id = trg_payload.get("customer_id") or trg_payload.get("target_customer_id")
        if not customer_id and isinstance(trg_payload.get("payload"), dict):
            customer_id = trg_payload["payload"].get("customer_id") or trg_payload["payload"].get("target_customer_id")

        if not merchant_id:
            continue

        merchant = get_payload("merchant", merchant_id)
        if not merchant:
            logger.warning(f"Merchant {merchant_id} not in context store")
            continue

        category_slug = merchant.get("category_slug", "")
        category = get_payload("category", category_slug)
        if not category:
            logger.warning(f"Category {category_slug} not in context store")
            continue

        customer = get_payload("customer", customer_id) if customer_id else None

        # Build trigger dict for bot (trigger payload IS the trigger object here)
        trigger = trg_payload  # already the full trigger context

        try:
            result = Bot.compose(category, merchant, trigger, customer)
        except Exception as e:
            logger.error(f"Compose error for {merchant_id}/{trg_id}: {e}")
            continue

        body_text = result.get("body", "")
        if not body_text:
            continue

        # Track suppression
        if suppression_key:
            sent_suppressions.add(suppression_key)

        # Generate conversation_id (unique per merchant+trigger)
        conv_id = f"conv_{merchant_id}_{trg_id}_{uuid.uuid4().hex[:6]}"

        # Create conversation state for future replies
        state = Bot.ConversationState(
            conversation_id=conv_id,
            merchant=merchant,
            category=category,
            trigger=trigger,
            customer=customer,
        )
        state.add_turn("vera", body_text)
        conversations[conv_id] = state

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": result.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
            "template_params": [
                merchant.get("identity", {}).get("owner_first_name") or merchant.get("identity", {}).get("name", ""),
                trigger.get("kind", ""),
                body_text[:50],
            ],
            "body": body_text,
            "cta": result.get("cta", "open_ended"),
            "suppression_key": suppression_key,
            "rationale": result.get("rationale", ""),
        }
        actions.append(action)
        logger.info(f"Action queued: conv={conv_id} merchant={merchant_id} kind={trigger.get('kind')}")

    return {"actions": actions}


# ── POST /v1/reply ────────────────────────────────────────────────────────────

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str = ""
    turn_number: int = 1

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id
    state = conversations.get(conv_id)

    # If we don't have state (e.g., new conversation the judge started), build it
    if not state:
        if body.merchant_id:
            merchant  = get_payload("merchant", body.merchant_id) or {}
            cat_slug  = merchant.get("category_slug", "")
            category  = get_payload("category", cat_slug) or {}
            customer  = get_payload("customer", body.customer_id) if body.customer_id else None
            trigger   = {}  # no trigger context for ad-hoc reply
            state = Bot.ConversationState(conv_id, merchant, category, trigger, customer)
            conversations[conv_id] = state
        else:
            return {"action": "end", "rationale": "Unknown conversation and no merchant_id provided."}

    if state.ended:
        return {"action": "end", "rationale": "Conversation already ended."}

    try:
        result = Bot.respond(state, body.message, body.from_role)
    except Exception as e:
        logger.error(f"Reply error conv={conv_id}: {e}")
        return {"action": "send", "body": "Got it — let me get back to you shortly.", "cta": "none", "rationale": f"Error handled: {e}"}

    # Track Vera's reply in state
    if result.get("action") == "send" and result.get("body"):
        state.add_turn("vera", result["body"])

    logger.info(f"Reply handled: conv={conv_id} action={result.get('action')}")
    return result
