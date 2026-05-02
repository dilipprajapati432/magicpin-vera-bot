"""
bot.py — Magicpin AI Challenge submission
Implements: compose(category, merchant, trigger, customer) -> dict
            respond(state, merchant_message) -> dict

Strategy:
- Single LLM call per composition with all 4 contexts fully structured in prompt
- Trigger-kind dispatch: different prompt framing per trigger kind for better specificity
- Post-LLM validation: CTA shape, body length, anti-repetition guard
- Multi-turn: tracks conversation state, detects auto-replies, handles intent transitions
"""

import os, re, json
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.3-70b-versatile"   # fast, free, 70B — best Groq option for structured output

# ── Helpers ────────────────────────────────────────────────────────────────────

def _name(merchant: dict) -> str:
    """Owner first name or clinic/biz name — whichever is most personal."""
    owner = merchant.get("identity", {}).get("owner_first_name")
    if owner:
        return owner
    return merchant.get("identity", {}).get("name", "there")

def _lang(merchant: dict, customer: dict | None = None) -> str:
    if customer:
        return customer.get("identity", {}).get("language_pref", "en")
    langs = merchant.get("identity", {}).get("languages", ["en"])
    if "hi" in langs:
        return "hi-en mix"
    return "en"

def _find_digest_item(category: dict, item_id: str) -> dict | None:
    for item in category.get("digest", []):
        if item.get("id") == item_id:
            return item
    return None

def _active_offers(merchant: dict) -> list[dict]:
    return [o for o in merchant.get("offers", []) if o.get("status") == "active"]

def _signals(merchant: dict) -> list[str]:
    return merchant.get("signals", [])

def _prior_bodies(history: list[dict]) -> list[str]:
    return [h.get("body", "") for h in history if h.get("from") == "vera"]


# ── Auto-reply detection ───────────────────────────────────────────────────────

AUTO_REPLY_PATTERNS = [
    r"thank you for (contacting|reaching out|your message)",
    r"i (am|'m) an? (automated|automatic) (assistant|reply|response)",
    r"this is an? (automated|automatic)",
    r"aapki (madad|jaankari) ke liye",
    r"bahut.bahut shukriya",
    r"main aapki.+team tak pahuncha",
    r"out of office",
    r"currently unavailable",
    r"will get back to you",
]

def is_auto_reply(message: str) -> bool:
    msg_lower = message.lower()
    return any(re.search(p, msg_lower) for p in AUTO_REPLY_PATTERNS)


# ── Intent detection ───────────────────────────────────────────────────────────

POSITIVE_INTENT = [
    r"\byes\b", r"\bha(n|in)\b", r"\bhaa\b", r"\bok\b", r"\bsure\b", r"\blet'?s? (do|go)\b",
    r"sounds good", r"go ahead", r"please (send|draft|do|proceed|update)",
    r"karo", r"kar do", r"chalega", r"theek hai", r"bahut accha",
    r"i('d| would) like", r"interested", r"great idea",
]

NEGATIVE_INTENT = [
    r"\bno\b", r"\bnahi\b", r"\bnot interested\b", r"stop", r"unsubscribe",
    r"don'?t (want|need|send)", r"please (stop|remove)", r"not now",
    r"abhi nahi", r"nahin chahiye",
]

def detect_intent(message: str) -> str:
    msg = message.lower()
    if any(re.search(p, msg) for p in POSITIVE_INTENT):
        return "positive"
    if any(re.search(p, msg) for p in NEGATIVE_INTENT):
        return "negative"
    return "neutral"


# ── Main compose function ──────────────────────────────────────────────────────

def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
) -> dict:
    """
    Compose a WhatsApp message from 4 contexts using Claude.
    Returns: { body, cta, send_as, suppression_key, rationale }
    """
    kind      = trigger.get("kind", "generic")
    scope     = trigger.get("scope", "merchant")
    send_as   = "merchant_on_behalf" if (scope == "customer" and customer) else "vera"
    lang      = _lang(merchant, customer)
    signals   = _signals(merchant)
    offers    = _active_offers(merchant)
    history   = merchant.get("conversation_history", [])
    prior     = _prior_bodies(history)

    # Resolve digest item if trigger references one
    digest_item = None
    top_item_id = trigger.get("payload", {}).get("top_item_id")
    if top_item_id:
        digest_item = _find_digest_item(category, top_item_id)

    # Build the structured prompt
    system = _build_system(category, lang)
    user   = _build_user_prompt(
        category, merchant, trigger, customer,
        kind, send_as, lang, signals, offers, prior, digest_item
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=600,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON output
    try:
        # Strip markdown fences if present
        clean = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(clean)
    except Exception:
        # Fallback: treat whole response as body
        result = {
            "body": raw,
            "cta": "open_ended",
            "rationale": f"Composed for trigger kind={kind}",
        }

    # Enforce suppression_key from trigger
    result["suppression_key"] = trigger.get("suppression_key", f"{kind}:{merchant.get('merchant_id', 'unk')}")
    result["send_as"] = send_as

    # Validate CTA
    valid_ctas = {"binary_yes_stop", "open_ended", "none"}
    if result.get("cta") not in valid_ctas:
        result["cta"] = "open_ended"

    # Anti-repetition: if body is too similar to a prior message, ask Claude to vary it
    body = result.get("body", "")
    if prior and any(_similarity(body, p) > 0.7 for p in prior):
        result["body"] = _vary_body(body, prior, system)

    return result


def _similarity(a: str, b: str) -> float:
    """Simple token-overlap similarity."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _vary_body(body: str, prior: list[str], system: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"The following message is too similar to one already sent:\n\n{body}\n\n"
                    f"Prior sent messages:\n" + "\n---\n".join(prior[-3:]) +
                    "\n\nRewrite the message to cover the same topic but with clearly different wording and structure. "
                    "Return only the new message body, no JSON wrapper."
                ),
            },
        ],
    )
    return resp.choices[0].message.content.strip()


def _build_system(category: dict, lang: str) -> str:
    voice = category.get("voice", {})
    tone = voice.get("tone", "professional")
    taboos = ", ".join(voice.get("vocab_taboo", []))
    allowed = ", ".join(voice.get("vocab_allowed", [])[:8])
    code_mix_note = ""
    if "hi" in lang:
        code_mix_note = (
            "Use natural Hindi-English code-mix where it feels authentic. "
            "Do NOT force Hindi on a clearly English-preferring merchant. "
            "Hinglish examples: 'aapka profile', 'yeh batao', 'chalega', 'theek hai'."
        )
    return f"""You are Vera, magicpin's AI merchant assistant. You compose WhatsApp messages to merchants and their customers.

VOICE: {tone}. Peer/colleague register — NOT promotional. {code_mix_note}
ALLOWED VOCAB: {allowed}
TABOO WORDS (never use): {taboos}

RULES:
1. Ground every claim in the context data. No invented numbers, no fake citations, no fabricated offers.
2. Single primary CTA. Binary (Reply YES / STOP) for action triggers. open_ended for info/curiosity. none for pure-info.
3. No preambles like "I hope you're well" or "I'm reaching out today to".
4. No re-introductions after the first message.
5. Specificity wins: anchor on a verifiable number, date, stat, or source citation from the data.
6. Keep body concise: 40-100 words for merchant messages, 50-120 for customer messages.
7. CTA goes in the last sentence.

OUTPUT FORMAT (JSON only, no markdown fences):
{{
  "body": "<the WhatsApp message>",
  "cta": "binary_yes_stop" | "open_ended" | "none",
  "rationale": "<1-2 sentences: why this message, what compulsion lever>"
}}"""


def _build_user_prompt(
    category, merchant, trigger, customer,
    kind, send_as, lang, signals, offers, prior, digest_item
) -> str:

    cat_slug    = category.get("slug", "")
    peer_stats  = category.get("peer_stats", {})
    perf        = merchant.get("performance", {})
    cust_agg    = merchant.get("customer_aggregate", {})
    identity    = merchant.get("identity", {})
    sub         = merchant.get("subscription", {})
    trg_payload = trigger.get("payload", {})

    merchant_block = f"""MERCHANT:
- ID: {merchant.get('merchant_id')}
- Name: {identity.get('name')} | Owner: {identity.get('owner_first_name', 'N/A')}
- City: {identity.get('city')} | Locality: {identity.get('locality')}
- Verified: {identity.get('verified')} | Languages: {identity.get('languages')}
- Subscription: {sub.get('plan')} plan, {sub.get('days_remaining')} days remaining, status={sub.get('status')}
- Performance (30d): views={perf.get('views')}, calls={perf.get('calls')}, CTR={perf.get('ctr')} (peer median={peer_stats.get('avg_ctr')})
  7d delta: views {perf.get('delta_7d',{}).get('views_pct',0):+.0%}, calls {perf.get('delta_7d',{}).get('calls_pct',0):+.0%}
- Customer aggregate: {cust_agg.get('total_unique_ytd')} unique YTD, {cust_agg.get('lapsed_180d_plus')} lapsed 180d+, {cust_agg.get('retention_6mo_pct',0):.0%} 6-mo retention
- Active offers: {'; '.join(o['title'] for o in offers) if offers else 'none'}
- Signals: {', '.join(signals) if signals else 'none'}"""

    trigger_block = f"""TRIGGER:
- ID: {trigger.get('id')} | Kind: {kind} | Urgency: {trigger.get('urgency')}/5
- Source: {trigger.get('source')} | Scope: {trigger.get('scope')}
- Payload: {json.dumps(trg_payload, ensure_ascii=False)}"""

    if digest_item:
        trigger_block += f"""
- Resolved digest item: title="{digest_item.get('title')}", source="{digest_item.get('source')}", trial_n={digest_item.get('trial_n')}, patient_segment={digest_item.get('patient_segment')}"""

    category_block = f"""CATEGORY: {cat_slug}
- Peer stats: avg_rating={peer_stats.get('avg_rating')}, avg_ctr={peer_stats.get('avg_ctr')}
- Seasonal beats: {json.dumps(category.get('seasonal_beats', [])[:2], ensure_ascii=False)}
- Top trend signals: {json.dumps(category.get('trend_signals', [])[:2], ensure_ascii=False)}
- Sample offer catalog: {'; '.join(o['title'] for o in category.get('offer_catalog', [])[:4])}"""

    customer_block = ""
    if customer:
        cust_id   = customer.get("identity", {})
        rel       = customer.get("relationship", {})
        prefs     = customer.get("preferences", {})
        state     = customer.get("state", "unknown")
        customer_block = f"""
CUSTOMER:
- Name: {cust_id.get('name')} | Language: {cust_id.get('language_pref')}
- State: {state} | Last visit: {rel.get('last_visit')} | Total visits: {rel.get('visits_total')}
- Services received: {rel.get('services_received')}
- Preferences: {prefs.get('preferred_slots')} slots, channel={prefs.get('channel')}
- Consent scope: {customer.get('consent', {}).get('scope')}"""

    prior_block = ""
    if prior:
        prior_block = f"\nPRIOR MESSAGES SENT (avoid repeating):\n" + "\n---\n".join(prior[-3:])

    instruction = _kind_instruction(kind, send_as, trg_payload, digest_item, merchant, customer, signals, offers, category, perf, peer_stats, cust_agg)

    return f"""{category_block}

{merchant_block}

{trigger_block}
{customer_block}
{prior_block}

TASK:
{instruction}

Compose now. Output valid JSON only."""


def _kind_instruction(kind, send_as, trg_payload, digest_item, merchant, customer, signals, offers, category, perf, peer_stats, cust_agg):
    """Per-trigger-kind instruction sharpens the prompt for specificity."""
    identity = merchant.get("identity", {})
    name = identity.get("owner_first_name") or identity.get("name")

    if kind == "research_digest":
        item = digest_item or {}
        return (
            f"Write a merchant-facing Vera message about this research digest item. "
            f"Lead with the specific finding (trial_n={item.get('trial_n')}, stat, source). "
            f"Connect it to THIS merchant's patient/customer cohort using the signals. "
            f"End with a single low-friction offer: e.g., 'Want me to pull it + draft a patient-ed WhatsApp?'"
        )

    elif kind == "regulation_change":
        deadline = trg_payload.get("deadline_iso", "")
        return (
            f"Write a merchant-facing Vera message about a regulatory change. "
            f"State the specific change, deadline ({deadline}), and what the merchant needs to do. "
            f"Frame as: 'here's what it means for you + I can help with X'."
        )

    elif kind in ("recall_due", "recall_reminder"):
        slots = trg_payload.get("available_slots", [])
        slot_str = " or ".join(s.get("label", "") for s in slots[:2]) if slots else ""
        last_service = trg_payload.get("last_service_date", "")
        return (
            f"Write a customer-facing message (sent from merchant's WhatsApp) reminding the customer "
            f"their service recall is due. Include last visit date ({last_service}), "
            f"available slots ({slot_str}), price from active offers if available. "
            f"Match customer's language preference exactly. End with slot-choice CTA."
        )

    elif kind == "perf_dip":
        delta = perf.get("delta_7d", {})
        return (
            f"Write a merchant-facing Vera message. Their performance dipped: {delta}. "
            f"Peer CTR is {peer_stats.get('avg_ctr')} vs their {perf.get('ctr')}. "
            f"Don't just state the problem — give ONE specific, actionable recommendation grounded in data. "
            f"If the dip is seasonal, say so with the benchmark range."
        )

    elif kind == "perf_spike":
        delta = perf.get("delta_7d", {})
        return (
            f"Write a merchant-facing Vera message celebrating a performance spike: {delta}. "
            f"Acknowledge the specific number. Immediately turn it into an opportunity: "
            f"'momentum is high — want me to push X to convert the traffic?'"
        )

    elif kind in ("milestone_reached", "milestone"):
        return (
            f"Write a merchant-facing Vera message celebrating a milestone. "
            f"Name the specific milestone from the trigger payload. "
            f"Connect it to a next-step: 'this is a great moment to do X'. Use social proof if relevant."
        )

    elif kind in ("ipl_match", "festival_upcoming", "local_event"):
        return (
            f"Write a merchant-facing Vera message about an upcoming event/trigger. "
            f"Don't just announce the event — give a specific recommendation on how to leverage it "
            f"(or avoid a common mistake). Include a concrete deliverable offer. "
            f"Use category-appropriate tone: restaurants=operator, gyms=coach, etc."
        )

    elif kind in ("bridal_followup", "appointment_tomorrow"):
        slots = trg_payload.get("available_slots", [])
        slot_str = " or ".join(s.get("label", "") for s in slots[:2]) if slots else ""
        return (
            f"Write a customer-facing message for appointment followup/reminder. "
            f"Be warm and specific: reference the appointment date/service, slots ({slot_str}). "
            f"Keep it under 80 words. End with confirmation CTA."
        )

    elif kind in ("curious_ask_due", "curious_ask", "scheduled_recurring"):
        return (
            f"Write a merchant-facing Vera message for the 'curious ask' cadence — Vera asks the merchant "
            f"a question about their business this week. This should be a low-stakes, engaging question "
            f"(e.g., 'what service has been most asked-for this week?') with a reciprocal offer "
            f"('I'll turn the answer into X — takes 5 min')."
        )

    elif kind in ("dormant_with_vera", "dormancy"):
        return (
            f"Write a merchant-facing Vera message to re-engage a dormant merchant. "
            f"Don't guilt-trip. Lead with something new or valuable — a stat, a question, a quick win. "
            f"Keep it very short (under 50 words). Single open-ended CTA."
        )

    elif kind in ("winback", "customer_lapsed_hard", "customer_lapsed_soft"):
        if customer:
            return (
                f"Write a customer-facing winback message. No shame, no guilt. "
                f"Reference their past services specifically. Offer something concrete (new class, offer, slot). "
                f"'No commitment' framing removes the barrier. Single binary CTA."
            )
        return (
            f"Write a merchant-facing message about lapsed customers. "
            f"Give the exact count ({cust_agg.get('lapsed_180d_plus')} lapsed 180d+). "
            f"Offer to draft a winback campaign."
        )

    elif kind in ("active_planning_intent", "planning_intent"):
        return (
            f"The merchant has signalled planning intent (see trigger payload). "
            f"Don't ask qualifying questions — they said yes. Deliver a complete draft artifact. "
            f"Structure it clearly, use real data from merchant context. "
            f"End with a follow-up offer (e.g., draft outreach to targets)."
        )

    elif kind in ("competitor_opened", "competitor"):
        return (
            f"Write a merchant-facing message about a new competitor. "
            f"Don't alarm them — reframe as an opportunity. "
            f"Give a specific recommendation (e.g., 'best time to refresh your GBP profile'). "
            f"Anchor on what THIS merchant does better using their signals."
        )

    elif kind in ("chronic_refill_due", "refill_reminder"):
        meds = trg_payload.get("medications", trg_payload.get("items", []))
        run_out_date = trg_payload.get("run_out_date", trg_payload.get("due_date", ""))
        return (
            f"Write a customer-facing pharmacy refill reminder. "
            f"Name the medications ({meds}), run-out date ({run_out_date}). "
            f"Include total cost + savings if senior discount applies. "
            f"Respectful tone. Give two contact options (Reply + Call). "
        )

    elif kind in ("supply_alert", "compliance_alert"):
        return (
            f"Write an urgent merchant-facing Vera message about a supply/compliance alert. "
            f"Be specific: batch numbers, manufacturer, what the merchant must do. "
            f"Derive how many of their customers are affected from the aggregate data. "
            f"Offer to draft customer WhatsApps + workflow."
        )

    elif kind in ("renewal_due", "subscription_expiry"):
        days = merchant.get("subscription", {}).get("days_remaining", 0)
        return (
            f"Write a merchant-facing message about subscription renewal ({days} days remaining). "
            f"Don't be pushy. Frame the value they'd lose. "
            f"Single binary CTA: 'Reply YES to renew, or let me know if you have questions.'"
        )

    elif kind in ("unverified_gbp", "profile_incomplete"):
        return (
            f"Write a merchant-facing Vera message about their incomplete/unverified Google Business Profile. "
            f"Mention the specific percentage or missing fields from the trigger/signals. "
            f"Offer to fix it: 'I can update X in 5 min — want me to go ahead?'"
        )

    else:  # generic fallback
        return (
            f"Write a merchant-facing Vera message appropriate for trigger kind='{kind}'. "
            f"Ground it in the specific data from merchant context (performance, offers, signals). "
            f"Use one compulsion lever: curiosity, loss aversion, social proof, or effort externalization. "
            f"Single CTA in the last sentence."
        )


# ── Multi-turn conversation handler ───────────────────────────────────────────

class ConversationState:
    def __init__(self, conversation_id: str, merchant: dict, category: dict,
                 trigger: dict, customer: dict | None = None):
        self.conversation_id = conversation_id
        self.merchant  = merchant
        self.category  = category
        self.trigger   = trigger
        self.customer  = customer
        self.turns: list[dict] = []  # [{"from": "vera"|"merchant", "body": str}]
        self.auto_reply_count = 0
        self.ended = False

    def add_turn(self, from_role: str, body: str):
        self.turns.append({"from": from_role, "body": body})

    def last_vera_body(self) -> str:
        for t in reversed(self.turns):
            if t["from"] == "vera":
                return t["body"]
        return ""

    def merchant_turns(self) -> list[str]:
        return [t["body"] for t in self.turns if t["from"] in ("merchant", "customer")]


def respond(state: ConversationState, merchant_message: str) -> dict:
    """
    Handle a reply from the merchant/customer in an ongoing conversation.
    Returns: { action: "send"|"wait"|"end", body?, cta?, rationale }
    """
    # Auto-reply detection
    if is_auto_reply(merchant_message):
        state.auto_reply_count += 1
        if state.auto_reply_count == 1:
            # First auto-reply: one soft retry
            return {
                "action": "send",
                "body": (
                    f"Samajh gayi — message team tak pahunch gaya. "
                    f"Kya aap directly dekhna chahenge ki exact kya improve ho sakta hai? "
                    f"2 minute ka kaam hai. Chalega?"
                ),
                "cta": "binary_yes_stop",
                "rationale": "First auto-reply detected; one soft retry to reach the human.",
            }
        else:
            # Second auto-reply: graceful exit
            state.ended = True
            mname = _name(state.merchant)
            return {
                "action": "end",
                "rationale": f"Repeated auto-reply confirmed ({state.auto_reply_count}×). Gracefully exiting — no point burning turns.",
            }

    state.add_turn("merchant", merchant_message)
    intent = detect_intent(merchant_message)

    # Explicit negative: graceful exit
    if intent == "negative":
        state.ended = True
        mname = _name(state.merchant)
        return {
            "action": "end",
            "rationale": "Merchant signalled not-interested. Exiting gracefully to preserve relationship.",
        }

    # Explicit positive intent: action mode — deliver the artifact, don't qualify further
    if intent == "positive":
        return _action_mode_reply(state, merchant_message)

    # Neutral / question: use Claude to reply naturally
    return _neutral_reply(state, merchant_message)


def _action_mode_reply(state: ConversationState, message: str) -> dict:
    """Merchant said YES — deliver the artifact or confirm the action."""
    kind     = state.trigger.get("kind", "generic")
    merchant = state.merchant
    category = state.category
    mname    = _name(merchant)
    lang     = _lang(merchant, state.customer)

    system = _build_system(category, lang)

    # Build conversation history for context
    history_str = "\n".join(
        f"[{t['from'].upper()}]: {t['body']}" for t in state.turns[-6:]
    )

    user = f"""The merchant ({mname}) has replied positively to your message. They want to proceed.

CONVERSATION SO FAR:
{history_str}
[MERCHANT]: {message}

MERCHANT CONTEXT: {json.dumps({'identity': merchant.get('identity'), 'offers': merchant.get('offers', [])[:3], 'signals': merchant.get('signals', [])}, ensure_ascii=False)}
TRIGGER KIND: {kind}

TASK: The merchant said YES. Switch from pitch mode to action mode immediately.
- Do NOT ask another qualifying question.
- Either: (a) deliver the promised artifact/action directly in this message, OR
  (b) if the artifact is complex, confirm what you're doing and give a clear next step.
- Keep it concrete and under 120 words.

Output JSON: {{"action": "send", "body": "...", "cta": "open_ended"|"none", "rationale": "..."}}"""

    try:
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=400, temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        return {"action": "wait", "wait_seconds": 60, "rationale": f"Rate limit pause (Action mode): {e}"}
    try:
        clean = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(clean)
        result.setdefault("action", "send")
        return result
    except Exception:
        return {"action": "send", "body": raw, "cta": "open_ended", "rationale": "Action mode response."}


def _neutral_reply(state: ConversationState, message: str) -> dict:
    """Merchant asked a question or gave a neutral response — answer naturally."""
    merchant = state.merchant
    category = state.category
    mname    = _name(merchant)
    lang     = _lang(merchant, state.customer)

    system = _build_system(category, lang)

    history_str = "\n".join(
        f"[{t['from'].upper()}]: {t['body']}" for t in state.turns[-6:]
    )

    merchant_ctx = {
        "identity": merchant.get("identity"),
        "performance": merchant.get("performance"),
        "offers": merchant.get("offers", [])[:5],
        "signals": merchant.get("signals", []),
        "customer_aggregate": merchant.get("customer_aggregate"),
    }

    user = f"""You are mid-conversation with merchant {mname}.

CONVERSATION:
{history_str}
[MERCHANT]: {message}

MERCHANT DATA: {json.dumps(merchant_ctx, ensure_ascii=False)}
CATEGORY: {category.get('slug')}

TASK: Reply to the merchant's message.
- Answer their question/comment grounded in the context data.
- Don't restart the pitch. Continue naturally.
- Keep it under 80 words.
- Know when to end: if they're disengaging, respond with action=end.

Output JSON: {{"action": "send"|"wait"|"end", "body": "...", "cta": "open_ended"|"none"|"binary_yes_stop", "rationale": "..."}}"""

    try:
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=350, temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        return {"action": "wait", "wait_seconds": 60, "rationale": f"Rate limit pause (Neutral reply): {e}"}
    try:
        clean = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(clean)
        result.setdefault("action", "send")
        return result
    except Exception:
        return {"action": "send", "body": raw, "cta": "open_ended", "rationale": "Neutral reply."}
