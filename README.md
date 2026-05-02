# Vera Engine — magicpin AI Challenge Submission

## Approach

**Single Claude call per composition** with all 4 context layers (category, merchant, trigger, customer) structured into a single prompt. The key design decisions:

### 1. Per-trigger-kind dispatch
Rather than one generic prompt, `bot.py` routes each trigger kind to a tailored `_kind_instruction()`. A `research_digest` prompt tells Claude to lead with trial sample size and source citation; a `perf_dip` prompt insists on one actionable recommendation; a `curious_ask` prompt enforces the ask-the-merchant pattern. This prevents the single biggest failure mode: a well-structured context producing a generic message because the model didn't know what shape to aim for.

### 2. Context fully serialised, not summarised
Every fact the judge might check — CTR vs peer median, customer lapse days, offer prices, signal flags — is present verbatim in the prompt. Claude can't hallucinate "your CTR is 2.1%" if it's literally in the prompt as `CTR=0.021 (peer median=0.030)`.

### 3. Post-LLM output validation
- CTA is validated against the allowed set (`binary_yes_stop`, `open_ended`, `none`)
- Anti-repetition check using token-overlap similarity against prior Vera messages; if too similar, a re-prompt is fired
- JSON fallback: if Claude returns non-JSON, the raw text is used as the body

### 4. Multi-turn conversation handling (`respond()`)
- **Auto-reply detection**: regex-based pattern matching for common WA Business canned replies (Hindi + English). First auto-reply → one soft retry ("Samajh gayi, kya aap directly dekhna chahenge…"). Second auto-reply → graceful `action: end`.
- **Intent transition**: positive-intent messages (yes/sure/karo/chalega) switch immediately to `_action_mode_reply()` which delivers the artifact without asking qualifying questions.
- **Negative exit**: graceful `action: end` on clear negative signals.
- **Neutral/question**: Claude replies naturally with full merchant context.

### 5. HTTP server (`server.py`)
Implements all 5 judge endpoints with exact schema compliance:
- `/v1/context` — idempotent by `(scope, context_id, version)`; 409 on stale version
- `/v1/tick` — suppression-key dedup; 20-action cap; per-merchant conversation state
- `/v1/reply` — routes to `bot.respond()`
- `/v1/healthz` — per-scope context counts
- `/v1/metadata` — team info

## Model choice

`llama-3.3-70b-versatile` via Groq — best balance of output quality, latency, and cost at zero API cost. Temperature=0 for full determinism. Groq's low-latency inference easily fits within the 30s timeout budget.

## What additional context would have helped most

1. **Real merchant conversation histories** — the seed data has 1-2 turns; production Vera handles 4.7 average turns. More history would let the bot calibrate re-engagement cadence.
2. **Slot availability API** — recall/appointment messages improve sharply when real open slots are injected; the dataset has sample slots but not a live query interface.
3. **Peer benchmark distributions** (not just means) — "your CTR 2.1% vs median 3.0%" is weaker than "your CTR puts you in the 31st percentile of Delhi dental clinics"; the latter drives more actionable framing.

## Tradeoffs

- **One LLM call per compose vs retrieval pipeline**: simpler, predictable latency, easier to debug. Tradeoff: can't retrieve the most relevant digest item from a large knowledge base — but the dataset's digest items are already pre-filtered so this isn't a problem at this scale.
- **Regex auto-reply detection vs LLM classification**: regex is deterministic and zero-latency; an LLM classifier would handle edge cases better but adds ~2-3s per reply turn.
- **In-memory state**: fine for a 60-min test window; would need Redis/SQLite for production.

## Running locally

```bash
pip install -r requirements.txt

# Generate submission.jsonl (key already set in bot.py, or override via env)
export GROQ_API_KEY=gsk_...   # optional — key is already embedded in bot.py
python generate_submission.py --expanded /path/to/expanded --out submission.jsonl

# Run the HTTP server
uvicorn server:app --host 0.0.0.0 --port 8080

# Run local self-test
export BOT_URL=http://localhost:8080
python /path/to/judge_simulator.py
```

## Deploying (Render — free tier)

1. Push this folder to GitHub
2. render.com → New Web Service → connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
5. Add env var: `GROQ_API_KEY=gsk_...`
6. Deploy → submit the public URL
