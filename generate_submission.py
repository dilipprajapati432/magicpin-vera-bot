#!/usr/bin/env python3
"""
generate_submission.py — runs bot.compose() on all 30 canonical test pairs
and writes submission.jsonl (one JSON object per line).

Usage:
    cd /path/to/submission
    ANTHROPIC_API_KEY=sk-... python generate_submission.py \
        --expanded /path/to/expanded \
        --out submission.jsonl
"""

import os, sys, json, argparse, pathlib, time

# Add submission dir to path so we can import bot
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bot as Bot

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def find_file(directory: str, id_str: str) -> str | None:
    """Find a file in directory whose stem matches id_str."""
    d = pathlib.Path(directory)
    # Exact match first
    for ext in (".json",):
        p = d / f"{id_str}{ext}"
        if p.exists():
            return str(p)
    # Prefix match
    matches = list(d.glob(f"{id_str}*.json"))
    if matches:
        return str(matches[0])
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded", default="expanded", help="Path to expanded dataset dir")
    parser.add_argument("--out", default="submission.jsonl", help="Output file")
    parser.add_argument("--pairs", default=None, help="Path to test_pairs.json (default: expanded/test_pairs.json)")
    args = parser.parse_args()

    expanded = pathlib.Path(args.expanded)
    pairs_path = args.pairs or str(expanded / "test_pairs.json")

    pairs_data = load_json(pairs_path)
    pairs = pairs_data.get("pairs", pairs_data)

    print(f"Loaded {len(pairs)} test pairs from {pairs_path}")
    print(f"Expanded dataset: {expanded}")
    print(f"Output: {args.out}\n")

    results = []

    for i, pair in enumerate(pairs):
        test_id     = pair["test_id"]
        merchant_id = pair["merchant_id"]
        trigger_id  = pair["trigger_id"]
        customer_id = pair.get("customer_id")

        print(f"[{i+1:02d}/{len(pairs)}] {test_id}: merchant={merchant_id}, trigger={trigger_id}, customer={customer_id}")

        # Load merchant
        merchant_path = find_file(str(expanded / "merchants"), merchant_id)
        if not merchant_path:
            print(f"  WARNING: merchant file not found for {merchant_id}")
            continue
        merchant = load_json(merchant_path)

        # Load category from merchant's category_slug
        category_slug = merchant.get("category_slug", "")
        # normalize: "pharmacies" might be stored as "pharmacie" in filenames
        cat_path = find_file(str(expanded / "categories"), category_slug)
        if not cat_path:
            # Try without trailing 's'
            alt = category_slug.rstrip("s")
            cat_path = find_file(str(expanded / "categories"), alt)
        if not cat_path:
            print(f"  WARNING: category file not found for {category_slug}")
            continue
        category = load_json(cat_path)

        # Load trigger
        trigger_path = find_file(str(expanded / "triggers"), trigger_id)
        if not trigger_path:
            # Try in expanded root
            trigger_path = find_file(str(expanded), trigger_id)
        if not trigger_path:
            print(f"  WARNING: trigger file not found for {trigger_id}")
            continue
        trigger = load_json(trigger_path)

        # Load customer (optional)
        customer = None
        if customer_id:
            cust_path = find_file(str(expanded / "customers"), customer_id)
            if cust_path:
                customer = load_json(cust_path)
            else:
                print(f"  WARNING: customer {customer_id} not found, proceeding without")

        # Run compose
        try:
            result = Bot.compose(category, merchant, trigger, customer)
            result["test_id"] = test_id
            # Reorder keys for readability
            ordered = {
                "test_id": test_id,
                "body":    result.get("body", ""),
                "cta":     result.get("cta", "open_ended"),
                "send_as": result.get("send_as", "vera"),
                "suppression_key": result.get("suppression_key", ""),
                "rationale": result.get("rationale", ""),
            }
            results.append(ordered)
            print(f"  ✓ {len(ordered['body'])} chars | cta={ordered['cta']} | send_as={ordered['send_as']}")
            print(f"    BODY: {ordered['body'][:120]}...")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            # Write placeholder so test_id isn't missing
            results.append({
                "test_id": test_id,
                "body": f"[COMPOSE ERROR: {e}]",
                "cta": "open_ended",
                "send_as": "vera",
                "suppression_key": "",
                "rationale": "Error during composition",
            })

        # Rate limit courtesy (increased for Groq TPM limits)
        time.sleep(6.0)

    # Write JSONL
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✓ Wrote {len(results)} entries to {args.out}")

if __name__ == "__main__":
    main()
