#!/usr/bin/env python3
"""
Evaluation script for receipt extraction results.

Scoring:
1. Schema check: predicted keys must match ground truth keys exactly, otherwise score=0.
2. Non-item fields: 1 error per mismatched field.
3. Items:
   - Item count mismatch: 1 penalty per extra/missing item (each missing item also adds
     4 field errors since all its fields are wrong).
   - Matched items (fuzzy matched by name): 1 error per wrong field (quantity, unit_price,
     total_price). Name itself is not penalized if fuzzy-matched.
   - Unmatched predicted items: 4 errors each (all fields considered wrong).
   - Unmatched ground truth items: 4 errors each.

Final score per receipt = 1 - (errors / total_fields)
"""

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

NON_ITEM_KEYS = ["date", "total", "currency", "store", "vat_rate", "payment_method"]
ITEM_FIELDS = ["name", "quantity", "unit_price", "total_price"]


def fuzzy_ratio(a: str, b: str) -> float:
    if a is None or b is None:
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def values_equal(a, b) -> bool:
    """Compare two values, treating int/float equivalence (1 == 1.0)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def match_items(pred_items: list, gt_items: list, threshold: float = 0.6) -> list:
    """
    Match predicted items to ground truth items using fuzzy name matching.
    Returns list of (pred_idx, gt_idx) pairs. Greedy best-first matching.
    """
    if not pred_items or not gt_items:
        return []

    # Build similarity matrix
    scores = []
    for pi, p in enumerate(pred_items):
        for gi, g in enumerate(gt_items):
            ratio = fuzzy_ratio(p.get("name"), g.get("name"))
            if ratio >= threshold:
                scores.append((ratio, pi, gi))

    # Greedy matching: best scores first
    scores.sort(reverse=True)
    used_pred = set()
    used_gt = set()
    matches = []
    for ratio, pi, gi in scores:
        if pi not in used_pred and gi not in used_gt:
            matches.append((pi, gi))
            used_pred.add(pi)
            used_gt.add(gi)

    return matches


def score_receipt(pred: dict, gt: dict) -> dict:
    """Score a single receipt extraction against ground truth."""
    result = {
        "schema_ok": True,
        "errors": [],
        "non_item_errors": 0,
        "item_errors": 0,
        "total_fields": 0,
        "total_errors": 0,
        "score": 0.0,
    }

    # 1. Schema check: top-level keys must match
    pred_keys = set(pred.keys())
    gt_keys = set(gt.keys())
    if pred_keys != gt_keys:
        result["schema_ok"] = False
        result["errors"].append(
            f"Schema mismatch: predicted={sorted(pred_keys)}, expected={sorted(gt_keys)}"
        )
        result["score"] = 0.0
        return result

    # Also check item schema if items exist
    if gt.get("items") and pred.get("items"):
        gt_item_keys = set(gt["items"][0].keys()) if gt["items"] else set()
        pred_item_keys = set(pred["items"][0].keys()) if pred["items"] else set()
        if gt_item_keys and pred_item_keys and gt_item_keys != pred_item_keys:
            result["schema_ok"] = False
            result["errors"].append(
                f"Item schema mismatch: predicted={sorted(pred_item_keys)}, expected={sorted(gt_item_keys)}"
            )
            result["score"] = 0.0
            return result

    # 2. Non-item fields
    non_item_errors = 0
    for key in NON_ITEM_KEYS:
        if key not in gt:
            continue
        result["total_fields"] += 1
        if not values_equal(pred.get(key), gt.get(key)):
            non_item_errors += 1
            result["errors"].append(
                f"Field '{key}': predicted={pred.get(key)!r}, expected={gt.get(key)!r}"
            )
    result["non_item_errors"] = non_item_errors

    # 3. Items
    pred_items = pred.get("items", []) or []
    gt_items = gt.get("items", []) or []

    # Total possible item fields = gt items * fields per item
    result["total_fields"] += len(gt_items) * len(ITEM_FIELDS)

    matches = match_items(pred_items, gt_items)
    matched_pred = {m[0] for m in matches}
    matched_gt = {m[1] for m in matches}

    item_errors = 0

    # Score matched items
    for pi, gi in matches:
        p_item = pred_items[pi]
        g_item = gt_items[gi]
        # Check name too (even though matched by fuzzy, exact match counts)
        for field in ITEM_FIELDS:
            if not values_equal(p_item.get(field), g_item.get(field)):
                item_errors += 1
                result["errors"].append(
                    f"Item '{g_item.get('name')}' field '{field}': "
                    f"predicted={p_item.get(field)!r}, expected={g_item.get(field)!r}"
                )

    # Unmatched ground truth items: all fields wrong
    for gi in range(len(gt_items)):
        if gi not in matched_gt:
            item_errors += len(ITEM_FIELDS)
            result["errors"].append(
                f"Missing item: '{gt_items[gi].get('name')}' (not found in prediction)"
            )

    # Unmatched predicted items: all fields wrong, but we also need to account for them
    # in total_fields since they represent extra work
    unmatched_pred_count = len(pred_items) - len(matched_pred)
    if unmatched_pred_count > 0:
        # Add to total fields so extra items penalize the score
        result["total_fields"] += unmatched_pred_count * len(ITEM_FIELDS)
        item_errors += unmatched_pred_count * len(ITEM_FIELDS)
        for pi in range(len(pred_items)):
            if pi not in matched_pred:
                result["errors"].append(
                    f"Extra item: '{pred_items[pi].get('name')}' (not in ground truth)"
                )

    result["item_errors"] = item_errors
    result["total_errors"] = non_item_errors + item_errors
    result["total_fields"] = max(result["total_fields"], 1)  # avoid div by 0
    result["score"] = 1.0 - (result["total_errors"] / result["total_fields"])

    return result


def evaluate_file(filepath: str) -> dict:
    with open(filepath) as f:
        data = json.load(f)

    results = []
    for i, entry in enumerate(data["results"]):
        try:
            pred = json.loads(entry["model_output"])
        except json.JSONDecodeError:
            results.append({
                "index": i,
                "schema_ok": False,
                "errors": ["Failed to parse model_output as JSON"],
                "score": 0.0,
            })
            continue

        gt = json.loads(entry["ground_truth"])
        score_result = score_receipt(pred, gt)
        score_result["index"] = i
        results.append(score_result)

    # Summary
    scores = [r["score"] for r in results]
    schema_failures = sum(1 for r in results if not r["schema_ok"])
    avg_score = sum(scores) / len(scores) if scores else 0.0

    summary = {
        "file": filepath,
        "model": data.get("config", {}).get("model_name", "unknown"),
        "num_receipts": len(results),
        "schema_failures": schema_failures,
        "average_score": round(avg_score, 4),
        "min_score": round(min(scores), 4) if scores else 0.0,
        "max_score": round(max(scores), 4) if scores else 0.0,
        "per_receipt": results,
    }

    return summary


def print_report(summary: dict):
    print(f"{'=' * 70}")
    print(f"Model: {summary['model']}")
    print(f"File:  {summary['file']}")
    print(f"{'=' * 70}")
    print(f"Receipts: {summary['num_receipts']}")
    print(f"Schema failures: {summary['schema_failures']}")
    print(f"Average score: {summary['average_score']:.2%}")
    print(f"Min score:     {summary['min_score']:.2%}")
    print(f"Max score:     {summary['max_score']:.2%}")
    print(f"{'=' * 70}")

    for r in summary["per_receipt"]:
        idx = r["index"]
        score = r["score"]
        marker = "SCHEMA_ERROR" if not r["schema_ok"] else ""
        print(f"\nReceipt #{idx + 1}: {score:.2%} {marker}")
        if r["errors"]:
            for err in r["errors"]:
                print(f"  - {err}")

    print(f"\n{'=' * 70}")
    print(f"FINAL SCORE: {summary['average_score']:.2%}")
    print(f"{'=' * 70}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <result_file.json> [result_file2.json ...]")
        sys.exit(1)

    all_summaries = []
    for filepath in sys.argv[1:]:
        if not Path(filepath).exists():
            print(f"File not found: {filepath}", file=sys.stderr)
            continue
        summary = evaluate_file(filepath)
        print_report(summary)
        all_summaries.append(summary)

    if len(all_summaries) > 1:
        print(f"\n{'=' * 70}")
        print("COMPARISON")
        print(f"{'=' * 70}")
        for s in all_summaries:
            print(f"  {s['model']:40s} → {s['average_score']:.2%}")


if __name__ == "__main__":
    main()
