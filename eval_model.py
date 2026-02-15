import asyncio
import html as html_module
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import AsyncOpenAI

from score import score_receipt

# ====================
# Configuration
# ====================
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
LORA_ADAPTER = None # "aidan3e4/receipt-lora-qwen3-4b"  # set to None to eval base model only
MODEL_BASE_URL = "http://localhost:8000/v1"  # vLLM default
MODEL_API_KEY = "token-abc123"  # vLLM doesn't need a real key
DATASET_FILE = "receipt_dataset_1000_test.jsonl"
EVAL_DIR = "eval"
MAX_CONCURRENT = 10
NUM_ENTRIES = None  # None = all entries

# =====================================================================
# vLLM launch commands:
#
# Base model only:
#   vllm serve Qwen/Qwen3-4B-Instruct-2507
#
# With LoRA adapter:
#   vllm serve Qwen/Qwen3-4B-Instruct-2507 \
#     --enable-lora \
#     --lora-modules receipt-lora=aidan3e4/receipt-lora-qwen3-4b
#
# When using LoRA, set MODEL_NAME below to the lora module name:
#   MODEL_NAME = "receipt-lora"
# =====================================================================

# ====================
# Client
# ====================
load_dotenv()
client = AsyncOpenAI(base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY)

# ====================
# System prompt (same as used in gen_data_receipts.py extraction)
# ====================
with open("prompt_v0.txt") as f:
    SYSTEM_PROMPT = f.read().strip()
USER_PROMPT_TEMPLATE = "{instruction}\n\nTicket de caisse :\n\n{receipt_text}"


# ====================
# Inference
# ====================


async def call_model(semaphore: asyncio.Semaphore, instruction: str, receipt_text: str) -> str:
    """Send a receipt to the local model and return the raw response text."""
    async with semaphore:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(instruction=instruction, receipt_text=receipt_text)},
            ],
        )
    return response.choices[0].message.content.strip()


# ====================
# Dataset loading
# ====================


def load_dataset(path: str, limit: int | None = None) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if limit:
        entries = entries[:limit]
    return entries


# ====================
# HTML report
# ====================


def generate_html(config: dict, results: list[dict], scoring: dict, output_path: str):
    avg_score = scoring["average_score"]
    score_color = "#4caf50" if avg_score >= 0.9 else "#ff9800" if avg_score >= 0.7 else "#f44336"

    header = f"""
    <html>
    <head>
        <title>Évaluation — {html_module.escape(config["model_name"])}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .config {{ background-color: #e3f2fd; border: 1px solid #90caf9; padding: 15px; border-radius: 5px; margin-bottom: 25px; }}
            .config h2 {{ margin-top: 0; }}
            .score-banner {{ background-color: {score_color}; color: white; padding: 20px; border-radius: 5px; margin-bottom: 25px; text-align: center; }}
            .score-banner h2 {{ margin: 0; font-size: 2em; }}
            .score-banner p {{ margin: 5px 0 0 0; }}
            .entry {{ border: 1px solid #ccc; margin-bottom: 20px; padding: 10px; border-radius: 5px; }}
            .header {{ font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }}
            .score-badge {{ display: inline-block; padding: 2px 8px; border-radius: 3px; color: white; font-weight: bold; margin-left: 10px; }}
            .score-good {{ background-color: #4caf50; }}
            .score-mid {{ background-color: #ff9800; }}
            .score-bad {{ background-color: #f44336; }}
            .errors {{ background-color: #fff3e0; padding: 8px; margin-top: 8px; border-radius: 3px; font-size: 0.85em; }}
            .errors li {{ color: #e65100; }}
            .container {{ display: flex; flex-wrap: wrap; }}
            .col {{ flex: 1; min-width: 30%; padding: 10px; }}
            .col-input {{ background-color: #f9f9f9; border-right: 1px solid #ddd; }}
            .col-model {{ background-color: #fff8e1; border-right: 1px solid #ddd; }}
            .col-truth {{ background-color: #e8f5e9; }}
            pre {{ white-space: pre-wrap; word-wrap: break-word; font-size: 0.85em; }}
        </style>
    </head>
    <body>
        <h1>Évaluation du modèle — Extraction de tickets de caisse</h1>
        <div class="score-banner">
            <h2>{avg_score:.1%}</h2>
            <p>Score moyen sur {scoring["num_receipts"]} tickets (min: {scoring["min_score"]:.1%}, max: {scoring["max_score"]:.1%})</p>
            <p>Erreurs de schéma : {scoring["schema_failures"]}</p>
        </div>
        <div class="config">
            <h2>Configuration</h2>
            <p><strong>Modèle :</strong> {html_module.escape(config["model_name"])}</p>
            <p><strong>LoRA adapter :</strong> {html_module.escape(config.get("lora_adapter") or "None (base model)")}</p>
            <p><strong>Date :</strong> {html_module.escape(config["datetime"])}</p>
            <p><strong>Entrées évaluées :</strong> {config["num_entries"]}</p>
            <p><strong>System prompt :</strong></p>
            <pre>{html_module.escape(config["system_prompt"])}</pre>
            <p><strong>User prompt template :</strong></p>
            <pre>{html_module.escape(config["user_prompt_template"])}</pre>
        </div>
    """

    body = ""
    per_receipt = scoring["per_receipt"]
    for idx, r in enumerate(results):
        raw = html_module.escape(r["input"])
        model_out = html_module.escape(r["model_output"])
        truth = html_module.escape(r["ground_truth"])
        sr = per_receipt[idx]
        score_val = sr["score"]
        css_class = "score-good" if score_val >= 0.9 else "score-mid" if score_val >= 0.7 else "score-bad"
        schema_tag = ' <span class="score-badge score-bad">SCHEMA ERROR</span>' if not sr["schema_ok"] else ""

        errors_html = ""
        if sr["errors"]:
            items = "".join(f"<li>{html_module.escape(e)}</li>" for e in sr["errors"])
            errors_html = f'<div class="errors"><ul>{items}</ul></div>'

        body += f"""
        <div class="entry">
            <div class="header">Entrée {idx + 1} <span class="score-badge {css_class}">{score_val:.0%}</span>{schema_tag}</div>
            <div class="container">
                <div class="col col-input">
                    <h3>Ticket (texte brut)</h3>
                    <pre>{raw}</pre>
                </div>
                <div class="col col-model">
                    <h3>Sortie du modèle</h3>
                    <pre>{model_out}</pre>
                </div>
                <div class="col col-truth">
                    <h3>Vérité terrain</h3>
                    <pre>{truth}</pre>
                </div>
            </div>
            {errors_html}
        </div>
        """

    footer = """
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + body + footer)

    print(f"Rapport HTML sauvegardé dans {output_path}")


# ====================
# Main
# ====================


async def main():
    entries = load_dataset(DATASET_FILE, NUM_ENTRIES)
    print(f"Évaluation de {len(entries)} entrées avec le modèle {MODEL_NAME}...")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def process_one(entry: dict) -> dict:
        receipt_text = entry["input"]
        instruction = entry["instruction"]
        ground_truth = json.dumps(
            json.loads(entry["output"]), indent=4, ensure_ascii=False
        )
        model_raw = await call_model(semaphore, instruction, receipt_text)
        # Try to pretty-print model output if it's valid JSON
        try:
            model_json = json.loads(model_raw)
            model_output = json.dumps(model_json, indent=4, ensure_ascii=False)
        except json.JSONDecodeError:
            model_output = model_raw
        return {
            "input": receipt_text,
            "model_output": model_output,
            "ground_truth": ground_truth,
        }

    tasks = [process_one(e) for e in entries]
    results = await asyncio.gather(*tasks)

    # Build config and save everything to eval/
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    config = {
        "model_name": MODEL_NAME,
        "lora_adapter": LORA_ADAPTER,
        "datetime": now.isoformat(),
        "num_entries": len(results),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
    }

    # Score each receipt
    print("Calcul des scores...")
    per_receipt = []
    for i, r in enumerate(results):
        try:
            pred = json.loads(r["model_output"])
        except json.JSONDecodeError:
            per_receipt.append({
                "index": i,
                "schema_ok": False,
                "errors": ["Failed to parse model_output as JSON"],
                "score": 0.0,
            })
            continue
        gt = json.loads(r["ground_truth"])
        sr = score_receipt(pred, gt)
        sr["index"] = i
        per_receipt.append(sr)

    scores = [r["score"] for r in per_receipt]
    scoring = {
        "average_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "min_score": round(min(scores), 4) if scores else 0.0,
        "max_score": round(max(scores), 4) if scores else 0.0,
        "schema_failures": sum(1 for r in per_receipt if not r["schema_ok"]),
        "num_receipts": len(per_receipt),
        "per_receipt": per_receipt,
    }
    print(f"Score moyen : {scoring['average_score']:.2%}")

    os.makedirs(EVAL_DIR, exist_ok=True)
    safe_model_name = MODEL_NAME.replace("/", "_")
    lora_suffix = f"_lora_{LORA_ADAPTER.split('/')[-1]}" if LORA_ADAPTER else "_base"
    base_name = f"{safe_model_name}{lora_suffix}_{timestamp}"
    json_path = os.path.join(EVAL_DIR, f"{base_name}.json")
    html_path = os.path.join(EVAL_DIR, f"{base_name}.html")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"config": config, "scoring": scoring, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Données sauvegardées dans {json_path}")

    generate_html(config, results, scoring, html_path)
    print("Terminé !")


if __name__ == "__main__":
    asyncio.run(main())
