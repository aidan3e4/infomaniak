import asyncio
import html as html_module
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import AsyncOpenAI

# ====================
# Configuration
# ====================
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_BASE_URL = "http://localhost:8000/v1"  # vLLM default
MODEL_API_KEY = "token-abc123"  # vLLM doesn't need a real key
DATASET_FILE = "receipt_dataset.jsonl"
EVAL_DIR = "eval"
MAX_CONCURRENT = 5
NUM_ENTRIES = None  # None = all entries

# ====================
# Client
# ====================
load_dotenv()
client = AsyncOpenAI(base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY)

# ====================
# System prompt (same as used in gen_data_receipts.py extraction)
# ====================
SYSTEM_PROMPT = (
    "Tu es un extracteur parfait de tickets de caisse suisses. "
    "Extrais les informations en respectant strictement le schéma. "
    "Utilise null pour les valeurs manquantes ou illisibles. "
    "Les prix sont des nombres décimaux (utilise . et non ,). "
    "Les dates sont toujours au format JJ.MM.AAAA si présentes. "
    "N'invente rien — utilise uniquement ce qui est dans le texte. "
    "Renvoie uniquement l'objet JSON — pas de texte supplémentaire."
)
USER_PROMPT_TEMPLATE = "Ticket de caisse :\n\n{receipt_text}"


# ====================
# Inference
# ====================


async def call_model(semaphore: asyncio.Semaphore, receipt_text: str) -> str:
    """Send a receipt to the local model and return the raw response text."""
    async with semaphore:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(receipt_text=receipt_text)},
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


def generate_html(config: dict, results: list[dict], output_path: str):
    header = f"""
    <html>
    <head>
        <title>Évaluation — {html_module.escape(config["model_name"])}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .config {{ background-color: #e3f2fd; border: 1px solid #90caf9; padding: 15px; border-radius: 5px; margin-bottom: 25px; }}
            .config h2 {{ margin-top: 0; }}
            .entry {{ border: 1px solid #ccc; margin-bottom: 20px; padding: 10px; border-radius: 5px; }}
            .header {{ font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }}
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
        <div class="config">
            <h2>Configuration</h2>
            <p><strong>Modèle :</strong> {html_module.escape(config["model_name"])}</p>
            <p><strong>Date :</strong> {html_module.escape(config["datetime"])}</p>
            <p><strong>Entrées évaluées :</strong> {config["num_entries"]}</p>
            <p><strong>System prompt :</strong></p>
            <pre>{html_module.escape(config["system_prompt"])}</pre>
            <p><strong>User prompt template :</strong></p>
            <pre>{html_module.escape(config["user_prompt_template"])}</pre>
        </div>
    """

    body = ""
    for idx, r in enumerate(results, 1):
        raw = html_module.escape(r["input"])
        model_out = html_module.escape(r["model_output"])
        truth = html_module.escape(r["ground_truth"])
        body += f"""
        <div class="entry">
            <div class="header">Entrée {idx}</div>
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
        ground_truth = json.dumps(
            json.loads(entry["output"]), indent=4, ensure_ascii=False
        )
        model_raw = await call_model(semaphore, receipt_text)
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
        "datetime": now.isoformat(),
        "num_entries": len(results),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
    }

    os.makedirs(EVAL_DIR, exist_ok=True)
    safe_model_name = MODEL_NAME.replace("/", "_")
    base_name = f"{safe_model_name}_{timestamp}"
    json_path = os.path.join(EVAL_DIR, f"{base_name}.json")
    html_path = os.path.join(EVAL_DIR, f"{base_name}.html")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"config": config, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Données sauvegardées dans {json_path}")

    generate_html(config, results, html_path)
    print("Terminé !")


if __name__ == "__main__":
    asyncio.run(main())
