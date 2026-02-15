import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# ====================
# Configuration
# ====================
GENERATION_MODEL = "gpt-4o"
EXTRACTION_MODEL = "gpt-5"
NUM_RECEIPTS = 1000
BATCH_SIZE = 20
MAX_CONCURRENT = 10
OUTPUT_FILE = "receipt_dataset_1000.jsonl"

# ====================
# OpenAI client
# ====================
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ====================
# Pydantic schemas
# ====================

class ReceiptItem(BaseModel):
    name: str = Field(..., description="Nom du produit")
    quantity: float | None = Field(None, description="Quantité")
    unit_price: float | None = Field(None, description="Prix unitaire en CHF")
    total_price: float | None = Field(None, description="Prix total de la ligne en CHF")


class ReceiptExtraction(BaseModel):
    date: str | None = Field(None, description="Date au format JJ.MM.AAAA")
    total: float | None = Field(None, description="Total en CHF")
    currency: str = Field("CHF", description="Devise")
    store: str | None = Field(None, description="Nom du magasin")
    items: list[ReceiptItem] = Field(..., description="Liste des articles")
    vat_rate: float | None = Field(None, description="Taux de TVA principal (ex: 8.1 ou 2.6)")
    payment_method: str | None = Field(None, description="Mode de paiement (TWINT, espèces, Visa, etc.)")


INSTRUCTION = (
    "Extrais les données structurées de ce ticket de caisse sous forme de JSON.\n"
    "Respecte strictement le schéma suivant :\n"
    f"{json.dumps(ReceiptExtraction.model_json_schema(), indent=2, ensure_ascii=False)}\n"
    "Utilise null pour les valeurs manquantes. Les prix sont des nombres décimaux."
)

# ====================
# Generation
# ====================

GENERATION_PROMPT = """
Génère {batch_size} tickets de caisse suisses réalistes en texte brut, exactement comme ils apparaissent sur un vrai ticket (copié depuis un email, un PDF, une photo ou un scan OCR).

Concentre-toi sur les magasins de Suisse romande : Migros, Coop, Aldi, Lidl, Denner, Manor, Restaurant Migros, Aligro, etc.

Inclus des variations :
- Dates au format JJ.MM.AAAA (récentes ou aléatoires sur les 2 dernières années)
- Produits suisses courants (lait, pain, fromage, chocolat, bière, légumes, viande, café, etc.)
- Quantités, prix unitaires, totaux par ligne, rabais (-5%, -2.00 Fr.), TVA (8.1% ou 2.6%)
- Total général
- Parfois le mode de paiement (TWINT, carte, espèces), numéro de ticket, caissier/ère
- Rends-les réalistes : sauts de ligne, espaces supplémentaires, abréviations, fautes de frappe, erreurs OCR (espaces bizarres, caractères manquants)

Varie la longueur : du petit ticket de café au long ticket de supermarché.

Format de sortie : numérote-les de 1 à {batch_size}, sépare chaque ticket par ---
Ne mets PAS de JSON, d'explications ni de markdown. Juste le texte brut des tickets.

Tout doit être en français.
""".strip()


async def generate_receipt_text_batch(
    semaphore: asyncio.Semaphore, batch_size: int
) -> list[str]:
    prompt = GENERATION_PROMPT.format(batch_size=batch_size)
    async with semaphore:
        response = await client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
    content = response.choices[0].message.content.strip()
    receipts = [r.strip() for r in content.split("---") if r.strip()]
    # Strip leading numbering like "1." or "1)"
    cleaned = []
    for r in receipts:
        for i in range(1, batch_size + 1):
            for sep in (".", ")"):
                prefix = f"{i}{sep}"
                if r.startswith(prefix):
                    r = r[len(prefix) :].strip()
                    break
        cleaned.append(r)
    return cleaned[:batch_size]


# ====================
# Extraction
# ====================

EXTRACTION_SYSTEM = (
    "Tu es un extracteur parfait de tickets de caisse suisses. "
    "Extrais les informations en respectant strictement le schéma. "
    "Utilise null pour les valeurs manquantes ou illisibles. "
    "Les prix sont des nombres décimaux (utilise . et non ,). "
    "Les dates sont toujours au format JJ.MM.AAAA si présentes. "
    "N'invente rien — utilise uniquement ce qui est dans le texte. "
    "Renvoie uniquement l'objet JSON — pas de texte supplémentaire."
)


async def extract_receipt_json(
    semaphore: asyncio.Semaphore, receipt_text: str
) -> dict:
    async with semaphore:
        completion = await client.beta.chat.completions.parse(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": f"Ticket de caisse :\n\n{receipt_text}"},
            ],
            response_format=ReceiptExtraction,
        )
    parsed = completion.choices[0].message.parsed
    return parsed.model_dump(mode="json")


# ====================
# Batch processing
# ====================


async def process_batch(
    semaphore: asyncio.Semaphore, batch_size: int, batch_num: int
) -> list[dict]:
    print(f"  [lot {batch_num}] Génération de {batch_size} tickets...")
    texts = await generate_receipt_text_batch(semaphore, batch_size)
    print(f"  [lot {batch_num}] {len(texts)} tickets générés, extraction en cours...")

    async def extract_one(text: str) -> dict | None:
        try:
            output = await extract_receipt_json(semaphore, text)
            return {
                "instruction": INSTRUCTION,
                "input": text,
                "output": json.dumps(output, ensure_ascii=False),
            }
        except Exception as e:
            print(f"  [lot {batch_num}] Erreur d'extraction : {e}")
            return None

    results = await asyncio.gather(*[extract_one(t) for t in texts])
    entries = [r for r in results if r is not None]
    print(f"  [lot {batch_num}] {len(entries)}/{len(texts)} extractions réussies")
    return entries


# ====================
# Main
# ====================


async def main():
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    num_batches = (NUM_RECEIPTS + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Génération de {NUM_RECEIPTS} tickets en {num_batches} lots (parallélisme max : {MAX_CONCURRENT})...")

    batch_tasks = [
        process_batch(
            semaphore,
            min(BATCH_SIZE, NUM_RECEIPTS - i * BATCH_SIZE),
            i + 1,
        )
        for i in range(num_batches)
    ]
    batch_results = await asyncio.gather(*batch_tasks)

    all_entries = [entry for batch in batch_results for entry in batch]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nTerminé ! {len(all_entries)} entrées sauvegardées dans {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
