"""
Harrier Multilingual Embedding Evaluation — Indian Languages
=============================================================
Tests Microsoft's Harrier-OSS-v1 embedding models on Indian language
legal text to verify cross-lingual quality for languages NOT explicitly
listed on the model card.

Currently configured for: Telugu ↔ English (legal domain)

To test YOUR language:
  1. Replace the sentences in the CONFIG section below
  2. Keep the English translations paired with your language sentences
  3. Run on Colab free tier: pip install sentence-transformers torch

Supports testing: 270M, 0.6B, and 27B model sizes
Expected runtime: ~2-5 minutes per model on Colab T4 GPU

Author: Kanthi
License: MIT
"""

import numpy as np
import gc
import torch
from sentence_transformers import SentenceTransformer


# ═════════════════════════════════════════════════════════════════
# ██  CONFIG — CHANGE THIS SECTION FOR YOUR LANGUAGE  ██
# ═════════════════════════════════════════════════════════════════

LANGUAGE_NAME = "Telugu"  # Change to: Hindi, Kannada, Tamil, Malayalam, etc.

# ── Test 1: Cross-lingual pairs ────────────────────────────────
# Your language sentence + its English equivalent.
# Use domain-specific text (legal, medical, finance, etc.) for
# a meaningful evaluation. Generic text won't test real-world quality.

CROSS_LINGUAL_PAIRS = [
    {
        "source": "ఈ కేసు న్యాయస్థానంలో విచారణకు వచ్చింది",
        "english": "This case came up for hearing before the court",
        "label": "court_hearing"
    },
    {
        "source": "విక్రయదారుడు కొనుగోలుదారునికి ఆస్తిని బదిలీ చేస్తాడు",
        "english": "The vendor transfers the property to the purchaser",
        "label": "property_transfer"
    },
    {
        "source": "ఈ ఆస్తి సర్వే నంబరు 123లో ఉంది",
        "english": "This property is situated in survey number 123",
        "label": "survey_number"
    },
    {
        "source": "ముద్దాయిపై భారతీయ శిక్షాస్మృతి సెక్షన్ 420 కింద కేసు నమోదు చేయబడింది",
        "english": "A case was registered against the accused under Section 420 of the Indian Penal Code",
        "label": "criminal_charge"
    },
    {
        "source": "పిటిషనర్ తరఫున న్యాయవాది వాదనలు వినిపించారు",
        "english": "The advocate for the petitioner presented arguments",
        "label": "legal_argument"
    },
    {
        "source": "ఈ భూమి పట్టాదారు పేరు రామయ్య",
        "english": "The pattadar name for this land is Ramaiah",
        "label": "land_record"
    },
    {
        "source": "రిజిస్ట్రేషన్ ఫీజు చెల్లించి డాక్యుమెంట్ నమోదు చేయబడింది",
        "english": "The document was registered after paying registration fee",
        "label": "registration"
    },
    {
        "source": "న్యాయమూర్తి తీర్పు రిజర్వ్ చేశారు",
        "english": "The judge reserved the judgment",
        "label": "judgment_reserved"
    },
]

# ── Test 2: Paraphrase pairs ──────────────────────────────────
# Two sentences in YOUR language that mean the same thing but
# use different wording.

PARAPHRASE_PAIRS = [
    {
        "sent_a": "ఈ ఆస్తి అమ్మకం రాతప్రతి ఇది",
        "sent_b": "ఈ డాక్యుమెంట్ భూమి విక్రయ పత్రం",
        "label": "sale_deed_paraphrase"
    },
    {
        "sent_a": "కోర్టు ఆదేశాలు పాటించవలెను",
        "sent_b": "న్యాయస్థానం ఉత్తర్వులను అమలు చేయాలి",
        "label": "court_order_paraphrase"
    },
]

# ── Test 3: Document type clustering ──────────────────────────
# Different document types, each with one sentence in YOUR language
# and one in English. Same-type pairs should cluster together.

DOC_TYPE_SAMPLES = {
    "sale_deed": [
        "ఈ విక్రయ పత్రం ద్వారా విక్రయదారుడు కొనుగోలుదారునికి భూమిని బదిలీ చేస్తున్నాడు",
        "This sale deed witnesses that the vendor hereby conveys the property to the purchaser",
    ],
    "court_order": [
        "న్యాయస్థానం క్రింది ఉత్తర్వులు జారీ చేస్తున్నది",
        "The court hereby passes the following order",
    ],
    "fir": [
        "ఫిర్యాదుదారు పోలీసు స్టేషన్‌లో ఫిర్యాదు దాఖలు చేశాడు",
        "The complainant filed a complaint at the police station",
    ],
}

# ── Test 4: Negative controls ─────────────────────────────────
# Unrelated pairs — these should score LOW.

NEGATIVE_PAIRS = [
    {
        "sent_a": "ఈ కేసు న్యాయస్థానంలో విచారణకు వచ్చింది",
        "sent_b": "The weather forecast predicts rain tomorrow",
        "label": "legal_vs_weather"
    },
    {
        "sent_a": "విక్రయదారుడు కొనుగోలుదారునికి ఆస్తిని బదిలీ చేస్తాడు",
        "sent_b": "The recipe calls for two cups of flour and one egg",
        "label": "legal_vs_recipe"
    },
]

# ── Models to test ────────────────────────────────────────────
# Uncomment/add models as needed. 27B requires A100 or better.

MODELS = [
    {
        "name": "harrier-oss-v1-270m",
        "hf_id": "microsoft/harrier-oss-v1-270m",
        "params": "270M",
        "embed_dim": 640,
        "approx_size": "~536 MB",
    },
    {
        "name": "harrier-oss-v1-0.6b",
        "hf_id": "microsoft/harrier-oss-v1-0.6b",
        "params": "0.6B",
        "embed_dim": 1024,
        "approx_size": "~1.2 GB",
    },
    # ── Uncomment ONLY if you have A100/L4 GPU (40GB+ VRAM) ──
    # {
    #     "name": "harrier-oss-v1-27b",
    #     "hf_id": "microsoft/harrier-oss-v1-27b",
    #     "params": "27B",
    #     "embed_dim": 5376,
    #     "approx_size": "~54 GB",
    # },
]

# ═════════════════════════════════════════════════════════════════
# ██  END OF CONFIG — No changes needed below this line  ██
# ═════════════════════════════════════════════════════════════════

# Thresholds
CROSS_LINGUAL_THRESHOLD = 0.70
PARAPHRASE_THRESHOLD = 0.75
SEPARATION_THRESHOLD = 0.10
NEGATIVE_THRESHOLD = 0.40


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def evaluate_model(model_info):
    name = model_info["name"]
    hf_id = model_info["hf_id"]

    print("\n" + "=" * 70)
    print(f"  EVALUATING: {name}")
    print(f"  Params: {model_info['params']}  |  Embed dim: {model_info['embed_dim']}  |  Size: {model_info['approx_size']}")
    print("=" * 70)

    print(f"\n  Loading {hf_id}...")
    try:
        model = SentenceTransformer(hf_id, model_kwargs={"dtype": "auto"})
    except Exception as e:
        print(f"\n  ✗ FAILED to load model: {e}")
        print(f"  Likely cause: insufficient GPU VRAM for {model_info['approx_size']}")
        print(f"  Skipping {name}.\n")
        return None

    print("  Model loaded successfully.\n")

    results = {"name": name, "params": model_info["params"]}

    # ── TEST 1: Cross-lingual similarity ──────────────────────
    print("  " + "-" * 60)
    print(f"  TEST 1: CROSS-LINGUAL SIMILARITY ({LANGUAGE_NAME} ↔ English)")
    print(f"  Threshold: > {CROSS_LINGUAL_THRESHOLD}")
    print("  " + "-" * 60)

    cross_scores = []
    for pair in CROSS_LINGUAL_PAIRS:
        src_emb = model.encode(pair["source"], prompt_name="sts_query")
        en_emb = model.encode(pair["english"], prompt_name="sts_query")
        score = cosine_similarity(src_emb, en_emb)
        cross_scores.append(score)
        status = "✓" if score > CROSS_LINGUAL_THRESHOLD else "✗"
        print(f"    {status} {pair['label']:<25s} → {score:.4f}")

    results["cross_avg"] = np.mean(cross_scores)
    results["cross_scores"] = {p["label"]: s for p, s in zip(CROSS_LINGUAL_PAIRS, cross_scores)}
    print(f"\n    AVERAGE: {results['cross_avg']:.4f}")

    # ── TEST 2: Paraphrase similarity ─────────────────────────
    print(f"\n  " + "-" * 60)
    print(f"  TEST 2: {LANGUAGE_NAME.upper()} PARAPHRASE SIMILARITY")
    print(f"  Threshold: > {PARAPHRASE_THRESHOLD}")
    print("  " + "-" * 60)

    para_scores = []
    for pair in PARAPHRASE_PAIRS:
        a_emb = model.encode(pair["sent_a"], prompt_name="sts_query")
        b_emb = model.encode(pair["sent_b"], prompt_name="sts_query")
        score = cosine_similarity(a_emb, b_emb)
        para_scores.append(score)
        status = "✓" if score > PARAPHRASE_THRESHOLD else "✗"
        print(f"    {status} {pair['label']:<30s} → {score:.4f}")

    results["para_avg"] = np.mean(para_scores)
    print(f"\n    AVERAGE: {results['para_avg']:.4f}")

    # ── TEST 3: Document type clustering ──────────────────────
    print(f"\n  " + "-" * 60)
    print("  TEST 3: DOCUMENT TYPE CLUSTERING")
    print(f"  Same type ({LANGUAGE_NAME}↔English) should score HIGHER than different types")
    print("  " + "-" * 60)

    doc_embeddings = {}
    for doc_type, texts in DOC_TYPE_SAMPLES.items():
        doc_embeddings[doc_type] = [model.encode(t) for t in texts]

    intra_scores = []
    print(f"\n    Intra-type (same doc type, {LANGUAGE_NAME}↔English):")
    for doc_type, embs in doc_embeddings.items():
        score = cosine_similarity(embs[0], embs[1])
        intra_scores.append(score)
        print(f"      {doc_type:<15s} → {score:.4f}")

    inter_scores = []
    types = list(doc_embeddings.keys())
    print("\n    Inter-type (different doc types):")
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            score = cosine_similarity(doc_embeddings[types[i]][1], doc_embeddings[types[j]][1])
            inter_scores.append(score)
            print(f"      {types[i]} vs {types[j]:<15s} → {score:.4f}")

    results["intra_avg"] = np.mean(intra_scores)
    results["inter_avg"] = np.mean(inter_scores)
    results["separation"] = results["intra_avg"] - results["inter_avg"]
    print(f"\n    Avg intra: {results['intra_avg']:.4f}  |  Avg inter: {results['inter_avg']:.4f}  |  Gap: {results['separation']:.4f}")

    # ── TEST 4: Negative controls ─────────────────────────────
    print(f"\n  " + "-" * 60)
    print(f"  TEST 4: NEGATIVE CONTROLS (unrelated pairs, should be < {NEGATIVE_THRESHOLD})")
    print("  " + "-" * 60)

    neg_scores = []
    for pair in NEGATIVE_PAIRS:
        a_emb = model.encode(pair["sent_a"])
        b_emb = model.encode(pair["sent_b"])
        score = cosine_similarity(a_emb, b_emb)
        neg_scores.append(score)
        status = "✓" if score < NEGATIVE_THRESHOLD else "✗"
        print(f"    {status} {pair['label']:<25s} → {score:.4f}")

    results["neg_avg"] = np.mean(neg_scores)
    print(f"\n    AVERAGE: {results['neg_avg']:.4f}")

    # ── Per-model verdict ─────────────────────────────────────
    passes = (
        results["cross_avg"] > CROSS_LINGUAL_THRESHOLD
        and results["para_avg"] > PARAPHRASE_THRESHOLD
        and results["separation"] > SEPARATION_THRESHOLD
        and results["neg_avg"] < NEGATIVE_THRESHOLD
    )
    results["passed"] = passes
    verdict = "ALL PASS ✓" if passes else "FAIL ✗"
    print(f"\n  ► {name}: {verdict}")

    # Clean up GPU memory before loading next model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def print_comparison(all_results):
    """Side-by-side comparison table."""
    valid = [r for r in all_results if r is not None]
    if not valid:
        print("\n  No models were successfully evaluated.")
        return

    print("\n\n" + "=" * 70)
    print(f"  SIDE-BY-SIDE COMPARISON — HARRIER ON {LANGUAGE_NAME.upper()}")
    print("=" * 70)

    # Header
    header = f"  {'Metric':<28s}"
    for m in valid:
        header += f"  {m['params']:>10s}"
    print(header)
    print("  " + "-" * (28 + 12 * len(valid)))

    # Rows
    metrics = [
        ("Cross-lingual avg", "cross_avg", f"> {CROSS_LINGUAL_THRESHOLD}"),
        ("Paraphrase avg", "para_avg", f"> {PARAPHRASE_THRESHOLD}"),
        ("Doc-type intra (same)", "intra_avg", "higher=better"),
        ("Doc-type inter (diff)", "inter_avg", "lower=better"),
        ("Separation gap", "separation", f"> {SEPARATION_THRESHOLD}"),
        ("Negative control avg", "neg_avg", f"< {NEGATIVE_THRESHOLD}"),
    ]

    for label, key, note in metrics:
        row = f"  {label:<28s}"
        for m in valid:
            val = m.get(key, None)
            if val is not None:
                row += f"  {val:>10.4f}"
            else:
                row += f"  {'N/A':>10s}"
        row += f"   ({note})"
        print(row)

    # Verdicts
    print("\n  " + "-" * (28 + 12 * len(valid)))
    verdict_row = f"  {'VERDICT':<28s}"
    for m in valid:
        verdict_row += f"  {'ALL PASS ✓':>10s}" if m["passed"] else f"  {'FAIL ✗':>10s}"
    print(verdict_row)

    # Recommendation
    passing = [m for m in valid if m["passed"]]
    failing = [m for m in valid if not m["passed"]]

    print("\n" + "=" * 70)
    print(f"  RECOMMENDATION FOR {LANGUAGE_NAME.upper()}")
    print("=" * 70)

    if len(passing) == len(valid):
        best = max(passing, key=lambda m: m["cross_avg"])
        smallest = min(passing, key=lambda m: float(m["params"].replace("B", "000").replace("M", "")))
        print(f"""
  All models pass for {LANGUAGE_NAME}!

  Best quality:       {best['name']} (cross-lingual: {best['cross_avg']:.4f})
  Most efficient:     {smallest['name']} (cross-lingual: {smallest['cross_avg']:.4f})

  If scores are close, prefer the smaller model — less memory,
  faster inference, lower deployment cost, same quality.
        """)
    elif passing:
        print(f"\n  Passing: {', '.join(m['name'] for m in passing)}")
        print(f"  Failing: {', '.join(m['name'] for m in failing)}")
        best = max(passing, key=lambda m: m["cross_avg"])
        print(f"\n  ★ Recommended: {best['name']} (cross-lingual: {best['cross_avg']:.4f})")
        print(f"\n  Bigger model does NOT always mean better for {LANGUAGE_NAME}.")
        print("  Always benchmark on your specific language and domain.\n")
    else:
        print(f"""
  No models passed all thresholds for {LANGUAGE_NAME}.

  Alternatives to consider for Indian languages:
    • Cohere Embed Multilingual v3 (100+ languages)
    • Ola Krutrim Vyakyarth (10 Indian languages)
    • AI4Bharat IndicBERT (12 Indian languages)
    • OpenAI text-embedding-3-large (multilingual)
        """)

    print("=" * 70)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print(f"HARRIER EMBEDDING EVALUATION — {LANGUAGE_NAME.upper()}")
    print("Testing Microsoft Harrier-OSS-v1 on Indian language text")
    print("=" * 70)
    print(f"\nLanguage: {LANGUAGE_NAME}")
    print(f"Models to test: {', '.join(m['name'] for m in MODELS)}")
    print(f"Test pairs: {len(CROSS_LINGUAL_PAIRS)} cross-lingual, "
          f"{len(PARAPHRASE_PAIRS)} paraphrase, "
          f"{len(DOC_TYPE_SAMPLES)} doc types, "
          f"{len(NEGATIVE_PAIRS)} negative controls\n")

    all_results = []
    for model_info in MODELS:
        result = evaluate_model(model_info)
        all_results.append(result)

    print_comparison(all_results)
