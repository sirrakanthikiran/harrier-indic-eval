# Harrier Indian Language Embedding Evaluation

Tests Microsoft's [Harrier-OSS-v1](https://huggingface.co/microsoft/harrier-oss-v1-270m) 
embedding models on Indian languages not explicitly listed on the model card.

## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sirrakanthikiran/harrier-indic-eval/blob/main/harrier_indic_eval.ipynb)
```bash
# Or run locally
pip install sentence-transformers torch numpy
python harrier_indic_eval.py
```

## What it tests

- **Cross-lingual similarity** — your language ↔ English sentence pairs
- **Paraphrase detection** — same meaning, different wording
- **Document type clustering** — can it tell different doc types apart?
- **Negative controls** — unrelated pairs should score low

## Test your language

Edit the CONFIG section at the top of the notebook or script. Replace the 
Telugu sentences with your language — Hindi, Kannada, Tamil, Malayalam, Bengali, 
Marathi, or any other. Run it. Share your scores!

## Telugu Results (Legal Domain)

| Metric | 270M | 0.6B |
|--------|------|------|
| Cross-lingual avg | 0.9002 ✓ | 0.9018 ✓ |
| Paraphrase avg | 0.8903 ✓ | 0.9004 ✓ |
| Doc-type separation | 0.1594 ✓ | 0.0997 ✗ |
| Negative control | 0.3909 ✓ | 0.4031 ✗ |
| **Verdict** | **ALL PASS** | **FAIL** |

The smaller 270M model outperformed the 0.6B for Telugu legal text.

## Files

| File | Purpose |
|------|---------|
| `harrier_indic_eval.ipynb` | Colab notebook — click the badge above to run instantly |
| `harrier_indic_eval.py` | Standalone script — for local runs or pipeline integration |

## License

MIT
