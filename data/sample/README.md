# Sample data

This directory contains a **synthetic** testing-and-inspection dataset used for smoke
tests and tutorials. It is not the real-world dataset from the paper.

| File | Rows | Description |
| --- | ---: | --- |
| `users.csv` | 240 | `user_id,region,industry` |
| `businesses.csv` | 80 | `business_id,region` |
| `services.csv` | 24 | `service_id,category,standard_id` |
| `business_services.csv` | 360 | provider → accredited service |
| `interactions.csv` | 3,098 | user → business → service with timestamp |

The generator creates latent category and region preferences, so a meta-path model can
outperform a popularity baseline. See [`docs/dataset.md`](../docs/dataset.md) for the
schema and the split protocol.

Regenerate with:

```bash
python scripts/generate_synthetic.py --out data/sample --seed 42
```
