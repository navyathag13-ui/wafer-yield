# Wafer Yield: defect patterns and lot SPC on 811K real wafer maps

A chip fab tests every die on a wafer and marks it pass or fail. Drawn as a map, the fails
often form a shape (a ring at the edge, a blob in the centre, a thin scratch), and each shape
points to a different process or tool problem. This project takes the public **WM-811K**
dataset (811,457 wafer maps from real fab production, Wu et al. 2015) and builds four
things a yield team uses:

1. **A checked data pipeline:** the raw pickle is loaded safely, every wafer is checked and the result lands in Parquet.
2. **Pattern classifiers:** a hand-built feature model, a small CNN and an ensemble of the two that flags wafers it is unsure about for an engineer to check.
3. **Similar-wafer search:** given a bad wafer, the most similar past wafers and their labels ("have we seen this before?").
4. **Lot-level SPC:** p-charts and Laney p' charts per product that flag abnormal lots, plus a FastAPI service over all of it.

Every number below comes from `results/` and was produced by the scripts in this repo.

![Real Test wafers with the ensemble's answers, and similar-wafer search for a scratch](results/examples.png)

*Randomly picked Test wafers, misses included (`scripts/examples.py`).*

## Results

### Data pipeline (`scripts/run_pipeline.py`, 36 to 86 s for the ETL on a laptop)

| | |
|---|---|
| Wafers read / loaded / quarantined | 811,457 / 811,457 / 0 |
| Labeled with a pattern | 172,950 (21%); Training 54,355, Test 118,595 |
| Lots | 46,293 (median 24 wafers) |
| Distinct map sizes | 632 |
| Parquet output | 144 MB (zstd), from a 2.1 GB pickle |

The checks (die codes only 0/1/2, at least one die, stored `dieSize` equal to the dies on the
map, wafer index 1 to 25, no duplicate lot and index) found nothing to quarantine or flag.
I confirmed that with a separate script on the raw DataFrame rather than trusting a zero. The
checks still run on every load and are tested on bad inputs in `tests/test_etl.py`.

Yield by labeled pattern (mean die yield): Near-full 12%, Random 52%, Donut 72%, Center 77%,
Edge-Loc 82%, Edge-Ring 85%, Loc 85%, no pattern 89%, Scratch 90%.

### Pattern classification (Test split: 118,595 wafers from lots never seen in training)

Both models train on the same 49,188 Training wafers. Another 5,167 wafers, from whole lots
held out of Training, pick the CNN's best epoch, the ensemble weight and the review threshold.

| Model | Accuracy | Macro F1 (9 classes) | Macro F1 (8 defect classes) | Train time |
|---|---|---|---|---|
| Gradient-boosted trees on 20 hand-made features | 93.6% | 0.670 | 0.632 | 39 s (CPU) |
| CNN, 73K weights, 15 epochs | **95.5%** | 0.678 | 0.640 | 7.7 min (Apple MPS) |
| **Ensemble** (0.3 CNN + 0.7 trees) | 94.7% | **0.700** | **0.665** | (both) |

Per class F1 for trees / CNN / ensemble:

| Pattern | Trees | CNN | Ensemble |
|---|---|---|---|
| Center | 0.45 | 0.49 | 0.50 |
| Donut | 0.66 | 0.56 | 0.68 |
| Edge-Loc | 0.58 | 0.64 | 0.61 |
| Edge-Ring | 0.76 | 0.75 | 0.79 |
| Loc | 0.51 | 0.49 | 0.55 |
| Near-full | 0.94 | 0.93 | 0.95 |
| Random | 0.70 | 0.71 | 0.71 |
| Scratch | 0.47 | 0.56 | 0.52 |
| none | 0.97 | 0.98 | 0.98 |

The ensemble trades a little accuracy on clean wafers for more defects caught. Defect wafers
called "none" drop from 1,820 (CNN) to 1,197 of 7,894. Clean wafers called a defect rise from
2,330 to 4,029 of 110,701. For a fab, a missed defect usually costs more than a second look.
`results/confusion_ensemble.png` has the full matrix.

**"Needs review" flag.** When the ensemble's top probability is below 0.785, the wafer goes to
an engineer instead of being labeled automatically:

| Confidence cut-off | Wafers sent to review | Accuracy on the rest | Macro F1 on the rest |
|---|---|---|---|
| none (label everything) | 0% | 94.7% | 0.700 |
| 0.6 | 4.2% | 96.7% | 0.767 |
| **0.785 (chosen on validation)** | **10.7%** | **98.9%** | **0.880** |
| 0.9 | 17.0% | 99.6% | 0.924 |

The cut-off was set to send 5% of validation wafers to review. On Test it sends 10.7%, another
sign that the Test lots differ from Training. Confidence is not a full safety net: some wrong
answers are confident. The first Scratch and Center wafers in Test both come back as "none"
with confidence above 0.9.

### Similar-wafer search (`wafer_yield/similar.py`)

Each wafer becomes 64 numbers from the CNN's last layer, and search returns the closest past
wafers by cosine similarity. Scored on all 7,894 Test defect wafers searching the 49,188
Training wafers (always different lots): what share of the 10 nearest wafers carry the same label?

| Pattern | CNN embedding | Plain pixel match |
|---|---|---|
| **All defects** | **59.6%** | 26.0% |
| Center | 58.6% | 19.3% |
| Donut | 44.7% | 14.1% |
| Edge-Loc | 65.6% | 31.7% |
| Edge-Ring | 73.7% | 71.7% |
| Loc | 46.2% | 4.1% |
| Near-full | 82.3% | 100% |
| Random | 66.6% | 0.7% |
| Scratch | 50.0% | 0.6% |

Pixel matching only works when the defect sits in the same place (an edge ring, an almost fully
failed wafer). A scratch or random fails at different spots look nothing alike pixel by pixel,
while the CNN embedding still finds them. One query over 172,950 wafers takes about 50 ms
through the API.

### Lot SPC (`wafer_yield/spc.py`)

| Setup | Lots flagged by p-chart | Laney sigma_z | Lots flagged by Laney p' | Run-rule hits |
|---|---|---|---|---|
| One chart for every product | 96.6% | 31.3 | 5.2% | 18,286 |
| One chart per product (202 products, 44,221 lots) | 83.2% | 11.1 (median) | **4.2% (1,861 lots)** | 6,209 |

## What went wrong and what I changed

These are the most useful parts of the project, so they stay in the README.

- **Diagonal scratches were invisible to the cluster features.** Fail clusters were joined by
  4-connectivity (up, down, left, right), so a diagonal scratch split into single dies and
  looked like random noise. A synthetic-scratch test caught it, and clusters now use
  8-connectivity.
- **One SPC chart across all products measured product mix, not process.** The first chart put
  all 46,293 lots on one centre line (6.7% fail). Lots 6 to 290 sat steadily near 17%,
  nowhere near that line, and the plain p-chart flagged 96.6% of lots. Charting each product (here, each map size)
  separately cut run-rule hits from 18,286 to 6,209.
- **Plain p-chart limits are too tight for fab lots.** A lot has thousands of dies, so binomial
  limits are narrow and normal lot-to-lot spread lands outside them: 83% of lots flagged even
  per product. Laney's p' measured that extra spread (sigma_z 11) and widened the limits, leaving
  4.2% flagged. Limits are also capped at 1.0, since a fail rate can't exceed 100%.
- **My first validation split leaked.** I held out random wafers from the Training split to pick
  the best epoch. Wafers from one lot share their defect, so the same lots sat on both sides
  and validation said 0.937 macro F1 while Test said 0.675
  (`results/metrics_random_wafer_validation.json`). Validation now holds out whole lots, as the
  official split does, and reads 0.889.
- **The remaining gap to Test is real distribution shift, not a bug.** The official Test split
  is 93% "none" against 68% in Training, has far fewer Center and Edge-Ring wafers and holds
  8,263 wafers from map sizes that never appear in Training. Per-class precision drops when
  the clean class dominates.
- **Loading the data safely.** WM-811K ships as a Python 2 pandas pickle, and a plain
  `pickle.load` will run any code a tampered copy contains. `wafer_yield/source.py` allows only
  the numpy and pandas classes a DataFrame needs, and translates Python 2 names before
  checking. Tests confirm it blocks `os.system` and `pandas.read_csv` payloads.
- **A fix that did not move the number.** Shrinking a large map to 64 x 64 by sampling rows and
  columns can skip a one-die-wide scratch, so the CNN now uses a max-pool resize that keeps any
  failed die (tested on a scratch nearest-neighbour misses). On real data CNN Scratch F1 stayed
  at 0.557 (0.5571 before, 0.5570 after). Only 2% of maps are larger than 64 on a side, and
  185 of 1,193 scratches, so the loss was smaller than I expected. The fix stays because it is
  correct, but the gain on this dataset is zero.

## Limits

- Map size stands in for "product". Inside the biggest size, wafers split into a near-zero-fail
  group (20% of wafers) and a group with roughly 15 to 20% fails, so it is a rough product ID.
- Lots have names, not timestamps. Ordering by lot number shows how the run rule works, but it
  is not proof of a drift over time.
- Scratch, Center and Loc stay the weakest classes (ensemble F1 0.50 to 0.55). Most misses are
  called "none", and some of those are made with high confidence, so the review flag does not
  catch them.
- The Training and Test splits differ (93% vs 68% clean wafers, unseen map sizes), so a
  threshold tuned on Training lots sends twice the expected share to review on Test.

## Run it

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
# LSWMD.pkl from kaggle.com/datasets/qingyi/wm811k-wafer-map (CC0, 2.1 GB, sign-in needed)
.venv/bin/python scripts/run_pipeline.py ~/Downloads/LSWMD.pkl
.venv/bin/python scripts/train.py --epochs 15
.venv/bin/uvicorn wafer_yield.api:app
```

API:

- `POST /classify` takes a map (`{"wafer_map": [[0,1,2,...], ...]}`) and returns the ensemble's pattern, confidence, `needs_review`, top 3 and die yield.
- `POST /similar` takes a map and `k` and returns the k most similar labeled wafers.
- `GET /similar/{lot}/{wafer_index}` does the same for a known wafer, skipping its own lot.
- `GET /lots/{lot}` returns a lot's yield, limits and status.
- `GET /spc/signals` lists lots outside the Laney limits, worst first.

Bad maps return 422 with the reason (for example `unknown die codes [3]`).

## Tests

`pytest -q`: 26 tests, all on synthetic wafers (`tests/synthetic.py`), so CI never needs the
2 GB file. They cover the checks and quarantine counts, the features on known shapes, p-chart
and Laney math, product separation, the run rule, the safe loader, the fail-keeping resize, similarity ranking and every API endpoint. GitHub Actions
runs them on every push.

## Layout

```
wafer_yield/source.py    safe loader for the pickle
wafer_yield/etl.py       checks, quarantine, Parquet
wafer_yield/features.py  20 features: rings, sectors, fail clusters
wafer_yield/maps.py      die codes, nearest and fail-keeping resize
wafer_yield/cnn.py       CNN, augmentation, prediction, embeddings
wafer_yield/similar.py   cosine search over embeddings, precision@k
wafer_yield/spc.py       lot table, p and Laney p' charts per product, run rule
wafer_yield/api.py       FastAPI service
scripts/                 run_pipeline.py, train.py, examples.py
results/                 metrics, charts, logs, trained models, embeddings
```

Data: Wu, Jang and Chen, "Wafer Map Failure Pattern Recognition and Similarity Ranking for
Large-Scale Data Sets," IEEE Transactions on Semiconductor Manufacturing, 2015. Released as WM-811K.
