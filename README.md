# Attention-Log

A training-free framework for log anomaly detection that repurposes the internal
attention of instruction-tuned LLMs. Given a log entry and a fixed system
instruction, normal logs cause the model's first generated token to attend
strongly to the instruction; anomalous logs disrupt this pattern. A small subset
of "important heads" identified from 50 labeled samples per class is sufficient
to turn this divergence into a reliable anomaly score — no fine-tuning, no log
parsing, no gradient updates.

The framework was originally derived from
[Attention Tracker](https://arxiv.org/abs/2411.00348) (Hung et al., 2024) for
prompt injection detection, and adapted here to the log anomaly detection
setting.

---

## Repository Layout

```
Attention-Tracker/
├── configs/model_configs/      # per-model config (heads, weights, prompt template)
├── data/                       # dataset loaders + raw logs (gitignored)
│   ├── bgl.py / bgl/
│   ├── spirit.py / spirit/     # the "Liberty" dataset in the report
│   ├── thunderbird.py / thunderbird/
│   └── windowed.py             # multi-line sliding-window loader
├── detector/attn.py            # AttentionDetector: scoring + thresholding
├── models/attn_model.py        # HF wrapper that returns attention maps
├── prepare_data/               # sliding/session window preprocessing
├── select_head.py              # find important heads on a small calibration set
├── run_dataset.py              # full evaluation loop (AUC, F1, P, R, FPR, FNR)
├── search_hyperparams.py       # grid search over instructions / num_data / head method
├── visualize_heads.py          # heatmaps of selected heads
├── scripts/                    # entry-point shell scripts (see below)
└── result/                     # JSON outputs by dataset
```

### Supported models

`llama3_8b-attn`, `mistral_7b-attn`, `granite3_8b-attn`, `qwen2-attn`,
`gemma2_9b-attn`, `phi3-attn` (configs in `configs/model_configs/`).

### Supported datasets

`bgl`, `spirit` (= **Liberty** in the report), `thunderbird`. Single-line and
windowed (multi-line sequence) modes are both supported.

---

## Setup

```bash
pip install -r requirements.txt
```

Requires CUDA + a GPU large enough to host the chosen LLM (one 24 GB card is
enough for the 7–9 B models used here).

---

## Quickstart

### 1. Download datasets

```bash
bash scripts/download_bgl.sh
bash scripts/download_spirit.sh         # truncated 5 M-line version (~50 MB)
bash scripts/download_thunderbird.sh    # ~2 GB compressed
```

### 2. Run the full pipeline (single-line logs)

```bash
# Pick the heads (writes them to stdout — copy the top-k list)
bash scripts/find_heads_all.sh llama3_8b-attn

# Paste those heads into scripts/run_dataset_all.sh, then evaluate
bash scripts/run_dataset_all.sh llama3_8b-attn
```

Per-sample scores and metrics are written to
`result/<dataset>/<model>-<seed>.json` and appended to
`result/<dataset>/result.jsonl`.

### 3. Run on windowed (multi-line) sequences

```bash
bash scripts/preprocess.sh bgl                    # → data/bgl/{train,test}.csv
bash scripts/run_windowed.sh llama3_8b-attn bgl
```

### 4. Visualize selected heads

```bash
bash scripts/visualize_all.sh llama3_8b-attn      # → plots/llama3_8b-attn/<dataset>/
```

---

## Ablation Studies

The codebase ships with a single grid-search driver, `search_hyperparams.py`,
that owns three knobs simultaneously:

| Knob | Variable | Meaning |
|------|----------|---------|
| Instruction prompt | `DEFAULT_INSTRUCTIONS` | The fixed system instruction prepended to each log entry. |
| Calibration size  | `DEFAULT_NUM_DATA_VALUES` | Number of normal+anomaly samples used for head selection. |
| Head selection    | `HEAD_METHODS` | `pos_div(n)` (mean − n·std > 0) and `top_k(portion)` (top-k% by divergence). |

It loads the LLM **once**, runs inference on the full pool **once per
instruction**, and then re-uses the cached attention maps for every
`(num_data, head_method)` combination. This makes a full sweep cost only
slightly more than a single evaluation run.

### How to run one ablation

```bash
bash scripts/search_ablation.sh llama3_8b-attn bgl
```

Output: `result/search/bgl/llama3_8b-attn_search.json` with `best` and
`all_results` (sorted by AUC).

### How to design each ablation

All four ablations below are run by editing the constants at the top of
[`search_hyperparams.py`](search_hyperparams.py) and then re-running
`scripts/search_ablation.sh`.

#### A. Calibration size sensitivity (how few labels do we really need?)

```python
DEFAULT_INSTRUCTIONS = ["Summarize this standard server log entry"]
DEFAULT_NUM_DATA_VALUES = [10, 25, 50, 100, 200, 500]
HEAD_METHODS = [("top_k", {"portion": 0.01})]
```

Run on all three datasets × five models. **Plot:** AUC vs. `num_data` (one line
per model, faceted by dataset). **Expected story:** AUC plateaus by
`num_data ≈ 50`, supporting the "training-free, ≤100 labels" claim.

#### B. Head selection percentage `k`

```python
DEFAULT_INSTRUCTIONS = ["Summarize this standard server log entry"]
DEFAULT_NUM_DATA_VALUES = [50]
HEAD_METHODS = [
    ("top_k", {"portion": 0.005}),
    ("top_k", {"portion": 0.01}),
    ("top_k", {"portion": 0.02}),
    ("top_k", {"portion": 0.05}),
    ("top_k", {"portion": 0.10}),
]
```

**Plot:** AUC and number-of-selected-heads vs. `k`. **Expected story:** very
small `k` (≤2 %) is enough; performance degrades with too many heads as noisy
heads dilute the signal.

#### C. Instruction sensitivity

```python
DEFAULT_INSTRUCTIONS = [
    "Say xxxxxx",
    "Repeat this normal log entry exactly",
    "Summarize this standard server log entry",
    "Classify whether this log line is normal or abnormal",
    "Analyze the following system log message",
]
DEFAULT_NUM_DATA_VALUES = [50]
HEAD_METHODS = [("top_k", {"portion": 0.01})]
```

**Table:** AUC × instruction × dataset for one model (e.g. LLaMA-3-8B).
**Expected story:** task-relevant prompts ("Summarize…", "Classify…") beat the
nonsense baseline `Say xxxxxx`, but the spread is small — the method is robust
to prompt phrasing.

#### D. Head selection method (`pos_div` vs. `top_k`)

```python
DEFAULT_INSTRUCTIONS = ["Summarize this standard server log entry"]
DEFAULT_NUM_DATA_VALUES = [50]
HEAD_METHODS = [
    ("pos_div", {"n": 1}),
    ("pos_div", {"n": 2}),
    ("pos_div", {"n": 3}),
    ("top_k", {"portion": 0.01}),
    ("top_k", {"portion": 0.05}),
]
```

**Table:** AUC, # heads selected, both methods. **Expected story:** `top_k`
gives more stable head counts; `pos_div` adapts to signal strength but can
return zero heads on weak datasets (BGL with some models).

### After the sweep

Each `result/search/<dataset>/<model>_search.json` contains
`all_results` sorted by AUC. To produce the final ablation tables/figures, write
a small notebook that loads these JSONs across models and datasets and emits
matplotlib plots (no extra inference needed — the search already cached
everything).

---

## Citation

This project builds on Attention Tracker:

```bibtex
@misc{hung2024attentiontrackerdetectingprompt,
  title  = {Attention Tracker: Detecting Prompt Injection Attacks in LLMs},
  author = {Kuo-Han Hung and Ching-Yun Ko and Ambrish Rawat and I-Hsin Chung
            and Winston H. Hsu and Pin-Yu Chen},
  year   = {2024},
  eprint = {2411.00348},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  url    = {https://arxiv.org/abs/2411.00348}
}
```

### License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en)
