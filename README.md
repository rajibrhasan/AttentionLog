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
`gemma2_9b-attn`(configs in `configs/model_configs/`).

### Supported datasets

`bgl`, `spirit`, `thunderbird`. Single-line and
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
