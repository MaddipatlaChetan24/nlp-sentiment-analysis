# Reproducibility

## Random Seeds

All randomness is fixed via `config.py`:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Dependencies

All dependencies are pinned in `requirements.txt` with minimum versions:

```
torch>=1.12.0
transformers>=4.20.0
datasets>=2.0.0
scikit-learn>=1.0.0
flask>=2.0.0
...
```

Install with:

```bash
pip install -r requirements.txt
```

## Dataset

The IMDB dataset is downloaded automatically from HuggingFace `datasets` on first run. No manual download needed.

## Hardware

Results were produced on:
- Apple M2 (MPS backend for PyTorch)
- 16 GB unified memory
- macOS

Results may vary slightly on different hardware, but the fixed seed ensures determinism within the same environment.
