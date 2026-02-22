```
  __  __ _ _ _
 |  \/  (_) | |
 | \  / |_| | |_ ___  _ __
 | |\/| | | | __/ _ \| '_ \
 | |  | | | | || (_) | | | |
 |_|  |_|_|_|\__\___/|_| |_|
```

A language model trained on Paradise Lost and nothing else.

35M parameter transformer, custom BPE tokenizer, trained from scratch on a single text. No pre-training, no fine-tuning, no instruction data. Milton's entire knowledge of language comes from one poem.

## Quickstart

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 get_data.py
python3 tokenizer.py
python3 train.py
python3 chat.py
```

Training takes ~75 minutes on an M-series Mac (MPS) or a T4 GPU.

## Model

| | |
|---|---|
| Parameters | 35.6M |
| Architecture | Decoder-only transformer |
| Layers | 8 |
| Embedding dim | 512 |
| Attention heads | 8 |
| FFN | SwiGLU, 2048 hidden |
| Positional encoding | RoPE |
| Context window | 512 tokens |
| Vocabulary | 4,096 BPE tokens trained on the text |
| Training data | Paradise Lost, Books I–XII (124,831 tokens) |

## Files

| File | |
|---|---|
| `get_data.py` | Download and clean Paradise Lost from Project Gutenberg |
| `tokenizer.py` | Train a BPE tokenizer on the text |
| `model.py` | Transformer architecture |
| `train.py` | Training loop |
| `chat.py` | Terminal chat interface |
