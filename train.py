"""Train Milton on Paradise Lost."""

import math
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from model import Milton
from retriever import Retriever
from tokenizer import Tokenizer

DATA_DIR = Path(__file__).parent / "data"
CKPT_DIR = Path(__file__).parent / "checkpoints"

# --- Hyperparameters ---
SEQ_LEN = 512
BATCH_SIZE = 16
EPOCHS = 120
LR = 3e-4
MIN_LR = 1e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.1
DROPOUT = 0.2
WORD_DROPOUT = 0.05
GRAD_CLIP = 1.0
DIM = 512
N_LAYERS = 8
N_HEADS = 8
FF_DIM = 2048

# Fraction of training data formatted as chat turns
CHAT_FRACTION = 0.30


class ParadiseLostDataset(Dataset):
    """Paradise Lost as overlapping sequences for causal LM training.

    Mixes raw text sequences with chat-formatted sequences so the model
    learns both Milton's language and the turn-taking pattern.
    """

    def __init__(self, tokens: list[int], tok: Tokenizer, seq_len: int = SEQ_LEN, stride: int = 256):
        self.seq_len = seq_len
        self.sequences: list[list[int]] = []

        # Raw text sequences (overlapping windows)
        for i in range(0, len(tokens) - seq_len, stride):
            self.sequences.append(tokens[i : i + seq_len])

        # Chat-formatted sequences: <|user|> [context] <|milton|> [continuation] <|eos|>
        n_chat = int(len(self.sequences) * CHAT_FRACTION / (1 - CHAT_FRACTION))
        chat_seqs = self._make_chat_sequences(tokens, tok, n_chat, seq_len)
        self.sequences.extend(chat_seqs)

    def _make_chat_sequences(
        self, tokens: list[int], tok: Tokenizer, count: int, seq_len: int
    ) -> list[list[int]]:
        """Create topically aligned chat sequences from Paradise Lost.

        For each passage: extract distinctive keywords via TF-IDF and use
        them (or a short phrase from the passage) as the user prompt, with
        the passage text as Milton's response. This teaches the model to
        associate topic words with relevant passages.
        """
        import random

        random.seed(42)
        retriever = Retriever()
        seqs: list[list[int]] = []
        n_passages = len(retriever.passages)
        reps = max(1, count // n_passages + 1)

        for rep in range(reps):
            for pidx, passage in enumerate(retriever.passages):
                if len(seqs) >= count:
                    break

                passage_tokens = tok.encode(passage)
                if len(passage_tokens) < 30:
                    continue

                if rep % 2 == 0:
                    # Keyword prompt: top TF-IDF keywords for this passage
                    kw = retriever.keywords(pidx, top_n=random.randint(3, 5))
                    prompt_text = " ".join(kw)
                else:
                    # Phrase prompt: random short phrase extracted from the passage
                    words = passage.split()
                    if len(words) > 6:
                        start = random.randint(0, len(words) - 6)
                        prompt_text = " ".join(words[start : start + random.randint(3, 6)])
                    else:
                        prompt_text = passage[:60]

                prompt_tokens = tok.encode(prompt_text)

                max_response = seq_len - len(prompt_tokens) - 3
                if max_response < 30:
                    continue

                response_tokens = passage_tokens[:max_response]

                seq = (
                    [tok.user_id]
                    + prompt_tokens
                    + [tok.milton_id]
                    + response_tokens
                    + [tok.eos_id]
                )

                if len(seq) < seq_len:
                    seq = seq + [tok.pad_id] * (seq_len - len(seq))
                else:
                    seq = seq[:seq_len]

                seqs.append(seq)

            if len(seqs) >= count:
                break

        random.shuffle(seqs)
        return seqs[:count]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_lr(step: int, total_steps: int) -> float:
    """Cosine schedule with linear warmup."""
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def train():
    device = get_device()
    print(f"Device: {device}")

    # Load tokenizer and data
    tok = Tokenizer.load(DATA_DIR / "tokenizer.json")
    tokens_path = DATA_DIR / "tokens.pt"
    if tokens_path.exists():
        tokens = torch.load(tokens_path, weights_only=True).tolist()
        print(f"Paradise Lost: {len(tokens)} tokens (cached)")
    else:
        print("Encoding Paradise Lost (first run only)...")
        text = (DATA_DIR / "paradise_lost.txt").read_text(encoding="utf-8")
        tokens = tok.encode(text)
        torch.save(torch.tensor(tokens, dtype=torch.int32), tokens_path)
        print(f"Paradise Lost: {len(tokens)} tokens")

    # Dataset
    dataset = ParadiseLostDataset(tokens, tok, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"Dataset: {len(dataset)} sequences ({len(dataset) - int(len(dataset) * CHAT_FRACTION / (1 + CHAT_FRACTION))} raw + chat-formatted)")

    # Model
    model = Milton(
        vocab_size=tok.vocab_size,
        dim=DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ff_dim=FF_DIM,
        max_seq_len=SEQ_LEN,
        dropout=DROPOUT,
    ).to(device)
    print(f"Milton: {model.count_parameters():,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95)
    )

    total_steps = EPOCHS * len(loader)
    print(f"Training: {EPOCHS} epochs, {len(loader)} steps/epoch, {total_steps} total steps")
    print()

    CKPT_DIR.mkdir(exist_ok=True)
    best_loss = float("inf")
    step = 0

    t_start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t_epoch = time.time()

        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            # Word-level dropout: randomly replace input tokens to prevent memorization
            if WORD_DROPOUT > 0:
                mask = torch.rand_like(x, dtype=torch.float) < WORD_DROPOUT
                random_tokens = torch.randint(0, tok.vocab_size, x.shape, device=device)
                x = torch.where(mask, random_tokens, x)

            lr = get_lr(step, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=tok.pad_id,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            non_pad = (y != tok.pad_id).sum().item()
            epoch_loss += loss.item() * non_pad
            epoch_tokens += non_pad
            step += 1

        avg_loss = epoch_loss / epoch_tokens
        ppl = math.exp(min(avg_loss, 20))
        dt = time.time() - t_epoch

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t_start
            print(f"epoch {epoch + 1:>3}/{EPOCHS}  loss={avg_loss:.4f}  ppl={ppl:.1f}  lr={lr:.2e}  {dt:.1f}s/ep  [{elapsed:.0f}s elapsed]")

        # Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": avg_loss,
                },
                CKPT_DIR / "milton_best.pt",
            )

        if (epoch + 1) % 50 == 0:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "loss": avg_loss},
                CKPT_DIR / f"milton_epoch{epoch + 1}.pt",
            )

            # Generate a sample
            model.eval()
            prompt = tok.encode("Of Man")
            prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
            out = model.generate(prompt_t, max_new_tokens=100, temperature=0.8, top_p=0.9)
            sample = tok.decode(out[0].tolist())
            print(f"  sample: {sample[:200]}")
            model.train()

    # Save final
    torch.save(
        {"model": model.state_dict(), "epoch": EPOCHS - 1, "loss": avg_loss},
        CKPT_DIR / "milton_final.pt",
    )
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {CKPT_DIR}")


if __name__ == "__main__":
    train()
