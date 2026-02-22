"""Milton: a small GPT-style transformer decoder.

8 layers, 512 embedding dim, 8 heads, RoPE, ~25M parameters.
Trained on exactly one text.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    """Rotary positional embeddings."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: (batch, heads, seq, head_dim)
    # cos, sin: (seq, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, n_heads, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim)
        self.attn = Attention(dim, n_heads, dropout)
        self.ff_norm = nn.RMSNorm(dim)
        self.ff = FeedForward(dim, ff_dim, dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ff(self.ff_norm(x))
        return x


class Milton(nn.Module):
    """A language model that knows only Paradise Lost."""

    def __init__(
        self,
        vocab_size: int = 4096,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        ff_dim: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.rope = RoPE(dim // n_heads, max_seq_len)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [TransformerBlock(dim, n_heads, ff_dim, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (batch, seq_len) token indices
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = idx.shape
        x = self.tok_emb(idx)
        x = self.drop(x)

        cos, sin = self.rope(T)
        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation with nucleus sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = idx[:, -self.max_seq_len :]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

            if next_token.item() == 1:  # eos
                break

        return idx

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = Milton()
    print(f"Milton: {model.count_parameters():,} parameters")
    dummy = torch.randint(0, 4096, (1, 64))
    logits = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {logits.shape}")
