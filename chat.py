"""Talk to Milton."""

import argparse
import os
import shutil
import sys
import textwrap
import threading
import time
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import torch
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.prompt import Prompt

from model import Milton
from tokenizer import Tokenizer

DATA_DIR = Path(__file__).parent / "data"
CKPT_DIR = Path(__file__).parent / "checkpoints"

console = Console()

# --- Colors ---
GOLD = "#c9a84c"
DIM = "#5a5a5a"
IVORY = "#d4cfc4"
FADED = "#8a8578"
RUST = "#a0522d"


def load_milton(checkpoint: str | None = None) -> tuple[Milton, Tokenizer, torch.device]:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tok = Tokenizer.load(DATA_DIR / "tokenizer.json")

    ckpt_path = Path(checkpoint) if checkpoint else CKPT_DIR / "milton_final.pt"
    if not ckpt_path.exists():
        ckpt_path = CKPT_DIR / "milton_best.pt"
    if not ckpt_path.exists():
        console.print(f"[red]No checkpoint found at {ckpt_path}[/red]")
        console.print("[dim]Run train.py first.[/dim]")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    model = Milton(
        vocab_size=tok.vocab_size,
        dim=512,
        n_layers=8,
        n_heads=8,
        ff_dim=2048,
        max_seq_len=512,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, tok, device


def generate_response(
    model: Milton,
    tok: Tokenizer,
    device: torch.device,
    user_input: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
) -> str:
    prompt = f"<|user|>{user_input}<|milton|>"
    prompt_ids = tok.encode(prompt)
    prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    output = model.generate(
        prompt_t,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    generated_ids = output[0].tolist()[len(prompt_ids):]
    if tok.eos_id in generated_ids:
        generated_ids = generated_ids[: generated_ids.index(tok.eos_id)]

    return tok.decode(generated_ids).strip()


def format_response(text: str, width: int) -> Text:
    """Format Milton's response as wrapped verse with line breaks preserved."""
    styled = Text()
    lines = text.split("\n")
    inner_width = width - 6

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            styled.append("\n")
            continue

        wrapped = textwrap.fill(line, width=inner_width)
        styled.append(wrapped, style=IVORY)
        if i < len(lines) - 1:
            styled.append("\n")

    return styled


LOGO = r"""
  __  __ _ _ _
 |  \/  (_) | |
 | \  / |_| | |_ ___  _ __
 | |\/| | | | __/ _ \| '_ \
 | |  | | | | || (_) | | | |
 |_|  |_|_|_|\__\___/|_| |_|
"""


def print_banner():
    console.print()

    for line in LOGO.strip("\n").split("\n"):
        t = Text(line, style=f"bold {GOLD}")
        console.print(t)
    console.print()

    subtitle = Text()
    subtitle.append("an LLM trained on ", style=DIM)
    subtitle.append("Paradise Lost", style=f"italic {FADED}")
    subtitle.append(" and nothing else", style=DIM)
    console.print(subtitle)
    console.print()

    console.print(Rule(style=DIM))
    console.print()

    help_text = Text()
    help_text.append("/temp ", style=FADED)
    help_text.append("0.5", style=f"dim {IVORY}")
    help_text.append("  adjust temperature    ", style=DIM)
    help_text.append("/top_p ", style=FADED)
    help_text.append("0.9", style=f"dim {IVORY}")
    help_text.append("  nucleus sampling    ", style=DIM)
    help_text.append("quit", style=FADED)
    help_text.append("  exit", style=DIM)
    console.print(help_text)

    console.print()
    console.print(Rule(style=DIM))
    console.print()


THINKING_FRAMES = [
    "  milton is thinking",
    "  milton is thinking .",
    "  milton is thinking . .",
    "  milton is thinking . . .",
]


def main():
    parser = argparse.ArgumentParser(description="Talk to Milton")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    with console.status(f"[{DIM}]Loading Milton...[/{DIM}]", spinner="dots", spinner_style=GOLD):
        model, tok, device = load_milton(args.checkpoint)

    print_banner()

    temperature = args.temperature
    top_p = args.top_p
    term_width = shutil.get_terminal_size().columns
    panel_width = min(term_width - 2, 76)

    while True:
        try:
            console.print(f"[{FADED}]you[/{FADED}]")
            user_input = console.input(f"[{IVORY}]> [/{IVORY}]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            console.print(Rule(style=DIM))
            console.print()
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            console.print()
            console.print(Rule(style=DIM))
            console.print()
            break

        if user_input.startswith("/temp "):
            try:
                temperature = float(user_input.split()[1])
                console.print(f"  [{DIM}]temperature = {temperature}[/{DIM}]")
            except (IndexError, ValueError):
                console.print(f"  [{DIM}]usage: /temp 0.8[/{DIM}]")
            continue
        if user_input.startswith("/top_p "):
            try:
                top_p = float(user_input.split()[1])
                console.print(f"  [{DIM}]top_p = {top_p}[/{DIM}]")
            except (IndexError, ValueError):
                console.print(f"  [{DIM}]usage: /top_p 0.9[/{DIM}]")
            continue

        console.print()

        # Generate with a subtle loading state
        result = [None]
        def _generate():
            result[0] = generate_response(
                model, tok, device, user_input,
                max_tokens=args.max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=args.top_k,
            )
        gen_thread = threading.Thread(target=_generate)
        gen_thread.start()

        frame_idx = 0
        with Live(Text(THINKING_FRAMES[0], style=DIM), console=console, refresh_per_second=4, transient=True) as live:
            while gen_thread.is_alive():
                gen_thread.join(timeout=0.3)
                frame_idx += 1
                live.update(Text(THINKING_FRAMES[frame_idx % len(THINKING_FRAMES)], style=DIM))

        response = result[0]
        if response:
            styled_response = format_response(response, panel_width)
            panel = Panel(
                styled_response,
                title=f"[bold {GOLD}]milton[/bold {GOLD}]",
                title_align="left",
                border_style=DIM,
                width=panel_width,
                padding=(1, 2),
            )
            console.print(panel)
        else:
            console.print(f"  [{DIM}](silence)[/{DIM}]")

        console.print()


if __name__ == "__main__":
    main()
