#!/usr/bin/env python3
"""
Stress test attention backends (FlashAttention-2 vs SDPA vs eager) in the same way GRPO loads models.

This is useful when GPU faults (e.g. NVRM Xid 13) happen only rarely during training.
If FlashAttention is the trigger, running this for long enough often reproduces the crash faster.

Example (run on GPU0 only):
  CUDA_VISIBLE_DEVICES=0 uv run --extra compile python -u stress_attn.py \
    --model_name_or_path /path/to/your/model \
    --attn_implementation flash_attention_2 \
    --seq_len 8192 --batch_size 1 --steps 2000

Try SDPA as a "safe" A/B test:
  CUDA_VISIBLE_DEVICES=0 uv run --extra compile python -u stress_attn.py \
    --model_name_or_path /path/to/your/model \
    --attn_implementation sdpa \
    --seq_len 8192 --batch_size 1 --steps 2000
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Literal, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


AttnImpl = Literal["flash_attention_2", "sdpa", "eager"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--model_revision", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--seq_len", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument(
        "--use_backbone_only",
        action="store_true",
        help="If set, run only the transformer backbone (no LM head / logits). This is much more memory-friendly and still stresses attention kernels.",
    )
    p.add_argument("--backward", action="store_true", help="If set, also run backward() to stress backward kernels (uses more memory).")
    p.add_argument("--optimizer", type=str, default="none", choices=["adamw", "none"])
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--use_cache", action="store_true")
    p.add_argument("--vary_seq_len", action="store_true", help="Randomize sequence length per step up to --seq_len")
    p.add_argument("--min_seq_len", type=int, default=256, help="Only used with --vary_seq_len")
    return p.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


@torch.no_grad()
def _smoke_tokenizer(model_name_or_path: str, revision: Optional[str]) -> int:
    # We use the tokenizer only to get vocab size and pad/eos ids reasonably.
    tok = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision, use_fast=True)
    if tok.pad_token_id is None:
        # Many causal LMs have no pad token; for synthetic random inputs this is fine.
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    return int(tok.vocab_size)


def main() -> None:
    args = parse_args()
    attn_impl: AttnImpl = args.attn_implementation  # type: ignore[assignment]

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This stress test needs an NVIDIA GPU.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    dtype = resolve_dtype(args.dtype)

    # Helpful for catching async kernel errors closer to the source.
    # (This slows things down, so we default it off.)
    cuda_launch_blocking = os.environ.get("CUDA_LAUNCH_BLOCKING")
    if cuda_launch_blocking:
        print(f"CUDA_LAUNCH_BLOCKING={cuda_launch_blocking} (enabled)")

    print(
        f"Loading model: {args.model_name_or_path} (rev={args.model_revision}) "
        f"attn={attn_impl} dtype={dtype} device={device}"
    )
    vocab_size = _smoke_tokenizer(args.model_name_or_path, args.model_revision)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        use_cache=args.use_cache,
    ).to(device)
    model.train()

    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        # Transformers sets use_cache=False when gradient checkpointing is enabled in many configs,
        # but we keep args.use_cache as the explicit user choice above.

    opt = None
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Main loop: do forward+backward and synchronize regularly.
    max_len = int(args.seq_len)
    bsz = int(args.batch_size)

    # We use random token IDs and a simple LM loss (shifted logits) to exercise attention + matmul paths.
    def sample_batch(seq_len: int) -> torch.Tensor:
        return torch.randint(0, vocab_size, (bsz, seq_len), device=device, dtype=torch.long)

    print(
        f"Starting stress loop: steps={args.steps} warmup={args.warmup} "
        f"batch_size={bsz} max_seq_len={max_len} vary_seq_len={bool(args.vary_seq_len)}"
    )

    def forward_loss(input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute a simple scalar loss that exercises the model.

        IMPORTANT:
        - If we're not doing backward, we run under `torch.no_grad()` so we don't
          store activations for autograd (huge memory savings).
        """

        def _inner() -> torch.Tensor:
            if args.use_backbone_only:
                backbone = getattr(model, model.base_model_prefix)
                out = backbone(input_ids=input_ids, use_cache=args.use_cache, return_dict=True)
                return out.last_hidden_state.mean()
            out = model(input_ids=input_ids, use_cache=args.use_cache, return_dict=True)
            return out.logits.mean()

        if args.backward:
            return _inner()
        with torch.no_grad():
            return _inner()

    # Warmup
    for _ in range(args.warmup):
        ids = sample_batch(max_len if not args.vary_seq_len else max(args.min_seq_len, max_len // 2))
        loss = forward_loss(ids)

        if args.backward:
            if opt is not None:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            if opt is not None:
                opt.step()
        torch.cuda.synchronize()

    start = time.time()
    last_t = start

    for step in range(1, args.steps + 1):
        if args.vary_seq_len:
            seq_len = int(torch.randint(low=max(2, args.min_seq_len), high=max_len + 1, size=(1,)).item())
        else:
            seq_len = max_len

        ids = sample_batch(seq_len)

        loss = forward_loss(ids)

        if args.backward:
            if opt is not None:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            if opt is not None:
                opt.step()

        # Sync every step so kernel faults surface near the right step.
        torch.cuda.synchronize()

        if step % args.log_every == 0:
            now = time.time()
            dt = now - last_t
            total = now - start
            last_t = now
            mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(
                f"step={step:6d}/{args.steps} seq_len={seq_len:5d} loss={loss.item():.4f} "
                f"dt={dt:.2f}s total={total/60:.1f}m max_mem={mem_gb:.2f}GB"
            )

    print("✅ Completed without Python-side CUDA errors. (If you were chasing Xid 13, also check `dmesg`.)")


if __name__ == "__main__":
    main()


