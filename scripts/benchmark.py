#!/usr/bin/env python3
"""Cost per image pair: FLOPs, latency and peak GPU memory at batch size one.

    python scripts/benchmark.py                          # JLD and JLD-fast, 512 x 384 inputs
    python scripts/benchmark.py --height 768 --width 1024
    python scripts/benchmark.py --baselines psnr ssim lpips-vgg dists   # needs pyiqa
    python scripts/benchmark.py --device cpu --threads 12 --repeats 5 --warmup 2

Protocol of the paper's cost columns: fp32 inputs of 512 x 384 (JLD crops them to
504 x 378, the nearest lower multiples of 14), batch size one, one pair = one
reference and one distorted image. Time is the median wall time per pair over
``--repeats`` calls after ``--warmup`` calls and includes the chroma prefilter,
the CLS term and the copy of the result to the CPU. FLOPs are the tensor
operations seen by ``torch.utils.flop_counter.FlopCounterMode`` for both images;
attention is run unfused while counting, because the counter does not see inside
fused scaled-dot-product attention on every device.

The paper reports 118.5 GFLOPs and 12.2 ms for JLD and 10.7 GFLOPs and 2.5 ms
for JLD-fast on an NVIDIA A100. This script counts 118.5 GFLOPs for JLD and
10.8 for JLD-fast; the paper's 10.7 is the count without the chroma prefilter
(``--variants lens``). Times depend on the hardware.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from jld import JLD, VARIANTS

NAMES = {"full": "JLD", "fast": "JLD-fast", "lens": "JLD lens only", "raw": "raw block-1 tokens"}


def measure(call, modules, ref: torch.Tensor, dist: torch.Tensor, device: str, warmup: int, repeats: int) -> dict:
    """GFLOPs, median milliseconds and peak GPU megabytes of ``call(ref, dist)``.

    ``modules`` are the network modules behind ``call``; their attention is
    switched to the unfused path for the FLOP count only.
    """
    cuda = device.startswith("cuda")
    fused = [m for m in modules if getattr(m, "fused_attn", False)]
    try:
        for module in fused:
            module.fused_attn = False
        with FlopCounterMode(display=False) as counter, torch.no_grad():
            call(ref, dist)
        gflops = counter.get_total_flops() / 1e9
    except Exception:  # FLOP counting is optional
        gflops = float("nan")
    finally:
        for module in fused:
            module.fused_attn = True
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    times = []
    for index in range(warmup + repeats):
        if cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        call(ref, dist)
        if cuda:
            torch.cuda.synchronize()
        if index >= warmup:
            times.append((time.perf_counter() - started) * 1e3)
    peak = torch.cuda.max_memory_allocated() / 2**20 if cuda else float("nan")
    return {"gflops": gflops, "ms": statistics.median(times), "peak_mb": peak}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=["full", "fast"], choices=sorted(VARIANTS))
    parser.add_argument("--baselines", nargs="*", default=[], metavar="NAME",
                        help="pyiqa full-reference metric names to time under the same protocol")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads for PyTorch (default: its own choice)")
    args = parser.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(0)
    ref = torch.rand(1, 3, args.height, args.width, generator=generator)
    dist = (ref + 0.05 * torch.randn(ref.shape, generator=generator)).clamp(0, 1)
    ref, dist = ref.to(args.device), dist.to(args.device)
    cuda = args.device.startswith("cuda")
    name = torch.cuda.get_device_name() if cuda else f"CPU, {torch.get_num_threads()} threads"
    print(f"device: {name}; input {args.height}x{args.width}; batch 1; "
          f"median of {args.repeats} calls after {args.warmup} warm-up calls")
    print(f"{'metric':<20} {'params (M)':>10} {'GFLOPs/pair':>12} {'ms/pair':>9} {'peak GPU MB':>12}")

    def report(label: str, params: int, row: dict) -> None:
        print(f"{label:<20} {params / 1e6:10.2f} {row['gflops']:12.1f} {row['ms']:9.2f} {row['peak_mb']:12.1f}")

    for variant in args.variants:
        metric = JLD.pretrained(variant, device=args.device)
        params = sum(p.numel() for p in metric.encoder.parameters())
        modules = list(metric.encoder.modules())
        report(NAMES[variant], params, measure(metric, modules, ref, dist, args.device, args.warmup, args.repeats))
    for baseline in args.baselines:
        import pyiqa

        model = pyiqa.create_metric(baseline, device=torch.device(args.device))
        params = sum(p.numel() for p in model.parameters())

        def call(x1, x2, model=model):
            with torch.inference_mode():
                return model(x2, x1).float().cpu()  # pyiqa takes (distorted, reference)

        report(baseline, params, measure(call, list(model.modules()), ref, dist, args.device, args.warmup, args.repeats))


if __name__ == "__main__":
    main()
