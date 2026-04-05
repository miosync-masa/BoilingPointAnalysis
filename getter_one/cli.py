"""
GETTER One CLI - Command Line Interface
=========================================

All CLI functionality: banners, argument parsing, command dispatch.
Invoked via:
    $ getter-one [command] [options]
    $ python -m getter_one [command] [options]
"""

from __future__ import annotations

import argparse
import os
import random
import sys

from getter_one import GPU_AVAILABLE, GPU_MEMORY, GPU_NAME, __version__, get_gpu_info

# ===============================
# Banner System
# ===============================


def _should_show_banner() -> bool:
    if os.environ.get("GETTER_ONE_NO_BANNER"):
        return False
    if not sys.stdout.isatty():
        return False
    return "--no-banner" not in sys.argv


def _gpu_status_line() -> str:
    if GPU_AVAILABLE:
        return f"GPU: {GPU_NAME} ({GPU_MEMORY:.1f} GB)"
    return "CPU Mode (install CuPy for GPU acceleration)"


def print_banner(style: str | None = None):
    if not _should_show_banner():
        return

    style = style or os.environ.get("GETTER_ONE_BANNER_STYLE", "random").lower()

    banners = {
        "simple": _banner_simple,
        "ascii": _banner_ascii,
        "getter": _banner_getter,
        "tamaki": _banner_tamaki,
    }

    if style == "random":
        random.choice(list(banners.values()))()
    elif style in banners:
        banners[style]()
    else:
        _banner_simple()


def _banner_simple():
    print()
    print("=" * 60)
    print(f"  GETTER One v{__version__}")
    print("  Geometric Event-driven Tensor-based")
    print("  Time-series Extraction & Recognition")
    print("=" * 60)
    print(f"  {_gpu_status_line()}")
    print("=" * 60)
    print()


def _banner_ascii():
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║    ██████  ███████ ████████ ████████ ███████ ██████       ║
║   ██       ██         ██       ██    ██      ██   ██      ║
║   ██   ███ █████      ██       ██    █████   ██████       ║
║   ██    ██ ██         ██       ██    ██      ██   ██      ║
║    ██████  ███████    ██       ██    ███████ ██   ██      ║
║                                           ONE             ║
║                                                           ║
║   Geometric Event-driven Tensor-based                     ║
║   Time-series Extraction & Recognition                    ║
║   v{__version__:<53s}║
╠═══════════════════════════════════════════════════════════╣
║   {_gpu_status_line():<55s}║
╚═══════════════════════════════════════════════════════════╝
""")


def _banner_getter():
    """ゲッター風バナー"""
    forms = [
        ("EAGLE", "Aerial Type  // Structure Detection"),
        ("JAGUAR", "Ground Type  // Causal Network"),
        ("BEAR", "Marine Type  // Confidence Engine"),
    ]
    form = random.choice(forms)
    print(f"""
╔══════════════════════════════════════════════════════════╗
║   ██████  ███████ ████████ ████████ ███████ ██████      ║
║   GETTER ONE  //  CHANGE {form[0]:>6s}                       ║
║   {form[1]:<53s}║
╠══════════════════════════════════════════════════════════╣
║   Structural event detection for N-dimensional          ║
║   time-series with causal network extraction            ║
║   v{__version__:<52s}║
╠══════════════════════════════════════════════════════════╣
║   {_gpu_status_line():<55s}║
║   THREE PRINCIPLES UNITED:                              ║
║     [DISCRETE] × [GEOMETRIC] × [CORRELATIONAL]         ║
╚══════════════════════════════════════════════════════════╝
""")


def _banner_tamaki():
    """環ちゃんバナー"""
    faces = ["(◕‿◕)", "(｡♥‿♥｡)", "(✧ω✧)", "(*´▽`*)"]
    messages = [
        "GETTER One is online! Let's go!",
        "Structural changes? I'll find them!",
        "Bring me the data, Master!",
        "Causal network analysis, here I come!",
        "CHAAANGE!! GETTER ONE!!",
    ]
    face = random.choice(faces)
    msg = random.choice(messages)

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║   GETTER One v{__version__:<44s}║
║   Geometric Event-driven Tensor-based                     ║
║   Time-series Extraction & Recognition                    ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║   {face} < {msg:<43s}║
║                                                           ║
║   {_gpu_status_line():<55s}║
║   Built with 💕 by Masamichi & Tamaki                     ║
╚═══════════════════════════════════════════════════════════╝
""")


# ===============================
# CLI Commands
# ===============================


def cmd_run(args):
    """パイプライン実行 (V1)"""
    from getter_one.pipeline import PipelineConfig, run

    print("\n🚀 Starting GETTER One analysis...")
    print(f"   Source:   {args.source}")
    if args.target:
        print(f"   Target:   {args.target}")
    print(f"   Window:   {args.window}")
    print(f"   Max lag:  {args.max_lag}")
    if GPU_AVAILABLE:
        print(f"   GPU:      {GPU_NAME}")
    else:
        print("   Mode:     CPU")
    print()

    config = PipelineConfig(
        window_steps=args.window,
        max_lag=args.max_lag,
        n_permutations=args.n_perm,
        report_path=args.report,
        enable_confidence=not args.no_confidence,
        enable_network=not args.no_network,
    )

    try:
        result = run(
            args.source,
            config=config,
            target=args.target,
            time_column=args.time,
            normalize=args.normalize,
        )

        if result.report:
            print(result.report)

        if result.computation_time > 0:
            print(f"\n✅ Complete! ({result.computation_time:.2f}s)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def cmd_run_v2(args):
    """パイプライン実行 (V2 DualCore)"""
    from getter_one.pipeline_v2 import PipelineV2Config, run_v2

    print("\n🚀 Starting GETTER One V2 (DualCore) analysis...")
    print(f"   Source:   {args.source}")
    if args.target:
        print(f"   Target:   {args.target}")
    print(f"   Window:   {args.window or 'adaptive'}")
    print(f"   Max lag:  {args.max_lag}")
    print(
        f"   Network:  local_std={'ON' if args.local_std else 'OFF'}, "
        f"event_mask={'ON' if args.event_mask else 'OFF'}"
    )
    if GPU_AVAILABLE:
        print(f"   GPU:      {GPU_NAME}")
    else:
        print("   Mode:     CPU")
    print()

    config = PipelineV2Config(
        window_steps=args.window,
        adaptive_window=args.window is None,
        max_lag=args.max_lag,
        use_local_std_for_network=args.local_std,
        use_event_mask_for_network=args.event_mask,
        enable_cascade=not args.no_cascade,
        enable_inverse_checker=not args.no_inverse,
        enable_confidence=not args.no_confidence,
        enable_network=not args.no_network,
        n_permutations=args.n_perm,
        report_path=args.report,
    )

    try:
        result = run_v2(
            args.source,
            config=config,
            target=args.target,
            time_column=args.time,
            normalize=args.normalize,
        )

        if result.report:
            print(result.report)

        if result.computation_time > 0:
            print(f"\n✅ Complete! ({result.computation_time:.2f}s)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def cmd_load(args):
    """データ準備"""
    from getter_one.data.loader import main as loader_main

    # loader.pyのCLIに委譲
    sys.argv = ["getter-one-loader", args.loader_command] + args.loader_args
    loader_main()


def cmd_info(args):
    """システム情報"""
    info = get_gpu_info()

    print("\n📊 GETTER One System Information")
    print(f"   Version:     {__version__}")
    print(f"   GPU:         {'Available' if info['available'] else 'Not available'}")
    if info["available"]:
        print(f"   Device:      {info['name']}")
        print(f"   Memory:      {info['memory_gb']:.1f} GB")
        print(f"   CUDA:        {info['cuda_version']}")
        print(f"   Compute:     {info['compute_capability']}")
    print(f"   CuPy:        {'Installed' if info['has_cupy'] else 'Not installed'}")
    print(f"   Python:      {sys.version.split()[0]}")
    print()


def cmd_check_gpu(args):
    """GPU動作確認"""
    if not GPU_AVAILABLE:
        print("❌ GPU not available")
        print("   Install CuPy for GPU acceleration")
        sys.exit(1)

    print(f"✅ GPU OK: {GPU_NAME} ({GPU_MEMORY:.1f} GB)")

    try:
        import cupy as cp

        x = cp.ones(1000)
        assert float(cp.sum(x)) == 1000.0
        print("✅ CuPy computation OK")
    except Exception as e:
        print(f"❌ CuPy test failed: {e}")
        sys.exit(1)


# ===============================
# Argument Parser
# ===============================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="getter-one",
        description=(
            "GETTER One: Geometric Event-driven Tensor-based "
            "Time-series Extraction & Recognition\n"
            "Structural event detection and causal network extraction "
            "for N-dimensional time series."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  getter-one run weather.csv --target precipitation --window 24
  getter-one run-v2 financial.csv --local-std --event-mask
  getter-one run-v2 market.csv --no-local-std --no-event-mask
  getter-one info
  getter-one check-gpu

Data preparation:
  getter-one-loader load weather.csv -o prepared.csv
  getter-one-loader merge a.csv b.json --time date -o merged.csv

GitHub: https://github.com/miosync-masa/getter-one
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"GETTER One v{__version__}",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress startup banner",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- run (V1) ---
    p_run = sub.add_parser(
        "run",
        help="Run GETTER One analysis (V1 pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  getter-one run weather.csv --target precipitation
  getter-one run sensor_data.json --window 48 --max-lag 24
  getter-one run experiment.parquet --report results.md --no-confidence
        """,
    )
    p_run.add_argument("source", help="Input file (csv/json/parquet/xlsx/npy)")
    p_run.add_argument("--target", help="Target column name")
    p_run.add_argument("--time", help="Time column name")
    p_run.add_argument(
        "--normalize", default="range", choices=["range", "zscore", "none"]
    )
    p_run.add_argument(
        "--window", type=int, default=24, help="Window steps for Λ³ (default: 24)"
    )
    p_run.add_argument(
        "--max-lag", type=int, default=12, help="Max lag for causal (default: 12)"
    )
    p_run.add_argument(
        "--n-perm", type=int, default=1000, help="Permutations for confidence"
    )
    p_run.add_argument("--report", help="Output report path (.md)")
    p_run.add_argument("--no-confidence", action="store_true")
    p_run.add_argument("--no-network", action="store_true")
    p_run.add_argument("--verbose", "-v", action="store_true")
    p_run.set_defaults(func=cmd_run)

    # --- run-v2 (DualCore) ---
    p_v2 = sub.add_parser(
        "run-v2",
        help="Run GETTER One V2 (DualCore: Local+Global, Cascade, InverseChecker)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
V2 DualCore Architecture:
  Lambda → Detection (event_mask) → Network → Cascade → InverseChecker

Network mode flags:
  --local-std / --no-local-std    Dimensionless displacement (default: ON)
  --event-mask / --no-event-mask  Event-filtered correlation (default: ON)

Examples:
  getter-one run-v2 financial.csv                          # Full V2
  getter-one run-v2 market.csv --no-local-std --no-event-mask  # V1-equivalent network
  getter-one run-v2 data.csv --no-cascade --no-inverse     # DualCore Lambda only
        """,
    )
    p_v2.add_argument("source", help="Input file (csv/json/parquet/xlsx/npy)")
    p_v2.add_argument("--target", help="Target column name")
    p_v2.add_argument("--time", help="Time column name")
    p_v2.add_argument(
        "--normalize", default="range", choices=["range", "zscore", "none"]
    )
    p_v2.add_argument(
        "--window",
        type=int,
        default=None,
        help="Window steps (default: adaptive)",
    )
    p_v2.add_argument(
        "--max-lag", type=int, default=12, help="Max lag for causal (default: 12)"
    )
    p_v2.add_argument(
        "--n-perm", type=int, default=1000, help="Permutations for confidence"
    )
    p_v2.add_argument("--report", help="Output report path (.md)")

    # Network mode flags
    p_v2.add_argument(
        "--local-std",
        action="store_true",
        default=True,
        dest="local_std",
        help="Enable dimensionless displacement (default)",
    )
    p_v2.add_argument(
        "--no-local-std",
        action="store_false",
        dest="local_std",
        help="Disable dimensionless displacement",
    )
    p_v2.add_argument(
        "--event-mask",
        action="store_true",
        default=True,
        dest="event_mask",
        help="Enable event-filtered correlation (default)",
    )
    p_v2.add_argument(
        "--no-event-mask",
        action="store_false",
        dest="event_mask",
        help="Disable event-filtered correlation",
    )

    # Feature toggles
    p_v2.add_argument("--no-cascade", action="store_true")
    p_v2.add_argument("--no-inverse", action="store_true")
    p_v2.add_argument("--no-confidence", action="store_true")
    p_v2.add_argument("--no-network", action="store_true")
    p_v2.add_argument("--verbose", "-v", action="store_true")
    p_v2.set_defaults(func=cmd_run_v2)

    # --- info ---
    p_info = sub.add_parser("info", help="Show system information")
    p_info.set_defaults(func=cmd_info)

    # --- check-gpu ---
    p_gpu = sub.add_parser("check-gpu", help="Verify GPU availability")
    p_gpu.set_defaults(func=cmd_check_gpu)

    return parser


# ===============================
# Main Entry Point
# ===============================


def main():
    print_banner()

    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\n💡 Quick start:")
        print("  getter-one run your_data.csv --target y")
        print("  getter-one run-v2 your_data.csv  # DualCore pipeline")
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
