#!/usr/bin/env python3
"""Run python/sglang/bench_serving.py without importing heavy runtime modules.

This keeps the benchmark request-generation path intact for local HTTP payload
inspection, while stubbing the SRT modules that otherwise pull in torch/triton.
"""

import runpy
import sys
import types
from pathlib import Path


def install_stubs(repo_root: Path) -> None:
    python_dir = repo_root / "python"
    sglang_dir = python_dir / "sglang"

    sys.path.insert(0, str(python_dir))

    sglang_pkg = types.ModuleType("sglang")
    sglang_pkg.__path__ = [str(sglang_dir)]
    sys.modules["sglang"] = sglang_pkg

    srt_pkg = types.ModuleType("sglang.srt")
    srt_pkg.__path__ = [str(sglang_dir / "srt")]
    sys.modules["sglang.srt"] = srt_pkg

    disagg_pkg = types.ModuleType("sglang.srt.disaggregation")
    disagg_pkg.__path__ = [str(sglang_dir / "srt" / "disaggregation")]
    sys.modules["sglang.srt.disaggregation"] = disagg_pkg

    disagg_utils = types.ModuleType("sglang.srt.disaggregation.utils")
    disagg_utils.FAKE_BOOTSTRAP_HOST = "2.2.2.2"
    sys.modules["sglang.srt.disaggregation.utils"] = disagg_utils

    srt_utils_pkg = types.ModuleType("sglang.srt.utils")
    srt_utils_pkg.__path__ = [str(sglang_dir / "srt" / "utils")]
    sys.modules["sglang.srt.utils"] = srt_utils_pkg


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    install_stubs(repo_root)
    sys.argv[0] = str(repo_root / "python" / "sglang" / "bench_serving.py")
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
