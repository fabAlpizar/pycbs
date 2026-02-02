#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple
import importlib.util

logger = logging.getLogger(__name__)

BASIS_MAP = {
    "vdz": "cc-pvdz",
    "vtz": "cc-pvtz",
    "vqz": "cc-pvqz",
    "v5z": "cc-pv5z",
    "avdz": "aug-cc-pvdz",
    "avtz": "aug-cc-pvtz",
    "avqz": "aug-cc-pvqz",
    "VDZ": "cc-pvdz",
    "VTZ": "cc-pvtz",
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def map_basis(name: str) -> str | None:
    if not name:
        return None
    name = name.strip()
    if not name:
        return None
    low = name.lower()
    if "cc-" in low or "aug" in low or "-" in name:
        return name
    return BASIS_MAP.get(name) or BASIS_MAP.get(low) or name


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def load_params_from_ini(ini_path: Path) -> Dict[str, Any]:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    if not cfg.read(str(ini_path)):
        raise FileNotFoundError(f"Cannot read config file: {ini_path}")

    if "OPTIMIZATION" in cfg:
        section = "OPTIMIZATION"
    elif cfg.sections():
        section = cfg.sections()[0]
    else:
        section = cfg.default_section

    raw = dict(cfg[section])
    return {k.lower(): v for k, v in raw.items()}


def prepare_options_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize options coming from the INI file into a dict passed to the optimizer.
    Provides safe defaults so the caller can omit keys.
    """
    def _f(key, default):
        v = params.get(key)
        if v is None:
            return float(default)
        try:
            return float(v)
        except Exception:
            return float(default)

    method = params.get("method", "ccsd(t)")
    spin = int(params.get("spin", 0)) if params.get("spin") is not None else 0

    # optimizer selection (default 'lbfgs' or 'sqm')
    optimizer = params.get("optimizer", "lbfgs").strip().lower()

    return {
        "method": method,
        "spin": spin,
        "X1": _f("x1",  1.85),
        "X2": _f("x2",2.639),
        "Xhf1": _f("xhf1", 3.02),
        "Xhf2": _f("xhf2", 3.64),
        "beta": _f("beta", 1.62),
        "optimizer": optimizer,
        "input_xyz": params.get("input_xyz") or params.get("input") or params.get("geometry"),
        "output_dir": params.get("output_dir"),  # optional override
    }


def find_optimization_module(pkg_dir: Path, optimizer_name: str) -> Tuple[str, Path]:
    """
    Search for optimizer implementation files under the repository.
    Returns (module_name, path_to_file).
    Recognized names: 'lbfgs', 'sqm' (case-insensitive) mapped to candidate files.
    """
    # candidate filenames mapping (lowercase key)
    mapping = {
        "lbfgs": ["L-BFGS-B-based.py", "L-BFGS-B-based.py".lower(), "lbfgs-based.py", "lbfgs.py"],
        "sqm": ["SQM-based.py", "SQM-based.py".lower(), "sqm-based.py", "sqm.py"],
    }

    candidates = mapping.get(optimizer_name, [optimizer_name + ".py"])
    # first look in pkg_dir / 'pycbs-opt' (some files are in a separate dir), then same folder
    search_dirs = [pkg_dir / ".." / "pycbs-opt", pkg_dir / "pycbs-opt", pkg_dir]
    for s in search_dirs:
        s = s.resolve()
        if not s.exists():
            continue
        for cand in candidates:
            p = s / cand
            if p.exists():
                # produce a stable module name from path
                mod_name = f"pycbs_opt_{optimizer_name}"
                return mod_name, p
    # fallback: try direct file name under pkg_dir
    for s in search_dirs:
        s = s.resolve()
        if not s.exists():
            continue
        for f in s.iterdir():
            if f.is_file() and optimizer_name in f.name.lower():
                return f"pycbs_opt_{optimizer_name}", f
    raise FileNotFoundError(f"Could not find optimizer implementation for '{optimizer_name}' under {pkg_dir} or its pycbs-opt subfolder.")


def import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ensure_pycb_outputs(base_dir: Path | None) -> Path:
    """Create and return the PyCBS-OUTPUTS directory path"""
    base = Path(base_dir) if base_dir else Path.cwd()
    out = base / "PyCBS-OUTPUTS"
    ensure_output_dir(out)
    return out


def main():
    parser = argparse.ArgumentParser(description="pycbs optimizer CLI (select optimization pathway)")
    parser.add_argument("config", help="INI-style configuration file describing the job")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print("Config file not found:", cfg_path, file=sys.stderr)
        sys.exit(2)

    try:
        raw_params = load_params_from_ini(cfg_path)
        params = prepare_options_from_params(raw_params)

        pkg_dir = Path(__file__).resolve().parent
        optimizer_name = params.get("optimizer", "lbfgs")
        mod_name, mod_path = find_optimization_module(pkg_dir, optimizer_name)

        mod = import_module_from_path(mod_name, mod_path)

        # outputs directory: prefer config output_dir if provided, else project cwd
        base_out = Path(params["output_dir"]) if params.get("output_dir") else Path.cwd()
        outputs_dir = ensure_pycb_outputs(base_out)

        # Standardized entry point expected: run_optimization(params: dict, outputs_dir: Path) -> dict
        if not hasattr(mod, "run_optimization"):
            # attempt to find 'main' or 'optimize' functions as fallbacks
            if hasattr(mod, "main"):
                # wrap into run_optimization by calling main with environment variables
                def _fallback_run(params_in, out_dir):
                    # create a minimal args replacement if script expects CLI; not recommended.
                    return mod.main()
                run_func = _fallback_run
            elif hasattr(mod, "optimize") and callable(getattr(mod, "optimize")):
                run_func = getattr(mod, "optimize")
            else:
                raise AttributeError(f"Selected optimizer module {mod_path} does not expose run_optimization(params, outputs_dir). Please update the module to provide that API.")
        else:
            run_func = getattr(mod, "run_optimization")

        print(f"Using optimizer module: {mod_path}")
        print("Starting optimization... outputs will be written to:", outputs_dir)

        result = run_func(params, outputs_dir)

        # result is expected to be a dict with keys: 'history' (list), 'final_xyz' or 'final_cart' and 'final_energy'
        if isinstance(result, dict):
            # basic reporting
            hist = result.get("history")
            fe = result.get("final_energy")
            print("Optimization finished.")
            if hist is not None:
                print(f"Cycles performed: {len(hist)}")
            if fe is not None:
                print(f"Final CBS energy (Ha): {fe:.10f}")
        else:
            print("Optimizer returned no structured result (not a dict).")

    except Exception as exc:
        logger.error("Optimization failed: %s", exc)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()