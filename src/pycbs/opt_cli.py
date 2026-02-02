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
    return {
        "method": params.get("method", "ccsd(t)"),
        "spin": int(params.get("spin", 0)),
        "X1": int(params.get("x1", 2)),
        "X2": int(params.get("x2", 3)),
        "Xhf1": int(params.get("xhf1", 2)),
        "Xhf2": int(params.get("xhf2", 3)),
    }


def find_optimization_module() -> Tuple[str, object]:
    this_file = Path(__file__).resolve()
    pkg_dir = this_file.parent

    candidates = [
        pkg_dir.parent / "pycbs-opt" / "optimization.py",
        pkg_dir / "pycbs-opt" / "optimization.py",
    ]

    for p in sys.path:
        candidates.append(Path(p) / "pycbs-opt" / "optimization.py")

    for c in candidates:
        try:
            if c.exists():
                spec = importlib.util.spec_from_file_location("pycbs_opt_module", str(c))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return str(c), mod
        except Exception:
            continue

    raise FileNotFoundError("Could not locate pycbs-opt/optimization.py")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="pycbs-opt",
        description="Run pyCBS geometry optimization from an INI file",
    )
    parser.add_argument("config", type=Path, help="Optimization input INI file")
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        params = load_params_from_ini(args.config)
    except Exception as e:
        logger.error("Failed to read config: %s", e)
        return 2

    if "xyz_file" not in params and "xyz" not in params:
        logger.error("Missing 'xyz_file' entry in INI")
        return 2

    xyz_file = Path(params.get("xyz_file", params.get("xyz"))).expanduser()
    if not xyz_file.exists():
        logger.error("XYZ file does not exist: %s", xyz_file)
        return 2

    basis1 = map_basis(params.get("basis1", "vdz"))
    basis2 = map_basis(params.get("basis2", "vtz"))
    basis_pair = (basis1, basis2)

    outdir = Path(params.get("output_dir", "PyCBS-OUTPUTS"))
    ensure_output_dir(outdir)

    try:
        _, opt_mod = find_optimization_module()
    except Exception as e:
        logger.error("Cannot locate optimization module: %s", e)
        traceback.print_exc()
        return 3

    if not hasattr(opt_mod, "read_xyz") or not hasattr(opt_mod, "optimize_geometry"):
        logger.error("optimization.py missing required API")
        return 4

    symbols, coords0 = opt_mod.read_xyz(str(xyz_file))
    options = prepare_options_from_params(params)

    logger.info(
        "Starting geometry optimization (basis pair: %s, %s)",
        basis1,
        basis2,
    )

    res = opt_mod.optimize_geometry(
        symbols,
        coords0,
        basis_pair=basis_pair,
        options=options,
    )

    # -----------------------------------------------------------------
    # Postprocessing: XYZ + optimization history (CANONICAL)
    # -----------------------------------------------------------------

    from . import writer as _writer

    # ---- XYZ ----
    opt_xyz = unique_path(outdir / params.get("output_xyz", "optimized.xyz"))

    coords = []
    if isinstance(res, dict):
        if res.get("final_cart"):
            coords = res["final_cart"]
        elif res.get("coords"):
            coords = res["coords"]

    if hasattr(opt_mod, "write_xyz"):
        opt_mod.write_xyz(
            str(opt_xyz),
            symbols,
            coords,
            comment="Optimized by pycbs-opt",
        )
    else:
        with opt_xyz.open("w", encoding="utf-8") as fh:
            fh.write(f"{len(symbols)}\n")
            fh.write("Optimized by pycbs-opt\n")
            for s, c in zip(symbols, coords):
                fh.write(f"{s} {c[0]} {c[1]} {c[2]}\n")

    # ---- Optimization history ----
    opt_cycles = []
    hist = res.get("history") or res.get("opt_history") or res.get("cycles")

    if isinstance(hist, list) and hist:
        for i, h in enumerate(hist, start=1):
            if isinstance(h, dict):
                e = (
                    h.get("cbs_energy")
                    or h.get("energy")
                    or h.get("total_energy")
                )
                opt_cycles.append({"cycle": h.get("cycle", i), "cbs_energy": e})
            else:
                opt_cycles.append({"cycle": i, "cbs_energy": h})
    else:
        final_energy = (
            res.get("final_energy")
            or (
                res.get("final_hf_cbs")
                and res.get("final_corr_cbs")
                and res["final_hf_cbs"] + res["final_corr_cbs"]
            )
            or res.get("opt_result", {}).get("fun")
        )
        opt_cycles = [{"cycle": 0, "cbs_energy": final_energy}]

    history_path = unique_path(outdir / params.get("summary_file", "opt_history.txt"))

    _writer.write_reports(
        str(history_path),
        calculations=[],
        opt_cycles=opt_cycles,
    )

    logger.info("Optimization finished. Results written to: %s", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
