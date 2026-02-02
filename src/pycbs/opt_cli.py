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


# Helper to defensively convert array-like coords to nested lists of floats
def _coords_to_list(obj) -> list:
    if obj is None:
        return []
    try:
        if not hasattr(obj, "__len__") or len(obj) == 0:
            return []
    except Exception:
        return []
    out = []
    try:
        for row in obj:
            if row is None:
                continue
            try:
                coords = [float(x) for x in row]
                if len(coords) >= 3:
                    out.append([coords[0], coords[1], coords[2]])
                else:
                    continue
            except Exception:
                continue
    except Exception:
        return []
    return out


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

    # Call optimizer (signature expected: optimize_geometry(symbols, coords0, basis_pair=basis_pair, options=options))
    res = opt_mod.optimize_geometry(
        symbols,
        coords0,
        basis_pair=basis_pair,
        options=options,
    )

    # -----------------------------------------------------------------
    # Postprocessing: XYZ + optimization history (CANONICAL)
    # -----------------------------------------------------------------
    # Use package writer for formatted output
    from . import writer as _writer

    # ---- XYZ ----
    opt_xyz = unique_path(outdir / params.get("output_xyz", "optimized.xyz"))

    coords = []
    if isinstance(res, dict):
        # prefer final_cart, then coords
        fc = res.get("final_cart", None)
        cc = res.get("coords", None)

        if fc is not None and (hasattr(fc, "__len__") and len(fc) > 0):
            coords = _coords_to_list(fc)
        elif cc is not None and (hasattr(cc, "__len__") and len(cc) > 0):
            coords = _coords_to_list(cc)

    # If coords is still empty, do not try to write internals as cartesian
    if not coords:
        logger.warning("No cartesian coordinates found in optimizer result (final_cart/coords). Writing empty geometry file placeholder.")
        coords = []

    # Write optimized geometry using module writer if present, with fallback
    try:
        if hasattr(opt_mod, "write_xyz"):
            try:
                opt_mod.write_xyz(str(opt_xyz), symbols, coords, comment="Optimized by pycbs-opt")
            except TypeError:
                opt_mod.write_xyz(str(opt_xyz), symbols, coords)
        else:
            with opt_xyz.open("w", encoding="utf-8") as fh:
                fh.write(f"{len(symbols)}\n")
                fh.write("Optimized by pycbs-opt\n")
                for s, c in zip(symbols, coords):
                    fh.write(f"{s} {c[0]} {c[1]} {c[2]}\n")
    except Exception:
        logger.exception("Failed to write optimized XYZ using optimization module; attempting simple fallback writer.")
        try:
            with opt_xyz.open("w", encoding="utf-8") as fh:
                fh.write(f"{len(symbols)}\n")
                fh.write("Optimized by pycbs-opt\n")
                for s, c in zip(symbols, coords):
                    fh.write(f"{s} {c[0]} {c[1]} {c[2]}\n")
        except Exception:
            logger.exception("Final fallback failed to write optimized.xyz. File may be missing or empty.")

    # ---- Optimization history ----
    opt_cycles = []
    hist = None
    if isinstance(res, dict):
        hist = res.get("history") or res.get("opt_history") or res.get("cycles") or res.get("cycle_history")

    if isinstance(hist, list) and hist:
        for i, h in enumerate(hist, start=1):
            if isinstance(h, dict):
                e = h.get("cbs_energy") or h.get("energy") or h.get("total_energy") or h.get("E_total")
                opt_cycles.append({"cycle": h.get("cycle", i), "cbs_energy": e})
            elif isinstance(h, (list, tuple)) and len(h) >= 2:
                opt_cycles.append({"cycle": h[0], "cbs_energy": h[1]})
            else:
                opt_cycles.append({"cycle": i, "cbs_energy": h})
    else:
        final_energy = None
        if isinstance(res, dict):
            final_energy = res.get("final_energy")
            if final_energy is None:
                hf = res.get("final_hf_cbs")
                corr = res.get("final_corr_cbs")
                if hf is not None and corr is not None:
                    try:
                        final_energy = float(hf) + float(corr)
                    except Exception:
                        final_energy = None
            if final_energy is None and isinstance(res.get("opt_result"), dict):
                final_energy = res["opt_result"].get("fun")
        opt_cycles = [{"cycle": 0, "cbs_energy": final_energy}]

    history_path = unique_path(outdir / params.get("summary_file", "opt_history.txt"))
    try:
        _writer.write_reports1(str(history_path), calculations=[], opt_cycles=opt_cycles)
    except Exception:
        logger.exception("writer.write_reports1 failed; writing compact history fallback.")
        try:
            with history_path.open("w", encoding="utf-8") as fh:
                fh.write("pycbs-opt history / summary\n")
                fh.write(f"config: {args.config}\n")
                fh.write(f"xyz: {xyz_file}\n")
                fh.write(f"basis_pair: {basis_pair}\n")
                fh.write(f"method: {options.get('method')}\n\n")
                for row in opt_cycles:
                    fh.write(f"{row.get('cycle')}\t{row.get('cbs_energy')}\n")
        except Exception:
            logger.exception("Fallback history write failed.")

    logger.info("Optimization finished. Results written to: %s", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
