#!/usr/bin/env python3
from __future__ import annotations
import argparse
import configparser
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import importlib.util
import traceback

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


def map_basis(name: str) -> str:
    if not name:
        return None
    name = name.strip()
    if not name:
        return None
    low = name.lower()
    if "cc-" in low or "aug" in low or "-" in name:
        return name
    mapped = BASIS_MAP.get(name) or BASIS_MAP.get(name.lower())
    return mapped if mapped else name


def find_optimization_module() -> Tuple[str, object]:
    this_file = Path(__file__).resolve()
    pkg_dir = this_file.parent
    candidate = pkg_dir.parent / "pycbs-opt" / "optimization.py"
    if candidate.exists():
        spec = importlib.util.spec_from_file_location("pycbs_opt_module", str(candidate))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return str(candidate), mod
    candidate2 = pkg_dir / "pycbs-opt" / "optimization.py"
    if candidate2.exists():
        spec = importlib.util.spec_from_file_location("pycbs_opt_module", str(candidate2))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return str(candidate2), mod
    for p in map(Path, sys.path):
        try:
            c = p / "pycbs-opt" / "optimization.py"
            if c.exists():
                spec = importlib.util.spec_from_file_location("pycbs_opt_module", str(c))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return str(c), mod
        except Exception:
            continue
    raise FileNotFoundError("Could not locate pycbs-opt/optimization.py on sys.path or relative package path.")


def load_params_from_ini(ini_path: Path) -> Dict[str, Any]:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    read_ok = cfg.read(str(ini_path))
    if not read_ok:
        raise FileNotFoundError(f"Cannot read config file: {ini_path}")
    section = None
    if "OPTIMIZATION" in cfg:
        section = "OPTIMIZATION"
    elif cfg.sections():
        section = cfg.sections()[0]
    else:
        section = configparser.DEFAULTSECT
    params_raw = dict(cfg[section]) if section != configparser.DEFAULTSECT else dict(cfg.defaults())
    params = {k.lower(): v for k, v in params_raw.items()}
    return params


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
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def prepare_options_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    method = params.get("method", "ccsd(t)")
    spin = int(params.get("spin", 0))
    X1 = int(params.get("x1", 2))
    X2 = int(params.get("x2", 3))
    Xhf1 = int(params.get("xhf1", 2))
    Xhf2 = int(params.get("xhf2", 3))
    return {
        "method": method,
        "spin": spin,
        "X1": X1,
        "X2": X2,
        "Xhf1": Xhf1,
        "Xhf2": Xhf2,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pycbs-opt", description="Run pyCBS geometry optimizer from a single INI file")
    parser.add_argument("config", type=Path, help="Optimization input INI file (first section used)")
    parser.add_argument("-v", "--verbose", action="count", default=1, help="verbosity (default: 1)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose >= 1 else logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        params = load_params_from_ini(args.config)
    except Exception as e:
        logger.error("Failed to read config: %s", e)
        sys.exit(2)

    if "xyz_file" not in params and "xyz" not in params:
        logger.error("Missing required 'xyz_file' entry in INI. Provide the XYZ path under key 'xyz_file'.")
        sys.exit(2)

    xyz_file = Path(params.get("xyz_file", params.get("xyz"))).expanduser()
    if not xyz_file.exists():
        logger.error("XYZ file does not exist: %s", xyz_file)
        sys.exit(2)

    basis1_name = params.get("basis1", "vdz")
    basis2_name = params.get("basis2", "vtz")
    basis1 = map_basis(basis1_name)
    basis2 = map_basis(basis2_name)

    outdir = Path(params.get("output_dir", "PyCBS-OUTPUTS"))
    ensure_output_dir(outdir)

    try:
        path_used, opt_mod = find_optimization_module()
    except Exception as e:
        logger.error("Cannot locate optimization.py: %s", e)
        traceback.print_exc()
        sys.exit(3)

    if not hasattr(opt_mod, "read_xyz") or not hasattr(opt_mod, "optimize_geometry"):
        logger.error("Loaded optimization module doesn't expose expected functions (read_xyz, optimize_geometry)")
        sys.exit(4)

    symbols, coords0 = opt_mod.read_xyz(str(xyz_file))

    options = prepare_options_from_params(params)
    if "method" in params:
        options["method"] = params["method"]

    logger.info("Starting optimization (basis pair: %s, %s) ...", basis1, basis2)
    basis_pair = (basis1, basis2)

    # run optimization (user's optimization function may have different signature;
    # we assume optimize_geometry(symbols, coords0, basis_pair=basis_pair, options=options))
    res = opt_mod.optimize_geometry(symbols, coords0, basis_pair=basis_pair, options=options)

    # Make unique output filenames in the shared PyCBS-OUTPUTS dir
    opt_xyz_candidate = outdir / (params.get("output_xyz", "optimized.xyz"))
    opt_xyz = unique_path(opt_xyz_candidate)
    # write optimized geometry (use module write_xyz if available)
    if hasattr(opt_mod, "write_xyz"):
        opt_mod.write_xyz(str(opt_xyz), symbols, res.get("coords", res.get("x", [])), comment="Optimized by pycbs-opt")
    else:
        # fallback simple writer
        with opt_xyz.open("w", encoding="utf-8") as fh:
            fh.write(f"{len(symbols)}\n")
            fh.write("Optimized by pycbs-opt\n")
            coords = res.get("coords", res.get("x", []))
            for sym, c in zip(symbols, coords):
                fh.write(f"{sym} {c[0]} {c[1]} {c[2]}\n")

    # write optimization history summary/table
    history_path_candidate = outdir / (params.get("summary_file", "opt_history.txt"))
    history_path = unique_path(history_path_candidate)

    try:
        with history_path.open("w", encoding="utf-8") as fh:
            fh.write("pycbs-opt history / summary\n")
            fh.write(f"config: {args.config}\n")
            fh.write(f"xyz: {xyz_file}\n")
            fh.write(f"basis_pair: {basis_pair}\n")
            fh.write(f"method: {options.get('method')}\n\n")

            # try to write a nice table if we have a history list in the result
            history = res.get("history") or res.get("opt_history") or res.get("cycles")
            if isinstance(history, list) and history:
                # assume list of dict-like rows
                keys = set()
                for row in history:
                    if isinstance(row, dict):
                        keys.update(row.keys())
                keys = sorted(keys)
                if keys:
                    fh.write("\t".join(keys) + "\n")
                    for row in history:
                        if isinstance(row, dict):
                            fh.write("\t".join(str(row.get(k, "")) for k in keys) + "\n")
                        else:
                            fh.write(str(row) + "\n")
                else:
                    # fallback: just dump each item
                    for item in history:
                        fh.write(str(item) + "\n")
            else:
                # fallback: print keys from result
                for k in sorted(res.keys()):
                    fh.write(f"{k}: {res[k]}\n")
    except Exception:
        logger.exception("Failed writing optimization history")

    logger.info("Optimization finished. Results written to: %s", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
