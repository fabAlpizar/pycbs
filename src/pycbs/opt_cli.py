#!/usr/bin/env python3
"""
pycbs-opt CLI wrapper.

This wrapper accepts a single INI file (first section is used) with the keys:
  - xyz_file (required)
  - method (optional)         default: "ccsd(t)"
  - basis1, basis2 (optional) default: vdz, vtz
  - spin (optional)           default: 0
  - X1, X2, Xhf1, Xhf2 (optional) defaults: 2,3,2,3
  - output_dir (optional)     default: PyCBS-OUTPUTS (directory)
The wrapper maps short basis names (vdz, vtz, avdz, ...) to pyscf-style names (cc-pvdz, cc-pvtz, aug-cc-pvdz, ...).
It then loads the optimization.py module and calls optimize_geometry(), writing outputs
into the chosen output directory.
"""

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


# Simple mapping from short basis keys to pyscf-style basis names.
# Extend this mapping as needed.
BASIS_MAP = {
    "vdz": "cc-pvdz",
    "vtz": "cc-pvtz",
    "vqz": "cc-pvqz",
    "v5z": "cc-pv5z",
    "avdz": "aug-cc-pvdz",
    "avtz": "aug-cc-pvtz",
    "avqz": "aug-cc-pvqz",
    # allow uppercase variants
    "VDZ": "cc-pvdz",
    "VTZ": "cc-pvtz",
}


def map_basis(name: str) -> str:
    if not name:
        return None
    name = name.strip()
    if not name:
        return None
    # if user gives a name that already looks like a pyscf basis, use it as-is
    low = name.lower()
    if "cc-" in low or "aug" in low or "-" in name:
        return name
    # map common short forms (vdz -> cc-pvdz)
    mapped = BASIS_MAP.get(name) or BASIS_MAP.get(name.lower())
    return mapped if mapped else name


def find_optimization_module() -> Tuple[str, object]:
    """
    Attempt to locate optimization.py in a few likely places and import it as a module.
    Returns (path_used, module_object).
    """
    # 1) Look relative to this package directory: <...>/site-packages/pycbs/../pycbs-opt/optimization.py
    this_file = Path(__file__).resolve()
    pkg_dir = this_file.parent  # .../pycbs
    candidate = pkg_dir.parent / "pycbs-opt" / "optimization.py"
    if candidate.exists():
        spec = importlib.util.spec_from_file_location("pycbs_opt_module", str(candidate))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return str(candidate), mod

    # 2) Look for optimization.py next to this module (defensive)
    candidate2 = pkg_dir / "pycbs-opt" / "optimization.py"
    if candidate2.exists():
        spec = importlib.util.spec_from_file_location("pycbs_opt_module", str(candidate2))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return str(candidate2), mod

    # 3) As a last resort, search sys.path for a file named optimization.py inside a folder named 'pycbs-opt'
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
    cfg.optionxform = str  # keep case of keys (but we'll handle lowercasing)
    read_ok = cfg.read(str(ini_path))
    if not read_ok:
        raise FileNotFoundError(f"Cannot read config file: {ini_path}")

    # prefer an explicit "OPTIMIZATION" section; otherwise take the first section or defaults
    section = None
    if "OPTIMIZATION" in cfg:
        section = "OPTIMIZATION"
    elif cfg.sections():
        section = cfg.sections()[0]
    else:
        # no sections: use DEFAULT
        section = configparser.DEFAULTSECT

    params_raw = dict(cfg[section]) if section != configparser.DEFAULTSECT else dict(cfg.defaults())
    # normalize keys to lowercase
    params = {k.lower(): v for k, v in params_raw.items()}
    return params


def ensure_output_dir(dirpath: Path) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def prepare_options_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    # defaults
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

    # configure logging minimal by default
    logging.basicConfig(level=logging.INFO if args.verbose >= 1 else logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        params = load_params_from_ini(args.config)
    except Exception as e:
        logger.error("Failed to read config: %s", e)
        sys.exit(2)

    # mandatory xyz_file
    if "xyz_file" not in params and "xyz" not in params:
        logger.error("Missing required 'xyz_file' entry in INI. Provide the XYZ path under key 'xyz_file'.")
        sys.exit(2)

    xyz_file = Path(params.get("xyz_file", params.get("xyz"))).expanduser()
    if not xyz_file.exists():
        logger.error("XYZ file does not exist: %s", xyz_file)
        sys.exit(2)

    # basis handling
    basis1_name = params.get("basis1", "vdz")
    basis2_name = params.get("basis2", "vtz")
    basis1 = map_basis(basis1_name)
    basis2 = map_basis(basis2_name)

    # output directory: config may optionally specify output_dir
    outdir = Path(params.get("output_dir", "PyCBS-OUTPUTS"))
    ensure_output_dir(outdir)

    # find and import optimization module
    try:
        path_used, opt_mod = find_optimization_module()
    except Exception as e:
        logger.error("Cannot locate optimization.py: %s", e)
        traceback.print_exc()
        sys.exit(3)

    # Read xyz using the optimization module's read_xyz if available, otherwise parse simple xyz
    if not hasattr(opt_mod, "read_xyz") or not hasattr(opt_mod, "optimize_geometry"):
        logger.error("Loaded optimization module doesn't expose expected functions (read_xyz, optimize_geometry)")
        sys.exit(4)

    symbols, coords0 = opt_mod.read_xyz(str(xyz_file))

    options = prepare_options_from_params(params)
    # options dict can contain extra keys used by the module's optimize_geometry
    if "method" in params:
        options["method"] = params["method"]

    # run optimization -> returns a dict (optimize_geometry semantics)
    logger.info("Starting optimization (basis pair: %s, %s) ...", basis1, basis2)
    basis_pair = (basis1, basis2)
    res = opt_mod.optimize_geometry(symbols, coords0, basis_pair=basis_pair, options=options)

    # write optimized geometry to an output file inside outdir
    out_xyz = outdir / (params.get("output_xyz", "optimized.xyz"))
    opt_mod.write_xyz(str(out_xyz), symbols, res.get("coords", res.get("x", [])), comment="Optimized by pycbs-opt")

    # produce a tiny summary file
    summary_file = outdir / (params.get("summary_file", "optimization_summary.txt"))
    with summary_file.open("w", encoding="utf-8") as fh:
        fh.write("pycbs-opt summary\n")
        fh.write(f"config: {args.config}\n")
        fh.write(f"xyz: {xyz_file}\n")
        fh.write(f"basis_pair: {basis_pair}\n")
        fh.write(f"method: {options.get('method')}\n")
        fh.write("Result keys:\n")
        for k in sorted(res.keys()):
            fh.write(f"  {k}: {res[k]}\n")

    logger.info("Optimization finished. Results written to: %s", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
