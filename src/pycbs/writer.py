# src/pycbs/writer.py
"""
Writer utilities for pyCBS results.

- Produces two main sections:
    1) CBS Extrapolations (single summary table)
    2) Geometrical Optimization (Cycle vs CBS Energy table)
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterable, Tuple, Union
import datetime

# Visual header / info block (keeps similarity to your prior writer)
LOGO = """
                             $$$$$$\  $$$$$$$\   $$$$$$\  
                            $$  __$$\ $$  __$$\ $$  __$$\ 
        $$$$$$\  $$\   $$\  $$ /  \__|$$ |  $$ |$$ /  \__|
        $$  __$$\ $$ |  $$ |$$ |      $$$$$$$\ |\$$$$$$\  
        $$ /  $$ |$$ |  $$ |$$ |      $$  __$$\  \____$$\  
        $$ |  $$ |$$ |  $$ |$$ |  $$\ $$ |  $$ |$$\   $$ |
        $$$$$$$  |\$$$$$$$ |\$$$$$$  |$$$$$$$  |\$$$$$$  |
        $$  ____/  \____$$ | \______/ \_______/  \______/ 
        $$ |      $$\   $$ |                              
        $$ |      \$$$$$$  |                              
        \__|       \______/                               
"""

INFO_BLOCK = """
        *******************************************************
        *               Alberto Guerra-Barroso,               *
        *              Fabio J. Delgado-Alpízar               *
        *    Lab of Computational and Theoretical Chemistry   *
        *      Faculty of Chemistry, University of Havana     *
        *                                                     *
        *                        and                          *
        *                                                     *
        *              Antonio J. C. Varandas                 *
        *    Department of Chemistry, and Chemistry Centre    *
        *                University of Coimbra                *                    
        *******************************************************
"""

# Defaults for scheme classification (lowercase)
DEFAULT_HF_COMPONENTS = {"feller", "truhlar_hf", "jensen", "klopper", "hf_e"}
DEFAULT_CORR_COMPONENTS = {"martin", "truhlar_corr", "oanc", "bakowies", "huh-lee", "halkier-helgaker"}
DEFAULT_MIXED_SCHEMES = {"uste1", "uste2", "uspe"}

# Global collector (optional compatibility)
RESULTS_SUMMARY: List[Dict[str, Any]] = []


# -------------------------
# Formatting & rendering
# -------------------------
def _format_numeric(v: Any) -> str:
    """Return numeric values as plain decimal text (10 fractional digits).
    None -> '-'."""
    if v is None:
        return "-"
    try:
        if isinstance(v, bool):
            return str(v)
        val = float(v)
        # Plain decimal with fixed precision (no scientific notation)
        return f"{val:.10f}"
    except Exception:
        return str(v)



def _format_generic(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return _format_numeric(v)
    return str(v)


def _render_table(headers: List[str], rows: Iterable[List[Any]]) -> str:
    """Render a simple left-aligned ASCII table.

    When there are no data rows, render a single row filled with '-' so the
    table body is not empty (user requested).
    """
    # minimal table renderer (original behaviour preserved)
    cols = len(headers)
    rows = list(rows)
    if not rows:
        rows = [["-"] * cols]
    # compute widths
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    sep = " | "
    header_line = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    bar = "-+-".join("-" * widths[i] for i in range(len(headers)))
    body = "\n".join(sep.join(str(c).ljust(widths[i]) for i, c in enumerate(r)) for r in rows)
    return f"{header_line}\n{bar}\n{body}"


# -------------------------
# Output helpers for optimizations
# -------------------------
def ensure_outputs_dir(base: Path | None = None) -> Path:
    base_path = Path(base) if base is not None else Path.cwd()
    out = base_path / "PyCBS-OUTPUTS"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_cycle_energies(out_dir: Path, prefix: str, history: list):
    """
    Write cycle-by-cycle energies to a CSV-like file under out_dir.
    history: list of dicts with keys e.g. {'cycle': int, 'energy': float, 'parameters': ...}
    File name: {prefix}_cycle_energies.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{prefix}_cycle_energies.csv"
    with open(fname, "w") as fh:
        fh.write("cycle,step,E_cbs\n")
        for entry in history:
            cycle = entry.get("cycle", "")
            if "energies" in entry and isinstance(entry["energies"], (list, tuple)):
                # if entry contains per-step energies, write them
                for i, e in enumerate(entry["energies"]):
                    fh.write(f"{cycle},{i},{e:.10f}\n")
            else:
                e = entry.get("energy", entry.get("E_cbs"))
                fh.write(f"{cycle},0,{float(e):.10f}\n")
    return fname


def write_final_xyz(out_dir: Path, prefix: str, symbols: list, coords: list, final_energy: float):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{prefix}_final_opt.xyz"
    nat = len(symbols)
    with open(fname, "w") as fh:
        fh.write(f"{nat}\n")
        fh.write(f"Optimized by pyCBS, E_CBS = {final_energy:.10f} Ha\n")
        for s, c in zip(symbols, coords):
            fh.write(f"{s} {c[0]: .10f} {c[1]: .10f} {c[2]: .10f}\n")
    return fname


# -------------------------
# New: Write a top-level header file (output.txt)
# -------------------------
def write_extrapolations(out_path: Path, extrap: Dict[str, Any]):
    """
    Write a small extrapolation summary table to the given path (appends).
    extrap is a dict like:
      {'a_corr': 0.123, 'b_hf': 0.456, 'E_hf_cbs': -76.1, 'E_corr_cbs': -0.05}
    """
    out_path = Path(out_path)
    with open(out_path, "a") as fh:
        fh.write("\nEXTRAPOLATION SUMMARY\n")
        for k, v in extrap.items():
            fh.write(f"{k:20s} : {_format_generic(v)}\n")
    return out_path


def write_header(output_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None, title: Optional[str] = None):
    """
    Write a human-readable header file (overwrites existing file).
    output_path: path to the header/output file (e.g. PyCBS-OUTPUTS/output.txt or path passed by CLI)
    metadata: optional dict of key->value pairs to include (method, bases, params...)
    title: optional title line
    Returns the Path written.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write(LOGO + "\n")
        fh.write(INFO_BLOCK + "\n")
        fh.write(f"pyCBS results generated: {datetime.datetime.utcnow().isoformat()} UTC\n\n")
        if title:
            fh.write(f"{title}\n\n")
        if metadata:
            fh.write("RUN METADATA\n")
            for k, v in metadata.items():
                fh.write(f"{k:20s} : {_format_generic(v)}\n")
            fh.write("\n")
        fh.write("Notes:\n")
        fh.write(" - Cycle-by-cycle energies are written as CSV files: <PREFIX>_cycle_energies.csv\n")
        fh.write(" - Final geometries are written as XYZ: <PREFIX>_final_opt.xyz\n")
    return out