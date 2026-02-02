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
    str_rows = [[_format_generic(c) for c in row] for row in rows]

    # If there are no data rows, create one row of '-' placeholders (one per header)
    if not str_rows:
        str_rows = [["-" for _ in headers]]

    # Compute columns and widths
    columns = list(zip(*([headers] + str_rows)))
    col_widths = [max(len(str(x)) for x in col) + 2 for col in columns]

    sep = "+" + "+".join("-" * w for w in col_widths) + "+"

    def render_row(cells: List[str]) -> str:
        rendered = []
        for cell, w in zip(cells, col_widths):
            if cell == "-":
                rendered.append(" " + cell.center(w - 1) + "|")
            else:
                rendered.append(" " + cell.ljust(w - 1) + "|")
        return "|" + "".join(rendered)

    lines = [sep, render_row(headers), sep]
    for r in str_rows:
        lines.append(render_row(r))
    lines.append(sep)
    return "\n".join(lines)



# -------------------------
# Normalization & helpers
# -------------------------
def _norm_scheme(scheme: Optional[str]) -> str:
    if scheme is None:
        return "unknown"
    return str(scheme).strip().lower().replace("_", "-")



def _coerce_entry(entry: Any) -> Dict[str, Any]:
    if entry is None:
        return {}
    if isinstance(entry, dict):
        return entry
    try:
        return dict(entry)
    except Exception:
        return {"value": entry}


def _classify_and_build_row(
    entry: Dict[str, Any],
    hf_components: Iterable[str],
    corr_components: Iterable[str],
    mixed_schemes: Iterable[str],
) -> List[Any]:
    """
    Return row in exact column order:
    ['calculation', 'scheme', 'HF_CBS', 'Corr_CBS', 'Freq_CBS', 'TensProp', 'Total Energy']
    """
    label = entry.get("calculation") or entry.get("label") or entry.get("name") or entry.get("section") or "unnamed"
    scheme_raw = entry.get("scheme", entry.get("method", "unknown"))
    scheme = _norm_scheme(scheme_raw)

    hf_val = entry.get("hf_cbs") if "hf_cbs" in entry else entry.get("EHF") if "EHF" in entry else entry.get("hf") if "hf" in entry else None
    corr_val = entry.get("corr_cbs") if "corr_cbs" in entry else entry.get("dc") if "dc" in entry else entry.get("corr") if "corr" in entry else None
    freq_val = entry.get("freq_cbs") if "freq_cbs" in entry else entry.get("frequency") if "frequency" in entry else entry.get("freq") if "freq" in entry else None
    tens_val = entry.get("tens_prop") if "tens_prop" in entry else entry.get("tensprop") if "tensprop" in entry else entry.get("tensor") if "tensor" in entry else None
    total_val = entry.get("total_energy") if "total_energy" in entry else entry.get("energy") if "energy" in entry else entry.get("total") if "total" in entry else None
    prop_hint = (entry.get("property_type") or entry.get("property") or "").strip().lower()

    hf_set = {s.lower() for s in hf_components}
    corr_set = {s.lower() for s in corr_components}
    mixed_set = {s.lower() for s in mixed_schemes}

    # Exact rules as requested:
    if scheme in hf_set:
        return [label, scheme_raw, hf_val, None, None, None, None]
    if scheme in corr_set:
        return [label, scheme_raw, None, corr_val, None, None, None]
    if scheme in mixed_set:
        if prop_hint.startswith("freq") or (freq_val is not None):
            return [label, scheme_raw, hf_val, corr_val, freq_val, None, None]
        if prop_hint.startswith("tens") or (tens_val is not None):
            return [label, scheme_raw, hf_val, corr_val, None, tens_val, None]
        return [label, scheme_raw, hf_val, corr_val, None, None, total_val]
    # Unknown: infer from fields (prefer explicit fields)
    if hf_val is not None and corr_val is None:
        return [label, scheme_raw, hf_val, None, None, None, None]
    if corr_val is not None and hf_val is None:
        return [label, scheme_raw, None, corr_val, None, None, None]
    return [label, scheme_raw, hf_val, corr_val, freq_val, tens_val, total_val]


# -------------------------
# Primary writer API
# -------------------------
def write_header(path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(LOGO + "\n\n")
        f.write("             pyCBS: Complete Basis Set Extrapolation Tool\n\n")
        f.write(INFO_BLOCK + "\n\n")


def write_error(path: Union[str, Path], section_name: str, message: str) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(f"\nERROR in [{section_name}]: {message}\n")


def write_reports(
    filename: Union[str, Path],
    calculations: Iterable[Any],
    *,
    hf_components: Optional[Iterable[str]] = None,
    corr_components: Optional[Iterable[str]] = None,
    mixed_schemes: Optional[Iterable[str]] = None,
    detailed_blocks: bool = False,
) -> Dict[str, str]:
    """
    Writes a single file (overwrites) containing:
      - header
      - CBS Extrapolations summary table (exact column order)
      - Geometrical Optimization table (Cycle, CBS Energy)

    No CSV/LaTeX exports. No examples are written.
    """
    out = {}
    path = Path(filename)
    outdir = path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    hf_components = set(hf_components) if hf_components is not None else set(DEFAULT_HF_COMPONENTS)
    corr_components = set(corr_components) if corr_components is not None else set(DEFAULT_CORR_COMPONENTS)
    mixed_schemes = set(mixed_schemes) if mixed_schemes is not None else set(DEFAULT_MIXED_SCHEMES)

    # Clear previous global summary and write header (overwrite)
    global RESULTS_SUMMARY
    RESULTS_SUMMARY = []
    write_header(path)

    # Prepare calculation rows
    calc_list = [_coerce_entry(c) for c in calculations] if calculations is not None else []
    headers = ["calculation", "scheme", "HF_CBS", "Corr_CBS", "Freq_CBS", "TensProp", "Total Energy"]
    rows: List[List[Any]] = []

    # Optional detailed blocks, if caller wants them
    if detailed_blocks:
        for entry in calc_list:
            scheme = entry.get("scheme") or entry.get("method") or "unknown"
            EHF = entry.get("hf_cbs") or entry.get("EHF") or entry.get("hf")
            dc = entry.get("corr_cbs") or entry.get("dc") or entry.get("corr")
            energy = entry.get("total_energy") or entry.get("energy") or entry.get("total")
            with path.open("a", encoding="utf-8") as f:
                f.write("\n" + "=" * 70 + "\n")
                f.write(f"                       Extrapolation Scheme: {scheme}\n")
                f.write("=" * 70 + "\n")
                f.write("Input Parameters:\n")
                f.write("-" * 70 + "\n")
                if entry:
                    for key in sorted(entry.keys()):
                        try:
                            f.write(f"{key:>20}: {entry[key]}\n")
                        except Exception:
                            f.write(f"{key:>20}: {str(entry[key])}\n")
                else:
                    f.write(" (no input parameters)\n")
                f.write("-" * 70 + "\n")
                f.write("Extrapolation Results:\n")
                f.write("-" * 70 + "\n")
                if (EHF is not None) and (dc is not None):
                    try:
                        f.write(f"{'Hartree-Fock (CBS):':>30} {float(EHF):.10f}\n")
                    except Exception:
                        f.write(f"{'Hartree-Fock (CBS):':>30} {EHF}\n")
                    try:
                        f.write(f"{'Dynamic Correlation:':>30} {float(dc):.10f}\n")
                    except Exception:
                        f.write(f"{'Dynamic Correlation:':>30} {dc}\n")
                    try:
                        f.write(f"{'Total CBS Energy:':>30} {float(energy):.10f}\n")
                    except Exception:
                        f.write(f"{'Total CBS Energy:':>30} {energy}\n")
                else:
                    if energy is not None:
                        try:
                            f.write(f"{'CBS Extrapolated Energy:':>30} {float(energy):.10f}\n")
                        except Exception:
                            f.write(f"{'CBS Extrapolated Energy:':>30} {energy}\n")
                    elif EHF is not None:
                        try:
                            f.write(f"{'HF (CBS):':>30} {float(EHF):.10f}\n")
                        except Exception:
                            f.write(f"{'HF (CBS):':>30} {EHF}\n")
                    elif dc is not None:
                        try:
                            f.write(f"{'Dynamic Corr (only):':>30} {float(dc):.10f}\n")
                        except Exception:
                            f.write(f"{'Dynamic Corr (only):':>30} {dc}\n")
                    else:
                        f.write("No numeric result available\n")
                f.write("=" * 70 + "\n\n")

    # Build summary rows
    for e in calc_list:
        row = _classify_and_build_row(e, hf_components, corr_components, mixed_schemes)
        rows.append(row)
        RESULTS_SUMMARY.append({
            "calculation": row[0],
            "scheme": row[1],
            "EHF": row[2],
            "dc": row[3],
            "freq": row[4],
            "tens": row[5],
            "energy": row[6],
        })

    # Write CBS summary table
    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("                      CBS EXTRAPOLATIONS\n")
        f.write("=" * 70 + "\n\n")
        f.write(_render_table(headers, rows) + "\n\n")
    return out



def write_reports1(
    filename: Union[str, Path],
    calculations: Iterable[Any],
    opt_cycles: Iterable[Any]
) -> Dict[str, str]:
    """
    Write a standalone optimization report file containing:
      - header/logo
      - GEOMETRICAL OPTIMIZATION table (Cycle, CBS Energy)

    This function creates the parent directory, writes a fresh header
    (overwrites any existing file of the same name), then appends the
    optimization table. Returns a small dict with the final filename.
    """
    out = {}
    path = Path(filename)
    outdir = path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    # Write header (overwrite existing file with logo + info)
    write_header(path)

    # Build optimization rows in canonical two-column form
    opt_headers = ["Cycle", "CBS Energy"]
    opt_rows: List[List[Any]] = []
    for itm in opt_cycles:
        if isinstance(itm, dict):
            cycle = itm.get("cycle", itm.get("step", "-"))
            energy = itm.get("cbs_energy", itm.get("energy", itm.get("total_energy", "-")))
        elif isinstance(itm, (list, tuple)) and len(itm) >= 2:
            cycle, energy = itm[0], itm[1]
        else:
            # ignore malformed entries
            continue
        opt_rows.append([cycle, energy])

    # Write optimization table
    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("                      GEOMETRICAL OPTIMIZATION\n")
        f.write("=" * 70 + "\n\n")
        f.write(_render_table(opt_headers, opt_rows) + "\n\n")

    out["txt"] = str(path)
    return out

def clear_results_summary() -> None:
    """Clear global RESULTS_SUMMARY collector."""
    global RESULTS_SUMMARY
    RESULTS_SUMMARY = []
