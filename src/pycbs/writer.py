# src/pycbs/writer.py
"""
Writer utilities for pyCBS results.
Single, consistent implementation for header, per-job results, errors,
optimization history, and summary table.
"""

from pathlib import Path
from typing import Any, Dict, Optional, List

LOGO = """
                             $$$$$$\  $$$$$$$\   $$$$$$\  
                            $$  __$$\ $$  __$$\ $$  __$$\ 
        $$$$$$\  $$\   $$\ $$ /  \__|$$ |  $$ |$$ /  \__|
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
        *      Faculty of Chemistry,University of Havana      *
        *                                                     *
        *                        and                          *
        *                                                     *
        *              Antonio J. C. Varandas                 *
        *    Department of Chemistry, and Chemistry Centre    *
        *                University of Coimbra                *                    
        *******************************************************
"""

GENERAL_CITATION_WRITTEN = False
RESULTS_SUMMARY: List[Dict[str, Any]] = []


def write_header(filename: str) -> None:
    """Write initial header to output file (overwrites file)."""
    global GENERAL_CITATION_WRITTEN, RESULTS_SUMMARY
    RESULTS_SUMMARY = []
    with open(filename, "w") as f:
        f.write(LOGO)
        f.write("\n\n")
        f.write("             pyCBS: Complete Basis Set Extrapolation Tool\n\n")
        f.write(INFO_BLOCK)
        f.write("\n\n")
        if not GENERAL_CITATION_WRITTEN:
            GENERAL_CITATION_WRITTEN = True


def write_error(filename: str, section_name: str, message: str) -> None:
    """Append an error message for the named section."""
    with open(filename, "a") as f:
        f.write(f"\nERROR in [{section_name}]: {message}\n")


def _coerce_data_dict(data: Any) -> Dict[str, Any]:
    """Ensure 'data' used for printing parameters is a dict. If it's a string, return {'input': data}."""
    if data is None:
        return {}
    if isinstance(data, dict):
        return data
    # Try to coerce Path-like or other objects
    try:
        return dict(data)
    except Exception:
        return {"value": data}


def write_result(filename: str, scheme: str, data: Any, EHF: Optional[float] = None,
                 dc: Optional[float] = None, energy: Optional[float] = None) -> None:
    """
    Write a detailed result block.

    Args:
        filename: path to file (string)
        scheme: scheme name (string)
        data: dict-like of input parameters (or any, will be coerced)
        EHF: Hartree-Fock CBS energy (optional)
        dc: dynamic correlation contribution (optional)
        energy: total CBS energy (optional)
    """
    global RESULTS_SUMMARY
    datad = _coerce_data_dict(data)

    # determine whether HF+dc components present
    has_components = (EHF is not None) and (dc is not None)

    with open(filename, "a") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"                       Extrapolation Scheme: {scheme}\n")
        f.write("=" * 70 + "\n")
        f.write("Input Parameters:\n")
        f.write("-" * 70 + "\n")
        if datad:
            # Print sorted keys for stable output
            for key in sorted(datad.keys()):
                try:
                    f.write(f"{key:>20}: {datad[key]}\n")
                except Exception:
                    f.write(f"{key:>20}: {str(datad[key])}\n")
        else:
            f.write(" (no input parameters)\n")
        f.write("-" * 70 + "\n")
        f.write("Extrapolation Results:\n")
        f.write("-" * 70 + "\n")

        if has_components:
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
            # If only one value is provided, prefer energy; otherwise write whatever we have
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

    # Append to summary record
    RESULTS_SUMMARY.append({
        'scheme': scheme,
        'energy': energy,
        'EHF': EHF,
        'dc': dc,
        'has_components': has_components
    })


def write_optimization_summary(filename: str, history: list) -> None:
    """Write optimization cycle history (if provided)."""
    with open(filename, "a") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("                      OPTIMIZATION CYCLE HISTORY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Cycle':>5} {'p0':>12} {'p1':>12} {'Energy[Ha]':>18} {'DispFactor':>12}\n")
        f.write("-" * 70 + "\n")
        for step in history:
            try:
                cyc = int(step.get('cycle', 0))
            except Exception:
                cyc = 0
            params = step.get('parameters', [None, None])
            try:
                p0 = float(params[0])
            except Exception:
                p0 = 0.0
            try:
                p1 = float(params[1])
            except Exception:
                p1 = 0.0
            try:
                en = float(step.get('energy', 0.0))
            except Exception:
                en = 0.0
            try:
                dff = float(step.get('displacement_factor', 0.0))
            except Exception:
                dff = 0.0
            f.write(f"{cyc:5d} {p0:12.8f} {p1:12.8f} {en:18.10f} {dff:12.6f}\n")
        f.write("\n" + "=" * 70 + "\n\n")


def write_summary_table(filename: str) -> None:
    """Write the final summary table appended to the results file."""
    global RESULTS_SUMMARY
    if not RESULTS_SUMMARY:
        return
    with open(filename, "a") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("                      SUMMARY OF RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Scheme':<20}{'HF (CBS)':>18}{'Dynamic Corr.':>18}{'Total Energy':>18}\n")
        f.write("-" * 70 + "\n")
        for item in RESULTS_SUMMARY:
            scheme = item.get('scheme', '')
            energy = item.get('energy', None)
            EHF = item.get('EHF', None)
            dc = item.get('dc', None)
            has_components = item.get('has_components', False)

            if has_components:
                try:
                    f.write(f"{scheme:<20}{float(EHF):18.10f}{float(dc):18.10f}{float(energy):18.10f}\n")
                except Exception:
                    f.write(f"{scheme:<20}{str(EHF):>18}{str(dc):>18}{str(energy):>18}\n")
            else:
                try:
                    f.write(f"{scheme:<20}{'':18}{'':18}{float(energy):18.10f}\n")
                except Exception:
                    f.write(f"{scheme:<20}{'':18}{'':18}{str(energy):>18}\n")
        f.write("\n" + "=" * 70 + "\n")
