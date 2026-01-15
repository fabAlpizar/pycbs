"""
writer.py

A robust, professional writer utility for pyCBS that creates human- and machine-readable
reports for:
  1) CBS Extrapolations (summary table)
  2) Geometrical Optimization (cycle vs CBS-energy table)

Design goals
- Follow the mapping rules provided by the user: HF-only schemes, correlation-only schemes,
  mixed (uste/uste2/uspe) that return both HF and correlation CBS and total/other properties.
- Produce a single summary table for CBS extrapolations with these 7 columns (in order):
    calculation, scheme, HF_CBS, Corr_CBS, Freq_CBS, TensProp, Total Energy
- Produce an optimization table with two columns: Cycle, CBS Energy
- Be robust: accept inputs as simple dictionaries (order-agnostic), validate them, and
  fall back to sensible defaults when values are missing.
- Provide plain-text (ASCII) output plus optional CSV and LaTeX exports when pandas is
  available. Avoid hard dependency on third-party packages for terminal output.

Usage
- The repository caller should collect per-calculation results into a list of dicts where each
  dict minimally contains:
      {
          'calculation': '<input-section-name-or-user-label>',
          'scheme': '<scheme-name>',
          'hf_cbs': <float or None>,         # optional depending on scheme
          'corr_cbs': <float or None>,       # optional depending on scheme
          'freq_cbs': <float or None>,       # optional
          'tens_prop': <str or float or None>, # optional (tensorial properties)
          'total_energy': <float or None>,   # optional
          'property_type': '<energy|frequency|tensprop|total>' # optional hint
      }
- Optimization cycles: list of (cycle_index, cbs_energy) or list of dicts

"""
from __future__ import annotations
import os
import math
from typing import List, Dict, Any, Optional, Iterable, Tuple
import shutil

# Try importing pandas only for enhanced exports; otherwise continue without it.
try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore


# ---------------------------------------------------------------------------
# Scheme classification (defaults). These can be extended at runtime by the caller.
# ---------------------------------------------------------------------------
DEFAULT_HF_COMPONENTS = {"feller", "truhlar_hf", "jensen", "klopper", "hf_e"}
DEFAULT_CORR_COMPONENTS = {"martin", "truhlar_corr", "oanc", "bakowies", "huh-lee", "halkier-helgaker"}
DEFAULT_MIXED_SCHEMES = {"uste1", "uste2", "uspe"}  # these return HF + CORR + Total (or property)


# ---------------------------------------------------------------------------
# Helper: plain text table renderer (dependency-free)
# ---------------------------------------------------------------------------

def _format_value(v: Any) -> str:
    """Format numeric values in scientific format; keep strings as-is; None -> '-'."""
    if v is None:
        return "-"
    if isinstance(v, (float, int)) and (not isinstance(v, bool)):
        # Use scientific notation for energies; but if number is small integer-like show as int
        try:
            if abs(float(v)) >= 1e-4:
                return f"{float(v):.6e}"
            else:
                return f"{float(v):.6e}"
        except Exception:
            return str(v)
    return str(v)


def _render_table(headers: List[str], rows: Iterable[List[Any]]) -> str:
    """Render a simple left-aligned ASCII table.

    This function does not require third-party packages and aims for a clean scientific look.
    """
    # Convert all cells to strings with formatting rules
    str_rows = [[_format_value(c) for c in row] for row in rows]

    # Compute column widths
    columns = list(zip(*([headers] + str_rows))) if str_rows else [(h,) for h in headers]
    col_widths = [max(len(str(x)) for x in col) + 2 for col in columns]

    # Helper to render a single row
    def render_row(cells: List[str]) -> str:
        return "|" + "".join(f" {cell.ljust(w-1)}|" for cell, w in zip(cells, col_widths))

    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    lines = [sep, render_row(headers), sep]
    for r in str_rows:
        lines.append(render_row(r))
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Writer implementation
# ---------------------------------------------------------------------------
class Writer:
    """Create publication-style, machine-friendly reports for CBS extrapolations
    and geometry optimization cycles.

    The Writer does not assume the internal structure of the rest of the package; it
    operates on simple Python structures (lists/dicts). This keeps it easy to integrate.
    """

    def __init__(
        self,
        outdir: Optional[str] = "outputs",
        hf_components: Optional[Iterable[str]] = None,
        corr_components: Optional[Iterable[str]] = None,
        mixed_schemes: Optional[Iterable[str]] = None,
        write_csv: bool = True,
        write_latex: bool = False,
    ) -> None:
        self.outdir = outdir or "outputs"
        self.hf_components = set(hf_components) if hf_components is not None else set(DEFAULT_HF_COMPONENTS)
        self.corr_components = set(corr_components) if corr_components is not None else set(DEFAULT_CORR_COMPONENTS)
        self.mixed_schemes = set(mixed_schemes) if mixed_schemes is not None else set(DEFAULT_MIXED_SCHEMES)
        self.write_csv = write_csv
        self.write_latex = write_latex and (pd is not None)

        # Prepare output directory
        if os.path.exists(self.outdir):
            # keep it but ensure it's writeable
            if not os.path.isdir(self.outdir):
                raise RuntimeError(f"Output path {self.outdir} exists and is not a directory")
        else:
            os.makedirs(self.outdir, exist_ok=True)

    # ---------------------- Public API ----------------------
    def write_reports(self, calculations: List[Dict[str, Any]], opt_cycles: List[Tuple[int, float]]) -> Dict[str, str]:
        """Write the CBS summary and optimization tables.

        Returns a dict with paths to the generated textual/CSV/LaTeX files (where applicable).
        """
        cbs_path = os.path.join(self.outdir, "cbs_extrapolations.txt")
        opt_path = os.path.join(self.outdir, "geometry_optimization.txt")

        # Build tables
        cbs_headers, cbs_rows = self._build_cbs_table(calculations)
        opt_headers, opt_rows = self._build_opt_table(opt_cycles)

        # Render to text and write
        with open(cbs_path, "w", encoding="utf-8") as f:
            f.write("CBS Extrapolations\n")
            f.write("=" * 80 + "\n\n")
            f.write(_render_table(cbs_headers, cbs_rows))
            f.write("\n")

        with open(opt_path, "w", encoding="utf-8") as f:
            f.write("Geometrical Optimization\n")
            f.write("=" * 80 + "\n\n")
            f.write(_render_table(opt_headers, opt_rows))
            f.write("\n")

        outputs = {"cbs_txt": cbs_path, "opt_txt": opt_path}

        # Optional: export CSV/LaTeX using pandas if available
        if self.write_csv or self.write_latex:
            if pd is None:
                # graceful fallback: notify user via return value that pandas was not available
                outputs["csv_warning"] = "pandas not installed; CSV/LaTeX exports skipped"
            else:
                # Build DataFrames
                df_cbs = pd.DataFrame([dict(zip(cbs_headers, r)) for r in cbs_rows])
                df_opt = pd.DataFrame([dict(zip(opt_headers, r)) for r in opt_rows])

                if self.write_csv:
                    csv_cbs = os.path.join(self.outdir, "cbs_extrapolations.csv")
                    csv_opt = os.path.join(self.outdir, "geometry_optimization.csv")
                    df_cbs.to_csv(csv_cbs, index=False)
                    df_opt.to_csv(csv_opt, index=False)
                    outputs["cbs_csv"] = csv_cbs
                    outputs["opt_csv"] = csv_opt

                if self.write_latex:
                    tex_cbs = os.path.join(self.outdir, "cbs_extrapolations.tex")
                    tex_opt = os.path.join(self.outdir, "geometry_optimization.tex")
                    with open(tex_cbs, "w", encoding="utf-8") as f:
                        f.write(df_cbs.to_latex(index=False, float_format="%.6e"))
                    with open(tex_opt, "w", encoding="utf-8") as f:
                        f.write(df_opt.to_latex(index=False, float_format="%.6e"))
                    outputs["cbs_tex"] = tex_cbs
                    outputs["opt_tex"] = tex_opt

        return outputs

    # ---------------------- Internal helpers ----------------------
    def _build_cbs_table(self, calculations: List[Dict[str, Any]]) -> Tuple[List[str], List[List[Any]]]:
        """Given a list of calculation-result dicts, produce headers and rows for the
        CBS summary table with the exact column order requested by the user.

        Column order: calculation, scheme, HF_CBS, Corr_CBS, Freq_CBS, TensProp, Total Energy
        """
        headers = ["calculation", "scheme", "HF_CBS", "Corr_CBS", "Freq_CBS", "TensProp", "Total Energy"]
        rows: List[List[Any]] = []

        for entry in calculations:
            # Normalize keys to lower-case for flexible input
            calculation_label = entry.get("calculation") or entry.get("label") or entry.get("name") or "unnamed"
            scheme = entry.get("scheme", "unknown").lower()

            hf_val = entry.get("hf_cbs")
            corr_val = entry.get("corr_cbs")
            freq_val = entry.get("freq_cbs")
            tens_val = entry.get("tens_prop") or entry.get("tensprop")
            total_val = entry.get("total_energy")
            property_hint = (entry.get("property_type") or "").lower()

            # Decide filling according to scheme classification (user rules)
            if scheme in self.hf_components:
                # Only HF CBS produced
                row = [calculation_label, scheme, hf_val, None, None, None, None]
            elif scheme in self.corr_components:
                # Only correlation CBS produced
                row = [calculation_label, scheme, None, corr_val, None, None, None]
            elif scheme in self.mixed_schemes:
                # Mixed: fill HF, CORR, and either Total Energy or property columns per hint
                if property_hint.startswith("freq") or (freq_val is not None):
                    row = [calculation_label, scheme, hf_val, corr_val, freq_val, None, None]
                elif property_hint.startswith("tens") or (tens_val is not None):
                    row = [calculation_label, scheme, hf_val, corr_val, None, tens_val, None]
                else:
                    row = [calculation_label, scheme, hf_val, corr_val, None, None, total_val]
            else:
                # Unknown scheme: try to infer from available fields. Prioritize explicit fields.
                if hf_val is not None and corr_val is None:
                    row = [calculation_label, scheme, hf_val, None, None, None, None]
                elif corr_val is not None and hf_val is None:
                    row = [calculation_label, scheme, None, corr_val, None, None, None]
                else:
                    # If both or neither present, preserve whatever is available and prefer total_energy
                    row = [
                        calculation_label,
                        scheme,
                        hf_val,
                        corr_val,
                        freq_val,
                        tens_val,
                        total_val,
                    ]

            rows.append(row)

        return headers, rows

    def _build_opt_table(self, opt_cycles: List[Tuple[int, float]]) -> Tuple[List[str], List[List[Any]]]:
        """Build the two-column optimization table. Accepts either a list of (cycle, energy)
        or a list of dicts containing 'cycle' and 'cbs_energy'."""
        headers = ["Cycle", "CBS Energy"]
        rows: List[List[Any]] = []

        for item in opt_cycles:
            if isinstance(item, dict):
                cycle = item.get("cycle")
                energy = item.get("cbs_energy") or item.get("energy") or item.get("total_energy")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                cycle, energy = item[0], item[1]
            else:
                # skip malformed entries
                continue
            rows.append([cycle, energy])

        return headers, rows


# ---------------------------------------------------------------------------
# Minimal self-test / example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example input that demonstrates the mapping rules requested by the user.
    sample_calculations = [
        {"calculation": "calculation_H2_FELLER", "scheme": "feller", "hf_cbs": -1.23456789},
        {"calculation": "calculation_N2_HALKIER_HELGAKER", "scheme": "halkier-helgaker", "corr_cbs": -0.0012345},
        {
            "calculation": "calculation_CH4_USTE1",
            "scheme": "uste1",
            "hf_cbs": -10.123456789,
            "corr_cbs": -0.987654321,
            "total_energy": -11.11111111,
        },
        {
            "calculation": "calculation_CO_USPE_FREQ",
            "scheme": "uspe",
            "hf_cbs": -200.1,
            "corr_cbs": -0.9,
            "freq_cbs": 3450.12,
            "property_type": "frequency",
        },
    ]

    sample_opt = [(0, -11.11111111), (1, -11.11120000), (2, -11.11125000)]

    writer = Writer(outdir="outputs_example", write_csv=True, write_latex=False)
    out = writer.write_reports(sample_calculations, sample_opt)
    print("Generated files:")
    for k, v in out.items():
        print(f"  {k}: {v}")
