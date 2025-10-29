#!/usr/bin/env python3
"""
PYCBS.py - Improved entrypoint for pyCBS with automated basis handling.
"""
from __future__ import annotations

import argparse
import configparser
import logging
import platform
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Dict, Optional

import writer
import basis

# extrapolation modules
try:
    import USTE1
except Exception as e:
    USTE1 = None
    logging.getLogger(__name__).debug("USTE1 not available: %s", e)

try:
    import USTE2
except Exception as e:
    USTE2 = None
    logging.getLogger(__name__).debug("USTE2 not available: %s", e)

try:
    import USPE
except Exception as e:
    USPE = None
    logging.getLogger(__name__).debug("USPE not available: %s", e)

try:
    import tensorial_properties1 as TP
except Exception as e:
    TP = None
    logging.getLogger(__name__).debug("tensorial_properties1 not available: %s", e)


OPT_DEFAULTS = {
    "init_parameters": [0.96654, 103.93761],
    "geo_init": ["r1", "teta"],
    "basis_sets": ["cc-pvtz", "cc-pvqz"],
    "METHOD": "CCSD(T)",
    "x1": 2.792,
    "x2": 3.719,
    "x1_hf": 2.96,
    "x2_hf": 3.87,
    "beta": 1.62,
    "maxcycle": 20,
    "energy_criterion": 1e-8,
    "fac_mult": 0.05,
    "cut": 0.75,
    "workers": None,
}


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pyCBS", description="pyCBS: Complete Basis Set extrapolation tool")
    p.add_argument("-i", "--input", type=Path, default=Path("inputfile.inp"),
                   help="Input file (INI format)")
    p.add_argument("-o", "--output", type=Path, default=Path("results.out"),
                   help="Output results file")
    p.add_argument("--no-opt", action="store_true", help="Disable optimizer even if requested in input")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (use -vv for debug)")
    p.add_argument("--workers", type=int, default=None, help="Override number of worker processes for optimization")
    return p


def setup_logging(verbosity: int = 0, logfile: Optional[Path] = None) -> None:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO

    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        handlers.append(logging.FileHandler(logfile, mode="a"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )


def read_config(input_path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(input_path)
    return cfg


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def parse_list_of_floats(txt: str) -> list[float]:
    parts = [p.strip() for p in txt.strip().split(",") if p.strip() != ""]
    return [float(p) for p in parts]


def parse_list_of_str(txt: str) -> list[str]:
    return [p.strip() for p in txt.strip().split(",") if p.strip() != ""]


def gather_optimization_params(section: configparser.SectionProxy) -> dict:
    params = dict(OPT_DEFAULTS)
    key_parsers = {
        "init_parameters": parse_list_of_floats,
        "basis_sets": parse_list_of_str,
        "METHOD": lambda s: s.strip(),
        "x1": float,
        "x2": float,
        "x1_hf": float,
        "x2_hf": float,
        "beta": float,
        "maxcycle": int,
        "energy_criterion": float,
        "fac_mult": float,
        "cut": float,
        "workers": int,
    }
    for key, parser in key_parsers.items():
        if key in section:
            try:
                params[key] = parser(section.get(key))
            except Exception as e:
                logging.warning(
                    "Could not parse optimization param '%s' value '%s': %s. Using default %s",
                    key, section.get(key), e, params.get(key)
                )
    # Normalize list lengths
    if isinstance(params.get("init_parameters"), list) and len(params["init_parameters"]) >= 2:
        params["init_parameters"] = [float(params["init_parameters"][0]), float(params["init_parameters"][1])]
    else:
        params["init_parameters"] = list(OPT_DEFAULTS["init_parameters"])
    if isinstance(params.get("basis_sets"), list) and len(params["basis_sets"]) >= 2:
        params["basis_sets"] = [params["basis_sets"][0], params["basis_sets"][1]]
    else:
        params["basis_sets"] = list(OPT_DEFAULTS["basis_sets"])
    return params


@contextmanager
def silence_output():
    try:
        with open(sys.devnull, "w") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield
    except Exception:
        yield


def run_uste1(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
    if USTE1 is None:
        logging.error("USTE1 module not found; skipping %s", calc_name)
        return
    try:
        method = section["method"]
        basis1 = section["basis1"]
        basis2 = section["basis2"]
        HF1 = float(section["HF1"])
        HF2 = float(section["HF2"])
        E1 = float(section["E1"])
        E2 = float(section["E2"])
    except KeyError as e:
        logging.error("Missing param in %s: %s", calc_name, e)
        return
    except ValueError as e:
        logging.error("Invalid numeric param in %s: %s", calc_name, e)
        return

    with output_file.open("a") as f:
        f.write("\n")
        f.write(f" JOB: {calc_name}\n")

    # build required dictionaries / energies
    hf_dict, corr_dict = USTE1.dictionaries(method, basis1, basis2)
    Ecr1, Ecr2 = USTE1.correlation_frequency(HF1, HF2, E1, E2)

    # ---- FIXED LINE: pass basis1 and basis2 (not calc_name) ----
    EHF, dc, CBS = USTE1.CBS_extrapolation(HF1, HF2, Ecr1, Ecr2, corr_dict, basis1, basis2)
    # ------------------------------------------------------------

    writer.write_result(str(output_file), calc_name, {"basis1": basis1, "basis2": basis2}, EHF, dc, CBS)



def run_uste2(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
    if USTE2 is None:
        logging.error("USTE2 module not found; skipping %s", calc_name)
        return
    try:
        method = section["method"]
        basis1 = section["basis1"]
        basis2 = section["basis2"]
        E1 = float(section["E1"])
        E2 = float(section["E2"])
    except KeyError as e:
        logging.error("Missing param in %s: %s", calc_name, e)
        return
    except ValueError as e:
        logging.error("Invalid numeric param in %s: %s", calc_name, e)
        return

    with output_file.open("a") as f:
        f.write("\n")
        f.write(f" JOB: {calc_name}\n")

    corr1, corr2 = USTE2.dynamic_correlation(method, basis1, basis2, E1, E2)
    writer.write_result(str(output_file), calc_name, {"basis1": basis1, "basis2": basis2}, None, corr1 + corr2, corr1 + corr2)


def run_uspe(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
    if USPE is None:
        logging.error("USPE module not found; skipping %s", calc_name)
        return
    try:
        method = section["method"]
        E1 = float(section["E1"])
        E2 = float(section["E2"])
        E3 = float(section["E3"])
    except KeyError as e:
        logging.error("Missing param in %s: %s", calc_name, e)
        return
    except ValueError as e:
        logging.error("Invalid numeric param in %s: %s", calc_name, e)
        return

    with output_file.open("a") as f:
        f.write("\n")
        f.write(f" JOB: {calc_name}\n")

    EHF, dc, CBS = USPE.CBS_extrapolation(method, E1, E2, E3)
    writer.write_result(str(output_file), calc_name, {"basis_set": method}, EHF, dc, CBS)


def run_tensorial(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
    if TP is None:
        logging.error("TensorialProperties module not found; skipping %s", calc_name)
        return
    try:
        basis1 = section["basis1"]
        basis2 = section["basis2"]
        basis3 = section["basis3"]
        basis4 = section["basis4"]
        labels = section.get("labels", None)
    except KeyError as e:
        logging.error("Missing param in %s: %s", calc_name, e)
        return

    with output_file.open("a") as f:
        f.write("\n")
        f.write(f" JOB: {calc_name}\n")

    TP.run_tensorial(basis1, basis2, basis3, basis4, labels, output_file, calc_name)


def main():
    BANNER = "\n".join([
        """
        
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

    ])
    INFO = "\n".join([

        """
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

    ])
    print(BANNER)
    print(INFO)

    parser = build_cli()
    args = parser.parse_args()

    setup_logging(args.verbose, logfile=Path("pycbs.log"))
    log = logging.getLogger(__name__)

    input_path: Path = args.input
    output_file: Path = args.output

    if not input_path.exists():
        log.error("Input file '%s' does not exist.", input_path)
        sys.exit(1)

    config = read_config(input_path)

    # prepare output file and header via writer
    output_file.write_text("")
    writer.write_header(str(output_file))
    # add reproducibility metadata
    try:
        import pyscf
        pyscf_version = getattr(pyscf, "__version__", "unknown")
    except Exception:
        pyscf_version = "unknown"
    with output_file.open("a") as f:
        f.write("\nRun metadata:\n")
        f.write(f"  Platform: {platform.platform()}\n")
        f.write(f"  Python: {platform.python_version()}\n")
        f.write(f"  PySCF: {pyscf_version}\n\n")

    # read OPTIMIZATION section
    opt_enabled = False
    opt_params: Dict[str, Any] = {}
    if config.has_section("OPTIMIZATION"):
        sec = config["OPTIMIZATION"]
        opt_enabled = parse_bool(sec.get("optimization", "False"))
        opt_params = gather_optimization_params(sec)
        log.info("OPTIMIZATION section detected: enabled=%s", opt_enabled)

        # === Handle user-specified basis sets ===
        orig_basis1, orig_basis2 = opt_params["basis_sets"][0], opt_params["basis_sets"][1]

        # Define mapping from friendly names to PySCF names
        pyscf_basis_map = {
            'VDZ': 'cc-pvdz', 'VTZ': 'cc-pvtz', 'VQZ': 'cc-pvqz', 'V5Z': 'cc-pv5z', 'V6Z': 'cc-pv6z',
            'AVDZ': 'aug-cc-pvdz', 'AVTZ': 'aug-cc-pvtz', 'AVQZ': 'aug-cc-pvqz', 'AV5Z': 'aug-cc-pv5z', 'AV6Z': 'aug-cc-pv6z',
            'd-AVDZ': 'd-aug-cc-pvdz', 'd-AVTZ': 'd-aug-cc-pvtz', 'd-AVQZ': 'd-aug-cc-pvqz', 'd-AV5Z': 'd-aug-cc-pv5z',
            'VDZ-F12': 'cc-pvdz-f12', 'VTZ-F12': 'cc-pvtz-f12', 'VQZ-F12': 'cc-pvqz-f12'
        }
        # Inverse map for inputs that may already be in PySCF format (lowercase keys)
        friendly_from_pyscf = {v.lower(): k for k, v in pyscf_basis_map.items()}

        # Determine friendly keys for each original basis name
        def get_friendly_and_pyscf(name: str):
            key = None
            name_stripped = name.strip()
            lower = name_stripped.lower()
            if lower in friendly_from_pyscf:
                key = friendly_from_pyscf[lower]
            elif name_stripped.upper() in basis.hf:
                key = name_stripped.upper()
            else:
                key = None
            pyscf_name = pyscf_basis_map.get(key, name_stripped) if key else name_stripped
            return key, pyscf_name

        key1, pyscf1 = get_friendly_and_pyscf(orig_basis1)
        key2, pyscf2 = get_friendly_and_pyscf(orig_basis2)

        # Retrieve hierarchical exponents (or defaults if missing)
        x1 = basis.dc3.get(key1) if key1 in basis.dc3 else None
        x2 = basis.dc3.get(key2) if key2 in basis.dc3 else None
        x1_hf = basis.hf.get(key1) if key1 in basis.hf else None
        x2_hf = basis.hf.get(key2) if key2 in basis.hf else None

        missing = False
        notes = []
        # Check for missing entries and issue warnings
        if x1 is None or x2 is None or x1_hf is None or x2_hf is None:
            missing = True
            log.warning("Some hierarchical values for basis sets '%s', '%s' were not found in lookup tables.",
                        orig_basis1, orig_basis2)
            notes.append("Missing hierarchical values replaced by defaults.")

        # If user manually provided x1, x2, x1_hf, x2_hf, ignore them
        if any(k in sec for k in ("x1", "x2", "x1_hf", "x2_hf")):
            log.info("User-provided exponent values (x1, x2, x1_hf, x2_hf) will be ignored; using dictionary values.")

        # Update opt_params with PySCF basis names and dictionary exponents
        opt_params["basis_sets"] = [pyscf1, pyscf2]
        # Use defaults from OPT_DEFAULTS if any value was missing
        opt_params["x1"] = x1 if x1 is not None else OPT_DEFAULTS["x1"]
        opt_params["x2"] = x2 if x2 is not None else OPT_DEFAULTS["x2"]
        opt_params["x1_hf"] = x1_hf if x1_hf is not None else OPT_DEFAULTS["x1_hf"]
        opt_params["x2_hf"] = x2_hf if x2_hf is not None else OPT_DEFAULTS["x2_hf"]

        # Summarize to terminal
        print("\n" + "="*60)
        print(" BASIS SETS SUMMARY ")
        print("="*60)
        print(f"Original basis sets: {orig_basis1}, {orig_basis2}")
        print(f"PySCF basis sets: {pyscf1}, {pyscf2}")
        print(f"x1 (from dc3) for {orig_basis1}: {opt_params['x1']}")
        print(f"x2 (from dc3) for {orig_basis2}: {opt_params['x2']}")
        print(f"x1_hf (from hf) for {orig_basis1}: {opt_params['x1_hf']}")
        print(f"x2_hf (from hf) for {orig_basis2}: {opt_params['x2_hf']}")
        if missing:
            print("⚠️ WARNING: Some values were missing and defaults were used.")
        print("="*60 + "\n")

        # Write summary into output file
        with output_file.open("a") as f:
            f.write("="*60 + "\n")
            f.write(" BASIS SETS SUMMARY \n")
            f.write("="*60 + "\n")
            f.write(f"Original basis sets: {orig_basis1}, {orig_basis2}\n")
            f.write(f"PySCF basis sets: {pyscf1}, {pyscf2}\n")
            f.write(f"x1 (dc3) for {orig_basis1}: {opt_params['x1']}\n")
            f.write(f"x2 (dc3) for {orig_basis2}: {opt_params['x2']}\n")
            f.write(f"x1_hf (hf) for {orig_basis1}: {opt_params['x1_hf']}\n")
            f.write(f"x2_hf (hf) for {orig_basis2}: {opt_params['x2_hf']}\n")
            if missing:
                f.write("NOTE: Some hierarchical values were missing; defaults were used.\n")
            f.write("="*60 + "\n\n")
    else:
        log.debug("No OPTIMIZATION section in input file; optimizer not requested.")

    # override workers from CLI if provided
    if args.workers is not None:
        opt_params["workers"] = args.workers

    # call optimizer (if requested)
    optimizer_result = None
    if opt_enabled and not args.no_opt:
        log.info("Optimization requested — launching optimizer...")
        try:
            import optimization as optmod  # type: ignore
        except Exception as e:
            log.error("Could not import optimization module: %s", e)
        else:
            run_opt = getattr(optmod, "run_optimization", None)
            if callable(run_opt):
                try:
                    log.info("Running optimizer (PySCF output suppressed)...")
                    with silence_output():
                        optimizer_result = run_opt(opt_params, output_file=str(output_file))
                    log.info("Optimizer finished.")
                except Exception as e:
                    log.exception("Error while running optimizer: %s", e)
            else:
                log.error("optimization.run_optimization(...) not found in optimization module.")

        # Write optimization summary to output file
        if isinstance(optimizer_result, dict):
            history = optimizer_result.get("history")
            final_params = optimizer_result.get("parameters")
            final_energy = optimizer_result.get("energy")
            converged = optimizer_result.get("converged", False)

            if history:
                try:
                    writer.write_optimization_summary(str(output_file), history)
                except Exception as e:
                    log.exception("Failed writing optimization summary to results: %s", e)

            # Write final result block in writer format for compatibility
            try:
                writer.write_result(str(output_file), "OPTIMIZATION", {
                    "method": opt_params.get("METHOD", OPT_DEFAULTS["METHOD"]),
                    "basis_sets": ",".join(opt_params.get("basis_sets", OPT_DEFAULTS["basis_sets"])),
                    "init_parameters": ",".join(map(str, opt_params.get("init_parameters", OPT_DEFAULTS["init_parameters"])))
                }, EHF=None, dc=None, energy=final_energy)
            except Exception:
                pass

    # Process remaining scheme sections (USTE1, USTE2, USPE, TENSORIAL)
    print('='*60)
    print(" CALCULATIONS STATUS ")
    print('='*60)
    calculation_counter = 0
    for section_name in config.sections():
        if section_name.upper() == "OPTIMIZATION":
            continue

        section = config[section_name]
        scheme = section.get("scheme", "").upper()

        try:
            if scheme == "USTE1":
                run_uste1(section, output_file, section_name)
            elif scheme == "USTE2":
                run_uste2(section, output_file, section_name)
            elif scheme == "USPE":
                run_uspe(section, output_file, section_name)
            elif scheme == "TENSORIAL":
                run_tensorial(section, output_file, section_name)
            else:
                print(f"⚠️  Unknown scheme '{scheme}' in section [{section_name}] — skipping.")
                continue
        except Exception as e:
            logging.getLogger(__name__).exception("Error processing section %s: %s", section_name, e)

        calculation_counter += 1
        print(f"Calculation {calculation_counter} done.")

    # final summary table
    try:
        writer.write_summary_table(str(output_file))
    except Exception:
        logging.getLogger(__name__).exception("Failed to write summary table using writer")

    try:
        abs_path = output_file.resolve()
    except Exception:
        abs_path = output_file
    print('='*60)
    print(f"\nResults saved to: {abs_path}")
    print('='*60)


if __name__ == "__main__":
    main()
