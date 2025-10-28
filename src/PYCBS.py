#!/usr/bin/env python3
"""
PYCBS.py - Improved entrypoint for pyCBS

Minimal terminal output, writes full results (including optimization cycle history)
into results.out via writer.
"""
from __future__ import annotations

import argparse
import configparser
import json
import logging
import platform
import sys
import time
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Dict, Optional

import writer

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
                logging.warning("Could not parse optimization param '%s' value '%s': %s. Using default %s",
                                key, section.get(key), e, params.get(key))
    # normalize
    if isinstance(params.get("init_parameters"), list) and len(params["init_parameters"]) >= 2:
        params["init_parameters"] = [float(params["init_parameters"][0]), float(params["init_parameters"][1])]
    else:
        params["init_parameters"] = list(OPT_DEFAULTS["init_parameters"])
    if isinstance(params.get("basis_sets"), list) and len(params["basis_sets"]) >= 2:
        params["basis_sets"] = [params["basis_sets"][0], params["basis_sets"][1]]
    else:
        params["basis_sets"] = list(OPT_DEFAULTS["basis_sets"])
    return params


from contextlib import redirect_stdout, redirect_stderr
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

    hf_dict, corr_dict = USTE1.dictionaries(method, basis1, basis2)
    Ecr1, Ecr2 = USTE1.correlation_frequency(HF1, HF2, E1, E2)
    EHF, dc, CBS = USTE1.CBS_extrapolation(HF1, HF2, Ecr1, Ecr2, corr_dict, basis1, basis2)

    result_data = {
        "method": method,
        "basis1": basis1,
        "basis2": basis2,
        "HF1": HF1,
        "HF2": HF2,
        "E1": E1,
        "E2": E2,
    }
    writer.write_result(str(output_file), "USTE1", result_data, EHF, dc, CBS)


def run_uste2(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
    if USTE2 is None:
        logging.error("USTE2 module not found; skipping %s", calc_name)
        return
    try:
        method = section["method"]
        basis1 = section["basis1"]
        basis2 = section["basis2"]
        basis3 = section["basis3"]
        basis4 = section["basis4"]
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

    hf_dict, corr_dict = USTE2.dictionaries(method, basis1, basis2, basis3, basis4)
    Ecr1, Ecr2 = USTE2.correlation_energy(HF1, HF2, E1, E2)
    EHF, dc, CBS = USTE2.CBS_extrapolation(HF1, HF2, Ecr1, Ecr2, corr_dict, basis1, basis2, basis3, basis4)

    result_data = {
        "method": method,
        "basis1": basis1,
        "basis2": basis2,
        "basis3": basis3,
        "basis4": basis4,
        "HF1": HF1,
        "HF2": HF2,
        "E1": E1,
        "E2": E2,
    }
    writer.write_result(str(output_file), "USTE2", result_data, EHF, dc, CBS)


def run_uspe(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
    if USPE is None:
        logging.error("USPE module not found; skipping %s", calc_name)
        return
    try:
        method = section["method"]
        constant = section["constant"]
        basis = section["basis"]
        HF = float(section["HF"])
        Etot = float(section["Etot"])
    except KeyError as e:
        logging.error("Missing param in %s: %s", calc_name, e)
        return
    except ValueError as e:
        logging.error("Invalid numeric param in %s: %s", calc_name, e)
        return

    with output_file.open("a") as f:
        f.write("\n")
        f.write(f" JOB: {calc_name}\n")

    resultado = USPE.CBS_extrapolation(HF, Etot, method, constant, basis)

    result_data = {"method": method, "constant": constant, "basis": basis, "HF": HF, "Etot": Etot}
    writer.write_result(str(output_file), "USPE", result_data, energy=resultado)


def run_tensorial(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
    if TP is None:
        logging.error("tensorial_properties1 not found; skipping %s", calc_name)
        return
    try:
        method = section["method"]
        basis1 = section["basis1"]
        basis2 = section["basis2"]
        zeta_HF1 = float(section["zeta_HF1"])
        zeta_HF2 = float(section["zeta_HF2"])
        zeta_E1 = float(section["zeta_E1"])
        zeta_E2 = float(section["zeta_E2"])
    except KeyError as e:
        logging.error("Missing param in %s: %s", calc_name, e)
        return
    except ValueError as e:
        logging.error("Invalid numeric param in %s: %s", calc_name, e)
        return

    with output_file.open("a") as f:
        f.write("\n")
        f.write(f" JOB: {calc_name}\n")

    hf_dict, corr_dict = TP.dictionaries(method, basis1, basis2)
    zeta_cor1, zeta_cor2 = TP.correlation_energy(zeta_HF1, zeta_HF2, zeta_E1, zeta_E2)
    zeta_HF, zeta_cor, zeta = TP.CBS_extrapolation(zeta_HF1, zeta_HF2, zeta_cor1, zeta_cor2, corr_dict, basis1, basis2)

    result_data = {
        "method": method,
        "basis1": basis1,
        "basis2": basis2,
        "zeta_HF1": zeta_HF1,
        "zeta_HF2": zeta_HF2,
        "zeta_E1": zeta_E1,
        "zeta_E2": zeta_E2,
    }
    writer.write_result(str(output_file), "TENSORIAL", result_data, zeta_HF, zeta_cor, zeta)


def main() -> None:
    BANNER = r"""
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

    INFO = """
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
                    # suppress PySCF chatter while optimizer runs
                    with silence_output():
                        optimizer_result = run_opt(opt_params, output_file=str(output_file))
                    log.info("Optimizer finished.")
                except Exception as e:
                    log.exception("Error while running optimizer: %s", e)
            else:
                log.error("optimization.run_optimization(...) not found in optimization module.")

        # If optimizer returned the history, write it to the results file using writer
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

            # write final optimization block (keeps compatibility with writer.write_result)
            try:
                writer.write_result(str(output_file), "OPTIMIZATION", {
                    "method": opt_params.get("METHOD", OPT_DEFAULTS["METHOD"]),
                    "basis_sets": ",".join(opt_params.get("basis_sets", OPT_DEFAULTS["basis_sets"])),
                    "init_parameters": ",".join(map(str, opt_params.get("init_parameters", OPT_DEFAULTS["init_parameters"])))
                }, EHF=None, dc=None, energy=final_energy)
            except Exception:
                pass
    # Process remaining sections
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

    # final summary
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

    print('=' * 60)


if __name__ == "__main__":
    main()
