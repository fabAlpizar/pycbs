#!/usr/bin/env python3
"""
pyCBS CLI entry point
"""

import argparse
import configparser
import logging
import sys
from pathlib import Path

# Package imports
from . import writer
from .config_validator import ConfigValidator
from .module_loader import SchemeModuleLoader

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

def setup_logging(verbosity: int = 0) -> None:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


# ----------------------------------------------------------------------
# Central execution entry point
# ----------------------------------------------------------------------

def run(params: dict):
    """
    Execute a CBS scheme based on validated parameters.

    This is the single execution gateway used by the CLI.
    """
    scheme = params["scheme"]
    module = SchemeModuleLoader.load_scheme(scheme)

    if not hasattr(module, "compute"):
        raise AttributeError(
            f"Scheme '{scheme}' does not expose a compute(params) function"
        )

    return module.compute(params)


# ----------------------------------------------------------------------
# Config handling
# ----------------------------------------------------------------------

def read_config(input_path: Path) -> configparser.ConfigParser:
    if not input_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {input_path}")

    cfg = configparser.ConfigParser()
    cfg.optionxform = str  # preserve case
    cfg.read(input_path)

    if not cfg.sections():
        raise ValueError(f"Configuration file is empty: {input_path}")

    return cfg


# ----------------------------------------------------------------------
# Section dispatcher
# ----------------------------------------------------------------------

def normalize_params(raw: dict) -> dict:
    """
    Convert configparser string values into typed params suitable for compute().
    - Keys are normalized to lowercase (case-insensitive input).
    - 'scheme' and 'method' values are uppercased.
    - Basis names and labels preserve case.
    - Numeric-looking values are converted to int or float.
    - Empty strings and None-like values become None.
    """
    params = {}

    for k, v in raw.items():
        key = k.strip().lower()

        if v is None:
            val = None
        else:
            val = v.strip()

        # Normalize empty / None-like values
        if isinstance(val, str) and val.lower() in {"", "none", "null"}:
            params[key] = None
            continue

        # Scheme and method → uppercase values
        if key in {"scheme", "method"}:
            params[key] = val.upper() if isinstance(val, str) else val
            continue

        # BASIS names and metadata → preserve case
        if key.startswith("basis") or key in {"constant", "comment", "label"}:
            params[key] = val
            continue

        # Try numeric coercion
        if isinstance(val, str):
            try:
                f = float(val)
                params[key] = int(f) if f.is_integer() else f
            except ValueError:
                params[key] = val
        else:
            params[key] = val

    return params


def process_section(
    section_name: str,
    section: configparser.SectionProxy,
    output_file: Path
) -> bool:
    try:
        # raw: preserve original strings from configparser for validation
        raw = dict(section)

        # Validate raw config (validator is now case-insensitive)
        ok, errors = ConfigValidator.validate_section(section_name, raw)
        if not ok:
            for err in errors:
                writer.write_error(output_file, section_name, err)
            return False

        # Normalize values for computation (lowercase keys, typed numbers)
        params = normalize_params(raw)

        # scheme is normalized to uppercase inside normalize_params
        scheme = params.get("scheme", "")
        if not scheme:
            raise ValueError(f"[{section_name}] Missing required key: scheme")

        logger.info(f"Processing [{section_name}] | scheme={scheme}")


        # Execute scheme
        result = run(params)

        # Write output
        # Prepare data for writer: use the raw input (strings) so writer prints the
        # parameters exactly as the user supplied them.
        data_for_writer = raw if isinstance(raw, dict) else dict(section)

        # Map compute() result into writer's expected fields:
        EHF = None
        dc = None
        energy = None

        # result can be:
        #  - dict with keys like {'EHF':..., 'E_corr':..., 'E_CBS':...}
        #  - tuple/list (EHF, dc, energy)
        #  - scalar (energy)
        if isinstance(result, dict):
            # attempt common keys
            EHF = result.get('EHF') or result.get('E_HF') or result.get('E_hf') or result.get('zeta_HF') or result.get('zeta_hf')
            dc = result.get('E_corr') or result.get('dc') or result.get('dynamic_corr') or result.get('zeta_cor')
            energy = result.get('E_CBS') or result.get('energy') or result.get('total') or result.get('E_total')
        elif isinstance(result, (list, tuple)):
            if len(result) == 3:
                EHF, dc, energy = result
            elif len(result) == 2:
                # ambiguous: assume (dc, energy)
                dc, energy = result
            else:
                energy = result[0]
        else:
            # scalar
            energy = result

        # Finally call writer with the standardized signature.
        # writer expects: filename (str), scheme (str), data (dict of inputs), EHF, dc, energy
        writer.write_result(str(output_file), scheme, data_for_writer, EHF=EHF, dc=dc, energy=energy)


        return True

    except Exception as e:
        logger.exception(f"Error in [{section_name}]")
        writer.write_error(output_file, section_name, str(e))
        return False


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pycbs",
        description="pyCBS – Complete Basis Set Extrapolation Tool"
    )
    parser.add_argument(
        "-input", "--input",
        type=Path,
        required=True,
        help="Path to input configuration file"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("results.out"),
        help="Output file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv)"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

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
    print(LOGO)
    print(INFO_BLOCK)

    logger.info("Starting pyCBS")
    logger.info(f"Input : {args.input}")
    logger.info(f"Output: {args.output}")

    try:
        config = read_config(args.input)
    except Exception as e:
        logger.error(e)
        sys.exit(1)

    # Initialize output file
    args.output.write_text("")
    writer.write_header(str(args.output))

    total = 0
    success = 0

    for section_name in config.sections():
        if section_name.upper() == "OPTIMIZATION":
            logger.warning("OPTIMIZATION section not yet supported")
            continue

        total += 1
        if process_section(section_name, config[section_name], args.output):
            success += 1

    logger.info(f"Completed {success}/{total} calculations")
    sys.exit(0 if success == total else 1)


if __name__ == "__main__":
    main()
