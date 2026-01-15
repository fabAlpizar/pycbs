#!/usr/bin/env python3
"""
pyCBS CLI entry point
"""

import argparse
import configparser
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

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
    cfg.optionxform = str
    cfg.read(input_path)

    if not cfg.sections():
        raise ValueError(f"Configuration file is empty: {input_path}")

    return cfg


# ----------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------

def normalize_params(raw: dict) -> dict:
    params = {}

    for k, v in raw.items():
        key = k.strip().lower()
        val = v.strip() if isinstance(v, str) else v

        if val is None or (isinstance(val, str) and val.lower() in {"", "none", "null"}):
            params[key] = None
            continue

        if key in {"scheme", "method"}:
            params[key] = val.upper()
            continue

        if key.startswith("basis") or key in {"constant", "comment", "label"}:
            params[key] = val
            continue

        try:
            f = float(val)
            params[key] = int(f) if f.is_integer() else f
        except Exception:
            params[key] = val

    return params


# ----------------------------------------------------------------------
# Section processing
# ----------------------------------------------------------------------

def process_section(
    section_name: str,
    section: configparser.SectionProxy,
    calculations: List[Dict[str, Any]]
) -> bool:
    try:
        raw = dict(section)

        ok, errors = ConfigValidator.validate_section(section_name, raw)
        if not ok:
            for err in errors:
                writer.write_error(None, section_name, err)
            return False

        params = normalize_params(raw)

        scheme = params.get("scheme")
        if not scheme:
            raise ValueError(f"[{section_name}] Missing required key: scheme")

        logger.info(f"Processing [{section_name}] | scheme={scheme}")

        result = run(params)

        record: Dict[str, Any] = {
            "calculation": raw.get("label", section_name),
            "scheme": scheme,
        }

        # ---- result mapping (robust but minimal) ----

        if isinstance(result, dict):
            record["hf_cbs"] = result.get("EHF") or result.get("E_HF")
            record["corr_cbs"] = result.get("E_corr") or result.get("dc")
            record["freq_cbs"] = result.get("freq_cbs")
            record["tens_prop"] = result.get("tens_prop")
            record["total_energy"] = (
                result.get("E_CBS")
                or result.get("E_total")
                or result.get("energy")
            )
            record["property_type"] = result.get("property_type")

        elif isinstance(result, (tuple, list)):
            if len(result) == 3:
                record["hf_cbs"], record["corr_cbs"], record["total_energy"] = result
            elif len(result) == 2:
                record["corr_cbs"], record["total_energy"] = result
            else:
                record["total_energy"] = result[0]

        else:
            record["total_energy"] = result

        calculations.append(record)
        return True

    except Exception as e:
        logger.exception(f"Error in [{section_name}]")
        writer.write_error(None, section_name, str(e))
        return False


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pycbs",
        description="pyCBS – Complete Basis Set Extrapolation Tool"
    )
    parser.add_argument("-input", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, default=Path("results.out"))
    parser.add_argument("-v", "--verbose", action="count", default=0)

    args = parser.parse_args()
    setup_logging(args.verbose)

    print("""
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
    """)

    try:
        config = read_config(args.input)
    except Exception as e:
        logger.error(e)
        sys.exit(1)

    args.output.write_text("")
    writer.write_header(str(args.output))

    calculations: List[Dict[str, Any]] = []

    total = success = 0

    for section_name in config.sections():
        if section_name.upper() == "OPTIMIZATION":
            logger.warning("OPTIMIZATION section not supported")
            continue

        total += 1
        if process_section(section_name, config[section_name], calculations):
            success += 1

    writer.write_reports(
        str(args.output),
        calculations=calculations,
        opt_cycles=[]
    )

    logger.info(f"Completed {success}/{total} calculations")
    sys.exit(0 if success == total else 1)


if __name__ == "__main__":
    main()
