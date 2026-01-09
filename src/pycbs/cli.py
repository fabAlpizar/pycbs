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

def process_section(
    section_name: str,
    section: configparser.SectionProxy,
    output_file: Path
) -> bool:
    try:
        params = dict(section)
        scheme = params.get("scheme", "").strip()

        if not scheme:
            raise ValueError(f"[{section_name}] Missing required key: scheme")

        logger.info(f"Processing [{section_name}] | scheme={scheme}")

        # ----------------------------
        # Validate input
        # ----------------------------
        ok, errors = ConfigValidator.validate_section(section_name, params)
        if not ok:
            for err in errors:
                writer.write_error(output_file, section_name, err)
            return False

        # ----------------------------
        # Execute scheme
        # ----------------------------
        result = run(params)

        # ----------------------------
        # Write output
        # ----------------------------
        writer.write_result(output_file, section_name, scheme, result)

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
