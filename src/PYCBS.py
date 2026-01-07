#!/usr/bin/env python3
"""
pyCBS Main Entry Point - Complete redesign with proper module discovery
"""

import argparse
import configparser
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Local imports (use relative imports within package)
from src.pycbs import writer, basis
from src.pycbs.config_validator import ConfigValidator
from src.pycbs.module_loader import SchemeModuleLoader

logger = logging.getLogger(__name__)


def setup_logging(verbosity: int = 0) -> None:
    """Configure logging system"""
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


def read_config(input_path: Path) -> configparser.ConfigParser:
    """Read and parse INI configuration file"""
    if not input_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {input_path}")

    cfg = configparser.ConfigParser()
    cfg.optionxform = str  # Preserve case
    cfg.read(input_path)

    if not cfg.sections():
        raise ValueError(f"Configuration file is empty:  {input_path}")

    return cfg


def process_section(section_name: str, section: configparser.SectionProxy,
                    output_file: Path) -> bool:
    """
    Process a single calculation section.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert section to dict for validation
        section_dict = dict(section)

        scheme = section_dict.get('scheme', '').upper().strip()

        # Validate configuration
        is_valid, errors = ConfigValidator.validate_section(section_name, section_dict)
        if not is_valid:
            logger.error(f"Invalid configuration in section [{section_name}]:")
            for error in errors:
                logger.error(f"  - {error}")
                with output_file.open("a") as f:
                    f.write(f"ERROR [{section_name}]: {error}\n")
            return False

        logger.info(f"Processing [{section_name}] with scheme={scheme}")

        # Dispatch to appropriate handler
        if scheme == "USTE1":
            return process_uste1(section, output_file, section_name)
        elif scheme == "USTE2":
            return process_uste2(section, output_file, section_name)
        elif scheme == "USPE":
            return process_uspe(section, output_file, section_name)
        elif scheme == "TENSORIAL":
            return process_tensorial(section, output_file, section_name)
        elif scheme == "FREQUENCY":
            return process_frequency(section, output_file, section_name)
        # Single-function extrapolation schemes
        elif scheme in ['FELLER', 'TRUHLAR_HF', 'KLOPPER', 'JENSEN',
                        'BAKOULES', 'OAN', 'TRUHLAR_CORR', 'MARTIN', 'HALKIER_HELGAKER', 'HUH_LEE']:
            return process_simple_extrapolation(section, scheme, output_file, section_name)
        elif scheme == 'HF_E':
            return process_hf_e(section, output_file, section_name)
        else:
            logger.warning(f"Unknown scheme: {scheme}")
            return False

    except Exception as e:
        logger.exception(f"Error processing section [{section_name}]:  {e}")
        return False


def process_hf_e(section: configparser.SectionProxy, output_file: Path, section_name: str) -> bool:
    """Process HF_E (Hartree-Fock energy) calculation"""
    try:
        basis1 = section.get('basis1', '').strip()
        basis2 = section.get('basis2', '').strip()
        HF1 = float(section.get('HF1'))
        HF2 = float(section.get('HF2'))

        # Load module
        module = SchemeModuleLoader.load_module('HF_E')
        if module is None:
            logger.error(f"Could not load HF_E module")
            return False

        # Call hartree_fock_energy function
        fn = getattr(module, 'hartree_fock_energy', None)
        if not callable(fn):
            logger.error(f"HF_E module missing 'hartree_fock_energy' function")
            return False

        result = fn(HF1, HF2, basis1, basis2)

        with output_file.open("a") as f:
            f.write(f"\n{'=' * 70}\n")
            f.write(f"JOB: {section_name}\n")
            f.write(f"Scheme: HF_E\n")
            f.write(f"Basis sets: {basis1}, {basis2}\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"HF Energy (CBS): {result:20.12f}\n")

        logger.info(f"✓ [{section_name}] completed successfully")
        return True

    except Exception as e:
        logger.exception(f"Error in HF_E processing [{section_name}]: {e}")
        return False


def process_simple_extrapolation(section: configparser.SectionProxy, scheme: str,
                                 output_file: Path, section_name: str) -> bool:
    """
    Process simple single-function extrapolation schemes.

    Handles schemes like FELLER, OAN, TRUHLAR_CORR, etc.  that have a single
    extrapolation function with parameters (Ehf_X, Ehf_Y, Ec_X, Ec_Y, X, Y, alfa, beta).
    """
    try:
        section_dict = dict(section)

        # Parse all available parameters (they're optional except the required ones)
        Ehf_X = float(section_dict.get('Ehf_X', 'nan')) if 'Ehf_X' in section_dict else None
        Ehf_Y = float(section_dict.get('Ehf_Y', 'nan')) if 'Ehf_Y' in section_dict else None
        Ec_X = float(section_dict.get('Ec_X', 'nan')) if 'Ec_X' in section_dict else None
        Ec_Y = float(section_dict.get('Ec_Y', 'nan')) if 'Ec_Y' in section_dict else None
        X = float(section_dict.get('X', 'nan')) if 'X' in section_dict else None
        Y = float(section_dict.get('Y', 'nan')) if 'Y' in section_dict else None
        alfa = float(section_dict.get('alfa', 'nan')) if 'alfa' in section_dict else None
        beta = float(section_dict.get('beta', 'nan')) if 'beta' in section_dict else None

        # Replace 'nan' with None for cleaner logic
        Ehf_X = None if (Ehf_X is not None and str(Ehf_X) == 'nan') else Ehf_X
        Ehf_Y = None if (Ehf_Y is not None and str(Ehf_Y) == 'nan') else Ehf_Y
        Ec_X = None if (Ec_X is not None and str(Ec_X) == 'nan') else Ec_X
        Ec_Y = None if (Ec_Y is not None and str(Ec_Y) == 'nan') else Ec_Y
        X = None if (X is not None and str(X) == 'nan') else X
        Y = None if (Y is not None and str(Y) == 'nan') else Y
        alfa = None if (alfa is not None and str(alfa) == 'nan') else alfa
        beta = None if (beta is not None and str(beta) == 'nan') else beta

        # Call the extrapolation function using the loader
        result = SchemeModuleLoader.call_extrapolation_function(
            scheme,
            Ehf_X=Ehf_X,
            Ehf_Y=Ehf_Y,
            Ec_X=Ec_X,
            Ec_Y=Ec_Y,
            X=X,
            Y=Y,
            alfa=alfa,
            beta=beta,
        )

        with output_file.open("a") as f:
            f.write(f"\n{'=' * 70}\n")
            f.write(f"JOB: {section_name}\n")
            f.write(f"Scheme: {scheme}\n")

            # Write input parameters
            if Ehf_X is not None:
                f.write(f"Ehf_X: {Ehf_X: 20.12f}\n")
            if Ehf_Y is not None:
                f.write(f"Ehf_Y: {Ehf_Y: 20.12f}\n")
            if Ec_X is not None:
                f.write(f"Ec_X: {Ec_X:20.12f}\n")
            if Ec_Y is not None:
                f.write(f"Ec_Y: {Ec_Y:20.12f}\n")
            if X is not None:
                f.write(f"X: {X:20.12f}\n")
            if Y is not None:
                f.write(f"Y: {Y:20.12f}\n")
            if alfa is not None:
                f.write(f"alfa: {alfa:20.12f}\n")
            if beta is not None:
                f.write(f"beta: {beta:20.12f}\n")

            f.write(f"{'=' * 70}\n")
            f.write(f"Extrapolated Result: {result:20.12f}\n")

        logger.info(f"✓ [{section_name}] completed successfully")
        return True

    except Exception as e:
        logger.exception(f"Error in {scheme} processing [{section_name}]: {e}")
        return False


def process_uste1(section: configparser.SectionProxy, output_file: Path, section_name: str) -> bool:
    """Process USTE1 calculation"""
    try:
        method = section.get('method', '').upper().strip()
        basis1 = section.get('basis1', '').strip()
        basis2 = section.get('basis2', '').strip()
        HF1 = float(section.get('HF1'))
        HF2 = float(section.get('HF2'))
        E1 = float(section.get('E1'))
        E2 = float(section.get('E2'))

        module = SchemeModuleLoader.load_module('USTE1')
        if module is None:
            logger.error(f"Could not load USTE1 module")
            return False

        # Call dictionaries function
        dicts_fn = getattr(module, 'dictionaries', None)
        if not callable(dicts_fn):
            logger.error(f"USTE1 module missing 'dictionaries' function")
            return False

        hf_dict, corr_dict = dicts_fn(method, basis1, basis2)

        # Calculate correlation energy
        corr_fn = getattr(module, 'correlation_energy', None)
        if callable(corr_fn):
            Ecr1, Ecr2 = corr_fn(HF1, HF2, E1, E2)
        else:
            Ecr1 = E1 - HF1
            Ecr2 = E2 - HF2

        # Calculate CBS extrapolation
        cbs_fn = getattr(module, 'CBS_extrapolation', None)
        if not callable(cbs_fn):
            logger.error(f"USTE1 module missing 'CBS_extrapolation' function")
            return False

        EHF, dc, CBS = cbs_fn(HF1, HF2, Ecr1, Ecr2, corr_dict, basis1, basis2)

        with output_file.open("a") as f:
            f.write(f"\n{'=' * 70}\n")
            f.write(f"JOB: {section_name}\n")
            f.write(f"Scheme: USTE1 | Method: {method}\n")
            f.write(f"Basis sets: {basis1}, {basis2}\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"HF (CBS):                   {EHF:20.12f}\n")
            f.write(f"Dynamic Correlation:        {dc:20.12f}\n")
            f.write(f"Total CBS Energy:           {CBS:20.12f}\n")

        logger.info(f"✓ [{section_name}] completed successfully")
        return True

    except Exception as e:
        logger.exception(f"Error in USTE1 processing [{section_name}]: {e}")
        return False


def process_uste2(section: configparser.SectionProxy, output_file: Path, section_name: str) -> bool:
    """Process USTE2 calculation"""
    try:
        method = section.get('method', '').upper().strip()
        basis1 = section.get('basis1', '').strip()
        basis2 = section.get('basis2', '').strip()
        basis3 = section.get('basis3', basis1).strip()
        basis4 = section.get('basis4', basis2).strip()
        HF1 = float(section.get('HF1'))
        HF2 = float(section.get('HF2'))
        E1 = float(section.get('E1'))
        E2 = float(section.get('E2'))

        module = SchemeModuleLoader.load_module('USTE2')
        if module is None:
            logger.error(f"Could not load USTE2 module")
            return False

        dicts_fn = getattr(module, 'dictionaries', None)
        if not callable(dicts_fn):
            logger.error(f"USTE2 module missing 'dictionaries' function")
            return False

        hf_dict, corr_dict = dicts_fn(method, basis1, basis2, basis3, basis4)

        corr_fn = getattr(module, 'correlation_energy', None)
        if callable(corr_fn):
            Ecr1, Ecr2 = corr_fn(HF1, HF2, E1, E2)
        else:
            Ecr1 = E1 - HF1
            Ecr2 = E2 - HF2

        cbs_fn = getattr(module, 'CBS_extrapolation', None)
        if not callable(cbs_fn):
            logger.error(f"USTE2 module missing 'CBS_extrapolation' function")
            return False

        EHF, dc, CBS = cbs_fn(HF1, HF2, Ecr1, Ecr2, corr_dict, basis1, basis2, basis3, basis4)

        with output_file.open("a") as f:
            f.write(f"\n{'=' * 70}\n")
            f.write(f"JOB: {section_name}\n")
            f.write(f"Scheme:  USTE2 | Method: {method}\n")
            f.write(f"Basis sets (HF): {basis1}, {basis2}\n")
            f.write(f"Basis sets (Corr): {basis3}, {basis4}\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"HF (CBS):                   {EHF:20.12f}\n")
            f.write(f"Dynamic Correlation:        {dc:20.12f}\n")
            f.write(f"Total CBS Energy:           {CBS:20.12f}\n")

        logger.info(f"✓ [{section_name}] completed successfully")
        return True

    except Exception as e:
        logger.exception(f"Error in USTE2 processing [{section_name}]: {e}")
        return False


def process_uspe(section: configparser.SectionProxy, output_file: Path, section_name: str) -> bool:
    """Process USPE calculation"""
    try:
        method = section.get('method', '').upper().strip()
        basis = section.get('basis', '').strip()
        HF = float(section.get('HF'))
        Etot = float(section.get('Etot'))
        constant = section.get('constant', 'normal').lower().strip()

        module = SchemeModuleLoader.load_module('USPE')
        if module is None:
            logger.error(f"Could not load USPE module")
            return False

        uspe_fn = getattr(module, 'USPE_CBS_extrapolation', None)
        if not callable(uspe_fn):
            logger.error(f"USPE module missing 'USPE_CBS_extrapolation' function")
            return False

        result = uspe_fn(HF, HF, Etot, method, constant, basis, basis)

        with output_file.open("a") as f:
            f.write(f"\n{'=' * 70}\n")
            f.write(f"JOB: {section_name}\n")
            f.write(f"Scheme: USPE | Method: {method}\n")
            f.write(f"Basis:  {basis} | Constant type: {constant}\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"Result:  {result:20.12f}\n")

        logger.info(f"✓ [{section_name}] completed successfully")
        return True

    except Exception as e:
        logger.exception(f"Error in USPE processing [{section_name}]: {e}")
        return False


def process_tensorial(section: configparser.SectionProxy, output_file: Path, section_name: str) -> bool:
    """Process TENSORIAL calculation"""
    try:
        method = section.get('method', '').upper().strip()
        basis1 = section.get('basis1', '').strip()
        basis2 = section.get('basis2', basis1).strip()
        dc_scheme = section.get('dc_scheme', 'USTE1').upper().strip()

        module = SchemeModuleLoader.load_module('TENSORIAL')
        if module is None:
            logger.error(f"Could not load TENSORIAL module")
            return False

        if dc_scheme == 'USPE':
            zeta_HF1 = float(section.get('zeta_HF1'))
            zeta_E1 = float(section.get('zeta_E1'))
            constant = section.get('constant', 'normal').lower().strip()

            uspe_fn = getattr(module, 'USPE_CBS_extrapolation', None)
            if callable(uspe_fn):
                result = uspe_fn(zeta_HF1, zeta_HF1, zeta_E1, method, constant, basis1, basis2)
            else:
                result = zeta_E1
        else:  # USTE1
            zeta_HF1 = float(section.get('zeta_HF1'))
            zeta_HF2 = float(section.get('zeta_HF2'))
            zeta_E1 = float(section.get('zeta_E1'))
            zeta_E2 = float(section.get('zeta_E2'))

            dicts_fn = getattr(module, 'dictionaries', None)
            if not callable(dicts_fn):
                logger.error(f"TENSORIAL module missing 'dictionaries' function")
                return False

            hf_dict, corr_dict = dicts_fn(method, basis1, basis2)

            corr_fn = getattr(module, 'correlation_energy', None)
            if callable(corr_fn):
                Ecr1, Ecr2 = corr_fn(zeta_HF1, zeta_HF2, zeta_E1, zeta_E2)
            else:
                Ecr1 = zeta_E1 - zeta_HF1
                Ecr2 = zeta_E2 - zeta_HF2

            uste_fn = getattr(module, 'USTE_CBS_extrapolation', None)
            if callable(uste_fn):
                result = uste_fn(zeta_HF1, zeta_HF2, Ecr1, Ecr2, corr_dict, basis1, basis2)
            else:
                result = zeta_E1

        with output_file.open("a") as f:
            f.write(f"\n{'=' * 70}\n")
            f.write(f"JOB:  {section_name}\n")
            f.write(f"Scheme: TENSORIAL | DC Scheme: {dc_scheme} | Method: {method}\n")
            f.write(f"Basis sets: {basis1}, {basis2}\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"Result: {result}\n")

        logger.info(f"✓ [{section_name}] completed successfully")
        return True

    except Exception as e:
        logger.exception(f"Error in TENSORIAL processing [{section_name}]: {e}")
        return False


def process_frequency(section: configparser.SectionProxy, output_file: Path, section_name: str) -> bool:
    """Process FREQUENCY calculation"""
    try:
        method = section.get('method', '').upper().strip()
        basis1 = section.get('basis1', '').strip()
        basis2 = section.get('basis2', '').strip()
        HF1 = float(section.get('HF1'))
        HF2 = float(section.get('HF2'))
        F1 = float(section.get('F1', section.get('E1')))
        F2 = float(section.get('F2', section.get('E2')))

        module = SchemeModuleLoader.load_module('FREQUENCY')
        if module is None:
            logger.error(f"Could not load FREQUENCY module")
            return False

        dicts_fn = getattr(module, 'dictionaries', None)
        if not callable(dicts_fn):
            logger.error(f"FREQUENCY module missing 'dictionaries' function")
            return False

        hf_dict, corr_dict = dicts_fn(method, basis1, basis2)

        freq_fn = getattr(module, 'correlation_frequency', None)
        if callable(freq_fn):
            Fcr1, Fcr2 = freq_fn(HF1, HF2, F1, F2)
        else:
            Fcr1 = F1 - HF1
            Fcr2 = F2 - HF2

        cbs_fn = getattr(module, 'CBS_extrapolation', None)
        if not callable(cbs_fn):
            logger.error(f"FREQUENCY module missing 'CBS_extrapolation' function")
            return False

        EHF, dc, CBS = cbs_fn(HF1, HF2, Fcr1, Fcr2, corr_dict, basis1, basis2)

        with output_file.open("a") as f:
            f.write(f"\n{'=' * 70}\n")
            f.write(f"JOB: {section_name}\n")
            f.write(f"Scheme: FREQUENCY | Method: {method}\n")
            f.write(f"Basis sets: {basis1}, {basis2}\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"HF (CBS):                   {EHF:20.12f}\n")
            f.write(f"Dynamic Correlation:        {dc:20.12f}\n")
            f.write(f"Total CBS Energy:           {CBS:20.12f}\n")

        logger.info(f"✓ [{section_name}] completed successfully")
        return True

    except Exception as e:
        logger.exception(f"Error in FREQUENCY processing [{section_name}]: {e}")
        return False


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog="pycbs",
        description="pyCBS - Complete Basis Set Extrapolation Tool"
    )
    parser.add_argument(
        "-input", "--input",
        type=Path,
        default=Path("inputfile.inp"),
        help="Path to configuration file (INI format)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("results.out"),
        help="Output results file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv)"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger.info(f"Starting pyCBS calculation")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")

    # Read configuration
    try:
        config = read_config(args.input)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Initialize output file
    args.output.write_text("")
    writer.write_header(str(args.output))

    # Process each calculation section
    total_calcs = 0
    successful_calcs = 0

    for section_name in config.sections():
        if section_name.upper() == "OPTIMIZATION":
            logger.warning("OPTIMIZATION section found but not yet implemented")
            continue

        total_calcs += 1
        if process_section(section_name, config[section_name], args.output):
            successful_calcs += 1

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Calculations completed: {successful_calcs}/{total_calcs}")
    logger.info(f"Results saved to: {args.output.resolve()}")
    logger.info(f"{'=' * 60}\n")

    sys.exit(0 if successful_calcs == total_calcs else 1)


if __name__ == "__main__":
    main()