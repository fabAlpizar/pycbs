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

def ensure_parent_dir(path: Path) -> None:
    """Make sure parent directory of a path exists."""
    path.parent.mkdir(parents=True, exist_ok=True)

def unique_path(path: Path) -> Path:
    """
    Return a non-existing Path based on `path`.
    If `path` does not exist return it; otherwise append _1, _2,... before suffix.
    """
    if not path.exists():
        return path
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1
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
    calculations: List[Dict[str, Any]],
    output_file: Path
) -> bool:
    """
    Run and map one section into writer-friendly record dicts.
    Normalizes scheme names and classification sets using the same rule so
    'HUH_LEE' matches 'huh-lee', etc.
    """
    try:
        raw = dict(section)

        ok, errors = ConfigValidator.validate_section(section_name, raw)
        if not ok:
            for err in errors:
                writer.write_error(str(output_file), section_name, err)
            return False

        params = normalize_params(raw)

        scheme = params.get("scheme")
        if not scheme:
            raise ValueError(f"[{section_name}] Missing required key: scheme")

        logger.info(f"Processing [{section_name}] | scheme={scheme}")

        result = run(params)

        logger.debug("Raw compute() result for [%s]: %r", section_name, result)

        record: Dict[str, Any] = {
            "calculation": raw.get("label", section_name),
            "scheme": scheme,
        }

        def _to_float(v):
            if v is None:
                return None
            if isinstance(v, (float, int)) and not isinstance(v, bool):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v)
                except Exception:
                    return None
            return None

        # --- canonical normalizer: lower + replace underscores with hyphens ---
        def _canon(s: str) -> str:
            return str(s).strip().lower().replace("_", "-")

        # Build normalized classification sets from writer defaults
        hf_set = {_canon(s) for s in getattr(writer, "DEFAULT_HF_COMPONENTS", set())}
        corr_set = {_canon(s) for s in getattr(writer, "DEFAULT_CORR_COMPONENTS", set())}
        mixed_set = {_canon(s) for s in getattr(writer, "DEFAULT_MIXED_SCHEMES", set())}

        scheme_low = _canon(scheme)

        # If result is dict-like: normalize keys, flatten one level, and extract candidates
        if isinstance(result, dict):
            res: Dict[str, Any] = {}
            for k, v in result.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        res[f"{k}.{k2}".lower()] = v2
                        res[k2.lower()] = v2
                else:
                    res[k.lower()] = v

            EHF = (
                res.get("ehf")
                or res.get("e_hf")
                or res.get("ehf_cbs")
                or res.get("e_hf_cbs")
                or res.get("hf_cbs")
                or res.get("hf")
            )
            dc = (
                res.get("e_corr")
                or res.get("ecorr")
                or res.get("corr_cbs")
                or res.get("corr")
                or res.get("dc")
                or res.get("dynamic_corr")
            )
            energy = (
                res.get("e_cbs")
                or res.get("ecbs")
                or res.get("e_cbs_total")
                or res.get("e_total")
                or res.get("total")
                or res.get("energy")
                or res.get("e_cbs_energy")
            )
            freq = res.get("freq_cbs") or res.get("frequency") or res.get("freq")
            tens = res.get("tens_prop") or res.get("tensprop") or res.get("tensor")
            prop_hint = (res.get("property_type") or res.get("property") or "").strip().lower()

            EHFf = _to_float(EHF)
            dcf = _to_float(dc)
            energyf = _to_float(energy)
            freqf = _to_float(freq)
            tensf = tens  # may be structured

            # Fallback: search first numeric if nothing explicit found
            if EHFf is None and dcf is None and energyf is None and freqf is None:
                for v in res.values():
                    nv = _to_float(v)
                    if nv is not None:
                        energyf = nv
                        break

            if EHFf is not None:
                record["hf_cbs"] = EHFf
            if dcf is not None:
                record["corr_cbs"] = dcf

            # Special-case mapping: frequency / tensorial
            if scheme_low == "frequency" or prop_hint.startswith("freq"):
                if freqf is not None:
                    record["freq_cbs"] = freqf
                elif energyf is not None:
                    record["freq_cbs"] = energyf
            elif scheme_low == "tensorial" or prop_hint.startswith("tens"):
                if tensf is not None:
                    record["tens_prop"] = tensf
                elif energyf is not None:
                    record["tens_prop"] = energyf
            else:
                # Place energy according to scheme classification
                if energyf is not None:
                    if ("corr_cbs" not in record) and (scheme_low in corr_set):
                        record["corr_cbs"] = energyf
                    elif ("hf_cbs" not in record) and (scheme_low in hf_set):
                        record["hf_cbs"] = energyf
                    else:
                        record["total_energy"] = energyf

            if freqf is not None and "freq_cbs" not in record and scheme_low != "frequency":
                record["freq_cbs"] = freqf
            if tensf is not None and "tens_prop" not in record and scheme_low != "tensorial":
                record["tens_prop"] = tensf
            if prop_hint:
                record["property_type"] = prop_hint

        elif isinstance(result, (tuple, list)):
            if len(result) == 3:
                record["hf_cbs"] = _to_float(result[0])
                record["corr_cbs"] = _to_float(result[1])
                record["total_energy"] = _to_float(result[2])
            elif len(result) == 2:
                first = _to_float(result[0])
                second = _to_float(result[1])
                if scheme_low in corr_set:
                    record["corr_cbs"] = first
                    record["total_energy"] = second
                elif scheme_low in hf_set:
                    record["hf_cbs"] = first
                    record["total_energy"] = second
                else:
                    record["corr_cbs"] = first
                    record["total_energy"] = second
            elif len(result) == 1:
                val = _to_float(result[0])
                if val is not None:
                    if scheme_low in corr_set:
                        record["corr_cbs"] = val
                    elif scheme_low in hf_set:
                        record["hf_cbs"] = val
                    else:
                        record["total_energy"] = val
            else:
                for v in result:
                    nv = _to_float(v)
                    if nv is not None:
                        if scheme_low in corr_set:
                            record["corr_cbs"] = nv
                        elif scheme_low in hf_set:
                            record["hf_cbs"] = nv
                        else:
                            record["total_energy"] = nv
                        break
        else:
            # scalar
            val = _to_float(result)
            if val is not None:
                if scheme_low in corr_set:
                    record["corr_cbs"] = val
                elif scheme_low in hf_set:
                    record["hf_cbs"] = val
                else:
                    record["total_energy"] = val

        calculations.append(record)
        return True

    except Exception as e:
        logger.exception("Error in [%s]", section_name)
        writer.write_error(str(output_file), section_name, str(e))
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
    # default output is a file inside the directory PyCBS-OUTPUTS
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("PyCBS-OUTPUTS") / "extrapolations_results.txt",
        help="Output report file (default: PyCBS-OUTPUTS/extrapolations_results.txt)",
    )
    parser.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity")


    args = parser.parse_args()
    setup_logging(args.verbose)

    print("""
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
    """)
    print("""
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
    """)
    try:
        config = read_config(args.input)
    except Exception as e:
        logger.error(e)
        sys.exit(1)

    ensure_parent_dir(args.output)
    args.output = unique_path(args.output)
    args.output.write_text("")
    writer.write_header(str(args.output))

    calculations: List[Dict[str, Any]] = []
    opt_cycles: List[Dict[str, Any]] = []

    total = success = 0

    for section_name in config.sections():
        total += 1
        if process_section(section_name, config[section_name], calculations, args.output):
            success += 1

    writer.write_reports(
        str(args.output),
        calculations=calculations
    )

    logger.info(f"Completed {success}/{total} calculations")
    sys.exit(0 if success == total else 1)


if __name__ == "__main__":
    main()
