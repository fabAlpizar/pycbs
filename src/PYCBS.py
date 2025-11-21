#!/usr/bin/env python3
"""
PYCBS.py - Improved entrypoint for pyCBS with automated basis handling.

This script is the CLI entrypoint for the pyCBS Complete Basis Set extrapolation
tool. It parses an INI-style input file with multiple sections (OPTIMIZATION and
calculation sections), optionally runs the optimizer, and dispatches calculation
sections to the appropriate extrapolation modules (USTE1, USTE2, USPE,
tensorial_properties1, frequency).

This version adds robust adapters so the main code tolerates small API
differences across the extrapolation modules (different function names,
argument orders or consolidated "run" functions). The adapters attempt a
best-effort mapping and log clear errors if a module does not expose the
expected functionality.
"""
from __future__ import annotations

import argparse
import configparser
import inspect
import logging
import platform
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import writer
import basis

# Attempt to import extrapolation modules; set None if not available.
try:
    import USTE1
except Exception as e:  # pragma: no cover - defensive
    USTE1 = None
    logging.getLogger(__name__).debug("USTE1 not available: %s", e)

try:
    import USTE2
except Exception as e:  # pragma: no cover - defensive
    USTE2 = None
    logging.getLogger(__name__).debug("USTE2 not available: %s", e)

try:
    import USPE
except Exception as e:  # pragma: no cover - defensive
    USPE = None
    logging.getLogger(__name__).debug("USPE not available: %s", e)

try:
    import tensorial_properties1 as TP
except Exception as e:  # pragma: no cover - defensive
    TP = None
    logging.getLogger(__name__).debug("tensorial_properties1 not available: %s", e)

try:
    import frequency
except Exception as e:  # pragma: no cover - defensive
    frequency = None
    logging.getLogger(__name__).debug("frequency module not available: %s", e)

# -----------------------------------------------------------------------------
# Defaults & CLI
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Small parsing utilities
# -----------------------------------------------------------------------------
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
                logging.getLogger(__name__).warning(
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
    """Context manager to suppress stdout/stderr (used when running optimizer)."""
    try:
        with open(sys.devnull, "w") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield
    except Exception:
        yield


# -----------------------------------------------------------------------------
# Generic adapter helpers (robust calling of module functions)
# -----------------------------------------------------------------------------
def _find_callable(module, *names):
    """Return the first callable attribute from module found in names, or None."""
    for name in names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None

def _compute_dynamic_correlation_with_provider(provider,
                                               corr_dict,
                                               basis1: str, basis2: str,
                                               Ecr1: float, Ecr2: float) -> float:
    """
    Given a provider module (USTE1/USTE2/USPE-like), try to compute the dynamic
    correlation 'dc' in a robust way.

    Strategy (in order):
      1. If provider exposes dynamic_correlation_energy(...), try calling it with
         common argument orders (Ecr1, Ecr2, corr_dict, basis1, basis2) and fallbacks.
      2. If not available or calling fails, fall back to the built-in inverse-cubic
         formula (the one you provided) using values in corr_dict for basis exponents
         (corr_dict[basis] is expected to hold the hierarchical exponent).
    Returns:
      dc (float) on success, else raises a TypeError.
    """
    # 1) Try provider.dynamic_correlation_energy if present
    dyn_fn = _find_callable(provider, "dynamic_correlation_energy", "dynamic_corr", "dynamic_correlation")
    if dyn_fn is not None:
        # Try a few calling patterns
        attempts = [
            (Ecr1, Ecr2, corr_dict, basis1, basis2),
            (Ecr1, Ecr2, basis1, basis2, corr_dict),
            (corr_dict, basis1, basis2, Ecr1, Ecr2),
            (Ecr1, Ecr2),  # maybe provider only needs the two Ecrs and uses its internal corr_dict
        ]
        for args in attempts:
            try:
                res = dyn_fn(*args)
                # allow either direct float return or tuple containing the dc
                if isinstance(res, tuple):
                    # if returns (dc, ...) take first
                    return float(res[0])
                else:
                    return float(res)
            except TypeError:
                continue
            except Exception:
                # provider function raised; continue trying other patterns
                continue

    # 2) Fallback: use inverse-cubic formula if corr_dict contains numeric exponents
    try:
        # attempt to find the exponent-like entries for basis1/basis2
        a1 = float(corr_dict.get(basis1, corr_dict.get(basis1.lower())))
        a2 = float(corr_dict.get(basis2, corr_dict.get(basis2.lower())))
        # inverse-cubic formula: dc = Ecr2 + (b2^-3 / (b1^-3 - b2^-3)) * (Ecr2 - Ecr1)
        denom = (a1 ** -3) - (a2 ** -3)
        if denom == 0:
            raise ZeroDivisionError("denominator zero in inverse-cubic fallback")
        dc = Ecr2 + ((a2 ** -3) / denom) * (Ecr2 - Ecr1)
        return float(dc)
    except Exception as e:
        raise TypeError("Unable to compute dynamic correlation with provider or fallback: " + str(e))


def _call_correlation_single(module, HF: float, Etot: float) -> float:
    """
    Call a provider correlation function that computes a single correlation contribution,
    e.g. USPE-style correlation_energy(HF, Etot) -> Ecr (float).
    Tries common names and argument orders; raises TypeError if not possible.
    """
    fn = _find_callable(module, "correlation_energy", "correlation", "correlation_uspe", "correlationEnergy")
    if fn is None:
        raise AttributeError(f"No single-value correlation function found in module {getattr(module, '__name__', module)}")

    # Try a few likely call patterns
    attempts = [
        (HF, Etot),
        (Etot, HF),
    ]
    for args in attempts:
        try:
            res = fn(*args)
            return float(res)
        except TypeError:
            continue
        except Exception:
            # provider function raised; try other patterns
            continue

    # try keyword mapping if signature available
    try:
        sig = inspect.signature(fn)
        kw = {}
        for pname in sig.parameters:
            ln = pname.lower()
            if "hf" in ln:
                kw[pname] = HF
            elif "tot" in ln or "etot" in ln or "e" == ln or "energy" in ln:
                kw[pname] = Etot
        if kw:
            res = fn(**kw)
            return float(res)
    except Exception:
        pass

    raise TypeError(f"Unable to call single-value correlation function of module {getattr(module,'__name__',module)} with plausible args")


def _call_correlation(module, HF1: float, HF2: float, E1: float, E2: float) -> Tuple[float, float]:
    """
    Call an available correlation function and return (correlation1, correlation2).

    Tries common names: correlation_energy, correlation_frequency, correlation.
    Expects a function that receives (HF1, HF2, E1, E2) in some order and returns a
    2-tuple (c1, c2). If a module provides a different signature, try to call it
    in the most-likely way; otherwise raise AttributeError/TypeError.
    """
    fn = _find_callable(module, "correlation_energy", "correlation_frequency", "correlation", "correlationEnergy")
    if fn is None:
        raise AttributeError(f"No correlation function found in module {getattr(module, '__name__', module)}")

    # Try a few likely call patterns
    for attempt in (
            (HF1, HF2, E1, E2),
            (E1, E2, HF1, HF2),  # some variants might accept Etot first
            (HF1, E1, HF2, E2),
    ):
        try:
            res = fn(*attempt)
            if isinstance(res, tuple) and len(res) >= 2:
                return res[0], res[1]
        except TypeError:
            continue
    # last resort: try keyword names mapping
    try:
        sig = inspect.signature(fn)
        kw = {}
        for pname in sig.parameters:
            ln = pname.lower()
            if "hf" in ln and "1" in ln:
                kw[pname] = HF1
            elif "hf" in ln and "2" in ln:
                kw[pname] = HF2
            elif ("f" in ln or "e" in ln or "corr" in ln) and "1" in ln:
                kw[pname] = E1
            elif ("f" in ln or "e" in ln or "corr" in ln) and "2" in ln:
                kw[pname] = E2
        if kw:
            res = fn(**kw)
            if isinstance(res, tuple) and len(res) >= 2:
                return res[0], res[1]
    except Exception:
        pass

    raise TypeError(
        f"Unable to call correlation function of module {getattr(module, '__name__', module)} with plausible args")


def _call_cbs_extrapolation(module, *args):
    """
    Call module.CBS_extrapolation with flexible argument shapes.

    The canonical expected signature (most modules) is:
        CBS_extrapolation(HF1, HF2, Ecr1, Ecr2, corr_dict, basis1, basis2[, basis3, basis4])

    This helper tries multiple orders and keyword calls and returns the function's
    result. On failure it raises a TypeError with a helpful message.
    """
    fn = getattr(module, "CBS_extrapolation", None)
    if not callable(fn):
        raise AttributeError(f"No 'CBS_extrapolation' function found in module {getattr(module, '__name__', module)}")

    # First: try direct pass-through (most likely)
    try:
        return fn(*args)
    except TypeError:
        pass

    # Second: try without basis args if module only expects HF/corr dicts
    try:
        # try first 5 args
        return fn(*args[:5])
    except TypeError:
        pass

    # Third: try some keyword-name mapping if possible
    try:
        sig = inspect.signature(fn)
        param_names = [p.name.lower() for p in sig.parameters.values()]
        kw = {}
        # args mapping heuristics
        # args expected: HF1, HF2, Ecr1, Ecr2, corr_dict, [bases...]
        if len(args) >= 5:
            HF1, HF2, Ecr1, Ecr2, corr_dict = args[:5]
            kw_map = {
                "hf1": HF1, "hf_1": HF1, "hf": HF1,
                "hf2": HF2, "hf_2": HF2,
                "ecr1": Ecr1, "ecorr1": Ecr1, "fcr1": Ecr1,
                "ecr2": Ecr2, "ecorr2": Ecr2, "fcr2": Ecr2,
                "dic": corr_dict, "corr": corr_dict, "corr_dict": corr_dict, "dic_correlacion": corr_dict
            }
            for name in param_names:
                if name in kw_map and name not in kw:
                    kw[name] = kw_map[name]
        # bases: map by 'basis1','basis2', etc.
        bases = args[5:] if len(args) > 5 else []
        for i, b in enumerate(bases, start=1):
            key = f"basis{i}"
            for pname in sig.parameters:
                if key in pname.lower() and pname not in kw:
                    kw[pname] = b
        if kw:
            return fn(**kw)
    except Exception:
        pass

    raise TypeError(f"Could not call {getattr(module, '__name__', module)}.CBS_extrapolation with available args")


# -----------------------------------------------------------------------------
# Module-specific small wrappers that use the adapters above
# -----------------------------------------------------------------------------
def uste1_run(module, method: str, basis1: str, basis2: str, HF1: float, HF2: float, E1: float, E2: float):
    """
    Execute a USTE1-style extrapolation and return (EHF, dc, CBS).
    Accepts modules that implement the reference interface:
      - dictionaries(method, basis1, basis2) -> (hf_dict, corr_dict)
      - correlation_energy(HF1, HF2, E1, E2) -> (Ecr1, Ecr2)  (or similar names)
      - CBS_extrapolation(...) -> (EHF, dc, CBS)
    """
    dicts_fn = _find_callable(module, "dictionaries")
    if dicts_fn is None:
        raise AttributeError("USTE1-like module must provide 'dictionaries' function")
    hf_dict, corr_dict = dicts_fn(method, basis1, basis2)

    Ecr1, Ecr2 = _call_correlation(module, HF1, HF2, E1, E2)
    EHF, dc, CBS = _call_cbs_extrapolation(module, HF1, HF2, Ecr1, Ecr2, corr_dict, basis1, basis2)
    return EHF, dc, CBS


def uste2_run(module, method: str, basis1: str, basis2: str, basis3: str, basis4: str,
              HF1: float, HF2: float, E1: float, E2: float):
    """
    Execute an USTE2-style extrapolation and return (EHF, dc, CBS).
    USTE2 reference interface expects dictionaries(method, basis1,basis2,basis3,basis4)
    but the wrapper will tolerate other small differences.
    """
    dicts_fn = _find_callable(module, "dictionaries")
    if dicts_fn is None:
        raise AttributeError("USTE2-like module must provide 'dictionaries' function")
    # try the most likely full signature first
    try:
        hf_dict, corr_dict = dicts_fn(method, basis1, basis2, basis3, basis4)
    except TypeError:
        # fallback to older signatures that accept fewer basis args (best-effort)
        hf_dict, corr_dict = dicts_fn(method, basis1, basis2)

    Ecr1, Ecr2 = _call_correlation(module, HF1, HF2, E1, E2)
    EHF, dc, CBS = _call_cbs_extrapolation(module, HF1, HF2, Ecr1, Ecr2, corr_dict, basis1, basis2, basis3, basis4)
    return EHF, dc, CBS


def uspe_run(module, *args, **kwargs):
    """
    Execute USPE-style extrapolation. USPE implementations vary widely:
      - Some expose CBS_extrapolation(HF, Etot, method, constant, basis) -> energy
      - Others expose different arg orders.

    This wrapper tries to call CBS_extrapolation and returns the single result energy.
    """
    fn = getattr(module, "CBS_extrapolation", None)
    if not callable(fn):
        raise AttributeError("USPE module must provide 'CBS_extrapolation' function")

    # Try calling with passed args first
    try:
        return fn(*args, **kwargs)
    except TypeError:
        # Try inverted orders commonly observed:
        # If user passed (method, E1, E2, E3) (older format), try that direct call too
        try:
            return fn(*args)
        except Exception as e:  # pragma: no cover - last resort
            raise


def tensorial_run(module, method: str, basis1: str, basis2: str,
                  zeta_HF1: float, zeta_HF2: float, zeta_E1: float, zeta_E2: float):
    """
    Runs tensorial extrapolation. Accepts either:
      - low-level functions in module (dictionaries, correlation_energy, CBS_extrapolation)
      - or a single run_tensorial(...) helper provided by the module.
    Returns (zeta_HF, zeta_cor, zeta_total) to align with writer expectations.
    """
    # If the module provides a run_tensorial function, prefer it
    run_fn = _find_callable(module, "run_tensorial", "run")
    if run_fn:
        # prefer the cleaned signature if possible
        try:
            # many implementations accept (basis1,basis2,basis3,basis4,labels,outputfile,calcname)
            # but we will call minimal signature if available
            sig = inspect.signature(run_fn)
            if len(sig.parameters) == 7:
                # caller will still want to handle opening/writing; use low-level path instead
                raise TypeError("module provides run_tensorial(7 args) which isn't compatible with this wrapper")
        except TypeError:
            pass

    # fallback to low-level approach (dictionaries + correlation + CBS)
    dicts_fn = _find_callable(module, "dictionaries")
    if dicts_fn is None:
        raise AttributeError("tensorial module must expose 'dictionaries' or 'run_tensorial'")

    hf_dict, corr_dict = dicts_fn(method, basis1, basis2)
    zcr1, zcr2 = _call_correlation(module, zeta_HF1, zeta_HF2, zeta_E1, zeta_E2)
    zeta_HF, zeta_cor, zeta_total = _call_cbs_extrapolation(module, zeta_HF1, zeta_HF2, zcr1, zcr2, corr_dict, basis1,
                                                            basis2)
    return zeta_HF, zeta_cor, zeta_total


def frequency_run(module, method: str, basis1: str, basis2: str,
                  HF1: float, HF2: float, F1: float, F2: float):
    """
    Execute the frequency module (mirrors USTE1-like behaviour but using frequency.* names).
    Returns (EHF, dc, CBS).
    """
    dicts_fn = _find_callable(module, "dictionaries")
    if dicts_fn is None:
        raise AttributeError("frequency module must provide 'dictionaries' function")
    hf_dict, corr_dict = dicts_fn(method, basis1, basis2)

    # correlation_frequency returns (Fcr1, Fcr2)
    corr_fn = _find_callable(module, "correlation_frequency", "correlation_energy", "correlation_frequency")
    if corr_fn is None:
        raise AttributeError("frequency module must provide 'correlation_frequency' function")
    Fcr1, Fcr2 = corr_fn(HF1, HF2, F1, F2)

    EHF, dc, CBS = _call_cbs_extrapolation(module, HF1, HF2, Fcr1, Fcr2, corr_dict, basis1, basis2)
    return EHF, dc, CBS


# -----------------------------------------------------------------------------
# High-level run_* functions used by the main loop (these read section keys,
# call the adapters above, and format writer output)
# -----------------------------------------------------------------------------
def run_uste1(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
    """
    Run a USTE1 calculation for a given config section and write to output_file.
    """
    if USTE1 is None:
        logging.getLogger(__name__).error("USTE1 module not found; skipping %s", calc_name)
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
        logging.getLogger(__name__).error("Missing param in %s: %s", calc_name, e)
        return
    except ValueError as e:
        logging.getLogger(__name__).error("Invalid numeric param in %s: %s", calc_name, e)
        return

    with output_file.open("a") as f:
        f.write("\n")
        f.write(f" JOB: {calc_name}\n")

    # after successful computation...
    try:
        EHF, dc, CBS = uste1_run(USTE1, method, basis1, basis2, HF1, HF2, E1, E2)
    except Exception as e:
        logging.getLogger(__name__).exception("Error while processing USTE1 calculation %s: %s", calc_name, e)
        return

    # Use user-provided scheme name (fallback to calc_name); show in UPPERCASE for clarity
    scheme_label = section.get("scheme", calc_name).strip().upper()
    writer.write_result(str(output_file), scheme_label, {"basis1": basis1, "basis2": basis2, "method": method},
                        EHF, dc, CBS)



def run_uste2(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
    """
    Run a USTE2 calculation and write results.

    Expected keys in section:
      - method, basis1, basis2, (optional) basis3, (optional) basis4,
      - HF1, HF2, E1, E2

    If basis3 or basis4 are missing, the function will fall back to sensible
    defaults (basis1/basis2) and log a warning. Prefer fixing the input file
    to provide explicit basis3/basis4 when they are really required.
    """
    if USTE2 is None:
        logging.getLogger(__name__).error("USTE2 module not found; skipping %s", calc_name)
        return

    try:
        method = section["method"]
        basis1 = section["basis1"]
        basis2 = section["basis2"]
        # basis3/basis4 **may** be missing in some input files -> fallback to basis1/basis2
        basis3 = section.get("basis3", None)
        basis4 = section.get("basis4", None)

        HF1 = float(section["HF1"])
        HF2 = float(section["HF2"])
        E1 = float(section["E1"])
        E2 = float(section["E2"])
    except KeyError as e:
        logging.getLogger(__name__).error("Missing param in %s: %s", calc_name, e)
        return
    except ValueError as e:
        logging.getLogger(__name__).error("Invalid numeric param in %s: %s", calc_name, e)
        return

    # If basis3/4 missing, fall back and warn the user
    if basis3 is None or basis4 is None:
        logging.getLogger(__name__).warning(
            "USTE2 section %s missing basis3/basis4 — falling back to basis1/basis2. "
            "If USTE2 requires distinct basis3/basis4, please add them to the input.",
            calc_name
        )
        # sensible defaults: copy basis1->basis3 and basis2->basis4 if absent
        if basis3 is None:
            basis3 = basis1
        if basis4 is None:
            basis4 = basis2

    # write job header
    with output_file.open("a") as f:
        f.write("\n")
        f.write(f" JOB: {calc_name}\n")

    try:
        EHF, dc, CBS = uste2_run(USTE2, method, basis1, basis2, basis3, basis4, HF1, HF2, E1, E2)
    except Exception as e:
        logging.getLogger(__name__).exception("Error while processing USTE2 calculation %s: %s", calc_name, e)
        return

    scheme_label = section.get("scheme", calc_name).strip().upper()
    writer.write_result(str(output_file), scheme_label, {
        "basis1": basis1, "basis2": basis2, "basis3": basis3, "basis4": basis4, "method": method
    }, EHF, dc, CBS)




def run_uspe(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
    """
    Run USPE scheme. Accepts fields:
      method, constant, basis, HF, Etot
    The USPE implementation may return only a single energy; writer will be called
    accordingly (non-USTE schemes use the 'energy' position).
    """
    if USPE is None:
        logging.getLogger(__name__).error("USPE module not found; skipping %s", calc_name)
        return
    try:
        method = section["method"]
        constant = section.get("constant", section.get("constant_type", "normal"))
        basis_name = section["basis"]
        HF = float(section["HF"])
        Etot = float(section["Etot"])
    except KeyError as e:
        logging.getLogger(__name__).error("Missing param in %s: %s", calc_name, e)
        return
    except ValueError as e:
        logging.getLogger(__name__).error("Invalid numeric param in %s: %s", calc_name, e)
        return

    with output_file.open("a") as f:
        f.write("\n")
        f.write(f" JOB: {calc_name}\n")

    try:
        resultado = uspe_run(USPE, HF, Etot, method, constant, basis_name)
    except Exception as e:
        logging.getLogger(__name__).exception("Error while processing USPE calculation %s: %s", calc_name, e)
        return

    scheme_label = section.get("scheme", calc_name).strip().upper()
    writer.write_result(str(output_file), scheme_label, {"basis_set": basis_name, "method": method},
                        EHF=None, dc=None, energy=resultado)



def run_frequency(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
    """
    Run frequency-based extrapolation (uses frequency module).
    Expected keys (method, basis1, basis2, HF1, HF2, F1/E1, F2/E2).
    """
    if frequency is None:
        logging.getLogger(__name__).error("frequency module not found; skipping %s", calc_name)
        return

    try:
        method = section["method"]
        basis1 = section["basis1"]
        basis2 = section["basis2"]
        HF1 = float(section["HF1"])
        HF2 = float(section["HF2"])
        raw_F1 = section.get("F1", section.get("E1", None))
        raw_F2 = section.get("F2", section.get("E2", None))
        if raw_F1 is None or raw_F2 is None:
            raise KeyError("Missing F1/F2 (or E1/E2) keys")
        F1 = float(raw_F1)
        F2 = float(raw_F2)
    except KeyError as e:
        logging.getLogger(__name__).error("Missing param in %s: %s", calc_name, e)
        return
    except ValueError as e:
        logging.getLogger(__name__).error("Invalid numeric param in %s: %s", calc_name, e)
        return

    with output_file.open("a") as f:
        f.write("\n")
        f.write(f" JOB: {calc_name}\n")

    # CORRECT: call frequency_run (previously this incorrectly called tensorial_run)
    try:
        EHF, dc, CBS = frequency_run(frequency, method, basis1, basis2, HF1, HF2, F1, F2)
    except Exception as e:
        logging.getLogger(__name__).exception("Error while processing frequency calculation %s: %s", calc_name, e)
        return

    scheme_label = section.get("scheme", calc_name).strip().upper()
    writer.write_result(str(output_file), scheme_label, {
        "basis1": basis1, "basis2": basis2, "method": method
    }, EHF, dc, CBS)




# ---------- Place into PYCBS.py (replace existing run_tensorial) ----------
def _run_tensorial_legacy(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
    """
    Legacy in-main implementation of tensorial handling. Keeps tolerant key parsing
    and supports both USTE (two-point) and USPE (single-point) dynamic-correlation
    providers. This is used only as a fallback when TP.run_tensorial_from_section
    is unavailable or fails.
    """
    # TP must exist (checked by caller), but provider modules may be None
    # read common inputs
    try:
        method = section["method"]
        basis1 = section["basis1"]
        basis2 = section.get("basis2", None)  # optional for single-point USPE
    except KeyError as e:
        logging.getLogger(__name__).error("Missing param in %s: %s", calc_name, e)
        return

    # provider selection (USTE or USPE)
    dc_scheme = section.get("dc_scheme", section.get("dc", "USTE1")).strip().upper()
    provider = None
    if dc_scheme.startswith("USTE"):
        provider = USTE1
    elif dc_scheme.startswith("USPE"):
        provider = USPE

    # If provider requested but not available, we'll try to use TP functions where possible.
    if provider is None and dc_scheme.startswith("USPE"):
        logging.getLogger(__name__).info("USPE provider not found; will attempt TP-only USPE flow.")

    # Branch: USPE (single-point)
    if dc_scheme.startswith("USPE"):
        raw_hf = section.get("zeta_HF1", section.get("zeta_HF", None))
        raw_e  = section.get("zeta_E1",  section.get("zeta_E", None))
        if (raw_hf is None) or (raw_e is None):
            logging.getLogger(__name__).error("USPE-style tensorial requires 'zeta_HF1' (or 'zeta_HF') and 'zeta_E1' (or 'zeta_E') keys for %s", calc_name)
            return
        try:
            zeta_HF = float(raw_hf)
            zeta_E  = float(raw_e)
        except ValueError as e:
            logging.getLogger(__name__).error("Invalid numeric param in %s: %s", calc_name, e)
            return

        # first try TP.USPE-style CBS_extrapolation if available
        try:
            zeta_total = _call_cbs_extrapolation(TP, zeta_HF, zeta_E, section.get("method", method),
                                                 section.get("constant", section.get("constant_type", "normal")), basis1)
            zeta_cor = zeta_E - zeta_HF
        except Exception:
            # try provider fallback (USPE module) if present
            if provider is not None:
                try:
                    zeta_total = _call_cbs_extrapolation(provider, zeta_HF, zeta_E, section.get("method", method),
                                                         section.get("constant", section.get("constant_type", "normal")), basis1)
                    zeta_cor = zeta_E - zeta_HF
                except Exception as e:
                    logging.getLogger(__name__).exception("USPE-style CBS extrapolation failed for %s: %s", calc_name, e)
                    return
            else:
                logging.getLogger(__name__).exception("USPE-style CBS extrapolation not available for %s (no TP or USPE provider)", calc_name)
                return

        scheme_label = section.get("scheme", calc_name).strip()
        writer.write_result(str(output_file), scheme_label, {
            "basis": basis1, "method": method, "dc_provider": dc_scheme,
            "constant": section.get("constant", section.get("constant_type", "normal"))
        }, zeta_HF, zeta_cor, zeta_total)
        return

    # Branch: USTE (two-point)
    # require two-point keys (allow alternate key names)
    try:
        raw_hf1 = section.get("zeta_HF1", section.get("zeta_HF", None))
        raw_hf2 = section.get("zeta_HF2", section.get("zeta_HF_2", None))
        raw_e1  = section.get("zeta_E1",  section.get("zeta_E", None))
        raw_e2  = section.get("zeta_E2",  section.get("zeta_E_2", None))
        if raw_hf1 is None or raw_hf2 is None or raw_e1 is None or raw_e2 is None:
            logging.getLogger(__name__).error("USTE-style tensorial requires zeta_HF1/zeta_HF2 and zeta_E1/zeta_E2 keys for %s", calc_name)
            return
        zeta_HF1 = float(raw_hf1); zeta_HF2 = float(raw_hf2)
        zeta_E1 = float(raw_e1);   zeta_E2 = float(raw_e2)
    except ValueError as e:
        logging.getLogger(__name__).error("Invalid numeric param in %s: %s", calc_name, e)
        return

    # Build dictionaries: prefer provider, else TP
    corr_dict = {}
    hf_dict = {}
    dicts_fn = None
    if provider is not None:
        dicts_fn = _find_callable(provider, "dictionaries")
    if dicts_fn is None:
        dicts_fn = _find_callable(TP, "dictionaries")

    if dicts_fn is None:
        logging.getLogger(__name__).error("No 'dictionaries' function found in provider or TP; cannot run USTE for %s", calc_name)
        return

    try:
        # try calling with (method, basis1, basis2)
        hf_dict, corr_dict = dicts_fn(method, basis1, section.get("basis2", None))
    except Exception:
        try:
            hf_dict, corr_dict = dicts_fn(method, basis1, basis2)
        except Exception as e:
            logging.getLogger(__name__).exception("dictionaries(...) call failed for %s: %s", calc_name, e)
            return

    # compute per-basis correlation contributions (Ecr1, Ecr2) using provider then TP
    try:
        Ecr1, Ecr2 = _call_correlation(provider if provider is not None else TP, zeta_HF1, zeta_HF2, zeta_E1, zeta_E2)
    except Exception:
        try:
            Ecr1, Ecr2 = _call_correlation(TP, zeta_HF1, zeta_HF2, zeta_E1, zeta_E2)
        except Exception as e:
            logging.getLogger(__name__).exception("Unable to obtain two-point correlation contributions for %s: %s", calc_name, e)
            return

    # compute dynamic correlation via provider helper or fallback
    try:
        dc = _compute_dynamic_correlation_with_provider(provider if provider is not None else TP, corr_dict, basis1, basis2, Ecr1, Ecr2)
    except Exception as e:
        logging.getLogger(__name__).exception("Failed to compute dynamic correlation for %s: %s", calc_name, e)
        return

    # final CBS extrapolation: prefer TP, else provider
    try:
        zeta_HF, zeta_cor, zeta_total = _call_cbs_extrapolation(TP, zeta_HF1, zeta_HF2, Ecr1, Ecr2, corr_dict, basis1, basis2)
    except Exception as e_tp:
        logging.getLogger(__name__).warning("TP.CBS_extrapolation failed for %s: %s — trying provider fallback", calc_name, e_tp)
        try:
            zeta_HF, zeta_cor, zeta_total = _call_cbs_extrapolation(provider if provider is not None else TP, zeta_HF1, zeta_HF2, Ecr1, Ecr2, corr_dict, basis1, basis2)
        except Exception as e_prov:
            logging.getLogger(__name__).exception("Provider CBS_extrapolation also failed for %s: %s", calc_name, e_prov)
            return

    scheme_label = section.get("scheme", calc_name).strip()
    writer.write_result(str(output_file), scheme_label, {
        "basis1": basis1, "basis2": basis2, "method": method, "dc_provider": dc_scheme
    }, zeta_HF, zeta_cor, zeta_total)


def run_tensorial(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
    """
    Delegating run_tensorial: prefer TP.run_tensorial_from_section (module-owned parser),
    otherwise fall back to the legacy _run_tensorial_legacy implementation above.
    """
    if TP is None:
        logging.getLogger(__name__).error("TensorialProperties module not found; skipping %s", calc_name)
        return

    # write job header
    with output_file.open("a") as f:
        f.write("\n")
        f.write(f" JOB: {calc_name}\n")

    # Prefer a helper inside the module (if added). Accept either name.
    run_helper = _find_callable(TP, "run_tensorial_from_section", "run_tensorial")
    if run_helper:
        try:
            # pass a plain dict (SectionProxy works but dict is clearer)
            zeta_HF, zeta_cor, zeta_total = run_helper(dict(section))
            scheme_label = section.get("scheme", calc_name).strip()
            writer.write_result(str(output_file), scheme_label, {
                "basis1": section.get("basis1"), "method": section.get("method"),
                "dc_provider": section.get("dc_scheme", section.get("dc", "USTE1"))
            }, zeta_HF, zeta_cor, zeta_total)
            return
        except Exception as e:
            logging.getLogger(__name__).exception("TP.run_tensorial helper failed for %s: %s — falling back to legacy", calc_name, e)

    # fallback to legacy in-main logic
    try:
        _run_tensorial_legacy(section, output_file, calc_name)
    except Exception as e:
        logging.getLogger(__name__).exception("Fallback tensorial processing failed for %s: %s", calc_name, e)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
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
        import pyscf  # type: ignore
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
            'AVDZ': 'aug-cc-pvdz', 'AVTZ': 'aug-cc-pvtz', 'AVQZ': 'aug-cc-pvqz', 'AV5Z': 'aug-cc-pv5z',
            'AV6Z': 'aug-cc-pv6z',
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
        print("\n" + "=" * 60)
        print(" BASIS SETS SUMMARY ")
        print("=" * 60)
        print(f"Original basis sets: {orig_basis1}, {orig_basis2}")
        print(f"PySCF basis sets: {pyscf1}, {pyscf2}")
        print(f"x1 (from dc3) for {orig_basis1}: {opt_params['x1']}")
        print(f"x2 (from dc3) for {orig_basis2}: {opt_params['x2']}")
        print(f"x1_hf (from hf) for {orig_basis1}: {opt_params['x1_hf']}")
        print(f"x2_hf (from hf) for {orig_basis2}: {opt_params['x2_hf']}")
        if missing:
            print("⚠️ WARNING: Some values were missing and defaults were used.")
        print("=" * 60 + "\n")

        # Write summary into output file
        with output_file.open("a") as f:
            f.write("=" * 60 + "\n")
            f.write(" BASIS SETS SUMMARY \n")
            f.write("=" * 60 + "\n")
            f.write(f"Original basis sets: {orig_basis1}, {orig_basis2}\n")
            f.write(f"PySCF basis sets: {pyscf1}, {pyscf2}\n")
            f.write(f"x1 (dc3) for {orig_basis1}: {opt_params['x1']}\n")
            f.write(f"x2 (dc3) for {orig_basis2}: {opt_params['x2']}\n")
            f.write(f"x1_hf (hf) for {orig_basis1}: {opt_params['x1_hf']}\n")
            f.write(f"x2_hf (hf) for {orig_basis2}: {opt_params['x2_hf']}\n")
            if missing:
                f.write("NOTE: Some hierarchical values were missing; defaults were used.\n")
            f.write("=" * 60 + "\n\n")
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
                    "init_parameters": ",".join(
                        map(str, opt_params.get("init_parameters", OPT_DEFAULTS["init_parameters"])))
                }, EHF=None, dc=None, energy=final_energy)
            except Exception:
                pass

    # Process remaining scheme sections (USTE1, USTE2, USPE, TENSORIAL, FREQUENCY)
    print('=' * 60)
    print(" CALCULATIONS STATUS ")
    print('=' * 60)
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
            elif scheme == "FREQUENCY":
                run_frequency(section, output_file, section_name)
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
    print('=' * 60)
    print(f"\nResults saved to: {abs_path}")
    print('=' * 60)


if __name__ == "__main__":
    main()
