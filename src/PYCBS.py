#!/usr/bin/env python3
"""
PYCBS.py - Entrypoint for pyCBS (CBS extrapolation).
Adapted to call tensorial_properties1 (USTE/USPE) using its native function names.
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

# Module imports (optional providers)
try:
    import USTE1
except Exception:
    USTE1 = None
try:
    import USTE2
except Exception:
    USTE2 = None
try:
    import USPE
except Exception:
    USPE = None
try:
    import tensorial_properties1 as TP
except Exception:
    TP = None
try:
    import frequency
except Exception:
    frequency = None

# Defaults
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


# -----------------------
# CLI / helpers
# -----------------------
def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pyCBS", description="pyCBS: Complete Basis Set extrapolation tool")
    p.add_argument("-i", "--input", type=Path, default=Path("inputfile.inp"), help="Input file (INI format)")
    p.add_argument("-o", "--output", type=Path, default=Path("results.out"), help="Output results file")
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
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)


def read_config(input_path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(input_path)
    return cfg


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def parse_list_of_floats(txt: str) -> list[float]:
    return [float(p.strip()) for p in txt.split(",") if p.strip()]


def parse_list_of_str(txt: str) -> list[str]:
    return [p.strip() for p in txt.split(",") if p.strip()]


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
                logging.getLogger(__name__).warning("Could not parse optimization param '%s': %s (using default)", key, e)
    # normalize lists
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


# -----------------------
# Module adapters
# -----------------------
def _find_callable(module, *names):
    for name in names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


def _compute_dynamic_correlation_with_provider(provider, corr_dict, basis1: str, basis2: str, Ecr1: float, Ecr2: float) -> float:
    dyn_fn = _find_callable(provider, "dynamic_correlation_energy", "dynamic_corr", "dynamic_correlation")
    if dyn_fn:
        # try several calling orders
        attempts = [
            (Ecr1, Ecr2, corr_dict, basis1, basis2),
            (Ecr1, Ecr2, basis1, basis2, corr_dict),
            (corr_dict, basis1, basis2, Ecr1, Ecr2),
            (Ecr1, Ecr2),
        ]
        for args in attempts:
            try:
                res = dyn_fn(*args)
                return float(res[0]) if isinstance(res, tuple) else float(res)
            except TypeError:
                continue
            except Exception:
                continue

    # fallback inverse-cubic using corr_dict entries (expects numeric hierarchical exponents)
    try:
        a1 = float(corr_dict.get(basis1, corr_dict.get(basis1.lower())))
        a2 = float(corr_dict.get(basis2, corr_dict.get(basis2.lower())))
        denom = (a1 ** -3) - (a2 ** -3)
        if denom == 0:
            raise ZeroDivisionError("denominator zero in inverse-cubic fallback")
        dc = Ecr2 + ((a2 ** -3) / denom) * (Ecr2 - Ecr1)
        return float(dc)
    except Exception as e:
        raise TypeError("Unable to compute dynamic correlation: " + str(e))


def _call_correlation(module, HF1: float, HF2: float, E1: float, E2: float) -> Tuple[float, float]:
    fn = _find_callable(module, "correlation_energy", "correlation_frequency", "correlation", "correlationEnergy")
    if fn is None:
        raise AttributeError(f"No correlation function found in module {getattr(module, '__name__', module)}")
    attempts = [
        (HF1, HF2, E1, E2),
        (E1, E2, HF1, HF2),
        (HF1, E1, HF2, E2),
    ]
    for args in attempts:
        try:
            res = fn(*args)
            if isinstance(res, tuple) and len(res) >= 2:
                return float(res[0]), float(res[1])
        except TypeError:
            continue
    # try keyword mapping
    try:
        sig = inspect.signature(fn)
        kw: dict = {}
        for pname in sig.parameters:
            ln = pname.lower()
            if "hf" in ln and "1" in ln:
                kw[pname] = HF1
            elif "hf" in ln and "2" in ln:
                kw[pname] = HF2
            elif ("corr" in ln or "e" in ln or "f" in ln) and "1" in ln:
                kw[pname] = E1
            elif ("corr" in ln or "e" in ln or "f" in ln) and "2" in ln:
                kw[pname] = E2
        if kw:
            res = fn(**kw)
            if isinstance(res, tuple) and len(res) >= 2:
                return float(res[0]), float(res[1])
    except Exception:
        pass
    raise TypeError(f"Unable to call correlation function of {getattr(module,'__name__',module)}")


def _call_cbs_extrapolation(module, *args):
    # prefer canonical CBS_extrapolation
    fn = getattr(module, "CBS_extrapolation", None)
    if callable(fn):
        try:
            return fn(*args)
        except TypeError:
            pass
    # try USTE_CBS_extrapolation
    fn_uste = getattr(module, "USTE_CBS_extrapolation", None)
    if callable(fn_uste):
        try:
            return fn_uste(*args)
        except TypeError:
            if len(args) >= 7:
                try:
                    return fn_uste(args[0], args[1], args[2], args[3], args[4], args[5], args[6])
                except Exception:
                    pass
    # try USPE_CBS_extrapolation (single-point style, adapt if needed)
    fn_uspe = getattr(module, "USPE_CBS_extrapolation", None)
    if callable(fn_uspe):
        try:
            return fn_uspe(*args)
        except TypeError:
            if len(args) >= 5:
                zeta_HF, zeta_E, method, constant, basis1 = args[:5]
                try:
                    # duplicate HF & basis to satisfy 7-arg signatures some modules use
                    return fn_uspe(zeta_HF, zeta_HF, zeta_E, method, constant, basis1, basis1)
                except Exception:
                    pass
    raise TypeError(f"Could not call any CBS extrapolation function on {getattr(module,'__name__',module)}")


# -----------------------
# Module-specific wrappers
# -----------------------
def uste1_run(module, method: str, basis1: str, basis2: str, HF1: float, HF2: float, E1: float, E2: float):
    dicts_fn = _find_callable(module, "dictionaries")
    if dicts_fn is None:
        raise AttributeError("USTE1-like module must provide 'dictionaries'")
    hf_dict, corr_dict = dicts_fn(method, basis1, basis2)
    Ecr1, Ecr2 = _call_correlation(module, HF1, HF2, E1, E2)
    EHF, dc, CBS = _call_cbs_extrapolation(module, HF1, HF2, Ecr1, Ecr2, corr_dict, basis1, basis2)
    return EHF, dc, CBS


def uste2_run(module, method: str, basis1: str, basis2: str, basis3: str, basis4: str,
              HF1: float, HF2: float, E1: float, E2: float):
    dicts_fn = _find_callable(module, "dictionaries")
    if dicts_fn is None:
        raise AttributeError("USTE2-like module must provide 'dictionaries'")
    try:
        hf_dict, corr_dict = dicts_fn(method, basis1, basis2, basis3, basis4)
    except TypeError:
        hf_dict, corr_dict = dicts_fn(method, basis1, basis2)
    Ecr1, Ecr2 = _call_correlation(module, HF1, HF2, E1, E2)
    EHF, dc, CBS = _call_cbs_extrapolation(module, HF1, HF2, Ecr1, Ecr2, corr_dict, basis1, basis2, basis3, basis4)
    return EHF, dc, CBS


def uspe_run(module, *args, **kwargs):
    fn_uspe = getattr(module, "USPE_CBS_extrapolation", None)
    if callable(fn_uspe):
        try:
            return fn_uspe(*args, **kwargs)
        except Exception:
            pass
    fn = getattr(module, "CBS_extrapolation", None)
    if callable(fn):
        return fn(*args, **kwargs)
    raise AttributeError("USPE module must provide 'USPE_CBS_extrapolation' or 'CBS_extrapolation'")


def tensorial_run(module, method: str, basis1: str, basis2: str,
                  zeta_HF1: float, zeta_HF2: float, zeta_E1: float, zeta_E2: float):
    run_fn = _find_callable(module, "run_tensorial", "run")
    if run_fn:
        try:
            sig = inspect.signature(run_fn)
            if len(sig.parameters) != 7:
                return run_fn(method, basis1, basis2, zeta_HF1, zeta_HF2, zeta_E1, zeta_E2)
        except Exception:
            pass
    dicts_fn = _find_callable(module, "dictionaries")
    if dicts_fn is None:
        raise AttributeError("tensorial module must expose 'dictionaries' or 'run_tensorial'")
    hf_dict, corr_dict = dicts_fn(method, basis1, basis2)
    zcr1, zcr2 = _call_correlation(module, zeta_HF1, zeta_HF2, zeta_E1, zeta_E2)
    zeta_HF, zeta_cor, zeta_total = _call_cbs_extrapolation(module, zeta_HF1, zeta_HF2, zcr1, zcr2, corr_dict, basis1, basis2)
    return zeta_HF, zeta_cor, zeta_total


def frequency_run(module, method: str, basis1: str, basis2: str, HF1: float, HF2: float, F1: float, F2: float):
    dicts_fn = _find_callable(module, "dictionaries")
    if dicts_fn is None:
        raise AttributeError("frequency module must provide 'dictionaries'")
    hf_dict, corr_dict = dicts_fn(method, basis1, basis2)
    corr_fn = _find_callable(module, "correlation_frequency", "correlation_energy", "correlation_frequency")
    if corr_fn is None:
        raise AttributeError("frequency module must provide 'correlation_frequency'")
    Fcr1, Fcr2 = corr_fn(HF1, HF2, F1, F2)
    EHF, dc, CBS = _call_cbs_extrapolation(module, HF1, HF2, Fcr1, Fcr2, corr_dict, basis1, basis2)
    return EHF, dc, CBS


# -----------------------
# High-level run_* handlers
# -----------------------
def run_uste1(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
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
    except (KeyError, ValueError) as e:
        logging.getLogger(__name__).error("Invalid/missing param in %s: %s", calc_name, e)
        return
    with output_file.open("a") as f:
        f.write(f"\n JOB: {calc_name}\n")
    try:
        EHF, dc, CBS = uste1_run(USTE1, method, basis1, basis2, HF1, HF2, E1, E2)
    except Exception as e:
        logging.getLogger(__name__).exception("Error while processing USTE1 %s: %s", calc_name, e)
        return
    scheme_label = section.get("scheme", calc_name).strip().upper()
    writer.write_result(str(output_file), scheme_label, {"basis1": basis1, "basis2": basis2, "method": method}, EHF, dc, CBS)


def run_uste2(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
    if USTE2 is None:
        logging.getLogger(__name__).error("USTE2 module not found; skipping %s", calc_name)
        return
    try:
        method = section["method"]
        basis1 = section["basis1"]
        basis2 = section["basis2"]
        basis3 = section.get("basis3", basis1)
        basis4 = section.get("basis4", basis2)
        HF1 = float(section["HF1"])
        HF2 = float(section["HF2"])
        E1 = float(section["E1"])
        E2 = float(section["E2"])
    except (KeyError, ValueError) as e:
        logging.getLogger(__name__).error("Invalid/missing param in %s: %s", calc_name, e)
        return
    with output_file.open("a") as f:
        f.write(f"\n JOB: {calc_name}\n")
    try:
        EHF, dc, CBS = uste2_run(USTE2, method, basis1, basis2, basis3, basis4, HF1, HF2, E1, E2)
    except Exception as e:
        logging.getLogger(__name__).exception("Error while processing USTE2 %s: %s", calc_name, e)
        return
    scheme_label = section.get("scheme", calc_name).strip().upper()
    writer.write_result(str(output_file), scheme_label, {"basis1": basis1, "basis2": basis2, "basis3": basis3, "basis4": basis4, "method": method}, EHF, dc, CBS)


def run_uspe(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
    if USPE is None:
        logging.getLogger(__name__).error("USPE module not found; skipping %s", calc_name)
        return
    try:
        method = section["method"]
        constant = section.get("constant", section.get("constant_type", "normal"))
        basis_name = section["basis"]
        HF = float(section["HF"])
        Etot = float(section["Etot"])
    except (KeyError, ValueError) as e:
        logging.getLogger(__name__).error("Invalid/missing param in %s: %s", calc_name, e)
        return
    with output_file.open("a") as f:
        f.write(f"\n JOB: {calc_name}\n")
    try:
        resultado = uspe_run(USPE, HF, Etot, method, constant, basis_name)
    except Exception as e:
        logging.getLogger(__name__).exception("Error while processing USPE %s: %s", calc_name, e)
        return
    scheme_label = section.get("scheme", calc_name).strip().upper()
    writer.write_result(str(output_file), scheme_label, {"basis_set": basis_name, "method": method}, EHF=None, dc=None, energy=resultado)


def run_frequency(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
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
            raise KeyError("Missing F1/F2 (or E1/E2)")
        F1 = float(raw_F1)
        F2 = float(raw_F2)
    except (KeyError, ValueError) as e:
        logging.getLogger(__name__).error("Invalid/missing param in %s: %s", calc_name, e)
        return
    with output_file.open("a") as f:
        f.write(f"\n JOB: {calc_name}\n")
    try:
        EHF, dc, CBS = frequency_run(frequency, method, basis1, basis2, HF1, HF2, F1, F2)
    except Exception as e:
        logging.getLogger(__name__).exception("Error while processing frequency %s: %s", calc_name, e)
        return
    scheme_label = section.get("scheme", calc_name).strip().upper()
    writer.write_result(str(output_file), scheme_label, {"basis1": basis1, "basis2": basis2, "method": method}, EHF, dc, CBS)


# -----------------------
# Simplified tensorial handlers (replace existing _run_tensorial_legacy and run_tensorial)
# -----------------------

def _run_tensorial_legacy(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
    """
    Simplified legacy handler that calls tensorial_properties1 (TP) functions directly.
    This intentionally expects the signatures used in your tensorial_properties1.py:
      - dictionaries(method, basis1, basis2) -> (hf_dict, corr_dict)
      - USTE_CBS_extrapolation(zeta_HF1,zeta_HF2,zeta_cor1,zeta_cor2,corr_dict,basis1,basis2) -> (zhf,zcor,ztot)
      - USPE_CBS_extrapolation(zeta_HF1,zeta_HF2,zeta_E,method,constant,basis1,basis2) -> (zhf,zcor,ztot)
      - optionally: correlation_energy(HF1,HF2,E1,E2) -> (Ecr1,Ecr2)
    """
    log = logging.getLogger(__name__)

    if TP is None:
        log.error("TP (tensorial_properties1) not available; cannot run legacy tensorial for %s", calc_name)
        return

    # Basic params
    try:
        method = section["method"]
        basis1 = section["basis1"]
        basis2 = section.get("basis2", basis1)
    except KeyError as e:
        log.error("Missing required keys in %s: %s", calc_name, e)
        return

    dc_scheme = section.get("dc_scheme", section.get("dc", "USTE1")).strip().upper()

    # --- USPE (single-point) ---
    if dc_scheme.startswith("USPE"):
        raw_hf = section.get("zeta_HF1", section.get("zeta_HF", None))
        raw_e = section.get("zeta_E1", section.get("zeta_E", None))
        if raw_hf is None or raw_e is None:
            log.error("USPE tensorial requires 'zeta_HF1' (or 'zeta_HF') and 'zeta_E1' (or 'zeta_E') for %s", calc_name)
            return
        try:
            zeta_HF = float(raw_hf)
            zeta_E = float(raw_e)
        except ValueError as e:
            log.error("Invalid numeric zeta in %s: %s", calc_name, e)
            return

        # Try direct TP.USPE_CBS_extrapolation (signature matches your TP module)
        tp_uspe = getattr(TP, "USPE_CBS_extrapolation", None)
        if callable(tp_uspe):
            try:
                zeta_HF_out, zeta_cor, zeta_total = tp_uspe(zeta_HF, zeta_HF, zeta_E, method,
                                                            section.get("constant", "normal"), basis1, basis2)
                writer.write_result(str(output_file),
                                    section.get("scheme", calc_name).strip(),
                                    {"basis": basis1, "method": method, "dc_provider": "TP-USPE", "constant": section.get("constant", "normal")},
                                    zeta_HF_out, zeta_cor, zeta_total)
                return
            except Exception as e:
                log.exception("TP.USPE_CBS_extrapolation failed for %s: %s", calc_name, e)
                # fall through to try alternative (simple) computation below

        # If TP.USPE_CBS_extrapolation not available or failed, try TP.USPE_correlation_energy (single value)
        corr_fn = getattr(TP, "USPE_correlation_energy", None)
        if callable(corr_fn):
            try:
                zeta_cor = corr_fn(method, basis1, section.get("constant", "normal"), zeta_E)
                zeta_HF_out = zeta_HF
                zeta_total = zeta_HF_out + float(zeta_cor)
                writer.write_result(str(output_file),
                                    section.get("scheme", calc_name).strip(),
                                    {"basis": basis1, "method": method, "dc_provider": "TP-USPE-corr", "constant": section.get("constant", "normal")},
                                    zeta_HF_out, float(zeta_cor), zeta_total)
                return
            except Exception as e:
                log.exception("TP.USPE_correlation_energy failed for %s: %s", calc_name, e)

        # Last resort: simple estimate zeta_cor = zeta_E - zeta_HF (naive but safe)
        try:
            zeta_cor = float(zeta_E) - float(zeta_HF)
            zeta_HF_out = float(zeta_HF)
            zeta_total = zeta_HF_out + zeta_cor
            writer.write_result(str(output_file),
                                section.get("scheme", calc_name).strip(),
                                {"basis": basis1, "method": method, "dc_provider": "naive", "constant": section.get("constant", "normal")},
                                zeta_HF_out, zeta_cor, zeta_total)
            return
        except Exception as e:
            log.exception("Final USPE fallback failed for %s: %s", calc_name, e)
            return

    # --- USTE (two-point) ---
    # require two-point keys
    try:
        raw_hf1 = section.get("zeta_HF1", section.get("zeta_HF", None))
        raw_hf2 = section.get("zeta_HF2", section.get("zeta_HF_2", None))
        raw_e1 = section.get("zeta_E1", section.get("zeta_E", None))
        raw_e2 = section.get("zeta_E2", section.get("zeta_E_2", None))
        if raw_hf1 is None or raw_hf2 is None or raw_e1 is None or raw_e2 is None:
            log.error("USTE tensorial requires zeta_HF1/zeta_HF2 and zeta_E1/zeta_E2 for %s", calc_name)
            return
        zeta_HF1 = float(raw_hf1); zeta_HF2 = float(raw_hf2)
        zeta_E1 = float(raw_e1); zeta_E2 = float(raw_e2)
    except ValueError as e:
        log.error("Invalid numeric zeta in %s: %s", calc_name, e)
        return

    # Use TP.dictionaries (expects method, basis1, basis2) to get corr_dict
    dicts_fn = getattr(TP, "dictionaries", None)
    if not callable(dicts_fn):
        log.error("TP.dictionaries not found; cannot run USTE for %s", calc_name)
        return

    try:
        hf_dict, corr_dict = dicts_fn(method, basis1, basis2)
    except Exception as e:
        log.exception("TP.dictionaries(...) failed for %s: %s", calc_name, e)
        return

    # Try TP.correlation_energy if available, otherwise Ecr = zeta_E - zeta_HF
    corr_fn = getattr(TP, "correlation_energy", None)
    try:
        if callable(corr_fn):
            Ecr1, Ecr2 = corr_fn(zeta_HF1, zeta_HF2, zeta_E1, zeta_E2)
            Ecr1 = float(Ecr1); Ecr2 = float(Ecr2)
        else:
            Ecr1 = float(zeta_E1) - float(zeta_HF1)
            Ecr2 = float(zeta_E2) - float(zeta_HF2)
    except Exception as e:
        log.exception("Failed to obtain per-basis correlation contributions for %s: %s", calc_name, e)
        return

    # Now call TP.USTE_CBS_extrapolation directly (signature matches your TP file)
    tp_uste = getattr(TP, "USTE_CBS_extrapolation", None)
    if callable(tp_uste):
        try:
            zeta_HF_out, zeta_cor, zeta_total = tp_uste(zeta_HF1, zeta_HF2, Ecr1, Ecr2, corr_dict, basis1, basis2)
            writer.write_result(str(output_file),
                                section.get("scheme", calc_name).strip(),
                                {"basis1": basis1, "basis2": basis2, "method": method, "dc_provider": "TP-USTE"},
                                zeta_HF_out, zeta_cor, zeta_total)
            return
        except Exception as e:
            log.exception("TP.USTE_CBS_extrapolation failed for %s: %s", calc_name, e)

    # If TP.USTE_CBS_extrapolation not available or failed, try generic _call_cbs_extrapolation with TP (keeps compatibility)
    try:
        zeta_HF_out, zeta_cor, zeta_total = _call_cbs_extrapolation(TP, zeta_HF1, zeta_HF2, Ecr1, Ecr2, corr_dict, basis1, basis2)
        writer.write_result(str(output_file),
                            section.get("scheme", calc_name).strip(),
                            {"basis1": basis1, "basis2": basis2, "method": method, "dc_provider": "TP-generic"},
                            zeta_HF_out, zeta_cor, zeta_total)
        return
    except Exception as e:
        log.exception("Final USTE CBS extrapolation failed for %s: %s", calc_name, e)
        return


def run_tensorial(section: configparser.SectionProxy, output_file: Path, calc_name: str) -> None:
    """
    High-level tensorial runner. Prefer TP-provided section helper if available,
    otherwise use the simplified legacy flow above that directly uses TP functions.
    """
    log = logging.getLogger(__name__)
    if TP is None:
        log.error("TensorialProperties module not available; skipping %s", calc_name)
        return

    # write header
    with output_file.open("a") as f:
        f.write(f"\n JOB: {calc_name}\n")

    # If module provides a helper, try it first
    helper = _find_callable(TP, "run_tensorial_from_section", "run_tensorial")
    if helper:
        try:
            # helper is expected to accept a dict-like section
            zeta_HF, zeta_cor, zeta_total = helper(dict(section))
            writer.write_result(str(output_file),
                                section.get("scheme", calc_name).strip(),
                                {"basis1": section.get("basis1"), "method": section.get("method"), "dc_provider": section.get("dc_scheme", section.get("dc", "USTE1"))},
                                zeta_HF, zeta_cor, zeta_total)
            return
        except Exception as e:
            log.exception("TP helper failed for %s: %s — falling back to internal handler", calc_name, e)

    # Fallback to simplified legacy implementation
    _run_tensorial_legacy(section, output_file, calc_name)



# -----------------------
# Main entrypoint
# -----------------------
def main():
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
    print(BANNER)
    print("pyCBS - Complete Basis Set extrapolation tool\n")
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

    # Optimization section (if present)
    opt_enabled = False
    opt_params: Dict[str, Any] = {}
    if config.has_section("OPTIMIZATION"):
        sec = config["OPTIMIZATION"]
        opt_enabled = parse_bool(sec.get("optimization", "False"))
        opt_params = gather_optimization_params(sec)
        log.info("OPTIMIZATION detected: enabled=%s", opt_enabled)

        # handle basis map + hierarchical values (best-effort)
        orig_basis1, orig_basis2 = opt_params["basis_sets"][0], opt_params["basis_sets"][1]
        pyscf_basis_map = {  # friendly -> pyscf
            'VDZ': 'cc-pvdz', 'VTZ': 'cc-pvtz', 'VQZ': 'cc-pvqz', 'V5Z': 'cc-pv5z', 'V6Z': 'cc-pv6z',
            'AVDZ': 'aug-cc-pvdz', 'AVTZ': 'aug-cc-pvtz', 'AVQZ': 'aug-cc-pvqz', 'AV5Z': 'aug-cc-pv5z',
            'AV6Z': 'aug-cc-pv6z',
        }
        friendly_from_pyscf = {v.lower(): k for k, v in pyscf_basis_map.items()}

        def get_friendly_and_pyscf(name: str):
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

        x1 = basis.dc3.get(key1) if key1 in basis.dc3 else None
        x2 = basis.dc3.get(key2) if key2 in basis.dc3 else None
        x1_hf = basis.hf.get(key1) if key1 in basis.hf else None
        x2_hf = basis.hf.get(key2) if key2 in basis.hf else None

        missing = False
        if x1 is None or x2 is None or x1_hf is None or x2_hf is None:
            missing = True
            log.warning("Some hierarchical values missing for '%s', '%s'", orig_basis1, orig_basis2)

        if any(k in sec for k in ("x1", "x2", "x1_hf", "x2_hf")):
            log.info("User-provided exponent values will be ignored; using dictionary values.")

        opt_params["basis_sets"] = [pyscf1, pyscf2]
        opt_params["x1"] = x1 if x1 is not None else OPT_DEFAULTS["x1"]
        opt_params["x2"] = x2 if x2 is not None else OPT_DEFAULTS["x2"]
        opt_params["x1_hf"] = x1_hf if x1_hf is not None else OPT_DEFAULTS["x1_hf"]
        opt_params["x2_hf"] = x2_hf if x2_hf is not None else OPT_DEFAULTS["x2_hf"]

        # write summary
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
                f.write("NOTE: Some hierarchical values were missing; defaults used.\n")
            f.write("=" * 60 + "\n\n")

    # override workers if CLI provided
    if args.workers is not None:
        opt_params["workers"] = args.workers

    # run optimizer if requested
    optimizer_result = None
    if opt_enabled and not args.no_opt:
        try:
            import optimization as optmod  # type: ignore
            run_opt = getattr(optmod, "run_optimization", None)
            if callable(run_opt):
                with silence_output():
                    optimizer_result = run_opt(opt_params, output_file=str(output_file))
        except Exception:
            log.exception("Optimizer failed to run")

        if isinstance(optimizer_result, dict):
            try:
                writer.write_optimization_summary(str(output_file), optimizer_result.get("history"))
            except Exception:
                log.exception("Failed writing optimizer summary")
            try:
                writer.write_result(str(output_file), "OPTIMIZATION", {
                    "method": opt_params.get("METHOD", OPT_DEFAULTS["METHOD"]),
                    "basis_sets": ",".join(opt_params.get("basis_sets", OPT_DEFAULTS["basis_sets"])),
                    "init_parameters": ",".join(map(str, opt_params.get("init_parameters", OPT_DEFAULTS["init_parameters"])))
                }, EHF=None, dc=None, energy=optimizer_result.get("energy"))
            except Exception:
                pass

    # iterate sections and dispatch
    print("=" * 60)
    print(" CALCULATIONS STATUS ")
    print("=" * 60)
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

    try:
        writer.write_summary_table(str(output_file))
    except Exception:
        logging.getLogger(__name__).exception("Failed to write summary table")

    try:
        abs_path = output_file.resolve()
    except Exception:
        abs_path = output_file
    print("=" * 60)
    print(f"\nResults saved to: {abs_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
