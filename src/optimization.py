# optimization.py  (añade/pega esto en tu módulo)
import numpy as np
from pyscf import gto, scf, cc, lib, mp
import os
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm

# -------------------------
# Defaults
# -------------------------
DEFAULTS = {
    'init_parameters': [0.96654, 103.93761],
    'geo_init': ['r1', 'teta'],
    'basis_sets': ['cc-pvtz', 'cc-pvqz'],
    'METHOD': 'CCSD(T)',
    'x1': 2.792,
    'x2': 3.719,
    'x1_hf': 2.96,
    'x2_hf': 3.87,
    'beta': 1.62,
    'maxcycle': 20,
    'energy_criterion': 1e-8,
    'fac_mult': 0.05,
    'cut': 0.75,
    # resources defaults
    'max_workers': max(1, cpu_count() - 1),
    'pyscf_threads': max(1, (max(1, cpu_count() - 1)) // 2),
}

# Utility: normalize basis list from config if necessary
def _parse_basis_list(basis_val):
    if isinstance(basis_val, (list, tuple)):
        return basis_val
    s = str(basis_val)
    return [b.strip() for b in s.split(',') if b.strip()]

# -------------------------
# MAIN ENTRYPOINT
# -------------------------
def run_optimization(config_dict=None, output_file=None):
    """
    Ejecuta la optimización.
    config_dict: dict con claves opcionales (cualquiera de DEFAULTS keys).
    output_file: si se proporciona, se escriben resultados adicionales (opcional).
    """
    # merge defaults with user config
    cfg = dict(DEFAULTS)
    if config_dict:
        # normalize keys: accept lower/upper case
        for k, v in config_dict.items():
            # allow keys passed as 'method' -> 'METHOD'
            if k.lower() in [kk.lower() for kk in DEFAULTS.keys()]:
                # find canonical key
                for key in DEFAULTS.keys():
                    if key.lower() == k.lower():
                        cfg[key] = v
                        break
            else:
                cfg[k] = v

    # cast types for important fields
    try:
        init_parameters = np.array(cfg['init_parameters'], dtype=float)
    except Exception:
        init_parameters = np.array(DEFAULTS['init_parameters'], dtype=float)

    geo_init = cfg.get('geo_init', DEFAULTS['geo_init'])
    basis_sets = _parse_basis_list(cfg.get('basis_sets', DEFAULTS['basis_sets']))

    METHOD = str(cfg.get('METHOD', DEFAULTS['METHOD'])).upper()
    x1 = float(cfg.get('x1', DEFAULTS['x1']))
    x2 = float(cfg.get('x2', DEFAULTS['x2']))
    x1_hf = float(cfg.get('x1_hf', DEFAULTS['x1_hf']))
    x2_hf = float(cfg.get('x2_hf', DEFAULTS['x2_hf']))
    beta = float(cfg.get('beta', DEFAULTS['beta']))

    maxcycle = int(cfg.get('maxcycle', DEFAULTS['maxcycle']))
    energy_criterion = float(cfg.get('energy_criterion', DEFAULTS['energy_criterion']))
    fac_mult = float(cfg.get('fac_mult', DEFAULTS['fac_mult']))
    cut = float(cfg.get('cut', DEFAULTS['cut']))

    # Resource tuning
    MAX_WORKERS = int(cfg.get('max_workers', DEFAULTS['max_workers']))
    PYSCF_THREADS = int(cfg.get('pyscf_threads', DEFAULTS['pyscf_threads']))

    # set environment threads before importing heavy libs (already done at top, but set again)
    os.environ['MKL_NUM_THREADS'] = str(PYSCF_THREADS)
    os.environ['OMP_NUM_THREADS'] = str(PYSCF_THREADS)
    os.environ['OPENBLAS_NUM_THREADS'] = str(PYSCF_THREADS)
    lib.num_threads(PYSCF_THREADS)

    # constants for extrapolation (local)
    a_corr = (x1 ** 3) / (x2 ** 3 - x1 ** 3)
    b_hf = (np.exp(beta * x1_hf)) / (np.exp(beta * x2_hf) - np.exp(beta * x1_hf))

    # -------------------------
    # Define inner functions using the local constants
    # (I keep compute_cbs_energy etc inside so they close over a_corr/b_hf, BASIS etc)
    # -------------------------
    def raw_energy(ex1, ex2, ex1hf, ex2hf):
        return ex2 + a_corr * (ex2 - ex1) + (a_corr - b_hf) * (ex2hf - ex1hf)

    def xyz_from_params(params):
        r1, teta = params
        theta_rad = np.deg2rad(teta)
        z_h1 = r1 * np.cos(theta_rad / 2)
        y_h1 = r1 * np.sin(theta_rad / 2)
        z_h2 = r1 * np.cos(theta_rad / 2)
        y_h2 = -r1 * np.sin(theta_rad / 2)
        return f"O 0 0 0\nH 0 {y_h1:.8f} {z_h1:.8f}\nH 0 {y_h2:.8f} {z_h2:.8f}"

    # Use a small local cache keyed by (method, params) to avoid lru_cache issues across processes
    _local_cache = {}

    def compute_cbs_energy_local(method, parameters_tuple):
        # simple memoization
        key = (method, tuple(parameters_tuple))
        if key in _local_cache:
            return _local_cache[key]

        parameters = np.array(parameters_tuple)
        xyz = xyz_from_params(parameters)
        results = {'scf': [], 'corr': []}

        for basis in basis_sets:
            mol = gto.Mole()
            mol.atom = xyz
            mol.basis = basis
            mol.spin = 0
            mol.charge = 0
            mol.nthread = PYSCF_THREADS
            mol.max_memory = 8000
            mol.build()

            mf = scf.RHF(mol)
            mf.max_memory = 14330
            mf.conv_tol = 1e-9
            mf.max_cycle = 100
            scf_energy = mf.kernel()

            corr_energy = None
            if method == 'CCSD(T)':
                mycc = cc.CCSD(mf)
                mycc.conv_tol = 1e-7
                mycc.max_cycle = 100
                mycc.kernel()
                try:
                    et = mycc.ccsd_t()
                except Exception:
                    et = None
                corr_energy = mycc.e_tot + (et if et is not None else 0.0)

            elif method == 'MP2':
                mymp = mp.MP2(mf)
                mymp.max_memory = 14330
                mp_res = mymp.run()
                mp2_total = getattr(mp_res, 'e_tot', None)
                if mp2_total is None:
                    mp2_total = getattr(mymp, 'e_tot', None)
                if mp2_total is None:
                    e_corr = getattr(mp_res, 'e_corr', getattr(mymp, 'e_corr', None))
                    if e_corr is not None:
                        mp2_total = scf_energy + e_corr
                if mp2_total is None:
                    raise RuntimeError("Could not recuperar energía MP2")
                corr_energy = float(mp2_total)
            else:
                raise ValueError("Unknown method: " + str(method))

            results['scf'].append(float(scf_energy))
            results['corr'].append(float(corr_energy))

        energy_cbs = raw_energy(results['corr'][0], results['corr'][1], results['scf'][0], results['scf'][1])
        _local_cache[key] = energy_cbs
        return energy_cbs

    # Parabolic fit (same que tenías)
    def parabolic_minimum(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size < 3 or y.size < 3:
            idx = np.argmin(y)
            return float(x[idx]), float(y[idx])
        try:
            coeffs = np.polyfit(x, y, 2)
            a, b, c = coeffs
            if abs(a) < 1e-20:
                idx = np.argmin(y)
                return float(x[idx]), float(y[idx])
            x_min = -b / (2.0 * a)
            y_min = a * x_min ** 2 + b * x_min + c
            return float(x_min), float(y_min)
        except Exception:
            idx = np.argmin(y)
            return float(x[idx]), float(y[idx])

    def get_displacements(current_param, displacement_factor):
        f = float(displacement_factor)
        return np.array([
            current_param * (1 - 2 * f),
            current_param * (1 - f),
            current_param,
            current_param * (1 + f),
            current_param * (1 + 2 * f)
        ])

    # worker functions that call compute_cbs_energy_local
    def process_single_displacement(args):
        current_params, param_idx, disp = args
        test_params = current_params.copy()
        test_params[param_idx] = disp
        if param_idx == 1 and (disp < 90.0 or disp > 120.0):
            return None
        try:
            energy = compute_cbs_energy_local(METHOD, tuple(test_params))
            return disp, energy
        except Exception as e:
            # bubble up error to caller by returning an informative tuple
            return ('error', str(e))

    def optimize_parameter(args):
        current_params, param_idx, displacement_factor = args
        displacements = get_displacements(current_params[param_idx], displacement_factor)
        param_values = []
        energies = []
        for disp in displacements:
            result = process_single_displacement((current_params, param_idx, disp))
            if result is None:
                continue
            if isinstance(result, tuple) and result[0] == 'error':
                # raise so higher-level knows
                raise RuntimeError(f"Calculation error for disp {disp}: {result[1]}")
            disp_v, energy = result
            param_values.append(disp_v)
            energies.append(energy)
        return param_idx, np.array(param_values), np.array(energies)

    def parallel_optimize_parameters(current_params, cycle, displacement_factor, executor):
        args_list = [(current_params.copy(), idx, displacement_factor) for idx in range(len(geo_init))]
        results = []
        with tqdm(total=len(geo_init), desc=f"Cycle {cycle}") as pbar:
            futures = [executor.submit(optimize_parameter, args) for args in args_list]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                except Exception as e:
                    # propagate: stop optimization and raise
                    raise
        return sorted(results, key=lambda x: x[0])

    # -------------------------
    # Run the optimization loop (main code)
    # -------------------------
    # choose executor; safer to start with ThreadPoolExecutor to avoid process-fork issues with PySCF
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    current_params = init_parameters.copy()
    displacement_factor = fac_mult
    converged = False
    optimization_history = []

    print("=" * 60)
    print("OPTIMIZACIÓN CBS GEOMÉTRICA-PySCF")
    print("=" * 60)
    print(f"Configuración inicial:")
    print(f"  r(OH) = {current_params[0]:.5f} Å")
    print(f"  ∠HOH = {current_params[1]:.5f}°")
    print(f"Método: RHF/{METHOD} con bases {basis_sets}")
    print(f"Criterio convergencia: < {energy_criterion:.1e} Hartree")
    print("=" * 60)

    try:
        for cycle in range(1, maxcycle + 1):
            print(f"\n>>> CICLO {cycle}/{maxcycle}")
            print(f"Factor desplazamiento: {displacement_factor:.6f}")

            optimization_results = parallel_optimize_parameters(
                current_params,
                cycle,
                displacement_factor,
                executor
            )

            for param_idx, param_values, energies in optimization_results:
                param_name = geo_init[param_idx]
                print(f"\n  > Optimizando parámetro: {param_name}")

                if len(energies) >= 3:
                    opt_value, opt_energy = parabolic_minimum(param_values, energies)
                    if param_name == 'r1':
                        opt_value = max(0.8, min(1.2, opt_value))
                    elif param_name == 'teta':
                        opt_value = max(95.0, min(115.0, opt_value))

                    print(f"Mínimo parabólico: {param_name} = {opt_value:.6f}, E = {opt_energy:.10f} Ha")

                    current_energy = compute_cbs_energy_local(METHOD, tuple(current_params))
                    if opt_energy < current_energy:
                        current_params[param_idx] = opt_value
                        print(f"    ACTUALIZADO a {opt_value:.6f}")
                    else:
                        print("    sin mejora")
                else:
                    print("    !! No hay suficientes puntos para ajuste parabólico")

            current_energy = compute_cbs_energy_local(METHOD, tuple(current_params))
            optimization_history.append({
                'cycle': cycle,
                'parameters': current_params.copy(),
                'energy': current_energy,
                'displacement_factor': displacement_factor
            })

            if cycle > 1:
                energy_diff = abs(optimization_history[-2]['energy'] - optimization_history[-1]['energy'])
                print(f"\n  ΔE desde último ciclo: {energy_diff:.4e} Ha")
                if energy_diff < energy_criterion:
                    print("\n" + "=" * 60)
                    print("CONVERGENCIA ALCANZADA!")
                    converged = True
                    break

            displacement_factor *= cut
            print(f"  Nuevo factor desplazamiento: {displacement_factor:.6f}")

    finally:
        executor.shutdown(wait=True)

    # Result printing / optional writing to output_file
    print("\n" + "=" * 60)
    print("RESULTADOS FINALES (optimizer)")
    print("=" * 60)

    final_energy = compute_cbs_energy_local(METHOD, tuple(current_params))

    print("\nGEOMETRÍA OPTIMIZADA:")
    print(f"  r(OH) = {current_params[0]:.8f} Å")
    print(f"  ∠HOH = {current_params[1]:.8f}°")
    print(f"Energía CBS final: {final_energy:.10f} Hartree")

    if output_file:
        # Append results to the output file via writer if available, otherwise plain append
        try:
            import writer
            result_data = {
                'method': METHOD,
                'basis_sets': ','.join(basis_sets),
                'init_parameters': ','.join(map(str, init_parameters))
            }
            writer.write_result(output_file, "OPTIMIZATION", result_data, EHF=None, dc=None, energy=final_energy)
        except Exception:
            with open(output_file, "a") as f:
                f.write("\nOPTIMIZATION RESULT\n")
                f.write(f"r(OH) = {current_params[0]:.8f}\n")
                f.write(f"theta = {current_params[1]:.8f}\n")
                f.write(f"final CBS = {final_energy:.10f}\n")

    # return structure for caller
    return {
        'parameters': current_params.copy(),
        'energy': final_energy,
        'history': optimization_history,
        'converged': converged
    }
