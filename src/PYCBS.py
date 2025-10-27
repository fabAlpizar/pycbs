############################################################################################
##                                pyCBS: CBS Extrapolation Tool                           ##
############################################################################################

import sys
import os
import configparser
import writer

# Importar módulos de extrapolación
try:
    import USTE1
except ImportError:
    print("Error: Módulo USTE1 no encontrado")
    sys.exit(1)

try:
    import USTE2
except ImportError:
    print("Error: Módulo USTE2 no encontrado")
    sys.exit(1)

try:
    import USPE
except ImportError:
    print("Error: Módulo USPE no encontrado")
    sys.exit(1)

try:
    import tensorial_properties1 as TP
except ImportError:
    print("Error: Módulo tensorial_properties1 no encontrado")
    sys.exit(1)

def main():
    # Verificar argumentos
    if len(sys.argv) < 2:
        print("Uso: python PYCBS.py archivo_entrada.inp")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: Archivo de entrada '{input_file}' no encontrado")
        sys.exit(1)

    # Después de leer config:
    config = read_config(input_file)

    # Inicializar archivo de resultados
    output_file = "results.out"
    open(output_file, "w").close()
    writer.write_header(output_file)

    # --- LEER OPCIONES DE OPTIMIZACIÓN (sección OPTIMIZATION) ---
    opt_enabled = False
    opt_params = {}
    if config.has_section('OPTIMIZATION'):
        opt_section = config['OPTIMIZATION']
        opt_enabled = opt_section.get('optimization', 'False').lower() in ('1', 'true', 'yes', 'on')
        # Leer parámetros opcionales y guardarlos en dict (si existen)
        if 'init_parameters' in opt_section:
            val = opt_section.get('init_parameters').split(',')
            opt_params['init_parameters'] = [float(v.strip()) for v in val]
        if 'basis_sets' in opt_section:
            opt_params['basis_sets'] = [v.strip() for v in opt_section.get('basis_sets').split(',')]
        # strings / scalars (use get with fallback)
        for key in ('METHOD', 'x1', 'x2', 'x1_hf', 'x2_hf', 'beta', 'energy_criterion', 'fac_mult', 'cut', 'maxcycle'):
            if key in opt_section:
                sval = opt_section.get(key)
                # try parse numeric where applicable
                try:
                    # interpret booleans or strings accordingly
                    if key in ('METHOD',):
                        opt_params[key.lower()] = sval.strip()
                    elif '.' in sval or 'e' in sval.lower():
                        opt_params[key.lower()] = float(sval)
                    else:
                        opt_params[key.lower()] = int(sval)
                except Exception:
                    # fallback to raw string
                    opt_params[key.lower()] = sval

    # Si la optimización está activada, importar y lanzar el módulo de optimization
    if opt_enabled:
        print("Optimization requested in input file — launching optimizer...")
        try:
            import optimization
        except ImportError as e:
            print("Error: módulo 'optimization' no encontrado o con error:", e)
        else:
            # llamar a la función pública run_optimization del módulo,
            # pasándole el dict opt_params que puede estar vacío (usa defaults).
            try:
                optimization.run_optimization(opt_params, output_file=output_file)
            except Exception as e:
                print("Error al ejecutar la optimización:", e)

    # Procesar cada cálculo como antes
    for section in config.sections():
        # omitimos la sección OPTIMIZATION al procesar JOBs
        if section.upper() == 'OPTIMIZATION':
            continue

        print(f"Processing {section}...", end="", flush=True)
        scheme = config[section].get('scheme', '').upper()

        if scheme == "USTE1":
            run_uste1(config[section], output_file, section)
        elif scheme == "USTE2":
            run_uste2(config[section], output_file, section)
        elif scheme == "USPE":
            run_uspe(config[section], output_file, section)
        elif scheme == "TENSORIAL":
            run_tensorial(config[section], output_file, section)
        else:
            print(f"Error: Esquema '{scheme}' no reconocido en cálculo {section}.")
        print(" DONE")

    # Escribir tabla resumen al final
    writer.write_summary_table(output_file)
    print("\nAll calculations completed successfully!")
    print(f"Results saved to: {output_file}")

def read_config(input_file):
    """Lee y parsea el archivo de configuración con múltiples secciones"""
    config = configparser.ConfigParser()
    config.read(input_file)
    return config

def run_uste1(section, output_file, calc_name):
    """Ejecuta el esquema USTE1 para una sección específica"""
    try:
        method = section['method']
        basis1 = section['basis1']
        basis2 = section['basis2']
        HF1 = float(section['HF1'])
        HF2 = float(section['HF2'])
        E1 = float(section['E1'])
        E2 = float(section['E2'])
    except KeyError as e:
        print(f"\nError en cálculo {calc_name}: Falta parámetro obligatorio - {e}")
        return
    except ValueError as e:
        print(f"\nError en cálculo {calc_name}: Valor numérico inválido - {e}")
        return
    
    # Escribir nombre del cálculo en el archivo de salida
    with open(output_file, "a") as f:
        f.write(f"\n")
        f.write(f" JOB: {calc_name}")
    
    # Realizar extrapolación
    hf_dict, corr_dict = USTE1.dictionaries(method, basis1, basis2)
    Ecr1, Ecr2 = USTE1.correlation_energy(HF1, HF2, E1, E2)
    EHF, dc, CBS = USTE1.CBS_extrapolation(HF1, HF2, Ecr1, Ecr2, corr_dict, basis1, basis2)
    
    # Escribir resultados
    result_data = {
        'method': method,
        'basis1': basis1,
        'basis2': basis2,
        'HF1': HF1,
        'HF2': HF2,
        'E1': E1,
        'E2': E2
    }
    writer.write_result(output_file, "USTE1", result_data, EHF, dc, CBS)

def run_uste2(section, output_file, calc_name):
    """Ejecuta el esquema USTE2 para una sección específica"""
    try:
        method = section['method']
        basis1 = section['basis1']
        basis2 = section['basis2']
        basis3 = section['basis3']
        basis4 = section['basis4']
        HF1 = float(section['HF1'])
        HF2 = float(section['HF2'])
        E1 = float(section['E1'])
        E2 = float(section['E2'])
    except KeyError as e:
        print(f"\nError en cálculo {calc_name}: Falta parámetro obligatorio - {e}")
        return
    except ValueError as e:
        print(f"\nError en cálculo {calc_name}: Valor numérico inválido - {e}")
        return
    
    # Escribir nombre del cálculo
    with open(output_file, "a") as f:
        f.write(f"\n")
        f.write(f" JOB: {calc_name}")
        
    # Realizar extrapolación
    hf_dict, corr_dict = USTE2.dictionaries(method, basis1, basis2, basis3, basis4)
    Ecr1, Ecr2 = USTE2.correlation_energy(HF1, HF2, E1, E2)
    EHF, dc, CBS = USTE2.CBS_extrapolation(HF1, HF2, Ecr1, Ecr2, corr_dict, basis1, basis2, basis3, basis4)
    
    # Escribir resultados
    result_data = {
        'method': method,
        'basis1': basis1,
        'basis2': basis2,
        'basis3': basis3,
        'basis4': basis4,
        'HF1': HF1,
        'HF2': HF2,
        'E1': E1,
        'E2': E2
    }
    writer.write_result(output_file, "USTE2", result_data, EHF, dc, CBS)

def run_uspe(section, output_file, calc_name):
    """Ejecuta el esquema USPE para una sección específica"""
    try:
        method = section['method']
        constant = section['constant']
        basis = section['basis']
        HF = float(section['HF'])
        Etot = float(section['Etot'])
    except KeyError as e:
        print(f"\nError en cálculo {calc_name}: Falta parámetro obligatorio - {e}")
        return
    except ValueError as e:
        print(f"\nError en cálculo {calc_name}: Valor numérico inválido - {e}")
        return
    
    # Escribir nombre del cálculo
    with open(output_file, "a") as f:
        f.write(f"\n")
        f.write(f" JOB: {calc_name}")
    
    # Realizar extrapolación
    resultado = USPE.CBS_extrapolation(HF, Etot, method, constant, basis)
    
    # Escribir resultados
    result_data = {
        'method': method,
        'constant': constant,
        'basis': basis,
        'HF': HF,
        'Etot': Etot
    }
    writer.write_result(output_file, "USPE", result_data, energy=resultado)

def run_tensorial(section, output_file, calc_name):
    """Ejecuta el esquema TENSORIAL (idéntico a USTE1)"""
    try:
        method = section['method']
        basis1 = section['basis1']
        basis2 = section['basis2']
        zeta_HF1 = float(section['zeta_HF1'])
        zeta_HF2 = float(section['zeta_HF2'])
        zeta_E1 = float(section['zeta_E1'])
        zeta_E2 = float(section['zeta_E2'])
    except KeyError as e:
        print(f"\nError en cálculo {calc_name}: Falta parámetro obligatorio - {e}")
        return
    except ValueError as e:
        print(f"\nError en cálculo {calc_name}: Valor numérico inválido - {e}")
        return
    
    # Escribir nombre del cálculo
    with open(output_file, "a") as f:
        f.write(f"\n")
        f.write(f" JOB: {calc_name}")
    
    # Realizar extrapolación (usando el módulo tensorial_properties1)
    hf_dict, corr_dict = TP.dictionaries(method, basis1, basis2)
    zeta_cor1, zeta_cor2 = TP.correlation_energy(zeta_HF1, zeta_HF2, zeta_E1, zeta_E2)
    zeta_HF, zeta_cor, zeta = TP.CBS_extrapolation(
        zeta_HF1, zeta_HF2, zeta_cor1, zeta_cor2, corr_dict, basis1, basis2
    )
    
    # Escribir resultados
    result_data = {
        'method': method,
        'basis1': basis1,
        'basis2': basis2,
        'zeta_HF1': zeta_HF1,
        'zeta_HF2': zeta_HF2,
        'zeta_E1': zeta_E1,
        'zeta_E2': zeta_E2
    }
    writer.write_result(output_file, "TENSORIAL", result_data, zeta_HF, zeta_cor, zeta)

if __name__ == "__main__":
    main()
