# writer.py (debe tener este contenido para que funcione)
import citations

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

# Variables globales
GENERAL_CITATION_WRITTEN = False
RESULTS_SUMMARY = []  # Almacena los resultados para la tabla final

def write_header(filename):
    """Escribe el encabezado con logo e información institucional"""
    global GENERAL_CITATION_WRITTEN, RESULTS_SUMMARY
    RESULTS_SUMMARY = []  # Reiniciar la tabla para nuevo cálculo
    with open(filename, "w") as f:
        f.write(LOGO)
        f.write("\n\n")
        f.write("             pyCBS: Complete Basis Set Extrapolation Tool\n")
        f.write("\n\n")
        f.write(INFO_BLOCK)
        f.write("\n\n")
        
        if not GENERAL_CITATION_WRITTEN:
            write_general_citation(f)
            GENERAL_CITATION_WRITTEN = True

def write_general_citation(file):
    """Escribe la cita general del programa"""
    file.write("\n" + "=" * 70 + "\n")
    file.write("                          CITATION INFORMATION\n")
    file.write("=" * 70 + "\n\n")
    
    default_citations = citations.CITATIONS.get("DEFAULT", [])
    if default_citations:
        file.write("Please cite this program as:\n\n")
        for ref in default_citations:
            file.write(f"   {ref}\n")
        file.write("\n")

def write_scheme_citations(file, scheme):
    """Escribe citas específicas del esquema si son necesarias"""
    specific_citations = citations.CITATIONS.get(scheme, [])
    if specific_citations:
        file.write(f"Additionally, cite the following references when using the {scheme} scheme:\n\n")
        for ref in specific_citations:
            file.write(f"   {ref}\n")
        file.write("\n")

def write_result(filename, scheme, data, EHF=None, dc=None, energy=None):
    """
    Escribe resultados en el archivo de salida y almacena para el resumen
    """
    global RESULTS_SUMMARY
    with open(filename, "a") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"                       Extrapolation Scheme: {scheme}\n")
        f.write("=" * 70 + "\n")
        
        write_scheme_citations(f, scheme)
        
        f.write("Input Parameters:\n")
        f.write("-" * 70 + "\n")
        for key, value in data.items():
            f.write(f"{key:>20}: {value}\n")
        
        f.write("-" * 70 + "\n")
        f.write("Extrapolation Results:\n")
        f.write("-" * 70 + "\n")
        
        if scheme in ["USTE1", "USTE2"]:
            f.write(f"{'Hartree-Fock (CBS):':>25} {EHF:.10f}\n")
            f.write(f"{'Dynamic Correlation:':>25} {dc:.10f}\n")
            f.write(f"{'Total CBS Energy:':>25} {energy:.10f}\n")
        else:
            f.write(f"{'CBS Extrapolated Energy:':>25} {energy:.10f}\n")
        f.write("=" * 70 + "\n\n")
    
    # Almacenar resultados para la tabla final
    RESULTS_SUMMARY.append({
        'scheme': scheme,
        'energy': energy,
        'EHF': EHF,
        'dc': dc
    })

def write_summary_table(filename):
    """Escribe una tabla resumen con todos los resultados al final del archivo"""
    global RESULTS_SUMMARY
    if not RESULTS_SUMMARY:
        return
    
    with open(filename, "a") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("                      SUMMARY OF RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        # Cabecera para esquemas USTE
        f.write(f"{'Scheme':<10}{'HF (CBS)':>20}{'Dynamic Corr.':>20}{'Total Energy':>20}\n")
        f.write("-" * 70 + "\n")
        
        for item in RESULTS_SUMMARY:
            if item['scheme'] in ["USTE1", "USTE2"]:
                f.write(f"{item['scheme']:<10}{item['EHF']:>20.10f}{item['dc']:>20.10f}{item['energy']:>20.10f}\n")
            else:
                f.write(f"{item['scheme']:<10}{'':>20}{'':>20}{item['energy']:>20.10f}\n")
        
        f.write("\n" + "=" * 70 + "\n")