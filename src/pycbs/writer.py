# writer.py
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

GENERAL_CITATION_WRITTEN = False
RESULTS_SUMMARY = []  # stores per-scheme final results for summary table


def write_header(filename):
    global GENERAL_CITATION_WRITTEN, RESULTS_SUMMARY
    RESULTS_SUMMARY = []
    with open(filename, "w") as f:
        f.write(LOGO)
        f.write("\n\n")
        f.write("             pyCBS: Complete Basis Set Extrapolation Tool\n\n")
        f.write(INFO_BLOCK)
        f.write("\n\n")
        if not GENERAL_CITATION_WRITTEN:
            write_general_citation(f)
            GENERAL_CITATION_WRITTEN = True


def write_general_citation(file):
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
    specific_citations = citations.CITATIONS.get(scheme, [])
    if specific_citations:
        file.write(f"Additionally, cite the following references when using the {scheme} scheme:\n\n")
        for ref in specific_citations:
            file.write(f"   {ref}\n")
        file.write("\n")


def write_result(filename, scheme, data, EHF=None, dc=None, energy=None):
    """
    Write a detailed result block into the results file.

    - If EHF and dc are provided (not None), print Hartree-Fock (CBS), Dynamic Correlation,
      and Total CBS Energy.
    - Otherwise, print a single CBS Extrapolated Energy line (existing behavior).
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

        # If both EHF and dc were provided, print the detailed breakdown
        if (EHF is not None) and (dc is not None):
            try:
                f.write(f"{'Hartree-Fock (CBS):':>25} {float(EHF):.10f}\n")
            except Exception:
                f.write(f"{'Hartree-Fock (CBS):':>25} {EHF}\n")
            try:
                f.write(f"{'Dynamic Correlation:':>25} {float(dc):.10f}\n")
            except Exception:
                f.write(f"{'Dynamic Correlation:':>25} {dc}\n")
            try:
                f.write(f"{'Total CBS Energy:':>25} {float(energy):.10f}\n")
            except Exception:
                f.write(f"{'Total CBS Energy:':>25} {energy}\n")
            has_components = True
        else:
            # Fallback / existing behaviour for items without HF/dc components
            try:
                f.write(f"{'CBS Extrapolated Energy:':>25} {float(energy):.10f}\n")
            except Exception:
                f.write(f"{'CBS Extrapolated Energy:':>25} {energy}\n")
            has_components = False

        f.write("=" * 70 + "\n\n")

    # Store a record for the final summary table. Include a flag whether HF/dc components are present.
    RESULTS_SUMMARY.append({
        'scheme': scheme,
        'energy': energy,
        'EHF': EHF,
        'dc': dc,
        'has_components': has_components
    })


def write_optimization_summary(filename, history):
    """
    Write a dedicated optimization cycle summary into the results file.
    history: list of dicts {'cycle', 'parameters', 'energy', 'displacement_factor'}
    """
    with open(filename, "a") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("                      OPTIMIZATION CYCLE HISTORY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Cycle':>5} {'r(OH)[Å]':>12} {'HOH[°]':>12} {'Energy[Ha]':>18} {'DispFactor':>12}\n")
        f.write("-" * 70 + "\n")
        for step in history:
            cyc = step.get('cycle', 0)
            params = step.get('parameters', [None, None])
            energy = step.get('energy', 0.0)
            df = step.get('displacement_factor', 0.0)
            # protect against numpy arrays or Nones
            try:
                p0 = float(params[0])
            except Exception:
                p0 = 0.0
            try:
                p1 = float(params[1])
            except Exception:
                p1 = 0.0
            try:
                en = float(energy)
            except Exception:
                en = 0.0
            try:
                dff = float(df)
            except Exception:
                dff = 0.0
            f.write(f"{cyc:5d} {p0:12.8f} {p1:12.8f} {en:18.10f} {dff:12.6f}\n")
        f.write("\n" + "=" * 70 + "\n\n")


def write_summary_table(filename):
    """
    Write the summary table. Rows that have has_components=True show HF & Dynamic Corr columns;
    others show only the total energy (but columns preserve alignment).
    """
    global RESULTS_SUMMARY
    if not RESULTS_SUMMARY:
        return
    with open(filename, "a") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("                      SUMMARY OF RESULTS\n")
        f.write("=" * 70 + "\n\n")
        # Header (we always print the three columns for clarity)
        f.write(f"{'Scheme':<20}{'HF (CBS)':>18}{'Dynamic Corr.':>18}{'Total Energy':>18}\n")
        f.write("-" * 70 + "\n")
        for item in RESULTS_SUMMARY:
            scheme = item.get('scheme', '')
            energy = item.get('energy', None)
            EHF = item.get('EHF', None)
            dc = item.get('dc', None)
            has_components = item.get('has_components', False)

            if has_components:
                # print HF, DC and Total energy if available
                try:
                    f.write(f"{scheme:<20}{float(EHF):18.10f}{float(dc):18.10f}{float(energy):18.10f}\n")
                except Exception:
                    # fallback with simple str formatting if floats fail
                    f.write(f"{scheme:<20}{str(EHF):>18}{str(dc):>18}{str(energy):>18}\n")
            else:
                # fill HF and DC columns with blanks, print only total energy
                try:
                    f.write(f"{scheme:<20}{'':18}{'':18}{float(energy):18.10f}\n")
                except Exception:
                    f.write(f"{scheme:<20}{'':18}{'':18}{str(energy):>18}\n")

        f.write("\n" + "=" * 70 + "\n")
