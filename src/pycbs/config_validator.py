"""
Configuration Validator:  Ensures all required parameters are present
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates configuration sections based on scheme and method"""

    # Required parameters per scheme
    REQUIRED_PARAMS = {
        'USTE1': {
            'required_all': ['scheme', 'method', 'basis1', 'basis2', 'HF1', 'HF2', 'E1', 'E2'],
            'required_if': {
                'MP2': [],
                'CCSD(T)': [],
                'MP2+CCSD(T)': [],
            }
        },
        'USTE2': {
            'required_all': ['scheme', 'method', 'basis1', 'basis2', 'basis3', 'basis4', 'HF1', 'HF2', 'E1', 'E2'],
            'required_if': {}
        },
        'USPE': {
            'required_all': ['scheme', 'method', 'basis', 'HF', 'Etot'],
            'optional': ['constant'],
            'required_if': {}
        },
        'TENSORIAL': {
            'required_all': ['scheme', 'method', 'basis1', 'basis2'],
            'required_if': {
                'USPE': ['zeta_HF1', 'zeta_E1'],
                'USTE1': ['zeta_HF1', 'zeta_HF2', 'zeta_E1', 'zeta_E2'],
            },
            'optional': ['dc_scheme', 'constant']
        },
        'FREQUENCY': {
            'required_all': ['scheme', 'method', 'basis1', 'basis2', 'HF1', 'HF2', 'F1', 'F2'],
            'required_if': {}
        }
    }

    @classmethod
    def validate_section(cls, section_name: str, section_dict: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a configuration section.

        Args:
            section_name: Name of the configuration section (e.g., 'Calculation1')
            section_dict: Dictionary of key-value pairs from the section

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        scheme = section_dict.get('scheme', '').upper().strip()
        method = section_dict.get('method', '').upper().strip()

        if not scheme:
            errors.append("Missing 'scheme' parameter")
            return False, errors

        if not method:
            errors.append("Missing 'method' parameter")
            return False, errors

        if scheme not in cls.REQUIRED_PARAMS:
            errors.append(f"Unknown scheme: '{scheme}'. Valid schemes: {', '.join(cls.REQUIRED_PARAMS.keys())}")
            return False, errors

        validation_rules = cls.REQUIRED_PARAMS[scheme]

        # Check required parameters
        for param in validation_rules.get('required_all', []):
            if param not in section_dict or not str(section_dict[param]).strip():
                errors.append(f"Missing required parameter: '{param}' for scheme '{scheme}'")

        # Check method-specific requirements
        required_if_dict = validation_rules.get('required_if', {})
        if method in required_if_dict:
            for param in required_if_dict[method]:
                if param not in section_dict or not str(section_dict[param]).strip():
                    errors.append(f"Missing required parameter for method '{method}': '{param}'")

        # Validate numeric parameters
        numeric_params = [k for k, v in section_dict.items()
                          if k.lower() in ['hf1', 'hf2', 'e1', 'e2', 'f1', 'f2', 'hf', 'etot',
                                           'zeta_hf1', 'zeta_hf2', 'zeta_e1', 'zeta_e2', 'zeta_hf', 'zeta_e']]

        for param in numeric_params:
            try:
                float(section_dict[param])
            except (ValueError, TypeError):
                errors.append(f"Parameter '{param}' must be numeric, got: '{section_dict[param]}'")

        # Validate basis sets are known
        basis_params = [k for k in section_dict.keys() if k.lower().startswith('basis')]
        for basis_param in basis_params:
            # This is best-effort; we can expand the basis validation later
            pass

        is_valid = len(errors) == 0
        return is_valid, errors