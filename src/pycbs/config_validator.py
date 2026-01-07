"""
Configuration Validator: Ensures all required parameters are present
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates configuration sections based on scheme and method"""

    REQUIRED_PARAMS = {
        'USTE1':  {
            'required_all': ['scheme', 'method', 'basis1', 'basis2', 'HF1', 'HF2', 'E1', 'E2'],
            'required_if': {}
        },
        'USTE2': {
            'required_all': ['scheme', 'method', 'basis1', 'basis2', 'basis3', 'basis4', 'HF1', 'HF2', 'E1', 'E2'],
            'required_if': {}
        },
        'USPE': {
            'required_all': ['scheme', 'method', 'basis', 'HF', 'Etot'],
            'optional':  ['constant'],
            'required_if': {}
        },
        'TENSORIAL': {
            'required_all': ['scheme', 'method', 'basis1', 'basis2'],
            'required_if': {
                'USPE': ['zeta_HF1', 'zeta_E1'],
                'USTE1':  ['zeta_HF1', 'zeta_HF2', 'zeta_E1', 'zeta_E2'],
            },
            'optional': ['dc_scheme', 'constant']
        },
        'FREQUENCY': {
            'required_all': ['scheme', 'method', 'basis1', 'basis2', 'HF1', 'HF2', 'F1', 'F2'],
            'required_if': {}
        },
        'FELLER': {
            'required_all': ['scheme', 'Ehf_X', 'Ehf_Y', 'X', 'Y'],
            'optional': ['alfa'],
        },
        'HF_E': {
            'required_all': ['scheme', 'HF1', 'HF2', 'basis1', 'basis2'],
        },
        'TRUHLAR_HF': {
            'required_all': ['scheme', 'Ehf_X', 'Ehf_Y', 'X', 'Y'],
            'optional': ['alfa'],
        },
        'KLOPPER': {
            'required_all': ['scheme', 'Ehf_X', 'Ehf_Y', 'X', 'Y'],
            'optional': ['alfa'],
        },
        'JENSEN': {
            'required_all': ['scheme', 'Ehf_X', 'Ehf_Y', 'X', 'Y'],
            'optional': ['alfa'],
        },
        'BAKOULES': {
            'required_all': ['scheme', 'Ec_X', 'Ec_Y', 'X', 'Y'],
            'optional': ['beta'],
        },
        'OAN': {
            'required_all': ['scheme', 'Ec_X', 'Ec_Y'],
            'optional': ['beta'],
        },
        'TRUHLAR_CORR': {
            'required_all': ['scheme', 'Ec_X', 'Ec_Y', 'X', 'Y'],
            'optional': ['beta'],
        },
        'MARTIN': {
            'required_all': ['scheme', 'Ec_X', 'Ec_Y', 'X', 'Y'],
            'optional': ['beta'],
        },
        'HALKIER_HELGAKER': {
            'required_all': ['scheme', 'Ec_X', 'Ec_Y', 'X', 'Y'],
        },
        'HUH_LEE': {
            'required_all': ['scheme', 'Ec_X', 'Ec_Y', 'X', 'Y'],
            'optional': ['beta'],
        },
    }

    @classmethod
    def validate_section(cls, section_name: str, section_dict: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a configuration section.

        Args:
            section_name: Name of the configuration section
            section_dict: Dictionary of key-value pairs from the section

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        scheme = section_dict.get('scheme', '').upper().strip()
        method = section_dict.get('method', '').upper().strip() if 'method' in section_dict else None

        if not scheme:
            errors.append("Missing 'scheme' parameter")
            return False, errors

        if scheme not in cls.REQUIRED_PARAMS and scheme not in ['FELLER', 'HF_E', 'TRUHLAR_HF', 'KLOPPER', 'JENSEN',
                                                                   'BAKOULES', 'OAN', 'TRUHLAR_CORR', 'MARTIN',
                                                                   'HALKIER_HELGAKER', 'HUH_LEE']:
            errors.append(f"Unknown scheme:  '{scheme}'")
            return False, errors

        # Get validation rules for this scheme
        validation_rules = cls.REQUIRED_PARAMS. get(scheme, {})

        # Check required parameters
        for param in validation_rules.get('required_all', []):
            if param not in section_dict or not str(section_dict[param]).strip():
                errors.append(f"Missing required parameter: '{param}' for scheme '{scheme}'")

        # Check method-specific requirements (if method exists)
        if method:
            required_if_dict = validation_rules.get('required_if', {})
            if method in required_if_dict:
                for param in required_if_dict[method]:
                    if param not in section_dict or not str(section_dict[param]).strip():
                        errors.append(f"Missing required parameter for method '{method}': '{param}'")

        # Validate numeric parameters
        numeric_params_to_check = [
            'HF1', 'HF2', 'E1', 'E2', 'F1', 'F2', 'HF', 'Etot',
            'Ehf_X', 'Ehf_Y', 'Ec_X', 'Ec_Y', 'X', 'Y',
            'zeta_HF1', 'zeta_HF2', 'zeta_E1', 'zeta_E2', 'zeta_HF', 'zeta_E',
            'alfa', 'beta'
        ]

        for param in numeric_params_to_check:
            if param in section_dict:
                try:
                    float(section_dict[param])
                except (ValueError, TypeError):
                    errors.append(f"Parameter '{param}' must be numeric, got: '{section_dict[param]}'")

        is_valid = len(errors) == 0
        return is_valid, errors