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
        'BAKOWIES': {
            'required_all': ['scheme', 'Ec_X', 'Ec_Y','Ehf_X', 'Ehf_Y', 'X', 'Y'],
            'optional': ['beta'],
        },
        'OANc': {
            'required_all': ['scheme', 'Ec_X', 'Ec_Y','Ehf_X', 'Ehf_Y'],
            'optional': ['beta'],
        },
        'TRUHLAR_CORR': {
            'required_all': ['scheme', 'Ec_X', 'Ec_Y','Ehf_X', 'Ehf_Y', 'X', 'Y'],
            'optional': ['beta'],
        },
        'MARTIN': {
            'required_all': ['scheme', 'Ec_X', 'Ec_Y','Ehf_X', 'Ehf_Y', 'X', 'Y'],
            'optional': ['beta'],
        },
        'HALKIER_HELGAKER': {
            'required_all': ['scheme', 'Ec_X', 'Ec_Y','Ehf_X', 'Ehf_Y', 'X', 'Y'],
        },
        'HUH_LEE': {
            'required_all': ['scheme', 'Ec_X', 'Ec_Y','Ehf_X', 'Ehf_Y', 'X', 'Y'],
            'optional': ['beta'],
        },
    }

    @classmethod
    def validate_section(cls, section_name: str, section_dict: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a configuration section.

        This validator is case-insensitive for keys: it lower-cases incoming keys,
        then compares against required parameter names case-insensitively.
        """
        errors = []

        # Normalize incoming keys to lowercase for robust validation
        sec = {str(k).strip().lower(): v for k, v in section_dict.items()}

        scheme = str(sec.get('scheme', '')).upper().strip()
        method = sec.get('method')
        method_up = str(method).upper().strip() if method is not None else None

        if not scheme:
            errors.append("Missing 'scheme' parameter")
            return False, errors

        # Allowed schemes set = keys of REQUIRED_PARAMS plus explicit single-function schemes
        allowed = set(cls.REQUIRED_PARAMS.keys()) | {
            'FELLER', 'HF_E', 'TRUHLAR_HF', 'KLOPPER', 'JENSEN',
            'BAKOWIES', 'OANc', 'TRUHLAR_CORR', 'MARTIN',
            'HALKIER_HELGAKER', 'HUH_LEE'
        }

        if scheme not in allowed:
            errors.append(f"Unknown scheme: '{scheme}'")
            return False, errors

        # Get validation rules for this scheme (if present)
        validation_rules = cls.REQUIRED_PARAMS.get(scheme, {})

        # Check required parameters (case-insensitive)
        for param in validation_rules.get('required_all', []):
            if param.lower() not in sec or not str(sec.get(param.lower(), '')).strip():
                errors.append(f"Missing required parameter: '{param}' for scheme '{scheme}'")

        # Check method-specific required_if (case-insensitive)
        if method_up:
            required_if_dict = validation_rules.get('required_if', {})
            req_for_method = required_if_dict.get(method_up, [])
            for param in req_for_method:
                if param.lower() not in sec or not str(sec.get(param.lower(), '')).strip():
                    errors.append(f"Missing required parameter for method '{method_up}': '{param}'")

        # Validate numeric parameters (case-insensitive)
        numeric_params_to_check = [
            'HF1', 'HF2', 'E1', 'E2', 'F1', 'F2', 'HF', 'Etot',
            'Ehf_X', 'Ehf_Y', 'Ec_X', 'Ec_Y', 'X', 'Y',
            'zeta_HF1', 'zeta_HF2', 'zeta_E1', 'zeta_E2', 'zeta_HF', 'zeta_E',
            'alfa', 'beta'
        ]

        for param in numeric_params_to_check:
            key = param.lower()
            if key in sec and sec[key] is not None and str(sec[key]).strip() != '':
                try:
                    float(sec[key])
                except (ValueError, TypeError):
                    errors.append(f"Parameter '{param}' must be numeric, got: '{sec[key]}'")

        is_valid = len(errors) == 0
        return is_valid, errors
