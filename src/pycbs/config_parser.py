"""Configuration parser for INI files with methodology selection support.

This module provides functionality to parse INI configuration files and
handle methodology selection for the pyCBS application.
"""

import configparser
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when there is an error in configuration parsing or validation."""

    pass


class ConfigParser:
    """Parser for INI configuration files with methodology selection support.

    This class handles reading, parsing, and validating INI configuration files
    with special support for methodology selection and section management.

    Attributes:
        config_path: Path to the configuration file.
        parser: Internal ConfigParser instance.
        methodology: Selected methodology for processing.
    """

    # Default methodology selection
    DEFAULT_METHODOLOGY = "default"

    # Valid sections in configuration
    VALID_SECTIONS = {
        "general",
        "methodology",
        "paths",
        "processing",
        "output",
        "logging",
    }

    def __init__(self, config_path: str | Path) -> None:
        """Initialize the configuration parser.

        Args:
            config_path: Path to the INI configuration file.

        Raises:
            ConfigurationError: If the config file does not exist.
        """
        self.config_path = Path(config_path)
        self.parser = configparser.ConfigParser()
        self.methodology: Optional[str] = None

        if not self.config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {self.config_path}"
            )

        self._load_config()

    def _load_config(self) -> None:
        """Load and parse the configuration file.

        Raises:
            ConfigurationError: If there is an error parsing the configuration file.
        """
        try:
            files_read = self.parser.read(self.config_path)
            if not files_read:
                raise ConfigurationError(
                    f"Failed to read configuration file: {self.config_path}"
                )
            logger.info(f"Configuration loaded from {self.config_path}")
        except configparser.Error as e:
            raise ConfigurationError(f"Error parsing configuration file: {e}") from e

    def get_methodology(self) -> str:
        """Get the selected methodology from configuration.

        Returns:
            The selected methodology name.

        Raises:
            ConfigurationError: If methodology is not properly configured.
        """
        if self.methodology is None:
            try:
                self.methodology = self.parser.get(
                    "methodology", "selected", fallback=self.DEFAULT_METHODOLOGY
                )
                logger.info(f"Selected methodology: {self.methodology}")
            except configparser.Error as e:
                raise ConfigurationError(f"Error reading methodology: {e}") from e

        return self.methodology

    def set_methodology(self, methodology: str) -> None:
        """Set the methodology for processing.

        Args:
            methodology: Name of the methodology to use.

        Raises:
            ConfigurationError: If the methodology is not valid.
        """
        available = self.get_available_methodologies()
        if methodology not in available:
            raise ConfigurationError(
                f"Invalid methodology '{methodology}'. "
                f"Available: {', '.join(available)}"
            )

        self.methodology = methodology
        logger.info(f"Methodology changed to: {methodology}")

    def get_available_methodologies(self) -> List[str]:
        """Get list of available methodologies in configuration.

        Returns:
            List of available methodology names.
        """
        methodologies = []
        if self.parser.has_section("methodology"):
            for option in self.parser.options("methodology"):
                if option != "selected":
                    methodologies.append(option)

        return sorted(methodologies) if methodologies else [self.DEFAULT_METHODOLOGY]

    def get_methodology_config(self, methodology: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific methodology.

        Args:
            methodology: Name of the methodology. If None, uses selected methodology.

        Returns:
            Dictionary of methodology configuration parameters.

        Raises:
            ConfigurationError: If the methodology section is not found.
        """
        if methodology is None:
            methodology = self.get_methodology()

        if not self.parser.has_section("methodology"):
            return {}

        try:
            config_dict = {}
            for key in self.parser.options("methodology"):
                if key == "selected":
                    continue
                value = self.parser.get("methodology", key)
                config_dict[key] = self._parse_value(value)
            return config_dict
        except configparser.Error as e:
            raise ConfigurationError(
                f"Error reading methodology '{methodology}' config: {e}"
            ) from e

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get all values from a configuration section.

        Args:
            section: Name of the section to retrieve.

        Returns:
            Dictionary of section configuration.

        Raises:
            ConfigurationError: If the section does not exist.
        """
        if not self.parser.has_section(section):
            raise ConfigurationError(f"Configuration section not found: {section}")

        try:
            section_dict = {}
            for key in self.parser.options(section):
                value = self.parser.get(section, key)
                section_dict[key] = self._parse_value(value)
            return section_dict
        except configparser.Error as e:
            raise ConfigurationError(f"Error reading section '{section}': {e}") from e

    def get_value(
        self,
        section: str,
        key: str,
        fallback: Any = None,
        value_type: type = str,
    ) -> Any:
        """Get a specific configuration value.

        Args:
            section: Section name.
            key: Configuration key.
            fallback: Default value if key is not found.
            value_type: Expected type for the value (str, int, float, bool).

        Returns:
            The configuration value, converted to the specified type.

        Raises:
            ConfigurationError: If the value cannot be converted to the specified type.
        """
        if not self.parser.has_section(section):
            if fallback is not None:
                return fallback
            raise ConfigurationError(f"Configuration section not found: {section}")

        if not self.parser.has_option(section, key):
            if fallback is not None:
                return fallback
            raise ConfigurationError(
                f"Configuration key not found: {section}.{key}"
            )

        try:
            value = self.parser.get(section, key)

            if value_type == bool:
                return self.parser.getboolean(section, key)
            elif value_type == int:
                return self.parser.getint(section, key)
            elif value_type == float:
                return self.parser.getfloat(section, key)
            else:
                return value

        except (ValueError, configparser.Error) as e:
            if fallback is not None:
                logger.warning(
                    f"Could not parse {section}.{key}, using fallback: {e}"
                )
                return fallback
            raise ConfigurationError(
                f"Error parsing configuration value {section}.{key}: {e}"
            ) from e

    def get_paths(self) -> Dict[str, Path]:
        """Get all path configurations as Path objects.

        Returns:
            Dictionary of path configurations.

        Raises:
            ConfigurationError: If the paths section is not found.
        """
        paths_config = self.get_section("paths")
        return {key: Path(value) for key, value in paths_config.items()}

    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration.

        Returns:
            Dictionary of processing parameters.

        Raises:
            ConfigurationError: If the processing section is not found.
        """
        return self.get_section("processing")

    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration.

        Returns:
            Dictionary of output parameters.

        Raises:
            ConfigurationError: If the output section is not found.
        """
        return self.get_section("output")

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration.

        Returns:
            Dictionary of logging parameters.

        Raises:
            ConfigurationError: If the logging section is not found.
        """
        return self.get_section("logging")

    def validate_configuration(self) -> bool:
        """Validate the configuration for required sections and values.

        Returns:
            True if configuration is valid.

        Raises:
            ConfigurationError: If required configuration is missing.
        """
        try:
            # Check for required sections
            required_sections = {"general", "methodology"}
            available_sections = set(self.parser.sections())

            missing_sections = required_sections - available_sections
            if missing_sections:
                raise ConfigurationError(
                    f"Missing required sections: {', '.join(missing_sections)}"
                )

            # Validate methodology selection
            self.get_methodology()

            logger.info("Configuration validation successful")
            return True

        except ConfigurationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    def get_all_sections(self) -> Dict[str, Dict[str, Any]]:
        """Get all sections and their values from the configuration.

        Returns:
            Dictionary with section names as keys and their configurations as values.
        """
        all_config = {}
        for section in self.parser.sections():
            try:
                all_config[section] = self.get_section(section)
            except ConfigurationError:
                logger.warning(f"Could not read section: {section}")

        return all_config

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "config_path": str(self.config_path),
            "methodology": self.get_methodology(),
            "sections": self.get_all_sections(),
        }

    @staticmethod
    def _parse_value(value: str) -> Any:
        """Parse a configuration value to its appropriate type.

        Args:
            value: String value to parse.

        Returns:
            Parsed value (str, int, float, bool, or list).
        """
        if not isinstance(value, str):
            return value

        # Try to parse as boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Try to parse as integer
        try:
            if "." not in value:
                return int(value)
        except ValueError:
            pass

        # Try to parse as float
        try:
            return float(value)
        except ValueError:
            pass

        # Try to parse as list (comma-separated)
        if "," in value:
            return [item.strip() for item in value.split(",")]

        return value

    def __repr__(self) -> str:
        """Return string representation of the parser."""
        return (
            f"ConfigParser(config_path={self.config_path}, "
            f"methodology={self.get_methodology()})"
        )


class MethodologyRegistry:
    """Registry for managing available methodologies and their configurations."""

    def __init__(self) -> None:
        """Initialize the methodology registry."""
        self._methodologies: Dict[str, Dict[str, Any]] = {}
        self._aliases: Dict[str, str] = {}

    def register_methodology(
        self,
        name: str,
        config: Dict[str, Any],
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Register a new methodology.

        Args:
            name: Name of the methodology.
            config: Configuration dictionary for the methodology.
            aliases: Alternative names for this methodology.

        Raises:
            ValueError: If methodology name is already registered.
        """
        if name in self._methodologies:
            raise ValueError(f"Methodology '{name}' is already registered")

        self._methodologies[name] = config

        if aliases:
            for alias in aliases:
                if alias in self._aliases:
                    raise ValueError(f"Alias '{alias}' is already in use")
                self._aliases[alias] = name

        logger.info(f"Registered methodology: {name}")

    def get_methodology(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a registered methodology configuration.

        Args:
            name: Name or alias of the methodology.

        Returns:
            Configuration dictionary, or None if not found.
        """
        resolved_name = self._aliases.get(name, name)
        return self._methodologies.get(resolved_name)

    def get_available_methodologies(self) -> List[str]:
        """Get list of all registered methodologies.

        Returns:
            Sorted list of methodology names.
        """
        return sorted(self._methodologies.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a methodology is registered.

        Args:
            name: Name or alias of the methodology.

        Returns:
            True if methodology is registered.
        """
        resolved_name = self._aliases.get(name, name)
        return resolved_name in self._methodologies

    def __repr__(self) -> str:
        """Return string representation of the registry."""
        return (
            f"MethodologyRegistry(methodologies={len(self._methodologies)}, "
            f"aliases={len(self._aliases)})"
        )
