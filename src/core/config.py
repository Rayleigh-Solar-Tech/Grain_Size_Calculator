"""
Configuration management for grain size analysis.
Handles analysis variants, user settings, and default parameters.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class AnalysisVariant:
    """Configuration for a single analysis variant."""
    name: str
    clip: float = 1.3          # CLAHE clip limit
    tile: int = 8              # CLAHE tile grid size
    gamma: float = 1.0         # Gamma correction
    unsharp_amount: float = 0.7  # Unsharp masking amount (us in original)
    unsharp_sigma: float = 1.0   # Unsharp masking sigma
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisVariant':
        """Create from dictionary."""
        # Handle legacy 'us' parameter name
        if 'us' in data:
            data['unsharp_amount'] = data.pop('us')
        if 'sigma' in data:
            data['unsharp_sigma'] = data.pop('sigma')
            
        return cls(**data)


@dataclass
class ProcessingConfig:
    """Main processing configuration."""
    # Model settings - using large model for best accuracy
    model_gpu: str = "sam_l.pt"    # Large model for best accuracy
    model_cpu: str = "sam_l.pt"    # Large model even on CPU
    max_side_cpu: int = 1536       # Higher resolution for better accuracy
    
    # Analysis settings
    min_area_px: int = 50
    apply_feret_cap: bool = True
    feret_cap_um: float = 5.0
    
    # Output settings
    save_overlays: bool = True
    save_contact_sheet: bool = False
    annotate_measurements: bool = True
    
    # Default frame width (can be overridden by OCR)
    default_frame_width_um: float = 21.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingConfig':
        """Create from dictionary."""
        return cls(**data)


class ConfigManager:
    """Manage analysis configurations and variants."""
    
    # Default analysis variants from original code
    DEFAULT_VARIANTS = [
        AnalysisVariant("v22_cl1.3_t12_g0.98_us0.70_s1.1", 1.3, 12, 0.98, 0.70, 1.1),
        AnalysisVariant("v14_cl1.3_t10_g1.00_us0.70_s1.2", 1.3, 10, 1.00, 0.70, 1.2),
        AnalysisVariant("v5_cl1.5_t8_g1.00_us0.80_s1.0", 1.5, 8, 1.00, 0.80, 1.0),
        AnalysisVariant("v10_cl1.5_t8_g0.98_us0.80_s1.0", 1.5, 8, 0.98, 0.80, 1.0),
        AnalysisVariant("v4_cl3_t8_g0.85_us1.50_s1.2", 3.0, 8, 0.85, 1.50, 1.2),
        AnalysisVariant("v2_cl2.5_t8_g0.95_us1.20_s1.0", 2.5, 8, 0.95, 1.20, 1.0),
    ]
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.config_file = config_file
        self.processing_config = ProcessingConfig()
        self.variants = self.DEFAULT_VARIANTS.copy()
    
    def load_config(self, config_file: str = None) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration file
        """
        file_path = config_file or self.config_file
        if not file_path or not os.path.exists(file_path):
            print(f"Config file not found: {file_path}, using defaults")
            return
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Load processing config
            if 'processing' in data:
                self.processing_config = ProcessingConfig.from_dict(data['processing'])
            
            # Load variants
            if 'variants' in data:
                self.variants = [
                    AnalysisVariant.from_dict(variant_data) 
                    for variant_data in data['variants']
                ]
            
            print(f"Configuration loaded from: {file_path}")
            
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
    
    def save_config(self, config_file: str = None) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config_file: Path to save configuration file
        """
        file_path = config_file or self.config_file
        if not file_path:
            raise ValueError("No config file specified")
        
        data = {
            'processing': self.processing_config.to_dict(),
            'variants': [variant.to_dict() for variant in self.variants]
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Configuration saved to: {file_path}")
    
    def add_variant(self, variant: AnalysisVariant) -> None:
        """Add a new analysis variant."""
        self.variants.append(variant)
    
    def remove_variant(self, name: str) -> bool:
        """
        Remove variant by name.
        
        Returns:
            True if variant was found and removed, False otherwise
        """
        for i, variant in enumerate(self.variants):
            if variant.name == name:
                del self.variants[i]
                return True
        return False
    
    def get_variant_by_name(self, name: str) -> Optional[AnalysisVariant]:
        """Get variant by name."""
        for variant in self.variants:
            if variant.name == name:
                return variant
        return None
    
    def update_processing_config(self, **kwargs) -> None:
        """Update processing configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.processing_config, key):
                setattr(self.processing_config, key, value)
            else:
                print(f"Warning: Unknown config parameter: {key}")
    
    def get_variant_names(self) -> List[str]:
        """Get list of all variant names."""
        return [variant.name for variant in self.variants]
    
    def create_custom_variant(self, name: str, **params) -> AnalysisVariant:
        """
        Create a custom variant with specified parameters.
        
        Args:
            name: Variant name
            **params: Variant parameters
            
        Returns:
            New AnalysisVariant instance
        """
        # Start with default values
        default_variant = AnalysisVariant(name)
        
        # Update with provided parameters
        for key, value in params.items():
            if hasattr(default_variant, key):
                setattr(default_variant, key, value)
            else:
                print(f"Warning: Unknown variant parameter: {key}")
        
        return default_variant
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'processing_config': self.processing_config.to_dict(),
            'num_variants': len(self.variants),
            'variant_names': self.get_variant_names()
        }


def create_default_config_file(file_path: str) -> None:
    """Create a default configuration file."""
    config_manager = ConfigManager()
    config_manager.config_file = file_path
    config_manager.save_config()


def load_config_from_file(file_path: str) -> ConfigManager:
    """Load configuration from file."""
    config_manager = ConfigManager(file_path)
    config_manager.load_config()
    return config_manager


# Quick access functions for common configurations
def get_default_processing_config() -> ProcessingConfig:
    """Get default processing configuration."""
    return ProcessingConfig()


def get_default_variants() -> List[AnalysisVariant]:
    """Get default analysis variants."""
    return ConfigManager.DEFAULT_VARIANTS.copy()


def create_simple_variant(name: str, clip: float = 1.3, tile: int = 8, 
                         gamma: float = 1.0, unsharp: float = 0.7) -> AnalysisVariant:
    """Create a simple variant with basic parameters."""
    return AnalysisVariant(
        name=name,
        clip=clip,
        tile=tile,
        gamma=gamma,
        unsharp_amount=unsharp,
        unsharp_sigma=1.0
    )