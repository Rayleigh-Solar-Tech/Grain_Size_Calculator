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
    # Model settings - using SAM-B for high precision (correct grain separation)
    model_gpu: str = "sam_b.pt"    # SAM-B for precision (no merged grains)
    model_cpu: str = "sam_b.pt"    # SAM-B even on CPU
    max_side_cpu: int = 1536       # Higher resolution for better accuracy
    
    # Analysis settings
    min_area_px: int = 50
    apply_feret_cap: bool = True
    feret_cap_um: float = 5.0
    
    # Ridge filtering settings (NEW - experimental feature)
    apply_ridge_filtering: bool = True   # Enabled by default with SAM-B
    ridge_threshold: float = 0.15        # Minimum ridge density to accept grain (0.10-0.20 range)
    ridge_tv_weight: float = 0.01        # Total variation denoising weight
    ridge_percentile: int = 70           # Percentile for ridge threshold calculation
    ridge_min_size: int = 50             # Minimum edge object size to keep
    
    # Image tiling settings (for large images)
    enable_tiling: bool = True           # Enable automatic tiling for large images
    tile_size: int = 1024                # Size of each tile in pixels
    tile_overlap: int = 128              # Overlap between tiles to avoid edge artifacts
    min_image_size_for_tiling: int = 2048  # Only tile images larger than this
    
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
    
    # Default analysis variants - optimized experimental profiles
    DEFAULT_VARIANTS = [
        AnalysisVariant("Balanced_Default_clahe_strong_8x8", 4.0, 8, 1.0, 0.7, 1.0),
        AnalysisVariant("Low_Contrast_clahe_strong_4x4", 8.0, 4, 1.0, 0.7, 1.0),
        AnalysisVariant("Low_Contrast_clahe_vstrong_4x4", 14.0, 4, 1.0, 0.7, 1.0),
        AnalysisVariant("Aggressive_clahe_vstrong_8x8", 15.0, 8, 1.0, 0.7, 1.0),
        AnalysisVariant("Aggressive_clahe_ultra_8x8", 20.0, 8, 1.0, 0.7, 1.0),
        AnalysisVariant("Ultra_Aggressive_clahe_ultra_4x4", 30.0, 4, 1.0, 0.7, 1.0),
    ]
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        # Auto-locate default config if not provided
        if config_file is None:
            # Try to find default_config.json relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            config_file = os.path.join(project_root, 'configs', 'default_config.json')
        
        self.config_file = config_file
        self.processing_config = ProcessingConfig()
        self.variants = self.DEFAULT_VARIANTS.copy()
        
        # Auto-load config if file exists
        if self.config_file and os.path.exists(self.config_file):
            self.load_config()
    
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