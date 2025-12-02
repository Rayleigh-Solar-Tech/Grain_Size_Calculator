"""
Results processing module for grain size analysis.
Handles statistics calculation, data export, and results visualization.
"""

import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class StatisticalSummary:
    """Statistical summary for measurements."""
    n_samples: int
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    percentile_25: float
    percentile_75: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'n_samples': self.n_samples,
            'mean': self.mean,
            'median': self.median,
            'std': self.std,
            'min': self.min_val,
            'max': self.max_val,
            'percentile_25': self.percentile_25,
            'percentile_75': self.percentile_75
        }


class ResultsProcessor:
    """Process and analyze grain measurement results."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize results processor.
        
        Args:
            output_dir: Directory for saving results
        """
        self.output_dir = output_dir or "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def calculate_statistics(self, values: List[float]) -> StatisticalSummary:
        """
        Calculate comprehensive statistics for a list of values.
        
        Args:
            values: List of measurement values
            
        Returns:
            StatisticalSummary object
        """
        if not values:
            return StatisticalSummary(0, float('nan'), float('nan'), float('nan'),
                                    float('nan'), float('nan'), float('nan'), float('nan'))
        
        arr = np.array(values, dtype=float)
        
        return StatisticalSummary(
            n_samples=len(arr),
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
            percentile_25=float(np.percentile(arr, 25)),
            percentile_75=float(np.percentile(arr, 75))
        )
    
    def process_variant_results(self, variant_name: str, grain_data: List[Dict],
                              apply_cap: bool = True, feret_cap_um: float = 5.0) -> Dict[str, Any]:
        """
        Process results for a single analysis variant.
        
        Args:
            variant_name: Name of the analysis variant
            grain_data: List of grain measurement dictionaries
            apply_cap: Whether to apply Feret diameter cap
            feret_cap_um: Maximum Feret diameter in microns
            
        Returns:
            Processed results dictionary
        """
        # Extract measurements
        all_chords_um = [grain.get('length_um', 0) for grain in grain_data]
        all_areas_um2 = [grain.get('area_um2', 0) for grain in grain_data]
        
        # Apply cap if requested
        if apply_cap:
            filtered_indices = [i for i, chord in enumerate(all_chords_um) 
                              if chord <= feret_cap_um]
            chords_used = [all_chords_um[i] for i in filtered_indices]
            areas_used = [all_areas_um2[i] for i in filtered_indices]
        else:
            chords_used = all_chords_um
            areas_used = all_areas_um2
        
        # Calculate statistics
        chord_stats = self.calculate_statistics(chords_used)
        area_stats = self.calculate_statistics(areas_used)
        
        return {
            'variant_name': variant_name,
            'total_grains': len(grain_data),
            'grains_used': len(chords_used),
            'cap_applied': apply_cap,
            'feret_cap_um': feret_cap_um if apply_cap else None,
            'chord_statistics': chord_stats,
            'area_statistics': area_stats,
            'raw_chords_um': chords_used,
            'raw_areas_um2': areas_used
        }
    
    def combine_variant_results(self, variant_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine results from multiple variants into overall statistics.
        
        Args:
            variant_results: List of variant result dictionaries
            
        Returns:
            Combined results dictionary
        """
        # Pool all measurements
        pooled_chords = []
        pooled_areas = []
        
        for result in variant_results:
            pooled_chords.extend(result.get('raw_chords_um', []))
            pooled_areas.extend(result.get('raw_areas_um2', []))
        
        # Calculate overall statistics
        overall_chord_stats = self.calculate_statistics(pooled_chords)
        overall_area_stats = self.calculate_statistics(pooled_areas)
        
        return {
            'total_variants': len(variant_results),
            'total_grains_pooled': len(pooled_chords),
            'overall_chord_statistics': overall_chord_stats,
            'overall_area_statistics': overall_area_stats,
            'per_variant_summary': [
                {
                    'variant': result['variant_name'],
                    'grains_used': result['grains_used'],
                    'mean_chord_um': result['chord_statistics'].mean,
                    'mean_area_um2': result['area_statistics'].mean
                }
                for result in variant_results
            ]
        }


class ResultsExporter:
    """Export results to various formats."""
    
    def __init__(self, base_output_path: str):
        """
        Initialize exporter.
        
        Args:
            base_output_path: Base path for output files (without extension)
        """
        self.base_path = base_output_path
        self.output_dir = os.path.dirname(base_output_path)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def export_variant_chord_summary(self, variant_results: List[Dict[str, Any]]) -> str:
        """
        Export per-variant chord length summary to CSV.
        
        Args:
            variant_results: List of variant result dictionaries
            
        Returns:
            Path to exported CSV file
        """
        csv_path = f"{self.base_path}_per_variant_chords.csv"
        
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['variant', 'n_used', 'mean_chord_um']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in variant_results:
                stats = result['chord_statistics']
                writer.writerow({
                    'variant': result['variant_name'],
                    'n_used': stats.n_samples,
                    'mean_chord_um': stats.mean
                })
        
        return csv_path
    
    def export_variant_area_summary(self, variant_results: List[Dict[str, Any]]) -> str:
        """
        Export per-variant area summary to CSV.
        
        Args:
            variant_results: List of variant result dictionaries
            
        Returns:
            Path to exported CSV file
        """
        csv_path = f"{self.base_path}_per_variant_areas.csv"
        
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['variant', 'n_used', 'mean_area_um2']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in variant_results:
                stats = result['area_statistics']
                writer.writerow({
                    'variant': result['variant_name'],
                    'n_used': stats.n_samples,
                    'mean_area_um2': stats.mean
                })
        
        return csv_path
    
    def export_overall_summary(self, combined_results: Dict[str, Any], 
                             processing_config: Dict[str, Any]) -> str:
        """
        Export overall analysis summary to CSV.
        
        Args:
            combined_results: Combined results dictionary
            processing_config: Processing configuration
            
        Returns:
            Path to exported CSV file
        """
        csv_path = f"{self.base_path}_overall_averages.csv"
        
        chord_stats = combined_results['overall_chord_statistics']
        area_stats = combined_results['overall_area_statistics']
        
        with open(csv_path, 'w', newline='') as f:
            fieldnames = [
                'apply_cap', 'feret_cap_um', 'overall_mean_chord_um', 'overall_mean_area_um2',
                'n_chords', 'n_areas'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            writer.writerow({
                'apply_cap': int(processing_config.get('apply_feret_cap', True)),
                'feret_cap_um': processing_config.get('feret_cap_um', 5.0),
                'overall_mean_chord_um': chord_stats.mean,
                'overall_mean_area_um2': area_stats.mean,
                'n_chords': chord_stats.n_samples,
                'n_areas': area_stats.n_samples
            })
        
        return csv_path
    
    def export_detailed_json(self, variant_results: List[Dict[str, Any]], 
                           combined_results: Dict[str, Any], 
                           metadata: Dict[str, Any] = None) -> str:
        """
        Export detailed results to JSON format.
        
        Args:
            variant_results: List of variant result dictionaries
            combined_results: Combined results dictionary
            metadata: Additional metadata
            
        Returns:
            Path to exported JSON file
        """
        json_path = f"{self.base_path}_detailed_results.json"
        
        # Prepare data for JSON serialization
        def prepare_for_json(obj):
            if isinstance(obj, StatisticalSummary):
                return obj.to_dict()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: prepare_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [prepare_for_json(item) for item in obj]
            else:
                return obj
        
        data = {
            'metadata': metadata or {},
            'variant_results': prepare_for_json(variant_results),
            'combined_results': prepare_for_json(combined_results),
            'export_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return json_path
    
    def export_all_formats(self, variant_results: List[Dict[str, Any]], 
                          combined_results: Dict[str, Any], 
                          processing_config: Dict[str, Any],
                          metadata: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Export results in all available formats.
        
        Args:
            variant_results: List of variant result dictionaries
            combined_results: Combined results dictionary
            processing_config: Processing configuration
            metadata: Additional metadata
            
        Returns:
            Dictionary mapping format names to file paths
        """
        exported_files = {}
        
        try:
            exported_files['chord_summary'] = self.export_variant_chord_summary(variant_results)
        except Exception as e:
            print(f"Error exporting chord summary: {e}")
        
        try:
            exported_files['area_summary'] = self.export_variant_area_summary(variant_results)
        except Exception as e:
            print(f"Error exporting area summary: {e}")
        
        try:
            exported_files['overall_summary'] = self.export_overall_summary(combined_results, processing_config)
        except Exception as e:
            print(f"Error exporting overall summary: {e}")
        
        try:
            exported_files['detailed_json'] = self.export_detailed_json(variant_results, combined_results, metadata)
        except Exception as e:
            print(f"Error exporting detailed JSON: {e}")
        
        return exported_files


class ResultsVisualizer:
    """Create visualizations for analysis results."""
    
    def __init__(self, output_dir: str):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory for saving visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_distribution_plots(self, variant_results: List[Dict[str, Any]], 
                                base_filename: str) -> List[str]:
        """
        Create distribution plots for chord lengths and areas.
        
        Args:
            variant_results: List of variant result dictionaries
            base_filename: Base filename for plots
            
        Returns:
            List of created plot file paths
        """
        plot_files = []
        
        # Chord length distribution
        plt.figure(figsize=(12, 8))
        for result in variant_results:
            chords = result.get('raw_chords_um', [])
            if chords:
                plt.hist(chords, bins=30, alpha=0.6, label=result['variant_name'])
        
        plt.xlabel('Chord Length (µm)')
        plt.ylabel('Frequency')
        plt.title('Chord Length Distribution by Variant')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        chord_plot_path = os.path.join(self.output_dir, f"{base_filename}_chord_distribution.png")
        plt.savefig(chord_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(chord_plot_path)
        
        # Area distribution
        plt.figure(figsize=(12, 8))
        for result in variant_results:
            areas = result.get('raw_areas_um2', [])
            if areas:
                plt.hist(areas, bins=30, alpha=0.6, label=result['variant_name'])
        
        plt.xlabel('Area (µm²)')
        plt.ylabel('Frequency')
        plt.title('Area Distribution by Variant')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        area_plot_path = os.path.join(self.output_dir, f"{base_filename}_area_distribution.png")
        plt.savefig(area_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(area_plot_path)
        
        return plot_files
    
    def create_summary_comparison(self, variant_results: List[Dict[str, Any]], 
                                base_filename: str) -> str:
        """
        Create comparison plot of variant statistics.
        
        Args:
            variant_results: List of variant result dictionaries
            base_filename: Base filename for plot
            
        Returns:
            Path to created plot file
        """
        variant_names = [result['variant_name'] for result in variant_results]
        mean_chords = [result['chord_statistics'].mean for result in variant_results]
        mean_areas = [result['area_statistics'].mean for result in variant_results]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Chord length comparison
        bars1 = ax1.bar(variant_names, mean_chords, alpha=0.7, color='skyblue')
        ax1.set_ylabel('Mean Chord Length (µm)')
        ax1.set_title('Mean Chord Length by Variant')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, mean_chords):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(mean_chords),
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Area comparison
        bars2 = ax2.bar(variant_names, mean_areas, alpha=0.7, color='lightcoral')
        ax2.set_ylabel('Mean Area (µm²)')
        ax2.set_xlabel('Variant')
        ax2.set_title('Mean Area by Variant')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, mean_areas):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(mean_areas),
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        comparison_plot_path = os.path.join(self.output_dir, f"{base_filename}_variant_comparison.png")
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return comparison_plot_path


def create_complete_results_processor(output_dir: str, base_filename: str):
    """
    Create complete results processing setup.
    
    Args:
        output_dir: Output directory
        base_filename: Base filename for outputs
        
    Returns:
        tuple: (ResultsProcessor, ResultsExporter, ResultsVisualizer)
    """
    base_path = os.path.join(output_dir, base_filename)
    
    processor = ResultsProcessor(output_dir)
    exporter = ResultsExporter(base_path)
    visualizer = ResultsVisualizer(output_dir)
    
    return processor, exporter, visualizer