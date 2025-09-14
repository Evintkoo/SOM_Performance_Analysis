"""
Experiment configuration management for performance benchmarking.
Defines and manages benchmark experiment parameters and settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """
    Configuration class for performance benchmarking experiments.
    """
    
    # Experiment identification
    experiment_name: str = "SOM_vs_Sklearn_Benchmark"
    experiment_description: str = "Comprehensive performance comparison between SOM and sklearn clustering algorithms"
    
    # Dataset configuration
    datasets_root: Optional[str] = None
    dataset_categories: List[str] = field(default_factory=lambda: ["all"])  # "all" means use all available
    max_samples_per_dataset: Optional[int] = 5000  # Limit for memory management
    normalize_features: bool = True
    has_labels: bool = False  # Whether datasets contain true labels
    label_column: Union[str, int] = -1  # Column index or name for labels (default: last column)
    
    # Algorithm configuration
    include_som_algorithms: bool = True
    include_sklearn_algorithms: bool = True
    som_epochs: int = 100
    som_excluded_methods: List[str] = field(default_factory=lambda: ["kde", "kmeans", "kmeans++"])
    
    # Performance metrics configuration
    measure_speed: bool = True
    measure_memory: bool = True
    measure_quality: bool = True
    measure_evaluation: bool = True  # When true labels available
    
    # Quality metrics to compute
    quality_metrics: List[str] = field(default_factory=lambda: [
        "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"
    ])
    
    # Evaluation metrics to compute (when true labels available)
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "adjusted_rand_index", "normalized_mutual_info", "homogeneity", 
        "completeness", "v_measure"
    ])
    
    # Experiment control
    random_state: int = 42
    n_runs: int = 1  # Number of runs per algorithm-dataset combination
    timeout_seconds: float = 300.0  # Timeout per algorithm run
    
    # Multiple runs configuration
    aggregate_runs: bool = True  # Whether to aggregate results across runs
    aggregation_methods: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max", "median"])
    save_individual_runs: bool = True  # Save individual run results
    statistical_significance: bool = False  # Compute statistical tests between algorithms
    confidence_level: float = 0.95  # Confidence level for statistical tests
    
    # Output configuration
    results_dir: str = "results"
    save_models: bool = False
    save_predictions: bool = True
    save_detailed_results: bool = True
    
    # Parallel processing
    n_jobs: int = 1  # Currently not implemented, reserved for future use
    
    # Logging configuration
    log_level: str = "INFO"
    log_to_file: bool = True
    log_style: str = "detailed"  # "simple", "detailed", "json"
    log_performance: bool = True
    log_progress: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate configuration
        self._validate_config()
        
        # Setup paths
        self.results_dir = Path(self.results_dir)
        if self.datasets_root:
            self.datasets_root = Path(self.datasets_root)
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.n_runs < 1:
            raise ValueError("n_runs must be >= 1")
        
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        
        if self.som_epochs < 1:
            raise ValueError("som_epochs must be >= 1")
        
        if self.max_samples_per_dataset and self.max_samples_per_dataset < 10:
            raise ValueError("max_samples_per_dataset must be >= 10 or None")
        
        if self.confidence_level <= 0 or self.confidence_level >= 1:
            raise ValueError("confidence_level must be between 0 and 1")
        
        valid_aggregation_methods = ["mean", "std", "min", "max", "median", "q25", "q75"]
        invalid_methods = [m for m in self.aggregation_methods if m not in valid_aggregation_methods]
        if invalid_methods:
            raise ValueError(f"Invalid aggregation methods: {invalid_methods}. "
                           f"Valid methods: {valid_aggregation_methods}")
        
        valid_log_styles = ["simple", "detailed", "json"]
        if self.log_style not in valid_log_styles:
            raise ValueError(f"log_style must be one of {valid_log_styles}")
        
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif hasattr(value, '__dict__'):
                config_dict[key] = value.__dict__
            else:
                config_dict[key] = value
        return config_dict
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Convert string paths back to Path objects
        if 'results_dir' in config_dict:
            config_dict['results_dir'] = str(config_dict['results_dir'])
        if 'datasets_root' in config_dict and config_dict['datasets_root']:
            config_dict['datasets_root'] = str(config_dict['datasets_root'])
        
        logger.info(f"Configuration loaded from {filepath}")
        return cls(**config_dict)
    
    def create_run_config(self, algorithm_name: str, dataset_name: str, 
                         run_number: int) -> Dict[str, Any]:
        """
        Create a configuration for a specific algorithm-dataset run.
        
        Args:
            algorithm_name: Name of the algorithm
            dataset_name: Name of the dataset
            run_number: Run number
            
        Returns:
            Dictionary with run-specific configuration
        """
        return {
            'experiment_name': self.experiment_name,
            'algorithm_name': algorithm_name,
            'dataset_name': dataset_name,
            'run_number': run_number,
            'random_state': self.random_state + run_number,  # Ensure reproducibility
            'timeout_seconds': self.timeout_seconds,
            'normalize_features': self.normalize_features,
            'som_epochs': self.som_epochs,
            'quality_metrics': self.quality_metrics.copy(),
            'evaluation_metrics': self.evaluation_metrics.copy(),
            'measure_speed': self.measure_speed,
            'measure_memory': self.measure_memory,
            'measure_quality': self.measure_quality,
            'measure_evaluation': self.measure_evaluation
        }
    
    def get_output_paths(self) -> Dict[str, Path]:
        """Get all output file paths."""
        base_dir = self.results_dir
        
        paths = {
            'base_dir': base_dir,
            'raw_results': base_dir / 'raw_results',
            'processed_results': base_dir / 'processed_results',
            'visualizations': base_dir / 'visualizations',
            'reports': base_dir / 'reports',
            'models': base_dir / 'models',
            'predictions': base_dir / 'predictions',
            'logs': base_dir / 'logs',
            'configs': base_dir / 'configs'
        }
        
        # Create directories
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        return paths
    
    def summary(self) -> str:
        """Get a human-readable summary of the configuration."""
        summary_parts = [
            f"Experiment: {self.experiment_name}",
            f"Description: {self.experiment_description}",
            f"",
            f"Dataset Configuration:",
            f"  - Categories: {self.dataset_categories}",
            f"  - Max samples per dataset: {self.max_samples_per_dataset}",
            f"  - Normalize features: {self.normalize_features}",
            f"",
            f"Algorithm Configuration:",
            f"  - Include SOM: {self.include_som_algorithms}",
            f"  - Include sklearn: {self.include_sklearn_algorithms}",
            f"  - SOM epochs: {self.som_epochs}",
            f"  - SOM excluded methods: {self.som_excluded_methods}",
            f"",
            f"Metrics Configuration:",
            f"  - Speed: {self.measure_speed}",
            f"  - Memory: {self.measure_memory}",
            f"  - Quality: {self.measure_quality}",
            f"  - Evaluation: {self.measure_evaluation}",
            f"",
            f"Multiple Runs Configuration:",
            f"  - Number of runs: {self.n_runs}",
            f"  - Aggregate runs: {self.aggregate_runs}",
            f"  - Aggregation methods: {self.aggregation_methods}",
            f"  - Save individual runs: {self.save_individual_runs}",
            f"  - Statistical significance: {self.statistical_significance}",
            f"",
            f"Experiment Control:",
            f"  - Random state: {self.random_state}",
            f"  - Timeout: {self.timeout_seconds}s",
            f"",
            f"Logging Configuration:",
            f"  - Level: {self.log_level}",
            f"  - Style: {self.log_style}",
            f"  - Log to file: {self.log_to_file}",
            f"  - Log performance: {self.log_performance}",
            f"  - Log progress: {self.log_progress}",
            f"",
            f"Output Configuration:",
            f"  - Results directory: {self.results_dir}",
            f"  - Save models: {self.save_models}",
            f"  - Save predictions: {self.save_predictions}",
            f"  - Save detailed results: {self.save_detailed_results}"
        ]
        
        return "\n".join(summary_parts)


# Predefined configuration templates
class ConfigTemplates:
    """Predefined configuration templates for common use cases."""
    
    @staticmethod
    def quick_test() -> ExperimentConfig:
        """Configuration for quick testing."""
        return ExperimentConfig(
            experiment_name="Quick_Test",
            experiment_description="Quick test run with limited datasets and algorithms",
            max_samples_per_dataset=500,
            som_epochs=20,
            n_runs=1,
            timeout_seconds=60.0,
            save_models=False
        )
    
    @staticmethod
    def comprehensive() -> ExperimentConfig:
        """Configuration for comprehensive benchmarking."""
        return ExperimentConfig(
            experiment_name="Comprehensive_Benchmark",
            experiment_description="Full comprehensive benchmark across all algorithms and datasets",
            max_samples_per_dataset=None,  # No limit
            som_epochs=200,
            n_runs=5,  # Multiple runs for statistical significance
            aggregate_runs=True,
            aggregation_methods=["mean", "std", "min", "max", "median", "q25", "q75"],
            statistical_significance=True,
            timeout_seconds=600.0,
            save_models=True,
            save_detailed_results=True,
            save_individual_runs=True,
            log_performance=True,
            log_progress=True
        )
    
    @staticmethod
    def memory_constrained() -> ExperimentConfig:
        """Configuration for memory-constrained environments."""
        return ExperimentConfig(
            experiment_name="Memory_Constrained_Benchmark",
            experiment_description="Benchmark optimized for limited memory environments",
            max_samples_per_dataset=1000,
            som_epochs=50,
            n_runs=2,
            timeout_seconds=180.0,
            save_models=False,
            measure_memory=True
        )
    
    @staticmethod
    def speed_focused() -> ExperimentConfig:
        """Configuration focused on speed benchmarking."""
        return ExperimentConfig(
            experiment_name="Speed_Focused_Benchmark",
            experiment_description="Benchmark focused on algorithm speed comparison",
            max_samples_per_dataset=2000,
            som_epochs=50,
            n_runs=10,  # More runs for better speed statistics
            aggregate_runs=True,
            aggregation_methods=["mean", "std", "median"],
            statistical_significance=True,
            timeout_seconds=120.0,
            measure_speed=True,
            measure_memory=False,
            measure_quality=False,
            save_models=False,
            save_individual_runs=True,
            log_performance=True,
            log_progress=True
        )