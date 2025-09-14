"""
Performance Benchmark Framework for SOM vs SKlearn Clustering Algorithms

This package provides comprehensive benchmarking tools to compare 
SOM_plus_clustering algorithms with scikit-learn clustering algorithms
across multiple performance metrics including speed, memory usage, 
clustering quality, and evaluation scores.
"""

__version__ = "1.0.0"
__author__ = "Performance Analysis Team"

# Import modules that can be imported without issues
try:
    from .core.benchmark_runner import BenchmarkRunner
    from .core.experiment_config import ExperimentConfig
    from .utils.dataset_loader import DatasetLoader
    from .metrics.performance_metrics import PerformanceMetrics
    
    __all__ = [
        "BenchmarkRunner",
        "ExperimentConfig", 
        "DatasetLoader",
        "PerformanceMetrics"
    ]
except ImportError:
    # If imports fail, provide a minimal interface
    __all__ = []