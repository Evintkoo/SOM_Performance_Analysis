"""
Performance metrics module for comprehensive benchmarking of clustering algorithms.
Measures speed, memory usage, clustering quality, and evaluation scores.
"""

import time
import tracemalloc
import psutil
import gc
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    adjusted_mutual_info_score, homogeneity_score, completeness_score,
    v_measure_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.metrics.cluster import fowlkes_mallows_score
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Comprehensive performance measurement for clustering algorithms.
    
    Measures:
    - Speed: Training time, prediction time
    - Memory: Peak memory usage, memory efficiency  
    - Quality: Internal clustering metrics (silhouette, calinski-harabasz, etc.)
    - Evaluation: External metrics when true labels available (ARI, NMI, etc.)
    """
    
    def __init__(self):
        """Initialize the performance metrics collector."""
        self.reset()
    
    def reset(self):
        """Reset all collected metrics."""
        self.metrics = {
            'speed': {},
            'memory': {},
            'quality': {},
            'evaluation': {},
            'general': {}
        }
        self._start_time = None
        self._memory_peak = 0
        self._memory_start = 0
    
    def start_timing(self):
        """Start timing measurement."""
        # Force garbage collection before measurement
        gc.collect()
        
        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()
        self._memory_start = process.memory_info().rss / 1024 / 1024  # MB
        
        # Start time measurement
        self._start_time = time.perf_counter()
        
    def stop_timing(self, operation_name: str) -> float:
        """
        Stop timing and record the elapsed time.
        
        Args:
            operation_name: Name of the operation being timed
            
        Returns:
            Elapsed time in seconds
        """
        if self._start_time is None:
            raise RuntimeError("Timing not started. Call start_timing() first.")
        
        # Stop time measurement
        elapsed_time = time.perf_counter() - self._start_time
        self.metrics['speed'][operation_name] = elapsed_time
        
        # Stop memory tracking
        try:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Also get process memory
            process = psutil.Process()
            memory_end = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_end - self._memory_start
            
            self.metrics['memory'][f'{operation_name}_peak_mb'] = peak / 1024 / 1024
            self.metrics['memory'][f'{operation_name}_used_mb'] = max(memory_used, 0)
            
        except Exception as e:
            logger.warning(f"Memory tracking failed: {e}")
            self.metrics['memory'][f'{operation_name}_peak_mb'] = 0
            self.metrics['memory'][f'{operation_name}_used_mb'] = 0
        
        # Reset timing
        self._start_time = None
        
        return elapsed_time
    
    def measure_clustering_quality(self, X: np.ndarray, 
                                  labels: np.ndarray, 
                                  algorithm_name: str) -> Dict[str, float]:
        """
        Measure internal clustering quality metrics.
        
        Args:
            X: Feature matrix
            labels: Predicted cluster labels
            algorithm_name: Name of the algorithm
            
        Returns:
            Dictionary of quality metrics
        """
        quality_metrics = {}
        
        # Check if we have valid clustering results
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters <= 1:
            logger.warning(f"Only {n_clusters} cluster(s) found for {algorithm_name}")
            # Return default values for single cluster
            quality_metrics.update({
                'silhouette_score': -1.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': float('inf'),
                'n_clusters': n_clusters,
                'n_noise_points': np.sum(labels == -1) if -1 in labels else 0
            })
        else:
            try:
                # Silhouette Score (higher is better, range: [-1, 1])
                if len(X) > 1 and n_clusters > 1:
                    sil_score = silhouette_score(X, labels)
                    quality_metrics['silhouette_score'] = sil_score
                else:
                    quality_metrics['silhouette_score'] = -1.0
                
                # Calinski-Harabasz Index (higher is better)
                if n_clusters > 1:
                    ch_score = calinski_harabasz_score(X, labels)
                    quality_metrics['calinski_harabasz_score'] = ch_score
                else:
                    quality_metrics['calinski_harabasz_score'] = 0.0
                
                # Davies-Bouldin Index (lower is better)
                if n_clusters > 1:
                    db_score = davies_bouldin_score(X, labels)
                    quality_metrics['davies_bouldin_score'] = db_score
                else:
                    quality_metrics['davies_bouldin_score'] = float('inf')
                
                # Cluster statistics
                quality_metrics['n_clusters'] = n_clusters
                quality_metrics['n_noise_points'] = np.sum(labels == -1) if -1 in labels else 0
                
            except Exception as e:
                logger.error(f"Error computing quality metrics for {algorithm_name}: {e}")
                quality_metrics.update({
                    'silhouette_score': -1.0,
                    'calinski_harabasz_score': 0.0,
                    'davies_bouldin_score': float('inf'),
                    'n_clusters': n_clusters,
                    'n_noise_points': np.sum(labels == -1) if -1 in labels else 0
                })
        
        # Store in metrics
        for metric, value in quality_metrics.items():
            self.metrics['quality'][f'{algorithm_name}_{metric}'] = value
        
        return quality_metrics
    
    def measure_evaluation_scores(self, true_labels: np.ndarray, 
                                 predicted_labels: np.ndarray,
                                 algorithm_name: str) -> Dict[str, float]:
        """
        Measure external evaluation metrics when true labels are available.
        
        Args:
            true_labels: Ground truth labels
            predicted_labels: Predicted cluster labels  
            algorithm_name: Name of the algorithm
            
        Returns:
            Dictionary of evaluation metrics
        """
        eval_metrics = {}
        
        try:
            # Adjusted Rand Index (higher is better, range: [-1, 1])
            ari = adjusted_rand_score(true_labels, predicted_labels)
            eval_metrics['adjusted_rand_index'] = ari
            
            # Normalized Mutual Information (higher is better, range: [0, 1])
            nmi = normalized_mutual_info_score(true_labels, predicted_labels)
            eval_metrics['normalized_mutual_info'] = nmi
            
            # Adjusted Mutual Information (higher is better, range: [-1, 1])
            ami = adjusted_mutual_info_score(true_labels, predicted_labels)
            eval_metrics['adjusted_mutual_info'] = ami
            
            # Homogeneity (higher is better, range: [0, 1])
            homogeneity = homogeneity_score(true_labels, predicted_labels)
            eval_metrics['homogeneity'] = homogeneity
            
            # Completeness (higher is better, range: [0, 1])
            completeness = completeness_score(true_labels, predicted_labels)
            eval_metrics['completeness'] = completeness
            
            # V-measure (higher is better, range: [0, 1])
            v_measure = v_measure_score(true_labels, predicted_labels)
            eval_metrics['v_measure'] = v_measure
            
            # Fowlkes-Mallows Score (higher is better, range: [0, 1])
            fmi = fowlkes_mallows_score(true_labels, predicted_labels)
            eval_metrics['fowlkes_mallows'] = fmi
            
        except Exception as e:
            logger.error(f"Error computing evaluation metrics for {algorithm_name}: {e}")
            # Return default values
            eval_metrics = {
                'adjusted_rand_index': 0.0,
                'normalized_mutual_info': 0.0,
                'adjusted_mutual_info': 0.0,
                'homogeneity': 0.0,
                'completeness': 0.0,
                'v_measure': 0.0,
                'fowlkes_mallows': 0.0
            }
        
        # Store in metrics
        for metric, value in eval_metrics.items():
            self.metrics['evaluation'][f'{algorithm_name}_{metric}'] = value
        
        return eval_metrics
    
    def measure_memory_efficiency(self, X: np.ndarray, model: Any, 
                                 algorithm_name: str) -> Dict[str, float]:
        """
        Measure memory efficiency metrics.
        
        Args:
            X: Input data
            model: Trained model
            algorithm_name: Name of the algorithm
            
        Returns:
            Dictionary of memory efficiency metrics
        """
        memory_metrics = {}
        
        try:
            # Data size in MB
            data_size_mb = X.nbytes / 1024 / 1024
            memory_metrics['data_size_mb'] = data_size_mb
            
            # Model size estimation (if possible)
            try:
                import pickle
                model_size_mb = len(pickle.dumps(model)) / 1024 / 1024
                memory_metrics['model_size_mb'] = model_size_mb
            except:
                memory_metrics['model_size_mb'] = 0.0
            
            # Memory efficiency ratio (data size / peak memory)
            peak_memory = self.metrics['memory'].get(f'{algorithm_name}_peak_mb', 0)
            if peak_memory > 0:
                memory_metrics['memory_efficiency'] = data_size_mb / peak_memory
            else:
                memory_metrics['memory_efficiency'] = 0.0
            
        except Exception as e:
            logger.error(f"Error computing memory efficiency for {algorithm_name}: {e}")
            memory_metrics = {
                'data_size_mb': 0.0,
                'model_size_mb': 0.0,
                'memory_efficiency': 0.0
            }
        
        # Store in metrics
        for metric, value in memory_metrics.items():
            self.metrics['memory'][f'{algorithm_name}_{metric}'] = value
        
        return memory_metrics
    
    def measure_scalability(self, fit_times: List[float], 
                           sample_sizes: List[int],
                           algorithm_name: str) -> Dict[str, float]:
        """
        Measure algorithm scalability based on timing data.
        
        Args:
            fit_times: List of fitting times
            sample_sizes: List of corresponding sample sizes
            algorithm_name: Name of the algorithm
            
        Returns:
            Dictionary of scalability metrics
        """
        scalability_metrics = {}
        
        try:
            if len(fit_times) >= 2 and len(sample_sizes) >= 2:
                # Compute time complexity coefficient (approximate)
                # Assuming T = a * N^b, we estimate b
                log_times = np.log(fit_times)
                log_sizes = np.log(sample_sizes)
                
                # Linear regression to estimate complexity
                coeffs = np.polyfit(log_sizes, log_times, 1)
                time_complexity_exp = coeffs[0]
                
                scalability_metrics['time_complexity_exponent'] = time_complexity_exp
                
                # Time growth rate (ratio of last to first)
                time_growth = fit_times[-1] / fit_times[0] if fit_times[0] > 0 else float('inf')
                size_growth = sample_sizes[-1] / sample_sizes[0] if sample_sizes[0] > 0 else float('inf')
                
                scalability_metrics['time_growth_ratio'] = time_growth
                scalability_metrics['size_growth_ratio'] = size_growth
                scalability_metrics['scalability_efficiency'] = size_growth / time_growth if time_growth > 0 else 0
                
            else:
                scalability_metrics = {
                    'time_complexity_exponent': 1.0,
                    'time_growth_ratio': 1.0,
                    'size_growth_ratio': 1.0,
                    'scalability_efficiency': 1.0
                }
                
        except Exception as e:
            logger.error(f"Error computing scalability metrics for {algorithm_name}: {e}")
            scalability_metrics = {
                'time_complexity_exponent': 1.0,
                'time_growth_ratio': 1.0,
                'size_growth_ratio': 1.0,
                'scalability_efficiency': 1.0
            }
        
        # Store in metrics
        for metric, value in scalability_metrics.items():
            self.metrics['general'][f'{algorithm_name}_{metric}'] = value
        
        return scalability_metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all collected metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        return {
            'speed_metrics': self.metrics['speed'],
            'memory_metrics': self.metrics['memory'],
            'quality_metrics': self.metrics['quality'],
            'evaluation_metrics': self.metrics['evaluation'],
            'general_metrics': self.metrics['general']
        }
    
    def get_algorithm_summary(self, algorithm_name: str) -> Dict[str, Any]:
        """
        Get metrics summary for a specific algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Dictionary of metrics for the specified algorithm
        """
        summary = {}
        
        # Collect metrics for this algorithm
        for category, metrics in self.metrics.items():
            algo_metrics = {k.replace(f'{algorithm_name}_', ''): v 
                           for k, v in metrics.items() 
                           if k.startswith(f'{algorithm_name}_')}
            if algo_metrics:
                summary[category] = algo_metrics
        
        return summary
    
    @staticmethod
    def compare_algorithms(metrics_dict: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Compare metrics across multiple algorithms.
        
        Args:
            metrics_dict: Dictionary mapping algorithm names to their metrics
            
        Returns:
            Comparison summary with rankings and relative performance
        """
        comparison = {}
        
        # Collect all metric names
        all_metrics = set()
        for algo_metrics in metrics_dict.values():
            for category_metrics in algo_metrics.values():
                all_metrics.update(category_metrics.keys())
        
        # For each metric, rank algorithms
        for metric in all_metrics:
            metric_values = {}
            for algo_name, algo_metrics in metrics_dict.items():
                for category_metrics in algo_metrics.values():
                    if metric in category_metrics:
                        metric_values[algo_name] = category_metrics[metric]
            
            if metric_values:
                # Determine if higher is better based on metric name
                higher_is_better = any(term in metric.lower() for term in [
                    'silhouette', 'calinski', 'rand', 'mutual', 'homogeneity',
                    'completeness', 'v_measure', 'fowlkes', 'efficiency'
                ])
                
                # Sort algorithms by metric value
                sorted_algos = sorted(metric_values.items(), 
                                    key=lambda x: x[1], 
                                    reverse=higher_is_better)
                
                comparison[metric] = {
                    'ranking': [algo for algo, _ in sorted_algos],
                    'values': dict(sorted_algos),
                    'best': sorted_algos[0][0],
                    'worst': sorted_algos[-1][0]
                }
        
        return comparison