"""
Scikit-learn clustering algorithms wrapper for performance benchmarking.
Provides a unified interface for various sklearn clustering algorithms.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering,
    MeanShift, Birch, OPTICS, AffinityPropagation
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


class SklearnClusteringWrapper:
    """
    Wrapper class for scikit-learn clustering algorithms.
    Provides unified interface for benchmarking different clustering methods.
    """
    
    def __init__(self):
        """Initialize the sklearn clustering wrapper."""
        self.available_algorithms = {
            'KMeans': self._create_kmeans,
            'DBSCAN': self._create_dbscan,
            'AgglomerativeClustering': self._create_agglomerative,
            'SpectralClustering': self._create_spectral,
            'MeanShift': self._create_meanshift,
            'Birch': self._create_birch,
            'OPTICS': self._create_optics,
            'AffinityPropagation': self._create_affinity_propagation,
            'GaussianMixture': self._create_gaussian_mixture
        }
        
        logger.info(f"Sklearn wrapper initialized with {len(self.available_algorithms)} "
                   f"algorithms: {list(self.available_algorithms.keys())}")
    
    def get_available_configurations(self, n_samples: int = 1000, 
                                   n_features: int = 2) -> List[Dict[str, Any]]:
        """
        Get all available sklearn clustering configurations for benchmarking.
        
        Args:
            n_samples: Number of samples (used for parameter optimization)
            n_features: Number of features (used for parameter optimization)
            
        Returns:
            List of configuration dictionaries
        """
        configurations = []
        
        # Estimate optimal number of clusters based on data size
        optimal_k = max(2, min(20, int(np.sqrt(n_samples / 50))))
        
        for algo_name in self.available_algorithms.keys():
            # Get multiple configurations for each algorithm with different parameters
            if algo_name == 'KMeans':
                for init_method in ['k-means++', 'random']:
                    for k in [optimal_k, optimal_k // 2, optimal_k * 2]:
                        if k >= 2:
                            config = {
                                'algorithm_type': 'sklearn',
                                'algorithm_class': algo_name,
                                'algorithm_name': f'{algo_name}_k{k}_{init_method}',
                                'n_clusters': k,
                                'init': init_method,
                                'n_init': 10,
                                'random_state': 42
                            }
                            configurations.append(config)
            
            elif algo_name == 'DBSCAN':
                # Different eps values for DBSCAN
                for eps in [0.3, 0.5, 0.8]:
                    for min_samples in [3, 5]:
                        config = {
                            'algorithm_type': 'sklearn',
                            'algorithm_class': algo_name,
                            'algorithm_name': f'{algo_name}_eps{eps}_min{min_samples}',
                            'eps': eps,
                            'min_samples': min_samples
                        }
                        configurations.append(config)
            
            elif algo_name == 'AgglomerativeClustering':
                for linkage in ['ward', 'complete', 'average']:
                    for k in [optimal_k, optimal_k // 2]:
                        if k >= 2:
                            config = {
                                'algorithm_type': 'sklearn',
                                'algorithm_class': algo_name,
                                'algorithm_name': f'{algo_name}_k{k}_{linkage}',
                                'n_clusters': k,
                                'linkage': linkage
                            }
                            configurations.append(config)
            
            elif algo_name == 'SpectralClustering':
                for k in [optimal_k]:
                    if k >= 2:
                        config = {
                            'algorithm_type': 'sklearn',
                            'algorithm_class': algo_name,
                            'algorithm_name': f'{algo_name}_k{k}',
                            'n_clusters': k,
                            'random_state': 42,
                            'n_init': 10
                        }
                        configurations.append(config)
            
            elif algo_name == 'Birch':
                for threshold in [0.3, 0.5, 0.8]:
                    for k in [optimal_k]:
                        if k >= 2:
                            config = {
                                'algorithm_type': 'sklearn',
                                'algorithm_class': algo_name,
                                'algorithm_name': f'{algo_name}_k{k}_t{threshold}',
                                'n_clusters': k,
                                'threshold': threshold
                            }
                            configurations.append(config)
            
            elif algo_name == 'OPTICS':
                for min_samples in [3, 5]:
                    config = {
                        'algorithm_type': 'sklearn',
                        'algorithm_class': algo_name,
                        'algorithm_name': f'{algo_name}_min{min_samples}',
                        'min_samples': min_samples,
                        'xi': 0.05
                    }
                    configurations.append(config)
            
            elif algo_name == 'GaussianMixture':
                for k in [optimal_k, optimal_k // 2]:
                    if k >= 2:
                        config = {
                            'algorithm_type': 'sklearn',
                            'algorithm_class': algo_name,
                            'algorithm_name': f'{algo_name}_k{k}',
                            'n_components': k,
                            'random_state': 42
                        }
                        configurations.append(config)
            
            else:
                # Default configuration for other algorithms
                config = {
                    'algorithm_type': 'sklearn',
                    'algorithm_class': algo_name,
                    'algorithm_name': algo_name,
                    'random_state': 42
                }
                configurations.append(config)
        
        return configurations
    
    def create_algorithm_instance(self, config: Dict[str, Any]) -> Any:
        """
        Create an algorithm instance based on configuration.
        
        Args:
            config: Algorithm configuration dictionary
            
        Returns:
            Configured algorithm instance
        """
        algorithm_class = config['algorithm_class']
        
        if algorithm_class not in self.available_algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_class}")
        
        try:
            # Remove non-parameter keys from config
            params = config.copy()
            for key in ['algorithm_type', 'algorithm_class', 'algorithm_name']:
                params.pop(key, None)
            
            # Create algorithm instance
            algorithm = self.available_algorithms[algorithm_class](**params)
            
            logger.info(f"Created {algorithm_class} instance with config: {params}")
            return algorithm
            
        except Exception as e:
            logger.error(f"Failed to create {algorithm_class} instance: {e}")
            raise
    
    def fit_algorithm(self, algorithm: Any, X: np.ndarray, 
                     normalize: bool = True) -> Any:
        """
        Fit algorithm to the data.
        
        Args:
            algorithm: Algorithm instance
            X: Training data
            normalize: Whether to normalize data before fitting
            
        Returns:
            Fitted algorithm instance
        """
        try:
            # Normalize data if requested
            if normalize:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                algorithm._scaler = scaler  # Store for later use
            else:
                X_scaled = X
                algorithm._scaler = None
            
            # Fit the algorithm
            algorithm.fit(X_scaled)
            
            return algorithm
            
        except Exception as e:
            logger.error(f"Error fitting algorithm: {e}")
            raise
    
    def predict_algorithm(self, algorithm: Any, X: np.ndarray) -> np.ndarray:
        """
        Get cluster predictions from fitted algorithm.
        
        Args:
            algorithm: Fitted algorithm instance
            X: Data to predict
            
        Returns:
            Cluster labels
        """
        try:
            # Apply same normalization as training if it was used
            if hasattr(algorithm, '_scaler') and algorithm._scaler is not None:
                X_scaled = algorithm._scaler.transform(X)
            else:
                X_scaled = X
            
            # Get predictions
            if hasattr(algorithm, 'predict'):
                labels = algorithm.predict(X_scaled)
            elif hasattr(algorithm, 'labels_'):
                # For algorithms that don't have predict method
                labels = algorithm.labels_
            elif hasattr(algorithm, 'fit_predict'):
                # For algorithms that only have fit_predict
                labels = algorithm.fit_predict(X_scaled)
            else:
                raise ValueError(f"Algorithm {type(algorithm)} doesn't support prediction")
            
            return labels
            
        except Exception as e:
            logger.error(f"Error predicting with algorithm: {e}")
            raise
    
    def get_cluster_centers(self, algorithm: Any) -> Optional[np.ndarray]:
        """
        Get cluster centers from fitted algorithm (if available).
        
        Args:
            algorithm: Fitted algorithm instance
            
        Returns:
            Cluster centers array or None if not available
        """
        try:
            if hasattr(algorithm, 'cluster_centers_'):
                centers = algorithm.cluster_centers_
                # Transform back to original space if normalization was used
                if hasattr(algorithm, '_scaler') and algorithm._scaler is not None:
                    centers = algorithm._scaler.inverse_transform(centers)
                return centers
            elif hasattr(algorithm, 'means_'):  # GaussianMixture
                centers = algorithm.means_
                if hasattr(algorithm, '_scaler') and algorithm._scaler is not None:
                    centers = algorithm._scaler.inverse_transform(centers)
                return centers
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not get cluster centers: {e}")
            return None
    
    def get_algorithm_info(self, algorithm: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get information about the algorithm configuration and state.
        
        Args:
            algorithm: Algorithm instance
            config: Configuration used to create the algorithm
            
        Returns:
            Dictionary with algorithm information
        """
        info = {
            'algorithm_class': config.get('algorithm_class', 'Unknown'),
            'algorithm_name': config.get('algorithm_name', 'Unknown'),
            'config': config.copy()
        }
        
        # Add algorithm-specific information
        try:
            if hasattr(algorithm, 'n_clusters'):
                info['n_clusters'] = algorithm.n_clusters
            if hasattr(algorithm, 'n_components'):
                info['n_components'] = algorithm.n_components
            if hasattr(algorithm, 'labels_'):
                unique_labels = np.unique(algorithm.labels_)
                info['actual_n_clusters'] = len(unique_labels)
                info['n_noise_points'] = np.sum(algorithm.labels_ == -1)
            if hasattr(algorithm, 'inertia_'):
                info['inertia'] = algorithm.inertia_
            if hasattr(algorithm, 'n_iter_'):
                info['n_iterations'] = algorithm.n_iter_
        except Exception as e:
            logger.warning(f"Could not extract algorithm info: {e}")
        
        return info
    
    def run_complete_benchmark(self, X: np.ndarray, 
                              config: Dict[str, Any],
                              normalize: bool = True,
                              true_labels: Optional[np.ndarray] = None) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
        """
        Run a complete sklearn algorithm benchmark for given configuration.
        
        Args:
            X: Training data
            config: Algorithm configuration
            normalize: Whether to normalize data
            true_labels: Ground truth labels (if available)
            
        Returns:
            Tuple of (trained_model, predictions, results_dict)
        """
        results = {
            'algorithm_name': config['algorithm_name'],
            'config': config.copy(),
            'dataset_info': {
                'n_samples': X.shape[0],
                'n_features': X.shape[1]
            }
        }
        
        try:
            # Create algorithm instance
            algorithm = self.create_algorithm_instance(config)
            
            # Fit algorithm
            fitted_algorithm = self.fit_algorithm(algorithm, X, normalize=normalize)
            
            # Get predictions
            predictions = self.predict_algorithm(fitted_algorithm, X)
            
            # Get algorithm info
            algo_info = self.get_algorithm_info(fitted_algorithm, config)
            results['algorithm_info'] = algo_info
            
            # Get cluster centers if available
            centers = self.get_cluster_centers(fitted_algorithm)
            if centers is not None:
                results['cluster_centers'] = centers
            
            results['success'] = True
            results['error'] = None
            
            return fitted_algorithm, predictions, results
            
        except Exception as e:
            logger.error(f"Error in sklearn benchmark: {e}")
            results['success'] = False
            results['error'] = str(e)
            
            # Return dummy results
            dummy_predictions = np.zeros(X.shape[0], dtype=int)
            return None, dummy_predictions, results
    
    # Algorithm creation methods
    def _create_kmeans(self, **kwargs) -> KMeans:
        """Create KMeans instance."""
        default_params = {
            'n_clusters': 3,
            'init': 'k-means++',
            'n_init': 10,
            'random_state': 42
        }
        default_params.update(kwargs)
        return KMeans(**default_params)
    
    def _create_dbscan(self, **kwargs) -> DBSCAN:
        """Create DBSCAN instance."""
        default_params = {
            'eps': 0.5,
            'min_samples': 5
        }
        default_params.update(kwargs)
        return DBSCAN(**default_params)
    
    def _create_agglomerative(self, **kwargs) -> AgglomerativeClustering:
        """Create AgglomerativeClustering instance."""
        default_params = {
            'n_clusters': 3,
            'linkage': 'ward'
        }
        default_params.update(kwargs)
        return AgglomerativeClustering(**default_params)
    
    def _create_spectral(self, **kwargs) -> SpectralClustering:
        """Create SpectralClustering instance."""
        default_params = {
            'n_clusters': 3,
            'random_state': 42,
            'n_init': 10
        }
        default_params.update(kwargs)
        return SpectralClustering(**default_params)
    
    def _create_meanshift(self, **kwargs) -> MeanShift:
        """Create MeanShift instance."""
        default_params = {}
        default_params.update(kwargs)
        return MeanShift(**default_params)
    
    def _create_birch(self, **kwargs) -> Birch:
        """Create Birch instance."""
        default_params = {
            'n_clusters': 3,
            'threshold': 0.5
        }
        default_params.update(kwargs)
        return Birch(**default_params)
    
    def _create_optics(self, **kwargs) -> OPTICS:
        """Create OPTICS instance."""
        default_params = {
            'min_samples': 5,
            'xi': 0.05,
            'min_cluster_size': 0.05
        }
        default_params.update(kwargs)
        return OPTICS(**default_params)
    
    def _create_affinity_propagation(self, **kwargs) -> AffinityPropagation:
        """Create AffinityPropagation instance."""
        default_params = {
            'random_state': 42
        }
        default_params.update(kwargs)
        return AffinityPropagation(**default_params)
    
    def _create_gaussian_mixture(self, **kwargs) -> GaussianMixture:
        """Create GaussianMixture instance."""
        default_params = {
            'n_components': 3,
            'random_state': 42
        }
        default_params.update(kwargs)
        return GaussianMixture(**default_params)