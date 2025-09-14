"""
SOM algorithm wrapper for performance benchmarking.
Provides a unified interface for different SOM initialization methods
while excluding specified methods (KDE, Kmeans, Kmeans++).
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

# Add SOM_plus_clustering to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
som_path = os.path.join(project_root, 'SOM_plus_clustering')
if som_path not in sys.path:
    sys.path.append(som_path)

try:
    from modules.som import SOM
    from modules.variables import INITIATION_METHOD_LIST, DISTANCE_METHOD_LIST
except ImportError as e:
    logging.error(f"Failed to import SOM modules: {e}")
    raise ImportError("Cannot import SOM modules. Check SOM_plus_clustering installation.")

logger = logging.getLogger(__name__)


class SOMWrapper:
    """
    Wrapper class for SOM algorithms with different initialization methods.
    Excludes KDE, Kmeans, and Kmeans++ initialization methods as requested.
    """
    
    # Excluded initialization methods as per user request
    EXCLUDED_METHODS = {'kde', 'kmeans', 'kmeans++'}
    
    def __init__(self):
        """Initialize the SOM wrapper."""
        # Get available initialization methods (excluding specified ones)
        self.available_init_methods = [
            method for method in INITIATION_METHOD_LIST 
            if method.lower() not in self.EXCLUDED_METHODS
        ]
        
        self.available_distance_methods = DISTANCE_METHOD_LIST.copy()
        
        logger.info(f"SOM wrapper initialized with {len(self.available_init_methods)} "
                   f"initialization methods: {self.available_init_methods}")
        logger.info(f"Available distance methods: {self.available_distance_methods}")
    
    def get_available_configurations(self) -> List[Dict[str, Any]]:
        """
        Get all available SOM configurations for benchmarking.
        
        Returns:
            List of configuration dictionaries
        """
        configurations = []
        
        for init_method in self.available_init_methods:
            for distance_method in self.available_distance_methods:
                config = {
                    'algorithm_type': 'SOM',
                    'init_method': init_method,
                    'distance_method': distance_method,
                    'algorithm_name': f'SOM_{init_method}_{distance_method}'
                }
                configurations.append(config)
        
        return configurations
    
    def create_som_instance(self, n_samples: int, n_features: int, 
                           init_method: str, distance_method: str = 'euclidean',
                           m: Optional[int] = None, n: Optional[int] = None,
                           learning_rate: float = 0.5, neighbour_rad: int = 1,
                           max_iter: Optional[int] = None) -> SOM:
        """
        Create a SOM instance with specified parameters.
        
        Args:
            n_samples: Number of samples in dataset
            n_features: Number of features in dataset
            init_method: Initialization method
            distance_method: Distance function to use
            m: Grid height (auto-calculated if None)
            n: Grid width (auto-calculated if None)
            learning_rate: Initial learning rate
            neighbour_rad: Initial neighborhood radius
            max_iter: Maximum iterations (auto-calculated if None)
            
        Returns:
            Configured SOM instance
        """
        # Validate initialization method
        if init_method.lower() in self.EXCLUDED_METHODS:
            raise ValueError(f"Initialization method '{init_method}' is excluded from benchmarking")
        
        if init_method not in self.available_init_methods:
            raise ValueError(f"Invalid initialization method: {init_method}. "
                           f"Available: {self.available_init_methods}")
        
        if distance_method not in self.available_distance_methods:
            raise ValueError(f"Invalid distance method: {distance_method}. "
                           f"Available: {self.available_distance_methods}")
        
        # Auto-calculate grid dimensions if not provided
        if m is None or n is None:
            # Rule of thumb: grid size should be roughly sqrt(n_samples)
            grid_size = max(2, int(np.sqrt(n_samples / 5)))
            m = m or grid_size
            n = n or grid_size
        
        # Auto-calculate max iterations if not provided
        if max_iter is None:
            # Rule of thumb: 500 * grid_size
            max_iter = 500 * (m * n)
        
        # Adjust learning rate and neighbor radius based on grid size
        adjusted_learning_rate = min(learning_rate, 0.8)
        adjusted_neighbour_rad = max(neighbour_rad, max(m, n) // 4)
        
        logger.info(f"Creating SOM: {m}x{n} grid, {init_method} init, "
                   f"{distance_method} distance, lr={adjusted_learning_rate}, "
                   f"radius={adjusted_neighbour_rad}, max_iter={max_iter}")
        
        try:
            som = SOM(
                m=m,
                n=n, 
                dim=n_features,
                initiate_method=init_method,
                learning_rate=adjusted_learning_rate,
                neighbour_rad=adjusted_neighbour_rad,
                distance_function=distance_method,
                max_iter=max_iter
            )
            return som
            
        except Exception as e:
            logger.error(f"Failed to create SOM instance: {e}")
            raise
    
    def fit_som(self, som: SOM, X: np.ndarray, epochs: int = 100,
               shuffle: bool = True, batch_size: Optional[int] = None) -> SOM:
        """
        Fit SOM to the data.
        
        Args:
            som: SOM instance to train
            X: Training data
            epochs: Number of training epochs
            shuffle: Whether to shuffle data each epoch
            batch_size: Batch size for training
            
        Returns:
            Trained SOM instance
        """
        try:
            # Determine appropriate number of epochs based on dataset size
            if epochs is None:
                epochs = max(50, min(200, 1000 // (X.shape[0] // 1000 + 1)))
            
            # Determine batch size if not provided
            if batch_size is None:
                batch_size = min(X.shape[0], max(32, X.shape[0] // 50))
            
            logger.info(f"Training SOM with {epochs} epochs, batch_size={batch_size}")
            
            som.fit(X, epoch=epochs, shuffle=shuffle, batch_size=batch_size)
            
            return som
            
        except Exception as e:
            logger.error(f"Error training SOM: {e}")
            raise
    
    def predict_som(self, som: SOM, X: np.ndarray) -> np.ndarray:
        """
        Get cluster predictions from trained SOM.
        
        Args:
            som: Trained SOM instance
            X: Data to predict
            
        Returns:
            Cluster labels
        """
        try:
            labels = som.predict(X)
            return labels
        except Exception as e:
            logger.error(f"Error predicting with SOM: {e}")
            raise
    
    def get_cluster_centers(self, som: SOM) -> np.ndarray:
        """
        Get cluster centers from trained SOM.
        
        Args:
            som: Trained SOM instance
            
        Returns:
            Cluster centers array
        """
        try:
            return som.cluster_center_
        except Exception as e:
            logger.error(f"Error getting cluster centers: {e}")
            raise
    
    def evaluate_som(self, som: SOM, X: np.ndarray, 
                    eval_methods: List[str] = None) -> Dict[str, float]:
        """
        Evaluate SOM clustering performance.
        
        Args:
            som: Trained SOM instance
            X: Data used for evaluation
            eval_methods: List of evaluation methods
            
        Returns:
            Dictionary of evaluation scores
        """
        if eval_methods is None:
            eval_methods = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
        
        try:
            scores = som.evaluate(X, eval_methods)
            
            # Convert to standard format if needed
            if isinstance(scores, list):
                score_dict = {}
                for i, method in enumerate(eval_methods):
                    if i < len(scores):
                        score_dict[method] = scores[i]
                return score_dict
            else:
                return scores
                
        except Exception as e:
            logger.error(f"Error evaluating SOM: {e}")
            # Return default scores
            return {method: 0.0 for method in eval_methods}
    
    def get_som_info(self, som: SOM) -> Dict[str, Any]:
        """
        Get information about the SOM configuration.
        
        Args:
            som: SOM instance
            
        Returns:
            Dictionary with SOM information
        """
        return {
            'grid_shape': (som.m, som.n),
            'n_neurons': som.m * som.n,
            'dimensions': som.dim,
            'init_method': som.init_method,
            'distance_function': som.dist_func,
            'initial_learning_rate': som.initial_learning_rate,
            'initial_neighbour_rad': som.initial_neighbour_rad,
            'current_learning_rate': som.cur_learning_rate,
            'current_neighbour_rad': som.cur_neighbour_rad,
            'max_iter': som.max_iter,
            'is_trained': som._trained
        }
    
    def run_complete_benchmark(self, X: np.ndarray, 
                              config: Dict[str, Any],
                              epochs: int = 100,
                              true_labels: Optional[np.ndarray] = None) -> Tuple[SOM, np.ndarray, Dict[str, Any]]:
        """
        Run a complete SOM benchmark for given configuration.
        
        Args:
            X: Training data
            config: Algorithm configuration
            epochs: Number of training epochs
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
            # Create SOM instance
            som = self.create_som_instance(
                n_samples=X.shape[0],
                n_features=X.shape[1],
                init_method=config['init_method'],
                distance_method=config['distance_method']
            )
            
            # Train SOM
            trained_som = self.fit_som(som, X, epochs=epochs)
            
            # Get predictions
            predictions = self.predict_som(trained_som, X)
            
            # Get SOM info
            som_info = self.get_som_info(trained_som)
            results['som_info'] = som_info
            
            # Evaluate SOM
            evaluation_scores = self.evaluate_som(trained_som, X)
            results['som_evaluation'] = evaluation_scores
            
            results['success'] = True
            results['error'] = None
            
            return trained_som, predictions, results
            
        except Exception as e:
            logger.error(f"Error in SOM benchmark: {e}")
            results['success'] = False
            results['error'] = str(e)
            
            # Return dummy results
            dummy_predictions = np.zeros(X.shape[0], dtype=int)
            return None, dummy_predictions, results
    
    def get_optimal_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Get optimal parameters for dataset.
        
        Args:
            X: Input data
            
        Returns:
            Dictionary of recommended parameters
        """
        n_samples, n_features = X.shape
        
        # Grid size based on data size
        grid_size = max(2, int(np.sqrt(n_samples / 5)))
        
        # Learning rate based on data size
        learning_rate = max(0.1, min(0.8, 1.0 / np.log(n_samples + 1)))
        
        # Neighborhood radius based on grid size
        neighbour_rad = max(1, grid_size // 3)
        
        # Epochs based on data complexity
        epochs = max(50, min(200, int(100 * np.log(n_samples + 1))))
        
        # Max iterations
        max_iter = epochs * n_samples
        
        return {
            'grid_m': grid_size,
            'grid_n': grid_size,
            'learning_rate': learning_rate,
            'neighbour_rad': neighbour_rad,
            'epochs': epochs,
            'max_iter': max_iter
        }