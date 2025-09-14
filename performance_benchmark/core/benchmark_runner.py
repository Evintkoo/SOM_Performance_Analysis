"""
Main benchmark runner for comprehensive performance comparison.
Orchestrates the entire benchmarking process across all algorithms and datasets.
"""

import os
import sys
import time
import logging
import signal
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import pickle
from concurrent.futures import TimeoutError
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from .experiment_config import ExperimentConfig
from ..utils.dataset_loader import DatasetLoader
from ..utils.logging_config import setup_logging, LoggingConfig, LoggingTemplates
from ..algorithms.som_wrapper import SOMWrapper
from ..algorithms.sklearn_wrapper import SklearnClusteringWrapper
from ..metrics.performance_metrics import PerformanceMetrics


class BenchmarkRunner:
    """
    Main class for running comprehensive clustering algorithm benchmarks.
    
    Coordinates dataset loading, algorithm execution, performance measurement,
    and result collection across SOM and sklearn clustering algorithms.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the benchmark runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        
        # Setup enhanced logging system
        self._setup_logging()
        self.logger = self.logging_system.get_logger("benchmark_runner")
        
        # Initialize components
        self.dataset_loader = DatasetLoader(config.datasets_root)
        self.som_wrapper = SOMWrapper() if config.include_som_algorithms else None
        self.sklearn_wrapper = SklearnClusteringWrapper() if config.include_sklearn_algorithms else None
        
        # Results storage
        self.results = {
            'experiment_info': {},
            'dataset_info': {},
            'algorithm_results': {},
            'individual_runs': {},  # Store individual run results
            'aggregated_results': {},  # Store aggregated results across runs
            'statistical_tests': {},  # Store statistical test results
            'summary_stats': {},
            'errors': []
        }
        
        # Setup output paths
        self.output_paths = config.get_output_paths()
        
        self.logger.info("BenchmarkRunner initialized successfully")
        self.logger.info(f"Experiment: {config.experiment_name}")
        
        # Log experiment start
        self.logging_system.log_experiment_start(
            config.experiment_name, 
            config.to_dict()
        )
        
    def _setup_logging(self):
        """Setup enhanced logging system."""
        # Create logging configuration based on experiment config
        if self.config.log_style == "json":
            logging_config = LoggingTemplates.production()
        elif self.config.log_level == "DEBUG":
            logging_config = LoggingTemplates.development()
        elif self.config.log_performance:
            logging_config = LoggingTemplates.performance_focused()
        else:
            logging_config = LoggingConfig()
        
        # Override with experiment-specific settings
        logging_config.level = self.config.log_level
        logging_config.format_style = self.config.log_style
        logging_config.file_enabled = self.config.log_to_file
        logging_config.performance_enabled = self.config.log_performance
        logging_config.log_dir = str(self.config.get_output_paths()['logs'])
        
        # Setup the logging system
        self.logging_system = setup_logging(logging_config)
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Run the complete benchmark across all datasets and algorithms.
        
        Returns:
            Dictionary containing all benchmark results
        """
        start_time = time.time()
        self.logger.info("Starting full benchmark")
        
        try:
            # Save configuration
            self._save_experiment_config()
            
            # Load datasets
            datasets = self._load_datasets()
            
            # Get algorithm configurations
            algorithm_configs = self._get_algorithm_configurations(datasets)
            
            # Run benchmarks
            self._run_algorithm_benchmarks(datasets, algorithm_configs)
            
            # Process and summarize results
            self._process_results()
            
            # Save results
            self._save_results()
            
            total_time = time.time() - start_time
            self.logger.info(f"Full benchmark completed in {total_time:.2f} seconds")
            
            self.results['experiment_info']['total_runtime'] = total_time
            self.results['experiment_info']['completion_status'] = 'completed'
            
            # Log experiment completion
            self.logging_system.log_experiment_end(
                self.config.experiment_name, 
                total_time, 
                success=True
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"Benchmark failed: {e}")
            self.logger.error(traceback.format_exc())
            
            self.results['experiment_info']['completion_status'] = 'failed'
            self.results['experiment_info']['error'] = str(e)
            self.results['experiment_info']['total_runtime'] = total_time
            
            # Log experiment failure
            self.logging_system.log_experiment_end(
                self.config.experiment_name, 
                total_time, 
                success=False
            )
            
            # Still save partial results
            self._save_results()
            
            raise
        
        finally:
            # Close logging system
            self.logging_system.close()
        
        return self.results
    
    def _load_datasets(self) -> List[Tuple[np.ndarray, Optional[np.ndarray], Dict]]:
        """Load all datasets for benchmarking."""
        self.logger.info("Loading datasets...")
        
        if "all" in self.config.dataset_categories:
            datasets = self.dataset_loader.load_all_datasets(
                normalize=self.config.normalize_features,
                max_samples=self.config.max_samples_per_dataset,
                has_labels=getattr(self.config, 'has_labels', False),
                label_column=getattr(self.config, 'label_column', -1)
            )
        else:
            datasets = []
            for category in self.config.dataset_categories:
                category_datasets = self.dataset_loader.load_datasets_by_category(
                    category=category,
                    normalize=self.config.normalize_features,
                    has_labels=getattr(self.config, 'has_labels', False),
                    label_column=getattr(self.config, 'label_column', -1)
                )
                datasets.extend(category_datasets)
        
        # Apply max samples limit if needed
        if self.config.max_samples_per_dataset:
            for i, (X, labels, metadata) in enumerate(datasets):
                if X.shape[0] > self.config.max_samples_per_dataset:
                    indices = np.random.RandomState(self.config.random_state).choice(
                        X.shape[0], self.config.max_samples_per_dataset, replace=False
                    )
                    X = X[indices]
                    if labels is not None:
                        labels = labels[indices]
                    metadata['n_samples'] = self.config.max_samples_per_dataset
                    metadata['subsampled'] = True
                    datasets[i] = (X, labels, metadata)
        
        self.logger.info(f"Loaded {len(datasets)} datasets")
        
        # Store dataset info
        self.results['dataset_info'] = {
            metadata['filename']: {
                'category': metadata['category'],
                'n_samples': metadata['n_samples'],
                'n_features': metadata['n_features'],
                'has_labels': metadata['has_labels'],
                'n_classes': metadata.get('n_classes'),
                'subsampled': metadata.get('subsampled', False)
            }
            for _, _, metadata in datasets
        }
        
        return datasets
    
    def _get_algorithm_configurations(self, datasets: List[Tuple]) -> List[Dict[str, Any]]:
        """Get all algorithm configurations for benchmarking."""
        configurations = []
        
        # Estimate typical dataset characteristics for parameter optimization
        if datasets:
            avg_samples = np.mean([X.shape[0] for X, _, _ in datasets])
            avg_features = np.mean([X.shape[1] for X, _, _ in datasets])
        else:
            avg_samples, avg_features = 1000, 10
        
        # SOM configurations
        if self.som_wrapper:
            som_configs = self.som_wrapper.get_available_configurations()
            configurations.extend(som_configs)
            self.logger.info(f"Added {len(som_configs)} SOM configurations")
        
        # Sklearn configurations
        if self.sklearn_wrapper:
            sklearn_configs = self.sklearn_wrapper.get_available_configurations(
                n_samples=int(avg_samples),
                n_features=int(avg_features)
            )
            configurations.extend(sklearn_configs)
            self.logger.info(f"Added {len(sklearn_configs)} sklearn configurations")
        
        self.logger.info(f"Total algorithm configurations: {len(configurations)}")
        return configurations
    
    def _run_algorithm_benchmarks(self, datasets: List[Tuple], 
                                 algorithm_configs: List[Dict[str, Any]]) -> None:
        """Run benchmarks for all algorithm-dataset combinations."""
        total_combinations = len(datasets) * len(algorithm_configs) * self.config.n_runs
        current_combination = 0
        
        self.logger.info(f"Running {total_combinations} algorithm-dataset combinations "
                        f"({len(datasets)} datasets × {len(algorithm_configs)} algorithms × {self.config.n_runs} runs)")
        
        # Track results for aggregation
        individual_results = {}  # For storing individual runs
        
        for dataset_idx, (X, true_labels, dataset_metadata) in enumerate(datasets):
            dataset_name = dataset_metadata['filename']
            self.logger.info(f"Processing dataset: {dataset_name} "
                           f"({X.shape[0]} samples, {X.shape[1]} features)")
            
            for config in algorithm_configs:
                algorithm_name = config['algorithm_name']
                
                # Initialize storage for this algorithm-dataset combination
                combo_key = f"{dataset_name}_{algorithm_name}"
                individual_results[combo_key] = []
                
                for run_num in range(self.config.n_runs):
                    current_combination += 1
                    
                    # Log progress
                    if self.config.log_progress:
                        self.logging_system.log_run_progress(
                            current=current_combination,
                            total=total_combinations,
                            algorithm=algorithm_name,
                            dataset=dataset_name,
                            run_number=run_num + 1
                        )
                    
                    self.logger.info(f"[{current_combination}/{total_combinations}] "
                                   f"Running {algorithm_name} on {dataset_name} (run {run_num + 1})")
                    
                    try:
                        # Run single algorithm benchmark
                        result = self._run_single_benchmark(
                            X=X,
                            true_labels=true_labels,
                            dataset_metadata=dataset_metadata,
                            algorithm_config=config,
                            run_number=run_num
                        )
                        
                        # Store individual result
                        result_key = f"{dataset_name}_{algorithm_name}_run{run_num}"
                        self.results['algorithm_results'][result_key] = result
                        individual_results[combo_key].append(result)
                        
                        # Log performance metrics
                        if self.config.log_performance and result['success']:
                            self._log_performance_metrics(result, algorithm_name, dataset_name, run_num)
                        
                    except Exception as e:
                        error_msg = f"Failed: {algorithm_name} on {dataset_name} (run {run_num}): {str(e)}"
                        self.logger.error(error_msg)
                        self.results['errors'].append({
                            'dataset': dataset_name,
                            'algorithm': algorithm_name,
                            'run': run_num,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        })
                        continue
        
        # Store individual results and create aggregations
        self.results['individual_runs'] = individual_results
        
        if self.config.aggregate_runs and self.config.n_runs > 1:
            self._aggregate_results()
        
        if self.config.statistical_significance and self.config.n_runs > 1:
            self._compute_statistical_tests()
    
    def _run_single_benchmark(self, X: np.ndarray, true_labels: Optional[np.ndarray],
                             dataset_metadata: Dict, algorithm_config: Dict[str, Any],
                             run_number: int) -> Dict[str, Any]:
        """Run a single algorithm benchmark on a dataset."""
        
        # Initialize performance metrics
        metrics = PerformanceMetrics()
        
        # Create run configuration
        run_config = self.config.create_run_config(
            algorithm_name=algorithm_config['algorithm_name'],
            dataset_name=dataset_metadata['filename'],
            run_number=run_number
        )
        
        result = {
            'run_config': run_config,
            'dataset_metadata': dataset_metadata.copy(),
            'algorithm_config': algorithm_config.copy(),
            'timestamp': datetime.now().isoformat(),
            'success': False
        }
        
        try:
            # Set random seed for reproducibility
            np.random.seed(run_config['random_state'])
            
            # Determine algorithm type and run appropriate benchmark
            if algorithm_config.get('algorithm_type') == 'SOM':
                trained_model, predictions, algorithm_results = self._run_som_benchmark(
                    X=X, 
                    config=algorithm_config,
                    metrics=metrics,
                    epochs=self.config.som_epochs
                )
            else:  # sklearn algorithm
                trained_model, predictions, algorithm_results = self._run_sklearn_benchmark(
                    X=X,
                    config=algorithm_config,
                    metrics=metrics
                )
            
            # Measure clustering quality
            if self.config.measure_quality:
                quality_scores = metrics.measure_clustering_quality(
                    X=X,
                    labels=predictions,
                    algorithm_name=algorithm_config['algorithm_name']
                )
                result['quality_metrics'] = quality_scores
            
            # Measure evaluation scores (if true labels available)
            if self.config.measure_evaluation and true_labels is not None:
                eval_scores = metrics.measure_evaluation_scores(
                    true_labels=true_labels,
                    predicted_labels=predictions,
                    algorithm_name=algorithm_config['algorithm_name']
                )
                result['evaluation_metrics'] = eval_scores
            
            # Measure memory efficiency
            if self.config.measure_memory and trained_model is not None:
                memory_scores = metrics.measure_memory_efficiency(
                    X=X,
                    model=trained_model,
                    algorithm_name=algorithm_config['algorithm_name']
                )
                result['memory_metrics'] = memory_scores
            
            # Store all metrics
            result['performance_metrics'] = metrics.get_summary()
            result['algorithm_results'] = algorithm_results
            result['predictions'] = predictions if self.config.save_predictions else None
            result['model'] = trained_model if self.config.save_models else None
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
            self.logger.error(f"Single benchmark failed: {e}")
            raise
        
        return result
    
    def _run_som_benchmark(self, X: np.ndarray, config: Dict[str, Any],
                          metrics: PerformanceMetrics, epochs: int) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
        """Run SOM algorithm benchmark with performance measurement."""
        
        # Start timing
        metrics.start_timing()
        
        # Run SOM benchmark
        trained_model, predictions, results = self.som_wrapper.run_complete_benchmark(
            X=X,
            config=config,
            epochs=epochs
        )
        
        # Stop timing
        fit_time = metrics.stop_timing('fit_predict')
        
        results['fit_time'] = fit_time
        
        return trained_model, predictions, results
    
    def _run_sklearn_benchmark(self, X: np.ndarray, config: Dict[str, Any],
                              metrics: PerformanceMetrics) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
        """Run sklearn algorithm benchmark with performance measurement."""
        
        # Start timing
        metrics.start_timing()
        
        # Run sklearn benchmark
        trained_model, predictions, results = self.sklearn_wrapper.run_complete_benchmark(
            X=X,
            config=config,
            normalize=self.config.normalize_features
        )
        
        # Stop timing
        fit_time = metrics.stop_timing('fit_predict')
        
        results['fit_time'] = fit_time
        
        return trained_model, predictions, results
    
    def _process_results(self) -> None:
        """Process and summarize benchmark results."""
        self.logger.info("Processing results...")
        
        # Create summary statistics
        summary_stats = {
            'total_runs': len(self.results['algorithm_results']),
            'successful_runs': sum(1 for r in self.results['algorithm_results'].values() if r['success']),
            'failed_runs': len(self.results['errors']),
            'algorithms_tested': len(set(r['algorithm_config']['algorithm_name'] 
                                       for r in self.results['algorithm_results'].values())),
            'datasets_tested': len(self.results['dataset_info'])
        }
        
        # Calculate average metrics per algorithm
        algorithm_summaries = {}
        for result_key, result in self.results['algorithm_results'].items():
            if not result['success']:
                continue
                
            algo_name = result['algorithm_config']['algorithm_name']
            
            if algo_name not in algorithm_summaries:
                algorithm_summaries[algo_name] = {
                    'runs': 0,
                    'total_fit_time': 0,
                    'quality_scores': [],
                    'evaluation_scores': [],
                    'memory_usage': []
                }
            
            summary = algorithm_summaries[algo_name]
            summary['runs'] += 1
            
            # Aggregate performance metrics
            if 'performance_metrics' in result:
                perf_metrics = result['performance_metrics']
                
                if 'speed_metrics' in perf_metrics:
                    summary['total_fit_time'] += perf_metrics['speed_metrics'].get('fit_predict', 0)
                
                if 'quality_metrics' in perf_metrics:
                    summary['quality_scores'].append(perf_metrics['quality_metrics'])
                
                if 'evaluation_metrics' in perf_metrics:
                    summary['evaluation_scores'].append(perf_metrics['evaluation_metrics'])
                
                if 'memory_metrics' in perf_metrics:
                    memory_used = perf_metrics['memory_metrics'].get('fit_predict_used_mb', 0)
                    summary['memory_usage'].append(memory_used)
        
        # Calculate averages
        for algo_name, summary in algorithm_summaries.items():
            if summary['runs'] > 0:
                summary['avg_fit_time'] = summary['total_fit_time'] / summary['runs']
                summary['avg_memory_usage'] = np.mean(summary['memory_usage']) if summary['memory_usage'] else 0
        
        summary_stats['algorithm_summaries'] = algorithm_summaries
        self.results['summary_stats'] = summary_stats
        
        self.logger.info(f"Processed {summary_stats['successful_runs']} successful runs")
    
    def _save_experiment_config(self) -> None:
        """Save experiment configuration."""
        config_file = self.output_paths['configs'] / f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.config.save_to_file(config_file)
    
    def _save_results(self) -> None:
        """Save all benchmark results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw results
        raw_results_file = self.output_paths['raw_results'] / f"raw_results_{timestamp}.pkl"
        with open(raw_results_file, 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save summary as JSON
        summary_file = self.output_paths['processed_results'] / f"summary_{timestamp}.json"
        
        # Prepare JSON-serializable summary
        json_summary = {
            'experiment_info': self.results['experiment_info'],
            'dataset_info': self.results['dataset_info'],
            'summary_stats': self.results['summary_stats'],
            'errors': self.results['errors']
        }
        
        # Add aggregated results if available
        if 'aggregated_results' in self.results:
            json_summary['aggregated_results'] = self.results['aggregated_results']
        
        # Add statistical tests if available
        if 'statistical_tests' in self.results:
            json_summary['statistical_tests'] = self.results['statistical_tests']
        
        with open(summary_file, 'w') as f:
            json.dump(json_summary, f, indent=2, default=str)
        
        # Save detailed results as CSV for easy analysis
        self._save_results_as_csv(timestamp)
        
        # Save aggregated results as CSV if available
        if 'aggregated_results' in self.results and self.results['aggregated_results']:
            self._save_aggregated_results_as_csv(timestamp)
        
        self.logger.info(f"Results saved to {self.output_paths['base_dir']}")
    
    def _save_results_as_csv(self, timestamp: str) -> None:
        """Save results in CSV format for easy analysis."""
        
        # Prepare data for CSV
        csv_data = []
        
        for result_key, result in self.results['algorithm_results'].items():
            if not result['success']:
                continue
            
            row = {
                'dataset_name': result['dataset_metadata']['filename'],
                'dataset_category': result['dataset_metadata']['category'],
                'n_samples': result['dataset_metadata']['n_samples'],
                'n_features': result['dataset_metadata']['n_features'],
                'algorithm_name': result['algorithm_config']['algorithm_name'],
                'algorithm_type': result['algorithm_config'].get('algorithm_type', 'sklearn'),
                'run_number': result['run_config']['run_number'],
                'success': result['success']
            }
            
            # Add performance metrics
            if 'performance_metrics' in result:
                perf_metrics = result['performance_metrics']
                
                # Speed metrics
                if 'speed_metrics' in perf_metrics:
                    for metric, value in perf_metrics['speed_metrics'].items():
                        row[f'speed_{metric}'] = value
                
                # Memory metrics
                if 'memory_metrics' in perf_metrics:
                    for metric, value in perf_metrics['memory_metrics'].items():
                        row[f'memory_{metric}'] = value
                
                # Quality metrics
                if 'quality_metrics' in perf_metrics:
                    for metric, value in perf_metrics['quality_metrics'].items():
                        row[f'quality_{metric}'] = value
                
                # Evaluation metrics
                if 'evaluation_metrics' in perf_metrics:
                    for metric, value in perf_metrics['evaluation_metrics'].items():
                        row[f'evaluation_{metric}'] = value
            
            csv_data.append(row)
        
        # Create DataFrame and save
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_file = self.output_paths['processed_results'] / f"results_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            self.logger.info(f"CSV results saved to {csv_file}")
    
    def _save_aggregated_results_as_csv(self, timestamp: str) -> None:
        """Save aggregated results in CSV format for easy analysis."""
        
        # Prepare data for CSV
        csv_data = []
        
        for combo_key, result in self.results['aggregated_results'].items():
            base_row = {
                'dataset_name': result['dataset_metadata']['filename'],
                'dataset_category': result['dataset_metadata']['category'],
                'n_samples': result['dataset_metadata']['n_samples'],
                'n_features': result['dataset_metadata']['n_features'],
                'algorithm_name': result['algorithm_config']['algorithm_name'],
                'algorithm_type': result['algorithm_config'].get('algorithm_type', 'sklearn'),
                'n_successful_runs': result['n_successful_runs'],
                'n_total_runs': result['n_total_runs'],
                'success_rate': result['success_rate']
            }
            
            # Add aggregated metrics
            if 'aggregated_metrics' in result:
                for category, metrics in result['aggregated_metrics'].items():
                    for metric_name, aggregations in metrics.items():
                        for agg_method, value in aggregations.items():
                            column_name = f"{category[:-8]}_{metric_name}_{agg_method}"  # Remove '_metrics' suffix
                            base_row[column_name] = value
            
            csv_data.append(base_row)
        
        # Create DataFrame and save
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_file = self.output_paths['processed_results'] / f"aggregated_results_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Aggregated CSV results saved to {csv_file}")
    
    def get_results_summary(self) -> str:
        """Get a human-readable summary of results."""
        if not self.results.get('summary_stats'):
            return "No results available yet."
        
        stats = self.results['summary_stats']
        
        summary_parts = [
            f"Benchmark Results Summary",
            f"=" * 50,
            f"Total runs: {stats['total_runs']}",
            f"Successful runs: {stats['successful_runs']}",
            f"Failed runs: {stats['failed_runs']}",
            f"Success rate: {stats['successful_runs'] / stats['total_runs'] * 100:.1f}%" if stats['total_runs'] > 0 else "N/A",
            f"",
            f"Algorithms tested: {stats['algorithms_tested']}",
            f"Datasets tested: {stats['datasets_tested']}",
            f"",
            f"Top performing algorithms (by average fit time):"
        ]
        
        # Sort algorithms by average fit time
        if 'algorithm_summaries' in stats:
            sorted_algos = sorted(
                stats['algorithm_summaries'].items(),
                key=lambda x: x[1].get('avg_fit_time', float('inf'))
            )
            
            for i, (algo_name, summary) in enumerate(sorted_algos[:5]):
                avg_time = summary.get('avg_fit_time', 0)
                runs = summary.get('runs', 0)
                summary_parts.append(f"  {i+1}. {algo_name}: {avg_time:.3f}s (avg over {runs} runs)")
        
        return "\n".join(summary_parts)
    
    def _log_performance_metrics(self, result: Dict[str, Any], algorithm: str, 
                                dataset: str, run_number: int):
        """Log performance metrics for a completed run."""
        if not result.get('performance_metrics'):
            return
        
        perf_metrics = result['performance_metrics']
        
        # Log speed metrics
        if 'speed_metrics' in perf_metrics:
            for metric_name, value in perf_metrics['speed_metrics'].items():
                self.logging_system.log_performance_metric(
                    metric_name=f"speed_{metric_name}",
                    value=value,
                    algorithm=algorithm,
                    dataset=dataset,
                    run_number=run_number
                )
        
        # Log memory metrics
        if 'memory_metrics' in perf_metrics:
            for metric_name, value in perf_metrics['memory_metrics'].items():
                self.logging_system.log_performance_metric(
                    metric_name=f"memory_{metric_name}",
                    value=value,
                    algorithm=algorithm,
                    dataset=dataset,
                    run_number=run_number
                )
        
        # Log quality metrics
        if 'quality_metrics' in perf_metrics:
            for metric_name, value in perf_metrics['quality_metrics'].items():
                self.logging_system.log_performance_metric(
                    metric_name=f"quality_{metric_name}",
                    value=value,
                    algorithm=algorithm,
                    dataset=dataset,
                    run_number=run_number
                )
        
        # Log evaluation metrics
        if 'evaluation_metrics' in perf_metrics:
            for metric_name, value in perf_metrics['evaluation_metrics'].items():
                self.logging_system.log_performance_metric(
                    metric_name=f"evaluation_{metric_name}",
                    value=value,
                    algorithm=algorithm,
                    dataset=dataset,
                    run_number=run_number
                )
    
    def _aggregate_results(self):
        """Aggregate results across multiple runs."""
        self.logger.info("Aggregating results across multiple runs...")
        
        aggregated = {}
        
        for combo_key, runs in self.results['individual_runs'].items():
            if not runs or not any(r['success'] for r in runs):
                continue
            
            successful_runs = [r for r in runs if r['success']]
            if len(successful_runs) < 2:
                continue
            
            aggregated[combo_key] = {
                'dataset_metadata': successful_runs[0]['dataset_metadata'],
                'algorithm_config': successful_runs[0]['algorithm_config'],
                'n_successful_runs': len(successful_runs),
                'n_total_runs': len(runs),
                'success_rate': len(successful_runs) / len(runs),
                'aggregated_metrics': {}
            }
            
            # Aggregate performance metrics
            self._aggregate_performance_metrics(successful_runs, aggregated[combo_key])
        
        self.results['aggregated_results'] = aggregated
        self.logger.info(f"Aggregated results for {len(aggregated)} algorithm-dataset combinations")
    
    def _aggregate_performance_metrics(self, runs: List[Dict], aggregated_result: Dict):
        """Aggregate performance metrics across runs."""
        metric_categories = ['speed_metrics', 'memory_metrics', 'quality_metrics', 'evaluation_metrics']
        
        for category in metric_categories:
            if category not in aggregated_result['aggregated_metrics']:
                aggregated_result['aggregated_metrics'][category] = {}
            
            # Collect all metric values across runs
            metric_values = {}
            for run in runs:
                if 'performance_metrics' in run and category in run['performance_metrics']:
                    for metric_name, value in run['performance_metrics'][category].items():
                        if metric_name not in metric_values:
                            metric_values[metric_name] = []
                        if value is not None and not np.isnan(value):
                            metric_values[metric_name].append(value)
            
            # Compute aggregation statistics
            for metric_name, values in metric_values.items():
                if len(values) == 0:
                    continue
                
                values_array = np.array(values)
                aggregated_metrics = {}
                
                for agg_method in self.config.aggregation_methods:
                    if agg_method == 'mean':
                        aggregated_metrics['mean'] = np.mean(values_array)
                    elif agg_method == 'std':
                        aggregated_metrics['std'] = np.std(values_array, ddof=1) if len(values) > 1 else 0
                    elif agg_method == 'min':
                        aggregated_metrics['min'] = np.min(values_array)
                    elif agg_method == 'max':
                        aggregated_metrics['max'] = np.max(values_array)
                    elif agg_method == 'median':
                        aggregated_metrics['median'] = np.median(values_array)
                    elif agg_method == 'q25':
                        aggregated_metrics['q25'] = np.percentile(values_array, 25)
                    elif agg_method == 'q75':
                        aggregated_metrics['q75'] = np.percentile(values_array, 75)
                
                # Add confidence interval for mean
                if len(values) > 1:
                    se = stats.sem(values_array)
                    ci = stats.t.interval(
                        self.config.confidence_level,
                        len(values) - 1,
                        loc=np.mean(values_array),
                        scale=se
                    )
                    aggregated_metrics['mean_ci_lower'] = ci[0]
                    aggregated_metrics['mean_ci_upper'] = ci[1]
                
                aggregated_result['aggregated_metrics'][category][metric_name] = aggregated_metrics
    
    def _compute_statistical_tests(self):
        """Compute statistical significance tests between algorithms."""
        self.logger.info("Computing statistical significance tests...")
        
        statistical_tests = {}
        
        # Group results by dataset
        datasets = {}
        for combo_key, runs in self.results['individual_runs'].items():
            parts = combo_key.split('_')
            if len(parts) < 2:
                continue
            dataset_name = parts[0]
            algorithm_name = '_'.join(parts[1:])
            
            if dataset_name not in datasets:
                datasets[dataset_name] = {}
            
            # Extract successful runs
            successful_runs = [r for r in runs if r['success']]
            if len(successful_runs) >= 2:  # Need at least 2 runs for statistical tests
                datasets[dataset_name][algorithm_name] = successful_runs
        
        # Perform pairwise statistical tests for each dataset
        for dataset_name, algorithms in datasets.items():
            if len(algorithms) < 2:
                continue
            
            statistical_tests[dataset_name] = {}
            algorithm_names = list(algorithms.keys())
            
            # Test each metric category
            metric_categories = ['speed_metrics', 'memory_metrics', 'quality_metrics', 'evaluation_metrics']
            
            for category in metric_categories:
                statistical_tests[dataset_name][category] = {}
                
                # Get all available metrics for this category
                all_metrics = set()
                for algo_runs in algorithms.values():
                    for run in algo_runs:
                        if 'performance_metrics' in run and category in run['performance_metrics']:
                            all_metrics.update(run['performance_metrics'][category].keys())
                
                # Test each metric
                for metric_name in all_metrics:
                    statistical_tests[dataset_name][category][metric_name] = {}
                    
                    # Collect values for each algorithm
                    algorithm_values = {}
                    for algo_name, runs in algorithms.items():
                        values = []
                        for run in runs:
                            if ('performance_metrics' in run and 
                                category in run['performance_metrics'] and
                                metric_name in run['performance_metrics'][category]):
                                value = run['performance_metrics'][category][metric_name]
                                if value is not None and not np.isnan(value):
                                    values.append(value)
                        if len(values) >= 2:
                            algorithm_values[algo_name] = values
                    
                    # Perform pairwise t-tests
                    if len(algorithm_values) >= 2:
                        for i, algo1 in enumerate(algorithm_names):
                            for j, algo2 in enumerate(algorithm_names[i+1:], i+1):
                                if algo1 in algorithm_values and algo2 in algorithm_values:
                                    values1 = algorithm_values[algo1]
                                    values2 = algorithm_values[algo2]
                                    
                                    # Perform Welch's t-test (unequal variances)
                                    try:
                                        t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
                                        
                                        test_key = f"{algo1}_vs_{algo2}"
                                        statistical_tests[dataset_name][category][metric_name][test_key] = {
                                            'algorithm_1': algo1,
                                            'algorithm_2': algo2,
                                            'mean_1': np.mean(values1),
                                            'mean_2': np.mean(values2),
                                            'std_1': np.std(values1, ddof=1),
                                            'std_2': np.std(values2, ddof=1),
                                            'n_1': len(values1),
                                            'n_2': len(values2),
                                            't_statistic': t_stat,
                                            'p_value': p_value,
                                            'significant': p_value < (1 - self.config.confidence_level),
                                            'effect_size': abs(np.mean(values1) - np.mean(values2)) / 
                                                         np.sqrt(((np.std(values1, ddof=1)**2) + (np.std(values2, ddof=1)**2)) / 2)
                                        }
                                    except Exception as e:
                                        self.logger.warning(f"Statistical test failed for {metric_name} "
                                                          f"between {algo1} and {algo2}: {e}")
        
        self.results['statistical_tests'] = statistical_tests
        self.logger.info(f"Computed statistical tests for {len(statistical_tests)} datasets")