"""
Results analysis and visualization tools for performance benchmarking.
Provides comprehensive analysis and visualization of benchmark results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import pickle
import json
from datetime import datetime

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """
    Comprehensive analysis and visualization of benchmark results.
    
    Provides methods for analyzing performance metrics, generating comparisons,
    and creating visualizations for benchmark results.
    """
    
    def __init__(self, results_dir: Union[str, Path]):
        """
        Initialize the results analyzer.
        
        Args:
            results_dir: Directory containing benchmark results
        """
        self.results_dir = Path(results_dir)
        self.results_data = None
        self.processed_df = None
        
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")
        
        logger.info(f"ResultsAnalyzer initialized with directory: {self.results_dir}")
    
    def load_results(self, results_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load benchmark results from file.
        
        Args:
            results_file: Specific results file to load (if None, loads most recent)
            
        Returns:
            Dictionary containing benchmark results
        """
        if results_file is None:
            # Find most recent results file
            raw_results_dir = self.results_dir / 'raw_results'
            if raw_results_dir.exists():
                pkl_files = list(raw_results_dir.glob('raw_results_*.pkl'))
                if pkl_files:
                    results_file = max(pkl_files, key=lambda x: x.stat().st_mtime)
                else:
                    raise FileNotFoundError("No results files found")
            else:
                raise FileNotFoundError("Raw results directory not found")
        else:
            results_file = Path(results_file)
        
        logger.info(f"Loading results from: {results_file}")
        
        with open(results_file, 'rb') as f:
            self.results_data = pickle.load(f)
        
        # Convert to DataFrame for easier analysis
        self._create_processed_dataframe()
        
        return self.results_data
    
    def _create_processed_dataframe(self) -> None:
        """Create a processed DataFrame from raw results."""
        if not self.results_data:
            raise ValueError("No results data loaded")
        
        rows = []
        
        for result_key, result in self.results_data['algorithm_results'].items():
            if not result['success']:
                continue
            
            row = {
                'result_key': result_key,
                'dataset_name': result['dataset_metadata']['filename'],
                'dataset_category': result['dataset_metadata']['category'],
                'n_samples': result['dataset_metadata']['n_samples'],
                'n_features': result['dataset_metadata']['n_features'],
                'has_labels': result['dataset_metadata']['has_labels'],
                'algorithm_name': result['algorithm_config']['algorithm_name'],
                'algorithm_type': result['algorithm_config'].get('algorithm_type', 'sklearn'),
                'run_number': result['run_config']['run_number']
            }
            
            # Extract performance metrics
            if 'performance_metrics' in result:
                perf = result['performance_metrics']
                
                # Speed metrics
                for metric, value in perf.get('speed_metrics', {}).items():
                    row[f'speed_{metric}'] = value
                
                # Memory metrics  
                for metric, value in perf.get('memory_metrics', {}).items():
                    row[f'memory_{metric}'] = value
                
                # Quality metrics
                for metric, value in perf.get('quality_metrics', {}).items():
                    row[f'quality_{metric}'] = value
                
                # Evaluation metrics
                for metric, value in perf.get('evaluation_metrics', {}).items():
                    row[f'evaluation_{metric}'] = value
            
            # Extract algorithm-specific metrics
            if 'quality_metrics' in result:
                for metric, value in result['quality_metrics'].items():
                    row[f'quality_{metric}'] = value
            
            if 'evaluation_metrics' in result:
                for metric, value in result['evaluation_metrics'].items():
                    row[f'evaluation_{metric}'] = value
            
            rows.append(row)
        
        self.processed_df = pd.DataFrame(rows)
        logger.info(f"Created processed DataFrame with {len(self.processed_df)} rows")
    
    def generate_performance_comparison(self, metric_category: str = 'speed') -> pd.DataFrame:
        """
        Generate performance comparison across algorithms.
        
        Args:
            metric_category: Category of metrics to compare ('speed', 'memory', 'quality', 'evaluation')
            
        Returns:
            DataFrame with performance comparison
        """
        if self.processed_df is None:
            raise ValueError("No processed data available. Load results first.")
        
        # Get relevant columns
        metric_cols = [col for col in self.processed_df.columns if col.startswith(f'{metric_category}_')]
        
        if not metric_cols:
            raise ValueError(f"No metrics found for category: {metric_category}")
        
        # Group by algorithm and calculate statistics
        groupby_cols = ['algorithm_name', 'algorithm_type']
        comparison_data = []
        
        for algo_info, group in self.processed_df.groupby(groupby_cols):
            algo_name, algo_type = algo_info
            
            row = {
                'algorithm_name': algo_name,
                'algorithm_type': algo_type,
                'n_runs': len(group)
            }
            
            for metric_col in metric_cols:
                values = group[metric_col].dropna()
                if len(values) > 0:
                    row[f'{metric_col}_mean'] = values.mean()
                    row[f'{metric_col}_std'] = values.std()
                    row[f'{metric_col}_min'] = values.min()
                    row[f'{metric_col}_max'] = values.max()
                    row[f'{metric_col}_median'] = values.median()
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def create_performance_plots(self, output_dir: Optional[Path] = None) -> List[Path]:
        """
        Create comprehensive performance visualization plots.
        
        Args:
            output_dir: Directory to save plots (defaults to results/visualizations)
            
        Returns:
            List of paths to created plot files
        """
        if output_dir is None:
            output_dir = self.results_dir / 'visualizations'
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_files = []
        
        # 1. Speed comparison
        try:
            fig_path = self._plot_speed_comparison(output_dir)
            if fig_path:
                plot_files.append(fig_path)
        except Exception as e:
            logger.error(f"Failed to create speed comparison plot: {e}")
        
        # 2. Memory usage comparison
        try:
            fig_path = self._plot_memory_comparison(output_dir)
            if fig_path:
                plot_files.append(fig_path)
        except Exception as e:
            logger.error(f"Failed to create memory comparison plot: {e}")
        
        # 3. Quality metrics comparison
        try:
            fig_path = self._plot_quality_comparison(output_dir)
            if fig_path:
                plot_files.append(fig_path)
        except Exception as e:
            logger.error(f"Failed to create quality comparison plot: {e}")
        
        # 4. Algorithm performance heatmap
        try:
            fig_path = self._plot_performance_heatmap(output_dir)
            if fig_path:
                plot_files.append(fig_path)
        except Exception as e:
            logger.error(f"Failed to create performance heatmap: {e}")
        
        # 5. Dataset-specific performance
        try:
            fig_path = self._plot_dataset_performance(output_dir)
            if fig_path:
                plot_files.append(fig_path)
        except Exception as e:
            logger.error(f"Failed to create dataset performance plot: {e}")
        
        logger.info(f"Created {len(plot_files)} visualization plots")
        return plot_files
    
    def _plot_speed_comparison(self, output_dir: Path) -> Optional[Path]:
        """Create speed comparison plot."""
        speed_cols = [col for col in self.processed_df.columns if 'speed_' in col and '_mean' not in col]
        
        if not speed_cols:
            logger.warning("No speed metrics found for plotting")
            return None
        
        # Use the main speed metric (fit_predict or similar)
        main_speed_col = 'speed_fit_predict'
        if main_speed_col not in speed_cols:
            main_speed_col = speed_cols[0]
        
        plt.figure(figsize=(12, 8))
        
        # Box plot of speed by algorithm
        speed_data = self.processed_df.dropna(subset=[main_speed_col])
        
        if len(speed_data) == 0:
            logger.warning("No valid speed data for plotting")
            return None
        
        sns.boxplot(data=speed_data, x='algorithm_name', y=main_speed_col)
        plt.xticks(rotation=45, ha='right')
        plt.title('Algorithm Speed Comparison')
        plt.ylabel('Time (seconds)')
        plt.tight_layout()
        
        fig_path = output_dir / 'speed_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig_path
    
    def _plot_memory_comparison(self, output_dir: Path) -> Optional[Path]:
        """Create memory usage comparison plot."""
        memory_cols = [col for col in self.processed_df.columns if 'memory_' in col and 'used_mb' in col]
        
        if not memory_cols:
            logger.warning("No memory metrics found for plotting")
            return None
        
        main_memory_col = memory_cols[0]
        
        plt.figure(figsize=(12, 8))
        
        memory_data = self.processed_df.dropna(subset=[main_memory_col])
        
        if len(memory_data) == 0:
            logger.warning("No valid memory data for plotting")
            return None
        
        sns.boxplot(data=memory_data, x='algorithm_name', y=main_memory_col)
        plt.xticks(rotation=45, ha='right')
        plt.title('Algorithm Memory Usage Comparison')
        plt.ylabel('Memory (MB)')
        plt.tight_layout()
        
        fig_path = output_dir / 'memory_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig_path
    
    def _plot_quality_comparison(self, output_dir: Path) -> Optional[Path]:
        """Create clustering quality comparison plot."""
        quality_cols = [col for col in self.processed_df.columns if 'quality_' in col]
        
        if not quality_cols:
            logger.warning("No quality metrics found for plotting")
            return None
        
        # Focus on silhouette score if available
        main_quality_col = 'quality_silhouette_score'
        if main_quality_col not in quality_cols:
            main_quality_col = quality_cols[0]
        
        plt.figure(figsize=(12, 8))
        
        quality_data = self.processed_df.dropna(subset=[main_quality_col])
        
        if len(quality_data) == 0:
            logger.warning("No valid quality data for plotting")
            return None
        
        sns.boxplot(data=quality_data, x='algorithm_name', y=main_quality_col)
        plt.xticks(rotation=45, ha='right')
        plt.title('Algorithm Clustering Quality Comparison')
        plt.ylabel('Quality Score')
        plt.tight_layout()
        
        fig_path = output_dir / 'quality_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig_path
    
    def _plot_performance_heatmap(self, output_dir: Path) -> Optional[Path]:
        """Create performance heatmap across algorithms and datasets."""
        
        # Create pivot table for heatmap
        speed_col = [col for col in self.processed_df.columns if 'speed_' in col]
        if not speed_col:
            logger.warning("No speed metrics for heatmap")
            return None
        
        speed_col = speed_col[0]
        
        # Group by algorithm and dataset, take mean of speed
        heatmap_data = self.processed_df.groupby(['algorithm_name', 'dataset_name'])[speed_col].mean().unstack()
        
        if heatmap_data.empty:
            logger.warning("No data for heatmap")
            return None
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title('Algorithm Performance Heatmap (Speed)')
        plt.ylabel('Algorithm')
        plt.xlabel('Dataset')
        plt.tight_layout()
        
        fig_path = output_dir / 'performance_heatmap.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig_path
    
    def _plot_dataset_performance(self, output_dir: Path) -> Optional[Path]:
        """Create dataset-specific performance analysis."""
        
        speed_col = [col for col in self.processed_df.columns if 'speed_' in col]
        if not speed_col:
            logger.warning("No speed metrics for dataset performance plot")
            return None
        
        speed_col = speed_col[0]
        
        # Create subplots for each dataset category
        categories = self.processed_df['dataset_category'].unique()
        
        if len(categories) == 0:
            logger.warning("No dataset categories found")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, category in enumerate(categories[:4]):  # Limit to 4 categories
            if i >= len(axes):
                break
                
            cat_data = self.processed_df[self.processed_df['dataset_category'] == category]
            
            if len(cat_data) > 0:
                sns.boxplot(data=cat_data, x='algorithm_name', y=speed_col, ax=axes[i])
                axes[i].set_title(f'Performance in {category} datasets')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(categories), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        fig_path = output_dir / 'dataset_performance.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig_path
    
    def generate_performance_report(self, output_file: Optional[Path] = None) -> Path:
        """
        Generate a comprehensive performance report.
        
        Args:
            output_file: Path to save the report (defaults to results/reports/report.html)
            
        Returns:
            Path to the generated report
        """
        if output_file is None:
            reports_dir = self.results_dir / 'reports'
            reports_dir.mkdir(parents=True, exist_ok=True)
            output_file = reports_dir / f'performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        
        # Generate performance comparisons
        speed_comparison = self.generate_performance_comparison('speed')
        memory_comparison = self.generate_performance_comparison('memory')
        quality_comparison = self.generate_performance_comparison('quality')
        
        # Create HTML report
        html_content = self._create_html_report(speed_comparison, memory_comparison, quality_comparison)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Performance report generated: {output_file}")
        return output_file
    
    def _create_html_report(self, speed_comp: pd.DataFrame, 
                           memory_comp: pd.DataFrame, 
                           quality_comp: pd.DataFrame) -> str:
        """Create HTML report content."""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Performance Benchmark Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary">
                <h2>Experiment Summary</h2>
                <p>Total successful runs: {len(self.processed_df) if self.processed_df is not None else 0}</p>
                <p>Algorithms tested: {len(self.processed_df['algorithm_name'].unique()) if self.processed_df is not None else 0}</p>
                <p>Datasets tested: {len(self.processed_df['dataset_name'].unique()) if self.processed_df is not None else 0}</p>
            </div>
            
            <h2>Speed Performance Comparison</h2>
            {speed_comp.to_html(index=False) if not speed_comp.empty else '<p>No speed data available</p>'}
            
            <h2>Memory Usage Comparison</h2>
            {memory_comp.to_html(index=False) if not memory_comp.empty else '<p>No memory data available</p>'}
            
            <h2>Quality Metrics Comparison</h2>
            {quality_comp.to_html(index=False) if not quality_comp.empty else '<p>No quality data available</p>'}
            
        </body>
        </html>
        """
        
        return html
    
    def get_top_performers(self, metric: str, n_top: int = 5) -> pd.DataFrame:
        """
        Get top performing algorithms for a specific metric.
        
        Args:
            metric: Metric name to rank by
            n_top: Number of top performers to return
            
        Returns:
            DataFrame with top performers
        """
        if self.processed_df is None:
            raise ValueError("No processed data available")
        
        # Check if metric exists
        if metric not in self.processed_df.columns:
            available_metrics = [col for col in self.processed_df.columns if any(cat in col for cat in ['speed_', 'memory_', 'quality_', 'evaluation_'])]
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {available_metrics}")
        
        # Group by algorithm and calculate mean
        top_performers = (self.processed_df.groupby('algorithm_name')[metric]
                         .agg(['mean', 'std', 'count'])
                         .reset_index())
        
        # Sort based on metric type (higher or lower is better)
        higher_is_better = any(term in metric.lower() for term in [
            'silhouette', 'calinski', 'rand', 'mutual', 'homogeneity',
            'completeness', 'v_measure', 'fowlkes', 'efficiency'
        ])
        
        top_performers = top_performers.sort_values('mean', ascending=not higher_is_better)
        
        return top_performers.head(n_top)