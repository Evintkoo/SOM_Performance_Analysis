"""
Main entry point for running the comprehensive performance benchmark.
This script provides command-line interface for running SOM vs sklearn clustering comparisons.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from performance_benchmark.core.experiment_config import ExperimentConfig, ConfigTemplates
    from performance_benchmark.core.benchmark_runner import BenchmarkRunner
    from performance_benchmark.visualization.results_analyzer import ResultsAnalyzer
    from performance_benchmark.utils.dataset_loader import DatasetLoader
    from performance_benchmark.utils.logging_config import setup_logging, LoggingConfig, LoggingTemplates
except ImportError as e:
    print(f"Error importing benchmark modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup basic logging configuration for main script."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def run_benchmark(config_template: str = "comprehensive", 
                 custom_config_file: str = None,
                 datasets_root: str = None) -> None:
    """
    Run the performance benchmark.
    
    Args:
        config_template: Name of the configuration template to use
        custom_config_file: Path to custom configuration file
        datasets_root: Custom datasets root directory
    """
    
    # Load or create configuration
    if custom_config_file:
        print(f"Loading configuration from {custom_config_file}")
        config = ExperimentConfig.load_from_file(custom_config_file)
    else:
        print(f"Using {config_template} configuration template")
        
        # Get configuration template
        if config_template == "quick":
            config = ConfigTemplates.quick_test()
        elif config_template == "comprehensive":
            config = ConfigTemplates.comprehensive()
        elif config_template == "memory_constrained":
            config = ConfigTemplates.memory_constrained()
        elif config_template == "speed_focused":
            config = ConfigTemplates.speed_focused()
        else:
            print(f"Unknown template: {config_template}")
            print("Available templates: quick, comprehensive, memory_constrained, speed_focused")
            return
    
    # Override datasets root if provided
    if datasets_root:
        config.datasets_root = datasets_root
    elif config.datasets_root is None:
        # Use default datasets directory
        project_root = Path(__file__).parent
        config.datasets_root = str(project_root / "datasets")
    
    # Setup basic logging for main script
    setup_logging(config.log_level)
    logger = logging.getLogger("main")
    
    print("\n" + "="*60)
    print("SOM vs SKlearn Clustering Performance Benchmark")
    print("="*60)
    print(f"Experiment: {config.experiment_name}")
    print(f"Datasets root: {config.datasets_root}")
    print(f"Results will be saved to: {config.results_dir}")
    print(f"Number of runs per algorithm-dataset: {config.n_runs}")
    if config.n_runs > 1:
        print(f"Aggregation methods: {config.aggregation_methods}")
        print(f"Statistical significance testing: {config.statistical_significance}")
    print("\nConfiguration Summary:")
    print(config.summary())
    print("="*60 + "\n")
    
    # Verify datasets directory exists
    if not Path(config.datasets_root).exists():
        print(f"Error: Datasets directory not found: {config.datasets_root}")
        print("Please check the path or provide a valid datasets directory using --datasets-root")
        return
    
    try:
        # Initialize and run benchmark
        runner = BenchmarkRunner(config)
        
        print("Starting benchmark execution...")
        start_time = datetime.now()
        
        results = runner.run_full_benchmark()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\nBenchmark completed successfully!")
        print(f"Total execution time: {duration}")
        print(f"Results saved to: {config.results_dir}")
        
        # Print summary
        print("\n" + runner.get_results_summary())
        
        # Generate analysis and visualizations
        print("\nGenerating analysis and visualizations...")
        analyzer = ResultsAnalyzer(config.results_dir)
        analyzer.load_results()
        
        # Create plots
        plot_files = analyzer.create_performance_plots()
        print(f"Created {len(plot_files)} visualization plots")
        
        # Generate report
        report_file = analyzer.generate_performance_report()
        print(f"Performance report generated: {report_file}")
        
        print(f"\nBenchmark complete! Check {config.results_dir} for all results.")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return


def analyze_existing_results(results_dir: str) -> None:
    """
    Analyze existing benchmark results.
    
    Args:
        results_dir: Directory containing benchmark results
    """
    print(f"Analyzing results from: {results_dir}")
    
    try:
        analyzer = ResultsAnalyzer(results_dir)
        analyzer.load_results()
        
        print("Generating analysis and visualizations...")
        
        # Create plots
        plot_files = analyzer.create_performance_plots()
        print(f"Created {len(plot_files)} visualization plots")
        
        # Generate report
        report_file = analyzer.generate_performance_report()
        print(f"Performance report generated: {report_file}")
        
        # Show top performers
        try:
            print("\nTop performers by speed:")
            speed_cols = [col for col in analyzer.processed_df.columns if 'speed_' in col]
            if speed_cols:
                top_speed = analyzer.get_top_performers(speed_cols[0])
                print(top_speed.to_string(index=False))
        except Exception as e:
            print(f"Could not generate top performers: {e}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()


def list_datasets(datasets_root: str = None) -> None:
    """
    List available datasets.
    
    Args:
        datasets_root: Datasets root directory
    """
    if datasets_root is None:
        datasets_root = Path(__file__).parent / "datasets"
    
    try:
        loader = DatasetLoader(datasets_root)
        available_datasets = loader.list_available_datasets()
        
        print(f"Available datasets in {datasets_root}:")
        print("="*50)
        
        for category, files in available_datasets.items():
            print(f"\n{category}:")
            for filename in files:
                print(f"  - {filename}")
        
        print(f"\nTotal: {sum(len(files) for files in available_datasets.values())} datasets")
        
        # Get dataset info
        info_df = loader.get_dataset_info()
        if not info_df.empty:
            print("\nDataset Statistics:")
            print(info_df.to_string(index=False))
        
    except Exception as e:
        print(f"Failed to list datasets: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SOM vs SKlearn Clustering Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run quick test benchmark
    uv run main.py --template quick
    
    # Run comprehensive benchmark
    uv run main.py --template comprehensive
    
    # Run with custom datasets directory
    uv run main.py --template comprehensive --datasets-root /path/to/datasets
    
    # Use custom configuration file
    uv run main.py --config config.json
    
    # Analyze existing results
    uv run main.py --analyze results/
    
    # List available datasets
    uv run main.py --list-datasets
        """
    )
    
    parser.add_argument(
        '--template', 
        choices=['quick', 'comprehensive', 'memory_constrained', 'speed_focused'],
        default='comprehensive',
        help='Configuration template to use (default: comprehensive)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file (JSON format)'
    )
    
    parser.add_argument(
        '--datasets-root',
        type=str,
        help='Root directory containing datasets'
    )
    
    parser.add_argument(
        '--analyze',
        type=str,
        help='Analyze existing results from specified directory'
    )
    
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List available datasets and exit'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup basic logging
    setup_logging(args.log_level)
    
    # Handle different modes
    if args.list_datasets:
        list_datasets(args.datasets_root)
    elif args.analyze:
        analyze_existing_results(args.analyze)
    else:
        run_benchmark(
            config_template=args.template,
            custom_config_file=args.config,
            datasets_root=args.datasets_root
        )


if __name__ == "__main__":
    main()