# SOM vs SKlearn Clustering Performance Benchmark

A comprehensive performance benchmarking framework that compares Self-Organizing Map (SOM) clustering algorithms from the `SOM_plus_clustering` module with various scikit-learn clustering algorithms across multiple performance metrics including speed, memory usage, clustering quality, and evaluation scores.

## Features

- **Comprehensive Algorithm Coverage**: Tests multiple SOM initialization methods (excluding KDE, Kmeans, Kmeans++ as requested) and 9 different sklearn clustering algorithms
- **Multi-Metric Evaluation**: Measures speed, memory usage, clustering quality, and evaluation scores
- **Flexible Dataset Support**: Works with Excel (.xlsx) and CSV files from various dataset categories
- **Automated Analysis**: Generates detailed reports, visualizations, and performance comparisons
- **Configurable Experiments**: Multiple configuration templates for different use cases
- **Professional Visualization**: Creates publication-quality plots and HTML reports

## Project Structure

```
SOM_Performance_Analysis/
├── performance_benchmark/           # Main benchmark framework
│   ├── core/                       # Core benchmark logic
│   │   ├── benchmark_runner.py     # Main benchmark orchestrator
│   │   └── experiment_config.py    # Configuration management
│   ├── algorithms/                 # Algorithm wrappers
│   │   ├── som_wrapper.py          # SOM algorithm wrapper
│   │   └── sklearn_wrapper.py     # Sklearn algorithms wrapper
│   ├── metrics/                    # Performance measurement
│   │   └── performance_metrics.py  # Comprehensive metrics collection
│   ├── utils/                      # Utility modules
│   │   └── dataset_loader.py       # Dataset loading and preprocessing
│   ├── visualization/              # Analysis and visualization
│   │   └── results_analyzer.py     # Results analysis and plotting
│   └── results/                    # Generated results (created during execution)
├── datasets/                       # Your dataset files (Excel/CSV)
├── SOM_plus_clustering/            # Existing SOM implementation
├── main.py                         # Main entry point script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd SOM_Performance_Analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify SOM_plus_clustering is available**:
   The framework automatically detects and uses the existing `SOM_plus_clustering` module in your project.

## Quick Start

### Run a Quick Test Benchmark
```bash
uv run main.py --template quick
```

### Run Comprehensive Benchmark
```bash
uv run main.py --template comprehensive
```

### List Available Datasets
```bash
uv run main.py --list-datasets
```

### Analyze Existing Results
```bash
uv run main.py --analyze results/
```

## Configuration Templates

The framework provides several pre-configured templates:

### 1. Quick Test (`--template quick`)
- Limited datasets (500 samples max)
- Fast execution (20 SOM epochs)
- 1 run per algorithm
- 60-second timeout
- Perfect for testing and development

### 2. Comprehensive (`--template comprehensive`)
- All available datasets (no sample limit)
- Full training (200 SOM epochs)
- 3 runs per algorithm for statistical significance
- 10-minute timeout
- Complete analysis with model saving

### 3. Memory Constrained (`--template memory_constrained`)
- Limited to 1000 samples per dataset
- Optimized for low-memory environments
- Focus on memory usage metrics
- 3-minute timeout

### 4. Speed Focused (`--template speed_focused`)
- 2000 samples max per dataset
- 5 runs per algorithm for better speed statistics
- Focus on timing metrics only
- 2-minute timeout

## Algorithms Tested

### SOM Algorithms (from SOM_plus_clustering)
The framework tests SOM with these initialization methods:
- **random**: Random weight initialization
- **som++**: SOM++ farthest-first traversal initialization
- **zero**: Zero initialization method
- **he**: He initialization
- **naive_sharding**: Naive sharding initialization
- **lecun**: LeCun initialization
- **lsuv**: LSUV (Layer-sequential unit-variance) initialization

*Note: KDE, Kmeans, and Kmeans++ initialization methods are excluded as requested.*

### Sklearn Algorithms
- **KMeans**: With k-means++ and random initialization
- **DBSCAN**: Density-based clustering
- **AgglomerativeClustering**: Hierarchical clustering
- **SpectralClustering**: Spectral clustering
- **MeanShift**: Mean shift clustering
- **Birch**: BIRCH clustering
- **OPTICS**: Ordering points clustering
- **AffinityPropagation**: Affinity propagation
- **GaussianMixture**: Gaussian mixture models

## Performance Metrics

### Speed Metrics
- **Fit Time**: Training/fitting time
- **Predict Time**: Prediction time (when applicable)
- **Total Time**: End-to-end execution time

### Memory Metrics
- **Peak Memory**: Maximum memory usage during execution
- **Memory Used**: Actual memory consumed
- **Memory Efficiency**: Data size to memory ratio

### Clustering Quality Metrics (Internal)
- **Silhouette Score**: Measure of cluster separation and cohesion
- **Calinski-Harabasz Index**: Ratio of within-cluster and between-cluster dispersion
- **Davies-Bouldin Index**: Average similarity between clusters

### Evaluation Metrics (External - when true labels available)
- **Adjusted Rand Index (ARI)**: Similarity to ground truth clustering
- **Normalized Mutual Information (NMI)**: Information shared with true labels
- **Homogeneity**: Each cluster contains only members of a single class
- **Completeness**: All members of a class are assigned to the same cluster
- **V-Measure**: Harmonic mean of homogeneity and completeness

## Dataset Format

The framework supports datasets in the following formats:

### Supported File Types
- **Excel files** (`.xlsx`): Most common format
- **CSV files** (`.csv`): Comma-separated values

### Directory Structure
Organize your datasets in categories:
```
datasets/
├── A/
│   ├── A1.xlsx
│   ├── A2.xlsx
│   └── A3.xlsx
├── Shape/
│   ├── Aggregation.xlsx
│   ├── Compound.xlsx
│   └── Flame.xlsx
└── G2/
    ├── g2-16-10.xlsx
    └── g2-1024-100.xlsx
```

### Data Format Requirements
- **Numeric data only**: All features must be numeric
- **No missing values**: Or they will be filled with column means
- **Optional labels**: Can include true cluster labels for evaluation
- **Automatic normalization**: Features are normalized to [0,1] range

## Usage Examples

### Basic Usage
```bash
# Run comprehensive benchmark
uv run main.py

# Use specific configuration template
uv run main.py --template speed_focused

# Specify custom datasets directory
uv run main.py --datasets-root /path/to/your/datasets

# Set logging level
uv run main.py --log-level DEBUG
```

### Custom Configuration
```bash
# Create custom configuration file
python -c "
from performance_benchmark.core.experiment_config import ExperimentConfig
config = ExperimentConfig(
    experiment_name='Custom_Experiment',
    som_epochs=50,
    max_samples_per_dataset=2000,
    n_runs=2
)
config.save_to_file('custom_config.json')
"

# Run with custom configuration
uv run main.py --config custom_config.json
```

### Analysis Only
```bash
# Analyze existing results without running new benchmarks
uv run main.py --analyze results/
```

## Output and Results

### Generated Files
The framework creates a comprehensive `results/` directory:

```
results/
├── raw_results/              # Raw benchmark data (pickle files)
├── processed_results/        # CSV and JSON summaries
├── visualizations/           # Performance plots and charts
├── reports/                  # HTML performance reports
├── models/                   # Trained models (if enabled)
├── predictions/              # Clustering predictions
├── logs/                     # Execution logs
└── configs/                  # Configuration files used
```

### Visualizations
- **Speed Comparison**: Box plots of algorithm execution times
- **Memory Usage**: Memory consumption across algorithms
- **Quality Metrics**: Clustering quality comparisons
- **Performance Heatmaps**: Algorithm performance across datasets
- **Dataset-Specific Analysis**: Performance broken down by dataset category

### Reports
- **HTML Reports**: Comprehensive performance analysis with tables and statistics
- **CSV Results**: Easy-to-analyze tabular data for further processing
- **Summary Statistics**: Aggregated performance metrics

## Advanced Usage

### Programmatic Access
```python
from performance_benchmark import BenchmarkRunner, ExperimentConfig
from performance_benchmark.visualization import ResultsAnalyzer

# Create custom configuration
config = ExperimentConfig(
    experiment_name="My_Custom_Benchmark",
    som_epochs=100,
    include_som_algorithms=True,
    include_sklearn_algorithms=True,
    measure_speed=True,
    measure_memory=True,
    measure_quality=True
)

# Run benchmark
runner = BenchmarkRunner(config)
results = runner.run_full_benchmark()

# Analyze results
analyzer = ResultsAnalyzer(config.results_dir)
analyzer.load_results()
analyzer.create_performance_plots()
report = analyzer.generate_performance_report()
```

### Custom Metrics
You can extend the framework by adding custom performance metrics:

```python
from performance_benchmark.metrics import PerformanceMetrics

class CustomMetrics(PerformanceMetrics):
    def measure_custom_metric(self, X, labels, algorithm_name):
        # Your custom metric implementation
        custom_score = your_metric_function(X, labels)
        self.metrics['custom'][f'{algorithm_name}_custom_score'] = custom_score
        return custom_score
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running the script from the project root directory
2. **Dataset Not Found**: Verify the datasets directory path and file formats
3. **Memory Issues**: Use `memory_constrained` template or reduce `max_samples_per_dataset`
4. **GPU Issues**: CuPy is optional; the framework works with CPU-only numpy

### Performance Tips

1. **For Large Datasets**: Use `max_samples_per_dataset` to limit memory usage
2. **For Speed**: Use `speed_focused` template with fewer epochs
3. **For Accuracy**: Use `comprehensive` template with multiple runs
4. **For Memory**: Use `memory_constrained` template

### Logging
Increase logging verbosity for debugging:
```bash
uv run main.py --log-level DEBUG
```

## Contributing

To extend the framework:

1. **Add New Algorithms**: Create wrappers in `algorithms/` directory
2. **Add New Metrics**: Extend the `PerformanceMetrics` class
3. **Add New Visualizations**: Extend the `ResultsAnalyzer` class
4. **Add New Datasets**: Place files in the `datasets/` directory

## License

This project uses the same license as the existing SOM_plus_clustering module.

## Citation

When using this benchmark framework in research, please cite both the original SOM_plus_clustering implementation and this benchmarking framework.

---

**Note**: This benchmarking framework is designed to work with your existing `SOM_plus_clustering` module without modifying any of its code, as requested. All SOM functionality is accessed through the wrapper interface to maintain compatibility and independence.