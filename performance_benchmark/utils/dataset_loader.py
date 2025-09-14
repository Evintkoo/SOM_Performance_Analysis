"""
Dataset loader utility for loading and preprocessing datasets from various formats.
Supports Excel, CSV files and handles different dataset structures.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Utility class for loading and preprocessing clustering datasets.
    
    Supports loading from Excel (.xlsx) and CSV files, with automatic
    data cleaning and preprocessing for clustering algorithms.
    """
    
    def __init__(self, datasets_root: str = None):
        """
        Initialize the dataset loader.
        
        Args:
            datasets_root: Root directory containing datasets
        """
        if datasets_root is None:
            # Default to datasets folder in project root
            project_root = Path(__file__).parent.parent.parent
            self.datasets_root = project_root / "datasets"
        else:
            self.datasets_root = Path(datasets_root)
            
        if not self.datasets_root.exists():
            raise FileNotFoundError(f"Datasets directory not found: {self.datasets_root}")
            
        logger.info(f"Dataset loader initialized with root: {self.datasets_root}")
    
    def list_available_datasets(self) -> Dict[str, List[str]]:
        """
        List all available datasets organized by category.
        
        Returns:
            Dictionary mapping category names to list of dataset files
        """
        datasets = {}
        
        # Check for datasets directly in the root datasets folder
        root_files = []
        for file_path in self.datasets_root.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.xlsx', '.csv', '.xls', '.txt', '.tsv']:
                root_files.append(file_path.name)
        
        if root_files:
            datasets['root'] = sorted(root_files)
        
        # Check for datasets in subdirectories (categories)
        for category_dir in self.datasets_root.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                files = []
                
                # Check all files in the category directory (including nested subdirectories)
                for file_path in category_dir.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in ['.xlsx', '.csv', '.xls', '.txt', '.tsv']:
                        # Store relative path within the category for nested files
                        relative_path = str(file_path.relative_to(category_dir))
                        files.append(relative_path)
                
                if files:
                    datasets[category_name] = sorted(files)
        
        return datasets
    
    def load_dataset(self, category: str, filename: str, 
                    has_labels: bool = False, 
                    label_column: Union[str, int] = -1,
                    normalize: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
        """
        Load a single dataset from the specified category and file.
        
        Args:
            category: Category subdirectory name or 'root' for files in datasets root
            filename: Dataset filename (can include subdirectory path for nested files)
            has_labels: Whether the dataset contains true labels
            label_column: Column index or name for labels (default: last column)
            normalize: Whether to normalize features to [0,1] range
            
        Returns:
            Tuple of (features, labels, metadata)
        """
        if category == 'root':
            file_path = self.datasets_root / filename
        else:
            file_path = self.datasets_root / category / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Load data based on file extension
        file_extension = file_path.suffix.lower()
        if file_extension in ['.xlsx', '.xls']:
            data = pd.read_excel(file_path)
        elif file_extension == '.csv':
            data = pd.read_csv(file_path)
        elif file_extension in ['.txt', '.tsv']:
            # Try to determine separator
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if '\t' in first_line:
                    data = pd.read_csv(file_path, sep='\t')
                else:
                    data = pd.read_csv(file_path, sep=None, engine='python')  # Auto-detect separator
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Loaded dataset {filename} with shape {data.shape}")
        
        # Handle missing values
        if data.isnull().any().any():
            logger.warning(f"Found missing values in {filename}, filling with column means")
            # Only fill numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
            # For non-numeric columns, fill with mode or 'unknown'
            non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_columns:
                if data[col].mode().empty:
                    data[col] = data[col].fillna('unknown')
                else:
                    data[col] = data[col].fillna(data[col].mode()[0])
        
        # Separate features and labels
        labels = None
        if has_labels:
            if isinstance(label_column, str):
                if label_column in data.columns:
                    labels = data[label_column].values
                    features = data.drop(columns=[label_column])
                else:
                    raise ValueError(f"Label column '{label_column}' not found in dataset")
            else:
                # Use integer index
                if label_column == -1:
                    label_column = data.shape[1] - 1
                labels = data.iloc[:, label_column].values
                features = data.drop(data.columns[label_column], axis=1)
        else:
            features = data
        
        # Convert to numpy arrays - only use numeric features
        numeric_features = features.select_dtypes(include=[np.number])
        X = numeric_features.values
        
        if X.shape[1] == 0:
            raise ValueError(f"No numeric features found in dataset {filename}")
        
        # Normalize features if requested
        if normalize:
            X = self._normalize_features(X)
        
        # Prepare metadata
        metadata = {
            'filename': filename,
            'category': category,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'has_labels': has_labels,
            'n_classes': len(np.unique(labels)) if labels is not None else None,
            'normalized': normalize,
            'feature_names': list(numeric_features.columns),
            'original_shape': data.shape,
            'file_path': str(file_path)
        }
        
        logger.info(f"Dataset {filename}: {metadata['n_samples']} samples, "
                   f"{metadata['n_features']} features")
        
        return X, labels, metadata
    
    def load_all_datasets(self, normalize: bool = True, 
                         max_samples: Optional[int] = None,
                         has_labels: bool = False,
                         label_column: Union[str, int] = -1) -> List[Tuple[np.ndarray, Optional[np.ndarray], Dict]]:
        """
        Load all available datasets.
        
        Args:
            normalize: Whether to normalize features
            max_samples: Maximum number of samples per dataset (for memory management)
            has_labels: Whether datasets contain true labels
            label_column: Column index or name for labels
            
        Returns:
            List of (features, labels, metadata) tuples
        """
        all_datasets = []
        available = self.list_available_datasets()
        
        for category, files in available.items():
            for filename in files:
                try:
                    X, labels, metadata = self.load_dataset(
                        category=category, 
                        filename=filename,
                        has_labels=has_labels,
                        label_column=label_column,
                        normalize=normalize
                    )
                    
                    # Subsample if dataset is too large
                    if max_samples and X.shape[0] > max_samples:
                        logger.info(f"Subsampling {filename} from {X.shape[0]} to {max_samples} samples")
                        indices = np.random.choice(X.shape[0], max_samples, replace=False)
                        X = X[indices]
                        if labels is not None:
                            labels = labels[indices]
                        metadata['n_samples'] = max_samples
                        metadata['subsampled'] = True
                    else:
                        metadata['subsampled'] = False
                    
                    all_datasets.append((X, labels, metadata))
                    
                except Exception as e:
                    logger.error(f"Failed to load {category}/{filename}: {str(e)}")
                    continue
        
        logger.info(f"Successfully loaded {len(all_datasets)} datasets")
        return all_datasets
    
    def load_datasets_by_category(self, category: str, 
                                 normalize: bool = True,
                                 has_labels: bool = False,
                                 label_column: Union[str, int] = -1) -> List[Tuple[np.ndarray, Optional[np.ndarray], Dict]]:
        """
        Load all datasets from a specific category.
        
        Args:
            category: Category name (or 'root' for files in datasets root)
            normalize: Whether to normalize features
            has_labels: Whether datasets contain true labels
            label_column: Column index or name for labels
            
        Returns:
            List of (features, labels, metadata) tuples
        """
        datasets = []
        available = self.list_available_datasets()
        
        if category not in available:
            raise ValueError(f"Category '{category}' not found. Available: {list(available.keys())}")
        
        for filename in available[category]:
            try:
                X, labels, metadata = self.load_dataset(
                    category=category,
                    filename=filename,
                    has_labels=has_labels,
                    label_column=label_column,
                    normalize=normalize
                )
                datasets.append((X, labels, metadata))
            except Exception as e:
                logger.error(f"Failed to load {category}/{filename}: {str(e)}")
                continue
        
        return datasets
    
    def get_dataset_info(self) -> pd.DataFrame:
        """
        Get summary information about all available datasets.
        
        Returns:
            DataFrame with dataset statistics
        """
        info_data = []
        available = self.list_available_datasets()
        
        for category, files in available.items():
            for filename in files:
                try:
                    if category == 'root':
                        file_path = self.datasets_root / filename
                    else:
                        file_path = self.datasets_root / category / filename
                    
                    # Load just to get basic info
                    file_extension = file_path.suffix.lower()
                    if file_extension in ['.xlsx', '.xls']:
                        data = pd.read_excel(file_path)
                    elif file_extension == '.csv':
                        data = pd.read_csv(file_path)
                    elif file_extension in ['.txt', '.tsv']:
                        # Try to determine separator
                        with open(file_path, 'r', encoding='utf-8') as f:
                            first_line = f.readline()
                            if '\t' in first_line:
                                data = pd.read_csv(file_path, sep='\t')
                            else:
                                data = pd.read_csv(file_path, sep=None, engine='python')
                    else:
                        continue  # Skip unsupported files
                    
                    info_data.append({
                        'category': category,
                        'filename': filename,
                        'n_samples': data.shape[0],
                        'n_features': data.shape[1],
                        'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                        'file_extension': file_extension
                    })
                except Exception as e:
                    logger.error(f"Failed to get info for {category}/{filename}: {str(e)}")
                    continue
        
        return pd.DataFrame(info_data)
    
    @staticmethod
    def _normalize_features(X: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0, 1] range using min-max scaling.
        
        Args:
            X: Feature matrix
            
        Returns:
            Normalized feature matrix
        """
        # Handle constant features (avoid division by zero)
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_range = X_max - X_min
        
        # Set range to 1 for constant features to avoid division by zero
        X_range[X_range == 0] = 1
        
        X_normalized = (X - X_min) / X_range
        return X_normalized
    
    def create_synthetic_dataset(self, n_samples: int = 1000, 
                                n_features: int = 2, 
                                n_clusters: int = 3,
                                noise: float = 0.1,
                                random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Create a synthetic clustering dataset for testing.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_clusters: Number of clusters
            noise: Noise level
            random_state: Random seed
            
        Returns:
            Tuple of (features, labels, metadata)
        """
        from sklearn.datasets import make_blobs
        
        X, y = make_blobs(
            n_samples=n_samples,
            centers=n_clusters,
            n_features=n_features,
            cluster_std=noise,
            random_state=random_state
        )
        
        # Normalize
        X = self._normalize_features(X)
        
        metadata = {
            'filename': f'synthetic_{n_samples}_{n_features}_{n_clusters}',
            'category': 'synthetic',
            'n_samples': n_samples,
            'n_features': n_features,
            'has_labels': True,
            'n_classes': n_clusters,
            'normalized': True,
            'synthetic': True
        }
        
        return X, y, metadata