"""
Script to create sample datasets for testing the benchmark system.
This will populate the datasets directory with various types of sample datasets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_blobs, make_circles, make_moons
import os

def create_sample_datasets():
    """Create various sample datasets for testing."""
    
    # Create datasets directory structure
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different categories
    (datasets_dir / "synthetic").mkdir(exist_ok=True)
    (datasets_dir / "real_world").mkdir(exist_ok=True)
    
    print("Creating sample datasets...")
    
    # 1. Simple blobs dataset (directly in datasets root)
    X_blobs, y_blobs = make_blobs(n_samples=1000, centers=4, n_features=2, 
                                  random_state=42, cluster_std=1.5)
    df_blobs = pd.DataFrame(X_blobs, columns=['feature_1', 'feature_2'])
    df_blobs['cluster'] = y_blobs
    df_blobs.to_csv(datasets_dir / "simple_blobs.csv", index=False)
    print("Created: simple_blobs.csv (root)")
    
    # 2. Circles dataset (synthetic category)
    X_circles, y_circles = make_circles(n_samples=800, noise=0.1, factor=0.5, random_state=42)
    df_circles = pd.DataFrame(X_circles, columns=['x', 'y'])
    df_circles['label'] = y_circles
    df_circles.to_csv(datasets_dir / "synthetic" / "circles.csv", index=False)
    print("Created: synthetic/circles.csv")
    
    # 3. Moons dataset (synthetic category)
    X_moons, y_moons = make_moons(n_samples=600, noise=0.15, random_state=42)
    df_moons = pd.DataFrame(X_moons, columns=['dim1', 'dim2'])
    df_moons['class'] = y_moons
    df_moons.to_csv(datasets_dir / "synthetic" / "moons.csv", index=False)
    print("Created: synthetic/moons.csv")
    
    # 4. High-dimensional blobs (synthetic category)
    X_high_dim, y_high_dim = make_blobs(n_samples=500, centers=3, n_features=10, 
                                        random_state=42, cluster_std=2.0)
    df_high_dim = pd.DataFrame(X_high_dim, columns=[f'feature_{i}' for i in range(10)])
    df_high_dim['target'] = y_high_dim
    df_high_dim.to_excel(datasets_dir / "synthetic" / "high_dimensional.xlsx", index=False)
    print("Created: synthetic/high_dimensional.xlsx")
    
    # 5. Unlabeled dataset for clustering (real_world category)
    np.random.seed(42)
    # Simulate customer data
    n_customers = 1200
    customer_data = pd.DataFrame({
        'age': np.random.normal(35, 12, n_customers).clip(18, 80),
        'income': np.random.lognormal(10, 0.5, n_customers).clip(20000, 200000),
        'spending_score': np.random.normal(50, 15, n_customers).clip(1, 100),
        'annual_purchases': np.random.poisson(12, n_customers),
        'satisfaction_rating': np.random.uniform(1, 5, n_customers)
    })
    customer_data.to_csv(datasets_dir / "real_world" / "customer_data.csv", index=False)
    print("Created: real_world/customer_data.csv (unlabeled)")
    
    # 6. Mixed data types dataset (real_world category)
    mixed_data = pd.DataFrame({
        'numerical_1': np.random.normal(0, 1, 800),
        'numerical_2': np.random.exponential(2, 800),
        'categorical_1': np.random.choice(['A', 'B', 'C'], 800),
        'numerical_3': np.random.uniform(-5, 5, 800),
        'categorical_2': np.random.choice(['X', 'Y', 'Z'], 800),
        'target_numeric': np.random.choice([0, 1, 2], 800)
    })
    # Convert categorical columns to numeric for clustering
    mixed_data['categorical_1_encoded'] = pd.Categorical(mixed_data['categorical_1']).codes
    mixed_data['categorical_2_encoded'] = pd.Categorical(mixed_data['categorical_2']).codes
    mixed_data_numeric = mixed_data[['numerical_1', 'numerical_2', 'categorical_1_encoded', 
                                    'numerical_3', 'categorical_2_encoded', 'target_numeric']]
    mixed_data_numeric.to_excel(datasets_dir / "real_world" / "mixed_types.xlsx", index=False)
    print("Created: real_world/mixed_types.xlsx")
    
    # 7. Small test dataset (tab-separated, directly in datasets root)
    small_data = pd.DataFrame({
        'x': [1, 2, 8, 9, 1.5, 2.5, 8.5, 9.5],
        'y': [1, 2, 8, 9, 1.5, 2.5, 8.5, 9.5],
        'group': [0, 0, 1, 1, 0, 0, 1, 1]
    })
    small_data.to_csv(datasets_dir / "small_test.tsv", sep='\t', index=False)
    print("Created: small_test.tsv (tab-separated)")
    
    print(f"\nAll sample datasets created in: {datasets_dir.absolute()}")
    print("Dataset summary:")
    print("- Root level: simple_blobs.csv, small_test.tsv")
    print("- synthetic/: circles.csv, moons.csv, high_dimensional.xlsx")
    print("- real_world/: customer_data.csv, mixed_types.xlsx")
    
    # Print directory structure
    print(f"\nDirectory structure:")
    for root, dirs, files in os.walk(datasets_dir):
        level = root.replace(str(datasets_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

if __name__ == "__main__":
    create_sample_datasets()