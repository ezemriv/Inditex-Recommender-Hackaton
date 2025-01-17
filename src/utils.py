import time
import tracemalloc
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline
import os
import pandas as pd

def timer_and_memory(func):
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.2f} seconds and Peak memory usage: {peak / 10**6:.3f} MB.")
        return result
    return wrapper

def timer(func):
    """
    Decorator to measure the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.2f} seconds.")
        return result
    return wrapper

def filter_polars(polars_df, column, value):
    return polars_df.filter(polars_df[column] == value)

def plot_cross_validation_scores(train_scores, valid_scores):
    """Plots and saves the cross-validation scores."""
    plt.figure()
    plt.plot(range(1, len(train_scores) + 1), train_scores,
             marker='o', label='Train F1 Score', linestyle='-')
    plt.plot(range(1, len(valid_scores) + 1), valid_scores,
             marker='o', label='Validation F1 Score', linestyle='-')
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.title('Train and Validation F1 Scores')
    plt.legend()
    plt.show()

def plot_feature_importances(model, FEATURES):
    """Plots and saves the feature importances."""
    # Handle pipeline if oversampling is used
    if isinstance(model, Pipeline):
        classifier = model.named_steps['classifier']
    else:
        classifier = model

    # Get feature importances
    importances = classifier.feature_importances_
    importances_df = pd.DataFrame({
        'feature': FEATURES,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importances_df)
    plt.title('Feature Importances')
    plt.show()