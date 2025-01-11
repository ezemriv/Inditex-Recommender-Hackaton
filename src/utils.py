import time
import tracemalloc

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