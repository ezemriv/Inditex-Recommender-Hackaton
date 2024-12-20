import polars as pl

class PolarsLoader():
    def __init__(self, sampling=False, file_type='csv'):
        """
        Initializes the PolarsLoader class.

        Parameters:
            sampling (bool): If True, loads a sample of 1,000,000 rows from the dataset. Defaults to False.
        """
        self.sampling = sampling
        self.file_type = file_type
    
    def load_data(self, path):
        
        """
        Loads the data from a CSV file.

        Parameters:
            path (str): Path to the CSV file.

        Returns:
            polars.DataFrame: Loaded DataFrame without selected columns.
        """
        if self.sampling:
            N_ROWS = 1_000_000
        else:
            N_ROWS = None

        if self.file_type == 'csv':
            # Read dataset as polars DataFrame
            df = pl.read_csv(path, low_memory=True,
                            batch_size=100_000, 
                            n_rows=N_ROWS,
                            try_parse_dates=True
                            )
        elif self.file_type == 'parquet':
            # Read dataset as polars DataFrame
            df = pl.read_parquet(path, low_memory=True,
                            n_rows=N_ROWS,
                            )
        else:
            raise ValueError("Unsupported file type. Supported types are 'csv' and 'parquet'.")

        return df