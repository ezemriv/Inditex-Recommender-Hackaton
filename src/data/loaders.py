import polars as pl

class PolarsLoader():
    def __init__(self, sampling=False, n_sample = 1_000_000, file_type='csv'):
        """
        Initializes the PolarsLoader class.

        Parameters:
            sampling (bool): If True, loads a sample of 1,000,000 rows from the dataset. Defaults to False.
        """
        self.sampling = sampling
        self.file_type = file_type
        self.n_sample = n_sample
    
    def load_data(self, path, select_cols=None):
        
        """
        Loads the data from a CSV file.

        Parameters:
            path (str): Path to the CSV file.

        Returns:
            polars.DataFrame: Loaded DataFrame without selected columns.
        """
        if self.sampling:
            N_ROWS = self.n_sample
        else:
            N_ROWS = None

        if self.file_type == 'csv':
            # Read dataset as polars DataFrame
            df = pl.read_csv(path, low_memory=True,
                            batch_size=100_000, 
                            n_rows=N_ROWS,
                            try_parse_dates=True,
                            columns=select_cols,
                            )
        elif self.file_type == 'parquet':
            # Read dataset as polars DataFrame
            df = pl.read_parquet(path, low_memory=True,
                            n_rows=N_ROWS,
                            columns=select_cols,
                            )
        else:
            raise ValueError("Unsupported file type. Supported types are 'csv' and 'parquet'.")

        return df
    
    def load_data_lazy(self, path):
        """
        Loads the data from a CSV file lazily.

        Parameters:
            path (str): Path to the CSV file.

        Returns:
            polars.LazyFrame: LazyFrame object for the data.
        """
        if self.sampling:
            N_ROWS = self.n_sample
        else:
            N_ROWS = None

        if self.file_type == 'csv':
            # Read dataset as polars LazyFrame
            lf = pl.scan_csv(path, low_memory=True,
                                n_rows=N_ROWS,
                                try_parse_dates=True,
                                )
        elif self.file_type == 'parquet':
            # Read dataset as polars LazyFrame
            lf = pl.scan_parquet(path, low_memory=True,
                                n_rows=N_ROWS,
                                )
        else:
            raise ValueError("Unsupported file type. Supported types are 'csv' and 'parquet'.")

        return lf