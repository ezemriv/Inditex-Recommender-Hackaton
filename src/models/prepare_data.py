import pandas as pd
import polars as pl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from src.data.loaders import PolarsLoader
from src.utils import timer_and_memory, timer

class PRODUCTS:
    def __init__(self, processed_path, matrix_out_path, engineered_path):
        self.processed_path = processed_path
        self.matrix_out_path = matrix_out_path
        self.engineered_path = engineered_path
        self.products_df = None

    def load_data(self):
        df = pl.read_parquet(self.processed_path, low_memory=True)
        self.products_df = df
        return df

    def calculate_similarity(self, df):
        embeddings = np.stack(df["embedding"].to_numpy())
        normalized_embeddings = normalize(embeddings)
        cosine_sim_matrix = cosine_similarity(normalized_embeddings)
        cosine_sim_df = pd.DataFrame(
            cosine_sim_matrix, 
            index=df["partnumber"], 
            columns=df["partnumber"]
        )
        cosine_sim_df.to_parquet(self.matrix_out_path)

    def feature_engineering(self, df):
        return df

    @timer
    def run(self):
        df = self.load_data()
        # self.calculate_similarity(df) --> DONE IN KAGGLE
        df = self.feature_engineering(df)
        return df

class TRAIN_TEST:
    def __init__(self, sampling=False, train_path=None, test_path=None, 
                 save_train_path=None, save_test_path=None, 
                 topN_global=10):
        self.sampling = sampling
        self.train_path = train_path
        self.test_path = test_path
        self.save_train_path = save_train_path
        self.save_test_path = save_test_path
        self.topN_global = topN_global

    def load_data(self):
        loader = PolarsLoader(sampling=self.sampling, file_type='parquet')
        train = loader.load_data(path=self.train_path)
        test = loader.load_data(path=self.test_path)
        return train, test
    
    def load_data_lazy(self):
        loader = PolarsLoader(sampling=self.sampling, file_type='parquet')
        train = loader.load_data_lazy(path=self.train_path)
        test = loader.load_data_lazy(path=self.test_path)
        return train, test
    
    def impute_train_test(self, df):
    
        return df.with_columns([pl.col("user_id").fill_null(-1).cast(pl.Int32),
                        pl.col("pagetype").fill_null(pl.col("pagetype").mode()),
                        ])
    @timer_and_memory
    def feature_engineering_no_candidate_dependent(self, df: pl.DataFrame) -> pl.DataFrame:

        # Cleaning and sorting
        df_ = (df.drop("date")
            .sort("timestamp_local")
        )

        df_ = df_.with_columns([
            # Calculate the difference in timestamps within each session
            (pl.col("timestamp_local").diff().over("session_id").cast(pl.Float32) / 1_000_000).alias("seconds_since_last_interaction"),
            # Total session duration in seconds
            ((pl.col("timestamp_local").max() - pl.col("timestamp_local").min()).over("session_id").cast(pl.Float32) / 1_000_000).alias("total_session_time"),
        ]).fill_null(strategy="zero")

        df_ = df_.with_columns([
                    pl.col("seconds_since_last_interaction").shift(-1).over("session_id").alias("interaction_length"),
        ]).fill_null(strategy="zero")

        # Date features
        df_ = df_.with_columns([
            # Extracting day number
            pl.col("timestamp_local").dt.day().alias("day_number"),
            
            # Extracting weekday number
            pl.col("timestamp_local").dt.weekday().alias("weekday_number"),
            
            # Extracting weekday name
            pl.col("timestamp_local").dt.strftime("%A").alias("weekday_name").cast(pl.Categorical),
            
            # Extracting hour
            pl.col("timestamp_local").dt.hour().alias("hour")
        ])
        
        df_ = df_.with_columns([
                    pl.when((pl.col("hour") >= 6) & (pl.col("hour") < 12)).then(pl.lit("Morning"))
                    .when((pl.col("hour") >= 12) & (pl.col("hour") < 18)).then(pl.lit("Afternoon"))
                    .when((pl.col("hour") >= 18) & (pl.col("hour") < 24)).then(pl.lit("Night"))
                    .otherwise(pl.lit("Late Night"))
                    .cast(pl.Categorical)
                    .alias("day_frame")
                ])

        return df_
    
    def select_global_candidates(self, train_df):

        cart_counts = (train_df
            .group_by('partnumber')
            .agg(pl.col('add_to_cart').count())
            .sort('add_to_cart', descending=True)
            )
        
        top_global_prods = cart_counts['partnumber'].head(self.topN_global).to_list()

        return top_global_prods
    
    def add_candidates_to_test(self, test_df: pl.DataFrame, candidate_products: list) -> pl.DataFrame:
        """Adds candidate products as new rows for each session in test data"""
        test_session_ids = test_df["session_id"].unique()

        # Create a dataframe with all the candidate products for each session
        candidates = [
            {"session_id": sid, "partnumber": pn}
            for sid in test_session_ids 
            for pn in candidate_products
        ]
        candidates_df = pl.DataFrame(candidates).with_columns([
            pl.col("session_id").cast(pl.UInt32),
            pl.col("partnumber").cast(pl.UInt16)
        ])
        
        # Add the candidate products to the test data
        test_extended = pl.concat([test_df, candidates_df], how="diagonal").unique()

        #Fill missing values with generated: Faster version
        backfilled_cols = ['timestamp_local',
                            'user_id',
                            'country',
                            'device_type',
                            'pagetype',
                            'total_session_time',
                            'day_number',
                            'weekday_number',
                            'weekday_name',
                            'hour',
                            'day_frame']
        
        statsfilled_cols = ['seconds_since_last_interaction',
                            'interaction_length']
        
        test_extended = (test_extended
                        .sort(["session_id", "timestamp_local"], descending=[False, True])
                        .with_columns([
                            pl.col(col).fill_null(strategy='backward').over("session_id") 
                            for col in backfilled_cols
                        ])
                        .with_columns([
                            pl.col(col).fill_null(strategy='mean').over("session_id") 
                            for col in statsfilled_cols
                        ])
                    )

        return test_extended
    
    def feature_engineering_candidate_dependent(self, df: pl.DataFrame) -> pl.DataFrame:
    
        df = df.with_columns([
            # Assign a cumulative count for each partnumber within a session
            pl.col("partnumber").cum_count().over(["session_id", "partnumber"]).alias("product_interaction_count")
        ]).fill_null(strategy="zero")

        return df
    
    @timer
    def run(self, process_train=True, candidates_from_train=None):
        """
        Processes train and test data, with options to skip training data processing
        and dynamically update candidates for the test dataset.
        
        Args:
            process_train (bool): Whether to process the training data.
            candidates_from_train (list or None): Precomputed list of candidate products from training data. 
                                                If None, it will compute the candidates.
        
        Returns:
            train_eng (pl.DataFrame): Processed training data (if process_train is True).
            test_extended (pl.DataFrame): Processed and candidate-augmented test data.
        """
        # Load data
        print("Loading train and test data...")
        # train, test = self.load_data()
        train, test = self.load_data_lazy()

        if process_train:
            print("Processing train data...")
            print("Imputing missing values...")
            train = self.impute_train_test(train)
            train_eng = self.feature_engineering_no_candidate_dependent(train)
            train_eng = self.feature_engineering_candidate_dependent(train_eng)

            #######################
            ### IF LAZY LOADING ###
            #######################
            train_eng = train_eng.collect()

            print("Saving parquet file...")
            train_eng.write_parquet(self.save_train_path)
        else:
            print("Skipping train processing. Loading preprocessed data...")
            train_eng = pl.read_parquet(self.save_train_path)

        # Test processing
        print("Processing test data...")
        test = self.impute_train_test(test)
        test_eng = self.feature_engineering_no_candidate_dependent(test)

        # Candidate selection
        if candidates_from_train is None:
            print("Selecting global candidates from training data...")
            global_candidates = self.select_global_candidates(train_eng)
        else:
            print("Using precomputed candidates...")
            global_candidates = candidates_from_train

        # Add candidates to test
        print(f"Adding {len(global_candidates)} candidate products to test data...")
        
        #######################
        ### IF LAZY LOADING ###
        #######################
        test_eng = test_eng.collect()

        test_extended = self.add_candidates_to_test(test_eng, global_candidates)
        test_extended = self.feature_engineering_candidate_dependent(test_extended)
        test_extended.write_parquet(self.save_test_path)

        print(f"Data processing completed.")
        return train_eng, test_extended