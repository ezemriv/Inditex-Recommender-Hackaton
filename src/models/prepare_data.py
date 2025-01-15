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

    def load_data(self):
        df = pl.read_parquet(self.processed_path, low_memory=True)
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
        df.write_parquet(self.engineered_path)
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
        
        # Add tag for known user first
        df = df.with_columns([
            pl.when(pl.col("user_id").is_null()).then(pl.lit(-1)).otherwise(pl.lit(1)).alias("known_user"),
        ])

        return df.with_columns([pl.col("user_id").fill_null(-1).cast(pl.Int32),
                        pl.col("pagetype").fill_null(pl.col("pagetype").mode()),
                        ])
    
    @timer_and_memory
    def create_concatenated_features(self, train: pl.DataFrame, 
                               test: pl.DataFrame) -> tuple[pl.DataFrame, 
                                                            pl.DataFrame]:
        """
        Create cart ratio features by concatenating train and test data, computing ratios,
        then returning the separated dataframes with new features.
        
        Args:
            train (pl.DataFrame): Training dataframe containing 'add_to_cart' column
            test (pl.DataFrame): Test dataframe without 'add_to_cart' column
        
        Returns:
            tuple[pl.DataFrame, pl.DataFrame]: Tuple containing (train, test) dataframes with new features
        """
        # Create dummy test dataframe with placeholder add_to_cart column
        dummy_test = (test.clone()
                    .with_columns([pl.lit(None).alias('add_to_cart')])
                    .select(sorted(train.columns, reverse=True))
                    .with_columns([pl.lit(0).alias('flag')])
        )
        
        # Prepare train dataframe with flag
        dummy_train = (train
                    .select(sorted(train.columns, reverse=True))
                    .with_columns([pl.lit(1).alias('flag')])
        )
        
        # Concatenate train and test
        traintest_concat = pl.concat([dummy_train, dummy_test])
        # Calculate page type cart addition ratio
        page_cart_ratio = (traintest_concat.group_by("pagetype")
                        .agg(pl.col("add_to_cart").mean().alias("page_cart_ratio").cast(pl.Float32))
                        .fill_null(pl.lit(-1))
                        .with_columns(pl.col("pagetype").cast(pl.UInt8))
        )
        traintest_concat = traintest_concat.join(page_cart_ratio, on="pagetype", how="left")
        # Calculate device type cart addition ratio
        device_cart_ratio = (traintest_concat.group_by("device_type")
            .agg(pl.col("add_to_cart").mean().alias("device_cart_ratio").cast(pl.Float32))
        )
        traintest_concat = traintest_concat.join(device_cart_ratio, on="device_type", how="left")
        # Calculate country cart addition ratio
        country_cart_ratio = (traintest_concat.group_by("country")
            .agg(pl.col("add_to_cart").mean().alias("country_cart_ratio").cast(pl.Float32))
        )
        traintest_concat = traintest_concat.join(country_cart_ratio, on="country", how="left")
        
        # Calculate cumulative user features
        traintest_concat = traintest_concat.with_columns([
                pl.col("add_to_cart").cum_sum().over("user_id").alias("user_previous_cart_additions"),
                pl.col("user_id").cum_count().over("user_id").alias("user_previous_interactions")
            ])
        
        # Split back into train and test based on flag
        train_processed = traintest_concat.filter(pl.col("flag") == 1).drop("flag")
        test_processed = traintest_concat.filter(pl.col("flag") == 0).drop(["flag", "add_to_cart"])
        
        return train_processed, test_processed

    @timer_and_memory
    def feature_engineering_no_candidate_dependent(self, df: pl.DataFrame) -> pl.DataFrame:

        # Cleaning and sorting
        df_ = (df.drop("date")
            .sort("timestamp_local")
        )

        df_ = df_.with_columns([
            # Calculate the difference in timestamps within each session
            (pl.col("timestamp_local").diff().over("session_id").cast(pl.Float32) / 1_000_000)
            .round(1).alias("seconds_since_last_interaction"),
            # Total session duration in seconds
            ((pl.col("timestamp_local").max() - pl.col("timestamp_local").min()).over("session_id").cast(pl.Float32) / 1_000_000)
            .round(1).alias("total_session_time"),
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
        print("Imputing missing values...")
        train = self.impute_train_test(train)
        test = self.impute_train_test(test)
        print("Creating concatenated features...")
        train, test = self.create_concatenated_features(train, test)

        if process_train:
            print("Processing train data...")
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
        print("Final size of test data:", len(test_extended))
        test_extended.write_parquet(self.save_test_path)

        print(f"Data processing completed.")
        return train_eng, test_extended
    
class USERS:
    def __init__(self, processed_path, engineered_path):
        self.processed_path = processed_path
        self.engineered_path = engineered_path

    def load_and_process_users(self):
        # Load and process user data
        users = (
            pl.read_parquet(self.processed_path, low_memory=True)
            .rename({'country': 'user_country'})
        )

        # Select one entry per user based on HIGH F and HIGH R
        users = (
            users.sort(['user_id', 'F', 'R'], descending=[False, True, True])
            .group_by('user_id')
            .agg(pl.all().first())
        )
        return users

    @staticmethod
    def create_initial_extra_features(df: pl.DataFrame) -> pl.DataFrame:

        df = df.with_columns([
            # Average value per purchase
            (pl.col('M') / pl.col('F')).alias('avg_value_per_purchase'),
            # Purchase frequency rate (F normalized by time window)
            (pl.col('F') / pl.col('R')).alias('purchase_rate').cast(pl.Float32),
            # Value density (M normalized by time window)
            (pl.col('M') / pl.col('R')).alias('spend_rate_per_day'),
        ])

        df = df.with_columns([
            # Value-frequency relationship
            (pl.col('M') * pl.col('F')).alias('total_value_frequency').cast(pl.Float32),
            
            # Recency-frequency relationship
            ((pl.col('R') / pl.col('F'))).alias('avg_days_between_purchases').cast(pl.Float32),
        ])

        # Country stats and users relatives to country
        contry_stats = df.group_by('user_country').agg([
                pl.col('M').mean().alias('country_avg_monetary'),
                pl.col('F').mean().alias('country_avg_frequency').cast(pl.Float32),
                pl.col('R').mean().alias('country_avg_recency').cast(pl.Float32)
            ])
        df = df.join(contry_stats, on='user_country').with_columns([
                (pl.col('M') / pl.col('country_avg_monetary')).alias('relative_monetary_value'),
                (pl.col('F') / pl.col('country_avg_frequency')).alias('relative_frequency').cast(pl.Float32),
                (pl.col('R') / pl.col('country_avg_recency')).alias('relative_recency').cast(pl.Float32),
            ])

        df = df.with_columns([
            # High expend customer flag
            (pl.col('M') > pl.col('M').mean().over('user_country')).cast(pl.Int8).alias('is_high_value_incountry'),
            # Frequent buyer flag
            (pl.col('F') > pl.col('F').mean().over('user_country')).cast(pl.Int8).alias('is_frequent_buyer_incountry'),
            # Recent customer flag
            (pl.col('R') > pl.col('R').mean().over('user_country')).cast(pl.Int8).alias('is_recent_customer_incountry'),
            # High expend customer flag
            (pl.col('M') > pl.col('M').mean()).cast(pl.Int8).alias('is_high_value'),
            # Frequent buyer flag
            (pl.col('F') > pl.col('F').mean()).cast(pl.Int8).alias('is_frequent_buyer'),
            # Recent customer flag
            (pl.col('R') > pl.col('R').mean()).cast(pl.Int8).alias('is_recent_customer'),
        ])
        
        # Replace NaN and Inf values with 0
        for col in ['purchase_rate', 'spend_rate_per_day', 'avg_days_between_purchases']:
                if col in df.columns:
                    df = df.with_columns(pl.when(pl.col(col).is_nan() | (pl.col(col)
                                                                                    .is_infinite()))
                                                                                    .then(0)
                                                                                    .otherwise(pl.col(col))
                                                                                    .alias(col),)

        return df

    @staticmethod
    def create_rfm_segments_and_ranks(users_df):
        # Create quintiles (5 segments) for each metric
        return users_df.with_columns([
                    # Recency quintile (1 is most recent, 5 is least recent)
                    pl.col('R')
                        .rank(descending=True)  # High R is better
                        .over('user_country')
                        .map_batches(lambda x: pd.qcut(x, q=5, labels=False) + 1)
                        .alias('r_segment').cast(pl.UInt8),
                        
                    # Frequency quintile (1 is highest frequency, 5 is lowest)
                    pl.col('F')
                        .rank(descending=True)   # higher F is better
                        .over('user_country')
                        .map_batches(lambda x: pd.qcut(x, q=5, labels=False) + 1)
                        .alias('f_segment').cast(pl.UInt8),
                        
                    # Monetary quintile (1 is highest value, 5 is lowest)
                    pl.col('M')
                        .rank(descending=True)   # higher M is better
                        .over('user_country')
                        .map_batches(lambda x: pd.qcut(x, q=5, labels=False) + 1)
                        .alias('m_segment').cast(pl.UInt8),

                    # Individual percentile ranks
                    pl.col('R').rank(descending=True)
                        .over('user_country').alias('r_rank_in_country').cast(pl.Int32),
                    pl.col('F').rank(descending=True)
                        .over('user_country').alias('f_rank_in_country').cast(pl.Int32),
                    pl.col('M').rank(descending=True)
                        .over('user_country').alias('m_rank_in_country').cast(pl.Int32),
                ])
    
    @timer
    def run(self):
        df = self.load_and_process_users()
        df = self.create_initial_extra_features(df)
        df = self.create_rfm_segments_and_ranks(df)
        df.write_parquet(self.engineered_path)
        return df