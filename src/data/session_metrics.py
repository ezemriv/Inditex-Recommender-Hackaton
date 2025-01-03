import pandas as pd


def get_session_metrics(df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """
    Given a pandas DataFrame in the format of the train dataset and a user_id, return the following metrics for every session_id of the user:
        - user_id (int) : the given user id.
        - session_id (int) : the session id.
        - total_session_time (float) : The time passed between the first and last interactions, in seconds. Rounded to the 2nd decimal.
        - cart_addition_ratio (float) : Percentage of the added products out of the total products interacted with. Rounded ot the 2nd decimal.

    If there's no data for the given user, return an empty Dataframe preserving the expected columns.
    The column order and types must be scrictly followed.

    Parameters
    ----------
    df : pandas DataFrame
       DataFrame  of the data to be used for the agent.
    user_id : int
        Id of the client.

    Returns
    -------
    Pandas Dataframe with some metrics for all the sessions of the given user.
    """
    
    # Filter data for the given user
    user_df = df[df["user_id"] == user_id]
    if user_df.empty:
        return pd.DataFrame(
            columns=["user_id", "session_id", "total_session_time", "cart_addition_ratio"]
        )

    # Ensure timestamp is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(user_df["timestamp_local"]):
        user_df["timestamp_local"] = pd.to_datetime(user_df["timestamp_local"])

    session_ids = user_df["session_id"].unique()
    metrics = []

    for session_id in session_ids:
        session_df = user_df[user_df["session_id"] == session_id]
        total_session_time = (session_df["timestamp_local"].max() - session_df["timestamp_local"].min()).total_seconds()
        cart_addition_ratio = 100*(session_df["add_to_cart"].sum() / len(session_df)) if len(session_df) > 0 else 0.0
        metrics.append(
            {
                "user_id": int(user_id),
                "session_id": int(session_id),
                "total_session_time": round(total_session_time, 2),
                "cart_addition_ratio": round(cart_addition_ratio, 2),
            }
        )

    return pd.DataFrame(metrics).sort_values(by=["user_id", "session_id"]).reset_index(drop=True)
