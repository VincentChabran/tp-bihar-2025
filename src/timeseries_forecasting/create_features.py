# def create_features(df, base_col="temperature_2m", lags=[1, 2, 3], ma_windows=[3, 6]):
#     df = df.copy()

#     # Lags
#     for lag in lags:
#         df[f"{base_col}_lag_{lag}"] = df[base_col].shift(lag)

#     # Moyennes mobiles
#     for w in ma_windows:
#         df[f"{base_col}_ma_{w}"] = df[base_col].rolling(window=w).mean()

#     # Features temporelles
#     df["hour"] = df.index.hour
#     df["month"] = df.index.month
#     df["dayofweek"] = df.index.dayofweek

#     # Nettoyage
#     return df.dropna()

def create_features(df, base_col="temperature_2m", lags=[1, 2, 3], ma_windows=[3, 6]):
    df = df.copy()

    # Lags : OK
    for lag in lags:
        df[f"{base_col}_lag_{lag}"] = df[base_col].shift(lag)

    # Moyennes mobiles : on décale pour ne pas inclure y[t]
    for w in ma_windows:
        df[f"{base_col}_ma_{w}"] = df[base_col].shift(1).rolling(window=w).mean()

    # Features temporelles
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["dayofweek"] = df.index.dayofweek

    return df.dropna()
