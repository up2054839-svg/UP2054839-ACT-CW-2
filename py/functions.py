from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


# ----------------------------
# 1) Loading + cleaning 
# ----------------------------
def get_data(file_path: str | Path) -> pd.DataFrame:
   
    df = pd.read_excel(file_path)
    df = df[["console", "genre", "total_sales(mil)"]]
    df = df.dropna()
    return df

# ----------------------------
# 2) Filtering
# ----------------------------
def filter_ps4_xone_pc_and_selected_genres(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:

    # Filtering each Console separately
    idxps4 = (df["console"] == "PS4")
    idxone = (df["console"] == "XOne")
    idxpc  = (df["console"] == "PC")

    # Filtering each Genre separately
    idxact = (df["genre"] == "Action")
    idxspo = (df["genre"] == "Sports")
    idxrp  = (df["genre"] == "Role-Playing")
    idxadv = (df["genre"] == "Adventure")

    # Filtering Console together
    idxt1 = idxps4 | idxone
    idxcon = idxt1 | idxpc

    # Filtering Genre together
    idxt2 = idxact | idxspo
    idxt3 = idxt2 | idxrp
    idxgen = idxt3 | idxadv

    # Combined filter
    idxcon_gen = idxcon & idxgen

    # Filtered dataframe
    F_DataFrame = df[idxcon_gen].copy()

    masks = {
        "idxps4": idxps4, "idxone": idxone, "idxpc": idxpc,
        "idxact": idxact, "idxspo": idxspo, "idxrp": idxrp, "idxadv": idxadv,
        "idxcon": idxcon, "idxgen": idxgen, "idxcon_gen": idxcon_gen,
    }

    return F_DataFrame, masks


# ----------------------------
# 3) Baselines
# ----------------------------
def baseline_means(df_full: pd.DataFrame, masks: Dict[str, pd.Series]) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
   
    idxcon = masks["idxcon"]
    idxgen = masks["idxgen"]
    idxcon_gen = masks["idxcon_gen"]

    mean_by_console = (
        df_full[idxcon]
        .groupby("console")["total_sales(mil)"]
        .mean()
        .sort_values(ascending=False)
    )

    mean_by_genre = (
        df_full[idxgen]
        .groupby("genre")["total_sales(mil)"]
        .mean()
        .sort_values(ascending=False)
    )

    mean_console_genre = (
        df_full[idxcon_gen]
        .groupby(["console", "genre"])["total_sales(mil)"]
        .mean()
        .reset_index()
        .sort_values(by="total_sales(mil)", ascending=False)
    )

    return mean_by_console, mean_by_genre, mean_console_genre


# ----------------------------
# 4) Encoding 
# ----------------------------
def encode_console_genre(F_DataFrame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:

    x = F_DataFrame[["console", "genre"]]
    y = F_DataFrame["total_sales(mil)"]
    x_encoded = pd.get_dummies(x)
    return x_encoded, y


# ----------------------------
# 5) Train/evaluate Random Forest
# ----------------------------
def train_and_evaluate_rf(
    x_encoded: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    n_jobs: int = -1,
) -> Tuple[RandomForestRegressor, pd.DataFrame]:


    x_train, x_test, y_train, y_test = train_test_split(
        x_encoded, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs
    )
    model.fit(x_train, y_train)

    y_prediction = model.predict(x_test)

    r2 = r2_score(y_test, y_prediction)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_prediction)))

    results = pd.DataFrame({
        "metric": ["r2", "rmse"],
        "value": [float(r2), rmse]
    })

    return model, results


# ----------------------------
# 6) Predict top console/genre combos
# ----------------------------
def top_console_genre_predictions(
    model: RandomForestRegressor,
    F_DataFrame: pd.DataFrame,
    x_encoded_columns: pd.Index,
    top_n: int = 12
) -> pd.DataFrame:
   
    console = F_DataFrame["console"].unique()
    genre = F_DataFrame["genre"].unique()

    combos = pd.DataFrame([(c, g) for c in console for g in genre], columns=["console", "genre"])

    combos_encoded = pd.get_dummies(combos)
    combos_encoded = combos_encoded.reindex(columns=x_encoded_columns, fill_value=0)

    combos["predicted_total_sales(mil)"] = model.predict(combos_encoded)

    top_combos = combos.sort_values(by="predicted_total_sales(mil)", ascending=False)
    return top_combos.head(top_n)
