from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge


# ------------------------------------------------------------
# 1. Download only the important data
# ------------------------------------------------------------

planets = NasaExoplanetArchive.query_criteria(
    table="ps",
    select="""
        pl_name,
        hostname,
        discoverymethod,

        pl_orbper,
        pl_orbsmax,
        pl_bmasse,
        st_mass,

        pl_orbeccen,
        pl_orbeccenerr1,
        pl_orbeccenerr2
    """
)

df = planets.to_pandas()


# ------------------------------------------------------------
# 2. Clean eccentricity values
# ------------------------------------------------------------

df = df[
    df["pl_orbeccen"].isna()
    | ((df["pl_orbeccen"] >= 0) & (df["pl_orbeccen"] < 1))
].copy()


# ------------------------------------------------------------
# 3. Select reliable eccentricities for training
# ------------------------------------------------------------

has_measured_eccentricity = (
    df["pl_orbeccen"].notna()
    & (
        df["pl_orbeccenerr1"].notna()
        | df["pl_orbeccenerr2"].notna()
    )
)

not_suspicious_zero = ~(
    (df["pl_orbeccen"] == 0.0)
    & df["pl_orbeccenerr1"].isna()
    & df["pl_orbeccenerr2"].isna()
)

training_mask = has_measured_eccentricity & not_suspicious_zero

train_df = df[training_mask].copy()


# ------------------------------------------------------------
# 4. Add physically useful log variables
# ------------------------------------------------------------

def add_log_features(data):
    data = data.copy()

    data["log_period"] = np.where(
        data["pl_orbper"] > 0,
        np.log10(data["pl_orbper"]),
        np.nan
    )

    data["log_semi_major_axis"] = np.where(
        data["pl_orbsmax"] > 0,
        np.log10(data["pl_orbsmax"]),
        np.nan
    )

    data["log_planet_mass"] = np.where(
        data["pl_bmasse"] > 0,
        np.log10(data["pl_bmasse"]),
        np.nan
    )

    data["log_stellar_mass"] = np.where(
        data["st_mass"] > 0,
        np.log10(data["st_mass"]),
        np.nan
    )

    return data


df = add_log_features(df)
train_df = add_log_features(train_df)


# ------------------------------------------------------------
# 5. Transform eccentricity
# ------------------------------------------------------------

def logit_e(e):
    """
    Converts eccentricity into logit space.

    This is used because eccentricity must remain between 0 and 1.
    """
    e = np.asarray(e, dtype=float)
    e = np.clip(e, 1e-4, 1 - 1e-4)
    return np.log(e / (1 - e))


def inverse_logit(y):
    """
    Converts model output back into eccentricity.
    """
    return 1 / (1 + np.exp(-y))


train_df["target"] = logit_e(train_df["pl_orbeccen"])


# ------------------------------------------------------------
# 6. Build the model using only important variables
# ------------------------------------------------------------

numeric_features = [
    "log_period",
    "log_semi_major_axis",
    "log_planet_mass",
    "log_stellar_mass"
]

categorical_features = [
    "discoverymethod"
]

features = numeric_features + categorical_features

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0))
    ]
)


# ------------------------------------------------------------
# 7. Train model and predict expected eccentricities
# ------------------------------------------------------------

X_train = train_df[features]
y_train = train_df["target"]

model.fit(X_train, y_train)

df["predicted_logit_e"] = model.predict(df[features])
df["expected_eccentricity"] = inverse_logit(df["predicted_logit_e"])


# ------------------------------------------------------------
# 8. Decide which eccentricity value to use
# ------------------------------------------------------------

suspicious_or_missing = (
    df["pl_orbeccen"].isna()
    | (
        (df["pl_orbeccen"] == 0.0)
        & df["pl_orbeccenerr1"].isna()
        & df["pl_orbeccenerr2"].isna()
    )
)

df["eccentricity_for_analysis"] = df["pl_orbeccen"]
df.loc[suspicious_or_missing, "eccentricity_for_analysis"] = df.loc[
    suspicious_or_missing,
    "expected_eccentricity"
]


# ------------------------------------------------------------
# 9. Save useful output
# ------------------------------------------------------------

output = df[
    [
        "pl_name",
        "hostname",
        "discoverymethod",

        "pl_orbper",
        "pl_orbsmax",
        "pl_bmasse",
        "st_mass",

        "pl_orbeccen",
        "expected_eccentricity",
        "eccentricity_for_analysis"
    ]
]

output.to_csv("exoplanet_expected_eccentricities.csv", index=False)

print(output.head(20))
print("Saved file: exoplanet_expected_eccentricities.csv")