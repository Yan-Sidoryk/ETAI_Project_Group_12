"""
The learned preprocessing: imputation, scaling and encoding. It is the first step of the model
Pipeline, so it is fitted on training rows only (again in every validation fold) and saved
together with the model.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Builds the unfitted preprocessor described by `preprocessing_config`.

      - categorical columns (listed in the config): a missing value becomes its own level
        "missing", then one-hot encoding. Levels seen in fewer than `min_frequency` training
        rows share one "infrequent" column, and a level never seen in training is encoded as
        all zeros instead of raising an error.
      - every other column is numeric: median imputation, with a 0/1 "was missing" column for
        each feature that has gaps in the training rows, then the configured scaler.
    """
    scaler = _SCALERS[preprocessing_config["scaler"]]
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", scaler if scaler == "passthrough" else scaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                 min_frequency=preprocessing_config.get("min_frequency"))),
    ])
    return ColumnTransformer(
        [("categorical", categorical, preprocessing_config["categorical_features"])],
        remainder=numeric,
    )
