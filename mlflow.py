import mlflow
import mlflow.sklearn

import pandas as pd
import requests

from imblearn.pipeline import Pipeline

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.feature_selection import (
    SelectFromModel
)

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

# ====================================
# LOAD DATASET
# ====================================

url = "https://data.mendeley.com/public-files/datasets/7m58kxs742/files/4d57de6a-a140-4607-bf38-146899f1a723/file_downloaded"

file_name = "Preprocessed_Balanced_dataset.csv"

response = requests.get(url)

with open(file_name, "wb") as f:
    f.write(response.content)

df = pd.read_csv(file_name)

# ====================================
# PREPROCESSING
# ====================================

le = LabelEncoder()

df['Attack_sub_category'] = le.fit_transform(
    df['Attack_sub_category']
)

X = df.drop(
    'Attack_sub_category',
    axis=1
)

y = df['Attack_sub_category']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ====================================
# PIPELINE TERBAIK
# ====================================

pipeline = Pipeline([
    (
        'scaler',
        StandardScaler()
    ),

    (
        'feature_selection',

        SelectFromModel(
            RandomForestClassifier(
                random_state=42
            ),
            max_features=10
        )
    ),

    (
        'model',

        RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    )
])

# ====================================
# MLFLOW TRACKING
# ====================================

with mlflow.start_run():

    # Training model
    pipeline.fit(
        X_train,
        y_train
    )

    # Prediksi
    y_pred = pipeline.predict(X_test)

    # Evaluasi
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average='macro'
    )

    # Logging parameter
    mlflow.log_params({

        "feature_selection":
        "SelectFromModel",

        "max_features":
        10,

        "n_estimators":
        100,

        "max_depth":
        10

    })

    # Logging metric
    mlflow.log_metrics({

        "accuracy":
        accuracy,

        "f1_score":
        f1

    })

    # Logging model
    mlflow.sklearn.log_model(
        pipeline,
        "pipeline_model"
    )

    print("MLflow tracking berhasil!")