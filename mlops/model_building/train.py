
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
import joblib
import os
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError


Xtrain_path = "hf://datasets/subhash33/bank-customer-churn/Xtrain.csv"
Xtest_path = "hf://datasets/subhash33/bank-customer-churn/Xtest.csv"
ytrain_path = "hf://datasets/subhash33/bank-customer-churn/ytrain.csv"
ytest_path = "hf://datasets/subhash33/bank-customer-churn/ytest.csv"


Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# List of numerical features in the dataset
numeric_features = [
    'CreditScore',       # Customer's credit score
    'Age',               # Customer's age
    'Tenure',            # Number of years the customer has been with the bank
    'Balance',           # Customer’s account balance
    'NumOfProducts',     # Number of products the customer has with the bank
    'HasCrCard',         # Whether the customer has a credit card (binary: 0 or 1)
    'IsActiveMember',    # Whether the customer is an active member (binary: 0 or 1)
    'EstimatedSalary'    # Customer’s estimated salary
]

# List of categorical features in the dataset
categorical_features = [
    'Geography',         # Country where the customer resides
]


# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features),
)

xgb_model = xbg.XBGClassifier(scale_pos_weight=class_wight, random_state=42)

param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 125, 150],    # number of tree to build
    'xgbclassifier__max_depth': [2, 3, 4],    # maximum depth of each tree
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each tree
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each level of a tree
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],    # learning rate
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],    # L2 regularization factor
}

model_pipeline = make_pipeline(preprocessor, xgb_model)
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, njobs=-1)
grid_search.fit(X_train, y_train)

grid_search.best_params_

best_model = grid_search.best_estimator_
best_model

classification_threshold = 0.45

y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

y_pred_test_proba = best_model.predict_proba(ytrain)[:, 1]
y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

# Generate a classification report to evaluate model performance on training set
print(classification_report(ytrain, y_pred_train))

# Generate a classification report to evaluate model performance on test set
print(classification_report(ytest, y_pred_test))

joblib.dump(best_model, 'best_churn_model.joblib')

# Upload to Hugging Face
repo_id = "subhash33/Bank-Churn-Model"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

api.upload_folder(
    folder_path="mlops/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
