import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from joblib import dump, load

categorical = ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"]
numerical = ["age", "avg_glucose_level", "bmi"]

# Load the dataset
def load_data():
    df = pd.read_csv('strokedata.csv')
    df = df.drop('id', axis=1)
    y = df['stroke']
    X = df.drop('stroke', axis=1)
    return X, y, categorical, numerical

def evaluate_model(X, y, models):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    scores = cross_val_score(models, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    return scores

# load data
X, y, categorical_features, numerical_features = load_data()

# Preprocessing pipeline
transformer = ColumnTransformer(transformers=[
    ('num', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('power', PowerTransformer(method='yeo-johnson', standardize=True))
    ]), numerical),
    ('cat', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ]), categorical)
])

# Using imblearn pipeline for SMOTE integration
pipeline = ImbalancedPipeline(steps=[
    ('transformer', transformer),
    ('smote', SMOTE(sampling_strategy='auto')),
    ('model', LinearDiscriminantAnalysis())
])

scores = evaluate_model(X, y, pipeline)
pipeline.fit(X, y)

dump(pipeline, 'stroke_prediction_model.joblib')