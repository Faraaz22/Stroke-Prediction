{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0d513cc8-0f29-4bb9-9c45-7d8e9304a0a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import PowerTransformer, OneHotEncoder\n",
    "from imblearn.pipeline import Pipeline as IMBPipeline\n",
    "from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score\n",
    "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "\n",
    "numerical = [\"age\", \"avg_glucose_level\", \"bmi\"]\n",
    "categorical = [\"gender\", \"hypertension\", \"heart_disease\", \"ever_married\", \"work_type\", \"Residence_type\", \"smoking_status\"]\n",
    "\n",
    "# Define the column transformer\n",
    "transformer = ColumnTransformer(transformers=[\n",
    "    ('num', Pipeline(steps=[\n",
    "        ('imputer', SimpleImputer(strategy='mean')),\n",
    "        ('power', PowerTransformer(method='yeo-johnson', standardize=True))\n",
    "    ]), numerical),\n",
    "    ('cat', Pipeline(steps=[\n",
    "        ('imputer', SimpleImputer(strategy='most_frequent')),\n",
    "        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))\n",
    "    ]), categorical)\n",
    "])\n",
    "\n",
    "\n",
    "# Define the function to evaluate models (focusing on LDA)\n",
    "def evaluate_lda(X, y):\n",
    "    model = LinearDiscriminantAnalysis()\n",
    "    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)\n",
    "    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)\n",
    "    print(f'LDA: {np.mean(scores)} ({np.std(scores)})')\n",
    "\n",
    "\n",
    "# Assuming 'df' is your DataFrame\n",
    "y = df['stroke']\n",
    "X = df.drop('stroke', axis=1)\n",
    "\n",
    "# Apply the transformer to the data\n",
    "X_transformed = transformer.fit_transform(X)\n",
    "\n",
    "# Evaluate LDA performance\n",
    "score = evaluate_lda(X_transformed, y)\n",
    "model_filename = 'lda_model.joblib'\n",
    "joblib.dump(model, model_filename)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
