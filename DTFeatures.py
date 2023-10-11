import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer, StandardScaler
import MLBenchmarks.regression_datasets_loaders as rdl
import MLBenchmarks.classification_datasets_loaders as cdl
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from xgboost import XGBRegressor,XGBClassifier
import numpy as np
from copy import copy,deepcopy
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=5, random_state=42)

dataset = cdl.load_student_dropout()
X = dataset['data']
y = dataset['target']

X, X_test, y, y_test = train_test_split(X, y, test_size=0.33)


# Function to generate a decision tree feature on a random subset of data
def create_decision_tree_feature(X, model):
    #random_indices = np.random.choice(len(X), size=int(0.8 * len(X)), replace=True)
    #X_subset = X[random_indices]
    #y_subset = y[random_indices]

    #tree = DecisionTreeClassifier()
    #tree.fit(X_subset, y_subset)
    return model.predict(X).reshape(-1, 1)


n_features = 5
feature_models = [None for _ in range(n_features)]
selected_feature_models = [None for _ in range(n_features)]

it = 0
maxit = 1000
quality = np.zeros(n_features)
quality_selected = np.zeros(n_features)

for it in tqdm(range(maxit)): 


    # generate a set of feature models 
    feature_models = [None for _ in range(n_features)]
    for i in range(n_features):
        random_indices = np.random.choice(len(X), size=int(0.8 * len(X)), replace=True)
        # allow random cols in the future
        X_subset = X[random_indices]
        y_subset = y[random_indices]
        tree = DecisionTreeClassifier()
        tree.fit(X_subset,y_subset)
        feature_models[i] = deepcopy(tree)


    # Create transformers for each decision tree feature
    feature_transformers = [FunctionTransformer(create_decision_tree_feature, kw_args={"model": feature_models[i]}) for i in range(n_features)]

    # Combine the features using FeatureUnion
    union = FeatureUnion(transformer_list=[("model_feature_{}".format(i), feature_transformer) 
                                       for i, feature_transformer in enumerate(feature_transformers)])

    # Create a scikit-learn pipeline
    pipeline = Pipeline([
        ('model_features', union)
    ])

    # Fit the pipeline and transform the data to get the new features
    X_new = pipeline.fit_transform(X)

    aggregator = LogisticRegression()

    aggregator.fit(X_new,y)

    quality = np.abs(aggregator.coef_[0])

    for i in range(n_features):
        if quality[i]>quality_selected[i]:
            selected_feature_models[i] = deepcopy(feature_models[i])
            quality_selected[i] = quality[i]

    if (it == 0) or (it == (maxit - 1)):
        print(quality_selected)




# Base Score
base_model = RandomForestClassifier()
base_model.fit(X,y)
print(f"score original: {f1_score(y_test, base_model.predict(X_test),average='micro')}")

# Create transformers for each decision tree feature
feature_transformers = [FunctionTransformer(create_decision_tree_feature, kw_args={"model": selected_feature_models[i]}) for i in range(n_features)]
# Combine the features using FeatureUnion
union = FeatureUnion(transformer_list=[("model_feature_{}".format(i), feature_transformer) 
                                       for i, feature_transformer in enumerate(feature_transformers)])

# Create a scikit-learn pipeline
pipeline = Pipeline([
        ('model_features', union)
    ])

# Fit the pipeline and transform the data to get the new features
X_new = pipeline.fit_transform(X)
base_model.fit(X_new,y)
print(f"score new: {f1_score(y_test, base_model.predict(pipeline.fit_transform(X_test)),average='micro')}")


X_new = pipeline.fit_transform(X)
base_model.fit(np.hstack([X_new,X]),y)
nox = np.hstack([pipeline.fit_transform(X_test),X_test])
print(f"score new+original: {f1_score(y_test, base_model.predict(nox), average='micro')}")

