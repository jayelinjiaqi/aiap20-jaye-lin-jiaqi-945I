from feature_engineering import feature_engineering
from load_data_from_db import load_data_from_db
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

def train_models(db_path, table_name):
    # Load and preprocess data
    df = load_data_from_db(db_path, table_name)
    df = feature_engineering(df)

    X = df.drop(columns='Subscription Status')
    y = df['Subscription Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    models = {
        'RandomForest': RandomForestClassifier(random_state=5, n_estimators=300, max_depth=20, min_samples_leaf=4),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = (model, X_test, y_test)

        # save model to disk
        with open(f'{name}_model.pkl', 'wb') as f:
            pickle.dump(model, f)

    return trained_models
