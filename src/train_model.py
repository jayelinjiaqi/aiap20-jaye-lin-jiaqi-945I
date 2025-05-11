from feature_engineering import feature_engineering
from load_data_from_db import load_data_from_db
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import argparse

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n--- Evaluation: {model.__class__.__name__} ---")
    if preds_proba is not None:
        auc = roc_auc_score(y_test, preds_proba)
        print('AUC-ROC:', auc)

    print('Accuracy:', accuracy_score(y_test, preds))
    print('Precision:', precision_score(y_test, preds))
    print('Recall:', recall_score(y_test, preds))
    print('F1 Score:', f1_score(y_test, preds))
    print('Confusion Matrix:\n', confusion_matrix(y_test, preds))

def train_models(db_path, table_name):
    # Load and preprocess data
    df = load_data_from_db(db_path, table_name)
    df = df = feature_engineering(db_path, table_name)

    X = df.drop(columns='Subscription Status')
    y = df['Subscription Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        'RandomForest': RandomForestClassifier(random_state=42, n_estimators=300, max_depth=30, max_features='sqrt', min_samples_leaf=1),
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=700, C=10, solver='liblinear', penalty='l1'),
        'KNN': KNeighborsClassifier(n_neighbors=20, weights='uniform', metric='manhattan')
    }

    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        trained_models[name] = model

    return trained_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', required=True, help='Path to the SQLite database file')
    parser.add_argument('--table_name', required=True, help='Name of the table in the database')

    args = parser.parse_args()

    train_models(args.db_path, args.table_name)