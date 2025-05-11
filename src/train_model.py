from feature_engineering import feature_engineering
from load_data_from_db import load_data_from_db
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def train_models(db_path, table_name):
    # Load and preprocess data
    df = load_data_from_db(db_path, table_name)
    df = feature_engineering(df)

    X = df.drop(columns='Subscription Status')
    y = df['Subscription Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        'RandomForest': RandomForestClassifier(random_state=42, n_estimators=300, max_depth=30, max_features='sqrt', min_samples_leaf=1),
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=700, C=10, solver='liblinear', penalty='l1'),
        'KNN': KNeighborsClassifier(n_neighbors=20, weights='uniform', metric='manhattan')
    }

    trained_models = {}

    return trained_models



