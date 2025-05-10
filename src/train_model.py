# train_model.py

from feature_engineering import feature_engineering
from data_preprocessing import load_data_from_db
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_and_evaluate_model(db_path, table_name):
    """Load data, preprocess, train model, and evaluate performance."""
    
    # Load data from the database
    df = load_data_from_db(db_path, table_name)
    
    # Apply feature engineering
    df = feature_engineering(df)
    print(df.head())
    
    # Build machine learning models and evaluate with AUC metric
    X = df.drop(columns='Subscription Status')
    y = df['Subscription Status']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Instantiate the model object
    rf_model = RandomForestClassifier(random_state=5, n_estimators=300, max_depth=20, min_samples_leaf=4)

    # Fit the model on the training data
    rf_model.fit(X_train, y_train)

    # Predict the target on the test dataset
    rf_preds = rf_model.predict(X_test)
    rf_preds_proba = rf_model.predict_proba(X_test)[:, 1]  # for ROC AUC

    # Calculate the AUC-ROC score on the test set
    auc = roc_auc_score(y_test, rf_preds_proba)
    print('\nAUC-ROC score on test set:', auc)

    # Calculate accuracy, precision, recall, f1 score
    accuracy = accuracy_score(y_test, rf_preds)
    precision = precision_score(y_test, rf_preds)
    recall = recall_score(y_test, rf_preds)
    f1 = f1_score(y_test, rf_preds)

    # Display the performance metrics
    print('\nModel Evaluation Metrics:')
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)

    # Display confusion matrix
    conf_matrix = confusion_matrix(y_test, rf_preds)
    print('\nConfusion Matrix:')
    print(conf_matrix)

if __name__ == '__main__':
    db_path = "/Users/jayelin/Downloads/docker_image/data/bmarket.db"
    table_name = "bank_marketing"
    
    # Train and evaluate the model
    train_and_evaluate_model(db_path, table_name)
