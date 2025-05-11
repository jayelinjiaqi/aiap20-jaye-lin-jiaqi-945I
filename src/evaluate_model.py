from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

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
