from train_model import train_models
from evaluate_model import evaluate_model

db_path = '/home/runner/work/aiap20-jaye-lin-jiaqi-945I/aiap20-jaye-lin-jiaqi-945I/data/bmarket.db'
table_name = 'bank_marketing'

trained_models = train_models(db_path, table_name)

for name, (model, X_test, y_test) in trained_models.items():
    evaluate_model(model, X_test, y_test)
