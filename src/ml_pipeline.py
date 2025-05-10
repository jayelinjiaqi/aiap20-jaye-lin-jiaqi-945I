import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import statistics
from sklearn.metrics import roc_auc_score
#from sklearn.preprocessing import StandardScaler

# Connect to database
#conn = sqlite3.connect("/home/runner/work/aiap20-jaye-lin-jiaqi-945I/aiap20-jaye-lin-jiaqi-945I/data/bmarket.db")
conn = sqlite3.connect("/Users/jayelin/Downloads/docker_image/data/bmarket.db")
# Fetch all rows from the 'lung_cancer' table
cursor = conn.cursor()
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
cursor.execute("SELECT * FROM bank_marketing")
rows = cursor.fetchall()

# Create DataFrame from the results
columns = [description[0] for description in cursor.description]  # Get column names
df = pd.DataFrame(rows, columns=columns)

# Data cleaning
def yes_no_binary(yes_no_str):
    
    if yes_no_str == 'no':
        return 0
    elif yes_no_str == 'yes':
        return 1
    else:
        return -1
    
def mobile_binary(mobile_str):
    
    if mobile_str == 'Telephone':
        return 'telephone'
    elif mobile_str == 'Cell':
        return 'cellular'
    else:
        return mobile_str

df['Housing Loan'] = df['Housing Loan'].apply(yes_no_binary)
df['Personal Loan'] = df['Personal Loan'].apply(yes_no_binary)
df['Subscription Status'] = df['Subscription Status'].apply(yes_no_binary)
df['Credit Default'] = df['Credit Default'].apply(yes_no_binary)
df['Contact Method'] = df['Contact Method'].apply(mobile_binary)
df.head()

# Feature engineering
def contact_binary(str):
    if str == 'telephone':
        return 1
    elif str == 'cellular':
        return 2

def occupation_binary(str):
    if str == 'technician':
        return 1
    elif str == 'blue-collar':
        return 2
    elif str == 'admin.':
        return 3
    elif str == 'housemaid':
        return 4
    elif str == 'retired':
        return 5
    elif str == 'services':
        return 6
    elif str == 'entrepreneur':
        return 7
    elif str == 'unemployed':
        return 8
    elif str == 'management':
        return 9
    elif str == 'self-employed':
        return 10
    elif str == 'student':
        return 11
    elif str == 'unknown':
        return -1
    
def marital_binary(str):
    if str == 'married':
        return 1
    elif str == 'divorced':
        return 2
    elif str == 'single':
        return 3
    elif str == 'unknown':
        return -1

def education_binary(str):
    if str == 'illiterate':
        return 1
    elif str == 'basic.4y':
        return 2
    elif str == 'basic.6y':
        return 3
    elif str == 'basic.9y':
        return 4
    elif str == 'high.school':
        return 5
    elif str == 'professional.course':
        return 6
    elif str == 'university.degree':
        return 7
    elif str == 'unknown':
        return -1
    
def previous_contact_days_binary(str):
    if str == 999:
        return -1
    else:
        return str

df['Age'] = df['Age'].str.extract(r'(\d+)').astype(int)
df['Contact Method'] = df['Contact Method'].apply(contact_binary)
df['Occupation'] = df['Occupation'].apply(occupation_binary)
df['Marital Status'] = df['Marital Status'].apply(marital_binary)
df['Education Level'] = df['Education Level'].apply(education_binary)
df['Previous Contact Days'] = df['Previous Contact Days'].apply(previous_contact_days_binary)
df.head(10)

# Build machine learning models and evaluate with AUC metric

X = df.drop(columns='Subscription Status')
y = df['Subscription Status']

# Standardize the features
#sc = StandardScaler()
#X = sc.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Instantiate the model object
rf_model = RandomForestClassifier(random_state=5, n_estimators=300, max_depth=20, min_samples_leaf=4)
#knn_model = KNeighborsClassifier(n_neighbors=20, weights='distance', metric='manhattan')
#xgb_model = XGBClassifier(learning_rate=0.012, n_estimators=1001, max_depth=7, min_child_weight=1, gamma=0, subsample=0.79, colsample_bytree=0.8, objective='binary:logistic', nthread=4, scale_pos_weight=1, reg_alpha=0.1, reg_lambda=1, random_state=5, eval_metric="logloss")

#xgb_model.fit(X_train,y_train)
rf_model.fit(X_train, y_train)
#knn_model.fit(X_train, y_train)

# Predict the target on the test dataset
#xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
rf_preds = rf_model.predict_proba(X_test)[:, 1]
#knn_preds = knn_model.predict_proba(X_test)[:, 1]

# Calculate the weighted average of the predictions
#final_preds = (0.9 * xgb_preds) + (0.05 * rf_preds) + (0.05 * knn_preds)

# Calculate the AUC-ROC score on the test set
#auc = roc_auc_score(y_test, final_preds)
auc = roc_auc_score(y_test, rf_preds)
print('\nAUC-ROC score on test set:', auc)

print(rf_preds)
