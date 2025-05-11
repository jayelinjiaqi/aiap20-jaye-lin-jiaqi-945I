import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from load_data_from_db import load_data_from_db

def contact_method_cleaning(mobile_str):
    """
    Cleans the 'Contact Method' feature by replacing 'Telephone' and 'Cell' with 
    lowercase versions.
    
    Args:
    - mobile_str: The original contact method string ('Telephone', 'Cell', etc.)
    
    Returns:
    - Cleaned contact method string.
    """
    if mobile_str == 'Telephone':
        return 'telephone'
    elif mobile_str == 'Cell':
        return 'cellular'
    else:
        return mobile_str

def yes_no_binary(yes_no_str):
    """
    Converts 'yes' to 1, 'no' to 0, and 'unknown' or NULL to -1.
    
    Args:
    - yes_no_str: The original string ('yes', 'no', 'unknown')
    
    Returns:
    - Corresponding numeric value (0, 1, or -1)
    """
    if yes_no_str == 'no':
        return 0
    elif yes_no_str == 'yes':
        return 1
    else:
        return -1

def education_mapping(education_str):
    """
    Maps education levels to numeric values.
    
    Args:
    - education_str: The education level string (e.g., 'high.school', 'university.degree')
    
    Returns:
    - Mapped numeric value corresponding to the education level.
    """
    mapping = {
        'illiterate': 1,
        'basic.4y': 2,
        'basic.6y': 3,
        'basic.9y': 4,
        'high.school': 5,
        'professional.course': 6,
        'university.degree': 7,
        'unknown': -1
    }
    return mapping.get(education_str, -1)

def previous_contact_days_mapping(contact_days):
    """
    Maps the 'Previous Contact Days' feature by replacing 999 with -1.
    
    Args:
    - contact_days: The number of previous contact days.
    
    Returns:
    - Updated number of contact days (or -1 if it was 999).
    """
    return -1 if contact_days == 999 else contact_days

def feature_engineering(db_path, table_name):
    """
    Loads data from DB and applies feature engineering.
    
    Args:
    - db_path (str): Path to the SQLite DB
    - table_name (str): Table name
    
    Returns:
    - df: Transformed DataFrame
    """
    df = load_data_from_db(db_path, table_name)

    # Extract numerical age from feature - 'Age'
    df['Age'] = df['Age'].str.extract(r'(\d+)').astype(int)

    # Remove outliers of 'Age' using IQR
    lower_bound = df['Age'].quantile(0.01)
    upper_bound = df['Age'].quantile(0.99)

    df = df[(df['Age'] > lower_bound) & (df['Age'] < upper_bound)]

    # Feature: 'Contact Method' - Clean the values
    df['Contact Method'] = df['Contact Method'].apply(contact_method_cleaning)

    # Apply yes_no_binary for categorical features
    df['Housing Loan'] = df['Housing Loan'].apply(yes_no_binary)
    df['Personal Loan'] = df['Personal Loan'].apply(yes_no_binary)
    df['Credit Default'] = df['Credit Default'].apply(yes_no_binary)
    df['Subscription Status'] = df['Subscription Status'].apply(yes_no_binary)

    # One-hot encoding for categorical features
    df = pd.get_dummies(df, columns=['Occupation'], prefix='occupation')
    df = pd.get_dummies(df, columns=['Marital Status'], prefix='marital_status')
    df = pd.get_dummies(df, columns=['Contact Method'], prefix='contact_method')
    
    # Apply education_mapping for 'Education Level'
    df['Education Level'] = df['Education Level'].apply(education_mapping)

    # Apply previous_contact_days_mapping for 'Previous Contact Days'
    df['Previous Contact Days'] = df['Previous Contact Days'].apply(previous_contact_days_mapping)

    # Apply normalization to 'Age' using MinMaxScaler
    #scaler = MinMaxScaler()
    #df['Age_Normalized'] = scaler.fit_transform(df[['Age']])
    
    # Drop columns with perfect correlation
    df = df.drop(['marital_status_single', 'contact_method_telephone'], axis=1)
    
    return df