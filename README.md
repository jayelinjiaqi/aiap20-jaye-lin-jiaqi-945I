# 📊 AIAP Batch 20 Technical Assessment

## 👤 Personal Details

- **Full Name**: JAYE LIN JIAQI
- **Email Address**: jiaqi.lin@mail.com

---

## 📁 Project Overview & Folder Structure

This repository contains the implementation of a machine learning pipeline designed for AI-Vive-Banking to develop predictive models to enhance resource allocation and customer engagement by accurately identifying clients that are most likely to respond positively to campaigns.

Project root folder: `aiap20-jaye-lin-jiaqi-945I`

<pre> <code> 
  . 
  ├── .github/     # GitHub workflow configuration file 
  ├── src/         # Python scripts for each pipeline step 
  │ ├── data_preprocessing.py 
  │ ├── feature_engineering.py 
  │ ├── train_model.py 
  │ └── evaluate_model.py 
  ├── README.md          # Project documentation 
  ├── eda.ipynb          # Jupyter notebook for exploratory data analysis 
  ├── requirements.txt   # Python dependencies 
  └── run.sh             # Shell script to execute the pipeline 
</code> </pre>

---

## 🛠️ Executing the pipeline

<pre> <code> 
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python src/train_model.py
</code> </pre>
---

## 🔄 Pipeline Flow & Design

---

## 📈 EDA Summary (From eda.ipynb)

    Key Summary:

    - There are 41188 rows and 12 columns
    - The data type for age is object
    - The 12 unique values for 'Occupation' are 'technician', 'blue-collar', 'admin.', 'housemaid', 'retired', 'services', 'entrepreneur', 'unemployed', 'management', 'self-employed', 'student' and 'unknown'
    - The 4 unique values for 'Marital Status' are 'married', 'divorced', 'single' and 'unknown'
    - The 8 unique values for 'Education' are 'illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'professional.course', 'university.degree' and 'unknown'
    - The 4 unqiue values for 'Credit Default' are 'yes', 'no' and 'None'
    - The 4 unqiue values for 'Housing Loan' are 'yes', 'no' and 'None'
    - The 4 unqiue values for 'Personal Loan' are 'yes', 'no' and 'None'
    - The 4 unqiue values for 'Contact Method' are 'cellular', 'Cell', 'telephone' and 'Telephone'
    - The 2 unqiue values for 'Subscription Status' are 'yes' and 'no'

    These findings guided:

    Feature engineering whereby:
    - Int is extracted from the 'Age' column
    - The none/null in 'Housing Loan' was replaced with 'unknown'
    - The none/null in 'Personal Loan' was replaced with 'unknown'
    - The 'Telephone' and 'Cell' in 'Contact Method' was replaced with 'telephone' and 'cellular' respectively

    Transformation: encoding for categorical variables.
    
---
