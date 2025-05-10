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

    Feature engineering (data cleaning, transformation and encoding/mapping for categorical variables).
    
---

## 🔧 Feature Processing Summary

### 🧹 Data Cleaning

| Feature                   | Data Cleaning Description                                                           |
|---------------------------|-------------------------------------------------------------------------------------|
| `Housing Loan`            | Replaced `'none'` or null to `'cellular'`                                           |
| `Personal Loan`           | Replaced `'none'` or null to `'cellular'`                                           |
| `Contact Method`          | Standardized `'Telephone'` to `'telephone'`, `'Cell'` to `'cellular'`               |

### 🔄 Data Transformation

| Feature                   | Transformation Description                                                          |
|---------------------------|-------------------------------------------------------------------------------------|
| `Age`                     | Extracted numerical age from string using regex and converted to integer            

### 🔢 Feature Encoding Mapping

#### ✅ Binary Encodings

| Feature               | Original Value | Encoded Value |
|-----------------------|----------------|---------------|
| Housing Loan          | `yes`          | 1             |
|                       | `no`           | 0             |
|                       | others/null    | -1            |
| Personal Loan         | `yes`          | 1             |
|                       | `no`           | 0             |
|                       | others/null    | -1            |
| Subscription Status   | `yes`          | 1             |
|                       | `no`           | 0             |
|                       | others/null    | -1            |
| Credit Default        | `yes`          | 1             |
|                       | `no`           | 0             |
|                       | others/null    | -1            |

#### ☎️ Contact Method Mapping

| Original Value | Normalized Value | Encoded Value |
|----------------|------------------|---------------|
| `Telephone`    | `telephone`      | 1             |
| `Cell`         | `cellular`       | 2             |

#### 🧑 Occupation Mapping

| Original Value      | Encoded Value |
|----------------------|---------------|
| `technician`         | 1             |
| `blue-collar`        | 2             |
| `admin.`             | 3             |
| `housemaid`          | 4             |
| `retired`            | 5             |
| `services`           | 6             |
| `entrepreneur`       | 7             |
| `unemployed`         | 8             |
| `management`         | 9             |
| `self-employed`      | 10            |
| `student`            | 11            |
| `unknown`            | -1            |

#### 💍 Marital Status Mapping

| Original Value | Encoded Value |
|----------------|---------------|
| `married`      | 1             |
| `divorced`     | 2             |
| `single`       | 3             |
| `unknown`      | -1            |

#### 🎓 Education Level Mapping

| Original Value        | Encoded Value |
|------------------------|---------------|
| `illiterate`           | 1             |
| `basic.4y`             | 2             |
| `basic.6y`             | 3             |
| `basic.9y`             | 4             |
| `high.school`          | 5             |
| `professional.course`  | 6             |
| `university.degree`    | 7             |
| `unknown`              | -1            |

#### 📅 Previous Contact Days

| Original Value | Encoded Value |
|----------------|---------------|
| 999            | -1            |
| All others     | Original value (unchanged) |

---

### 🤖 Model Choices

---

### 📊 Model Evaluation

---

### 🚀 Deployment Considerations



