# Census Income Project

This project analyzes the Census Income (Adult) dataset to predict whether an individual's income exceeds $50K/yr based on various features.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Citations](#citations)

## Introduction

The Census Income project uses various machine learning models to predict whether a person's income is greater than $50,000/year based on demographic and employment-related features.

## Dataset

The dataset used in this project is the [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult). It contains 14 features including age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, and native country.

The dataset is split into two files:
- `adult.data`
- `adult.test`

## Installation

To get started with this project, clone the repository and install the required dependencies:
```
git clone https://github.com/tanishk-ou/census-income.git
cd census-income
```
Create a virtual environment:
```
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate` ```
```
Install the required packages:
```
pip install -r requirements.txt
```

## Usage
### Data Preprocessing
Ensure your data files are in the data directory. Combine the 'adult.data' and 'adult.test' datasets, then preprocess using multiple label encoders and scalers. The preprocessed dataset is saved as dataset_pp.pkl.

Here is an example of preprocessing:

```import joblib
import pandas as pd

# Load data
data = pd.read_csv('data/adult.data', skipinitialspace = True)
test_data = pd.read_csv('data/adult.test', skipinitialspace = True)
combined_data = pd.concat([data, test_data])

# Preprocess data (apply encoders and scalers)
label_encoders = {}
categorical = ['marital-status', 'relationship', 'race', 'sex', 'native-country', 'workclass', 'occupation']
numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

for col in categorical:
    label_encoders[col] = joblib.load(f'{col}_label_encoder.pkl')

ss = joblib.load('standard_scaler.pkl')

# Transform categorical features using the saved encoders
for col in categorical:
    combined_data[col] = label_encoders[col].transform(combined_data[col])

# Transform numerical features using the saved scaler
combined_data[numerical] = ss.transform(combined_data[numerical])

# Save the processed dataset
joblib.dump(combined_data, 'data/dataset_pp.pkl')
```

### Model Training
All the training code is in the Jupyter notebook. Open and run the notebook to train your models:

```jupyter notebook notebooks/notebook.ipynb```

This notebook will load the data, preprocess it, train the model, and save the trained model to the models directory.

### Prediction
For making predictions, you can add a new section in your notebook to load the preprocessed data and trained model, then generate predictions.

Example:
```
import joblib
from sklearn.ensemble import GradientBoostingClassifier

# Load preprocessed data
data = joblib.load('data/dataset_pp.pkl')

# Use any model whichever you want
model = GradientBoostingClassifer()

# Make predictions
predictions = model.predict(data)
```

## Project Structure
census-income-project/
```
│
├── data/
│   ├── raw/              
│       ├── adult.data          # Raw training dataset
│       ├── adult.test          # Raw testing dataset
│       ├── dataset.pkl         # Combined raw dataset
│   └── processed/      
│       └── dataset_pp.pkl      # Pre-processed dataset
│
├── notebooks/
│   └── notebook.ipynb          # Jupyter notebook for EDA, preprocessing, and model training
│
├── models/
│   ├── scalers.pkl             # Dictionary of scalers
│   └── label_encoders.pkl      # Dictionary of label encoders
│
├── README.md
├── LICENSE
└── requirements.txt
```

## Model Training
To train the models, open and run the Jupyter notebook:

```jupyter notebook notebooks/notebook.ipynb```

## Prediction
To make predictions, add the prediction code to the notebook or create a new script to load the model and preprocessed data, then generate predictions.

## Results
Include your model evaluation results, such as accuracy, precision, recall, and F1-score, within the Jupyter notebook.

## Contributing
Contributions are welcome! Please create an issue or submit a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgements
The [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult) for providing the dataset.
Scikit-learn for providing the machine learning algorithms and tools.
The open-source community for their contributions and support.

## Citations
Becker,Barry and Kohavi,Ronny. (1996). Adult. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20.


