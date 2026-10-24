# Titanic Survival Prediction

A beginner-friendly machine learning project that predicts whether a Titanic passenger survived using passenger information.

## Dataset
- File: `Titanic-Dataset.csv`
- Rows: 891 passengers
- Target column: `Survived` (`0` = did not survive, `1` = survived)

## Features used
- `Pclass`: passenger class
- `Sex`: passenger sex
- `Age`: passenger age
- `SibSp`: number of siblings/spouses aboard
- `Parch`: number of parents/children aboard
- `Fare`: ticket fare
- `Embarked`: port of embarkation

Passenger names, IDs, ticket numbers, and cabin values are not used by this baseline model.

## Requirements
Python 3.9 or newer is recommended.

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the project
Place the dataset and `titanic_survival_prediction.py` in the same folder, then run:
```bash
python titanic_survival_prediction.py
```

The script splits the dataset into training and test sets, handles missing values, encodes categorical features, trains a Logistic Regression model, and prints accuracy and a classification report.

## Project files
- `Titanic-Dataset.csv` — dataset
- `titanic_survival_prediction.py` — model training and evaluation code
- `requirements.txt` — Python dependencies

## Disclaimer
This is an educational baseline project. Results depend on the dataset split and model settings; it is not intended for real-world decision-making.
