# Airfare Predictor Web Application with Streamlit

## Introduction
This repository stores the source code for an airfare predictor web app, written in Python using the Streamlit framework.

## Project Structure

```
.
├── airport_data_processing.ipynb
├── app
│   ├── main.py
│   ├── models
│   │   └── sets.py
│   ├── predict_nohops.py
│   ├── predict_nohops_return.py
│   └── predict_withhops.py
├── models
│   ├── airport_names.csv
│   ├── alex_xgboost_hyperopt.joblib
│   ├── alex_xgboost_hyperopt_new.joblib
│   ├── distance_data.csv
│   ├── distance_data.json
│   ├── names_data.json
│   ├── nicholas_mlbCabinCode.joblib
│   ├── nicholas_neuralnetwork_best.keras
│   ├── pine_xgb_pipeline_final.joblib
│   ├── travel_duration_data.csv
│   └── travel_duration_data.json
├── poetry.lock
├── pyproject.toml
└── requirements.txt
```

## Installation and Running

This is a private repository. If you can access this README, you are a part of it and may clone it to your machine for running. While it is private, you may follow these steps in your Terminal to install and run the app:

1. Clone the repository:

```
git clone https://<username>@github.com/nicnl31/airfare-streamlit
```

where `<username>` is your GitHub username. When prompted for your password, you must use your **personal access token (PAT)**. If you don't have one, you can create one following this guide: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic

After the repository is cloned, go into it:

```
cd airfare-streamlit
```

2. Create and activate a virtual environment on your machine:

```
python -m venv .venv
source .venv/bin/activate
```

3. Inside your virtual environment, upgrade `pip` and install the project's dependencies:

```
pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the app:

```
streamlit run app/main.py
```

5. Launch the app in your browser with localhost and port 8501 if it doesn't automatically do so:

```
http://localhost:8501
```

6. (Optional) If you no longer want the project and just need to get it out of your hair: Control+C to stop the app within Terminal. Then:

```
cd ..
rm -rf airfare-streamlit
```