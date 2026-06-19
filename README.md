# Industrial Motor Predictive Maintenance

Machine learning and Streamlit-based predictive maintenance system for industrial motors using vibration sensor data analysis.

## Project Overview

This project analyzes vibration sensor data collected from industrial motors to detect anomalies and predict potential failures before they occur.

The system uses machine learning techniques to classify motor conditions and support preventive maintenance strategies.

## Features

- Vibration data preprocessing
- Feature extraction
- Fault detection
- Predictive maintenance alerts
- Machine Learning model training
- Interactive prediction dashboard with Streamlit
- Performance evaluation

## Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Jupyter Notebook
- lightgbm
- streamlit

## Dataset

The dataset contains vibration measurements collected from industrial motors under different operating conditions.

## Installation

```bash
git clone https://github.com/yourusername/industrial-motor-predictive-maintenance.git
cd industrial-motor-predictive-maintenance
pip install -r requirements.txt
```
Usage
```bash
python src/visual.py
python src/split.py
python src/train.py
```
## Streamlit Web Application

The project includes an interactive Streamlit dashboard that allows users to:

- Upload vibration data
- Generate maintenance predictions
- Detect abnormal motor behavior
- Download the detailed report

Launch the application:

```bash
streamlit run predict.py
```
## Application Preview

<img width="290" height="217" alt="image" src="https://github.com/user-attachments/assets/cc0e16a2-b4fa-46d4-9e3a-0c431bf74e99" />
