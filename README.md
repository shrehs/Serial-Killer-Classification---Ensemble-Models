# Serial-Killer-Classification---Ensemble-Models
---

```markdown
# 🔪 Serial Killer Classification using Ensemble Models

Welcome to the dark side of data science — this project explores the patterns, behaviors, and profiles of serial killers using machine learning and interactive visualizations. From historical EDA to predictive modeling, the project takes a data-driven approach to better understand criminal psychology and classification using ensemble learning techniques.

🔗 **Live App**: [serial-killer-classification.streamlit.app](https://serial-killer-classification.streamlit.app/)

---

## 📌 Project Overview

This repository contains:
- 📊 **Exploratory Data Analysis** on serial killer profiles
- 🧠 **Classification models** using Random Forest, Gradient Boosting, and other ensemble methods
- 🌐 **Deployed Streamlit app** to interactively explore patterns and make predictions

---

## 📂 Contents

```

📁 Data understanding/
├── Serial\_kills.ipynb              <- EDA and modeling notebook
├── serial\_kills\_streamlit.py       <- Streamlit app code
├── requirements.txt                <- Python dependencies
├── .gitignore                      <- Ignores notebook checkpoints

````

---

## 🧪 Features

- Gender and age distribution of serial killers
- Behavioral and psychological patterns (e.g. childhood trauma, mental illness)
- Classification of individuals into "serial killer" or "non-serial killer"
- Interactive visualizations on timelines, methods, countries, etc.
- Real-time prediction UI using ensemble classification

---

## 🚀 Deployment

The app is deployed using **Streamlit Cloud**:  
[https://serial-killer-classification.streamlit.app/](https://serial-killer-classification.streamlit.app/)

---

## ⚙️ Setup Locally

Clone the repository:

```bash
git clone https://github.com/shrehs/Serial-Killer-Classification---Ensemble-Models.git
cd Serial-Killer-Classification---Ensemble-Models
````

Create a virtual environment and install dependencies:

```bash
conda create -n serialkiller python=3.10
conda activate serialkiller
pip install -r requirements.txt
```

Run the app locally:

```bash
streamlit run serial_kills_streamlit.py
```

---

## 🧠 Model Approach

* Preprocessing of raw data with pandas
* Label encoding for categorical variables
* Feature selection for psychological traits and demographic attributes
* Classification using:

  * Random Forest
  * Gradient Boosting (XGBoost / LightGBM)
  * Voting Classifier

Evaluation includes accuracy, F1-score, and confusion matrices.

---

## 📊 Visualization Tools

* **Seaborn & Matplotlib** for EDA
* **Plotly** for interactive plots (in the app)
* **Streamlit** for real-time user interaction

---

## 📚 Dataset

The dataset used includes behavioral and demographic attributes of known serial killers, curated from public datasets, law enforcement reports, and investigative research archives.

---

## 🙏 Acknowledgments

* Streamlit for app framework
* Scikit-learn, Pandas, and Seaborn for the machine learning pipeline
* Dataset sources from open crime archives and psychological research

---

## 📬 Contact

Developed by **Shreya H S**
Feel free to connect via [LinkedIn](https://www.linkedin.com/in/shreya-h-s) or raise issues in this repo for feedback.

---

The deployed app: [https://serial-killer-classification.streamlit.app/](https://serial-killer-classification.streamlit.app/)

---

## ⚠️ Disclaimer

This project is intended for educational purposes only. The predictions made by the model should not be used for real-world profiling, judgment, or legal purposes. Criminal behavior is a complex domain that requires ethical handling and domain expertise.

```

---

Would you like a `LICENSE` file (e.g., MIT) or a badge-based header with Python version, Streamlit status, etc., next?
```
