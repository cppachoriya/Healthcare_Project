# Healthcare Appointment No-Show Prediction

This project aims to predict whether a patient will miss their medical appointment using real-world data. The dataset includes patient demographics, appointment scheduling details, medical conditions, and SMS reminders. 

The goal is to help healthcare facilities reduce no-shows by identifying high-risk patients using machine learning.

---

## 🔍 Problem Statement

Healthcare no-shows lead to resource waste and decreased patient care efficiency. By building predictive models, this project seeks to:

- Identify patterns in no-show behavior
- Understand the impact of age, SMS reminders, weekday, and health conditions
- Provide data-driven insights for optimizing appointment scheduling

---

## 📂 Project Structure
Healthcare_Project/
├── KaggleV2-May-2016.csv # Original dataset
├── Cleaned_Datadet_Healthcare.csv # Cleaned and processed dataset for modeling & Power BI
├── Healthcare_Appointment_Project.py # Full Python script with modeling and visualizations
└── README.md # Project documentation

---

## 🧪 Machine Learning Models Used

1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **XGBoost Classifier**

Models were evaluated using:
- Precision, Recall, F1-Score
- Accuracy on test dataset

---

## 📊 Visualizations (Python)

- No-show Rate by Weekday
- Impact of SMS Reminders
- No-show Rate by Age Group
- No-show Rate by Chronic Conditions (Hypertension, Diabetes, Alcoholism, Scholarship)

---

## 💼 Power BI Ready

The cleaned dataset (`Cleaned_Datadet_Healthcare1.csv`) is formatted and saved specifically for Power BI dashboarding. Suggested KPIs and visuals include:

- Total Appointments
- No-Show Count and Rate
- Appointments by Weekday
- Demographics and Condition-based breakdown
- Geographic map of neighborhoods (if enabled)

---

## 🚀 How to Run This Project

1. Install required Python libraries:
    ```bash
    pip install pandas matplotlib seaborn scikit-learn xgboost
    ```

2. Run the Python script:
    ```bash
    python Healthcare_NoShow_Predictor.py
    ```

3. Open the cleaned CSV file in Power BI for further visualization.

---

## ✅ Dataset Source

- [Kaggle: Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments)

---

## 🙋‍♂️ Author

**Chandra**  
Aspiring Data Analyst | Passionate about AI, BI, and meaningful data-driven projects.

---

## 📬 Contact

- GitHub: [github.com/your-username](https://github.com/cppachoriya)
- LinkedIn: [linkedin.com/in/your-profile](www.linkedin.com/in/chandra-prakash-pachoriya)

---

## 📌 License

This project is open-source and available for use under the [MIT License](LICENSE).
