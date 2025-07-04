import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Set visual style for seaborn charts
sns.set(style="whitegrid")

# 1. Load the dataset
df = pd.read_csv(r"E:\Python_coding\Healthcare_Project\KaggleV2-May-2016.csv")

# 2. Display dataset structure and check for missing values
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Basic data cleaning and feature engineering
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['DaysWaiting'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

# Remove invalid entries
df = df[df['DaysWaiting'] >= 0]
df = df[df['Age'] >= 0]

# Encode target column to binary values
df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})

# Create a weekday feature from AppointmentDay
df['Weekday'] = df['AppointmentDay'].dt.day_name()

# 4. Drop unnecessary columns
df.drop(['ScheduledDay', 'AppointmentDay', 'PatientId', 'AppointmentID'], axis=1, inplace=True)

# 5. One-hot encode categorical variables
df = pd.get_dummies(df, columns=['Gender', 'Neighbourhood', 'Weekday'], drop_first=True)

# 6. Split data into features and target
X = df.drop('No-show', axis=1)
y = df['No-show']

# Create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train a Decision Tree Classifier and evaluate it
print("\nDecision Tree Results:\n")
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print(classification_report(y_test, dt_pred))

# 8. Train a Random Forest Classifier and evaluate it
print("\nRandom Forest Results:\n")
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print(classification_report(y_test, rf_pred))

# 9. Train an XGBoost Classifier and evaluate it
print("\nXGBoost Results:\n")
neg, pos = y_train.value_counts()
scale_weight = neg / pos  # Handle class imbalance

xgb_model = XGBClassifier(scale_pos_weight=scale_weight, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print(classification_report(y_test, xgb_pred))

# 10. Export cleaned data for Power BI dashboards
df.to_csv(r"E:\Python_coding\Healthcare_Project\Cleaned_Datadet_Healthcare.csv", index=False)
print("\nCleaned dataset saved for Power BI.")

# Reload the original dataset for visualization (before encoding)
df_viz = pd.read_csv(r"E:\Python_coding\Healthcare_Project\KaggleV2-May-2016.csv")

# Repeat basic preprocessing for charts
df_viz['ScheduledDay'] = pd.to_datetime(df_viz['ScheduledDay'])
df_viz['AppointmentDay'] = pd.to_datetime(df_viz['AppointmentDay'])
df_viz['DaysWaiting'] = (df_viz['AppointmentDay'] - df_viz['ScheduledDay']).dt.days
df_viz = df_viz[df_viz['DaysWaiting'] >= 0]
df_viz = df_viz[df_viz['Age'] >= 0]
df_viz['No-show'] = df_viz['No-show'].map({'Yes': 1, 'No': 0})
df_viz['Weekday'] = df_viz['AppointmentDay'].dt.day_name()

# Create age groups for plotting
df_viz['AgeGroup'] = pd.cut(df_viz['Age'], bins=[0, 18, 35, 50, 65, 100],
                            labels=["0-18", "19-35", "36-50", "51-65", "66+"])

# Chart 1: No-show rate by weekday
plt.figure(figsize=(8, 4))
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.barplot(x='Weekday', y='No-show', data=df_viz, order=weekday_order)
plt.title("No-Show Rate by Weekday")
plt.ylabel("No-Show Rate")
plt.xlabel("Weekday")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Chart 2: Impact of SMS reminders
plt.figure(figsize=(5, 4))
sns.barplot(x='SMS_received', y='No-show', data=df_viz)
plt.title("Impact of SMS on No-Shows")
plt.xticks([0, 1], ['No SMS', 'SMS Sent'])
plt.ylabel("No-Show Rate")
plt.xlabel("SMS Reminder")
plt.tight_layout()
plt.show()

# Chart 3: No-show rate by age group
plt.figure(figsize=(6, 4))
sns.barplot(x='AgeGroup', y='No-show', data=df_viz)
plt.title("No-Show Rate by Age Group")
plt.ylabel("No-Show Rate")
plt.xlabel("Age Group")
plt.tight_layout()
plt.show()

# Chart 4: No-show rate by chronic conditions
plt.figure(figsize=(8, 4))
chronic_conditions = ['Diabetes', 'Hipertension', 'Alcoholism', 'Scholarship']
melted = df_viz.melt(id_vars='No-show', value_vars=chronic_conditions)
sns.barplot(x='variable', y='value', hue='No-show', data=melted)
plt.title("Chronic Conditions vs No-Show")
plt.xlabel("Condition")
plt.ylabel("Proportion")
plt.legend(title="No-Show", labels=["Showed Up", "No-Show"])
plt.tight_layout()
plt.show()
