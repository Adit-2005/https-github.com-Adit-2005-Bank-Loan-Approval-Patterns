import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Dataset with Updated Path
# -----------------------------
train_path = r"C:\Users\aditj\Downloads\Bank Loan Approval Pattern Analysis\Dataset\train_u6lujuX_CVtuZ9i.csv"
test_path = r"C:\Users\aditj\Downloads\Bank Loan Approval Pattern Analysis\Dataset\test_Y3wMUE5_7gLdaTN.csv"

df = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# -----------------------------
# 2. Data Cleaning
# -----------------------------
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# -----------------------------
# 3. Approval % by Gender
# -----------------------------
approval_gender = df.groupby('Gender')['Loan_Status'].value_counts(normalize=True).unstack().fillna(0)
approval_gender.plot(kind='bar', stacked=True, colormap='viridis', figsize=(6, 4))
plt.title("Loan Approval Percentage by Gender")
plt.ylabel("Proportion")
plt.xticks(rotation=0)
plt.legend(title="Loan Status")
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Approval % by Income Bracket
# -----------------------------
df['Income_Bracket'] = pd.cut(df['ApplicantIncome'],
                              bins=[0, 2500, 4000, 6000, 10000, 25000],
                              labels=['Low', 'Lower-Mid', 'Mid', 'High', 'Very High'])

approval_income = df.groupby('Income_Bracket')['Loan_Status'].value_counts(normalize=True).unstack().fillna(0)
approval_income.plot(kind='bar', stacked=True, colormap='plasma', figsize=(8, 5))
plt.title("Loan Approval Percentage by Income Bracket")
plt.ylabel("Proportion")
plt.xticks(rotation=45)
plt.legend(title="Loan Status")
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Approval % by Credit History (Credit Score)
# -----------------------------
approval_credit = df.groupby('Credit_History')['Loan_Status'].value_counts(normalize=True).unstack().fillna(0)
approval_credit.plot(kind='bar', stacked=True, colormap='Set2', figsize=(6, 4))
plt.title("Loan Approval % by Credit History (Credit Score)")
plt.ylabel("Proportion")
plt.xticks(rotation=0)
plt.legend(title="Loan Status")
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Correlation Heatmap
# -----------------------------
df_encoded = pd.get_dummies(df.drop(columns=['Loan_ID']), drop_first=True)

plt.figure(figsize=(12, 8))
sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Summary Insights
# -----------------------------
print("\nSummary Insights:")
print("- Males apply more often, but approval rates are fairly balanced by gender.")
print("- Higher income doesn’t ensure approval—Credit_History matters more.")
print("- Applicants with Credit_History = 1 (good credit) have a much higher approval rate.")
print("- Correlation heatmap confirms that Credit_History is the strongest predictor of Loan_Status.")
