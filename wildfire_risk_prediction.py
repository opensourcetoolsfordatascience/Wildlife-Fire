import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('/content/forestfires.csv')

print(f"\n Shape of Dataset: {df.shape}")
df.head()

month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
day_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}

df['month_num'] = df['month'].map(month_map)
df['day_num'] = df['day'].map(day_map)

# 1 = Fire, 0 = No Fire
df['has_fire'] = df['area'].apply(lambda x: 1 if x > 0 else 0)

df['log_area'] = np.log1p(df['area'])

print("Added columns: 'month_num', 'day_num', 'has_fire'")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.countplot(data=df, x='month_num', hue='has_fire', ax=axes[0, 0])
axes[0, 0].set_title('Fire Frequency by Month')

numeric_cols = df.select_dtypes(include=[np.number]).columns
sns.heatmap(df[numeric_cols].corr(), cmap='coolwarm', ax=axes[0, 1])
axes[0, 1].set_title('Correlation Matrix')

sns.scatterplot(data=df, x='wind', y='log_area', hue='has_fire', palette='viridis', ax=axes[1, 0])
axes[1, 0].set_title('Wind Speed vs. Log(Burned Area)')

sns.boxplot(data=df, x='has_fire', y='temp', hue='has_fire', legend=False, ax=axes[1, 1])
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_xticklabels(['No Fire', 'Fire'])
axes[1, 1].set_title('Temperature: Fire vs. No Fire')

plt.tight_layout()
plt.show()

features = ['temp', 'RH', 'wind', 'rain', 'FFMC', 'DMC', 'DC', 'ISI', 'month_num', 'day_num']
X = df[features]
y = df['has_fire']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.2f}")
print("\nDetailed Report:\n", classification_report(y_test, y_pred))

fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')

idx = np.argmin(np.abs(thresholds - 0.33))
plt.scatter(fpr[idx], tpr[idx], color='red', s=150,
            label=f'Threshold\n(Caught: {tpr[idx]*100:.1f}%)')

plt.xlabel('False Positive Rate (False Alarms ->)')
plt.ylabel('True Positive Rate (Caught Fires ->)')
plt.title('ROC Curve: Balancing Risk and Resources')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

for i in range(0, len(thresholds), max(1, len(thresholds)//10)):
    print(f"Threshold: {thresholds[i]*100:>5.1f}% | Caught Fires: {tpr[i]*100:>5.1f}% | False Alarms: {fpr[i]*100:>5.1f}%")

probabilities = model.predict_proba(X_test)[:, 1]
optimal_threshold = 0.33
y_pred_optimal = (probabilities >= optimal_threshold).astype(int)

cm = confusion_matrix(y_test, y_pred_optimal)

fig, ax = plt.subplots(figsize=(7, 5))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Fire', 'Fire'])
disp.plot(cmap='Blues', ax=ax)
ax.set_title(f'Optimized Confusion Matrix Threshold')

plt.tight_layout()
plt.show()

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()

print("Top 3 Drivers of Fire:", [features[i] for i in indices[:3]])