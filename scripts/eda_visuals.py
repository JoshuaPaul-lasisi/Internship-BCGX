import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the directory exists
output_dir = "../report/images"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
data = pd.read_csv("your_dataset.csv")  # Update with the actual dataset path

# Basic information
print(data.info())
print(data.describe())

# Handle missing values
data.fillna(data.median(), inplace=True)  # Example of filling missing values

# Visualizations

# Histogram of numerical features
numerical_features = data.select_dtypes(include=[np.number]).columns
for col in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(data[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(f"{output_dir}/{col}_histogram.png")
    plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.savefig(f"{output_dir}/correlation_heatmap.png")
plt.close()

# Boxplots for numerical features
for col in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=data[col])
    plt.title(f'Boxplot of {col}')
    plt.savefig(f"{output_dir}/{col}_boxplot.png")
    plt.close()

# Pairplot for the first five numerical features (to avoid overloading)
sns.pairplot(data[numerical_features[:5]])
plt.savefig(f"{output_dir}/pairplot.png")
plt.close()

# Countplot for categorical features
categorical_features = data.select_dtypes(include=['object']).columns
for col in categorical_features:
    plt.figure(figsize=(8, 5))
    sns.countplot(y=data[col], order=data[col].value_counts().index)
    plt.title(f'Countplot of {col}')
    plt.savefig(f"{output_dir}/{col}_countplot.png")
    plt.close()

print("EDA completed. All visualizations are saved in ../report/images/")
