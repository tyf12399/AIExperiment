# Classify penguins using a decision tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Read the data
df = pd.read_csv("./PenguinsClassification/penguin.csv")

# replace missing values with -1
df = df.fillna(-1)

# drop the columns that are not needed
df = df.drop(["Island", "Body Mass (g)", "Sex", "Age"], axis=1)

# convert the Species to numbers
df["Species"] = df["Species"].map(
    {
        "Adelie Penguin (Pygoscelis adeliae)": 0,
        "Gentoo penguin (Pygoscelis papua)": 1,
        "Chinstrap penguin (Pygoscelis antarctica)": 2,
    }
)

# split the data into training and testing
train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

# extract the features and labels
train_features = train.drop(["Species"], axis=1)
train_labels = train["Species"]
test_features = test.drop(["Species"], axis=1)
test_labels = test["Species"]

# create the decision tree
model = DecisionTreeClassifier()
model.fit(train_features, train_labels)

# show the accuracy
print("Accuracy:", model.score(test_features, test_labels))

# show the confusion matrix by heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(
    pd.crosstab(
        test_labels,
        model.predict(test_features),
        rownames=["Actual"],
        colnames=["Predicted"],
    ),
    annot=True,
    cmap="Blues",
)

# show the decision tree
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=train_features.columns,
    class_names=["Adelie", "Gentoo", "Chinstrap"],
    filled=True,
)
plt.show()
