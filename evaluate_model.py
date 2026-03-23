import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# True labels
y_true = ["BPSK","BPSK","QPSK","BPSK","QPSK"]

# Predicted labels from CNN
y_pred = ["BPSK","QPSK","QPSK","BPSK","QPSK"]

# Accuracy
accuracy = accuracy_score(y_true,y_pred)

print("Model Accuracy:", accuracy*100,"%")

# Confusion Matrix
cm = confusion_matrix(y_true,y_pred,labels=["BPSK","QPSK"])

plt.figure(figsize=(5,4))
sns.heatmap(cm,annot=True,fmt="d",
            xticklabels=["BPSK","QPSK"],
            yticklabels=["BPSK","QPSK"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()