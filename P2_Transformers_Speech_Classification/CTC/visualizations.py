from sklearn.metrics import accuracy_score, precision_recall_fscore_support,confusion_matrix, classification_report, precision_score, recall_score
from sklearn.metrics import f1_score as f1_score_rep
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def print_score(y_pred, y_real, label_encoder):
    print("Accuracy: ", accuracy_score(y_real, y_pred))
    print("Precision:: ", precision_score(y_real, y_pred, average="micro"))
    print("Recall:: ", recall_score(y_real, y_pred, average="micro"))
    print("F1_Score:: ", f1_score_rep(y_real, y_pred, average="micro"))

    print()
    print("Confusion Matrix")
    cm = confusion_matrix(y_real, y_pred, labels = [i for i in label_encoder])
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, index = [i for i in label_encoder],
                  columns = [i for i in label_encoder])
    # df_cm = np.round(df_cm, decimals=2)
    plt.figure(figsize = (12, 10))
    sn.heatmap(df_cm, annot=True, fmt='g', cmap='viridis', norm=colors.LogNorm())#, fmt='g', cmap='viridis', norm=colors.LogNorm())
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    print()
    print("Classification Report")
    print(classification_report(y_real, y_pred, target_names=label_encoder))