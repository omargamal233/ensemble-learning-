import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_predict, cross_val_score


def evaluate_classifier(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    report = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    return {
        "accuracy_mean": scores.mean(),
        "accuracy_std": scores.std(),
        "report": report,
        "confusion_matrix": cm,
        "predictions": y_pred,
    }


def evaluate_classifier_with_proba(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    report = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    
    return {
        "accuracy_mean": scores.mean(),
        "accuracy_std": scores.std(),
        "report": report,
        "confusion_matrix": cm,
        "predictions": y_pred,
        "proba": y_proba,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
    }


def plot_confusion_matrix(cm, ax=None):
    if ax is None:
        ax = plt.gca()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return ax


def roc_curve_data(y, y_proba):
    fpr, tpr, _ = roc_curve(y, y_proba)
    return fpr, tpr, auc(fpr, tpr)
