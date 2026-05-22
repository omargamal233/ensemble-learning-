from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from ..config import RANDOM_STATE


def get_base_classifiers():
    return {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
        "extra_trees": ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
        "adaboost": AdaBoostClassifier(random_state=RANDOM_STATE, learning_rate=0.1, n_estimators=100),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE, learning_rate=0.1, n_estimators=200),
        "svc": SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE)
    }


def fit_classifiers(classifiers: dict, X, y):
    fitted = {}
    for name, clf in classifiers.items():
        clf.fit(X, y)
        fitted[name] = clf
    return fitted
