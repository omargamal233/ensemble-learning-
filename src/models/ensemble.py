from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression


def build_voting_classifier(models: dict, voting: str = "soft"):
    estimators = [(name, clf) for name, clf in models.items()]
    return VotingClassifier(estimators=estimators, voting=voting, n_jobs=1)


def build_stacking_classifier(models: dict):
    estimators = [(name, clf) for name, clf in models.items()]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=1,
        passthrough=False,
    )
