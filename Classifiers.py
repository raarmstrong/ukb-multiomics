from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

# 1. the base classifier
def get_base_classifier(n_cases):
    '''
    Returns a logistic regression classifier with elastic net regularization.
    Hyperparameter tuning depends on dataset
    '''

    if n_cases >=500:
        return LogisticRegressionCV(
            penalty='elasticnet',
            Cs=[0.01, 0.1, 1.0, 10.0],
            l1_ratios=[0.1, 0.5, 0.9],
            cv=3,
            solver='saga',
            random_state=42,
            scoring='balanced_accuracy',
            max_iter=10000
        )
    else:
        return LogisticRegression(
            penalty='elasticnet',
            C=1.0, 
            l1_ratio=0.5, 
            solver='saga', 
            random_state=42,
            max_iter=10000)

# 2. the parsimoniuos classifier
def get_simple_classifier(n_cases):
    '''
    Returns a logistic regression classifier with l2 regularization.
    Hyperparameter tuning depends on dataset
    '''

    if n_cases >=500:
        return LogisticRegressionCV(
            penalty='l2',
            Cs=[0.01, 0.1, 1.0, 10.0],
            cv=3,
            solver='lbfgs',
            random_state=42,
            scoring='balanced_accuracy',
            max_iter=10000
        )
    else:
        return LogisticRegression(
            penalty='l2',
            C=1.0, 
            solver='lbfgs', 
            random_state=42,
            max_iter=10000)

# 3. the balanced bagging classifier
def get_balanced_bagging_classifier(base_estimator):
    '''
    Returns a balanced bagging classifier with the given base estimator.
    '''
    return BalancedBaggingClassifier(
        base_estimator=base_estimator,
        n_estimators=50,
        sampling_strategy='not minority',
        replacement=False,
        random_state=42,
        n_jobs=1
    )

# 4. crossvalidation strategy
def get_cv_strategy(n_cases):
    '''
    Returns a stratified k-fold cross-validation strategy.
    '''
    if n_cases < 100:
        # Fewer splits, more repeats
        n_splits = 3
        n_repeats = 20
    else:
        # more splits, fewer repeats
        n_splits = 5
        n_repeats = 12

    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# 5. combine all into a function that returns the base classifier, the balanced bagging classifier and cv strategy
def get_classifier_and_cv(n_cases, transfer=False):
    '''
    Returns the base classifier, balanced bagging classifier and cross-validation strategy.
    '''
    if transfer:
        base_clf = get_simple_classifier(n_cases)
    else:       
        base_clf = get_base_classifier(n_cases)
    clf = get_balanced_bagging_classifier(base_clf)
    cv = get_cv_strategy(n_cases)
    return base_clf, clf, cv

