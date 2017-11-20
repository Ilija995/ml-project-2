import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample


class ResamplingAndLogisticRegression(BaseEstimator, TransformerMixin):
    """
    Resample new data points to obtain larger,
    and then perform SVM classification
    """
    def fit(self, X, y):
        # Resample
        no_resamples = 3
        X_new = X
        y_new = y
        for i in range(no_resamples):
            X_resampled, y_resampled = resample(X, y, random_state=0)
            X_new = np.concatenate((X_new, X_resampled), axis=0)
            y_new = np.concatenate((y_new, y_resampled), axis=0)

        self.classifier = LogisticRegression()
        # self.classifier = SVC(probability=True)
        self.classifier.fit(X_new, np.argmax(y_new, axis=1))
        return self

    def predict_proba(self, X):
        check_array(X)
        check_is_fitted(self, ["classifier"])
        y = self.classifier.predict_proba(X)
        y_res = np.zeros(y.shape)
        res = [0.1, 0.2, 0.3, 0.4]
        for i in range(len(y)):
            sort_indices = np.argsort(y[i])
            for j in range(len(sort_indices)):
                y_res[i, sort_indices[j]] = res[j]
        return y_res
