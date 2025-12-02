import numpy as np
import shap
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from catboost import CatBoostClassifier
import logging
import os

logger = logging.getLogger(__name__)
class GBE(BaseEstimator, ClassifierMixin):
    def __init__(self, fun_model, esize: int =30, **model_params):
        self.esize: int = esize
        self.model_params = model_params
        self.ensemble = []
        self.classes_ = None

        for e in range(self.esize):
            model = fun_model(**model_params, random_seed=100 + e)
            self.ensemble.append(model)
        logging.info(f"Initialized {self.esize} models in the ensemble.")

    def fit(self, X, y, **kwargs):
        self.classes_ = np.unique(y)

        for i, m in enumerate(self.ensemble, 1):
            logging.info(f"Training model {i}/{self.esize}...")
            m.fit(X, y, **kwargs)
            logging.info(f"Model {i}/{self.esize} trained successfully.")

        return self

    def predict(self, X):
        check_is_fitted(self, 'ensemble')
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        check_is_fitted(self, 'ensemble')

        all_probs = []
        for model in self.ensemble:
            all_probs.append(model.predict_proba(X))

        mean_probs = np.mean(all_probs, axis=0)
        return mean_probs

    def compute_shap_values(self, X):
        check_is_fitted(self, 'ensemble')

        all_shap_values = []
        all_base_values = []

        for model in self.ensemble:
            explainer = shap.TreeExplainer(model)
            explanation = explainer(X)

            all_shap_values.append(explanation.values)
            all_base_values.append(explanation.base_values)

        avg_shap_values = np.mean(all_shap_values, axis=0)
        avg_base_values = np.mean(all_base_values, axis=0)

        final_explanation = shap.Explanation(
            values=avg_shap_values,
            base_values=avg_base_values,
            data=X,
            feature_names=self.ensemble[0].feature_names_
        )

        return final_explanation

    @property
    def feature_names_(self):
        if self.ensemble:
            return self.ensemble[0].feature_names_
        return None

    def save(self, directory='gbe', file_prefix='ensemble_model'):
        if not os.path.exists(directory):
            os.makedirs(directory)

        for i, model in enumerate(self.ensemble):
            model_filename = os.path.join(directory, f"{file_prefix}_{i}.cbm")
            if hasattr(model, "save_model"):
                model.save_model(model_filename)
                logging.info(f"Model {i} saved to {model_filename}")
            else:
                logging.warning(f"Model {i} does not support 'save_model', skipping...")

    @classmethod
    def load(cls, directory='gbe', file_prefix='ensemble_model', fun_model=None, esize=30, **params):
        if fun_model is None:
            raise ValueError("fun_model must be provided to load models.")

        instance = cls(fun_model, esize=esize, **params)

        instance.ensemble = []
        for i in range(esize):
            model_filename = os.path.join(directory, f"{file_prefix}_{i}.cbm")
            model = fun_model(**params)
            if hasattr(model, "load_model"):
                model.load_model(model_filename)
                logging.info(f"Model {i} loaded from {model_filename}")
            else:
                logging.warning(f"Model {i} does not support 'load_model', skipping...")
            instance.ensemble.append(model)

        return instance