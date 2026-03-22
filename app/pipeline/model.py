from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, root_mean_squared_error


class Model:
    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type

        if task_type == "classification":
            self.model = RandomForestClassifier(random_state=42)
        elif task_type == "regression":
            self.model = RandomForestRegressor(random_state=42)
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")

    def train(self, X_train, y_train):
        """
        Train the model
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions
        """
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        """
        Evaluate model performance
        """
        if self.task_type == "classification":
            return {
                "accuracy": accuracy_score(y_test, y_pred)
            }

        elif self.task_type == "regression":
            return {
                "rmse": root_mean_squared_error(y_test, y_pred, squared=False)
            }

    def feature_importance(self, feature_names):
        """
        Get feature importance
        """
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(feature_names, self.model.feature_importances_))
        else:
            return None