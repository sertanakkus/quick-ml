from models.svm import SVC_, SVR_
from models.knn import KNN
from models.linear_regression import linear_regression
from models.decision_tree_regressor import decision_tree_regressor
from models.logistic_regression import logistic_regression
from models.random_forest_regressor import random_forest_regressor


class Algorithms:
    def apply(algorithm, target):
        try:
            algorithm = algorithm
            target_feature = target

            # Linear Regression
            if algorithm == "Linear Regression":
                print("linear regression is applying")

                score = linear_regression(target_feature)

            # Decision Tree Regressor
            elif algorithm == "Decision Tree Regressor":
                print("Decision Tree Regressor is applying")

                score = decision_tree_regressor(target_feature)

            # Random Forest Regressor
            elif algorithm == "Random Forest Regressor":
                print("Random Forest Regressor is applying")

                score = random_forest_regressor(target_feature)

            # Logistic Regression
            elif algorithm == "Logistic Regression":
                print("Logistic Regression is applying")

                score = logistic_regression(target_feature)

            # KNN
            elif algorithm == "KNN":
                print("KNN is applying")

                score = KNN(target_feature)

            # SVR
            elif algorithm == "SVR":
                print("SVR is applying")

                score = SVR_(target_feature)

            # SVC
            elif algorithm == "SVC":
                print("SVC is applying")

                score = SVC_(target_feature)

            return {
                "message": f"{algorithm} model trained successfully",
                "score": score,
            }

        except Exception as e:
            return {f"error: {e}"}
