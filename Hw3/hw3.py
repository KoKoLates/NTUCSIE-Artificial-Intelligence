
import ipdb
import numpy as np
import pandas as pd

# set random seed
np.random.seed(0)

"""
Tips for debugging:
- Use `print` to check the shape of your data. Shape mismatch is a common error.
- Use `ipdb` to debug your code
    - `ipdb.set_trace()` to set breakpoints and check the values of your variables in interactive mode
    - `python -m ipdb -c continue hw3.py` to run the entire script in debug mode. Once the script is paused, you can use `n` to step through the code line by line.
"""


# 1. Load datasets
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """ DO NOT MODIFY THIS FUNCTION. """
    # Load iris dataset
    iris = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )
    iris.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]

    # Load Boston housing dataset
    boston = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    return iris, boston


# 2. Preprocessing functions
def train_test_split(
    df: pd.DataFrame, 
    target: str, 
    test_size: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Shuffle and split dataset into train and test sets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # Split target and features
    X_train = train.drop(target, axis=1).values
    y_train = train[target].values
    X_test = test.drop(target, axis=1).values
    y_test = test[target].values

    return X_train, X_test, y_train, y_test


def normalize(X: np.ndarray) -> np.ndarray:
    """ feature normalization """
    # TODO: 1%
    X_normalize = np.empty((X.shape))
    min_value, max_value = np.amin(X, axis=0), np.amax(X, axis=0)
    for i, j in np.ndindex(X.shape):
        X_normalize[i, j] = (X[i, j] - min_value[j]) / (max_value[j] - min_value[j])
    
    return X_normalize


def encode_labels(y: np.ndarray) -> np.ndarray:
    """ Encode labels to integers """
    # TODO: 1%
    return np.array(
        [['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'].index(y[i]) \
         for i in range(y.size)]
    )


# 3. Models
class LinearModel:
    def __init__(
        self, 
        learning_rate: float = 0.01, 
        iterations: int = 1000, 
        model_type: str = 'linear'
    ) -> None:
        """ 
        @param learning_rate: the learning rate for gradient descent.
        @param iterations: training iteration.
        @param model_type: option for `linear` or `logistic` regression.
        """
        assert model_type in [
            'linear', 'logistic'
        ], "model_type must be either 'linear' or 'logistic'"
        self.model_type = model_type

        # You can try different learning rate and iterations
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ training based on the training dataset """
        X = np.insert(X, 0, 1, axis=1) # bias
        n_classes = len(np.unique(y))
        n_features = X.shape[1]

        # TODO: 2%
        self.weights = np.zeros(
            n_features if self.model_type == 'linear' \
                else (n_features, n_classes)
        )            
        
        for _ in range(self.iterations):
            self.weights -= self.learning_rate * self._compute_gradients(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ predict the results based on the pre-trained weights """
        X = np.insert(X, 0, 1, axis=1)
        # TODO: 4%
        return X @ self.weights if self.model_type == 'linear' else \
            np.argmax(self._softmax(X @ self.weights), axis=1)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """ softmax activation function """
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def _compute_gradients(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> np.ndarray:
        """ compute the gradient based on the error """
        n_samples, _ = X.shape
        n_classes = len(np.unique(y))

        if self.model_type == 'linear':
            # TODO: 3%
            return 1 / n_samples * (X.T @ (X @ self.weights - y))
        
        # TODO: 3%
        # gradient for logistic regression
        y_enc = np.zeros((n_samples, n_classes), dtype=int)
        for i in range(n_samples):
            y_enc[i, y[i]] = 1
            
        return -1 / n_samples * (X.T @ (y_enc - self._softmax(X @ self.weights)))    


class DecisionTree:
    def __init__(
        self, max_depth: int = 5, 
        model_type: str = "classifier"
    ) -> None:
        """
        @param max_depth: the maximum depth for the leaves of decision tree
        @param model_type: `classifier` or `regressor` for the model's purpose.
        """
        assert model_type in [
            "classifier",
            "regressor",
        ], "model_type must be either 'classifier' or 'regressor'"

        self.max_depth = max_depth
        self.model_type = model_type

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.tree = self._build_tree(X, y, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        depth: int
    ) -> dict[str, int | float]:
        """ building the node for tree or sub-tree
        @param X: splited training dataset.
        @param y: splited one-hot encoded label.
        @param depth: the depth of current node.
        """
        if depth >= self.max_depth or self._is_pure(y):
            return self._create_leaf(y)

        feature, threshold = self._find_best_split(X, y)

        # TODO: 4%
        n_samples, _ = X.shape
        x_left, x_right, y_left, y_right = [], [], [], []
        for i in range(n_samples):
            x_left.append(X[i]) if X[i, feature] <= threshold else x_right.append(X[i])
            y_left.append(y[i]) if X[i, feature] <= threshold else y_right.append(y[i])

        left_child = self._build_tree(
            np.array(x_left),  np.array(y_left),  depth + 1
        )
        right_child = self._build_tree(
            np.array(x_right), np.array(y_right), depth + 1
        )

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_child,
            "right": right_child,
        }

    def _is_pure(self, y: np.ndarray) -> bool:
        return len(set(y)) == 1

    def _create_leaf(self, y: np.ndarray) -> np.float64:
        if self.model_type == 'regressor':
            # TODO: 1%
            return y.mean()
        
        # TODO: 1%
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
    
    def _find_best_split(
        self, 
        X: np.ndarray,
        y: np.ndarray
    ) -> tuple[int, float]:
        """ find the best split of data 
        @param: input training data
        @param: input training label
        @return: a tuple of best feature and corresponding threshold
        """
        best_mse: float = float('inf')
        best_gini: float = float('inf')
        best_feature: int = None
        best_threshold: float = None

        for feature in range(X.shape[1]):
            sorted_indices = np.argsort(X[:, feature])
            for i in range(1, len(X)):
                if X[sorted_indices[i - 1], feature] == X[sorted_indices[i], feature]:
                    continue

                threshold = (
                    X[sorted_indices[i - 1], feature] + \
                    X[sorted_indices[i], feature]
                ) / 2

                mask = X[:, feature] <= threshold
                left_y, right_y = y[mask], y[~mask]

                if self.model_type == 'classifier':
                    gini: float = self._gini_index(left_y, right_y)
                    if gini >= best_gini: continue
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
                    continue

                mse: float = self._mse(left_y, right_y)
                if mse >= best_mse: continue
                best_mse = mse
                best_feature = feature
                best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        def _gini_calc(y: np.ndarray) -> float:
            p = np.array([(y == i).sum() / y.size for i in np.unique(y)])
            return 1 - (p ** 2).sum()
        
        return (left_y.size * _gini_calc(left_y) + \
                right_y.size * _gini_calc(right_y)) / (left_y.size + right_y.size)

    def _mse(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        def _mse_calc(y: np.ndarray) -> float:
            return np.square(y - y.mean()).mean()
        
        return (left_y.size * _mse_calc(left_y) + \
                right_y.size * _mse_calc(right_y)) / (left_y.size + right_y.size)

    def _traverse_tree(self, x: np.ndarray, node: dict | np.float64):
        if not isinstance(node, dict):
            return node
        
        feature, threshold = node['feature'], node['threshold']
        return self._traverse_tree(x, node['left' if x[feature] <= threshold else 'right'])


class RandomForest:
    def __init__(
        self, 
        n_estimators: int = 100, 
        max_depth: int = 5, 
        model_type: str = "classifier"
    ) -> None:
        """
        @param n_estimators: number of estimator of random forest
        @param max_depth: max depth for each decision tree
        @param model_type: functional type of decision tree based on purpose
        """
        # TODO: 1%
        self.model_type = model_type
        self.trees = [
            DecisionTree(max_depth=max_depth, model_type=model_type)
            for _ in range(n_estimators)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for tree in self.trees:
            # TODO: 2%
            boostrap_indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            sample_x = np.array([X[i] for i in boostrap_indices])
            sample_y = np.array([y[i] for i in boostrap_indices])
            tree.fit(sample_x, sample_y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: 2%
        n_samples, n_features = X.shape
        if self.model_type == 'classifier':
            predictions_matrix = np.array([tree.predict(X) for tree in self.trees]).T
            result = np.zeros(n_samples)
            for i in range(n_samples):
                values, counts = np.unique(predictions_matrix[i], return_counts=True)
                result[i] = values[np.argmax(counts)]
            
            return result
        else:
            result = np.zeros(n_samples)
            for tree in self.trees:
                result += tree.predict(X)
            
            result /= len(self.trees)
            return result


# 4. Evaluation metrics
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # TODO: 1%
    return (y_pred == y_true).sum() / y_true.size

def mean_squared_error(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> np.float64:
    # TODO: 1%
    return np.square(np.subtract(y_true, y_pred)).mean()


# 5. Main function
def main() -> None:
    iris, boston = load_data()

    # Iris dataset - Classification
    X_train, X_test, y_train, y_test = train_test_split(iris, "class")
    X_train, X_test = normalize(X_train), normalize(X_test)
    y_train, y_test = encode_labels(y_train), encode_labels(y_test)

    logistic_regression = LinearModel(model_type="logistic")
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy(y_test, y_pred))

    decision_tree_classifier = DecisionTree(model_type="classifier")
    decision_tree_classifier.fit(X_train, y_train)
    y_pred = decision_tree_classifier.predict(X_test)
    print("Decision Tree Classifier Accuracy:", accuracy(y_test, y_pred))

    random_forest_classifier = RandomForest(model_type="classifier")
    random_forest_classifier.fit(X_train, y_train)
    y_pred = random_forest_classifier.predict(X_test)
    print("Random Forest Classifier Accuracy:", accuracy(y_test, y_pred))

    # Boston dataset - Regression
    X_train, X_test, y_train, y_test = train_test_split(boston, "medv")
    X_train, X_test = normalize(X_train), normalize(X_test)

    linear_regression = LinearModel(model_type="linear")
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))

    decision_tree_regressor = DecisionTree(model_type="regressor")
    decision_tree_regressor.fit(X_train, y_train)
    y_pred = decision_tree_regressor.predict(X_test)
    print("Decision Tree Regressor MSE:", mean_squared_error(y_test, y_pred))

    random_forest_regressor = RandomForest(model_type="regressor")
    random_forest_regressor.fit(X_train, y_train)
    y_pred = random_forest_regressor.predict(X_test)
    print("Random Forest Regressor MSE:", mean_squared_error(y_test, y_pred))


if __name__ == "__main__":
    main()
