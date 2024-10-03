"""
    define models using classes from scikit-learn
"""


from typing import Any, List, Tuple

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, log_loss
)

import numpy as np
import matplotlib.pyplot as plt

def train_random_forest(train_features: np.ndarray, 
                        train_rts: np.ndarray,
                        filepath: str, 
                        n_estimators: List[int], 
                        min_samples_split: List[int | float],
                        min_samples_leaf: List[int | float], 
                        max_depth: List[int], 
                        max_features: List[int | str | None],
                        ) -> Any :
    """
    build a random forest model:

    - [input] (train_features, train_rts, n_estimators, 
                max_depth, min_samples_split,
                min_samples_leaf,max_features, filepath)
    - [output] trained random forest model

    Parameters
    ----------
    train_features : numpy.ndarray(dtype=numpy.uint8)
        train feature data, N-bit RDKit fingerprints x m_rows
    train_rts : numpy.ndarray(dtype=np.float64)
        retention times
    filepath: str
        the filepath where you want the trained model instance saved

    Lists of values used for grid search
    ________
    n_estimators : list(int)
        list of integers that represent the number of trees in the forest model
    max_depth : list(int)
        list of integers that represent the max depth of the tree
    min_samples_split: list(int)
        list of integers or floats that represent the minimum  number of samples required 
        to split an internal node
    min_samples_leaf: list(int)
        list of integers or floats that represent the minimum number of samples required 
        to be at a leaf node (value cannot be greater than number of samples)
    max_features : list(int)
        list of integers, floats or any of three {“sqrt”, “log2”, None} that represent the 
        number of features to consider when looking for the best split

    Returns
    -------
    best_model : ``random_forest_regressor.Model``
        best model from grid search
    """
    #create random forest regressor object
    model = RandomForestRegressor() 
    #create dictionary of parameters to search
    parameter_grid = {
        'n_estimators': n_estimators, 'max_depth': max_depth,
        'min_samples_split': min_samples_split, 'min_samples_leaf':min_samples_leaf,
        'max_features':max_features}
    #create grid search object to find hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=parameter_grid, scoring='neg_mean_squared_error')
    #perform grid search on training data
    grid_search.fit(train_features, train_rts)
    # # Best parameters and best score
    # best_params = grid_search.best_params_
    # best_score = grid_search.best_score_
    joblib.dump(grid_search.best_estimator_, filepath)

    # fitted model with the best parameters
    return grid_search.best_estimator_

def train_gradient_booster(train_features: np.ndarray, 
                           train_rts: np.ndarray,
                           filepath: str,
                           n_estimators: List[int], 
                           learning_rate: List[int | float], 
                           min_samples_split: List[int | float],
                           min_samples_leaf: List[int | float], 
                           max_depth: List[int], 
                           max_features: List[int | str | None],
                           ) -> Any :
    
    """
    build a gradient booster model:

    - [input] (train_features, train_rts, n_estimators, 
                learning_rate, min_samples_split, min_samples_leaf,
                max_depth, max_features, filepath)
    - [output] trained gradient booster model

    Parameters
    ----------
    train_features : numpy.ndarray(dtype=numpy.uint8)
        train feature data, N-bit RDKit fingerprints x m_rows
    train_rts : numpy.ndarray(dtype=np.float64)
        retention times
    filepath: str
        the filepath where you want the trained model instance saved

    Lists of values used for grid search
    ________
    n_estimators : list
        list of integers that represent the number of trees in the forest model
    learning_rate : list
        list of integers or floats that represent the contribution of each tree
    min_samples_split: list
        list of integers or floats that represent the minimum  number of samples required to split an internal node (value cannot be greater than number of samples)
    min_samples_leaf: list
        list of integers or floats that represent the minimum number of samples required to be at a leaf node 
    max_depth : list
        list of integers that represent the max depth of the tree
    max_features : list
        list of integers, floats or any of three {“sqrt”, “log2”, None} that represent the number of features to consider when looking for the best split

    Returns
    -------
    model : ``gradient_boosting_regressor.Model``
        best model from grid search

    """
    #create gradient booster object
    model = GradientBoostingRegressor()
    #create dictionary of parameters to search
    parameter_grid = {
        'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth,
        'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
        'max_features': max_features
    }
    #create grid search object to find hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=parameter_grid, scoring='neg_mean_squared_error')
    #perform grid search on training data
    grid_search.fit(train_features, train_rts)
    # best_params = grid_search.best_params_
    # best_score = grid_search.best_score_
    joblib.dump(grid_search.best_estimator_, filepath)

    # fitted model with the best parameters
    return grid_search.best_estimator_

def train_SVR(train_features: np.ndarray, 
              train_rts: np.ndarray,
              filepath: str,
              c: List[int | float], 
              gamma: List[ str | float]) -> Any :
    
    """
    build a SVR model:
    - [input] (train_features, train_rts, c, epsilon, filepath)
    - [output] trained LinearSVR model

    Parameters
    ----------
    train_features : numpy.ndarray(dtype=numpy.uint8)
        train feature data, N-bit RDKit fingerprints x m_rows
    train_rts : numpy.ndarray(dtype=np.float64)
        retention times
    filepath: str
        the filepath where you want the trained model instance saved

    Lists of values used for grid search
    ________
    c : list
        list of integers or floats that represent the regularization parameter
        ** must be positive values **
    gamma : list
        list of floats, 'scale' or 'auto' that defines influence of single training example

    Returns
    -------
    model : ``SVR.Model``
            best model from grid search

    """
    #create LinearSVR object
    model = SVR()
    #create dictionary of parameters to search
    parameter_grid = {
        'C': c, 'gamma': gamma
    }
    #create grid search object to find hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=parameter_grid, scoring='neg_mean_squared_error')
    #perform grid search on training data
    grid_search.fit(train_features, train_rts)
    # best_params = grid_search.best_params_
    # best_score = grid_search.best_score_
    joblib.dump(grid_search.best_estimator_, filepath)

    # fitted model with the best parameters
    return grid_search.best_estimator_

def train_model(model: Any,
                train_features: np.ndarray, 
                train_rts: np.ndarray,
                filepath: str
                )-> Any :
        """
    If the  model was not trained on the entire dataset -> train the model:
    - [input] (train_features, train_rts, filepath)

    Parameters
    ----------
    model : The model with parameters included from previous grid search
    train_features : numpy.ndarray(dtype=numpy.uint8)
        train feature data, N-bit RDKit fingerprints x m_rows
    train_rts : numpy.ndarray(dtype=np.float64)
        retention times
    filepath: str
        the filepath where you want the trained model instance saved

    """
        #fit the model with the data
        trained_model = model.fit(train_features, train_rts)

        #save a trained model instance
        joblib.dump(trained_model, filepath)

        # fitted model with the best parameters
        return trained_model

def model_predictions(model, test_features):
    """ 
    predict retention time values:

    - [input] (model, test_features)
    - [output] predicted retention time values

     Parameters
    ----------
    model : Any
        trained sklearn model
    test_features : numpy.ndarray(dtype=numpy.uint8)
        test feature data, N-bit RDKit fingerprints x m_rows
    """
    return model.predict(test_features)

def metrics(rt_pred: np.ndarray, 
            test_rts: np.ndarray
            ) -> Tuple[float, float, float] :
    """
    Evaluate model:

    - [input] (rt_preds, test_rts)
    - [output] mean absolute error, mean squared error and log loss, Tuple[float, float, float]

    Parameters
    ----------
    rt_pred : np.ndarray
        predicted retention time values from model_predictions(model, test_features
    test_rts : np.ndarray
        test retention time values
    """
    mae =  mean_absolute_error(test_rts, rt_pred)
    rmse = np.sqrt(mean_squared_error(test_rts, rt_pred))

    return "MAE", mae, "RMSE",  rmse

def plots(test_rts: np.ndarray,
          rt_pred: np.ndarray
          ) -> plt:
    
        """
    create plot to visualize model prediction:

    - [input] (test_rts, rt_pred)
    - [output] plotly scatter plot

    Parameters
    ----------
    test_rts : np.ndarray
        test retention time values
    rt_pred : np.ndarray
        predicted retention time values from model_predictions(model, test_features
    
    """
        rmse = np.sqrt(mean_squared_error(test_rts, rt_pred))
        plt.scatter(test_rts, rt_pred, color='blue', alpha=0.7)  # Set color and transparency
        plt.plot(test_rts, test_rts, color='red', linestyle='--', linewidth=2, label='Ideal Line')

        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values (RMSE: {:.2f})'.format(rmse))
        
        return plt.show()

