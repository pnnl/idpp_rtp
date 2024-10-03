"""
    unit tests for sklearn_models.py module
"""

import unittest
from unittest import TestCase 

import numpy as np

from rtp.data import assemble_dataset_arrays, split_train_test_data
from rtp.sklearn_models import (train_random_forest, train_gradient_booster, train_SVR, metrics, model_predictions)
from rtp.test.data import IDPP_DB_PATH
from rtp.data import _select_dataset_from_idppdb

# define all variables needed for testing sklearn_models.py
FP_SIZE = 1024
LABELS, FEATURES, RTS = assemble_dataset_arrays([544], 
                                                IDPP_DB_PATH, 
                                                FP_SIZE)
    
# split into training/test sets
TRAIN_IDX, TEST_IDX = split_train_test_data(RTS)
TRAIN_LABELS, TRAIN_FEATURES, TRAIN_RTS = LABELS[TRAIN_IDX], FEATURES[TRAIN_IDX], RTS[TRAIN_IDX]
TEST_LABELS, TEST_FEATURES, TEST_RTS = LABELS[TEST_IDX], FEATURES[TEST_IDX], RTS[TEST_IDX]

#filepath for saving model instance
FILEPATH = "./trained_RT_model"
class TestTrainRandomForest(TestCase):
    
    """ tests for the train_random_forest function """


    def setUp(self):
        """ parameters used for grid search, these values can be changed based on users preference """
        self.n_estimators = [10, 50, 100, 200]
        self.max_depth = [None, 10, 20, 30]
        self.min_samples_split = [2, 5, 10]
        self.min_samples_leaf = [1, 2, 4]
        self.max_features = [1, 5, 10]       

        """ initializing model """
        self.trained_model = train_random_forest(TRAIN_FEATURES, TRAIN_RTS, FILEPATH,
                                                 self.n_estimators, self.max_depth, self.min_samples_split,
                                                 self.min_samples_leaf, self.max_features)

    def test_TRF_valid_input(self):
        """ Test to see if all of the input types are valid """
        self.assertIsInstance(TRAIN_FEATURES, np.ndarray, "X_train should be a 2D array")
        self.assertIsInstance(TRAIN_RTS, np.ndarray, "y_train should be a 1D array")
        self.assertTrue(all(isinstance(lst, list) for lst in [ 
                self.n_estimators, self.max_depth, self.min_samples_split,
                self.min_samples_leaf, self.max_features]), "All lists are type list")
 
    def test_TRF_non_null(self):
        """ test if the resulting model has any non-null parameters"""
        self.assertTrue(hasattr(self.trained_model, 'estimators_'))
        self.assertTrue(hasattr(self.trained_model, 'feature_importances_'))

class TestTrainGradientBooster(TestCase):
    """ tests for the train_gradient_booster function """
    
    def setUp(self):
        """ parameters used for grid search, these values can be changed based on users preference """
        self.n_estimators = [10, 50, 100, 200]
        self.learning_rate = [.005, .05, .025, 1]
        self.min_samples_split = [2, 5, 10]
        self.min_samples_leaf = [1, 2, 4]
        self.max_depth = [None, 10, 20, 30]
        self.max_features = [1, 5, 10]       

        """ initializing model """
        self.trained_model = train_gradient_booster(TRAIN_FEATURES,TRAIN_RTS, FILEPATH,
                                                    self.n_estimators, self.learning_rate, self.min_samples_split,
                                                    self.min_samples_leaf, self.max_depth, self.max_features)
     
                                       
    def test_TGB_valid_input(self):
        """ Test to see if all of the input types are valid """
        self.assertIsInstance(TRAIN_FEATURES, np.ndarray, "X_train should be a 2D array")
        self.assertIsInstance(TRAIN_RTS, np.ndarray, "y_train should be a 1D list")
        self.assertTrue(all(isinstance(lst, list) for lst in [ 
                self.n_estimators, self.learning_rate, self.min_samples_split,
                self.min_samples_leaf, self.max_depth, self.max_features]), "All lists are type list")

    def test_TGB_non_null(self):
        """ test if the resulting model has any non-null parameters"""
        self.assertTrue(hasattr(self.trained_model, 'estimators_'))
        self.assertTrue(hasattr(self.trained_model, 'feature_importances_'))

class TestTrainSVR(TestCase):
    """ tests for the train_SVR function """
    
    def setUp(self):
        """ parameters used for grid search, these values can be changed based on users preference """
        self.c= [.001, .01, .1, 1, 10, 100, 1000], 
        self.gamma= ['scale', 'auto', .0001, .001, .01, .1, 1, 10, 100]

        """ initializing model """
        self.trained_model = train_SVR(TRAIN_FEATURES, TRAIN_RTS, FILEPATH, 
                                       self.c, self.gamma)

    def test_SVR_valid_input(self):
        """ Test to see if all of the input types are valid """
        self.assertIsInstance(TRAIN_FEATURES, np.ndarray, "X_train should be a 2D array")
        self.assertIsInstance(TRAIN_RTS, np.ndarray, "y_train should be a 1D list")
        self.assertTrue(all(isinstance(lst, list) for lst in [ 
                self.c, self.gamma]), "All lists are type list")

    def test_SVR_non_null(self):
        """ test if the resulting model has any non-null parameters"""
        self.assertTrue(hasattr(self.trained_model, 'estimators_'))
        self.assertTrue(hasattr(self.trained_model, 'feature_importances_'))

class TestModelPredictions(TestCase):
    """ tests for the model_predictions function """

    def setUp(self):
        """ Using the random forest model and parameters to test """
        """ parameters used for grid search, these values can be changed based on users preference """
        self.n_estimators = [10, 50, 100, 200]
        self.learning_rate = [.005, .05, .025, 1]
        self.min_samples_split = [2, 5, 10]
        self.min_samples_leaf = [1, 2, 4]
        self.max_depth = [None, 10, 20, 30]
        self.max_features = [1, 5, 10]   

        """ initializing random forest model """
        self.model_rf = train_random_forest(TRAIN_FEATURES, TRAIN_RTS, FILEPATH,
                                                 self.n_estimators, self.max_depth, self.min_samples_split,
                                                 self.min_samples_leaf, self.max_features)
        
        """ initiallizing model_prediction function """
        self.predictions = model_predictions(self.model_rf, TEST_FEATURES)

        """ identify the highest rt value """
        self.highest_value = np.max(TEST_RTS)
    
    def test_MP_data_types(self):
        """ Test to see if the arrays have data of the correct type """
        self.assertIsInstance(self.predictions, np.ndarray, "Predictions should be an array")
        self.assertIsInstance(TEST_FEATURES, np.ndarray, "X_test should be an array")
        self.assertIsInstance(TEST_RTS, np.ndarray, "y_test should be an array")

    def test_MP_predictions_negative(self):
        """ Test for negative predictions """
        self.assertFalse(np.any(self.predictions < 0), "A negative prediction exists")

    def test_MP_predictions_out_of_bounds(self):
        """ Test for predictions that are higher than the highest value in y_test """
        self.assertFalse(np.any(self.predictions > self.highest_value), "A prediction exists that is higher than the highest value in y_test")

    def test_MP_overfitting(self):
        """ Test to see if the model is overfitting """
        train_score = self.model_rf.score(TRAIN_FEATURES, TRAIN_RTS)
        test_score = self.model_rf.score(TEST_FEATURES, TEST_RTS)
        self.assertGreaterEqual(test_score, train_score, "Model is overfitting")


class TestMetrics(TestCase):
    """ tests for the metrics function """

    def setUp(self):
        """ Using the random forest model and parameters to test """
        """ parameters used for grid search, these values can be changed based on users preference """
        self.n_estimators = [10, 50, 100, 200]
        self.learning_rate = [.005, .05, .025, 1]
        self.min_samples_split = [2, 5, 10]
        self.min_samples_leaf = [1, 2, 4]
        self.max_depth = [None, 10, 20, 30]
        self.max_features = [1, 5, 10]   

        """ initializing random forest model """
        self.model_rf = train_random_forest(TRAIN_FEATURES, TRAIN_RTS, FILEPATH,
                                                 self.n_estimators, self.max_depth, self.min_samples_split,
                                                 self.min_samples_leaf, self.max_features)
        
        """ initiallizing model_prediction function """
        self.predictions = model_predictions(self.model_rf, TEST_FEATURES)
    
    def test_M_evaluation_acceptable(self):
        """ Test to see if the model is resulting in acceptable values """
        plot, mae, mse, accuracy = metrics(self.predictions, TEST_FEATURES, TEST_RTS)
        mse_threshold = 0.1
        mae_threshold = 0.2
        accuracy_threshold = 0.7
        # Check if MSE, MAE, and accuracy are within acceptable ranges
        self.assertTrue(mse < mse_threshold, f"MSE ({mse}) is not within acceptable range ({mse_threshold})")
        self.assertTrue(mae < mae_threshold, f"MAE ({mae}) is not within acceptable range ({mae_threshold})")
        self.assertTrue(accuracy < accuracy_threshold, f"Accuracy ({accuracy}) is not within acceptable range ({accuracy_threshold})")


if __name__ == "__main__":
    unittest.main(verbosity=2)

