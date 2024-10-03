"""
    unit tests for keras_models.py module
"""

import unittest
from unittest import TestCase 

import numpy as np

from rtp.data import assemble_dataset_arrays, split_train_test_data
from rtp.keras_models import (
    build_linear_model
)
from rtp.test.data import IDPP_DB_PATH

# define all variables needed for testing keras_models.py
FP_SIZE = 256
LABELS, FEATURES, RTS = assemble_dataset_arrays([544], 
                                                IDPP_DB_PATH, 
                                                FP_SIZE)
TRAIN_IDX, TEST_IDX = split_train_test_data(RTS)
TRAIN_LABELS, TRAIN_FEATURES, TRAIN_RTS = LABELS[TRAIN_IDX], FEATURES[TRAIN_IDX], RTS[TRAIN_IDX]
TEST_LABELS, TEST_FEATURES, TEST_RTS = LABELS[TEST_IDX], FEATURES[TEST_IDX], RTS[TEST_IDX]

class TestBuildLinearModel(TestCase):
    """ unit testing on keras linear model """
    def setUp(self):
        """ parameters used for grid search, these values can be changed based on users preference """
        self.model_keras = build_linear_model(FP_SIZE)
        self.hist = self.model_keras.fit(TRAIN_FEATURES, 
                       TRAIN_RTS, 
                       validation_data=(TEST_FEATURES, TEST_RTS),
                       batch_size=256, 
                       epochs=128,
                       verbose=2)


    def test_BLM_evaluation_acceptable(self):
        """ Tests to see if evaluations are in acceptable range"""
        loss, mse, mae = self.model_keras.evaluate(TEST_FEATURES, TEST_RTS)
        self.assertGreater(0.6, loss, 
                           "Validation loss is too high")
        self.assertGreater(0.5, mse, 
                           "Mean squared error is too high")
        self.assertGreater(0.5, mae, 
                           "Mean absolute error is too high")
    
    def test_BLM_model_fit(self):
        """ Tests for overfitting or underfitting"""
        train_loss = self.hist.history['loss']
        val_loss = self.hist.history['val_loss']        
        self.assertLess(train_loss[-1], val_loss[-1], 
                        "Model is overfitting")
        self.assertGreater(train_loss[0], val_loss[0], 
                           "Model is underfitting")

    def test_BLM_negative_predictions(self):
        """ Tests for negative predicted rt values"""
        predictions = self.model_keras.predict(TEST_FEATURES)
        self.assertGreaterEqual(np.min(predictions), 0, 
                                "Negative predicted results found")

    def test_BLM_out_of_bounds_prediction(self):
        """ Tests for predicted rt values over the highest value seen in the test data"""    
        highest_rt_value = np.max(TEST_RTS)
        predictions = self.model_keras.predict(TEST_FEATURES)
        self.assertLessEqual(np.max(predictions), highest_rt_value, 
                             "Predicted results above the highest retention time value"   )
  

if __name__ == "__main__":
    unittest.main(verbosity=2)

