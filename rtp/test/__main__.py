"""
    run all unittests from subpackage
"""


import unittest


from rtp.test.data import (
    Test_TestIdppDbExists,
    Test_BuildDataSelectionQuery, 
    Test_SelectDatasetFromIdppdb, 
    Test_PackIdsIntoLabel, 
    Test_Featurize, 
    TestGetFeatures,
    TestAssembleDatasetArrays, 
    Test_GetBinnedRts,
    TestSplitTrainTestData,
    TestCenterAndScale
)
from rtp.test.keras_models import (
    TestBuildLinearModel
)

from rtp.test.sklearn_models import (
    TestTrainRandomForest, 
    TestTrainGradientBooster, 
    TestTrainSVR,
    TestModelPredictions, 
    TestMetrics
)


if __name__ == "__main__":
    unittest.main(verbosity=2)


