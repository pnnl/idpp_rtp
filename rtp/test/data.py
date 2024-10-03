"""
    unit tests for data.py module
"""


import unittest
import pickle
from tempfile import TemporaryDirectory
import os

import numpy as np

from rtp.data import (
    FeatureSet,
    _build_dataset_selection_query,
    _select_dataset_from_idppdb,
    _pack_ids_into_label,
    _featurize,
    get_features,
    assemble_dataset_arrays,
    _get_binned_rts,
    split_train_test_data,
    center_and_scale
)
from rtp.test import TEST_INCLUDE_DIR


# path to built in test IdPP database file
# used for a few tests
IDPP_DB_PATH = os.path.join(TEST_INCLUDE_DIR, "test_idpp.db")


class Test_TestIdppDbExists(unittest.TestCase):
    """ ensure that the built in test database file exists """

    def test_db_file_exists(self):
        """ ensure that the built in test database file exists """
        self.assertTrue(os.path.isfile(IDPP_DB_PATH))


class Test_BuildDataSelectionQuery(unittest.TestCase):
    """ tests for the _build_dataset_selection_query function """

    def test_BDSQ_empty_src_ids(self):
        """ if src_ids is an empty list this raises a ValueError """
        with self.assertRaises(ValueError,
                               msg="empty src_ids list should raise a ValueError"):
            qry = _build_dataset_selection_query([])
    
    def test_BDSQ_single_src_id(self):
        """ generate a working query with a single src_id in src_ids list """
        qry = _build_dataset_selection_query([1])
        pat = r".+AND \(\s+src_id=1\s+\).+"
        self.assertRegex(qry, pat,
                         msg="query did not contain the expected conditional")

    def test_BDSQ_two_src_ids(self):
        """ generate a working query with two src_ids in src_ids list """
        qry = _build_dataset_selection_query([1, 2])
        pat = r".+AND \(\s+src_id=1 OR src_id=2\s+\).+"
        self.assertRegex(qry, pat,
                         msg="query did not contain the expected conditional")

    def test_BDSQ_three_src_ids(self):
        """ generate a working query with three src_ids in src_ids list """
        qry = _build_dataset_selection_query([1, 2, 3])
        pat = r".+AND \(\s+src_id=1 OR src_id=2 OR src_id=3\s+\).+"
        self.assertRegex(qry, pat,
                         msg="query did not contain the expected conditional")


class Test_SelectDatasetFromIdppdb(unittest.TestCase):
    """ tests for the _select_dataset_from_idppdb function """

    def test_SDFI_bad_db_path(self):
        """ if the path to idpp.db database file is invalid this should raise a ValueError """
        with self.assertRaises(ValueError,
                               msg="bad database file path should raise ValueError"):
            qry = "not important for this test"
            ids, smis, rts = _select_dataset_from_idppdb("bad/path/to/idpp.db", qry)

    def test_SDFI_bad_src_id(self):
        """ if the src_id in the selection query is invalid then this will return empty lists """                                                         
        # the query is fine in form but there are no entries with src_id=999
        # so it will return no rows
        qry = _build_dataset_selection_query([999])
        ids, smis, rts = _select_dataset_from_idppdb(IDPP_DB_PATH, qry)
        self.assertEqual(ids, [],
                         msg="ids should be empty list")
        self.assertEqual(smis, [],
                         msg="smis should be empty list")
        self.assertEqual(rts, [],
                         msg="rts should be empty list")

    def test_SDFI_noerr(self):
        """ extract a dataset and make sure the outputs look as expected """
        qry = _build_dataset_selection_query([554])
        # modify the query to only return 10 rows
        qry = qry.rstrip()[:-1] + " LIMIT 10;"
        ids, smis, rts = _select_dataset_from_idppdb(IDPP_DB_PATH, qry)
        self.assertEqual(len(ids), 10,
                         msg="ids should have 10 rows")
        for row in ids:
            self.assertAlmostEqual(len(row), 4,
                                   msg="each row in ids should have 4 values")
        self.assertEqual(len(smis), 10,
                         msg="smis should have 10 rows")
        self.assertEqual(len(rts), 10,
                         msg="rts should have 10 rows")


class Test_PackIdsIntoLabel(unittest.TestCase):
    """ tests for _pack_ids_into_label function """

    def test_PIIL_none_ids(self):
        """ non-integer IDs cause AssertionErrors """
        for ids in [(None, 1, 1, 1),
                    (1, None, 1, 1),
                    (1, 1, None, 1),
                    (1, 1, 1, None)]:
            with self.assertRaises(AssertionError,
                                   msg="any empty ID should cause an AssertionError"):
                lbl = _pack_ids_into_label(*ids)

    def test_PIIL_correct_labels(self):
        """ generate some labels and make sure they look as expected """
        for ids, exp_lbl in [((1, 2, 3, 4), "c1_s2_a3_r4"),
                             ((1, 22, 333, 4444), "c1_s22_a333_r4444"),
                             ((4444, 333, 22, 1), "c4444_s333_a22_r1")]:
            lbl = _pack_ids_into_label(*ids)
            self.assertEqual(exp_lbl, lbl,
                             msg=f"expected label: {exp_lbl} from ids: {ids}, got label: {lbl}")


class Test_Featurize(unittest.TestCase):
    """ tests for the _featurize function """

    def test_F_bad_smi(self):
        """ bad SMILES structure should return None """
        # test with molecular fingerprint feature set
        self.assertIsNone(_featurize("not a good SMILES structure", FeatureSet.FP, fp_size=1024),
                            msg="bad SMILES structure should return None")
        # test with MQN feature set
        self.assertIsNone(_featurize("not a good SMILES structure", FeatureSet.MQN),
                            msg="bad SMILES structure should return None")

    def test_F_good_smi(self):
        """ a good SMILES structure should produce a features array """
        # test with molecular fingerprint feature set
        features = _featurize("ICPOOP", FeatureSet.FP, fp_size=1024)
        self.assertEqual(features.shape, (1024,),
                         msg="molecular fingerprint should make a feature array with 1024 elements")
        self.assertEqual(features.dtype, np.uint8,
                         msg="molecular fingerprint feature set should have dtype=np.uint8")
        # test with MQN feature set
        features = _featurize("ICPOOP", FeatureSet.MQN)
        self.assertEqual(features.shape, (42,),
                         msg="MQNs should make a feature array with 42 elements")
        self.assertEqual(features.dtype, np.float32,
                         msg="molecular fingerprint feature set should have dtype=np.float32")
        
    def test_F_no_fp_size(self):
        """ raise a ValueError if fp_size kwarg is not set but FeatureSet.FP is specified """
        with self.assertRaises(ValueError):
            features = _featurize("ICPOOP", FeatureSet.FP)

    def test_F_unrecognized_feature_set(self):
        """ unrecognized feature set param should raise a ValueError """
        with self.assertRaises(ValueError):
            features = _featurize("ICPOOP", "not a valid feature set")


class TestGetFeatures(unittest.TestCase):
    """ tests for the get_features function """

    def test_GF_all_good_smiles(self):
        """ test input with all good SMILES structures """
        smis = [
            "ICPOOP",
            "CCCP",
            "NONONO"
        ]
        # test with molecular fingerprint feature set
        dataset, smiles_none = get_features(smis, FeatureSet.FP, fp_size=512)
        self.assertEqual(dataset.shape, (3, 512),
                         "wrong dataset shape")
        self.assertListEqual(smiles_none, [],
                             "no SMILES structures should be skipped")
        # test with MQN feature set
        dataset, smiles_none = get_features(smis, FeatureSet.MQN)
        self.assertEqual(dataset.shape, (3, 42),
                         "wrong dataset shape")
        self.assertListEqual(smiles_none, [],
                             "no SMILES structures should be skipped")

    def test_GF_some_bad_smiles(self):
        """ test input with some good and some bad SMILES structures """
        smis = [
            "ICPOOP",
            "this one is bad",
            "CCCP",
            "and this one too",
            "NONONO"
        ]
        # test with molecular fingerprint feature set
        dataset, smiles_none = get_features(smis, FeatureSet.FP, fp_size=512)
        self.assertEqual(dataset.shape, (3, 512),
                         "wrong dataset shape")
        self.assertListEqual(smiles_none, 
                             ["this one is bad", "and this one too"],
                             "should have skipped two of the SMILES structures")
        # test with MQN feature set
        dataset, smiles_none = get_features(smis, FeatureSet.MQN)
        self.assertEqual(dataset.shape, (3, 42),
                         "wrong dataset shape")
        self.assertListEqual(smiles_none, 
                             ["this one is bad", "and this one too"],
                             "should have skipped two of the SMILES structures")
    
    def test_GF_all_bad_smiles(self):
        """ test input with all bad SMILES structures """
        smis = [
            "this one is bad",
            "and this one too",
        ]
        # test with molecular fingerprint feature set
        dataset, smiles_none = get_features(smis, FeatureSet.FP, fp_size=512)
        self.assertEqual(dataset.shape, (0,),
                         "wrong dataset shape")
        self.assertListEqual(smiles_none, 
                             ["this one is bad", "and this one too"],
                             "should have skipped two of the SMILES structures")
        # test with MQN feature set
        dataset, smiles_none = get_features(smis, FeatureSet.MQN)
        self.assertEqual(dataset.shape, (0,),
                         "wrong dataset shape")
        self.assertListEqual(smiles_none, 
                             ["this one is bad", "and this one too"],
                             "should have skipped two of the SMILES structures")


class TestAssembleDatasetArrays(unittest.TestCase):
    """ tests for the assemble_dataset_arrays function """

    def test_ADA_dataset_shapes(self):
        """ assemble a dataset and ensure arrays have expected shapes """
        labels, features, rts = assemble_dataset_arrays([554], IDPP_DB_PATH, FeatureSet.FP, fp_size=1024)
        rows = 898
        self.assertEqual(labels.shape, (rows,),
                         msg=f"labels should have {rows} rows")
        self.assertEqual(features.shape, (rows, 1024),
                         msg=f"features should have {rows} rows with 1024 columns")
        self.assertEqual(rts.shape, (rows,),
                         msg=f"rts should have {rows} rows")
        

class Test_GetBinnedRts(unittest.TestCase):
    """ tests for the _get_binned_rts function """

    def test_GBR_size_of_bin_categories(self):
        """ ensure binning range 0 to 99 results in 5 20-bin categories """
        rts_binned = _get_binned_rts(np.arange(100))
        for i in range(5):
            self.assertEqual(len(rts_binned[rts_binned == i]), 20,
                             msg=f"category {i} should have 20 members")
        

class TestSplitTrainTestData(unittest.TestCase):
    """ tests for the split_train_test_data function """

    def test_STTD_split_sizes(self):
        """ make sure the training and test set splits are the expected sizes """
        rts_binned = _get_binned_rts(np.arange(100))
        train_idx, test_idx = split_train_test_data(rts_binned)
        train_data, test_data = rts_binned[train_idx], rts_binned[test_idx]
        self.assertEqual(len(train_data), 80, 
                         msg="training set split should have 80 members")
        self.assertEqual(len(test_data), 20, 
                         msg="test set split should have 20 members")


class TestCenterAndScale(unittest.TestCase):
    """ tests for the center_and_scale function """

    def test_CAS_dtype_is_preserved(self):
        """ make sure the datatype is preserved from input to transformed data """
        smis = [
            "ICPOOP",
            "CCCP",
            "NONONO"
        ]
        # test with MQN feature set
        dataset, _ = get_features(smis, FeatureSet.MQN)
        self.assertEqual(dataset.dtype, np.float32)
        scaled, _ = center_and_scale(dataset)
        self.assertEqual(scaled.dtype, np.float32)

    def test_CAS_save_load_scaler(self):
        """ save scaler to file and load it back again, make sure it produces same output """
        with TemporaryDirectory() as tmpdir:
            smis = [
                "ICPOOP",
                "CCCP",
                "NONONO"
            ]
            # test with MQN feature set
            dataset, _ = get_features(smis, FeatureSet.MQN)
            scaled, scaler = center_and_scale(dataset)
            # dump the scaler instance to pickle file then load it back again
            pkl = os.path.join(tmpdir, "scaler.pkl")
            with open(pkl, "wb") as pf:
                pickle.dump(scaler, pf)
            with open(pkl, "rb") as pf:
                scaler2 = pickle.load(pf)
            scaled2 = scaler2.transform(dataset)
            # make sure the output from the original scaler instance matches the
            # output from the saved and reloaded scaler instance
            self.assertEqual(scaled.shape, scaled2.shape,
                             ("output shapes from original and saved/reloaded scaler"
                              " should be the same"))
            # check each value
            for row, row2 in zip(scaled, scaled2):
                for val, val2 in zip(row, row2):
                    self.assertAlmostEqual(val, val2,
                                           ("output from original and saved/reloaded "
                                            "scaler should be the same"))
            

if __name__ == "__main__":
    unittest.main(verbosity=2)
