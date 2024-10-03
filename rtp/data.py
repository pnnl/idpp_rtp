"""
    data utilities
"""


from typing import Any, List, Tuple, Optional
from sqlite3 import connect
import os
import enum

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


# define an enum class that can be used to select feature set
# NOTE: The choice to use an enum here instead of, for instance,
#       defining a couple of string values to specify the different
#       feature sets is because that would involve a lot more
#       annoying boilerplate for validating those 
#       specifiers. With an enum we can just use a nice and clean
#       match/case statement which is clear and expressive, plus
#       the enum is a defined class with explicit allowed states that
#       the language server knows about, so there is no potential for
#       confusion on what string am I supposed to use for a given
#       feature set when writing code that needs to consider the 
#       different allowed options
class FeatureSet(enum.Enum):
    # molecular fingerprints
    FP = enum.auto()
    # molecular quantum numbers
    MQN = enum.auto()


_QRY_TEMPLATE = """
SELECT 
    Compounds.cmpd_id, 
    Smiles.smi_id, 
    Adducts.adduct_id, 
    RTs.rt_id, 
    Smiles.smi, 
    RTs.rt
FROM 
    Compounds 
    JOIN 
        Smiles 
        USING(smi_id) 
    JOIN 
        Adducts 
        USING(cmpd_id) 
    JOIN 
        RTs 
        USING(adduct_id)
WHERE 
    smi_id > 0 
    AND (
            {src_id_selection}
        ) 
ORDER BY 
    rt;
"""


def _build_dataset_selection_query(src_ids: List[int]
                                   ) -> str :
    """
    Builds a query to select desired dataset(s) from IdPPdb
    
    Parameters
    ----------
    src_ids : ``list(int)``
        list of src_ids for sources to include in dataset
    
    Returns
    -------
    query : ``str``
        generated query to select specified dataset(s)
    """
    if len(src_ids) < 1:
        msg = "src_ids should be a list of src_ids with at least 1 element"
        raise ValueError(msg)
    src_id_selection = " OR ".join([f"src_id={sid}" for sid in src_ids])
    return _QRY_TEMPLATE.format(src_id_selection=src_id_selection)


def _select_dataset_from_idppdb(path_to_idpp_db: str, 
                                query: str
                                ) -> Tuple[List[Tuple[int]], List[str], List[float]] :
    """
    Selects a RT dataset from the IdPPdb

    Parameters
    ----------
    path_to_idpp_db : ``str``
        path to idpp.db database file
    query : ``str``
        dataset selection query to run

    Returns
    -------
    ids : ``list(tuple(int))``
        tuples of the different IDs for each row in dataset (cmpd_id, smi_id, adduct_id, rt_id)
    smis : ``list(str)``
        list of SMILES structures for each row in dataset
    rts : ``list(float)``
        list of retention times for each row in dataset
    """
    # ensure valid db path
    if not os.path.isfile(path_to_idpp_db):
        msg = f"IdPPdb database file {path_to_idpp_db} not found"
        raise ValueError(msg)
    # connect to db
    con = connect(path_to_idpp_db)
    cur = con.cursor()
    # collect data
    ids, smis, rts = [], [], []
    for *_ids, smi, rt in cur.execute(query):
        ids.append(_ids)
        smis.append(smi)
        rts.append(rt)
    # cleanup
    con.close()
    # return
    return ids, smis, rts


def _pack_ids_into_label(cmpd_id: int, 
                         smi_id: int,
                         adduct_id: int,
                         rt_id: int
                         ) -> str :
    """
    pack the collection of identifiers for each row into a single convenient string label

    Parameters
    ----------
    cmpd_id : ``int``
    smi_id : ``int``
    adduct_id : ``int``
    rt_id : ``int``
        row identifiers
    
    Returns
    -------
    label : ``str``
        ids packed into a single string
    """
    # assert all of the IDs are ints, so do not allow any empties
    assert type(cmpd_id) is int
    assert type(smi_id) is int
    assert type(adduct_id) is int
    assert type(rt_id) is int
    return f"c{cmpd_id}_s{smi_id}_a{adduct_id}_r{rt_id}"


def _featurize(smi: str,
               feature_set: FeatureSet,
               fp_size: Optional[int] = None,
               ) -> npt.NDArray[Any] :
    """
    convert a SMILES structure into features (specified by the FeatureSet enum variant
    provided as a parameter)
    
    - FeatureSet.FP -> a vector of N bits (as np.uint8) using RDKit fingerprint, requires
        the fp_size kwarg to be set
    - FeatureSet.MQN -> a set of 42 molecular descriptors based on the molecular graph structure

    Parameters
    ----------
    smi : ``str``
        SMILES structure
    feature_set : ``FeatureSet``
        specify the feature set to use
    fp_size : ``int``, optional
        size of fingerprint (bits), required if using FetureSet.FP

    Returns
    -------
    feat_vec : ``numpy.ndarray(?)`` or ``None``
        feature vector, or `None` if unable to featurize

        - FeatureSet.FP -> numpy.ndarray(numpy.uint8)
        - FeatureSet.MQN -> numpy.ndarray(numpy.float32)
    """
    match feature_set:
        case FeatureSet.FP:
            # require fp_size
            if fp_size is None:
                raise ValueError(f"feature set {feature_set} requires fp_size to be set")
            try:
                # turn off rdkit logging messages
                RDLogger.DisableLog('rdApp.*')  
                return np.array(RDKFingerprint(Chem.MolFromSmiles(smi), fpSize=fp_size).ToList(), 
                                dtype=np.uint8)
            except:
                # TODO: Suggest avoiding bare excepts, it is preferable to handle specific errors 
                return None
        case FeatureSet.MQN:
            try:
                # turn off rdkit logging messages
                RDLogger.DisableLog('rdApp.*')  
                # NOTE: I think all of these features are ints so we could just use dtype np.int32, 
                #       however, I have elected to cast them to float32 because they will need to 
                #       be centered and scaled before being passed to a ML model for training which 
                #       will inherently convert them to floats anyways
                return np.array(Descriptors.rdMolDescriptors.MQNs_(Chem.MolFromSmiles(smi)), 
                                dtype=np.float32)
            except:
                # TODO: Suggest avoiding bare excepts, it is preferable to handle specific errors 
                return None
        case _:
            # any other unrecognized value of feature_set param raises a ValueError
            raise ValueError(f"unrecognized feature set: {feature_set}")
    

def get_features(smis: List[str],
                 feature_set: FeatureSet,
                 fp_size: Optional[int] = None,
                 ) -> Tuple[npt.NDArray[Any], List[str]] :
    """
    produce features for a list of SMILES structures using a specified feature set
    
    no need to deal with IDs or RTs since this function is meant to support doing 
    inference with trained models without using the IdPP database

    Parameters
    ----------
    smis : ``list(int)``
        list of input SMILES structures
    feature_set : ``FeatureSet``
        specify the feature set to use
    fp_size : ``int``, optional
        size of fingerprint (bits), required if using FetureSet.FP

    Returns
    -------
    dataset : ``numpy.ndarray(dtype=numpy.uint8)
        (X) feature data

        - FeatureSet.FP -> N-bit RDKit fingerprints x M rows, numpy.ndarray(numpy.uint8)
        - FeatureSet.MQN -> 42 descriptors x M rows, numpy.ndarray(numpy.float32)

    smiles_none : ``list(str)``
        list SMILES structures for which features were not able to be produced
    """
    dataset = []
    smiles_none = []
    for smi in smis:
        features = _featurize(smi, feature_set, fp_size=fp_size)
        if features is not None:
            dataset.append(features)
        else:
            smiles_none.append(smi)
    return np.array(dataset), smiles_none


def assemble_dataset_arrays(src_ids: List[int],
                            path_to_idpp_db: str,
                            feature_set: FeatureSet,
                            fp_size: Optional[int] = None,
                            ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[np.float32]] :
    """
    Fetch dataset from database, create labels, featurize using specified feature set, then return the
    data as numpy arrays

    Parameters
    ----------
    src_ids : ``list(int)``
        list of src_ids for sources to include in the dataset
    path_to_idpp_db : ``str``
        path to idpp.db database file
    feature_set : ``FeatureSet``
        specify the feature set to use
    fp_size : ``int``, optional
        size of fingerprint (bits), required if using FetureSet.FP

    Returns
    -------
    labels : ``numpy.ndarray(dtype=str)``
        row labels with IDs
    features : ``numpy.ndarray(dtype=numpy.uint8)
        (X) feature data

        - FeatureSet.FP -> N-bit RDKit fingerprints x M rows, numpy.ndarray(numpy.uint8)
        - FeatureSet.MQN -> 42 descriptors x M rows, numpy.ndarray(numpy.float64)

    rts : ``numpy.ndarray(dtype=np.float32)
        (y) retention times
    """
    # build the query
    qry = _build_dataset_selection_query(src_ids)
    # fetch data from database and build up dataset
    labels, features, rts = [], [], []
    for ids, smi, rt in zip(*_select_dataset_from_idppdb(path_to_idpp_db, qry)):
        feats = _featurize(smi, feature_set, fp_size=fp_size)
        if feats is not None:
            labels.append(_pack_ids_into_label(*ids))
            features.append(feats)
            rts.append(rt)
    # NOTE: I down-cast the RT values to half-precision floats (np.float32) since the dynamic range of 
    #       retention time values is not huge and half-precision floats should be amply precise while
    #       cutting the size of the y data in half. This can be reverted here if needed in the future,
    #       just make sure to propagate that through all of the type annotations in other functions
    #       that deal with RT
    return np.array(labels, dtype=str), np.array(features), np.array(rts, dtype=np.float32)


def _get_binned_rts(rts: npt.NDArray[np.float32], 
                    ) -> npt.NDArray[np.intp] :
    """
    convert RTs from continuous values into binned categorical values
    binning is performed based on distribution of RTs, split at the 
    20th, 40th, 60th, and 80th percentiles

    Parameters
    ----------
    rts : ``numpy.ndarray(numpy.float32)``
        RT array from assembled dataset
    
    Returns
    -------
    binned : ``numpy.ndarray(numpy.int32)``
        categorical (binned) RT data
    """
    return np.digitize(rts, np.percentile(rts, [20, 40, 60, 80]))


def split_train_test_data(rts: npt.NDArray[np.float32], 
                          test_fraction: float = 0.2,
                          random_state: int = 420
                          ) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """
    Takes RT array from an assembled dataset then splits into training and test sets
    with coarse stratification on RT distribution, returning the indices of training/test sets
 
    .. code-block:: python3
        :caption: Usage Example

        labels, features, rts = assemble_dataset_arrays([270, 271], "idpp_db", FeatureSet.FP, fp_size=512)
        train_idx, test_idx = split_train_test_data(rts)
        train_labels, train_features, train_rts = labels[train_idx], features[train_idx], rts[train_idx]
        test_labels, test_features, test_rts = labels[test_idx], features[test_idx], rts[test_idx]

    
    Parameters
    ----------
    rts : ``numpy.ndarray(dtype=np.float32)``
        RT array from assembled dataset
    test_fraction : ``float``, default=0.2
        fraction of overall dataset to reserve as a test set, training set fraction = 1 - test_fraction
    random_state : ``int``, default=420
        pRNG seed for deterministic splitting
        
    Returns
    -------
    idx_train : ``numpy.ndarray(int)``
    idx_test : ``numpy.ndarray(int)``
        indices of training and test set data 
    """
    # bin RTs for stratified splitting
    binned_rts = _get_binned_rts(rts)
    # create the split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state)
    return next(sss.split(np.zeros(len(binned_rts)), binned_rts))


def center_and_scale(feature_data: npt.NDArray[np.float32],
                     ) -> Tuple[npt.NDArray[np.float32], StandardScaler] :
        """
        Centers and scales feature data such that all features have an average of 0 and variance of 1 
        
        returns the scaled feature data and the fitted StandardScaler instance, which can be used to 
        scale other feature data via the scaler.transform() method

        .. note::

            This function is meant to be used during model training to scale the training data, then 
            the scaler instance should be saved and used in conjunction with the `get_features` 
            function to perform inference on new data. This initializes a new `StandardScaler`
            instance which will not perform the same scaling transformations that a different 
            instance produces. In order to get accurate predictions, the same instance must be used
            during model training and inference.

        Parameters
        ----------
        feature_data : ``numpy.ndarray(numpy.float32)``

        Returns
        -------
        scaled_data : ``numpy.ndarray(numpy.float32)``
            scaled feature data
        scaler : ``sklearn.preprocessing.StandardScaler
            StandardScaler instance

        """
        # NOTE: I am pretty sure StandardScaler will preserve the dtype between the input data
        #       and the transformed data, i.e. if providing an array with dtype=np.float32 as 
        #       input the scaled output will also have dtype=np.float32
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)
        return scaled_data, scaler
