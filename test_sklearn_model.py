"""
    test training a SVR model using sklearn 
    and 1024-bit molecular fingerprints
    with Chen, et al. dataset (src_id=412)
"""

import numpy as np

from rtp.data import assemble_dataset_arrays, split_train_test_data, FeatureSet
from rtp.sklearn_models import train_random_forest, metrics, model_predictions, train_gradient_booster, train_SVR, plots, train_model


def _main():
    
    # extract the dataset
    fp_size = 1024
    labels, features, rts = assemble_dataset_arrays([412], 
                                                    "./idpp.db", 
                                                    FeatureSet.FP,
                                                    fp_size)
    
    # split into training/test sets
    train_idx, test_idx = split_train_test_data(rts)
    train_labels, train_features, train_rts = labels[train_idx], features[train_idx], rts[train_idx]
    test_labels, test_features, test_rts = labels[test_idx], features[test_idx], rts[test_idx]
    
    #get some information about the dataset
    print(train_features.shape)
    print(train_rts.shape)
    print(test_features.shape)
    print(test_rts.shape)

    #create lists for parameter tuning
    n_estimators = [10,50,100,200]
    max_depth = [None,10,20,30]
    min_samples_split = [2, 3, 4, 5, 10, 20]
    min_samples_leaf = [1,2,4, 10, 20, 50]
    max_features = [1,5,10]
    learning_rate = [.005, .05, .025, 1]
    c = [.001, .01, .1, 1, 10, 100, 1000]
    gamma= ['scale', 'auto', .0001, .001, .01, .1, 1, 10, 100]

    #defining filepath where model instance will be saved        
    filepath = './test_SVR.pkl'

    #Using a grid search to find optimal parameters and training the model
    svm = train_SVR(train_features, 
                           train_rts,
                           filepath,
                           c,
                           gamma
                           )
    
    #getting model predictions
    svm_pred = model_predictions(svm, test_features) 

    #evaluating model performance
    eval_svm = metrics(svm_pred, test_rts)

    #plotting results
    plots(test_rts, svm_pred)


if __name__ == "__main__":
    _main()
