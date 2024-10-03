"""
    test training a simple linear model with early stopping
    using keras and MQN molecular features
    with Chen, et al. dataset (src_id=412)
"""


from rtp.data import assemble_dataset_arrays, split_train_test_data, FeatureSet
from rtp.keras_models import build_linear_model
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping


def _main():
    
    # extract the dataset
    labels, features, rts = assemble_dataset_arrays([412], 
                                                    "/Users/schw939/Library/CloudStorage/OneDrive-PNNL/idpp_stuff/idpp.db", 
                                                    FeatureSet.MQN)
    
    # split into training/test sets
    train_idx, test_idx = split_train_test_data(rts)
    train_labels, train_features, train_rts = labels[train_idx], features[train_idx], rts[train_idx]
    test_labels, test_features, test_rts = labels[test_idx], features[test_idx], rts[test_idx]
    
    # print some info about the dataset sizes
    print("train_labels:", train_labels.shape)
    print("train_features:", train_features.shape)
    print("train_rts:", train_rts.shape)
    print("test_labels:", test_labels.shape)
    print("test_features", test_features.shape)
    print("test_rts:", test_rts.shape)

    # initialize a linear model
    klm = build_linear_model(42)
    klm.summary()
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience= 5)

    # train the model with batches of 256 for 128 epochs
    hist = klm.fit(train_features, 
                   train_rts, 
                   validation_data=(test_features, test_rts),
                   batch_size=256, 
                   epochs=128, 
                   verbose=2
                   ,callbacks=[es])

    print(hist.history.keys())

    # save the optimized model to file
    klm.save("./test_linear.keras")

    plt.plot(hist.history['root_mean_squared_error'])
    plt.plot(hist.history['val_mean_absolute_error'])
    plt.title('model RMSE')
    plt.ylabel('root_mean_squared_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(hist.history['mean_absolute_error'])
    plt.plot(hist.history['val_mean_absolute_error'])
    plt.title('model MAE')
    plt.ylabel('mean_absolute_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    _main()

