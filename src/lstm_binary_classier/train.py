#!/usr/bin/env python3
# Based on https://www.kaggle.com/alphasis/xgboost-with-context-label-data-acc-99-637
# Make sure you have dataset/dataset1 and dataset/dataset2 in the right place in the folder above where this file is

from __future__ import print_function
import sys
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import gc
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, SimpleRNN
from keras.optimizers import RMSprop
from joblib import Parallel, delayed

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..",)))
from get_time_series import get_file



batch_size = 128
epochs = 10

max_num_features = 10
pad_size = 1
boundary_letter = -1
space_letter = 0
max_data_size = 9000000

FREQUENCY_UNIT_COUNT = 4432

dataset_path1 = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset1"))
dataset_path4 = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset4"))
dataset_path5 = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset5"))

out_path = os.path.realpath(os.path.join(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(out_path, 'lstm_model_pickle.hf5')

labels = ["up", "down", "silence"]

def load_file(file_path, class_path):
    file_path = os.path.join(class_path, file_path)
    # Load wav files
    x_data_from_file =  get_file(file_path)

    # Get the class from folder name

    class_name = os.path.basename(class_path)
    if class_name.startswith("all_"):
        class_name = class_name[len("all_"):]
    return x_data_from_file, labels.index(class_name)


def load_sound_file_for_network(file_path):
    """
    Loads a file for the network, should match what we are doing in the function load_dataset and load_file
    :param file_path:
    :return:
    """
    # Load file, just give some label
    data = load_file(file_path, labels[0])[0]
    x_data = np.zeros((1, FREQUENCY_UNIT_COUNT))
    # import code; code.interact(local=dict(globals(), **locals()))

    x_data[0] = np.array(data[1][:min(FREQUENCY_UNIT_COUNT, len(data[1]))])

    return_value =  np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    print(return_value.shape)
    return return_value

def load_dataset(path):
    outputs = []
    print("Loading dataset" + str(path))
    for class_path in os.listdir(path):
        class_path = os.path.join(path, class_path)
        if os.path.isdir(class_path):
            #for file_path in os.listdir(class_path):
            out = Parallel(n_jobs=4, backend="threading")(delayed(load_file)(file_path, class_path)
                                                              for file_path in os.listdir(class_path))
            outputs += out


    # Just arrange this stuff back
    x_data = np.zeros((len(outputs), FREQUENCY_UNIT_COUNT))
    y_data = np.zeros(len(outputs))

    for i, output in enumerate(outputs):
        good_length = min(FREQUENCY_UNIT_COUNT, len(output[0][1]))
        amplitude = np.abs(output[0][1])[:good_length]
        phase = np.angle(output[0][1])[:good_length]

        ampiphase = np.einsum('i,j->ij', amplitude, phase).ravel()
        
        # Dump wave data in to array, if its too long cut it in to WAV_LENGTH
        #import code; code.interact(local=dict(globals(), **locals()))
        x_data[i] = np.array(ampiphase[:min(FREQUENCY_UNIT_COUNT, len(ampiphase))])
        y_data[i] = output[1]
    return x_data, y_data



# model.add(Dense(1, activation='sigmoid'))

model = Sequential()
axis = 1
complexity_of_network = 16 # Change size to change how big it is
model.add(LSTM(complexity_of_network, dropout=0.2, recurrent_dropout=0.2, input_shape=(FREQUENCY_UNIT_COUNT, axis)))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(len(labels), activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

if __name__ == "__main__":
    # Load dataset
    x_data, y_data = load_dataset(dataset_path1)
    x_data_valid, y_data_valid = load_dataset(dataset_path4)
    x_data5, y_data5 = load_dataset(dataset_path5)

    print('Total number of samples:', len(x_data))

    x_train = x_data
    y_train = y_data

    # Add all of dataset 4 as validation, sperate recording from dataset 1
    x_train = np.concatenate([x_train, x_data_valid, x_data5])
    y_train = np.concatenate([y_train, y_data_valid, y_data5])

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          test_size=0.1, random_state=2017)



    #x_valid = x_data_valid
    #y_valid = y_data_valid





    # Load dataset to network for training

    # Number of classes to train
    num_classes = len(labels)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)

    print("current x train shape:" + str(x_train.shape))
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))
    print("current x train shape2:" + str(x_train.shape))
    # y_train= np.reshape(y_train,(y_train.shape[0],y_train.shape[1],1))
    # y_valid = np.reshape(y_valid,(y_valid.shape[0],y_valid.shape[1],1))
    gc.collect()



    print(x_train.shape, y_train.shape)
    # np.save("/tmp/yay.pkl", x_valid)
    keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=False,
                                    save_weights_only=False, mode='auto', period=1)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_valid, y_valid),
                        class_weight={
    0: 1.0,
    1: 1.0,
    2: 1.0
},
                        shuffle=True)

    gc.collect()

    """
    score = model.evaluate(x_valid, y_valid, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    from matplotlib import pyplot as plt
    
    
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(history.history['acc'], 'b')
    ax[0].set_title('Accuraccy')
    ax[1].plot(history.history['loss'], 'r')
    ax[1].set_title('Loss')
    plt.show()
    
    pred = model.predict(x_valid)
    # print 'pred', '=====', pred
    pred = [labels[np.argmax(x)] for x in pred]
    pred = [labels[np.argmax(x)] for x in pred]
    x_valid = [[chr(x) for x in y[2 + max_num_features: 2 + max_num_features * 2]] for y in x_valid]
    x_valid = [''.join(x) for x in x_valid]
    x_valid = [re.sub('a+$', '', x) for x in x_valid]
    
    gc.collect()
    
    
    
    df_pred = pd.DataFrame(columns=['data', 'predict', 'target'])
    df_pred['data'] = x_valid
    df_pred['predict'] = pred
    df_pred['target'] = y_valid
    df_pred.to_csv(os.path.join(out_path, 'pred_lstm.csv'))
    
    df_erros = df_pred.loc[df_pred['predict'] != df_pred['target']]
    df_erros.to_csv(os.path.join(out_path, 'errors_lstm.csv'), index=False)
    """

    print("Saving model")
    # model.save_weights(MODEL_PATH)
    model.save(MODEL_PATH)



