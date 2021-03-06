from train_svc import load_sound_file_for_network, load_dataset, FREQUENCY_UNIT_COUNT, labels, MODEL_PATH
import os
import numpy as np
import pickle
from get_time_series import get_file

if __name__ == "__main__":
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)

    #test_file = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset4", "all_up", "chunk86-97.56637.wav"))
    test_file = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset1", "all_down", "chunk10-98.0.wav"))




    """
    dataset_path4 = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset4"))
    x_data_valid, y_data_valid = load_dataset(dataset_path4)
    """
    # x_data_valid = np.reshape(x_data_valid, (x_data_valid.shape[0], x_data_valid.shape[1], 1))


    #y_pred = clf.predict(x_data_valid)



    test = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset4", "all_up", "chunk86-97.56637.wav"))

    x_data = np.zeros((1, FREQUENCY_UNIT_COUNT))
    #

    data = get_file(test)

    x_data[0] = np.array(data[1][:min(FREQUENCY_UNIT_COUNT, len(data[1]))])

    y_pred = clf.predict(x_data)

    print(list(labels[x] for x in  y_pred))

    print("Done")