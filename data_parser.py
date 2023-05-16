import pickle

PATH = "C:\\Users\\33785\\Maths\\MVA\\Kernel methods\\kaggle\\data"

def get_data():

    with open(PATH+'\\'+'test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)

    with open(PATH+'\\'+'training_data.pkl', 'rb') as f:
        training_data = pickle.load(f)

    with open(PATH+'\\'+'training_labels.pkl', 'rb') as f:
        training_labels = pickle.load(f)

    return training_data, training_labels, test_data

