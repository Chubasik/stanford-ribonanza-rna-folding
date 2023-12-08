import pickle


def load_dict_from_file(file_path):
    with open(file_path, 'rb') as f:
        cached_dict = pickle.load(f)
    return cached_dict