import pickle

def save_camstroke(camstroke_object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(camstroke_object, f)
    return filename

def load_camstroke(filename):
    with open(filename, 'rb') as f:
        camstroke = pickle.load(f)
        return camstroke
