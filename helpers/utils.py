import pickle
import pandas as pd

def save_camstroke(camstroke_object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(camstroke_object, f)
    return filename

def load_camstroke(filename):
    with open(filename, 'rb') as f:
        camstroke = pickle.load(f)
        return camstroke

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def unique_array_dict(L, key):
    return list({v[key]:v for v in L}.values())

def calc_average(arr):
    return sum(arr) / len(arr)