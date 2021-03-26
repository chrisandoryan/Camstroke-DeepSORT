import pickle
import pandas as pd
import os

def save_camstroke(camstroke_object, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(camstroke_object, f)
    return save_path

def load_camstroke(save_path):
    with open(save_path, 'rb') as f:
        camstroke = pickle.load(f)
        return camstroke

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def keep_unique(L, key):
    return list({v[key]:v for v in L}.values())

def calc_average(arr):
    return sum(arr) / len(arr)

def print_info(s):
    print("[INFO] %s" % s)

def epoch_to_millis(time):
    return round(time * 1000)

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)