from pohmm import Pohmm, PohmmClassifier
import pandas as pd

def preprocess(kpoints):
    data = [kp.get_timing_data() for kp in kpoints]
    df = pd.DataFrame(data)
    df.set_index('id', inplace=True)
    df.drop(['kunits'], axis=1, inplace=True)
    return df

def keystroke_model():
    """Generates a 2-state model with lognormal emissions and frequency smoothing"""
    model = Pohmm(n_hidden_states=2,
                  init_spread=2,
                  emissions=[('keyhold','lognormal'),('keydelay','lognormal')],
                  smoothing='freq',
                  init_method='obs',
                  thresh=1)
    return model

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def train(kpoints):
    model = keystroke_model()
    kpoints = preprocess(kpoints)
    print_full(kpoints)
    # model.fit_df(kpoints)