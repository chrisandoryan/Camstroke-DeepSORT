from pohmm import Pohmm, PohmmClassifier
import pandas as pd
import numpy as np


def preprocess(kpoints):
    data = [kp.get_timing_data() for kp in kpoints]
    df = pd.DataFrame(data)
    df.set_index('id', inplace=True)

    dataset = df[['keypress', 'keyrelease', 'keyhold', 'keydelay', 'keytext']].copy()
    for col_name in dataset:
        dataset.loc[df[col_name] == 0, col_name] = 0.01

    return dataset


def keystroke_model():
    # """Generates a 2-state model with lognormal emissions and frequency smoothing"""
    model = Pohmm(n_hidden_states=4,
                  init_spread=2,
                  emissions=[('keypress', 'lognormal'),
                             ('keyrelease', 'lognormal'),
                             ('keyhold', 'lognormal'),
                             ('keydelay', 'lognormal')],
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
    dataset = preprocess(kpoints)

    print_full(dataset)
    model.fit_df([dataset], pstate_col='keytext')

    np.set_printoptions(precision=3)
    print(model)