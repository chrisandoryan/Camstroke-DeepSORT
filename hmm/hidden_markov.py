from pohmm import Pohmm, PohmmClassifier
import pandas as pd
import numpy as np
from helpers.utils import print_full

def preprocess(kpoints):
    data = [kp.get_timing_data() for kp in kpoints]
    df = pd.DataFrame(data)
    df.set_index('id', inplace=True)

    dataset = df[['keypress', 'keyrelease',
                  'keyhold', 'keydelay', 'keytext']].copy()
    for col_name in dataset:
        dataset.loc[df[col_name] == 0, col_name] = 1e-5

    return dataset

def split_dataset(dataset, split_factor=0.7):
    dataset = dataset[:int(len(dataset)/3)]
    train_data, test_data = dataset[:int(len(dataset)*split_factor)], dataset[int(len(dataset)*split_factor):]
    return train_data, test_data

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


def train(kpoints):
    hmm_model = keystroke_model()
    dataset = preprocess(kpoints)
    print_full(dataset)
    
    train_data, test_data = split_dataset(dataset)
    # print_full(train_data)

    hmm_model.fit_df([train_data], pstate_col='keytext')
    print(hmm_model)
    # emissions
    # print(hmm_model.emission_distr)

    # hmm_model params
    # params = hmm_model.params()
    # emissions = params[0]
    # state_transitions = params[1]

    # print("Params: ", params)
    # print("Emissions: ", hmm_model.emission)
    # print("State Transitions: ", hmm_model.transmat)

    # np.set_printoptions(precision=3)
    # print(hmm_model)
    # result = hmm_model.predict_df(dataset, pstate_col='keytext')
    # print(result)

    return hmm_model, test_data
