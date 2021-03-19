from pohmm import Pohmm, PohmmClassifier
import pandas as pd
import numpy as np
from helpers.utils import print_full
from helpers.hmm import preprocess, split_dataset

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

    return hmm_model
