# from pohmm import Pohmm, PohmmClassifier
import pandas as pd

def preprocess(kpoints):
    data = [kp.get_timing_data() for kp in kpoints]
    df = pd.DataFrame(data)
    df.set_index('id', inplace=True)
    df.drop(['kunits'], axis=1, inplace=True)
    print(df)

def keystroke_model():
    """Generates a 2-state model with lognormal emissions and frequency smoothing"""
    model = Pohmm(n_hidden_states=2,
                  init_spread=2,
                  emissions=[('duration','lognormal'),('tau','lognormal')],
                  smoothing='freq',
                  init_method='obs',
                  thresh=1)
    return model

def train(kpoints):
    # model = keystroke_model()
    kpoints = preprocess(kpoints)
