from pohmm import Pohmm, PohmmClassifier

def preprocess(kpoints):
    for kp in kpoints:
        

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
    model = keystroke_model()
    kpoints = preprocess(kpoints)
