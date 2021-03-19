from hmmlearn import hmm
from helpers.utils import print_full
from helpers.hmm import preprocess, split_dataset

def keystroke_model(n_state):
    model = hmm.GaussianHMM(n_components=n_state, covariance_type="full")

def train(kpoints):
    dataset = preprocess(kpoints)
    print_full(dataset)
    return
