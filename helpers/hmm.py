import pandas as pd

def preprocess(fps, kpoints):
    data = [kp.get_timing_data(fps) for kp in kpoints]
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