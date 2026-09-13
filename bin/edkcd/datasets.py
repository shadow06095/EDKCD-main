import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class PredictionDataset(Dataset):
    """Dataset for (X, Y) time series pairs."""
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        assert X.shape == Y.shape

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def numpy2tensor(np_array):
    return torch.tensor(np_array).float()


def load_dataset(experiment_name, num_sim, base_dir="../datasets"):
    datasets = []
    structures = []

    if experiment_name == "lorenz96_0":
        for i in range(num_sim):
            data_i = pd.read_csv(f"{base_dir}/Lorenz96/Lorenz96_var20_force10_t250_data_{i}.csv", index_col=None)
            data_i[:] = StandardScaler().fit_transform(data_i[:])
            a_i = pd.read_csv(f"{base_dir}/Lorenz96/Lorenz96_var20_force10_t250_struct_{i}.csv", index_col=None)
            datasets.append(data_i.to_numpy())
            structures.append(a_i.to_numpy())
    elif experiment_name == "lorenz96_1":
        for i in range(num_sim):
            data_i = pd.read_csv(f"{base_dir}/Lorenz96/Lorenz96_var20_force40_t250_data_{i}.csv", index_col=None)
            data_i[:] = StandardScaler().fit_transform(data_i[:])
            a_i = pd.read_csv(f"{base_dir}/Lorenz96/Lorenz96_var20_force40_t250_struct_{i}.csv", index_col=None)
            datasets.append(data_i.to_numpy())
            structures.append(a_i.to_numpy())
    elif experiment_name == "lorenz96_2":
        for i in range(num_sim):
            data_i = pd.read_csv(f"{base_dir}/Lorenz96/Lorenz96_var100_force40_t500_data_{i}.csv", index_col=None)
            data_i[:] = StandardScaler().fit_transform(data_i[:])
            a_i = pd.read_csv(f"{base_dir}/Lorenz96/Lorenz96_var100_force40_t500_struct_{i}.csv", index_col=None)
            datasets.append(data_i.to_numpy())
            structures.append(a_i.to_numpy())
    elif experiment_name == "finance":
        for i in range(num_sim):
            data_i = pd.read_csv(f"{base_dir}/Finance/finance_data_{i}.csv", index_col=None)
            data_i[:] = StandardScaler().fit_transform(data_i[:])
            a_i = pd.read_csv(f"{base_dir}/Finance/finance_struct_{i}.csv", index_col=None)
            datasets.append(data_i.to_numpy())
            structures.append(a_i.to_numpy())
    elif experiment_name == "rivers":
        for i in range(num_sim):
            length = 2000
            combined = None
            for fname in ['rivers_train.csv', 'rivers_validation.csv', 'rivers_test.csv']:
                fpath = f"{base_dir}/Rivers/{fname}"
                df = pd.read_csv(fpath, index_col=None)
                combined = df.values.astype(np.float32) if combined is None else np.vstack([combined, df.values.astype(np.float32)])
            if (i + 1) * length > len(combined):
                raise ValueError(f"Rivers sample_id={i} out of range")
            data_slice = combined[i * length:(i + 1) * length]
            data_i = pd.DataFrame(data_slice)
            data_i[:] = StandardScaler().fit_transform(data_i[:])
            n_vars = data_slice.shape[1]
            struct = np.zeros((n_vars, n_vars))
            struct[3, 0] = 1; struct[4, 1] = 1; struct[5, 2] = 1
            struct[1, 0] = 1; struct[2, 0] = 1
            datasets.append(data_i.to_numpy())
            structures.append(struct)
    elif experiment_name == "airquality":
        data_full = np.load(f"{base_dir}/AirQuality/data.npy")
        gc_real = np.load(f"{base_dir}/AirQuality/graph.npy")
        nan_count = np.isnan(data_full).sum()
        if nan_count > 0:
            data_df = pd.DataFrame(data_full)
            data_df = data_df.ffill().bfill().fillna(data_df.mean())
            data_full = data_df.to_numpy()
        for i in range(num_sim):
            data_i = pd.DataFrame(data_full)
            data_i[:] = StandardScaler().fit_transform(data_i[:])
            datasets.append(data_i.to_numpy())
            structures.append(gc_real)
    else:
        raise NotImplementedError(f"Experiment {experiment_name} not supported")

    return datasets, structures