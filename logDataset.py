import torch
from torch.utils.data import Dataset, DataLoader
from utils import parse_log_file


def prepared_data(file, DA, DB, DELTA):
    data = parse_log_file(file)
    rough_entries = {h+i for i in range(DA,DB) for h in data['function']}
    X = [ data["mem"].get_byte(h) for h in rough_entries]
    y = [ 1 if h in data["function"] else 0 for h in rough_entries]
    y = [0]*DELTA+y[:-DELTA] # décalage de DELTA octets
    return X, y


def testing_data(file, DELTA): #toutes les données
    X, y = [], []
    addr = []
    data = parse_log_file(file)
    for seg in data["mem"].segments:
        X.extend(seg.data)
        addr.extend([seg.start + i for i in range(len(seg.data))])
    y.extend([ 1 if h in data["function"] else 0 for h in addr])
    y = [0]*DELTA+y[:-DELTA] # décalage de DELTA octets
    return X, y


class LogDataset(Dataset):

    log_files = ["kernel32.log", "user32.log", "msvcr100.log", "ntdll.log"]
    used_files = []

    def __init__(self, files, whitelist=True, training=True, DA=-2, DB=20, DELTA=6):
        data_files = list(set(self.log_files) & set(files)) # intersection
        data_files = data_files if whitelist else [f for f in self.log_files if f not in data_files]
        self.used_files = data_files
        
        self.data = []
        self.labels = []
        for file in data_files:
            if training:
                X, y = prepared_data(f"data/{file}", DA=DA, DB=DB, DELTA=DELTA)
            else:
                X, y = testing_data(f"data/{file}", DELTA=DELTA)
            self.data.append(X)
            self.labels.append(y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        Xt = torch.tensor(self.data[idx])
        yt = torch.tensor(self.labels[idx]).float()
        return Xt, yt


if __name__ == "__main__":
    dataset = LogDataset(files=["user32.log"], whitelist=False, training=True)
    print(f"Dataset size: {len(dataset)}")
    sample_data, sample_label = dataset[0]
    print(f"Sample data: {sample_data[:10]}, Sample label: {sample_label[:10]}")
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)