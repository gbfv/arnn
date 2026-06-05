import torch
from torch.utils.data import Dataset, DataLoader
from utils import parse_log_file
import random

def prepared_data(file, DA, DB, DELTA):
    data = parse_log_file(file)
    rough_entries = {h+i for i in range(DA,DB) for h in data['function']}
    X = [ data["mem"].get_byte(h) for h in rough_entries if data["mem"].seg_in_memory(h)]
    y = [ 1 if h in data["function"] else 0 for h in rough_entries if data["mem"].seg_in_memory(h)]
    if DELTA > 0:
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
    if DELTA > 0:
        y = [0]*DELTA+y[:-DELTA] # décalage de DELTA octets
    return X, y


def testing_data_headers(file):
    X, y = [], []
    addr = []
    data = parse_log_file(file)
    for seg in data["mem"].segments:
        X.extend(seg.data)
        addr.extend([seg.start + i for i in range(len(seg.data))])
    y.extend([1 if h in data["head"] else 0 for h in addr])
    return X, y


def testing_data_multi(file, DELTA):
    X, y = [], []
    addr = []
    data = parse_log_file(file)
    for seg in data["mem"].segments:
        X.extend(seg.data)
        addr.extend([seg.start + i for i in range(len(seg.data))])
    y.extend([ 1 if h in data["function"] else 0 for h in addr])
    for i in range(len(y)):
        if y[i] == 1:
            for j in range(DELTA):
                y[i-j] = DELTA-j
    y = [0]*DELTA+y[:-DELTA] # décalage de DELTA octets
    return X, y


def pick_negatives(log, DA, DB, nb_pos):
    segments = log["mem"].segments
    sizes = [len(segments[i].data) for i in range(len(segments))]

    negatives = []
    pool = []
    attempts = 0
    refresh_pool = min(10000, nb_pos)
    while len(negatives) < nb_pos and attempts < nb_pos*20:
        if attempts % refresh_pool == 0:
            pool = random.choices(range(len(segments)), weights=sizes, k=refresh_pool)
        seg = pool.pop()

        start = segments[seg].start + random.randint(0, len(segments[seg].data)-(DB-DA)-1) #mettre un max sur la borne sup au cas où le segment est trop petit ?
        sample = [ start + h for h in range(DA,DB) ]
        a = [ 1 if h in log["function"] else 0 for h in sample if log["mem"].seg_in_memory(h) ]
        if len(a) == (DB - DA) and a[2] != 1: #sum(a) == 0: # aucune fonction dans l'échantillon + sample complet
            negatives.append([ log["mem"].get_byte(h) for h in sample if log["mem"].seg_in_memory(h) ])
    return negatives
    



class LogDataset(Dataset):
    log_files = ["kernel32.log", "user32.log", "msvcr100.log", "ntdll.log", "basesrv.log", "clp64.log", "cmdext.log", "crypt32.log", "energy.log", "firewallAPI.log", "gdi32.log", "ieproxy.log", "kerberos.log", "libcrypto.log", "signdrv.log", "ws2_32.log"]
    used_files = []
    data = []
    labels = []

    def __init__(self, files, whitelist=True, x32=False):
        if x32:
            self.log_files = ["x32/crypt32.log", "x32/kerberos.log", "x32/kernel32.log", "x32/msvcr100.log", "x32/user32.log"]
        data_files = list(set(self.log_files) & set(files)) # intersection
        data_files = data_files if whitelist else [f for f in self.log_files if f not in data_files]
        self.used_files = data_files

    def prep_batch(self, randomize=False, DA=-2, DB=20): #Batch
        if not randomize:
            random.seed(42)
        
        #reset data and labels
        self.data = []
        self.labels = []

        for file in self.used_files:
            log = parse_log_file(f"data/{file}")
            fonctions = sorted(list(log['function']))
            positifs = 0
            for i in range(len(fonctions)):
                sample = [ fonctions[i]+h for h in range(DA,DB) ]
                sample_data = [ log["mem"].get_byte(h) for h in sample if log["mem"].seg_in_memory(h) ]
                if len(sample_data) == (DB - DA): # on vérifie que l'échantillon est complet
                    self.data.append(sample_data)
                    self.labels.append(1)
                    positifs += 1

            neg_samples = pick_negatives(log, DA, DB, nb_pos=positifs)
            self.data.extend(neg_samples)
            self.labels.extend([0]*len(neg_samples))
        print(f"Nbr 1 in y : {sum(self.labels)} / Nbr 0 in y : {len(self.labels)-sum(self.labels)}")


    def prep_concat(self, training=True, DA=-2, DB=20, DELTA=6): #Concatenation
        #reset data and labels
        self.data = []
        self.labels = []

        for file in self.used_files:
            if training:
                X, y = prepared_data(f"data/{file}", DA=DA, DB=DB, DELTA=DELTA)
            else:
                X, y = testing_data(f"data/{file}", DELTA=DELTA)
            self.data.append(X)
            self.labels.append(y)
            
        for i in range(len(self.data)):
            print(f"File {self.used_files[i]} - Nbr 1 in y : {sum(self.labels[i])} / Nbr 0 in y : {len(self.labels[i])-sum(self.labels[i])}")


    def prep_fs(self, DELTA=6, reverse=False): #Full Stream
        self.data = []
        self.labels = []

        for file in self.used_files:
            X, y = testing_data(f"data/{file}", DELTA=DELTA)
            if reverse:
                X.reverse()
                y.reverse()
            self.data.append(X)
            self.labels.append(y)
        
        for i in range(len(self.data)):
            print(f"File {self.used_files[i]} - Nbr 1 in y : {sum(self.labels[i])} / Nbr 0 in y : {len(self.labels[i])-sum(self.labels[i])}")


    def prep_header_fs(self): #pas de delta
        self.data = []
        self.labels = []

        for file in self.used_files:
            X, y = testing_data_headers(f"data/{file}")
            self.data.append(X)
            self.labels.append(y)

        total_pos = sum(sum(labels) for labels in self.labels)
        total_neg = sum(len(labels)-sum(labels) for labels in self.labels)
        print(f"Total - Nbr 1 in y : {total_pos} / Nbr 0 in y : {total_neg}")
        for i in range(len(self.data)):
            print(f"File {self.used_files[i]} - Nbr 1 in y : {sum(self.labels[i])} / Nbr 0 in y : {len(self.labels[i])-sum(self.labels[i])}")


    def prep_multi(self, DELTA=6): #Multi-class avec les DELTA classes de 1 à DELTA
        self.data = []
        self.labels = []

        for file in self.used_files:
            X, y = testing_data_multi(f"data/{file}", DELTA=DELTA)
            self.data.append(X)
            self.labels.append(y)
        


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        Xt = torch.tensor(self.data[idx])
        yt = torch.tensor(self.labels[idx]).float()
        return Xt, yt




if __name__ == "__main__":
    dataset = LogDataset(files=["libcrypto.log"], whitelist=False)
    dataset.prep_batch(randomize=True)
    print(f"Dataset size: {len(dataset)}")
    sample_data, sample_label = dataset[0]
    print(f"Sample data: {sample_data[:10]}, Sample label: {sample_label}")
    print(f"Len sample: {len(sample_data)}")
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)