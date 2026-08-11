import torch
from torch.utils.data import Dataset, DataLoader
from utils import parse_log_file, parse_log_file_instructions
import random
from zydis_wrapper import label

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


def testing_data_headers(file): #instructions
    X, y = [], []
    addr = []
    data = parse_log_file(file)
    for seg in data["mem"].segments:
        X.extend(seg.data)
        addr.extend([seg.start + i for i in range(len(seg.data))])
    y.extend([1 if h in data["head"] else 0 for h in addr])
    return X, y


def data_instructions(file):
    X, y = [], []
    data = parse_log_file_instructions(file)
    for page in data["pages"].segments:
        instr_addr = data["instructions"].get(page.start, [])
        X.append(page.data)
        data_addr = [page.start + i for i in range(len(page.data))]
        y.append([1 if h in instr_addr else 0 for h in data_addr])
    return X, y


def instr_multiclass(file):
    X, y = [], []
    data = parse_log_file_instructions(file)
    for page in data["pages"].segments:
        instr_addr = data["instructions"].get(page.start, [])
        X.append(page.data)

        y_temp = [0]*len(page.data)
        for i in range(len(instr_addr)-1):
            strt_index = instr_addr[i] - page.start
            end_index = instr_addr[i+1] - page.start
            instruction = page.data[strt_index:end_index]
            labels = label(instruction)

            if labels[-1] != 0:
                labels[-1] += 3 # Marque la fin de l'instruction
            else: # Chercher la fin de l'instruction
                for j in range(len(labels)-1, -1, -1):
                    if labels[j] != 0:
                        labels[j] += 3
                        break
            y_temp[strt_index:end_index] = labels
        y.append(y_temp)
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

    def __init__(self, path, files):
        self.used_files = files
        self.path = path

    def prep_batch(self, randomize=False, DA=-2, DB=20): #Batch
        if not randomize:
            random.seed(42)
        
        #reset data and labels
        self.data = []
        self.labels = []

        for file in self.used_files:
            log = parse_log_file(f"{self.path}/{file}")
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
                X, y = prepared_data(f"{self.path}/{file}", DA=DA, DB=DB, DELTA=DELTA)
            else:
                X, y = testing_data(f"{self.path}/{file}", DELTA=DELTA)
            self.data.append(X)
            self.labels.append(y)
            
        for i in range(len(self.data)):
            print(f"File {self.used_files[i]} - Nbr 1 in y : {sum(self.labels[i])} / Nbr 0 in y : {len(self.labels[i])-sum(self.labels[i])}")


    def prep_fs(self, DELTA=6, reverse=False): #Full Stream
        self.data = []
        self.labels = []

        for file in self.used_files:
            X, y = testing_data(f"{self.path}/{file}", DELTA=DELTA)
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
            X, y = testing_data_headers(f"{self.path}/{file}")
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
        cpt = [0,0]

        for file in self.used_files:
            X, y = testing_data_multi(f"{self.path}/{file}", DELTA=DELTA)
            self.data.append(X)
            self.labels.append(y)
            cpt[0] += len(X)
            cpt[1] += len(y)
            if len(X) != len(y):
                print(f"Erreur : {file} - len(X) = {len(X)} / len(y) = {len(y)}")

        print(f"Nombre de fichiers : {len(self.used_files)}")
        print(f"Nombre total d'octets : {cpt[0]}")
        print(f"Nombre total de labels : {cpt[1]}")
        

    def prep_instr(self): #Instructions
        self.data = []
        self.labels = []

        for file in self.used_files:
            X, y = data_instructions(f"{self.path}/{file}")
            self.data.extend(X)
            self.labels.extend(y)


    def prep_instr_multiclass(self): #Instructions multi-class
        self.data = []
        self.labels = []

        for file in self.used_files:
            X, y = instr_multiclass(f"{self.path}/{file}")
            self.data.extend(X)
            self.labels.extend(y)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        Xt = torch.tensor(self.data[idx])
        yt = torch.tensor(self.labels[idx]).float()
        return Xt, yt




if __name__ == "__main__":
    dataset = LogDataset(path="data", files=["libcrypto.log"])
    dataset.prep_batch(randomize=True)
    print(f"Dataset size: {len(dataset)}")
    sample_data, sample_label = dataset[0]
    print(f"Sample data: {sample_data[:10]}, Sample label: {sample_label}")
    print(f"Len sample: {len(sample_data)}")
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)