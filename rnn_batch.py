import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils import parse_log_file
import pathlib
from automaton import Automaton
from logDataset import LogDataset
import pickle
import json

LETTERS = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BRNN(nn.Module):
    #hyperparametres
    embedding_dim = 10
    hidden_dim = 200
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss pour les sorties binaires
    epochs = 500


    def __init__(self):
        super(BRNN, self).__init__()
        self.embedding = nn.Embedding(LETTERS, BRNN.embedding_dim)
        self.rnn = nn.RNN(BRNN.embedding_dim, BRNN.hidden_dim, batch_first=True)  # batch_first=False par défaut
        self.linear = nn.Linear(BRNN.hidden_dim, 1)

    def get_hidden_size(self):
        return BRNN.hidden_dim

    def forward(self, x, hidden_state=None):
        embedded = self.embedding(x)
        _, hs = self.rnn(embedded, hidden_state)
        output = torch.sigmoid(self.linear(hs))  # Prendre la dernière sortie de la séquence (many-to-one)
        return output, hs

    def get_default_hidden_state(self):
        return torch.zeros(1, 1, self.rnn.hidden_size).to(DEVICE)
    
    def step(self, hidden_state, input_char):
        embedded = self.embedding(torch.tensor(input_char).to(DEVICE)).unsqueeze(0).unsqueeze(1)
        _, next_hidden_state = self.rnn(embedded, hidden_state.to(DEVICE))
        return next_hidden_state

    def step_batch(self, hidden_state, input_char): #TODO à checker
        embedded = self.embedding(input_char.to(DEVICE)).unsqueeze(1)
        hs = hidden_state.expand(-1, embedded.size(0), -1).contiguous().to(DEVICE)
        _, next_hidden_state = self.rnn(embedded, hs)
        return next_hidden_state

    def get_hidden_state(self, x):
        embedded = self.embedding(x)
        _, hidden_state = self.rnn(embedded)
        return hidden_state
    
    def get_all_hidden_states(self, dataloader):
        hidden_states = []
        with torch.no_grad():
            for X, _ in dataloader:
                hidden_state = None
                X = X.to(DEVICE)
                for i in range(X.size(1)):
                    embedded = self.embedding(X[:,i]).unsqueeze(1)
                    _, hidden_state = self.rnn(embedded, hidden_state) # unsqueeze(1) permet d'avoir une dimension [1024,1] au lieu de [1024]
                    hidden_states.append(hidden_state)
        return torch.cat(hidden_states, dim=1)

    def train(self, dataloader):
        a = ""
        optimizer = optim.Adam(self.parameters(), lr=0.00001)  # Adam optimizer
        # Boucle d'entraînement
        for epoch in range(BRNN.epochs):
            total_loss = 0.0
            cpt = 0
            optimizer.zero_grad()  # Réinitialiser les gradients
            for X, y in dataloader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                outputs, _ = self(X)  # Prédiction du modèle
                outputs = outputs.squeeze()
                loss = BRNN.criterion(outputs, y)
                loss.backward()  # Rétropropagation
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()  # Mise à jour des poids

                total_loss += loss.item()
                cpt += 1
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / cpt
                print(f'Epoch [{epoch+1}/{BRNN.epochs}], Loss: {avg_loss:.4f}')

            if (epoch + 1) % 25 == 0:
                torch.save(self, f"test/{epoch+1}.pth")

            avg_loss = total_loss / cpt
            a += f"{avg_loss};"
        with open("loss_brnn.csv", "a") as f:
            f.write(a + "\n")

    def scores(self, y_true, y_pred):
        combined = list(zip(y_pred, y_true))
        true_positives  = sum([1 for pred, true in combined if pred == 1 and true == 1])
        false_positives = sum([1 for pred, true in combined if pred == 1 and true == 0])
        false_negatives = sum([1 for pred, true in combined if pred == 0 and true == 1])
        print(f"TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}")

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0 # VP / (VP + FP)
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0 # VP / (VP + FN)
        f1_score = 2*true_positives / (2*true_positives + false_positives + false_negatives) if (2*true_positives + false_positives + false_negatives) > 0 else 0 # 2*VP / (2*VP + FP + FN)
        return precision, recall, f1_score

    def get_hyperparameters(self):
        return {
            "embedding_dim": BRNN.embedding_dim,
            "hidden_dim": BRNN.hidden_dim,
            "epochs": BRNN.epochs
        }
    
    def predict(self, dataloader):
        outputs = None
        with torch.no_grad():
            for X, _ in dataloader:
                X = X.to(DEVICE)
                op, _ = self(X)
                op = op.squeeze()
                outputs = torch.cat((outputs, op), dim=0) if outputs is not None else op
        return outputs #raw output


    def predict_flow(self, X, y_true, window_size=22):
        with torch.no_grad():
            Xt = X.detach().cpu().numpy()
            y = np.zeros_like(Xt)
            starting_index = 0 # starting index of the window

            hallucinations = 0
            hidden_state = None

            while starting_index + window_size <= len(Xt):
                sample = Xt[starting_index:starting_index + window_size]
                pred, hs = self.forward(torch.tensor(sample).unsqueeze(0).to(DEVICE), hidden_state)
                if pred.item() > 0.8: # function found
                    if sum(y_true[starting_index:starting_index + window_size]) == 0: #hallucination check
                        hallucinations += 1
                    y[starting_index + 2] = 1
                    starting_index += 22 # jump to next window
                    hidden_state = None
                else:
                    starting_index += 1 # slide the window by 1
                    hidden_state = hs
            print(f"Total hallucinations: {hallucinations}")
            return y






def prepared_data(files, DA, DB, DELTA):
    X, y = [], []
    for file in files:
        data = parse_log_file(file)
        rough_entries = {h+i for i in range(DA,DB) for h in data['function']}
        X.extend([ data["mem"].get_byte(h) for h in rough_entries if data["mem"].seg_in_memory(h)]) # vérifie qu'on ne tape pas en dehors de la mémoire
        y.extend([ 1 if h in data["function"] else 0 for h in rough_entries if data["mem"].seg_in_memory(h)]) # vérifie qu'on ne tape pas en dehors de la mémoire
    y = [0]*DELTA+y[:-DELTA] # décalage de DELTA octets
    return X, y


def testing_data(files, DELTA): #toutes les données
    X, y = [], []
    addr = []
    for file in files:
        data = parse_log_file(file)
        for seg in data["mem"].segments:
            X.extend(seg.data)
            addr.extend([seg.start + i for i in range(len(seg.data))])
    y.extend([ 1 if h in data["function"] else 0 for h in addr])
    y = [0]*DELTA+y[:-DELTA] # décalage de DELTA octets
    return X, y


import time

if __name__ == "__main__":
    #training_files = ["kernel32.log", "msvcr100.log", "user32.log", "ntdll.log", "libcrypto.log", "firewallAPI.log", "ws2_32.log", "signdrv.log", "cmdext.log", "gdi32.log"]
    training_files = ["kernel32.log", "msvcr100.log", "user32.log"]
    testing_files = ["ntdll.log"]
    model_pth = "models/bkmu.500.pth"
    batch_size = 1024

    dataset = LogDataset(files=training_files, whitelist=True)
    dataset.prep_batch()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Fichier chargé pour l'entraînement : {dataset.used_files}")

    if pathlib.Path(model_pth).is_file():
        #if learning has been done
        model = torch.load(model_pth, weights_only=False).to(DEVICE)
    else:
        #otherwise, we learn the model
        model = BRNN().to(DEVICE)
        print(f"Hyperparamètres : {model.get_hyperparameters()}")
        model.train(dataloader)
        torch.save(model, model_pth)


    ds_test = LogDataset(files=testing_files, whitelist=True)
    ds_test.prep_batch(randomize=False)
    print(f"Taille d'une séquence de test : {len(ds_test.data[0])}")
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    y_t = torch.tensor(ds_test.labels).numpy()

    with torch.no_grad():
        predicted = (model.predict(dl_test).squeeze() > 0.8).int().detach().cpu().numpy()  # Seuil à 0.5 pour obtenir des 0 et des 1
        print(f'\nRNN  : {predicted}')
        print(f'Diff : {sum(abs(y_t - predicted))}')
        print(f"Scores : {model.scores(y_t, predicted)}")


    # Automaton
    states = 500
    #auto_path = f"automate/{model_pth.split('/')[-1].replace('.pth','.pkl')}"
    auto_path = f"automate/bkmu.500_{states}.pkl"
    if pathlib.Path(auto_path).is_file():
        with open(auto_path, "rb") as f:
            A = pickle.load(f)

        with open(auto_path, "rb") as f:
            B = pickle.load(f)
    else:
        print("Construction de l'automate...\n")
        start = time.time()
        A = Automaton(model, list(range(256)), states, dl_test, y_t)
        print(f"Temps de construction de l'automate : {time.time() - start} secondes")
        A.emonde()
        with open(auto_path, "wb") as f:
            pickle.dump(A, f)
    print(f"\nNombre d'état après émondage : {len(A.Q)} \nNombre d'états finaux : {len(A.F)}")

    predicted_auto = []
    history = []
    for x in ds_test.data:
        p, h  = A.predict(torch.tensor(x))
        predicted_auto.append(p[-1])
        history.append(h)

    print(f"Auto : {predicted_auto[:10]}")
    print(f"Diff :{sum(abs(predicted_auto - y_t))}")
    print(f"Scores Auto : {model.scores(y_t, predicted_auto)}")



    """ finals = A.path_to_finals()
    dot = "digraph {\n"
    json_save = []

    for state, path in finals.items():
        print(f"Chemins vers l'état final {state} : {path} / {len(path[0])} lettres")
        json_save.append(path[0])
        for i in range(len(path[1])-1):
            a = f"{path[1][i]} -> {path[1][i+1]} [label=\"{path[0][i]}\"];\n"
            if a not in dot:
                dot += a

    for state in A.F:
        dot += f"{state} [fillcolor=yellow, style=filled];\n"

    dot += "}"
    with open("automate/bfs/paths_to_finals.dot", "w") as f:
        f.write(dot)

    with open("automate/bfs/paths_to_finals.json", "w") as f:
        json.dump(json_save, f) """
    
    """ B.minimize()
    print(f"Nombre d'état après minimisation : {len(B.Q)} \nNombre d'états finaux : {len(B.F)}")
    predicted_auto_min = B.predict(X_t)
    print(f"Auto Min : {predicted_auto_min}")
    print(f"Diff :{sum(abs(predicted_auto_min - y_t.detach().numpy()))}")
    print(f"Scores Auto Min : {model.scores(y_t, predicted_auto_min)}")
    print(f"Diff A/B : {sum(abs(predicted_auto - predicted_auto_min))}") """


