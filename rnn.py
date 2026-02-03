import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from utils import parse_log_file
import pathlib
from automaton import Automaton
from logDataset import LogDataset
import pickle
import json

LETTERS = 257 #256 + 1 for padding
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ARNN(nn.Module):
    #hyperparametres
    embedding_dim = 10
    hidden_dim = 200
    criterion = nn.BCELoss(reduction="none")  # Binary Cross-Entropy Loss pour les sorties binaires
    epochs = 500


    def __init__(self):
        super(ARNN, self).__init__()
        self.embedding = nn.Embedding(LETTERS, ARNN.embedding_dim, padding_idx=256)
        self.rnn = nn.RNN(ARNN.embedding_dim, ARNN.hidden_dim, batch_first=True)  # batch_first=False par défaut
        self.linear = nn.Linear(ARNN.hidden_dim, 1)

    def get_hidden_size(self):
        return ARNN.hidden_dim

    def forward(self, x, hidden_state=None):
        embedded = self.embedding(x)
        output, hidden_state = self.rnn(embedded, hidden_state)
        output = torch.sigmoid(self.linear(output))
        return output, hidden_state

    def get_default_hidden_state(self):
        return torch.zeros(1, 1, self.rnn.hidden_size).to(DEVICE)
    
    def step(self, hidden_state, input_char):
        embedded = self.embedding(torch.tensor(input_char).to(DEVICE)).unsqueeze(0).unsqueeze(1)
        output, next_hidden_state = self.rnn(embedded, hidden_state.to(DEVICE))
        return next_hidden_state

    def step_batch(self, hidden_state, input_char): #TODO à checker
        embedded = self.embedding(input_char.to(DEVICE)).unsqueeze(1)
        hs = hidden_state.expand(-1, embedded.size(0), -1).contiguous().to(DEVICE)
        output, next_hidden_state = self.rnn(embedded, hs)
        return next_hidden_state

    def get_hidden_state(self, x):
        embedded = self.embedding(x)
        _, hidden_state = self.rnn(embedded)
        return hidden_state
    
    def get_all_hidden_states(self, x):
        hidden_states = []
        hidden_state = self.get_default_hidden_state()
        for char in x:
            embedded = self.embedding(char).unsqueeze(0).unsqueeze(1)
            _, hidden_state = self.rnn(embedded, hidden_state)
            hidden_states.append(hidden_state)
        return torch.cat(hidden_states, dim=1)


    def train(self, dataloader):
        a = ""
        optimizer = optim.Adam(self.parameters(), lr=0.001)  # Adam optimizer
        chunk_size = 65535  # Taille des chunks pour l'entrainement

        # Boucle d'entraînement
        for epoch in range(ARNN.epochs):
            total_loss = 0.0
            cpt = 0
            for X, y, mask in dataloader:
                X, y, mask = X.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
                num_chunks = X.size(1) // chunk_size if X.size(1) % chunk_size == 0 else X.size(1) // chunk_size + 1
                hidden_state = None
                outputs = None

                for chunk_idx in range(num_chunks):
                    optimizer.zero_grad()  # Réinitialiser les gradients
                    if hidden_state is not None:
                        hidden_state = hidden_state.detach()  # évite la rétropropagation à travers tout les chunks (utile pour la mémoire)
                    start = chunk_idx * chunk_size
                    end = min((chunk_idx + 1) * chunk_size, X.size(1))
                    op, hidden_state = self(X[:, start:end], hidden_state)  # Prédiction du modèle
                    #outputs = torch.cat((outputs, op), dim=1) if outputs is not None else op
                    op = op.squeeze(2)
                    mask_chunk = mask[:, start:end]
                    loss_raw = ARNN.criterion(op, y[:, start:end])
                    masked_loss = (loss_raw * mask_chunk).sum() / (mask_chunk.sum() + 1e-8)
                    masked_loss.backward()
                    optimizer.step()

                    total_loss += masked_loss.item()
                    cpt += 1

                """ outputs = outputs.squeeze(2)
                padded_loss = ARNN.criterion(outputs, y)  # Calcul de la perte total (padding inclus)
                masked_loss = padded_loss * mask # applique le masque pour ignorer les paddings
                loss = masked_loss.sum() / mask.sum()  # Moyenne sur les éléments non padding
                loss.backward()  # Rétropropagation
                optimizer.step()  # Mise à jour des poids

                total_loss += loss.item()
                cpt += 1 """
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{ARNN.epochs}], Loss: {total_loss/cpt:.4f}')
            """ a += f"{total_loss / cpt};"
            
        with open("loss_rnn.csv", "a") as f:
            f.write(a + "\n") """



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
            "embedding_dim": ARNN.embedding_dim,
            "hidden_dim": ARNN.hidden_dim,
            "epochs": ARNN.epochs
        }
    
    def predict(self, X):
        with torch.no_grad():
            chunk_size = 65535  # Taille des chunks pour l'entrainement
            num_chunks = X.size(0) // chunk_size if X.size(0) % chunk_size == 0 else X.size(0) // chunk_size + 1 #0 car pas de batch (normalement)
            hidden_state = None
            outputs = None
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, X.size(0))
                op, hidden_state = self(X[start_idx:end_idx], hidden_state)  # Prédiction du modèless
                outputs = torch.cat((outputs, op), dim=0) if outputs is not None else op
        return outputs #raw output

    #avoir une fonction predict_batch ? Pour le cas où X = [sample, seq_len, features] au lieu de [seq_len, features]


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


def pad_batch(batch):
    PADDING_ID_Y = 2
    sequences_x = [item[0] for item in batch]
    sequences_y = [item[1] for item in batch]
     
    X_padded = pad_sequence(sequences_x, padding_value=256, batch_first=True)
    Y_padded = pad_sequence(sequences_y, padding_value=PADDING_ID_Y, batch_first=True)
    
    # creating a mask to ignore padding in loss computation
    mask = (Y_padded != PADDING_ID_Y).float()
    Y_padded[Y_padded == PADDING_ID_Y] = 0.0
    
    return X_padded, Y_padded, mask




if __name__ == "__main__":
    training_files = ["kernel32.log", "msvcr100.log", "user32.log", "ntdll.log", "libcrypto.log", "firewallAPI.log", "ws2_32.log", "signdrv.log", "cmdext.log", "gdi32.log"]
    testing_files = ["data/ntdll.log"]
    model_pth = "models/KMU.500.pth"

    dataset = LogDataset(files=training_files, whitelist=True)
    dataset.prep_concat(training=True, DA=-2, DB=20, DELTA=6)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=pad_batch)
    print(f"Fichier chargé pour l'entraînement : {dataset.used_files}")

    if pathlib.Path(model_pth).is_file():
        #if learning has been done
        model = torch.load(model_pth, weights_only=False).to(DEVICE)
    else:
        #otherwise, we learn the model
        model = ARNN().to(DEVICE)
        print(f"Hyperparamètres : {model.get_hyperparameters()}")
        model.train(dataloader)
        torch.save(model, model_pth)



    X_test, y_test = prepared_data(testing_files, DA=-2, DB=20, DELTA=6)
    print(f"{len(X_test)} items in testing dataset")
    #X_test, y_test = testing_data(testing_files, DELTA=6)
    X_t, y_t = torch.tensor(X_test).to(DEVICE), torch.tensor(y_test)

    with torch.no_grad():
        predicted = (model.predict(X_t).squeeze() > 0.8).int().detach().cpu().numpy()  # Seuil à 0.5 pour obtenir des 0 et des 1
        print(f'\nRNN  : {predicted}')
        print(f'Diff : {sum(abs(y_t - predicted))}')
        print(f"Scores : {model.scores(y_t, predicted)}")



    # Automaton
    states = 1000
    #auto_path = f"automate/{model_pth.split('/')[-1].replace('.pth','.pkl')}"
    auto_path = f"automate/KMU.500_01k.pkl"
    if pathlib.Path(auto_path).is_file():
        with open(auto_path, "rb") as f:
            A = pickle.load(f)
    else:
        print("Construction de l'automate...\n")
        A = Automaton(model, list(range(256)), states, X_t, y_t)
        A.emonde()
        with open(auto_path, "wb") as f:
            pickle.dump(A, f)

    print(f"\nNombre d'état après émondage : {len(A.Q)} \nNombre d'états finaux : {len(A.F)}")
    predicted_auto, history = A.predict(X_t)
    print(f"Auto : {predicted_auto}")
    print(f"Diff :{sum(abs(predicted_auto - y_t.detach().numpy()))}")
    print(f"Scores Auto : {model.scores(y_t, predicted_auto)}")


    oblivion = [history[max(0, i-6)] for i, pred in enumerate(predicted_auto) if pred == 1]
    nb_oblivion = len(oblivion)
    ob_dict = {}
    for state in oblivion:
        if state not in ob_dict:
            ob_dict[state] = 0
        ob_dict[state] += 1
    oblivion = sorted(list(set(oblivion)))
    ob_dict = dict(sorted(ob_dict.items(), key=lambda x: x[1], reverse=True))
    print(f"États d'oubli : {oblivion}")
    for k,v in ob_dict.items():
        print(f"  État {k} : {v} fois ({v/nb_oblivion*100:.2f}%)")


    """ finals = A.path_to_finals(init_state=50)
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


