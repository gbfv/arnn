import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from utils import parse_log_file, progress_bar, get_device
import pathlib
import pickle
from automaton import Automaton
from logDataset import LogDataset
import time


LETTERS = 257 #256 + 1 for padding
DEVICE = get_device()


class TMGRU(nn.Module):
    #hyperparametres
    embedding_dim = 10
    hidden_dim = 100
    #criterion = nn.CrossEntropyLoss(reduction="none", weight=torch.tensor([1.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0]).to(DEVICE))  # Cross-Entropy Loss pour les sorties multi-classes
    criterion = nn.CrossEntropyLoss(reduction="none", weight=torch.tensor([1.0, 16.0, 16.0, 16.0, 16.0]).to(DEVICE))
    epochs = 500

    def __init__(self, nbClasses):
        super(TMGRU, self).__init__()
        self.embedding = nn.Embedding(LETTERS, TMGRU.embedding_dim, padding_idx=256)  # 256 pour le padding
        self.gru = nn.GRU(TMGRU.embedding_dim, TMGRU.hidden_dim, batch_first=True)  # batch_first=False par défaut
        self.linear = nn.Linear(TMGRU.hidden_dim, nbClasses)


    def get_hidden_size(self):
        return TMGRU.hidden_dim

    def forward(self, x, hidden_state=None):
        embedded = self.embedding(x)
        output, hidden_state = self.gru(embedded, hidden_state)
        output = self.linear(output)
        return output, hidden_state
    
    def get_default_hidden_state(self):
        return torch.zeros(1, 1, self.gru.hidden_size).to(DEVICE)

    def step(self, hidden_state, input_char):
        embedded = self.embedding(torch.tensor(input_char).to(DEVICE)).unsqueeze(0).unsqueeze(1)
        output, next_hidden_state = self.gru(embedded, hidden_state.to(DEVICE))
        return next_hidden_state

    def step_batch(self, hidden_state, input_char): #TODO à checker
            embedded = self.embedding(input_char.to(DEVICE)).unsqueeze(1)
            hs = hidden_state.expand(-1, embedded.size(0), -1).contiguous().to(DEVICE)
            output, next_hidden_state = self.gru(embedded, hs)
            return next_hidden_state

    def get_hidden_state(self, x):
        embedded = self.embedding(x)
        _, hidden_state = self.gru(embedded)
        return hidden_state
    
    def get_all_hidden_states(self, x):
        hidden_states = []
        hidden_state = self.get_default_hidden_state()
        for char in x:
            embedded = self.embedding(char).unsqueeze(0).unsqueeze(1)
            _, hidden_state = self.gru(embedded, hidden_state)
            hidden_states.append(hidden_state)
        return torch.cat(hidden_states, dim=1)
    
    def train(self, dataloader):
        optimizer = optim.Adam(self.parameters(), lr=0.001)  # Adam optimizer
        chunk_size = 30 #65535  # Taille des chunks pour l'entrainement
        a = ""

        # Boucle d'entraînement
        for epoch in range(TMGRU.epochs):
            timer = time.time()
            total_loss = 0
            cpt = 0
            for X, y, mask in dataloader:
                X, y, mask = X.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
                num_chunks = X.size(1) // chunk_size if X.size(1) % chunk_size == 0 else X.size(1) // chunk_size + 1
                hidden_state = None

                for chunk_idx in range(num_chunks):
                    optimizer.zero_grad()  # Réinitialiser les gradients
                    if hidden_state is not None:
                        hidden_state = hidden_state.detach()  # évite la rétropropagation à travers tout les chunks (utile pour la mémoire)
                    start = chunk_idx * chunk_size
                    end = min((chunk_idx + 1) * chunk_size, X.size(1))
                    op, hidden_state = self(X[:, start:end], hidden_state)  # Prédiction du modèle
                    op = op.squeeze(2)

                    mask_chunk = mask[:, start:end]
                    loss_raw = TMGRU.criterion(op.permute(0, 2, 1), y[:, start:end].long())
                    masked_loss = (loss_raw * mask_chunk).sum() / (mask_chunk.sum() + 1e-8)
                    masked_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0) # évite les gradients explosifs
                    optimizer.step()

                    total_loss += masked_loss.item()
                    cpt += 1
            if total_loss/cpt < 1e-4:
                print(f"Early stopping at epoch {epoch+1} with loss {total_loss/cpt:.4f}")
                break
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{TMGRU.epochs}], Loss: {total_loss/cpt:.4f}')
            a += f"{total_loss / cpt};"
            #print(f"Epoch {epoch+1}/{TMGRU.epochs} completed in {time.time() - timer:.2f} seconds")
        with open("losses_GRU.csv", "w") as f:
            f.write(a)

    def scores(self, y_true, y_pred): 
        combined = list(zip(y_pred, y_true))
        true_positives = sum([1 for pred, true in combined if pred == 1 and true == 1])
        false_positives = sum([1 for pred, true in combined if pred == 1 and true == 0])
        false_negatives = sum([1 for pred, true in combined if pred == 0 and true == 1])
        print(f"TP: {true_positives}, FP: {false_positives}, FN: {false_negatives} / {true_positives} & {false_positives} & {false_negatives}")

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0 # VP / (VP + FP)
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0 # VP / (VP + FN)
        f1_score = 2*true_positives / (2*true_positives + false_positives + false_negatives) if (2*true_positives + false_positives + false_negatives) > 0 else 0 # 2*VP / (2*VP + FP + FN)
        return precision, recall, f1_score
    
    def get_hyperparameters(self):
        return {
            "embedding_dim": TMGRU.embedding_dim,
            "hidden_dim": TMGRU.hidden_dim,
            "epochs": TMGRU.epochs
        }
    
    def predict(self, X, hidden_state=None):
        with torch.no_grad():
            chunk_size = 32768 #65535  # Taille des chunks pour l'entrainement
            num_chunks = X.size(0) // chunk_size if X.size(0) % chunk_size == 0 else X.size(0) // chunk_size + 1 #0 car pas de batch (normalement)
            hidden_state = None
            outputs = None
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, X.size(0))
                op, hidden_state = self(X[start_idx:end_idx], hidden_state)  # Prédiction du modèless
                proba = torch.softmax(op, dim=-1) # Convertir les logits en probabilités
                outputs = torch.cat((outputs, proba), dim=0) if outputs is not None else proba
        return outputs #raw output



def prepared_data(files, DA, DB, DELTA):
    X, y = [], []
    for file in files:
        data = parse_log_file(file)
        rough_entries = {h+i for i in range(DA,DB) for h in data['function']}
        X.extend([ data["mem"].get_byte(h) for h in rough_entries])
        y.extend([ 1 if h in data["function"] else 0 for h in rough_entries])
    y = [0]*DELTA+y[:-DELTA] # décalage de DELTA octets
    return X, y


def testing_data(files, DELTA): # toutes les données
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
