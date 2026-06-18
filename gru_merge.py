import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from utils import get_device

from log_custom import add_log
import time

LETTERS = 6
DEVICE = get_device()


class TOY_GRU(nn.Module):
    #hyperparametres (les passer en argument de la classe ?)
    embedding_dim:int = 2
    hidden_dim = 50
    epochs = 500

    def __init__(self, method, nbClasses=None, mots=None,weights=None):
        super(TOY_GRU, self).__init__()
        self.embedding = nn.Embedding(LETTERS, TOY_GRU.embedding_dim)  # 256 pour le padding
        self.gru = nn.GRU(TOY_GRU.embedding_dim, TOY_GRU.hidden_dim, batch_first=True)  # batch_first=False par défaut
        self.method = method

        if method == "multi-classe":
            self.multi_classe(nbClasses)
        elif method == "binaire":
            self.multi_classe(1)
        elif method == "multi-label":
            self.multi_label(mots, weights)


    def multi_classe(self, nbClasses): #nbClasses = 1 pour du binaire, >1 pour du multi-classe
        self.linear = nn.Linear(TOY_GRU.hidden_dim, nbClasses)
        self.nbClasses = nbClasses
        if nbClasses == 1: #TODO: voir pour gerer les poids
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(16.0))  # Binary Cross-Entropy Loss pour les sorties binaires
        else:
            self.criterion = nn.CrossEntropyLoss()

    def multi_label(self, mots, weights):
        length_mot = len(mots[0])+1 #position dans le mot + 0
        self.heads = nn.ModuleList([nn.Linear(TOY_GRU.hidden_dim, length_mot) for _ in range(len(mots))]) # une tête de classification par mot
        if not weights:
            self.criterion = nn.CrossEntropyLoss()
        else: # Gère les poids
            self.criterion = nn.ModuleList([nn.CrossEntropyLoss(weight=torch.tensor(weight)) for weight in weights])


    def get_hidden_size(self):
        return TOY_GRU.hidden_dim


    def forward(self, x, hidden_state=None):
        embedded = self.embedding(x)
        output, hidden_state = self.gru(embedded, hidden_state)
        if self.method == "multi-label":
            outputs = [linear(output) for linear in self.heads] # envoi les données vers chaque tête de classification
            return outputs, hidden_state
        else:
            output = self.linear(output)
        return output, hidden_state
    

    def get_default_hidden_state(self):
        return torch.zeros(1, 1, self.gru.hidden_size).to(DEVICE)


    def step(self, hidden_state, input_char): #TODO à checker
        embedded = self.embedding(torch.tensor(input_char).to(DEVICE)).unsqueeze(0).unsqueeze(1)
        output, next_hidden_state = self.gru(embedded, hidden_state.to(DEVICE))
        return next_hidden_state


    def step_batch(self, hidden_state, input_char): #TODO à checker
            embedded = self.embedding(input_char.to(DEVICE)).unsqueeze(1)
            hs = hidden_state.expand(-1, embedded.size(0), -1).contiguous().to(DEVICE)
            output, next_hidden_state = self.gru(embedded, hs)
            return next_hidden_state


    def get_hidden_state(self, x): # TODO à checker
        embedded = self.embedding(x)
        _, hidden_state = self.gru(embedded)
        return hidden_state
    

    def get_all_hidden_states(self, x): # TODO: à checker
        hidden_states = []
        hidden_state = self.get_default_hidden_state()
        for char in x:
            embedded = self.embedding(char).unsqueeze(0).unsqueeze(1)
            _, hidden_state = self.gru(embedded, hidden_state)
            hidden_states.append(hidden_state)
        return torch.cat(hidden_states, dim=1)
    
    ### Entrainement #####

    def loss_binaire(self, op, y):
        op = op.squeeze(2)
        loss = self.criterion(op, y)
        return loss
    
    def loss_multi_classe(self, op, y):
        loss = self.criterion(op.permute(0, 2, 1), y.long())
        return loss

    def loss_multi_label(self, ops, y):
        if isinstance(self.criterion, nn.ModuleList):
            loss = [self.criterion[i](ops[i].permute(0, 2, 1), y[:,:, i].long()) for i in range(len(ops))] # Calcul de la perte pour chaque tête de classification avec les poids
        else:
            loss = [self.criterion(ops[i].permute(0, 2, 1), y[:,:, i].long()) for i in range(len(ops))] # Calcul de la perte pour chaque tête de classification
        total_loss = sum(loss)
        return total_loss

    def train(self, dataloader):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)  # Adam optimizer
        chunk_size = 1000 #65535  # Taille des chunks pour l'entrainement

        # Boucle d'entraînement
        for epoch in range(TOY_GRU.epochs):
            time_begin_epoch = time.time()
            epoch_loss = 0
            cpt = 0
            for X, y, _ in dataloader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                num_chunks = X.size(1) // chunk_size if X.size(1) % chunk_size == 0 else X.size(1) // chunk_size + 1
                hidden_state = None

                for chunk_idx in range(num_chunks):
                    optimizer.zero_grad()  # Réinitialiser les gradients
                    if hidden_state is not None:
                        hidden_state = hidden_state.detach()  # évite la rétropropagation à travers tout les chunks (utile pour la mémoire)
                    start = chunk_idx * chunk_size
                    end = min((chunk_idx + 1) * chunk_size, X.size(1))
                    op, hidden_state = self(X[:, start:end], hidden_state)  # Prédiction du modèle

                    if self.method == "binaire":
                        loss = self.loss_binaire(op, y[:, start:end])
                    elif self.method == "multi-classe":
                        loss = self.loss_multi_classe(op, y[:, start:end])
                    elif self.method == "multi-label":
                        loss = self.loss_multi_label(op, y[:, start:end])

                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0) # évite les gradients explosifs
                    optimizer.step()

                    epoch_loss += loss.item()
                    cpt += 1
            time_end_epoch = time.time()
            add_log("epoch_time","epoch_time.log",f"EPOCH:{epoch},TIME:{time_end_epoch - time_begin_epoch}")
            add_log("epoch_loss","epoch_loss.log",f"EPOCH:{epoch},LOSS:{epoch_loss/cpt}")
            if epoch_loss/cpt < 1e-4:
                print(f"Early stopping at epoch {epoch+1} with loss {epoch_loss/cpt:.4f}")
                break
            if (epoch + 1) % 10 == 0:
                time_remaining = (TOY_GRU.epochs - epoch+1) * (time_end_epoch - time_begin_epoch)
                hours = int(time_remaining / 3600)
                mins = int((int(time_remaining) % 3600)/60)
                secs = int(time_remaining) % 60
                print(f'Epoch [{epoch+1}/{TOY_GRU.epochs}], Loss: {epoch_loss/cpt:.4f},Time remaining:{hours}h {mins}min {secs}s')

    ##### Evaluation #####

    def scores_binaire(self, y_true, y_pred):
        combined = list(zip(y_pred, y_true))
        true_positives = sum([1 for pred, true in combined if pred == 1 and true == 1])
        false_positives = sum([1 for pred, true in combined if pred == 1 and true == 0])
        false_negatives = sum([1 for pred, true in combined if pred == 0 and true == 1])
        print(f"TP: {true_positives}, FP: {false_positives}, FN: {false_negatives} / {true_positives} & {false_positives} & {false_negatives}")

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0 # VP / (VP + FP)
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0 # VP / (VP + FN)
        f1_score = 2*true_positives / (2*true_positives + false_positives + false_negatives) if (2*true_positives + false_positives + false_negatives) > 0 else 0 # 2*VP / (2*VP + FP + FN)
        return precision, recall, f1_score

    def scores_multi_classe(self, y_true, y_pred):
            combined = list(zip(y_pred, y_true))
            f1_scores = []

            for i in range(self.nbClasses):
                class_true_positives = sum([1 for pred, true in combined if pred == i and true == i])
                class_false_positives = sum([1 for pred, true in combined if pred == i and true != i])
                class_false_negatives = sum([1 for pred, true in combined if pred != i and true == i])
                print(f"Class {i} - TP: {class_true_positives}, FP: {class_false_positives}, FN: {class_false_negatives}")

                precision = class_true_positives / (class_true_positives + class_false_positives) if (class_true_positives + class_false_positives) > 0 else 0 # VP / (VP + FP)
                recall = class_true_positives / (class_true_positives + class_false_negatives) if (class_true_positives + class_false_negatives) > 0 else 0 # VP / (VP + FN)
                f1_score = (2*class_true_positives) / (2*class_true_positives + class_false_positives + class_false_negatives) if (2*class_true_positives + class_false_positives + class_false_negatives) > 0 else 0 # 2*VP / (2*VP + FP + FN)
                f1_scores.append(f1_score)
                print(f"Class {i} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

            macro_f1 = sum(f1_scores) / len(f1_scores)
            return macro_f1, f1_scores

    def scores_multi_label(self, y_true, y_pred): #F1 Macro
        macro = []
        macro_last = []
        for i in range(len(y_true[0])):
            print(f"--- Rapport pour la Tête {i} ---")
            list_f1 = f1_score([row[i] for row in y_true], [row[i] for row in y_pred], average=None)
            print(list_f1)
            macro_last.append(list_f1[-1])
            mf1 = f1_score([row[i] for row in y_true], [row[i] for row in y_pred], average='macro')
            print(f"F1 Macro: {mf1:.4f}")
            macro.append(mf1)
        
        print(f"[*] F1 Macro Moyenne: {sum(macro)/len(macro):.4f}")
        print(f"[*] F1 Macro Moyenne pour la classe finale: {sum(macro_last)/len(macro_last):.4f}")
        
        # exact match
        exact_matches = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        print(f"[*] Exact Match Accuracy: {exact_matches / len(y_true):.4f}")

        # distance de Hamming
        hamming_distance = 0
        for i in range(len(y_true)):
            for j in range(len(y_true[0])):
                hamming_distance += 1 if y_true[i][j] != y_pred[i][j] else 0
        print(f"[*] Hamming Distance: {hamming_distance/(len(y_true)*len(y_true[0])):.4f}")

    def scores(self, y_true, y_pred):
        if self.method == "binaire":
            return self.scores_binaire(y_true, y_pred)
        elif self.method == "multi-classe":
            return self.scores_multi_classe(y_true, y_pred)
        elif self.method == "multi-label":
            return self.scores_multi_label(y_true, y_pred)

    ######################

    def get_hyperparameters(self):
        return {
            "embedding_dim": TOY_GRU.embedding_dim,
            "hidden_dim": TOY_GRU.hidden_dim,
            "epochs": TOY_GRU.epochs
        }


    def predict(self, X, hidden_state=None):
        with torch.no_grad():
            chunk_size = 1000  # Taille des chunks pour l'entrainement
            num_chunks = X.size(0) // chunk_size if X.size(0) % chunk_size == 0 else X.size(0) // chunk_size + 1 #0 car pas de batch (normalement)
            hidden_state = None
            outputs = None
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, X.size(0))
                op, hidden_state = self(X[start_idx:end_idx], hidden_state) # Prédiction du modèle
                if self.method == "binaire":
                    proba = torch.sigmoid(op) # Convertir les logits en probabilités
                elif self.method == "multi-classe":
                    proba = torch.softmax(op, dim=1) # Convertir les logits en probabilités
                elif self.method == "multi-label":
                    proba = [torch.softmax(head, dim=-1) for head in op] # Convertir les logits en probabilités

                outputs = torch.cat((outputs, proba), dim=0) if outputs is not None else proba
        return outputs #raw output

    def give_f1_scores(self, y_true, y_pred):
        if self.method == "binaire":
            return self.scores_binaire(y_true, y_pred)
        elif self.method == "multi-classe":
            return self.scores_multi_classe(y_true, y_pred)[1]
        elif self.method == "multi-label":
            return [f1_score([row[i] for row in y_true], [row[i] for row in y_pred], average=None) for i in range(len(y_true[0]))]


