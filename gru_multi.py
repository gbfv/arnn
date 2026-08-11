import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from utils import parse_log_file, progress_bar, get_device
from sklearn.metrics import f1_score


LETTERS = 257 #256 + 1 for padding
DEVICE = get_device()


class TMGRU(nn.Module):
    #hyperparametres
    embedding_dim = 10
    hidden_dim = 50
    epochs = 100

    def __init__(self, method, nbClasses):
        super(TMGRU, self).__init__()
        self.embedding = nn.Embedding(LETTERS, TMGRU.embedding_dim, padding_idx=256)  # 256 pour le padding
        self.gru = nn.GRU(TMGRU.embedding_dim, TMGRU.hidden_dim, batch_first=True)  # batch_first=False par défaut
        self.method = method

        if method == "multi-classe":
            self.multi_classe(nbClasses)
        elif method == "binaire":
            self.multi_classe(1)
        elif method == "multi-label":
            self.multi_label(nbClasses)


    def multi_classe(self, nbClasses): #nbClasses = 1 pour du binaire, >1 pour du multi-classe
        self.linear = nn.Linear(TMGRU.hidden_dim, nbClasses)
        self.nbClasses = nbClasses
        if nbClasses == 1: #TODO: voir pour gerer les poids
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(16.0))  # Binary Cross-Entropy Loss pour les sorties binaires
        else:
            self.criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss pour les sorties multi-classes

    def multi_label(self, nbClasses):
        self.nbClasses = nbClasses
        #On crée un "buffer de lecture", s'il y a un chevauchement, on prédit le second label dpar la deuxième tête
        self.heads = nn.ModuleList([nn.Linear(TMGRU.hidden_dim, nbClasses) for _ in range(2)])
        self.criterion = nn.CrossEntropyLoss() #TODO: voir pour gerer les poids


    def get_hidden_size(self):
        return TMGRU.hidden_dim

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

    def get_all_hidden_states_gpu(self, dataloader):
        chunk_size = 32768
        all_hs = []
        with torch.no_grad():
            for X, _, mask in dataloader:
                X, mask = X.to(DEVICE), mask.to(DEVICE)
                num_chunks = X.size(1) // chunk_size if X.size(1) % chunk_size == 0 else X.size(1) // chunk_size + 1
                current_hs = None

                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min((chunk_idx + 1) * chunk_size, X.size(1))

                    embedded = self.embedding(X[:, start_idx:end_idx])
                    hidden_states, current_hs = self.gru(embedded, current_hs)

                    hidden_states = hidden_states[mask[:, start_idx:end_idx].bool()].cpu()
                    all_hs = torch.cat((all_hs, hidden_states), dim=0) if all_hs != [] else hidden_states
        return all_hs
    
    ### Entrainement ####

    def loss_binaire(self, op, y, mask):
        op = op.squeeze(2)
        loss_raw = self.criterion(op, y)
        masked_loss = (loss_raw * mask).sum() / (mask.sum() + 1e-8)
        return masked_loss

    def loss_multi_classe(self, op, y, mask):
        loss = self.criterion(op.permute(0, 2, 1), y.long())
        masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return masked_loss

    def loss_multi_label(self, ops, y, mask): #TODO: porter une grande attention s'il n'y a pas de bug
        loss = [self.criterion(ops[i].permute(0, 2, 1), y[:,:, i].long()) for i in range(len(ops))] # Calcul de la perte pour chaque tête de classification
        masked_loss = [(loss[i] * mask[:,:, i]).sum() / (mask[:,:, i].sum() + 1e-8) for i in range(len(loss))] # Application du masque à chaque perte
        total_loss = sum(masked_loss)
        return total_loss

    def train(self, dataloader):
        optimizer = optim.Adam(self.parameters(), lr=0.001)  # Adam optimizer
        chunk_size = 32768 #65535  # Taille des chunks pour l'entrainement

        # Boucle d'entraînement
        for epoch in range(TMGRU.epochs):
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

                    mask_chunk = mask[:, start:end]
                    if self.method == "binaire":
                        masked_loss = self.loss_binaire(op, y[:, start:end], mask_chunk)
                    elif self.method == "multi-classe":
                        masked_loss = self.loss_multi_classe(op, y[:, start:end], mask_chunk)
            
                    masked_loss.backward()
                    #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0) # évite les gradients explosifs
                    optimizer.step()

                    total_loss += masked_loss.item()
                    cpt += 1
            if total_loss/cpt < 1e-4:
                print(f"Early stopping at epoch {epoch+1} with loss {total_loss/cpt:.4f}")
                break
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{TMGRU.epochs}], Loss: {total_loss/cpt:.4f}')

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
            somme = 0
            for i in range(self.nbClasses):
                class_true_positives = sum([1 for pred, true in combined if pred == i and true == i])
                class_false_positives = sum([1 for pred, true in combined if pred == i and true != i])
                class_false_negatives = sum([1 for pred, true in combined if pred != i and true == i])
                print(f"Class {i} - TP: {class_true_positives}, FP: {class_false_positives}, FN: {class_false_negatives}")
                somme += class_true_positives + class_false_positives + class_false_negatives
                precision = class_true_positives / (class_true_positives + class_false_positives) if (class_true_positives + class_false_positives) > 0 else 0 # VP / (VP + FP)
                recall = class_true_positives / (class_true_positives + class_false_negatives) if (class_true_positives + class_false_negatives) > 0 else 0 # VP / (VP + FN)
                f1_score = (2*class_true_positives) / (2*class_true_positives + class_false_positives + class_false_negatives) if (2*class_true_positives + class_false_positives + class_false_negatives) > 0 else 0 # 2*VP / (2*VP + FP + FN)
                f1_scores.append(f1_score)
                print(f"Class {i} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
            print(f"[*] Total somme: {somme} / Total y_pred: {len(y_pred)} / Total y_true: {len(y_true)} / Total combined: {len(combined)}")
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
                op, hidden_state = self(X[start_idx:end_idx], hidden_state) # Prédiction du modèle
                if self.method == "binaire":
                    proba = torch.sigmoid(op) # Convertir les logits en probabilités
                elif self.method == "multi-classe":
                    proba = torch.softmax(op, dim=1) # Convertir les logits en probabilités

                outputs = torch.cat((outputs, proba), dim=0) if outputs is not None else proba
        return outputs #raw output
    

    def predict_dataloader(self, dataloader, hidden_state=None):
        #all_outputs, all_labels, all_masks = [], [], []
        outputs, y_final = [], []
        chunk_size = 32768 #65535  # Taille des chunks pour l'entrainement

        with torch.no_grad():
            for X, y, mask in dataloader:
                X, y, mask = X.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
                num_chunks = X.size(1) // chunk_size if X.size(1) % chunk_size == 0 else X.size(1) // chunk_size + 1
                hidden_state = None

                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min((chunk_idx + 1) * chunk_size, X.size(1))
                    op, hidden_state = self(X[:,start_idx:end_idx], hidden_state) # Prédiction du modèle
                    if self.method == "binaire":
                        proba = torch.sigmoid(op) # Convertir les logits en probabilités
                    elif self.method == "multi-classe":
                        proba = torch.softmax(op, dim=2) # Convertir les logits en probabilités

                    clean_proba = proba[mask[:,start_idx:end_idx].bool()]
                    outputs = torch.cat((outputs, clean_proba), dim=0) if outputs != [] else clean_proba
                    y_clean = y[:,start_idx:end_idx][mask[:,start_idx:end_idx].bool()]
                    y_final = torch.cat((y_final, y_clean), dim=0) if y_final != [] else y_clean
                """    tmp_outputs.append(proba)
                batch_outputs = torch.cat(tmp_outputs, dim=1)
                flattened_outputs = batch_outputs.reshape(-1, batch_outputs.size(-1))

                all_outputs.append(flattened_outputs)
                all_labels.append(y.reshape(-1))
                all_masks.append(mask.reshape(-1))
                
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_masks = torch.cat(all_masks, dim=0) """

        return outputs, y_final