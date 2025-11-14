import torch
import torch.nn as nn
import torch.optim as optim
from utils import parse_log_file, progress_bar
import pathlib
from automaton import Automaton



LETTERS = 256

class TGRU(nn.Module):
    #hyperparametres
    embedding_dim = 32
    hidden_dim = 256
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss pour les sorties binaires
    epochs = 500

    def __init__(self, num_layers=2):
        super(TGRU, self).__init__()
        self.embedding = nn.Embedding(LETTERS, TGRU.embedding_dim)
        self.gru = nn.GRU(TGRU.embedding_dim, TGRU.hidden_dim)  # batch_first=False par défaut
        self.linear = nn.Linear(TGRU.hidden_dim, 1)


    def get_hidden_size(self):
        return TGRU.hidden_dim

    def forward(self, x, hidden_state=None):
        embedded = self.embedding(x)
        output, hidden_state = self.gru(embedded, hidden_state)
        output = torch.sigmoid(self.linear(output))
        return output, hidden_state
    
    def get_default_hidden_state(self):
        return torch.zeros(1, 1, self.gru.hidden_size)

    def step(self, hidden_state, input_char):
        embedded = self.embedding(torch.tensor(input_char)).unsqueeze(0).unsqueeze(1)
        output, next_hidden_state = self.gru(embedded, hidden_state)
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
    
    def train(self, X, y):
        X = X.unsqueeze(1) # Ajout d'une dimension pour le batch
        y = y.float() # Convertir en float pour la fonction de perte
        optimizer = optim.Adam(self.parameters(), lr=0.001)  # Adam optimizer

        # Boucle d'entraînement
        for epoch in range(self.epochs):
            progress_bar(epoch+1, self.epochs)
            optimizer.zero_grad()  # Réinitialiser les gradients
            outputs, _ = model(X)  # Prédiction du modèle
            outputs = outputs.squeeze()
            loss = self.criterion(outputs, y)  # Calcul de la perte
            print(f"Loss : {loss.shape}\n{loss}")
            exit(0)
            loss.backward()  # Rétropropagation
            optimizer.step()  # Mise à jour des poids
            #if (epoch + 1) % 10 == 0:
                #print(f'Epoch [{epoch+1}/{ARNN.epochs}], Loss: {loss.item():.4f}')

    def scores(self, y_true, y_pred): 
        combined = list(zip(y_pred, y_true))
        true_positives = sum([1 for pred, true in combined if pred == 1 and true == 1])
        false_positives = sum([1 for pred, true in combined if pred == 1 and true == 0])
        false_negatives = sum([1 for pred, true in combined if pred == 0 and true == 1])

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0 # VP / (VP + FP)
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0 # VP / (VP + FN)
        f1_score = 2*true_positives / (2*true_positives + false_positives + false_negatives) if (2*true_positives + false_positives + false_negatives) > 0 else 0 # 2*VP / (2*VP + FP + FN)
        return precision, recall, f1_score



def prepared_data(files, DA, DB, DELTA):
    X, y = [], []
    for file in files:
        data = parse_log_file(file)
        rough_entries = {h+i for i in range(DA,DB) for h in data['function']}
        X.extend([ data["mem"].get_byte(h) for h in rough_entries])
        y.extend([ 1 if h in data["function"] else 0 for h in rough_entries])
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



if __name__ == "__main__":
    # Exemple d'utilisation
    model = TGRU()

    # Exemple de données
    #X = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 12, 1, 0, 1, 0, 0, 2, 0, 1, 0, 5, 1, 0, 0, 1, 3, 0, 1, 2, 0, 0, 3, 1, 0, 6, 0, 0, 10, 1, 0, 2, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
    #y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0 , 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 , 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    training_files = ["data/kernel32.log","data/msvcr100.log","data/ntdll.log"]
    testing_files = ["data/user32.log",]
    model_pth = "models/multi_gru.600.pth"
    DA,DB = -2,20 #on regarde autour de chaque fonction entre -2 octets et +20
    DELTA = 6
    X, y = prepared_data(training_files, DA, DB, DELTA)

    Xt, yt = torch.tensor(X),torch.tensor(y)

    if pathlib.Path(model_pth).is_file():
        #if learning has been done
        model = torch.load(model_pth, weights_only=False)
    else:
        #otherwise, we learn the model
        model = TGRU()
        model.train(Xt, yt)
        torch.save(model,model_pth)


    X_test, y_test = testing_data(testing_files, DELTA)
    X_t, y_t = torch.tensor(X_test), torch.tensor(y_test)
    
    with torch.no_grad():
        predicted = (model(X_t)[0].squeeze() > 0.8).int().detach().numpy()  # Seuil à 0.5 pour obtenir des 0 et des 1
        print(f'\nGRU  : {predicted}')
        print(f'Diff : {sum(abs(y_t - predicted))}')
        print(f"Scores : {model.scores(y_t, predicted)}")


    """ A = Automaton(model, list(range(LETTERS)), 1000, Xt, yt)
    A.emonde()
    print(f"Nombre d'état après émondage : {len(A.Q)} \nNombre d'états finaux : {len(A.F)}")
    #A.minimize()
    with open("hum_.dot", "w") as f:
        f.write(A.dot())
    predicted = A.predict(X)
    print(f"Auto : {predicted}")
    print(f"Diff :{sum(abs(predicted - yt.detach().numpy()))}")
    print(f"Size={len(A.Q)}") """
