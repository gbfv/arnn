import torch
import torch.nn as nn
import torch.optim as optim
from automaton import Automaton
from utils import parse_log_file
import time

class ARNN(nn.Module):
    #hyperparametres
    embedding_dim = 32
    hidden_dim = 256
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss pour les sorties binaires
    epochs = 500
        
    def __init__(self, num_layers=2):
        super(ARNN, self).__init__()
        self.embedding = nn.Embedding(256, ARNN.embedding_dim)#.to(self.device)
        self.rnn = nn.RNN(ARNN.embedding_dim, ARNN.hidden_dim)#.to(self.device)  # batch_first=False par défaut
        self.linear = nn.Linear(ARNN.hidden_dim, 1)#.to(self.device)

    def get_hidden_size(self):
        return ARNN.hidden_dim

    def forward(self, x, hidden_state=None):
        embedded = self.embedding(x)#.to(self.device))
        output, hidden_state = self.rnn(embedded, hidden_state)
        output = torch.sigmoid(self.linear(output))
        return output, hidden_state

    def get_default_hidden_state(self):
        return torch.zeros(1, 1, self.rnn.hidden_size)

    def step(self, hidden_state, input_char):
        embedded = self.embedding(torch.tensor(input_char)).unsqueeze(0).unsqueeze(1)
        output, next_hidden_state = self.rnn(embedded, hidden_state)
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

    def predict_from_hidden_state(self, hidden_state):
        output = torch.sigmoid(self.linear(hidden_state))
        return output

    def train(self, X, y):
        X = X.unsqueeze(1) # Ajout d'une dimension pour le batch
        y = y.float() # Convertir en float pour la fonction de perte
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

        # Boucle d'entraînement
        for epoch in range(ARNN.epochs):
            #t0 = time.clock_gettime(time.CLOCK_MONOTONIC)
            optimizer.zero_grad()  # Réinitialiser les gradients
            outputs, _ = model(X)  # Prédiction du modèle
            outputs = outputs.squeeze()
            loss = ARNN.criterion(outputs, y)#.to(self.device))  # Calcul de la perte
            loss.backward()  # Rétropropagation
            optimizer.step()  # Mise à jour des poids
            #print(time.clock_gettime(time.CLOCK_MONOTONIC) - t0)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{ARNN.epochs}], Loss: {loss.item():.4f}')
            


if __name__ == "__main__":
    # Exemple d'utilisation
    #
    data = parse_log_file("data/kernel32.log")
    rough_entries = {h+i for i in range(-2,20) for h in data['function']}
    X  = [ data["mem"].get_byte(h) for h in rough_entries]
    y = [
        1 if (h+2) in data["function"] else 0 
        for h in rough_entries
    ]
    y = [0]*6+y[:-6]

    # Exemple de données
    #X = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 12, 1, 0, 1, 0, 0, 2, 0, 1, 0, 5, 1, 0, 0, 1, 3, 0, 1, 2, 0, 0, 3, 1, 0, 6, 0, 0, 10, 1, 0, 2, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
    #y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0 , 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 , 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    Xt,yt = torch.tensor(X),torch.tensor(y)
    
    #learning
    #model = ARNN()
    #model.train(Xt, yt)
    #torch.save(model, "model.256.pth", weight_only=False)
    
    model = torch.load('model.256.pth', weights_only=False)
    

    with torch.no_grad():
        predicted = (model(Xt)[0].squeeze() > 0.5).int().detach().numpy()  # Seuil à 0.5 pour obtenir des 0 et des 1
        #print(f'RNN  : {predicted}')
        #print(f'Terr : {y}')
        print(f'Diff : {sum(abs(yt - predicted))}')

    A = Automaton(model, list(range(256)), 200, Xt, yt)
    with open("hum_2.dot", "w") as f:
        f.write(A.dot())
    print(f"Auto : {A.predict(X)}")
    print(f"Diff :{sum(abs(A.predict(X) - yt.detach().numpy()))} / sum={sum(A.predict(X))}")
    A.emonde()
    print(f"len(A) = {len(A.Q)}")

"""
#checks that h6 = r (* via single steps or all in once *)
h0 = model.get_default_hidden_state()
h1 = model.step(0, h0)
h2 = model.step(1, h1)
h3 = model.step(0, h2)
h4 = model.step(1, h3)
h5 = model.step(0, h4)
h6 = model.step(0, h5)

for h in [h0, h1, h2, h3, h4, h5, h6]:
    print(h)

r = model.get_hidden_state(torch.tensor([0, 1, 0, 1, 0, 0]))
print(r)
"""

