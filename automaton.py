import torch
import numpy as np
from sklearn.cluster import KMeans


class Automaton:
    def __init__(self, model, Sigma, state_number, X_val, y_val): #model is a RNN with some stuff
        hsize = model.get_hidden_size()

        H0 = model.get_all_hidden_states(X_val)
        H = H0.detach().numpy().reshape((-1,hsize))

        kmeans = KMeans(n_clusters=state_number,random_state=1)
        kmeans.fit(H)
        centers = {-1: model.get_default_hidden_state()}
        for i, c in enumerate(kmeans.cluster_centers_):
            centers[i] = torch.tensor(c).view(1, 1, hsize)

        Q = range(-1,state_number)
        delta = {(q,j): dict() for q in Q for j in Sigma}
        for (q,j) in delta:
            h = centers[q]
            k = model.step(h, j) #the new hidden layer
            p = kmeans.predict(k.detach().numpy().reshape((1,hsize)))[0] #the corresponding state
            delta[(q,j)] = int(p)  #complete the automaton, p has type np.int64 after prediction
        F = {kmeans.labels_[i] for i in range(len(kmeans.labels_)) if y_val[i] > 0 }
        self.Sigma = Sigma
        self.Q = Q #-1, 0, ..., n
        self.delta = delta #delta[ (q,k) ] = q' with q in Q, k in Sigma, q' in Q
        self.F = F #final states subseteq Q
    
    def dot(self, short = True):
        dot = "digraph {\n"
        for n in self.Q:
            color = "yellow" if n in self.F else "white"
            dot += f'N{n} [label="{n}",style=filled,fillcolor="{color}"];\n'.replace("-","_")

        blocs = {(q,p) : set() for q in self.Q for p in self.Q} #normalize a little bit the sets
        for q in self.Q:
            r = None
            start = -1
            end = -1
            for a in self.Sigma:
                end = a
                p = self.delta[(q,a)]
                if p != r:
                    if r != None:
                        blocs[(q,r)].add((start,end-1))
                    start = a
                    r = p
            blocs[(q,r)].add((start,end))
        for (q,r), sucs in blocs.items():
            if sucs:
                label = ",".join(f"{hex(s)[2:]}/{hex(e)[2:]}" if s != e else f"{hex(s)[2:]}" for (s,e) in sucs)
                dot += f'N{q} :> N{r} [label="{label}"];\n'.replace("-","_").replace(":","-")
            
        return dot + "}"

    def predict(self, X):
        y = np.zeros_like(X)
        q = -1
        for i in range(len(X)):
            q = self.delta[(q,X[i])]
            y[i] = 1 if q in self.F else 0
        return y

    def accessible(self):
        S = {-1}
        todo = [-1]
        while todo:
            q = todo.pop()
            for i in self.Sigma:
                t = self.delta[(q,i)]
                if t not in S:
                    S.add(t)
                    todo.append(t)
        return S
    
    def coaccessible(self):
        pass

    def emonde(self):
        acc = self.accessible()
        for q in self.Q:
            if q not in acc:
                for i in self.Sigma:
                    del self.delta[(q,i)]
        self.Q = acc




"""
    def __init__(self, model, Sigma, state_number, X_val, y_val): #model is a RNN with some stuff
        hsize = model.get_hidden_size()

        H0 = model.get_all_hidden_states(X_val).squeeze()
        H1 = torch.cat((H0, y_val.unsqueeze(1)), dim=1)
        H = H1.detach().numpy() #.reshape((-1,hsize))
        # z = model.predict_from_hidden_state(H0[0])

        kmeans = KMeans(n_clusters=state_number,random_state=1)
        kmeans.fit(H)

        centers = {-1: cat_hidden_to_float(model.get_default_hidden_state(), 0)}
        for i, c in enumerate(kmeans.cluster_centers_):
            centers[i] = torch.tensor(c).view(1, 1, hsize+1)

        Q = range(-1,state_number)
        delta = {(q,j): dict() for q in Q for j in Sigma}
        for (q,j) in delta:
            hb = centers[q]
            h = hb[:,:,:-1]
            k = model.step(h, j) #the new hidden layer
            b = model.predict_from_hidden_state(k)
            kb = torch.cat((k,b),dim=2)

            p = kmeans.predict(kb.detach().numpy().reshape((1,-1)))[0] #the corresponding state
            delta[(q,j)] = int(p)  #complete the automaton, p has type np.int64 after prediction
        F = {kmeans.labels_[i] for i in range(len(kmeans.labels_)) if y_val[i] > 0 }
        self.Sigma = Sigma
        self.Q = Q #-1, 0, ..., n
        self.delta = delta #delta[ (q,k) ] = q' with q in Q, k in Sigma, q' in Q
        self.F = F #final states subseteq Q
"""