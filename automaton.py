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
        F = {q for q in Q if torch.sigmoid(model.linear(centers[q])).detach().numpy().reshape(1)[0] > 0.3}
        
        self.Sigma = Sigma
        self.Q = Q #-1, 0, ..., n
        self.delta = delta #delta[ (q,k) ] = q' with q in Q, k in Sigma, q' in Q
        self.F = F #final states subseteq Q
    
    """
    def __init__(self, Sigma, Q, F, delta):
        self.Sigma = Sigma
        self.Q = Q
        self.F = F
        self.delta = delta
    """
    
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
    
    def emonde(self):
        acc = self.accessible()
        for q in self.Q:
            if q not in acc:
                for i in self.Sigma:
                    del self.delta[(q,i)]
        self.Q = acc

    def minimize(self):
        '''
        TODO : incorrect, a revoir, 
        '''
        partitions = [{q for q in self.F}, {q for q in self.Q if q not in self.F}]
        q2partition = {q : (0 if q in self.F else 1) for q in self.Q}
        lets_cut = True
        while lets_cut:
            new_partition = []
            new_q2partition = dict()
            new_id = 0
            for S in partitions:
                profiles = dict() #mapping a profile to its corresponding states
                for q in S:
                    profile = tuple(q2partition[self.delta[(q,j)]] for j in self.Sigma) 
                    if profile not in profiles:
                        profiles[profile] = (new_id,[])
                        new_id += 1
                    the_id, list_of_qs = profiles[profile]
                    list_of_qs.append(q)
                    new_q2partition[q] = the_id

                for _,subpartition in profiles.values():
                    new_partition.append(subpartition)
            if len(new_partition) == len(partitions):
                lets_cut = False
            else:
                partitions = new_partition
                q2partition = new_q2partition

        partition2q = dict()
        init_state = q2partition[-1]
        partition2q[init_state] = -1
        new_id = 0
        F = set()
        for q in self.Q:
            if q2partition[q] not in partition2q:
                partition2q[ q2partition[q] ] = new_id 
                new_id += 1
            if q in self.F:
                F.add(partition2q[ q2partition[q]])
            
        Q = set(range(-1,new_id))
        delta = {(q,j) : None for q in Q for j in self.Sigma}
        for q in self.Q:
            for a in self.Sigma:
                state = partition2q[ q2partition[q] ]
                if delta[(state, a)] == None:
                    next_state_ = self.delta[(q,a)]
                    new_state = partition2q[ q2partition[next_state_] ]
                    delta[(state, a)] = new_state
        self.Q = Q
        self.F = F
        self.delta = delta
        return self


if __name__ == "__main__":
    A = Automaton(list(range(2)), list(range(-1,3)), {0,2}, {(-1,0): 0, (-1,1): 1, 
    (0,0): 0, (0,1): 2, 
    (1,0): 1, (1,1): 2, 
    (2,0): 0, (2,1): 2, })
    A.minimize()


