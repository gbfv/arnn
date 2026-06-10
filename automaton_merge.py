import torch
import numpy as np
import random
from sklearn.cluster import KMeans
from utils import progress_bar, get_device

class TOY_Automaton:

    def __init__(self, model, Sigma, state_number, X_val, final = set(), init_build="brute"): #model is a RNN with some stuff
        device = get_device()
        self.method = model.method
        hsize = model.get_hidden_size()
        H = None

        for x in X_val:
            with torch.no_grad():
                H0 = model.get_all_hidden_states(torch.tensor(x).to(device))
                H_temp = H0.detach().cpu().numpy().reshape((-1,hsize))
                H = np.concatenate((H, H_temp), axis=0) if H is not None else H_temp
        print(len(H), "hidden states collected")
        print(len(H[0]), "dimensions per hidden state")

        ###### TEST ZONE ###### 
        #permet de réduire le nombre de points à clusteriser, pour accélérer le processus
        #indices = np.random.choice(H.shape[0], 400000, replace=False)
        #subsampling = H[indices]
        #######################

        kmeans = KMeans(n_clusters=state_number,random_state=1)
        kmeans.fit(H)
        if init_build == "brute":
            centers = {-1: model.get_default_hidden_state()} #force un état -1 à [0,0,0,...,0]
        else:
            centers = {}
        for i, c in enumerate(kmeans.cluster_centers_):
            centers[i] = torch.tensor(c).view(1, 1, hsize)
        print("KMeans clustering done")

        if init_build == "pred":
            node_pred = kmeans.predict(model.get_default_hidden_state().detach().cpu().numpy().reshape(-1, hsize))[0]
            print(f"Initial state predicted as state {node_pred} by KMeans")
            centers[-1] = centers[node_pred] #ce node devient l'état initial
            del centers[node_pred]

        Q = {q for q in centers.keys()} #Q = range(-1,state_number)
        print(f"Q:{Q}")
        delta = {(q,j): dict() for q in Q for j in Sigma}
        sigma_tensor = torch.tensor(Sigma)
        cpt = 1
        for q in Q:
            progress_bar(cpt, len(Q))
            cpt += 1
            h = centers[q]
            with torch.no_grad():
                k = model.step_batch(h, sigma_tensor) #the new hidden layer

            p = kmeans.predict(k.squeeze(0).detach().cpu().numpy()) #the corresponding state
            for j in Sigma:
                target = int(p[j]) if init_build != "pred" or int(p[j]) != node_pred else -1 #can't update kmeans labels
                delta[(q,j)] = target  #complete the automaton, p has type np.int64 after prediction

        if model.method == "binaire":
            rank = {}
            F = {q for q in Q if torch.sigmoid(model.linear(centers[q].to(device))).detach().cpu().numpy().reshape(1)[0] > 0.8}
        elif model.method == "multi-classe":
            rank = {q : torch.argmax(model.linear(centers[q].to(device))).item() for q in Q}
            if final == set():
                F = {q for q in Q if rank[q] == model.nbClasses-1}
            else:
                F = {q for q in Q if rank[q] in final}
            #self.qualite_classe(H, centers, kmeans)
        elif model.method == "multi-label":
            rank = {q : tuple(torch.argmax(model.heads[i](centers[q].to(device))).item() for i in range(len(model.heads))) for q in Q}
            final_classes = [model.heads[i].out_features-1 for i in range(len(model.heads))]
            F = {q for q in Q for i in range(len(model.heads)) if rank[q][i] == final_classes[i]} #final states are those that predict the final class for each head of classification
            #self.qualite_classe(H, centers, kmeans)

        self.Sigma = Sigma
        self.Q = Q #-1, 0, ..., n
        self.delta = delta #delta[ (q,k) ] = q' with q in Q, k in Sigma, q' in Q
        self.F = F #final states subseteq Q
        self.rank = rank

        if init_build in ["voteF", "voteQ"]:
            if init_build == "voteF":
                voted_init = self.vote_initial_state(delta, F, Sigma)
            else:
                voted_init = self.vote_initial_state(delta, Q, Sigma)
            print(f"\nInitial state voted as state {voted_init}")
            self.clean(voted_init)

        if init_build == "find":
            found_init = self.find_initial_state(0)
            print(f"\nInitial state found as state {found_init}")
            self.clean(found_init)



    def vote_initial_state(self, delta, pop, Sigma):
        """pop: voting population (F or Q)"""
        votes = []
        for q in pop:
            for j in Sigma:
                votes.append(delta[(q,j)])
        return max(set(votes), key=votes.count) #the most voted state is likely the initial state

    def clean(self, new_init):
        #delta
        for q in self.Q:
            for j in self.Sigma:
                if self.delta[(q,j)] == new_init:
                    self.delta[(q,j)] = -1
        for j in self.Sigma:
            self.delta[(-1,j)] = self.delta[(new_init, j)]
            del self.delta[(new_init, j)]
        #Q
        self.Q.remove(new_init)
        self.Q.add(-1)
        #F
        if new_init in self.F:
            self.F.remove(new_init)
            self.F.add(-1) #TODO: à voir si on laisse ça
        #rank
        if new_init in self.rank:
            self.rank[-1] = self.rank[new_init]
            del self.rank[new_init]

    def find_initial_state(self, init_state = -1):
        states = {q: 0 for q in self.Q}

        for _ in range(1000):
            mots = random.choices(list(self.Sigma), k=1000)
            _, hist = self.predict(torch.tensor(mots), init_state=init_state)
            states[hist[-1]] += 1
        
        states = sorted(states.items(), key=lambda x: x[1], reverse=True)
        for q, count in states:
            print(f"State {q} : {count} times reached")
        return states[0][0] #the most reached state is likely the initial state


    def qualite_classe(self, H, centers, kmeans): #TODO: à revoir pour adapter à chaque tête de classification
        success, failure = [0]*9, [0]*9
        H = H[1:,:]
        pred = kmeans.predict(H)
        for i in range(len(H)):
            class_pred = round(H[i][-1].item())
            if class_pred == round(centers[pred[i]][:,:,-1].item()):
                success[class_pred] += 1
            else:
                failure[class_pred] += 1
        for i in range(9):
            print(f"Class {i} : {success[i]} successes, {failure[i]} failures")

    

    def dot(self):
        dot = "digraph {\n"
        for n in self.Q:
            color = "yellow" if n in self.F else "white"
            rank = ""
            if self.method == "multi-classe":
                rank = f'/{self.rank[n]}'
            elif self.method == "multi-label":
                rank = f'/{"-".join(str(r) for r in self.rank[n])}'
            dot += f'N{n} [label="{n}{rank}",style=filled,fillcolor="{color}"];\n'.replace("-","_")

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
                label = ",".join(f"{chr(s+97)}/{chr(e+97)}" if s != e else f"{chr(s+97)}" for (s,e) in sucs)
                #label = label.replace("0","a").replace("1","b").replace("2","c").replace("3","d").replace("4","e").replace("5","f")
                dot += f'N{q} :> N{r} [label="{label}"];\n'.replace("-","_").replace(":","-")

        return dot + "}"


    def predict(self, X, init_state = -1):
        Xt = X.detach().cpu().numpy()
        if self.method in ["binaire", "multi-classe"]:
            y = np.zeros_like(Xt)
        elif self.method == "multi-label":
            nb_labels = len(self.rank[list(self.rank.keys())[0]])
            y = np.zeros((len(Xt), nb_labels))
        q = init_state
        history = [init_state]

        for i in range(len(Xt)):
            try:
                q = self.delta[(q,Xt[i])]
            except KeyError:
                print(f"{self.delta}\nTransition ({q}, {Xt[i]}) not found in delta.\n")
                exit(1)
            if self.method == "binaire":
                y[i] = 1 if q in self.F else 0
            elif self.method in ["multi-classe", "multi-label"]:
                y[i] = self.rank[q]
            history.append(q)
        return y, history

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
        self.F = self.F.intersection(acc) #cut final states too


    def minimize(self):
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

        rank = {}
        if self.method != "binaire":
            for q in Q:
                part = [v for v, part in partition2q.items() if part == q][0] #find the partition corresponding to q
                rank[q] = self.rank[partitions[part][0]]

        self.rank = rank
        self.Q = Q
        self.F = F
        self.delta = delta
        return self





########### TODO: Tester ces fonctions pour chaque methode ###########

    def path_to_finals(self, init_state = -1):
        '''BFS to find shortest paths to finals states'''
        queue = [(init_state, [[],[init_state]])] # (state, path to state)
        visited = set()
        paths = dict()
        
        while queue:
            state, path = queue.pop(0)

            if state in visited:
                continue
            visited.add(state)

            if state in self.F:
                paths[state] = path

            # letters leading to next states only
            filtered_j = [j for (q,j) in self.delta.keys() if q == state]
            for j in filtered_j:
                next_state = self.delta[(state, j)]
                if next_state not in visited:
                    queue.append((next_state, [path[0]+[f"0x{j:02X}"], path[1]+[next_state]]))
        print(f"Visited states: {len(visited)}")
        return paths


    def reverse_delta(self):
        reverse = {}
        for (src, j), dst in self.delta.items():
            if (dst, j) not in reverse:
                reverse[(dst, j)] = [src]
            else:
                reverse[(dst, j)].append(src)
        return reverse


    def get_paths(self, length, init_state = -1, delta = None):
        """BFS to find paths of a given length"""
        if delta is None:
            delta = self.delta

        queue = [([init_state], "")] # (path, string)
        path = []
        
        while queue:
            states, string = queue.pop(0)
            filtered_j = [j for (q,j) in delta.keys() if q == states[-1]]

            for j in filtered_j:
                next_states = delta[(states[-1], j)]

                for next_state in next_states:
                    if len(states) == length-1:
                        path.append((f"0x{j:02X}"+string, states+[next_state]))
                    else:
                        final = list(reversed(states+[next_state]))
                        queue.append((final, string+f"0x{j:02X}"))
        return path


    def score_path(self, data, score={}):

        """ if not score:
            for (q,j) in self.delta.keys():
                score[(q,j)] = 0 """

        for X in data:
            _, h = self.predict(torch.tensor(X))
            for i in range(len(X)):
                if (h[i], X[i]) not in score:
                    score[(h[i], X[i])] = 1
                else:
                    score[(h[i], X[i])] += 1
        return score
                




####### Deprecated functions #########


    def predict_flow(self, X, y_true, window_size=22):
        """
        Only for batch mode
        """
        Xt = X.detach().cpu().numpy()
        y = np.zeros_like(Xt)
        starting_index = 0 # starting index of the window

        hallucinations = 0
        init_state = -1

        while starting_index + window_size <= len(Xt):
            sample = torch.tensor(Xt[starting_index:starting_index + window_size])
            pred, hist = self.predict(sample, init_state=init_state)
            if pred[-1] == 1: # function found
                if sum(y_true[starting_index:starting_index + window_size]) == 0: #hallucination check
                    hallucinations += 1
                y[starting_index + 2] = 1
                starting_index += 22 # jump to next window
                #init_state = -1
            else:
                starting_index += 1 # slide the window by 1
                #init_state = hist[1]

        print(f"Total hallucinations: {hallucinations}")
        return y
                

    def predict_flowV2(self, X, y_true, window_size=22):
        """
        Only for batch mode
        """
        Xt = X.detach().cpu().numpy()
        y = np.zeros_like(Xt)
        starting_index = 0 # starting index of the window

        hallucinations = 0
        init_state = -1

        while starting_index + window_size <= len(Xt):
            sample = torch.tensor(Xt[starting_index:starting_index + window_size])
            pred, hist = self.predict(sample, init_state=init_state)
            if pred[-1] == 1: # function found
                if sum(y_true[starting_index:starting_index + window_size]) == 0: #hallucination check
                    hallucinations += 1
                y[starting_index + 2] = 1
            starting_index += 1


        print(f"Total hallucinations: {hallucinations}")
        return y
