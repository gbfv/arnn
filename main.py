import subprocess
from xml.parsers.expat import model
import torch
import pathlib
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
import json
from collections import Counter
import math
import time

from rnn import ARNN
from rnn_batch import BRNN
from gru_batch import BGRU
from gru_multi import TMGRU
from automaton import Automaton
from logDataset import LogDataset, prepared_data, testing_data, data_instructions, testing_data_multi
from nfa import NFA
from utils import get_device



DEVICE = get_device()




#################################################
########## Batch et Concat inutile ###########


def rnn_build(name, training_files, whitelist=True, save=True, training=True):
    model_path = f"models/{name}.pth"
    # Loading model
    if pathlib.Path(model_path).is_file():
        return torch.load(model_path, weights_only=False).to(DEVICE)
    
    # Building model
    dataset = LogDataset(files=training_files, whitelist=whitelist)
    dataset.prep_concat(training=training)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True, collate_fn=pad_batch)
    print(f"Fichier chargé pour l'entraînement : {dataset.used_files}")

    model = ARNN().to(DEVICE)
    print(f"Hyperparamètres : {model.get_hyperparameters()}")
    model.train(dataloader)
    if save:
        torch.save(model, model_path)
    return model


def batch_rnn_build(name, training_files, batch_size=1024, whitelist=True, save=True):
    model_path = f"models/{name}.pth"
    # Loading model
    if pathlib.Path(model_path).is_file():
        return torch.load(model_path, weights_only=False).to(DEVICE)
    
    # Building model
    dataset = LogDataset(files=training_files, whitelist=whitelist)
    dataset.prep_batch(DA=-2, DB=20)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Fichier chargé pour l'entraînement : {dataset.used_files}")

    model = BRNN().to(DEVICE)
    print(f"Hyperparamètres : {model.get_hyperparameters()}")
    model.train(dataloader)
    if save:
        torch.save(model, model_path)
    return model


def batch_gru_build(name, training_files, whitelist=True, save=True, DA=-2, DB=20):
    model_path = f"models/{name}.pth"
    # Loading model
    if pathlib.Path(model_path).is_file():
        return torch.load(model_path, weights_only=False).to(DEVICE)
    
    # Building model
    dataset = LogDataset(files=training_files, whitelist=whitelist)
    dataset.prep_batch(DA=DA, DB=DB)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
    print(f"Fichier chargé pour l'entraînement : {dataset.used_files}")

    model = BGRU().to(DEVICE)
    print(f"Hyperparamètres : {model.get_hyperparameters()}")
    model.train(dataloader)
    if save:
        torch.save(model, model_path)
    return model

def test_model_batch(model_name, files, batch_size=1024, whitelist=True, DA=-2, DB=20):
    model = load_model(model_name)
    
    # Preparing test data
    dataset = LogDataset(files=files, whitelist=whitelist)
    dataset.prep_batch(DA=DA, DB=DB)
    print(f"Taille d'une séquence de test : {len(dataset.data[0])}")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    y_t = torch.tensor(dataset.labels).numpy()

    with torch.no_grad():
        predicted = (model.predict(dataloader).squeeze() > 0.8).int().detach().cpu().numpy()  # Seuil à 0.5 pour obtenir des 0 et des 1
        print(f'\nRNN  : {predicted}')
        print(f'Diff : {sum(abs(y_t - predicted))}')
        prec, recall, f1 = model.scores(y_t, predicted)
        print(f"Scores : {prec}, {recall}, {f1}")
        return prec, recall, f1

def batch_automaton_build(model_name, files, states=1000, DA=-2, DB=20):
    model = load_model(model_name)

    auto_path = f"automate/{model_name}_{states}.pkl"
    if pathlib.Path(auto_path).is_file():
        with open(auto_path, "rb") as f:
            return pickle.load(f)
   
    #Preparing data
    dataset = LogDataset(files=files, whitelist=True)
    dataset.prep_batch(DA=DA, DB=DB)
    print(f"Taille d'une séquence de test : {len(dataset.data[0])}")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
    y_t = torch.tensor(dataset.labels).numpy()

    print("Construction de l'automate...\n")
    A = Automaton(model, list(range(256)), states, dataloader, y_t)
    A.emonde()
    with open(auto_path, "wb") as f:
        pickle.dump(A, f)
    
    print(f"Automate construit.\nNombre d'état après émondage : {len(A.Q)} \nNombre d'états finaux : {len(A.F)}")
    return A



def test_automaton_batch(automaton_name, files, model, DA=-2, DB=20):
    auto_path = f"automate/{automaton_name}.pkl"
    if pathlib.Path(auto_path).is_file():
        with open(auto_path, "rb") as f:
            A = pickle.load(f)
    else:
        raise FileNotFoundError(f"Automaton file {auto_path} not found. Please build the automaton before testing.")

    print(f"Testing automaton {automaton_name}...\n")

    # Preparing test data
    dataset = LogDataset(files=files, whitelist=True)
    dataset.prep_batch(DA=DA, DB=DB)
    print(f"Taille d'une séquence de test : {len(dataset.data[0])}")
    y_t = torch.tensor(dataset.labels).numpy()

    predicted_auto = []
    for x in dataset.data:
        p, _  = A.predict(torch.tensor(x))
        predicted_auto.append(p[-1])
    
    print(f'Diff : {sum(abs(y_t - predicted_auto))}')
    prec, recall, f1 = model.scores(y_t, predicted_auto)
    print(f"Scores : Precision: {prec}, Recall: {recall}, F1-score: {f1}")
    return prec, recall, f1

#################################################
#################################################



def pad_batch(batch):
    PADDING_ID_Y = 100
    sequences_x = [item[0] for item in batch]
    sequences_y = [item[1] for item in batch]
     
    X_padded = pad_sequence(sequences_x, padding_value=256, batch_first=True)
    Y_padded = pad_sequence(sequences_y, padding_value=PADDING_ID_Y, batch_first=True)
    
    # creating a mask to ignore padding in loss computation
    mask = (Y_padded != PADDING_ID_Y).float()
    Y_padded[Y_padded == PADDING_ID_Y] = 0.0
    
    return X_padded, Y_padded, mask

def load_model(name, path="models"):
    model_path = f"{path}/{name}.pth"
    # Loading model
    if pathlib.Path(model_path).is_file():
        model = torch.load(model_path, weights_only=False).to(DEVICE)
    else:
        raise FileNotFoundError(f"Model file {model_path} not found. Please train the model before testing.")
    return model


def gru_build(name, training_files, method, path, save=True, instr=False, DELTA=6):
    model_path = f"models/{name}.pth"
    # Loading model
    if pathlib.Path(model_path).is_file():
        return torch.load(model_path, weights_only=False).to(DEVICE)
    print("Modèle non trouvé. Construction du modèle...\n")
    # Building model
    dataset = LogDataset(path=path, files=training_files)
    if not instr:
        if method == "binaire":
            dataset.prep_fs(DELTA=DELTA, reverse=False)
        elif method == "multi-classe":
            dataset.prep_multi(DELTA=DELTA)
    else: # instructions
        if method == "binaire":
            dataset.prep_instr()
        elif method == "multi-classe":
            dataset.prep_instr_multiclass()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=pad_batch)
    print(f"Fichier chargé pour l'entraînement : {len(dataset.used_files)}\nMéthode d'entraînement : {method}")

    print("Construction du modèle TMGRU...\n")
    model = TMGRU(method, DELTA+1).to(DEVICE)
    print(f"Hyperparamètres : {model.get_hyperparameters()}")
    model.train(dataloader)
    if save:
        torch.save(model, model_path)
    return model



def test_model(model_name, path, files, instr=False, DELTA=6):
    model = load_model(model_name)
    
    method = model.method
    # Preparing test data
    y_test = []
    predicted = []
    for file in files:
        if not instr:
            if method == "binaire":
                X, y = testing_data(f"{path}/{file}", DELTA=DELTA)
            elif method == "multi-classe":
                X, y = testing_data_multi(f"{path}/{file}", DELTA=DELTA)
                
            y_test.extend(y)
            print(f"{len(X)} items in testing dataset")
            
            X_t = torch.tensor(X).to(DEVICE)

            with torch.no_grad():
                if method == "binaire":
                    pred = (model.predict(X_t).squeeze() > 0.8).int().detach().cpu().numpy()
                elif method == "multi-classe":
                    pred = torch.argmax(model.predict(X_t), dim=1).int().detach().cpu().numpy() # Prédiction de la classe avec la plus haute probabilité
                predicted.extend(pred)
        else: # instructions
            X, y = data_instructions(f"{path}/{file}")
            for y_i in y:
                y_test.extend(y_i)
            print(f"{sum(len(x) for x in X)} items in testing dataset")
            
            with torch.no_grad():
                for x_i in X:
                    X_t = torch.tensor(x_i).to(DEVICE)
                    pred = (model.predict(X_t).squeeze() > 0.8).int().detach().cpu().numpy()
                    predicted.extend(pred)
    print("Taille de y_test : ", len(y_test))
    print(f"Scores : {model.scores(y_test, predicted)}")


def test_model_dataloader(model_name, path, files, instr=False, DELTA=6):
    model = load_model(model_name)
    
    method = model.method
    # Preparing test data
    dataset = LogDataset(path=path, files=files)
    if instr:
        if method == "binaire":
            dataset.prep_instr()
        elif method == "multi-classe":
            dataset.prep_instr_multiclass()
    elif method == "binaire":
        dataset.prep_fs(DELTA=DELTA)
    elif method == "multi-classe":
        dataset.prep_multi(DELTA=DELTA)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=pad_batch)
    pred, y_test = model.predict_dataloader(dataloader)
    print(pred.shape, y_test.shape)

    if method == "binaire":
        pred = (pred.squeeze() > 0.8)
    elif method == "multi-classe":
        pred = torch.argmax(pred, dim=1)
    
    """ print(f"[*] pred shape: {pred.shape}, y_test shape: {y_test.shape}, mask shape: {mask.shape}")
    cleaned_y_test = y_test[mask.bool()].tolist()
    cleaned_pred = pred[mask.bool()].tolist()
    print("Taille clean_y_test : ", len(cleaned_y_test), " / Taille clean_pred : ", len(cleaned_pred)) """
    pred = pred.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()
    print(f"Taille y_test : {len(y_test)} / Taille pred : {len(pred)}")

    print(f"Scores : {model.scores(y_test, pred)}")


### Automaton functions ###

#deprecated
def automaton_build(model_name, path, files, states=1000, DELTA=6, init_build="brute"):
    model = load_model(model_name)

    auto_path = f"automate/{model_name}_{states}.pkl"
    if pathlib.Path(auto_path).is_file():
        with open(auto_path, "rb") as f:
            return pickle.load(f)
   
    method = model.method
    #Preparing data
    X_auto = []
    item = 0
    for file in files:
        if method == "binaire":
            X, _ = testing_data(f"{path}/{file}", DELTA=DELTA)
        elif method == "multi-classe":
            X, _ = testing_data_multi(f"{path}/{file}", DELTA=DELTA)
        X_auto.append(X)
        item += len(X)
    print(f"{item} items in training dataset")
    
    print("Construction de l'automate...\n")

    A = Automaton(model, list(range(256)), states, X_auto, init_build=init_build)
    A.emonde()

    with open(auto_path, "wb") as f:
        pickle.dump(A, f)
    
    print(f"Automate construit.\nNombre d'état après émondage : {len(A.Q)} \nNombre d'états finaux : {len(A.F)}")
    return A


def automaton_build_dataloader(model_name, path, files, instr=False, states=1000, DELTA=6, init_build="brute"):
    model = load_model(model_name)

    auto_path = f"automate/{model_name}_{states}.pkl"
    if pathlib.Path(auto_path).is_file():
        with open(auto_path, "rb") as f:
            return pickle.load(f)

    method = model.method
    # Preparing test data
    dataset = LogDataset(path=path, files=files)

    if method == "binaire":
        dataset.prep_fs(DELTA=DELTA)
    elif method == "multi-classe":
        dataset.prep_multi(DELTA=DELTA)
    print(f"{sum(len(x) for x in dataset.data)} items in training dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=pad_batch)

    A = Automaton(model, list(range(256)), states, dataloader, init_build=init_build)
    A.emonde()

    with open(auto_path, "wb") as f:
        pickle.dump(A, f)

    print(f"Automate construit.\nNombre d'état après émondage : {len(A.Q)} \nNombre d'états finaux : {len(A.F)}")
    return A


def test_automaton(automaton_name, path, files, model, init_state=-1, DELTA=6):
    auto_path = f"automate/{automaton_name}.pkl"
    if pathlib.Path(auto_path).is_file():
        with open(auto_path, "rb") as f:
            A = pickle.load(f)
    else:
        raise FileNotFoundError(f"Automaton file {auto_path} not found. Please build the automaton before testing.")

    print(f"Testing automaton {automaton_name}...\n")

    method = model.method
    #Preparing data
    X_test, y_test = [], []
    for file in files:
        if method == "binaire":
            X, y = testing_data(f"{path}/{file}", DELTA=DELTA)
        elif method == "multi-classe":
            X, y = testing_data_multi(f"{path}/{file}", DELTA=DELTA)
        #Voir pour add au lieu de extend, pour éviter qu'un echantillon en influence d'autres
        X_test.extend(X)
        y_test.extend(y)
    
    X_t, y_t = torch.tensor(X_test).to(DEVICE), torch.tensor(y_test)
    print(f"{len(X_test)} items in testing dataset")
    predicted, _ = A.predict(X_t, init_state=init_state)
    print(model.scores(y_t, predicted))



###### OTHER FUNCTIONS ######



def graph_final_states(auto_name, initial_state=-1):
    auto_path = f"automate/{auto_name}.pkl"
    with open(auto_path, "rb") as f:
        A = pickle.load(f)


    finals = A.path_to_finals(init_state=initial_state)
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
        json.dump(json_save, f)


#### Other functions ####




def big_run(build, test, model_name, automaton, states):
    """
    model_name : name of the model to load/build
    automaton : prefix of the automaton to build/test
    """
    res = dict()
    model = load_model(model_name)
    for file in test:
        print(f"Testing file {file}...")
        _, _, fs = test_model_batch(model_name, [file])
        if file not in res:
            res[file] = {"rnn" : round(fs*100,1), "automaton" : []}
        print("\n\n")

    for state in states:
        print(f"Building automaton {automaton}{state}...")
        A = batch_automaton_build(model_name, build, states=state)
        with open(f"automate/{automaton}{state}.pkl", "wb") as f:
            pickle.dump(A, f)
        test_automaton_batch(f"{automaton}{state}", build, model)
        
        for file in test:
            _, _, f1 = test_automaton_batch(f"{automaton}{state}", [file], model)
            res[file]["automaton"].append(round(f1*100,1))
            print(f"F1-score for {file} with {state} states: {round(f1*100,1)}")


    with open("automaton_results.json", "w") as f:
        json.dump(res, f)


def loss_stat(model_name, build_files=["kernel32.log", "msvcr100.log", "user32.log"], test_file=["ntdll.log"], rnn=True):
    res = []
    for _ in range(20):
        if rnn:
            batch_rnn_build("btest.500", build_files, save=False)
        else:
            batch_gru_build("bgru_test.500", build_files, save=False)
        
        files = os.listdir("test/")
        files = sorted([f for f in files if f.endswith(".pth")], key=lambda x: int(x.split(".")[0]))
        print(files)

        prf = [[], [], []]  # precision, recall, f1
        for file in files:
            print(f"Testing model saved at epoch {file.split('.')[0]}...")
            prec, recall, f1 = test_model_batch(f"{file.split('.')[0]}", test_file)
            prf[0].append(prec)
            prf[1].append(recall)
            prf[2].append(f1)
            os.remove(f"test/{file}")
        res.append(prf)

    with open("model_perfs.json", "w") as f:
        json.dump(res, f)


def stats_states_batch(auto_path, path, files):
    with open(auto_path, "rb") as f:
        A = pickle.load(f)

    dataset = LogDataset(path=path, files=files)
    dataset.prep_batch(DA=-2, DB=20)
    historique = []

    for i in range(len(dataset.data)):
        if dataset.labels[i] == 1:
            x = dataset.data[i]
            _, h = A.predict(torch.tensor(x))
            historique.append((x, h))
    
    states = dict()
    for _, hist in historique:
        for i in range(len(hist)):
            if hist[i] not in states:
                states[hist[i]] = [i]
            else:
                states[hist[i]].append(i)
    print(f"Nombre d'états : {len(states)} / Nombre d'échantillons : {len(historique)}")

    stats = dict()
    for s, positions in states.items():
        stats[s] = {
            "frequence" : len(positions) / len(historique),
            "count" : len(positions)
        }
        for i in range(23):
            stats[s][str(i)] = len([p for p in positions if p == i])
    return stats


def stats_states(auto_path, files, path):
    with open(auto_path, "rb") as f:
        A = pickle.load(f)

    dataset = LogDataset(path=path, files=files)
    dataset.prep_concat(training=False)
    historique = []

    for i in range(len(dataset.data)):
        x = dataset.data[i]
        p, h = A.predict(torch.tensor(x))
        positif = [i for i, pred in enumerate(p) if pred == 1]
        historique.extend([h[i-5:i+1] for i in positif])
    

    c = Counter([h[0] for h in historique])
    print(f"Top 10 des états initiaux : {c.most_common(10)}")

    """ states = dict()
    for hist in historique:
        for i in range(len(hist)):
            if hist[i] not in states:
                states[hist[i]] = [i]
            else:
                states[hist[i]].append(i)
    print(f"Nombre d'états : {len(states)} / Nombre d'échantillons : {len(historique)}")

    stats = dict()
    for s, positions in states.items():
        stats[s] = {
            "frequence" : len(positions) / len(historique),
            "count" : len(positions)
        }
        for i in range(23):
            stats[s][str(i)] = len([p for p in positions if p == i])
    return stats """
    

def score_to_proba(score, Laplace_smoothing=True):
    alpha = 1 if Laplace_smoothing else 0
    proba = {}
    states = set([s[0] for s in score.keys()])
    for s in states:
        filtered_j = [j for (q, j) in score.keys() if q == s]
        total = sum([score[(s, j)] for j in filtered_j])
        for j in filtered_j:
            proba[(s, j)] = (score[(s, j)] + alpha) / (total + alpha * len(states))
        #print(f"Total proba for state {s} : {sum([proba[(s, j)] for j in filtered_j])}")
    return proba



def dijkstra(graph, start, end):
    """
    graph : dict, where (q, i) : (q', w) means there is a transition from state q to state q' with label i and weight w
    """
    """ all_states = set()
    for (q,i), (q_next, w) in graph.items():
        all_states.add(q)
        all_states.add(q_next) """

    distances = {state: float('inf') for state in set(q for (q, _) in graph.keys())}
    distances[start] = 0
    previous = {state: None for state in distances}
    unvisited = set(distances.keys())

    while unvisited:
        current = min(unvisited, key=lambda state: distances[state])
        unvisited.remove(current)

        if current == end:
            break

        for (q, i), (q_next, w) in graph.items():
            if q == current and q_next in unvisited:
                alt = distances[current] + w
                if alt < distances[q_next]:
                    distances[q_next] = alt
                    previous[q_next] = (current, i)
    
    path = [(end, None)]
    current = end
    while current is not None:
        prev = previous.get(current, None)
        if prev is not None:
            path.append((prev[0], f"0x{prev[1]:02X}"))
            current = prev[0]
        else:
            current = None
    path.reverse()
    return path, distances[end]




def chemin_probable(automate, init_state=-1, test_files="", DELTA=6):
    if test_files == "Celica":
        test = ["kerberos.log","ieproxy.log","crypt32.log","clp64.log","energy.log","basesrv.log"]
    elif test_files == "Full":
        test = ["kernel32.log", "msvcr100.log", "user32.log", "ntdll.log", "libcrypto.log", "firewallAPI.log", "ws2_32.log", "signdrv.log", "cmdext.log", "gdi32.log", "kerberos.log","ieproxy.log","crypt32.log","clp64.log","energy.log","basesrv.log"]
    elif test_files == "x32":
        test = ["x32/crypt32.log", "x32/kerberos.log", "x32/kernel32.log", "x32/msvcr100.log", "x32/user32.log"]
    else:
        print("Test files not recognized. Please choose between 'Celica', 'Full' or 'x32'.")
        return

    #Charge l'automate
    with open(f"automate/{automate}_1000.pkl", "rb") as f:
        A = pickle.load(f)

    #Calcul des scores pour chaque transition
    multi_X = []
    print(f"Nombre d'état : {len(A.Q)} ")
    for file in test:
        X, _ = testing_data(f"data/{file}", DELTA=DELTA)
        multi_X.append(X)
    score = A.score_path(multi_X)

    #Transforme les scores en probabilités de transition
    proba = score_to_proba(score)

    graph = dict()
    for (q, i) in proba.keys():
        q_next = A.delta.get((q, i), None)
        if q_next is not None:
            poids = -math.log10(proba[(q, i)])
            graph[(q, i)] = (q_next, poids)

    digraph = "digraph {\n"
    for final in A.F:
        digraph += f"{final} [fillcolor=yellow, style=filled];\n"
        path, _ = dijkstra(graph, start=init_state, end=final)
        src, dest = "", ""
        label = ""
        for p in path:
            src = dest
            dest = p[0]
            if src != "":
                transition = f"{src} -> {dest} [label=\"{label}\"];\n" 
                digraph += transition if transition not in digraph else ""
            label = p[1]
            print(p, end=" ")
            if label is not None:
                print(f"-> ", end="")
        print("\n")

    digraph += "}"

    with open("djikstra.dot", "w") as f:
        f.write(digraph)
    


def n_plus_court(taille, graph, initial_state=-1, verbose=False):
    all_states = set()
    for (q,i), v in graph.items():
        all_states.add(q)
        transitions = v if type(v) == list else [v]

        for q_next, _ in transitions:
            all_states.add(q_next)

    distances = [{state: float('inf') for state in all_states} for _ in range(taille+1)]
    previous = [{state: None for state in distances[0]} for _ in range(taille+1)]
    
    distances[0][initial_state] = 0
    queue = [initial_state]
    for i in range(1,taille+1):
        if verbose:
            print(f"Étape {i} / {taille} - Nombre d'états à explorer : {len(queue)}")
        next_queue = []
        for current in queue:
            for (q, label), target in graph.items():
                if q == current:
                    if type(target) != list:
                        q_next, w = target
                        if distances[i-1][q] + w < distances[i][q_next]:
                            distances[i][q_next] = distances[i-1][q] + w
                            previous[i][q_next] = (current, label)
                            next_queue.append(q_next)
                    else:
                        for qn, w in target:
                            if distances[i-1][q] + w < distances[i][qn]:
                                distances[i][qn] = distances[i-1][q] + w
                                previous[i][qn] = (current, label)
                                next_queue.append(qn)
        queue = list(set(next_queue))
    
    
    last_sorted = sorted(distances[-1].items(), key=lambda x: x[1])
    paths = []

    for state, dist in last_sorted[:3]:
        end = state
        path = [(end, None)]
        current = end
        for i in range(taille, 0, -1):
            prev = previous[i].get(current, None)
            if prev is not None:
                path.append((prev[0], f"0x{prev[1]:02X}"))
                current = prev[0]
            else:
                current = None
        path.reverse()
        paths.append((path, dist))
    return paths



def reverse_graph(graph):
    reverse = {}
    for (q, i), (q_next, w) in graph.items():
        if reverse.get((q_next, i), None) is not None:
            reverse[(q_next, i)].append((q, w))
        else:
            reverse[(q_next, i)] = [(q, w)]
    return reverse


def reverse_graph_simple(graph):
    reverse = {}
    for (q, i), q_next in graph.items():
        reverse[(q_next, i)] = q
    return reverse



def find_signature(A, path, test_files, taille_chemin, comparaison=False):
    """ comparaison permet d'afficher les chemins de taille_chemin depuis les états d'oublie vers les états finaux, pour comparer avec les chemins trouvés dans l'autre sens. """

    precision = (-math.log10(1/7)) * taille_chemin
    print(f"Seuil de précision pour les chemins de longueur {taille_chemin} : {precision:.3f}")

    #Calcul des scores pour chaque transition
    multi_X = []
    print(f"Nombre d'état : {len(A.Q)} ")
    for file in test_files:
        X, _ = testing_data(f"{path}/{file}", DELTA=6) #delta n'a pas d'importance ici, on n'utilise pas les labels
        multi_X.append(X)
    score = A.score_path(multi_X)

    #Transforme les scores en probabilités de transition
    proba = score_to_proba(score)
    finaux = set()

    graph = dict()
    for (q, i) in proba.keys():
        q_next = A.delta.get((q, i), None)
        if q_next is not None:
            poids = -math.log10(proba[(q, i)])
            graph[(q, i)] = (q_next, poids)
        finaux.add(q_next) if q_next in A.F else None
        finaux.add(q) if q in A.F else None

    print(f"A.F : {len(A.F)} / finaux : {len(finaux)} ")
    print(A.F, finaux)
    reversed_graph = reverse_graph(graph)

    #Checher les états d'oublie

    print("\n\n[+]------------------ Chemins vers les états d'oublie ------------------[+]\n")
    oblivion_states = set()
    for state in finaux:
        print(f"\n------- État {state} -------")
        paths = n_plus_court(taille_chemin, graph=reversed_graph, initial_state=state)
        for path, dist in paths:
            oblivion_states.add(path[-1][0])
            chemin, binaire = "", ""
            for p in path:
                chemin += f"{p[0]:>3} -> " if p[1] is not None else f"\033[96m{p[0]:>3}\033[00m"
                binaire += f"\033[92m{p[1]}\033[00m " if p[1] is not None else ""
            print(f"{chemin} | {binaire}| Distance : {dist:.3f}")
    print(f"États d'oublie : {oblivion_states}")

    #Plus court chemin vers les états d'oublie
    
    print("\n\n[+]------------------ Signatures ------------------[+]\n")
    dijkstra_paths = []
    subgraph = {}
    for state in oblivion_states:
        if comparaison:
            print(f"\n------- État {state} -------")
            paths = n_plus_court(taille_chemin, graph=graph, initial_state=state)
            for path, dist in paths:
                chemin, binaire = "", ""
                for p in path:
                    chemin += f"{p[0]:>3} -> " if p[1] is not None else f"\033[96m{p[0]:>3}\033[00m"
                    binaire += f"\033[92m{p[1]}\033[00m " if p[1] is not None else ""
                print(f"{chemin} | {binaire}| Distance : {dist:.3f}")

        for fs in finaux:
            path, dist = dijkstra(graph, start=state, end=fs)
            chemin, binaire = "", ""
            for p in path:
                chemin += f"{p[0]:>3} -> " if p[1] is not None else f"\033[96m{p[0]:>3}\033[00m"
                binaire += f"\033[92m{p[1]}\033[00m " if p[1] is not None else ""
                dec = int(p[1], 16) if p[1] is not None else None
                if dec is not None and dist < precision and subgraph.get((p[0], dec)) is None:
                    subgraph[(p[0], dec)] = graph[(p[0], dec)]
            dijkstra_paths.append((chemin, binaire, dist))
    
    sorted_dijkstra = sorted(dijkstra_paths, key=lambda x: x[2])
    print("\n\n------- Chemins de Dijkstra triés par distance -------\n")
    for chemin, binaire, dist in sorted_dijkstra:
        print(f"{chemin} | {binaire}| Distance : {dist:.3f}")

    graph_to_dot(subgraph, filename="signature_subgraph.dot", highlight=[("yellow", A.F), ("cyan", oblivion_states)])
    return subgraph, oblivion_states



def graph_to_dot(graph, filename="graph.dot", highlight=[]):
    """ highlight : (color, [q1, q2, ...]) """
    dot = "digraph {\n"

    for color, states in highlight:
        for q in states:
            dot += f"{q} [fillcolor=\"{color}\", style=filled];\n"

    for (q, i), (q_next, w) in graph.items():
        dot += f"{q} -> {q_next} [label=\"0x{i:02X} ({w:.3f})\"];\n"
    dot += "}"
    with open(filename, "w") as f:
        f.write(dot)



def graph_to_dot_alt(graph, filename="graph.dot", highlight=[]):
    """ highlight : (color, [q1, q2, ...]) """
    dot = "digraph {\n"

    for color, states in highlight:
        for q in states:
            dot += f"{q} [fillcolor=\"{color}\", style=filled];\n"

    for (q, i), q_next in graph.items():
        dot += f"{q} -> {q_next} [label=\"{i}\"];\n"
    dot += "}"
    with open(filename, "w") as f:
        f.write(dot)







def real_signatures(A, path, files, DELTA=6):
    offset = DELTA-1
    graph = "digraph {\n"
    freq = dict()
    true_graph = dict()

    for final in A.F:
        graph += f"{final} [fillcolor=yellow, style=filled];\n"

    for file in files:
        X, _ = testing_data(f"{path}/{file}", DELTA=DELTA)
        p, h = A.predict(torch.tensor(X))
        functions = [i for i in range(len(p)) if p[i] == 1]
        for i in functions:
            data = X[i-offset:i+1]
            nodes = h[i-offset:i+2]
            for j in range(len(data)):
                freq[(nodes[j], nodes[j+1])] = freq.get((nodes[j], nodes[j+1]), 0) + 1
                true_graph[(nodes[j],data[j])] = nodes[j+1]
    

    max_freq = math.sqrt(max(freq.values()))
    min_freq = min(freq.values())
    range_freq = max_freq - min_freq if max_freq != min_freq else 1

    #print(f"Moyenne : {sum(freq.values())/len(freq):.2f} | Min : {min_freq} | Sqrt Max : {max_freq}")
    #print(f"std : {math.sqrt(sum([(f - sum(freq.values())/len(freq))**2 for f in freq.values()]) / len(freq)):.2f}")
    #print(f"Médiane : {sorted(freq.values())[len(freq)//2]}")
    #print(f"9e décile : {sorted(freq.values())[int(len(freq)*0.9)]}")

    for (q, q_next), f_value in freq.items():
        norm = (math.sqrt(f_value) - min_freq) / range_freq

        r = int(0   + (255 - 0)   * norm)
        g = int(0   + (0  - 0)   * norm)
        b = int(0 + (0   - 0) * norm)
        hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        graph += f"{q} -> {q_next} [color=\"{hex_color}\"];\n"

    graph += "}"

    with open(f"real_signatures.dot", "w") as f:
        f.write(graph)
    return true_graph


def all_path(A, length=6): #Trop volumineux
    subgraph = dict()
    reversed_delta = A.reverse_delta()
    visited = set()
    for f in A.F:
        queue = [(f, 0)]
        while queue:
            current, dist = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if dist < length:
                for (q, i), q_next in reversed_delta.items():
                    if q == current:
                        for trg in q_next:
                            print(f"Transition : {q} --0x{i:02X}--> {trg}")
                            subgraph[(trg, i)] = q
                            queue.append((trg, dist+1))
    
    graph_to_dot_alt(subgraph, filename=f"all_paths_{length}.dot", highlight=[("yellow", A.F), ("#AA00FF", A.get_ranking(1)), ("#AA22FF", A.get_ranking(2))])#, ("#AA44FF", A.get_ranking(3)), ("#AA66FF", A.get_ranking(4)), ("#AA88FF", A.get_ranking(5)), ("#AABBFF", A.get_ranking(6))])
    subprocess.run(["dot", "-Tsvg", f"all_paths_{length}.dot", "-o", f"all_paths_{length}.svg"])
    return subgraph

######## Arbre de signatures ########

class tree_signature:
    def __init__(self, data):
        self.data = data
        self.children = []
        
    def add_child(self, data):
        child = tree_signature(data)
        self.children.append(child)
        return child
    
    def remove_child(self, child):
        self.children.remove(child)
    
    def toGraph(self, first=False):
        dot = ""
        for c in self.children:
            if first and self.data == ():
                dot += f'"root" -> "{c.data}";\n'
            else:
                dot += f'"{self.data}" -> "{c.data}";\n'
            dot += c.toGraph()
        if first:
            dot = "digraph {\n" + dot + "}"
        return dot
    
    #Obsolete
    def predict(self, X, length=6):
        n = len(X)
        res = [0]*n
        ptr = 0
        while ptr < n:
            v = False
            for c in self.children:
                if X[ptr] == c.data:
                    if ptr in [17, 225, 545, 753]:
                        v = True
                    pred = c.child_predict("".join(X[ptr:ptr+length]),v)
                    if pred == 1:
                        res[ptr+length-1] = 1
                    if v:
                        print(f"Prediction for {X[ptr:ptr+length]} : {res[ptr:ptr+length]}")
            ptr += 1
        return res
    #Obsolete
    def child_predict(self, motif, verbose=False):
        if verbose:
            print(f"{self} -> {motif}")
        if not self.children: #feuille
            return 1
        
        for c in self.children:
            if motif.startswith(c.data):
                return c.child_predict(motif, verbose=verbose)

        return 0 # aucun match

    def __str__(self):
        return f"Node({self.data}) -- Children: {sorted([c.data for c in self.children])}"


def get_signature_tree(path, files, length=6, visualize=False):
    sig = set()
    for file in files:
        X, Y = testing_data(f"{path}/{file}", DELTA=length)
        index = [i for i in range(len(Y)) if Y[i] == 1]
        tmp = set()
        for i in index:
            tmp.add(tuple(X[i-(length-1):i+1]))
        sig = sig.union(tmp)
    print(f"Nombre de signatures uniques : {len(sig)}")
    print(f"Exemples de signatures : {list(sig)[:5]}")

    root = tree_signature(())
    queue = [(root, sig)]
    while queue:
        current, sig_set = queue.pop(0)

        prefixes = {}
        for s in sig_set:
            if len(s) >= len(current.data) + 1:
                prefix = s[:len(current.data)+1]
                if prefix not in prefixes:
                    prefixes[prefix] = set()
                prefixes[prefix].add(s)
        
        for prefix, s_set in prefixes.items():
            child = current.add_child(prefix)
            queue.append((child, s_set))

    
    dot = root.toGraph(first=True)
    print("Génération du fichier DOT pour l'arbre de signatures...")
    with open(f"experiences/signature/celica_sign_tree_ref_{length}.pkl", "wb") as f:
        pickle.dump(root, f)
    with open(f"experiences/signature/celica_sign_tree_ref_{length}.dot", "w") as f:
        f.write(dot)
    
    if visualize:
        print("Conversion du fichier DOT en SVG...")
        subprocess.run(["dot", "-Tsvg", f"experiences/signature/celica_sign_tree_ref_{length}.dot", "-o", f"experiences/signature/celica_sign_tree_ref_{length}.svg"])
    return root
        

#obsolete, mais je garde pour l'instant
def automaton_to_tree(delta, init_state=-1, final_states=set(), length=6):
    def prune(node, length): #TODO: c'est pas gracieux, mais jsp où le mettre
        remove_list = []
        for child in node.children:
            prune(child, length)
            if len(child.data) < length and not child.children:
                remove_list.append(child)
        for child in remove_list:
            node.remove_child(child)

    # Profondeur de chaque état
    queue = [init_state]
    profondeur = {init_state: 0}
    while queue:
        current = queue.pop(0)
        for (q, i), q_next in delta.items():
            if q == current and q_next not in profondeur:
                # On enregistre la profondeur dès la découverte
                profondeur[q_next] = profondeur[current] + 1
                queue.append(q_next)

    print(f"Profondeur : {profondeur}")

    root = tree_signature(())
    queue = [(root, init_state)]
    while queue:
        current, state = queue.pop(0)
        current_depth = len(current.data)
        if current_depth < length:
            for (q, i), q_next in delta.items():
                if q == state and profondeur[q_next] == current_depth + 1: # filtre par rapport à la profondeur pour accélérer la construction de l'arbre
                    if current_depth+1 == length: # dernier niveau
                        if q_next in final_states:
                            child = current.add_child(current.data+(i,))
                            #child.add_child("FINAL")
                            queue.append((child, q_next))
                    else: # < length
                        child = current.add_child(current.data+(i,))
                        queue.append((child, q_next))

    # élagage des branches qui ne mènent pas à un état final
    print("Élagage des branches de l'arbre...")
    #prune(root, length)

    dot = root.toGraph(first=True)
    with open(f"spanning.dot", "w") as f:
        f.write(dot)
    return root



def distance_tree_auto(node, state, A):
    """Calcul la distance entre l'arbre et l'automate"""
    if not node.children: #feuille
        #print("ok") if state in A.F else print(f"ko")
        return 0 if state in A.F else 2**-len(node.data)
    
    score = 0

    if state in A.F and node.data != ():
        #print(f"Final sans feuille / {node.data}")
        score += 2**-(len(node.data))

    explored = set()
    for child in node.children:
        if (state, child.data[-1]) in A.delta:
            explored.add(child.data[-1])
            next_state = A.delta[(state, child.data[-1])]
            score += distance_tree_auto(child, next_state, A)
        else: # pas de transition, on considère que c'est une erreur + pas d'exploration
            #print(f"Pas de transition alors que chemin / {child.data}")
            score += 2**-(len(child.data)+1)

    return score


def compteur_mots(A, init_state, length):
    """Compte le nombre de mots de longueur 1 à length que l'automate A accepte à partir de l'état init_state"""
    compteur_actuel = {q:0 for q in A.Q}
    compteur_actuel[init_state] = 1
    total = 0

    for _ in range(1, length+1):
        compteur_suivant = {q:0 for q in A.Q}
        for (q, _), q_next in A.delta.items():
            compteur_suivant[q_next] += compteur_actuel[q]
        compteur_actuel = compteur_suivant
        total += sum([compteur_actuel[q] for q in A.F])
    return total


##### Main #####


if __name__ == "__main__":
    #Celica
    #build = ["kernel32.log", "msvcr100.log", "user32.log", "ntdll.log", "libcrypto.log", "firewallAPI.log", "ws2_32.log", "signdrv.log", "cmdext.log", "gdi32.log"]
    #test = ["kerberos.log","ieproxy.log","crypt32.log","clp64.log","energy.log","basesrv.log"]

    with open("data/dataset.json", "r") as f:
        dataset = json.load(f)

    ds = "Levin"
    build = dataset[ds]["train"]
    test = dataset[ds]["test"]
    path_file = f"data/{ds.lower()}"

    #model = "Hgru_CelicaFS.500"
    #chemin_probable(model, init_state=580, test_files="Celica")


    delta = 7
    instr = True
    method = "multi-classe"
    gru_build("ABC_instrMC", build, method, path_file, instr=instr,save=True, DELTA=delta)
    test_model_dataloader("ABC_instrMC", path_file, test, instr=instr, DELTA=delta)
    exit(0)
    print("\n######### Automate #########\n")
    start = time.time()
    A = automaton_build_dataloader("ABC_test6", path_file, build, states=1000, DELTA=delta)
    exit(0)
    print(f"Automaton built in {round(time.time() - start, 2)} seconds")
    test_automaton("ABC_test6_1000", path_file, test, model=load_model("ABC_test6"), DELTA=delta)
    exit(0)

    #find_signature(A, test, taille_chemin=delta, comparaison=True)

    #for i in [-1, 563, 89, 67, 333, 0]:
    #    test_automaton("ABC_test6_1000", test, model=load_model("ABC_test6"), DELTA=delta, init_state=i)


    with open("automate/ABC_test_1000.pkl", "rb") as f:
        A = pickle.load(f)

    A.minimize()


    cpt = {q: 0 for q in A.Q}
    for (q,i), q_next in A.delta.items():
        cpt[q_next] += 1
    print(sorted(cpt.items(), key=lambda x: x[1], reverse=True)[:10])
    print(sorted(cpt.items(), key=lambda x: x[1])[:10])
    print("Moyenne : ", sum(cpt.values())/len(cpt))
    print("Médiane : ", sorted(cpt.values())[len(cpt)//2])


    root = get_signature_tree(build, length=delta)

    #root = automaton_to_tree(A.delta, init_state=-1, final_states=A.F, length=delta)
    #subprocess.run(["dot", "-Tsvg", f"spanning.dot", "-o", f"spanning.svg"])

    scores = []
    for i in range(-1, len(A.Q)-1):
        score = distance_tree_auto(root, i, A)
        scores.append((score, i))
        print(f"{i}: Score de l'arbre par rapport à l'automate : {score:.7f}")
    sorted_scores = sorted(scores, key=lambda x: x[0])
    print(f"Meilleurs états : {sorted_scores}")

    selected = [s[1] for s in sorted_scores if s[0] == sorted_scores[0][0]]
    print(f"États sélectionnés : {selected}")

    mots = []
    for s in selected:
        res = compteur_mots(A, s, delta)
        mots.append((s, res))
        print(f"Nombre de mots de longueur 1 à {delta} acceptés par l'automate à partir de l'état {s} : {res}")

    print(f"Nombre de mots : {sorted(mots, key=lambda x: x[1])}")
        




    """ model = "Hgru_CelicaFS.500"
    taille_chemin = 6
    with open(f"automate/{model}_1000.pkl", "rb") as f:
        A = pickle.load(f) """


    """ ds2 = LogDataset(files=test, whitelist=True)
    ds2.prep_fs(DELTA=8)

    for i in range(len(ds.labels)):
        for j in range(len(ds.labels[i])):
            if ds.labels[i][j] == 8 and ds2.labels[i][j] != 1:
                print(f"D1 / Mismatch at file {test[i]} index {j} : label {ds.labels[i][j]} vs {ds2.labels[i][j]}")

    for i in range(len(ds2.labels)):
        for j in range(len(ds2.labels[i])):
            if ds2.labels[i][j] == 1 and ds.labels[i][j] != 8:
                print(f"D2 / Mismatch at file {test[i]} index {j} : label {ds2.labels[i][j-6:j+6]} vs {ds.labels[i][j-6:j+6]}") """



    
    """ subgraph, oblivion_states = find_signature(A, test_files=test, taille_chemin=taille_chemin)
    subgraph = {k: {v[0]} for k, v in subgraph.items()}
    subgraph[(-2, 300)] = oblivion_states
    nfa = NFA(subgraph, init_state=-2, final_states=A.F)

    with open("automate/signatures/sig6T.pkl", "wb") as f:
        pickle.dump(nfa, f) """


    ### NFA signature ###
    """ with open("automate/signatures/sig6T.pkl", "rb") as f:
        nfa = pickle.load(f)
    
    model = load_model(model)

    for file in test:
        X, y = testing_data(f"data/{file}", DELTA=taille_chemin)
        predicted = nfa.predict(X)
        print(f"File {file} :")
        #print(sum(predicted), sum(y))
        scores = model.scores(y, predicted)
        print(f"Precision: {scores[0]:.3f}, Recall: {scores[1]:.3f}, F1-score: {scores[2]:.3f} / {scores[0]:.3f} & {scores[1]:.3f} & {scores[2]:.3f} \n") """


    ### NFA signature reelle ###
    """ subgraph = real_signatures(A, test, DELTA=taille_chemin)
    oblivions = {548, 580, 901, 79, 367}
    #in_nodes = set(q for q in subgraph.values())
    #out_nodes = set(q for (q,b) in subgraph.keys())
    #first_layers = out_nodes - in_nodes
    #print(f"In nodes : {len(in_nodes)}\nOut nodes : {len(out_nodes)}\nFirst layers : {first_layers}")
    
    subgraph = {k: {v} for k, v in subgraph.items()}
    subgraph[(-2, 300)] = oblivions  #first_layers
    nfa = NFA(subgraph, init_state=-2, final_states=A.F)

    with open("automate/signatures/sig6_real.pkl", "wb") as f:
        pickle.dump(nfa, f)

    model = load_model(model)

    for file in test:
        X, y = testing_data(f"data/{file}", DELTA=taille_chemin)
        predicted = nfa.predict(X)
        print(f"File {file} :")
        #print(sum(predicted), sum(y))
        scores = model.scores(y, predicted)
        print(f"Precision: {scores[0]:.3f}, Recall: {scores[1]:.3f}, F1-score: {scores[2]:.3f} / {scores[0]:.3f} & {scores[1]:.3f} & {scores[2]:.3f} \n") """




    ### GRU Instruction ###

    """ model_path = "models/gru_Celica_instr.500.pth"
    model = ""

    if pathlib.Path(model_path).is_file():
        model = torch.load(model_path, weights_only=False).to(DEVICE)
        print(f"Model {model_path} loaded successfully.")
    else:
        dataset = LogDataset(files=build, whitelist=True)
        dataset.prep_header_fs()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True, collate_fn=pad_batch)
        print(f"Fichier chargé pour l'entraînement : {dataset.used_files}")

        model = TGRU().to(DEVICE)
        print(f"Hyperparamètres : {model.get_hyperparameters()}")
        model.train(dataloader)
        torch.save(model, model_path)
 

    #Testing
    y_test = []
    predicted = []
    for file in test:
        X, y = testing_data_headers(f"data/{file}")
        y_test.extend(y)
        print(f"{len(X)} items in testing dataset")
        
        X_t = torch.tensor(X).to(DEVICE)

        with torch.no_grad():
            pred = (model.predict(X_t).squeeze() > 0.8).int().detach().cpu().numpy()
            predicted.extend(pred)
    
    print(f"Scores : {model.scores(y_test, predicted)}") """


    """ 
    ### GRU Multi ###
    #gru_multi_build("Hgru_CelicaFS_multi8.500", build, save=True, DELTA=8)
    test_model_multi("Hgru_CelicaFS_multi8.500", test, DELTA=8)
    exit()
    automaton_build_multi("Hgru_CelicaFS_multi8.500", build, states=1000, DELTA=8)
    for file in test:
        print(f"Testing file {file}...")
        test_automaton_multi(f"Hgru_CelicaFS_multi8.500_1000", [file], model=load_model("Hgru_CelicaFS_multi8.500"), DELTA=8)
    """




    """ with open(f"automate/ABC_test_1000.pkl", "rb") as f:
        A = pickle.load(f)
    #A.emonde()
    

    #with open("dot_gaph_auto.dot", "w") as f:
    #    f.write(A.dot())


    tmp = [0]*7
    for q in A.rank.keys():
        tmp[A.rank[q]] += 1
    print(tmp) 

    
    for i in range(0, 7):
        ranked = A.get_ranking(i)
        print(f"{len(ranked)} noeuds de rang {i} : {list(ranked)[:10]}")

    print(len(A.Q))

    
    #Calcul des scores pour chaque transition
    multi_X = []
    print(f"Nombre d'état : {len(A.Q)} ")
    for file in test:
        X, _ = testing_data(f"data/{file}", DELTA=6) #delta n'a pas d'importance ici, on n'utilise pas les labels
        multi_X.append(X)
    score = A.score_path(multi_X)


    #Transforme les scores en probabilités de transition
    proba = score_to_proba(score)

    graph = dict()
    for (q, i) in proba.keys():
        q_next = A.delta.get((q, i), None)
        if q_next is not None:
            poids = -math.log10(proba[(q, i)])
            graph[(q, i)] = (q_next, poids)

    subgraph = {}
    oblivion_states = A.get_ranking(1)
    for state in oblivion_states:
        for fs in A.F:
            path, _ = dijkstra(graph, start=state, end=fs)
            for p in path:
                dec = int(p[1], 16) if p[1] is not None else None
                if dec is not None:
                        subgraph[(p[0], dec)] = graph[(p[0], dec)]
    print(f"Finals : {A.F}")
    graph_to_dot(subgraph, filename="graph_sus.dot", highlight=[("yellow", A.F), ("cyan", oblivion_states), ("#005500", A.get_ranking(2)), ("#007700", A.get_ranking(3)), ("#009900", A.get_ranking(4)), ("#00BB00", A.get_ranking(5)), ("#00DD00", A.get_ranking(6)), ("#00FF00", A.get_ranking(7)), ("#00FF55", A.get_ranking(8)), ("#00FF77", A.get_ranking(9))])
     """

