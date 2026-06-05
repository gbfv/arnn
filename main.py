import time
import torch
import pathlib
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
import json
from collections import Counter
import math

from rnn import ARNN
from rnn_batch import BRNN
from gru import TGRU
from gru_batch import BGRU
from gru_multi import TMGRU
from automaton import Automaton
from automaton_multi import Automaton_multi
from logDataset import LogDataset, prepared_data, testing_data, testing_data_headers, testing_data_multi
from nfa import NFA
from utils import get_device




DEVICE = get_device()



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

def load_model(name, path="models"):
    model_path = f"{path}/{name}.pth"
    # Loading model
    if pathlib.Path(model_path).is_file():
        model = torch.load(model_path, weights_only=False).to(DEVICE)
    else:
        raise FileNotFoundError(f"Model file {model_path} not found. Please train the model before testing.")
    return model


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


    
def gru_build(name, training_files, whitelist=True, save=True, reverse=False, x32=False, DELTA=6):
    model_path = f"models/{name}.pth"
    # Loading model
    if pathlib.Path(model_path).is_file():
        return torch.load(model_path, weights_only=False).to(DEVICE)
    
    # Building model
    dataset = LogDataset(files=training_files, whitelist=whitelist, x32=x32)
    dataset.prep_fs(DELTA=DELTA, reverse=reverse)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True, collate_fn=pad_batch)
    print(f"Fichier chargé pour l'entraînement : {dataset.used_files}")

    model = TGRU().to(DEVICE)
    print(f"Hyperparamètres : {model.get_hyperparameters()}")
    model.train(dataloader)
    if save:
        torch.save(model, model_path)
    return model


def gru_multi_build(name, training_files, whitelist=True, save=True, x32=False, DELTA=6):
    model_path = f"models/{name}.pth"
    # Loading model
    if pathlib.Path(model_path).is_file():
        return torch.load(model_path, weights_only=False).to(DEVICE)
    
    # Building model
    dataset = LogDataset(files=training_files, whitelist=whitelist, x32=x32)
    dataset.prep_multi(DELTA=DELTA)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True, collate_fn=pad_batch)
    print(f"Fichier chargé pour l'entraînement : {dataset.used_files}")

    model = TMGRU(DELTA+1).to(DEVICE)
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



def test_model(model_name, files, DELTA=6, reverse=False):
    model = load_model(model_name)
    
    # Preparing test data
    y_test = []
    predicted = []
    for file in files:
        X, y = testing_data(f"data/{file}", DELTA=DELTA)
        if reverse:
            X.reverse()
            y.reverse()
        y_test.extend(y)
        print(f"{len(X)} items in testing dataset")
        
        X_t = torch.tensor(X).to(DEVICE)

        with torch.no_grad():
            pred = (model.predict(X_t).squeeze() > 0.8).int().detach().cpu().numpy()
            predicted.extend(pred)
    
    #print(f'Diff : {sum(abs(y_test - predicted))}')
    print(f"Scores : {model.scores(y_test, predicted)}")


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
    

def test_model_multi(model_name, files, DELTA=6):
    model = load_model(model_name)
    
    # Preparing test data
    y_test = []
    predicted = []
    for file in files:
        X, y = testing_data_multi(f"data/{file}", DELTA=DELTA)
        y_test.extend(y)
        print(f"{len(X)} items in testing dataset")
        
        X_t = torch.tensor(X).to(DEVICE)

        with torch.no_grad():
            pred = torch.argmax(model.predict(X_t), dim=1).int().detach().cpu().numpy()  # Prédiction de la classe avec la plus haute probabilité
            predicted.extend(pred)
            
    #print(f'Diff : {sum(abs(y_test - predicted))}') 
    print(f"Scores : {model.scores(y_test, predicted)}")


### Automaton functions ###


def automaton_build(model_name, files, states=1000, DELTA=6, reverse=False):
    model = load_model(model_name)

    auto_path = f"automate/{model_name}_{states}.pkl"
    if pathlib.Path(auto_path).is_file():
        with open(auto_path, "rb") as f:
            return pickle.load(f)
   
    #Preparing data
    X_test, y_test = [], []
    item = 0
    for file in files:
        X, y = testing_data(f"data/{file}", DELTA=DELTA)
        X_test.append(X)
        y_test.append(y)
        item += len(X)
    print(f"{item} items in testing dataset")
    
    #X_t, y_t = torch.tensor(X_test).to(DEVICE), torch.tensor(y_test)

    print("Construction de l'automate...\n")
    A = Automaton(model, list(range(256)), states, X_test, y_test)
    A.emonde()

    if reverse:
        r_delta = A.reverse_delta()
        A.delta = r_delta

    with open(auto_path, "wb") as f:
        pickle.dump(A, f)
    
    print(f"Automate construit.\nNombre d'état après émondage : {len(A.Q)} \nNombre d'états finaux : {len(A.F)}")
    return A


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


def automaton_build_multi(model_name, files, states=1000, DELTA=6):
    model = load_model(model_name)

    auto_path = f"automate/{model_name}_{states}.pkl"
    if pathlib.Path(auto_path).is_file():
        with open(auto_path, "rb") as f:
            return pickle.load(f)
    
    #Preparing data
    X_test, y_test = [], []
    item = 0
    for file in files:
        X, y = testing_data_multi(f"data/{file}", DELTA=DELTA)
        X_test.append(X)
        y_test.append(y)
        item += len(X)
    print(f"{item} items in testing dataset")
    
    #X_t, y_t = torch.tensor(X_test).to(DEVICE), torch.tensor(y_test)

    print("Construction de l'automate...\n")
    A = Automaton_multi(model, list(range(256)), states, X_test, y_test)
    #A.emonde()

    with open(auto_path, "wb") as f:
        pickle.dump(A, f)
    
    print(f"Automate construit.\nNombre d'état après émondage : {len(A.Q)} \nNombre d'états finaux : {len(A.F)}")
    return A




def test_automaton(automaton_name, files, model, DELTA=6):
    auto_path = f"automate/{automaton_name}.pkl"
    if pathlib.Path(auto_path).is_file():
        with open(auto_path, "rb") as f:
            A = pickle.load(f)
    else:
        raise FileNotFoundError(f"Automaton file {auto_path} not found. Please build the automaton before testing.")

    print(f"Testing automaton {automaton_name}...\n")

    #Preparing data
    X_test, y_test = [], []
    for file in files:
        X, y = testing_data(f"data/{file}", DELTA=DELTA)
        X_test.extend(X)
        y_test.extend(y)
    #print(f"{len(X_test)} items in testing dataset")
    
    X_t, y_t = torch.tensor(X_test).to(DEVICE), torch.tensor(y_test)

    predicted, _ = A.predict(X_t)
    print(f'Diff : {sum(abs(y_t - predicted))}')
    prec, recall, f1 = model.scores(y_t, predicted)
    print(f"Scores : Precision: {prec}, Recall: {recall}, F1-score: {f1}")
    return prec, recall, f1


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


def test_automaton_multi(automaton_name, files, model, DELTA=6):
    auto_path = f"automate/{automaton_name}.pkl"
    if pathlib.Path(auto_path).is_file():
        with open(auto_path, "rb") as f:
            A = pickle.load(f)
    else:
        raise FileNotFoundError(f"Automaton file {auto_path} not found. Please build the automaton before testing.")

    print(f"Testing automaton {automaton_name}...\n")

    #Preparing data
    X_test, y_test = [], []
    for file in files:
        X, y = testing_data_multi(f"data/{file}", DELTA=DELTA)
        X_test.extend(X)
        y_test.extend(y)
    #print(f"{len(X_test)} items in testing dataset")
    
    X_t, y_t = torch.tensor(X_test).to(DEVICE), torch.tensor(y_test)

    predicted, _ = A.predict(X_t)
    prec, recall, f1 = model.scores(y_t, predicted)
    print(f"Scores : Precision: {prec}, Recall: {recall}, F1-score: {f1}")
    return prec, recall, f1




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


def stats_states_batch(auto_path, files, whitelist=True):
    with open(auto_path, "rb") as f:
        A = pickle.load(f)

    dataset = LogDataset(files=files, whitelist=whitelist)
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


def stats_states(auto_path, files, whitelist=True):
    with open(auto_path, "rb") as f:
        A = pickle.load(f)

    dataset = LogDataset(files=files, whitelist=whitelist)
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



def find_signature(A, test_files, taille_chemin, comparaison=False):
    """ comparaison permet d'afficher les chemins de taille_chemin depuis les états d'oublie vers les états finaux, pour comparer avec les chemins trouvés dans l'autre sens. """

    precision = (-math.log10(1/7)) * taille_chemin
    print(f"Seuil de précision pour les chemins de longueur {taille_chemin} : {precision:.3f}")

    #Calcul des scores pour chaque transition
    multi_X = []
    print(f"Nombre d'état : {len(A.Q)} ")
    for file in test_files:
        X, _ = testing_data(f"data/{file}", DELTA=6) #delta n'a pas d'importance ici, on n'utilise pas les labels
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



def real_signatures(A, files, DELTA=6):
    offset = DELTA-1
    graph = "digraph {\n"
    freq = dict()
    true_graph = dict()

    for final in A.F:
        graph += f"{final} [fillcolor=yellow, style=filled];\n"

    for file in files:
        X, _ = testing_data(f"data/{file}", DELTA=DELTA)
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

##### Main #####


if __name__ == "__main__":
    #Celica
    build = ["kernel32.log", "msvcr100.log", "user32.log", "ntdll.log", "libcrypto.log", "firewallAPI.log", "ws2_32.log", "signdrv.log", "cmdext.log", "gdi32.log"]
    test = ["kerberos.log","ieproxy.log","crypt32.log","clp64.log","energy.log","basesrv.log"]

    #model = "Hgru_CelicaFS.500"
    #chemin_probable(model, init_state=580, test_files="Celica")

    """ delta = 10
    gru_build("Hgru_CelicaFS_10.500", build, save=True, DELTA=delta)
    test_model("Hgru_CelicaFS_10.500", test, DELTA=delta)
    print("\n######### Automate #########\n")
    start = time.time()
    automaton_build("Hgru_CelicaFS_10.500", build, states=3000, DELTA=delta)
    print(f"Automaton built in {round(time.time() - start, 2)} seconds")
    test_automaton("Hgru_CelicaFS_10.500_3000", test, model=load_model("Hgru_CelicaFS_10.500"), DELTA=delta)


    fs = []
    for file in test+build:
        _, _, f1_10 = test_automaton("Hgru_CelicaFS_10.500_3000", [file], model=load_model("Hgru_CelicaFS_10.500"), DELTA=10)
        _, _, f1_6 = test_automaton("Hgru_CelicaFS.500_1000", [file], model=load_model("Hgru_CelicaFS.500"), DELTA=6)
        fs.append((file, f1_10, f1_6))

    print("\n\n######### Résultats #########\n")
    for file, f1_10, f1_6 in fs:
        print(f"File {file:<16} : F1-score with DELTA=10 : {round(f1_10*100,1):<5.2f}% | DELTA=6 : {round(f1_6*100,1):<5.2f}%") """
   
    

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



    
    """ with open(f"automate/Hgru_CelicaFS_multi8.500_1000.pkl", "rb") as f:
        A = pickle.load(f)
    #A.emonde()
    

    #with open("dot_gaph_auto.dot", "w") as f:
    #    f.write(A.dot())


    tmp = [0]*9
    for q in A.rank.keys():
        tmp[A.rank[q]] += 1
    print(tmp) 

    
    for i in range(0, 9):
        ranked = A.get_ranking(i)
        print(f"{len(ranked)} noeuds de rang {i} : {list(ranked)[:10]}")

    print(len(A.Q)) """

    """
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

    graph_to_dot(subgraph, filename="graph_sus.dot", highlight=[("yellow", A.F), ("cyan", oblivion_states), ("#005500", A.get_ranking(2)), ("#007700", A.get_ranking(3)), ("#009900", A.get_ranking(4)), ("#00BB00", A.get_ranking(5)), ("#00DD00", A.get_ranking(6)), ("#00FF00", A.get_ranking(7))])
    """

