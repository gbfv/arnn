import pickle
import random
import torch
import pathlib
import pickle
import json
import argparse
from re import finditer

from logDataset import LogDataset
#from gru_multi_test import Test_TMGRU
#from gru_test import Test_TGRU
#from gru_labels_test import Test_LGRU
from gru_merge import TOY_GRU
#from automaton import Automaton
#from automaton_multi import Automaton_multi
#from automaton_multi_label import Automaton_multi_label
from automaton_merge import TOY_Automaton
from main import load_model, pad_batch
from utils import get_device
from build_auto import accept_stream, get_automate, light_automaton
from isomorphe import is_isomorphic, ged_nx


DEVICE = get_device()


def controlled_random(params, length, threshold=0.1):
    mots = params["mots"]
    core = []
    size = 0
    odds = 4

    while size < length*threshold:
        mot = random.choice(mots)
        if core == [] or random.randint(1, odds) != 1:
            core.append(mot)
            size += len(mot)
        else:
            core[-1] += mot[1:] # enlève le premier caractère pour forcer un potentiel chevauchement
            odds *= 2
            size += len(mot)-1
    return core, size


def generate_words(params, length = 20, nbr = 10):
    #objectif : avoir au moins 10% de mots
    words = set()
    attempts = 0
    reset_char = params["reset_char"]
    motifs = f"({'|'.join(params['mots'])})"

    while len(words) < nbr and attempts < nbr * 3:
        core, core_len = controlled_random(params, length)
        word = "".join(random.choices(params["alphabet"], k=length-core_len))

        for mot in core:
            pos = random.randint(0, len(word))
            word = word[:pos] + mot + word[pos:]

        finals = finditer(motifs, word) if motifs else None
        word = list(word)
        if reset_char is not None and finals is not None:
            for match in finals:
                if match.end() < len(word):
                    word[match.end()] = reset_char

        words.add("".join(word))
        attempts += 1
        if len(word) != length:
            print(f"Generated word of length {len(word)} instead of {length}. Word: {''.join(word)}")
    return words


def get_label(automate, words, motifs, final_states, method):
    labels = []
    lenght_motif = max(len(motif) for motif in motifs) #taille d'un mot
    for word in words:
        state = "0"
        label = []
        for letter in word:
            state = automate[state][letter]
            if method == "state":
                label.append(int(state)) # On prend l'état comme label
            else:
                if state not in final_states:
                    label.append(0)
                else:
                    label.append(1)

        if method == "multi-classe":
            hit = [i for i in range(len(label)) if label[i] == 1]
            for index in hit:
                debut = index-(lenght_motif-1)
                label[debut:index+1] = list(range(1, lenght_motif+1))
        
        elif method == "multi-label":
            lbl = [[0]*len(motifs) for _ in range(len(label))]
            hit = [i for i in range(len(label)) if label[i] == 1]
            for index in hit:
                debut = index-(lenght_motif-1)
                mot = motifs.index(word[debut:index+1])
                for j in range(debut, index+1):
                    lbl[j][mot] = j+1-debut
            label = lbl
        labels.append(label)
    return labels


def create_log(words, labels, nom="test/log.txt"):
    with open(nom, "w") as f:
        for word in words:
            f.write(word + "\n")
        f.write("LABEL\n")
        for label in labels:
            f.write("/".join(str(l) for l in label) + "\n")


def parse_log_file(file_path):
    with open(file_path, "r") as f:
        log = f.read()
    log = log.split("LABEL")
    raw_data = log[0].strip().split("\n")
    raw_label = log[1].strip().split("\n")

    data = [list(d) for d in raw_data]
    label = [l.split("/") for l in raw_label]

    if label[0][0][0] == '[': #multi-label
        label = json.loads(str(label).replace("'", ""))

    X = [[ord(x)-97 for x in d] for d in data] # Convertir les lettres en indices (a=0, b=1, c=2, ...)
    Y = [[int(y) for y in l] for l in label] if isinstance(label[0][0], str) else label

    return X, Y


###### Modèle et Automate ######

def get_model_name(prefix, method):
    return f"TOY_{prefix}_{method}_model.pth"

def get_automaton_name(prefix, method, states):
    return f"TOY_{prefix}_{method}_{states}.pkl"


def train_model(prefix, method, num_classes=None, mots=None, data_path="test/", model_path="models/"):
    X,Y = parse_log_file(f"{data_path}/{prefix}_{method}_train.txt")
    dataset = LogDataset(files=[], whitelist=True, x32=False)
    dataset.data = X
    dataset.labels = Y
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pad_batch)
    print(f"Fichier chargé pour l'entraînement : {dataset.used_files}")

    label_method = "multi-classe" if method == "state" else method # state est une version de multi-classe
    weights = None
    if method == "multi-label":
        weights = [ [1.0]+[16.0] * len(mot) for mot in mots ] # Poids pour chaque classe de chaque tête de classification
        print(f"Poids utilisés pour le multi-label : {weights}")

    model = TOY_GRU(label_method, nbClasses=num_classes, mots=mots, weights=weights).to(DEVICE)

    print(f"Hyperparamètres : {model.get_hyperparameters()}")
    model.train(dataloader)
    model_name = get_model_name(prefix, method)
    torch.save(model, f"{model_path}/{model_name}")


def test_model(prefix, method, data_path="test/", model_path="models/"):
    model_name = get_model_name(prefix, method).split(".")[0] # On enlève l'extension .pth car elle est ajoutée dans load_model (TODO:Fix ça)
    model = load_model(model_name, model_path).to(DEVICE)
    
    X, Y = parse_log_file(f"{data_path}/{prefix}_{method}_test.txt")
    predicted = []
    y_true = []
    for i in range(len(Y)):
        y_test = Y[i]
        y_true.extend(y_test)
        X_t = torch.tensor(X[i]).to(DEVICE)

        with torch.no_grad():
            if method == "multi-label":
                outputs = model.predict(X_t)  # pred_shape : [(lettres, classes)*nbr_tete]
                pred = [torch.argmax(op, dim=1).int().detach().cpu().numpy() for op in outputs]
                pred = [list(item) for item in zip(*pred)] # On regroupe les prédictions de chaque tête pour chaque lettre
            elif method in ["multi-classe", "state"]:
                pred = torch.argmax(model.predict(X_t), dim=1).int().detach().cpu().numpy()  # Prédiction de la classe avec la plus haute probabilité
            elif method == "binaire":
                pred = (torch.sigmoid(model.predict(X_t)).squeeze(1) > 0.6).int().detach().cpu().numpy()  # Prédiction binaire
            predicted.extend(pred)
    
    res = model.scores(y_true, predicted)
    print(f"Scores : {res}")

    if method == "multi-classe":
        labels = sorted(list(set(y_true)))
        for label in labels:
            print(f"Nombre de label {label} : {predicted.count(label)} / {y_true.count(label)}")
    elif method == "binaire":
        print(sum(predicted), sum(y_true))
    return res


def build_automate(model_name, states, prefix, method, sigma, init_build, path="test/", model_path="models/", final=set()):
    final = {int(f) for f in final} if final is not None else None
    model = load_model(model_name, model_path).to(DEVICE)
    X, _ = parse_log_file(f"{path}/{prefix}_{method}_test.txt")
    
    print("Construction de l'automate...\n")
    A = TOY_Automaton(model, sigma, states, X, final=final, init_build=init_build)
    #A.emonde()

    auto_name = get_automaton_name(prefix, method, states)
    with open(f"{path}/{auto_name}", "wb") as f:
        pickle.dump(A, f)
    
    print(f"Automate construit.\nNombre d'état après émondage : {len(A.Q)} \nNombre d'états finaux : {len(A.F)}")
    return A


def test_automaton(automaton_name, model, prefix, method, path="test/"):
    auto_path = f"{path}/{automaton_name}"
    if pathlib.Path(auto_path).is_file():
        with open(auto_path, "rb") as f:
            A = pickle.load(f)
    else:
        raise FileNotFoundError(f"Automaton file {auto_path} not found. Please build the automaton before testing.")
    print(f"Testing automaton {automaton_name}...\n")

    X, Y = parse_log_file(f"{path}/{prefix}_{method}_test.txt")
    X = torch.tensor(X).to(DEVICE)
    yt = [y for sublist in Y for y in sublist]
    predicted = []

    for x in X:
        pred, _ = A.predict(x) #Attention, ici on prédit depuis -1 (change assez peu)
        predicted.extend(pred.tolist())
    print(len(predicted), len(yt))
    res = model.scores(yt, predicted)

    if method == "multi-classe":
        f1_macro, _ = res
        print(f"F1-Score : {f1_macro}")
    elif method == "binaire":
        prec, recall, f1 = res
        print(f"Scores : Precision: {prec}, Recall: {recall}, F1-score: {f1}")
    return res


def find_motifs(A:TOY_Automaton, params, motif_length, method, init_state=-1):
    target = {motif_length} if method != "state" else {A.rank[q] for q in A.F}
    motifs = {}
    words = generate_words(params, length=4000, nbr=1000)
    deb = motif_length-1
    for X in words:
        X = list(X)
        for i in range(len(X)):
            X[i] = int(X[i], 16)-10 # Convertir les caractères 'a', 'b', 'c', ... en 0, 1, 2, ...

        X = torch.tensor(X).to(DEVICE)
        pred, hist = A.predict(X, init_state=init_state)
        for i in range(len(pred)):
            if (method != "multi-label" and pred[i] in target) or (method == "multi-label" and target.intersection(pred[i]) != set()):
                motif = X[max(i-deb,0):i+1].tolist()
                motif = "".join(chr(x+97) for x in motif)
                motif = motif.replace("0x", "")
                h = tuple(hist[max(i-deb,0):i+2])
                motifs[motif] = motifs.get(motif, {h}).union({h})

    print("\nMotifs trouvés :\n")
    for motif, hist in motifs.items():
        print(f"{motif} : {len(hist)} occurrences")
        print(f"Histories : {hist}\n")



def get_auto_simple(): #Pas de label state, juste bin, multi-classe et multi-label
    automate = {
        "0" : {"a": "1a", "b": "1b", "c": "0", "d": "0"},
        "1a" : {"a": "2aa", "b": "1b", "c": "0", "d": "0"},
        "1b" : {"a": "1a", "b": "2bb", "c": "0", "d": "0"},
        "2aa" : {"a": "2aa", "b": "3aab", "c": "0", "d": "0"},
        "2bb" : {"a": "1a", "b": "3bbb", "c": "0", "d": "0"},
        "3aab" : {"a": "1a", "b": "4aabb", "c": "4aabc", "d": "0"},
        "3bbb" : {"a": "1a", "b": "4bbbb", "c": "4bbbc", "d": "0"},
        "4aabb" : {"d": "0"},
        "4aabc" : {"d": "0"},
        "4bbbb" : {"d": "0"},
        "4bbbc" : {"d": "0"}
    }
    final_states = {"4aabb", "4aabc", "4bbbb", "4bbbc"}
    params = {
        "alphabet": ["a", "b", "c", "d"],
        "mots": ["aabb", "aabc", "bbbb", "bbbc"],
        "reset_char": None
    }
    return automate, final_states, params


def get_auto_safe():  #Pas de label state, juste bin, multi-classe et multi-label
    automate = {
        "0" : {"a": "1a", "b": "1b", "c": "0"},
        "1a" : {"a": "1a", "b": "2ab", "c": "0"},
        "1b" : {"a": "2ba", "b": "1b", "c": "0"},
        "2ab" : {"a": "2ba", "b": "1b", "c": "3abc"},
        "2ba" : {"a": "1a", "b": "2ab", "c": "3bac"},
        "3abc" : {"a": "1a", "b": "1b", "c": "0"},
        "3bac" : {"a": "1a", "b": "1b", "c": "0"}
    }
    final_states = {"3abc", "3bac"}
    params = {
        "alphabet": ["a", "b", "c"],
        "mots": ["abc", "bac"],
        "reset_char": None
    }

    return automate, final_states, params


def get_auto_state_safe():
    automate = {
        "0" : {"a": "1", "b": "2", "c": "0"},
        "1" : {"a": "1", "b": "3", "c": "0"}, #1a = 1
        "2" : {"a": "4", "b": "2", "c": "0"}, #1b = 2
        "3" : {"a": "4", "b": "2", "c": "5"}, #2ab = 3
        "4" : {"a": "1", "b": "3", "c": "6"}, #2ba = 4
        "5" : {"a": "1", "b": "2", "c": "0"}, #3abc = 5
        "6" : {"a": "1", "b": "2", "c": "0"}  #3bac = 6

    }
    final_states = {"5", "6"}
    params = {
        "alphabet": ["a", "b", "c"],
        "mots": ["abc", "bac"],
        "reset_char": None
    }

    return automate, final_states, params




def get_auto_5motif():  #Pas de label state, juste bin, multi-classe et multi-label
    automate = {
        "0" : {"a": "1a", "b": "0", "c": "0", "d": "0", "e": "1e", "f": "0"},
        "1a" : {"a": "1a", "b": "2ab", "c": "0", "d": "0", "e": "1e", "f": "0"},
        "1e" : {"a": "1a", "b": "0", "c": "0", "d": "0", "e": "2ee", "f": "0"},
        "2ab" : {"a": "1a", "b": "0", "c": "3abc", "d": "0", "e": "1e", "f": "0"},
        "2ee" : {"a": "3eea", "b": "0", "c": "0", "d": "0", "e": "3eee", "f": "0"},
        "3abc" : {"a": "1a", "b": "0", "c": "0", "d": "4abcd", "e": "4abce", "f": "0"},
        "3eea" : {"a": "1a", "b": "2ab", "c": "0", "d": "4eead", "e": "1e", "f": "0"},
        "3eee" : {"a": "3eea", "b": "0", "c": "0", "d": "0", "e": "4eeee", "f": "0"},
        "4abcd" : {"a": "1a", "b": "0", "c": "0", "d": "0", "e": "1e", "f": "0"},
        "4abce" : {"a": "1a", "b": "0", "c": "0", "d": "0", "e": "2ee", "f": "0"},
        "4eead" : {"a": "1a", "b": "0", "c": "0", "d": "0", "e": "1e", "f": "0"},
        "4eeee" : {"a": "3eea", "b": "0", "c": "0", "d": "0", "e": "4eeee", "f": "0"}
    }
    final_states = {"4abcd", "4abce", "4eead", "4eeee"}
    params = {
        "alphabet": ["a", "b", "c", "d", "e", "f"],
        "mots": ["abcd", "abce", "eeee", "eead"],
        "reset_char": "f"
    }
    return automate, final_states, params


def get_auto_state_motif():
    #automate similaire à get_auto_5motif
    automate = {
        "0" : {"a": "1", "b": "0", "c": "0", "d": "0", "e": "11", "f": "0"},
        "1" : {"a": "1", "b": "4", "c": "0", "d": "0", "e": "11", "f": "0"}, #1a = 1
        "11" : {"a": "1", "b": "0", "c": "0", "d": "0", "e": "2", "f": "0"}, #1e = 11
        "4" : {"a": "1", "b": "0", "c": "6", "d": "0", "e": "11", "f": "0"}, #2ab = 4
        "2" : {"a": "7", "b": "0", "c": "0", "d": "0", "e": "5", "f": "0"}, #2ee = 2
        "6" : {"a": "1", "b": "0", "c": "0", "d": "8", "e": "8", "f": "0"},#3abc = 6
        "7" : {"a": "1", "b": "4", "c": "0", "d": "10", "e": "11", "f": "0"},#3eea = 7
        "5" : {"a": "7", "b": "0", "c": "0", "d": "0", "e": "9", "f": "0"},#3eee = 5
        "8" : {"a": "1", "b": "0", "c": "0", "d": "0", "e": "11", "f": "0"}, #4abcd = 8
        "3" : {"a": "1", "b": "0", "c": "0", "d": "0", "e": "2", "f": "0"},#4abce = 3
        "10": {"a": "1", "b": "0", "c": "0", "d": "0", "e": "11", "f": "0"}, #4eead = 10
        "9" : {"a": "7", "b": "0", "c": "0", "d": "0", "e": "9", "f": "0"} #4eeee = 9
    }
    final_states = {"8", "3", "10", "9"}
    params = {
        "alphabet": ["a", "b", "c", "d", "e", "f"],
        "mots": ["abcd", "abce", "eeee", "eead"],
        "reset_char": "f"
    }
    return automate, final_states, params








######## Main expérimentale ########
if __name__ == "__main__":
    dict_automate = {
        "safe" : get_auto_safe,
        "5motif" : get_auto_5motif,
        "etat_motif" : get_auto_state_motif,
        "etat_safe" : get_auto_state_safe
    }

    methode_table = {
        "bin": "binaire", # 1 si fin du mot, 0 sinon
        "pos": "multi-classe", # position dans le mot
        "state": "state", # état courant
        "ml": "multi-label" # une position par mot, avec 0 la lettre n'est pas dans un motif.
    }


    parser = argparse.ArgumentParser()


    #arguments obligatoires
    parser.add_argument("-p","--prefix", help="Automaton to use", required=True)

    #arguemnts optionnels
    parser.add_argument("-s","--states", help="Number of states for the automaton to build", type=int, default=100)
    parser.add_argument("-l", "--label", help="How to label the data. bin : binary, pos : position in the word, state : current state", choices=["bin", "pos", "state", "ml"], default="pos")
    parser.add_argument("-d", "--dataset", help="Build the dataset (Warning : delete the previous one if existing)", action="store_true")
    parser.add_argument("-m", "--model", help="Build the model (Warning : delete the previous one if existing)", action="store_true")
    parser.add_argument("-a", "--automaton", help="Build the automaton (Warning : delete the previous one if existing)", action="store_true")
    parser.add_argument("-i", "--init", help="Define the method to build the initial state of an automaton.", choices=["brute", "pred", "voteF", "voteQ", "find"], default="brute")
    parser.add_argument("-dot", "--dot", help="Generate DOT file for the reduced automaton", action="store_true")
    parser.add_argument("-pm", "--path_model", help="Path to save/load the model", default="models")
    parser.add_argument("-dp", "--path_data", help="Path to save/load the dataset", default="test")
    parser.add_argument("--ds-len-words-train",help="Longueur des motsdu dataset",type=int,default=400)
    parser.add_argument("--ds-len-words-test",help="Longueur des motsdu dataset",type=int,default=400)
    parser.add_argument("--ds-nb-words",help="Longueur des motsdu dataset",type=int,default=1000)
    
    args = parser.parse_args()
    print(args)

    prefix = args.prefix
    states = args.states
    model_path = args.path_model
    data_path = args.path_data
    methode = methode_table[args.label]

    if prefix in dict_automate:
        automate, final_states, params = dict_automate[prefix]()
    else:
        automate, final_states, params = get_automate(prefix)


    # création du dataset
    if args.dataset:
        print("\n\n[*] Creating dataset...\n")
        dataset_name = f"{data_path}/{prefix}_{methode}"
        len_words_train = args.ds_len_words_train
        len_words_test = args.ds_len_words_test
        nb_words = args.ds_nb_words
        # Entrainement
        words = generate_words(params, length=len_words_train, nbr=nb_words)
        if prefix in dict_automate:
            labels = get_label(automate, words, params["mots"], final_states, methode)
        else:
            labels = [accept_stream(word, automate, params["mots"], methode) for word in words]
        create_log(words, labels, f"{dataset_name}_train.txt")
        # Test
        words = generate_words(params, length=len_words_test, nbr=nb_words)
        if prefix in dict_automate:
            labels = get_label(automate, words, params["mots"], final_states, methode)
        else:
            labels = [accept_stream(word, automate, params["mots"], methode) for word in words]
        create_log(words, labels, f"{dataset_name}_test.txt")

    #entrainement du modèle
    if args.model:
        print("\n\n[*] Training model...\n")
        if methode in ["state", "multi-classe"]:
            if methode == "state":
                num_classes = len(automate) if prefix in dict_automate else len(automate.states) # On suppose que les états sont numérotés de 0 à n-1
            else:
                num_classes = max([len(word) for word in params["mots"]]) + 1 # +1 pour la classe 0 (lettre n'appartenant pas à un motif)
            print(f"Number of classes : {num_classes}")
            train_model(prefix, methode, model_path=model_path, data_path=data_path, num_classes=num_classes)
        elif methode == "multi-label":
            train_model(prefix, methode, model_path=model_path, data_path=data_path, mots=params["mots"])
        elif methode == "binaire":
            train_model(prefix, methode, model_path=model_path, data_path=data_path)

    print("\n\n[*] Testing model...\n")
    test_model(prefix, methode, model_path=model_path, data_path=data_path)

    model_name = get_model_name(prefix, methode).split(".")[0] # On enlève l'extension .pth car elle est ajoutée dans load_model (TODO:Fix ça)
    model = load_model(model_name, model_path).to(DEVICE)
    #construction de l'automate
    if args.automaton:
        print("\n\n[*] Building automaton...\nInit build method : ", args.init)
        alphabet = list(range(len(params["alphabet"])))
        if methode == "state":
            build_automate(model_name, states, prefix, methode, alphabet, init_build=args.init, path=data_path, model_path=model_path, final=final_states)
        else:
            build_automate(model_name, states, prefix, methode, alphabet, init_build=args.init, path=data_path, model_path=model_path)
    
    print("\n\n[*] Testing automaton...\n")
    auto_name = get_automaton_name(prefix, methode, states)
    test_automaton(auto_name, model, prefix, methode, path=data_path)



    with open(f"{data_path}/{auto_name}", "rb") as f:
        A :TOY_Automaton= pickle.load(f)


    if methode != "binaire":
        rang = {}
        for q in A.Q:
            rang[A.rank[q]] = rang.get(A.rank[q], set()).union({q})

        for r in sorted(rang.keys()):
            print(f"Rang {r} : {len(rang[r])} états")


    A.minimize()
    print(f"Nombre d'état après minimisation : {len(A.Q)} \nNombre d'états finaux : {len(A.F)} | {A.F}")
    
    #Initial state
    print("\n\n[*] Finding initial state...\n")
    init_state = A.find_initial_state()
    print(f"Initial state found: {init_state}")

    #Motifs reconnus par l'automate
    motifs_length = max(len(motif) for motif in params["mots"])
    print(f"\n\n[*] Finding motifs from {init_state}...\n")
    find_motifs(A, params, motifs_length, methode, init_state=init_state)

    if prefix not in dict_automate: #besoin de fsm pour comparer
        B = light_automaton(automate)
        iso = is_isomorphic(B, A)[0]
        print(f"Isomorphisme avec l'automate de référence : {iso}")
        #Sprint(f"Distance d'édition (NetworkX) : {ged_nx(B, A)}")


    #Visualisation de l'automate réduit
    if args.dot:
        print("\n\n[*] Generating DOT file for the reduced automaton...\n")
        with open(f"{data_path}/{auto_name.split('.')[0]}_red.dot", "w") as f:
            f.write(A.dot())

    