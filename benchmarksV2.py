import log_custom
from log_custom import add_log
from collections import namedtuple
from typing import List
import numpy as np

import pickle
import random
import torch
import pathlib
import pickle
import json
import argparse
from re import finditer

from logDataset import LogDataset
# from gru_multi_test import Test_TMGRU
# from gru_test import Test_TGRU
# from gru_labels_test import Test_LGRU
from gru_merge import TOY_GRU
# from automaton import Automaton
# from automaton_multi import Automaton_multi
# from automaton_multi_label import Automaton_multi_label
from automaton_merge import TOY_Automaton
from main import load_model, pad_batch
from utils import get_device
from build_auto import accept_stream, get_automate, light_automaton, no_overlap, parse, save_fsm, load_fsm
from isomorphe import is_isomorphic, ged_nx, Weisfeiler_Leman

from logDataset import LogDataset
from torch.utils.data import DataLoader, TensorDataset

import toy_example

import random
import os


def dist(f1, f2):
    return abs(f1 - f2)


def get_index_closer(tab, v):
    return np.argmin([dist(x, v) for x in tab])


# On sort une note sur 5, 0 grosse différence 5 pareil
def compare_twof1score(f1: float, f2: float):
    diff = abs(f1 - f2)
    if diff < 1e-4:
        return 5
    if diff < 1e-3:
        return 4
    if diff < 1e-2:
        return 3
    if diff < 1e-1:
        return 2
    if diff < 1:
        return 1
    else:
        return 0


def comapre_tab_f1(tab: List[List[float]]):
    for model_i in range(len(tab)):
        for model_j in range(model_i+1, len(tab)):
            scores = []
            for mot_i in range(len(tab[model_i])):
                for label_i in range(len(tab[model_i][mot_i])):
                    scores.append(compare_twof1score(
                        tab[model_i][mot_i][label_i], tab[model_j][mot_i][label_i]))
            print(f"model {model_i} -> model{model_j} : {np.mean(scores)}/5")
        pass


def create_first_auto(prefix, len_words=None, nb_words=None, is_reset=True):
    automate, final_states, params = None, None, None

    print(f"Creating first_auto, generating it...")
    # configuration aléatoire de l'automate
    alpha_end = random.randint(100, 103)  # Entre [a-c] et [a-f]
    alphabet = [chr(i) for i in range(97, alpha_end)]
    if nb_words is None:
        nbr_max = random.randint(3, 20)
    else:
        nbr_max = nb_words
    if len_words is None:
        taille = random.randint(3, 10)
    else:
        taille = len_words
    if is_reset:
        reset = random.choice(alphabet)
    else:
        reset = None
    # genération du langage et de l'automate
    L, reset = no_overlap(alphabet=alphabet, nbr=nbr_max,
                          length=taille, reset=reset)
    regex = f"([{"".join(alphabet)}]|.)*({'|'.join(L)})"
    fsm = parse(regex).to_fsm()

    automate = fsm.reduce()
    final_states = set(fsm.finals)
    params = {
        "alphabet": alphabet,
        "mots": L,
        "reset_char": reset
    }
    print(f"Alphabet : {alphabet}\nRegex : {regex}\nReset : {reset}")
    print(automate)

    return automate, final_states, params


def create_dataset_and_save_it(automate, info_automate, methode, nb_words_test: int, nb_words_train: int, len_test: int, len_train: int, dataset_name: str):
    words = toy_example.generate_words(
        info_automate, length=len_train, nbr=nb_words_train)
    labels = [accept_stream(
        word, automate, info_automate["mots"], methode) for word in words]
    toy_example.create_log(words, labels, f"{dataset_name}_train.txt")

    words = toy_example.generate_words(
        info_automate, length=len_test, nbr=nb_words_test)
    labels = [accept_stream(
        word, automate, info_automate["mots"], methode) for word in words]
    toy_example.create_log(words, labels, f"{dataset_name}_test.txt")

    words = toy_example.generate_words(
        info_automate, length=len_test, nbr=nb_words_test)
    labels = [accept_stream(
        word, automate, info_automate["mots"], methode) for word in words]
    toy_example.create_log(words, labels, f"{dataset_name}_val.txt")


def create_model(mots, method: str, weights) -> TOY_GRU:
    num_classes = max([len(word) for word in mots]) + 1
    # state est une version de multi-classe
    label_method = "multi-classe" if method == "state" else method
    return TOY_GRU(label_method, nbClasses=num_classes, mots=mots, weights=weights).to(toy_example.DEVICE)


def train_model(model: TOY_GRU, epochs: int, dataset_name: str):
    X, Y = toy_example.parse_log_file(f"{dataset_name}_train.txt")
    dataset_tr = TensorDataset(torch.tensor(X), torch.tensor(Y))
    dataloader_tr = DataLoader(dataset_tr, batch_size=32, shuffle=True)

    X_val, Y_val = toy_example.parse_log_file(f"{dataset_name}_val.txt")
    dataset_val = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False)

    TOY_GRU.epochs = epochs
    print(f"Hyperparamètres : {model.get_hyperparameters()}")
    model.train_model(dataloader_tr, dataloader_val)

    return model


# Return the F1 scores
def test_model(model: TOY_GRU, method: str, dataset_name: str):
    X, Y = toy_example.parse_log_file(f"{dataset_name}_test.txt")
    predicted = []
    y_true = []
    for i in range(len(Y)):
        y_test = Y[i]
        y_true.extend(y_test)
        X_t = torch.tensor(X[i]).to(toy_example.DEVICE)
        with torch.no_grad():
            if method == "multi-label":
                # pred_shape : [(lettres, classes)*nbr_tete]
                outputs = model.predict(X_t)
                pred = [torch.argmax(op, dim=1).int(
                ).detach().cpu().numpy() for op in outputs]
                # On regroupe les prédictions de chaque tête pour chaque lettre
                pred = [list(item) for item in zip(*pred)]
            elif method in ["multi-classe", "state"]:
                pred = torch.argmax(model.predict(
                    X_t), dim=1).int().detach().cpu().numpy()
            predicted.extend(pred)
    model.scores(y_true, predicted)
    return model.give_f1_scores(y_true, predicted)


def get_automate_from_model(model: TOY_GRU, info_automate, nb_states: int, dataset_name: str, init_method: str,final=set()) -> TOY_Automaton:
    print("Création automate (peut être long...)")
    final = {int(f) for f in final} if final is not None else None
    X, _ = toy_example.parse_log_file(f"{dataset_name}_test.txt")
    alphabet = list(range(len(info_automate["alphabet"])))
    return TOY_Automaton(model, alphabet, nb_states, X, init_build=init_method,final=final)


def test_automate(A: TOY_Automaton, model: TOY_GRU, info_automate, mots, methode, init, dataset_name):
    X, Y = toy_example.parse_log_file(f"{dataset_name}_test.txt")
    X = torch.tensor(X).to(toy_example.DEVICE)
    yt = [y for sublist in Y for y in sublist]
    predicted = []
    for x in X:
        # Attention, ici on prédit depuis -1 (change assez peu)
        pred, _ = A.predict(x)
        predicted.extend(pred.tolist())
    print(len(predicted), len(yt))
    model.scores(yt, predicted)
    res = model.give_f1_scores(yt, predicted)
    #if methode == "multi-classe":
    #    f1_macro, _ = res
    #    print(f"F1-Score : {f1_macro}")
    #elif methode == "binaire":
    #    prec, recall, f1 = res
    #    print(f"Scores : Precision: {prec}, Recall: {recall}, F1-score: {f1}")
    #motifs_length = max(len(motif) for motif in info_automate["mots"])
    #toy_example.find_motifs(
    #    A, info_automate, motifs_length, methode, init_state=init)
    return res


def test_of_tests():
    automate, final_states, info_automate = get_fsm_by_id(121) # el_automate
    show_automation(automate,info_automate)
    dataset_name = "test/el_grand_test"
    create_dataset_and_save_it(
        automate, info_automate, "multi-label", 1000, 1000, 400, 40, dataset_name)
    mots = info_automate["mots"]
    weights = [[1.0]+[16.0]*(len(mot)) for mot in mots]
    M = create_model(mots, "multi-label", weights)
    train_model(M, 500, dataset_name)
    F1 = test_model(M, "multi-label", dataset_name)
    A = get_automate_from_model(M, info_automate, 100, dataset_name, "pred")
    F2 = test_automate(A, M, info_automate, mots,
                       "multi-label", -1, dataset_name)
    A.minimize()
    init_state = A.find_initial_state()
    B = light_automaton(automate)
    print(is_isomorphic(B, A)[0])
    print(F2)


def bug_hunt():
    automate, final_states, info_automate = create_first_auto(
        "ml", len_words=3, nb_words=10)  # el_automate
    dataset_name = "test/el_grand_test"
    create_dataset_and_save_it(
        automate, info_automate, "multi-label", 1000, 1000, 400, 30, dataset_name)
    mots = info_automate["mots"]
    weights = [[1.0]+[16.0]*(len(mot)) for mot in mots]
    M = create_model(mots, "multi-label", weights)
    B = light_automaton(automate)
    F2 = test_automate(B, M, info_automate, mots,
                       "multi-label", -1, dataset_name)


def best_method_init():
    automate, final_states, info_automate = create_first_auto(
        "ml", len_words=5)  # el_automate
    dataset_name = "test/el_grand_test"
    create_dataset_and_save_it(
        automate, info_automate, "multi-label", 1000, 1000, 400, 40, dataset_name)
    mots = info_automate["mots"]
    weights = [[1.0]+[16.0]*(len(mot)) for mot in mots]
    M = create_model(mots, "multi-label", weights)
    train_model(M, 500, dataset_name)
    options = ["brute", "pred", "voteF", "voteQ", "find"]
    for o in options:
        A = get_automate_from_model(M, info_automate, 100, dataset_name, o)
        A.minimize()
        init_st = A.find_initial_state()
        F2 = test_automate(A, M, info_automate, mots,
                           "multi-label", init_st, dataset_name)
        B = light_automaton(automate)
        print(np.mean([x[-1] for x in F2]))
        print(is_isomorphic(B, A)[0])


def gradual_epoch_loss_test():
    automate, final_states, info_automate = create_first_auto(
        "ml", len_words=5)  # el_automate
    dataset_name = "test/el_grand_test"
    create_dataset_and_save_it(
        automate, info_automate, "multi-label", 1000, 1000, 400, 40, dataset_name)
    mots = info_automate["mots"]
    weights = [[1.0]+[16.0]*(len(mot)) for mot in mots]
    B = light_automaton(automate)
    M = create_model(mots, "multi-label", weights)
    for i in range(50):
        train_model(M, 25, dataset_name)
        F1s = test_model(M, dataset_name)
        A = get_automate_from_model(
            M, info_automate, 100, dataset_name, "pred")
        A.minimize()
        ini = A.find_initial_state()
        F2s = test_automate(A, M, info_automate, mots,
                            "multi-label", ini, dataset_name)
        add_log("grad_ep", "gradual_ep_model.log", f"EPOCH:{(i+1)*25}" + ",MODEL:"+str(np.mean([x[-1] for x in F1s])))
        add_log("grad_ep", "gradual_ep_auto.log", f"EPOCH:{(i+1)*25}" + ",MODEL:"+str(np.mean([x[-1] for x in F2s]))+",ISO:"+str(is_isomorphic(B, A)[0]))


def create_automatas():
    os.makedirs(f"fsms/", exist_ok=True)
    i = 0
    while range(500):
        automate, final_states, info_automate = create_first_auto(
            "ml")  # el_automate
        N = len(automate.states)
        Z = len(info_automate["mots"]) * len(info_automate["mots"][0])
        pak = {"automate": automate,
               "final_states": final_states, "params": info_automate}
        save_fsm(pak, f"I_{i}_N_{N}_Z_{Z}", path="fsms/")
        i += 1

    pass


NB_IN_TEST = 150


def get_fsm_by_id(id: int):
    all_f = os.listdir("fsms/New_data")
    for f in all_f:
        if int(f.split("_")[1]) == id:
            data = load_fsm(f.split(".")[0], path="fsms/New_data/")
            automate = data["automate"]
            final_states = data["final_states"]
            params = data["params"]
            print("ID AUTO UTILISE:", id)
            return automate, final_states, params
    print(f"ERROR NO FSMS WITH ID {id} FOUND")


def gradual_epoch_loss_testV2():
    for N in range(25):
        automate, final_states, info_automate = get_fsm_by_id(N)  # el_automate
        log_custom.main_dir = f"auto{N}"
        dataset_name = "test/el_grand_test"
        create_dataset_and_save_it(
            automate, info_automate, "multi-label", 100, 100, 400, 400, dataset_name)
        mots = info_automate["mots"]
        weights = [[1.0]+[16.0]*(len(mot)) for mot in mots]
        B = light_automaton(automate)
        M = create_model(mots, "multi-label", weights)
        for i in range(1):
            train_model(M, 25, dataset_name)
            F1s = test_model(M, "multi-label", dataset_name)
            A = get_automate_from_model(
                M, info_automate, 100, dataset_name, "pred")
            A.minimize()
            ini = A.find_initial_state()
            F2s = test_automate(A, M, info_automate, mots,
                                "multi-label", ini, dataset_name)
            add_log("grad_ep", "gradual_ep_model.log", f"EPOCH:{(i+1)*25}" + ",MODEL:"+str(np.mean([x[-1] for x in F1s])))
            add_log("grad_ep", "gradual_ep_auto.log", f"EPOCH:{(i+1)*25}" + ",MODEL:"+str(np.mean([x[-1] for x in F2s]))+",ISO:"+str(is_isomorphic(B, A)[0]))


def ressemblance_score(A, B):
    # on applique WL et on compare les couleurs
    color_A = Weisfeiler_Leman(A)
    color_B = Weisfeiler_Leman(B)

    sorted_colors_A = sorted(color_A.values())
    sorted_colors_B = sorted(color_B.values())
    print(sorted_colors_A)
    print(sorted_colors_B)
    isomorphic = sorted_colors_A == sorted_colors_B

    if isomorphic:
        return 1
    return float(len([x for x in sorted_colors_A if x in sorted_colors_B])) / float(len(sorted_colors_A))


def trash_func():
    automate, final_states, info_automate = get_fsm_by_id(
        random.randint(0, NB_IN_TEST))  # el_automate
    dataset_name = "test/el_grand_test"
    create_dataset_and_save_it(
        automate, info_automate, "multi-label", 1000, 1000, 400, 40, dataset_name)
    mots = info_automate["mots"]
    weights = [[1.0]+[16.0]*(len(mot)) for mot in mots]
    M = create_model(mots, "multi-label", weights)
    M = train_model(M, 500, dataset_name)
    F1s = test_model(M, "multi-label", dataset_name)
    A = get_automate_from_model(M, info_automate, 100, dataset_name, "pred")
    A.minimize()
    B = light_automaton(automate)
    print(ressemblance_score(B, A))


def the_number_of_states_needed():
    for N in range(10):
        id_auto = random.randint(0, NB_IN_TEST)
        automate, final_states, info_automate = get_fsm_by_id(
            id_auto)  # el_automate
        dataset_name = "test/el_grand_test"
        create_dataset_and_save_it(
            automate, info_automate, "multi-label", 1000, 1000, 400, 40, dataset_name)
        mots = info_automate["mots"]
        weights = [[1.0]+[16.0]*(len(mot)) for mot in mots]
        M = create_model(mots, "multi-label", weights)
        train_model(M, 400, dataset_name)
        nb_states = int(len(automate.states) / 2)
        step = int(nb_states)
        B = light_automaton(automate)
        for i in range(20):
            A = get_automate_from_model(
                M, info_automate, nb_states, dataset_name, "pred")
            A.minimize()
            ini = A.find_initial_state()
            F1s = F2s = test_automate(
                A, M, info_automate, mots, "multi-label", ini, dataset_name)
            mean_f1 = np.mean([x[-1] for x in F1s])
            add_log("nb_st_test", f"nb_state_id{id_auto}", f"STATES:{nb_states},F1:{mean_f1},SAME:{ressemblance_score(A, B)}")
            nb_states += step


def nuage_de_points():
    l = [x for x in range(NB_IN_TEST+1)]
    random.shuffle(l)
    for id_auto in l:
        automate, final_states, info_automate = get_fsm_by_id(
            id_auto)  # el_automate
        dataset_name = "test/el_grand_test"
        for k in range(100, 2500, 100):
            create_dataset_and_save_it(
                automate, info_automate, "multi-label", k, k, 400, 40, dataset_name)
            mots = info_automate["mots"]
            weights = [[1.0]+[16.0]*(len(mot)) for mot in mots]
            M = create_model(mots, "multi-label", weights)
            train_model(M, 500, dataset_name)
            F1 = test_model(M, "multi-label", dataset_name)
            mean_f1 = np.mean([x[-1] for x in F1])
            add_log("nuage", "nuage_log.log", f"ID:{id_auto},NB_WORDS:{k},F1:{mean_f1}")


def perf_by_fsm():
    l = [x for x in range(NB_IN_TEST+1)]
    random.shuffle(l)
    for id_auto in l:
        automate, final_states, info_automate = get_fsm_by_id(id_auto)
        dataset_name = "test/el_grand_test"
        create_dataset_and_save_it(automate, info_automate, "multi-label", 1000, 1000, 400, 40, dataset_name)
        mots = info_automate["mots"]
        weights = [[1.0]+[16.0]*(len(mot)) for mot in mots]
        M = create_model(mots, "multi-label", weights)
        train_model(M, 500, dataset_name)
        F1 = test_model(M, "multi-label", dataset_name)
        A = get_automate_from_model(M, info_automate, 100, dataset_name, "pred")
        A.minimize()
        init_state = A.find_initial_state()
        F2 = test_automate(A, M, info_automate, mots,"multi-label", init_state, dataset_name)
        B = light_automaton(automate)
        iso = is_isomorphic(B, A)[0]
        mean_f1 = np.mean([x[-1] for x in F1])
        mean_f2 = np.mean([x[-1] for x in F2])
        mean_f1_0 = np.mean([x[0] for x in F1])
        mean_f2_0 = np.mean([x[0] for x in F2])
        add_log("perf","perf_log.log",f"ID:{id_auto},NB_WORDS:{len(mots)},LEN_WORDS:{len(mots[0])},F1_M:{mean_f1},F1_M0:{mean_f1_0},F1_A:{mean_f2},F1_A0:{mean_f2_0},ISO:{iso}")

def make_weights(id_w,mots):
    if id_w == 0:
        return [[1.0]+[16.0]*(len(mot)) for mot in mots]
    if id_w == 1:
        return [[1.0]+ ([0.1]*(len(mot)-1))+[256.0] for mot in mots]
    if id_w == 2:
        return [[float(pow(2,i)) for i in range(len(mot)+1)] for mot in mots]
    if id_w == 3:
        return [[float(pow(2,i)) for i in range(4,len(mot)+5)] for mot in mots]
    if id_w == 4:
        return [[256.0]+ ([0.1]*(len(mot)-1))+[256.0] for mot in mots]
    if id_w == 5:
        return [[float(pow(2,len(mot)-1))] + [float(pow(2,i)) for i in range(len(mot))] for mot in mots]
    if id_w == 6:
        return [[1.0]+[1.0]*(len(mot)) for mot in mots]
    if id_w == 7:
        return [[256.0]+ ([64.0]*(len(mot)-1))+[256.0] for mot in mots]


def test_weights():
    l = [x for x in range(NB_IN_TEST+1)]
    random.shuffle(l)
    for id_auto in l:
        automate, final_states, info_automate = get_fsm_by_id(id_auto)
        dataset_name = "test/el_grand_test"
        create_dataset_and_save_it(automate, info_automate, "multi-label", 1000, 1000, 400, 60, dataset_name)
        mots = info_automate["mots"]
        for weight_method in range(8):
            weights = make_weights(weight_method,mots)
            M = create_model(mots, "multi-label", weights)
            train_model(M, 500, dataset_name)
            F1 = test_model(M, "multi-label", dataset_name)
            A = get_automate_from_model(M, info_automate, 100, dataset_name, "pred")
            A.minimize()
            init_state = A.find_initial_state()
            F2 = test_automate(A, M, info_automate, mots,"multi-label", init_state, dataset_name)
            B = light_automaton(automate)
            iso = is_isomorphic(B, A)[0]
            mean_f1 = np.mean([x[-1] for x in F1])
            mean_f2 = np.mean([x[-1] for x in F2])
            add_log("perf","weight.log",f"ID:{id_auto},NB_WORDS:{len(mots)},LEN_WORDS:{len(mots[0])},F1_M:{mean_f1},F1_A:{mean_f2},ISO:{iso},WEIGHTS:{weight_method}")

def show_automation(auto,infos):
    print(f"{auto}\n{infos}")
    

def stress_test_dataset():
    l = [x for x in range(NB_IN_TEST+1)]
    random.shuffle(l)
    for id_auto in l:
        automate, final_states, info_automate = get_fsm_by_id(121)
        dataset_name = "test/el_grand_test"
        for i in range(20):
            create_dataset_and_save_it(automate, info_automate, "multi-label", 1000, 1000, 400, 40, dataset_name)
            mots = info_automate["mots"]
            weights = make_weights(0,mots)
            M = create_model(mots, "multi-label", weights)
            M = train_model(M, 500, dataset_name)
            F1 = test_model(M, "multi-label", dataset_name)

            A = get_automate_from_model(M, info_automate, 100, dataset_name, "pred")
            F2 = test_automate(A, M, info_automate, mots,"multi-label", -1, dataset_name)


            init_state = A.find_initial_state()
            A.minimize()


            B = light_automaton(automate)
            iso = is_isomorphic(B, A)[0]
            mean_f1 = np.mean([x[-1] for x in F1])
            mean_f2 = np.mean([x[-1] for x in F2])
            add_log("perf","stress.log",f"ID:{id_auto},N:{i},NB_WORDS:{len(mots)},LEN_WORDS:{len(mots[0])},F1_M:{mean_f1},F1_A:{mean_f2},ISO:{iso}")


def len_words_test():
    l = [x for x in range(NB_IN_TEST+1)]
    random.shuffle(l)
    for id_auto in l:
        automate, final_states, info_automate = get_fsm_by_id(
            id_auto)  # el_automate
        dataset_name = "test/el_grand_test"
        for k in range(10, 1000, 100):
            create_dataset_and_save_it(
                automate, info_automate, "multi-label",1000, 1000, k, k, dataset_name)
            mots = info_automate["mots"]
            weights = [[1.0]+[16.0]*(len(mot)) for mot in mots]
            M = create_model(mots, "multi-label", weights)
            M = train_model(M, 500, dataset_name)
            F1 = test_model(M, "multi-label", dataset_name)
            A = get_automate_from_model(M, info_automate,int(len(automate.states)*2.5), dataset_name, "pred")
            A.minimize()
            init_state = A.find_initial_state()
            F2 = test_automate(A, M, info_automate, mots,"multi-label", init_state, dataset_name)
            mean_f1 = np.mean([x[-1] for x in F1])
            mean_f2 = np.mean([x[-1] for x in F2])
            add_log("len_words", "len_words.log", f"ID:{id_auto},LEN_WORDS:{k},F1_M:{mean_f1},F1_A:{mean_f2}")

if __name__ == "__main__":
    test_of_tests()
