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
from utils_test import *
import random
import os



# Return the F1 scores




#===========================================================

def test_of_tests():
    """
    Le test qui test beacoup de fonction utilitaire

    Aussi un template pour les autres tests
    """
    automate, final_states, info_automate = get_fsm_by_id(121) # el_automate
    show_automation(automate,info_automate)
    dataset_name = "test/el_grand_test"
    create_dataset_and_save_it(
        automate, info_automate, "state", 1000, 1000, 400, 40, dataset_name)
    mots = info_automate["mots"]
    weights = [[1.0]+[16.0]*(len(mot)) for mot in mots]
    M = create_model(mots, "state", weights,automate)
    M = train_model(M, 500, dataset_name)
    F1 = test_model(M, "state", dataset_name)
    A = get_automate_from_model(M, info_automate, 100, dataset_name, "pred")
    F2 = test_automate(A, M, info_automate, mots,
                       "state", -1, dataset_name)
    A.minimize()
    init_state = A.find_initial_state()
    B = light_automaton(automate)
    print(is_isomorphic(B, A)[0])
    print(F1,F2)


def best_method_init():
    automate, final_states, info_automate = create_first_auto(
        "ml", len_words=5)  # el_automate
    dataset_name = "test/el_grand_test"
    create_dataset_and_save_it(
        automate, info_automate, "multi-label", 1000, 1000, 400, 40, dataset_name)
    mots = info_automate["mots"]
    weights = [[1.0]+[16.0]*(len(mot)) for mot in mots]
    M = create_model(mots, "multi-label", weights)
    M = train_model(M, 500, dataset_name)
    options = ["brute", "pred", "voteF", "voteQ", "find"]
    for o in options:
        A = get_automate_from_model(M, info_automate, 100, dataset_name, o)
        F2 = test_automate(A, M, info_automate, mots,
                           "multi-label", -1, dataset_name)
        A.minimize()
        init_st = A.find_initial_state()
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
        M = train_model(M, 25, dataset_name)
        F1s = test_model(M, dataset_name)
        A = get_automate_from_model(
            M, info_automate, 100, dataset_name, "pred")
        F2s = test_automate(A, M, info_automate, mots,
                            "multi-label", -1, dataset_name)
        A.minimize()
        ini = A.find_initial_state()
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
            M = train_model(M, 25, dataset_name)
            F1s = test_model(M, "multi-label", dataset_name)
            A = get_automate_from_model(
                M, info_automate, 100, dataset_name, "pred")
            F2s = test_automate(A, M, info_automate, mots,
                                "multi-label", -1, dataset_name)
            A.minimize()
            ini = A.find_initial_state()
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
        M = train_model(M, 400, dataset_name)
        nb_states = int(len(automate.states) / 2)
        step = int(nb_states)
        B = light_automaton(automate)
        for i in range(20):
            A = get_automate_from_model(
                M, info_automate, nb_states, dataset_name, "pred")
            F2s = test_automate(
                A, M, info_automate, mots, "multi-label", -1, dataset_name)
            A.minimize()
            ini = A.find_initial_state()
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
            M = train_model(M, 500, dataset_name)
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
        M = train_model(M, 500, dataset_name)
        F1 = test_model(M, "multi-label", dataset_name)
        A = get_automate_from_model(M, info_automate, 100, dataset_name, "pred")
        F2 = test_automate(A, M, info_automate, mots,"multi-label", -1, dataset_name)
        A.minimize()
        init_state = A.find_initial_state()
        B = light_automaton(automate)
        iso = is_isomorphic(B, A)[0]
        mean_f1 = np.mean([x[-1] for x in F1])
        mean_f2 = np.mean([x[-1] for x in F2])
        mean_f1_0 = np.mean([x[0] for x in F1])
        mean_f2_0 = np.mean([x[0] for x in F2])
        add_log("perf","perf_log.log",f"ID:{id_auto},NB_WORDS:{len(mots)},LEN_WORDS:{len(mots[0])},F1_M:{mean_f1},F1_M0:{mean_f1_0},F1_A:{mean_f2},F1_A0:{mean_f2_0},ISO:{iso}")



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
            M = train_model(M, 500, dataset_name)
            F1 = test_model(M, "multi-label", dataset_name)
            A = get_automate_from_model(M, info_automate, 100, dataset_name, "pred")
            F2 = test_automate(A, M, info_automate, mots,"multi-label", -1, dataset_name)
            A.minimize()
            init_state = A.find_initial_state()
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
            F2 = test_automate(A, M, info_automate, mots,"multi-label", -1, dataset_name)
            A.minimize()
            init_state = A.find_initial_state()
            mean_f1 = np.mean([x[-1] for x in F1])
            mean_f2 = np.mean([x[-1] for x in F2])
            add_log("len_words", "len_words.log", f"ID:{id_auto},LEN_WORDS:{k},F1_M:{mean_f1},F1_A:{mean_f2}")

if __name__ == "__main__":
    test_of_tests()