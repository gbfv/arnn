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
from gru_merge import TOY_GRU
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


def create_first_auto(prefix, len_words=None, nb_words=None, is_reset=True):
    """
    Crée l'automate du language de base, si les parametres sont None une valeure aléatoire est mise
    """
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

def create_dataset_and_save_it(automate, info_automate, methode:str, nb_words_test: int, nb_words_train: int, len_test: int, len_train: int, dataset_name: str):
    """
    Crée les 3 datasets en fonction des paramètres
    """
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



def create_model(mots, method: str, weights,automate_if_state=None,epochs=500,embedding_dim=2,hidden_dim=50) -> TOY_GRU:
    """
    Crée et retourne un model non entrainé sur un language et méthode donnée
    """
    num_classes = -1
    if method == "state":
        num_classes = len(automate_if_state.states)
    else:
        num_classes = max([len(word) for word in mots]) + 1
    # state est une version de multi-classe
    label_method = "multi-classe" if method == "state" else method
    return TOY_GRU(label_method, nbClasses=num_classes, mots=mots, weights=weights,epochs=epochs,embedding_dim=embedding_dim,hidden_dim=hidden_dim).to(toy_example.DEVICE)

def train_model(model: TOY_GRU, epochs: int, dataset_name: str):
    """
    Entraine le model donné.
    """
    X, Y = toy_example.parse_log_file(f"{dataset_name}_train.txt")
    dataset_tr = TensorDataset(torch.tensor(X), torch.tensor(Y))
    dataloader_tr = DataLoader(dataset_tr, batch_size=32, shuffle=True)

    X_val, Y_val = toy_example.parse_log_file(f"{dataset_name}_val.txt")
    dataset_val = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False)

    model.epochs = epochs
    print(f"Hyperparamètres : {model.get_hyperparameters()}")
    model.train_model(dataloader_tr, dataloader_val)

    return model

def make_weights(id_w,mots):
    """
    Génère les poids d'un modèle selon l'id_w
    """
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


def test_model(model: TOY_GRU, method: str, dataset_name: str):
    """
    Test un modèle donné
    """
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
    """
    Crée l'automate à partir du modèle
    """
    print("Création automate (peut être long...)")
    final = {int(f) for f in final} if final is not None else None
    X, _ = toy_example.parse_log_file(f"{dataset_name}_test.txt")
    alphabet = list(range(len(info_automate["alphabet"])))
    return TOY_Automaton(model, alphabet, nb_states, X, init_build=init_method,final=final)

def test_automate(A: TOY_Automaton, model: TOY_GRU, info_automate, mots, methode, init, dataset_name):
    """
    Test l'automate donné avec le dataset donné
    """
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
    return res
