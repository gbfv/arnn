import matplotlib
import matplotlib.pyplot as plt
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

import toy_example

#Tout les fichiers logs vont être de type VAR1:VAL1, VAR2:VAL2
def load_log_file_for_benchmark(filename):
    data = [x.removesuffix("\n") for x in  open(filename,"r").readlines()]
    #Get the attributes
    FirstLine:str = data[0]
    All_Fields = [x.split(":")[0] for x in FirstLine.split(",")]
    LogData = namedtuple("LogData",All_Fields)
    res :List[LogData] = []
    for line in data:
        line_data = [x.split(":")[1] for x in line.split(",")]
        res.append(LogData(*line_data))
    return res


def dist(f1,f2):
    return abs(f1 - f2)

def get_index_closer(tab,v):
    return np.argmin([dist(x,v) for x in tab])
    

def epoch_time_graph(filename):
    r = load_log_file_for_benchmark(filename)
    epochs = [int(getattr(x,"EPOCH")) for x in r]
    loss = [float(getattr(x,"LOSS"))for x in r]
    one_percent = loss[-1] * 1.01
    five_percent = loss[-1] * 1.05
    ten_percent = loss[-1] * 1.1
    fifty_percent = loss[-1] * 1.5
    one_percent_epoch_index = get_index_closer(loss,one_percent)
    five_percent_epoch_index = get_index_closer(loss,five_percent)
    ten_percent_epoch_index = get_index_closer(loss,ten_percent)
    fifty_percent_epoch_index = get_index_closer(loss,fifty_percent)

    plt.plot(epochs,loss,)
    plt.axis((min(epochs),max(epochs),1e-5,5))
    plt.title("Epoch Time POS")
    plt.axvline(one_percent_epoch_index,color="red",label=f"99% value (EPOCH:{one_percent_epoch_index},LOSS:{loss[one_percent_epoch_index]})")
    plt.axvline(ten_percent_epoch_index,color="orange",label=f"90% value (EPOCH:{ten_percent_epoch_index},LOSS:{loss[ten_percent_epoch_index]}))")
    plt.axvline(fifty_percent_epoch_index,color="yellow",label=f"50% value (EPOCH:{fifty_percent_epoch_index},,LOSS:{loss[fifty_percent_epoch_index]}))")
    plt.axvline(five_percent_epoch_index,color="blue",label=f"95% value (EPOCH:{five_percent_epoch_index},,LOSS:{loss[five_percent_epoch_index]}))")
    plt.xlabel("EPOCH")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.show()


def simple_graph(data,Field1:str,Field2:str,LabelX:str,LabelY:str):
    plt.cla()
    Y = [float(getattr(x,Field1)) for x in data]
    X = [float(getattr(x,Field2)) for x in data]
    plt.plot(Y,X)
    plt.axis((min(Y),max(Y),0,max(max(X),5)))
    plt.show()


#On sort une note sur 5, 0 grosse différence 5 pareil
def compare_twof1score(f1:float,f2:float):
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

def comapre_tab_f1(tab:List[List[float]]):
    for model_i in range(len(tab)):
        for model_j in range(model_i+1,len(tab)):
            scores = []
            for mot_i in range(len(tab[model_i])):
                for label_i in range(len(tab[model_i][mot_i])):
                    scores.append(compare_twof1score(tab[model_i][mot_i][label_i],tab[model_j][mot_i][label_i]))
            print(f"model {model_i} -> model{model_j} : {np.mean(scores)}/5")
        pass


def model_test(N:int):
    #create a automate and a dataset then we create N mdels and compare them
    automate, final_states, params = get_automate("ml") #el_automate
    print("\n\n[*] Creating dataset...\n")
    methode = "multi-label"
    prefix = "ml"
    model_path = "models"
    data_path = "test"
    dataset_name = f"{data_path}/{prefix}_{methode}"
    len_words_train = 40
    len_words_test = 400
    nb_words = 1000
    states = 100

    # Entrainement
    words = toy_example.generate_words(params, length=len_words_train, nbr=nb_words)
    labels = [accept_stream(word, automate, params["mots"], methode) for word in words]
    toy_example.create_log(words, labels, f"{dataset_name}_train.txt")

    words = toy_example.generate_words(params, length=len_words_test, nbr=nb_words)
    labels = [accept_stream(word, automate, params["mots"], methode) for word in words]
    toy_example.create_log(words, labels, f"{dataset_name}_test.txt")
    num_classes = max([len(word) for word in params["mots"]]) + 1 
    
    f1_scores = []
    automates = []
    for n in range(N):
        toy_example.train_model(prefix, methode, model_path=model_path, data_path=data_path, mots=params["mots"])
        model_name = toy_example.get_model_name(prefix, methode).split(".")[0] # On enlève l'extension .pth car elle est ajoutée dans load_model (TODO:Fix ça)
        model = load_model(model_name, model_path).to(toy_example.DEVICE)

        X, Y = toy_example.parse_log_file(f"{data_path}/{prefix}_{methode}_test.txt")
        predicted = []
        y_true = []
        for i in range(len(Y)):
            y_test = Y[i]
            y_true.extend(y_test)
            X_t = torch.tensor(X[i]).to(toy_example.DEVICE)

            with torch.no_grad():
                    outputs = model.predict(X_t)  # pred_shape : [(lettres, classes)*nbr_tete]
                    pred = [torch.argmax(op, dim=1).int().detach().cpu().numpy() for op in outputs]
                    pred = [list(item) for item in zip(*pred)] # On regroupe les prédictions de chaque tête pour chaque lettre
                    predicted.extend(pred)
                    
        f1_scores.append(model.give_f1_scores_ml(y_true, predicted))
        model_name = toy_example.get_model_name(prefix, methode).split(".")[0] # On enlève l'extension .pth car elle est ajoutée dans load_model (TODO:Fix ça)
        model = toy_example.load_model(model_name, model_path).to(toy_example.DEVICE)
        alphabet = list(range(len(params["alphabet"])))
        toy_example.build_automate(model_name, states, prefix, methode, alphabet, init_build="pred", path=data_path, model_path=model_path)
        auto_name = toy_example.get_automaton_name(prefix, methode, states)
        with open(f"{data_path}/{auto_name}", "rb") as f:
            automates.append(pickle.load(f))

    for i in f1_scores:
        print(i)
    isos = []
    for i in range(len(automates)):
        automates[i].minimize()
    for i in range(len(automates)):
        for j in range(i+1,len(automates)):
            isos.append(toy_example.is_isomorphic(automates[i],automates[j])[0])
    k = 0
    for i in range(len(automates)):
        for j in range(i+1,len(automates)):
            print(f"{i} et {j} -> {isos[k]}")
            k += 1
    comapre_tab_f1(f1_scores)
    

def dataset_test(N):
    #This test is the dataset one
    #We create One automata then we create N datasets
    #For each of those datasets we create M models
    #The goal is to check if the models between datasets have an influance on the model
    #create a automate and a dataset then we create N mdels and compare them
    automate, final_states, params = get_automate("ml") #el_automate
    print("\n\n[*] Creating dataset...\n")
    methode = "multi-label"
    prefix = "ml"
    model_path = "models"
    data_path = "test"
    dataset_name = f"{data_path}/{prefix}_{methode}"
    len_words_train = 40
    len_words_test = 400
    nb_words = 1000
    states = 100
    f1_scores = []
    automates = []
    # Entrainement
    for n in range(N):
        words = toy_example.generate_words(params, length=len_words_train, nbr=nb_words)
        labels = [accept_stream(word, automate, params["mots"], methode) for word in words]
        toy_example.create_log(words, labels, f"{dataset_name}_train.txt")

        words = toy_example.generate_words(params, length=len_words_test, nbr=nb_words)
        labels = [accept_stream(word, automate, params["mots"], methode) for word in words]
        toy_example.create_log(words, labels, f"{dataset_name}_test.txt")
        num_classes = max([len(word) for word in params["mots"]]) + 1 

        toy_example.train_model(prefix, methode, model_path=model_path, data_path=data_path, mots=params["mots"])
        model_name = toy_example.get_model_name(prefix, methode).split(".")[0] # On enlève l'extension .pth car elle est ajoutée dans load_model (TODO:Fix ça)
        model = load_model(model_name, model_path).to(toy_example.DEVICE)
        X, Y = toy_example.parse_log_file(f"{data_path}/{prefix}_{methode}_test.txt")
        predicted = []
        y_true = []
        for i in range(len(Y)):
            y_test = Y[i]
            y_true.extend(y_test)
            X_t = torch.tensor(X[i]).to(toy_example.DEVICE)
            with torch.no_grad():
                    outputs = model.predict(X_t)  # pred_shape : [(lettres, classes)*nbr_tete]
                    pred = [torch.argmax(op, dim=1).int().detach().cpu().numpy() for op in outputs]
                    pred = [list(item) for item in zip(*pred)] # On regroupe les prédictions de chaque tête pour chaque lettre
                    predicted.extend(pred)

        f1_scores.append(model.give_f1_scores_ml(y_true, predicted))
        model_name = toy_example.get_model_name(prefix, methode).split(".")[0] # On enlève l'extension .pth car elle est ajoutée dans load_model (TODO:Fix ça)
        model = toy_example.load_model(model_name, model_path).to(toy_example.DEVICE)
        alphabet = list(range(len(params["alphabet"])))
        toy_example.build_automate(model_name, states, prefix, methode, alphabet, init_build="pred", path=data_path, model_path=model_path)
        auto_name = toy_example.get_automaton_name(prefix, methode, states)
        with open(f"{data_path}/{auto_name}", "rb") as f:
            automates.append(pickle.load(f))
    for i in f1_scores:
        print(i)
    comapre_tab_f1(f1_scores)
    for i in range(len(automates)):
        automates[i].minimize()
    for i in range(len(automates)):
        for j in range(i+1,len(automates)):
            print(f"model_{i},model_{j} isomorphe:{toy_example.is_isomorphic(automates[i],automates[j])[0]}")
    pass


def train_model_incremantal(prefix, method,start,end,step, num_classes=None, mots=None, data_path="test/", model_path="models/"):
    X,Y = toy_example.parse_log_file(f"{data_path}/{prefix}_{method}_train.txt")
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

    model:TOY_GRU = TOY_GRU(label_method, nbClasses=num_classes, mots=mots, weights=weights).to(toy_example.DEVICE)

    print(f"Hyperparamètres : {model.get_hyperparameters()}")
    res = []
    nb_done = 0
    while nb_done < end:
        if nb_done == 0:
            TOY_GRU.epochs = start
            nb_done += start
        else:
            TOY_GRU.epochs = step
            nb_done += step
        model.train(dataloader)
        X, Y = toy_example.parse_log_file(f"{data_path}/{prefix}_{method}_test.txt")
        predicted = []
        y_true = []
        for i in range(len(Y)):
            y_test = Y[i]
            y_true.extend(y_test)
            X_t = torch.tensor(X[i]).to(toy_example.DEVICE)
            with torch.no_grad():
                    outputs = model.predict(X_t)  # pred_shape : [(lettres, classes)*nbr_tete]
                    pred = [torch.argmax(op, dim=1).int().detach().cpu().numpy() for op in outputs]
                    pred = [list(item) for item in zip(*pred)] # On regroupe les prédictions de chaque tête pour chaque lettre
                    predicted.extend(pred)
        res.append(model.give_f1_scores_ml(y_true, predicted))
        print(f"epoch {nb_done} done")
            

    model_name = toy_example.get_model_name(prefix, method)
    torch.save(model, f"{model_path}/{model_name}")
    return res

def test_encr_epoch(start,end,step):
        #create a automate and a dataset then we create N mdels and compare them
    automate, final_states, params = get_automate("ml") #el_automate
    print("\n\n[*] Creating dataset...\n")
    methode = "multi-label"
    prefix = "ml"
    model_path = "models"
    data_path = "test"
    dataset_name = f"{data_path}/{prefix}_{methode}"
    len_words_train = 40
    len_words_test = 400
    nb_words = 1000
    states = 100

    # Entrainement
    words = toy_example.generate_words(params, length=len_words_train, nbr=nb_words)
    labels = [accept_stream(word, automate, params["mots"], methode) for word in words]
    toy_example.create_log(words, labels, f"{dataset_name}_train.txt")

    words = toy_example.generate_words(params, length=len_words_test, nbr=nb_words)
    labels = [accept_stream(word, automate, params["mots"], methode) for word in words]
    toy_example.create_log(words, labels, f"{dataset_name}_test.txt")
    num_classes = max([len(word) for word in params["mots"]]) + 1 
    
    f1_scores = []
    automates = []
    return train_model_incremantal(prefix, methode,start,end,step, model_path=model_path, data_path=data_path, mots=params["mots"])


if __name__ == "__main__":
    print(test_encr_epoch(1,100,10))