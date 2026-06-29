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

import os
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


def simple_graph(data,Field1:str,Field2:str,LabelX:str,LabelY:str):
    plt.cla()
    Y = [float(getattr(x,Field1)) for x in data]
    X = [float(getattr(x,Field2)) for x in data]
    plt.plot(Y,X)
    plt.axis((min(Y),max(Y),0,max(max(X),5)))
    plt.show()


def states_graph(filename):
    all_f:List[str] = os.listdir(filename)
    plt.cla()
    for f in all_f:
        if f.startswith("nb_state_id"):
            data = load_log_file_for_benchmark(filename + f)
            STs = [int(getattr(x,"STATES")) for x in data]
            org = float(STs[3])
            Sts = [float(x) / org for x in STs]
            F1s = [float(getattr(x,"F1")) for x in data]
            plt.plot(STs,F1s,label=f"Auto:{f.split("nb_state_id")[1]}")
    plt.xlabel("Nombre de k clusters")
    plt.ylabel("F1 score")
    plt.title("F1 scores des automates en fonction du nombre de clusters crées")
    plt.legend()
    plt.show()


def perf_graph(filename):
    pass
    plt.cla()
    data = load_log_file_for_benchmark(filename)
    Z = [int(getattr(x,"NB_WORDS")) for x in data]
    F = [float(getattr(x,"F1_M")) for x in data]
    FA = [float(getattr(x,"F1_A")) for x in data]
    plt.scatter(Z,F,label="F1 score modele")
    plt.scatter(Z,FA,label="F1 score automate")
    plt.legend()
    plt.xlabel("Nombre de mots dans le motif")
    plt.ylabel("Score F1")
    plt.title("Score F1 en fonction du nombre de mots dans l'automate")
    plt.show()



def weight_graph(filename):
    plt.cla()
    data = load_log_file_for_benchmark(filename)
    ids = []
    for d in data:
        id_tmp = getattr(d,"ID")
        if id_tmp not in ids:
            ids.append(id_tmp)
    for_graph = [x for x in range(len(ids))]
    f1_0 = [float(getattr(x,"F1_M"))for x in data if int(getattr(x,"WEIGHTS")) == 0 ]
    f1_1 = [float(getattr(x,"F1_M"))for x in data if int(getattr(x,"WEIGHTS")) == 1 ]
    f1_2 = [float(getattr(x,"F1_M"))for x in data if int(getattr(x,"WEIGHTS")) == 2 ]
    f1_3 = [float(getattr(x,"F1_M"))for x in data if int(getattr(x,"WEIGHTS")) == 3 ]
    f1_4 = [float(getattr(x,"F1_M"))for x in data if int(getattr(x,"WEIGHTS")) == 4 ]
    f1_5 = [float(getattr(x,"F1_M"))for x in data if int(getattr(x,"WEIGHTS")) == 5 ]
    f1_6 = [float(getattr(x,"F1_M"))for x in data if int(getattr(x,"WEIGHTS")) == 6 ]
    f1_7 = [float(getattr(x,"F1_M"))for x in data if int(getattr(x,"WEIGHTS")) == 7 ]
    F1s = [f1_0,f1_1,f1_2,f1_3,f1_4,f1_5,f1_6,f1_7]
    colors = ['b','g','r','c','m','y',"k","navy"]
    isos = [0] * 8
    for i in range(len(isos)):
        isos[i] = len([1 for x in data if (int(getattr(x,"WEIGHTS")) == i and getattr(x,"ISO") == "True") ])
    for f1_score_i in range(len(F1s)):
        plt.hlines(np.mean(F1s[f1_score_i]),0,100,colors=colors[f1_score_i],label=f"moyenne weight {f1_score_i}, value:{np.mean(F1s[f1_score_i]):1.4}")
    plt.scatter(for_graph,f1_0,c=colors[0],label="Weight 0")
    plt.scatter(for_graph,f1_1,c=colors[1],label="Weight 1")
    plt.scatter(for_graph,f1_2,c=colors[2],label="Weight 2")
    plt.scatter(for_graph,f1_3,c=colors[3],label="Weight 3")
    plt.scatter(for_graph,f1_4,c=colors[4],label="Weight 4")
    plt.scatter(for_graph,f1_5,c=colors[5],label="Weight 5")
    plt.scatter(for_graph,f1_6,c=colors[6],label="Weight 6")
    plt.scatter(for_graph,f1_7,c=colors[7],label="Weight 7")
    plt.xticks([])
    plt.ylabel("F1 scrore")
    plt.title("F1 score de modèles par différentes fonctions de poids")
    plt.legend()
    plt.show()


def nuage_graph(filename):
    plt.cla()
    data = load_log_file_for_benchmark(filename)
    all_autos = list(set([getattr(x,"ID") for x in data]))
    for id in all_autos:
        nb_words_data = [int(getattr(x,"NB_WORDS")) for x in data if getattr(x,"ID") == id]
        F1_data = [float(getattr(x,"F1")) for x in data if getattr(x,"ID") == id]
        plt.plot(nb_words_data,F1_data,alpha=0.1)

    all_words = list(set([int(getattr(x,"NB_WORDS")) for x in data]))
    all_words.sort()
    moy = []
    for w in all_words:
        moy.append(np.mean([float(getattr(x,"F1")) for x in data if int(getattr(x,"NB_WORDS")) == w]))
    print(moy)
    plt.plot(all_words,moy)
    plt.plot()
    
    plt.show()

def stress_test_graph(filename):
    plt.cla()
    data = load_log_file_for_benchmark(filename)
    Z = [int(getattr(x,"NB_WORDS")) * int(getattr(x,"LEN_WORDS")) for x in data]
    I = [int(getattr(x,"ID")) for x in data]
    M = [float(getattr(x,"F1_M")) for x in data]
    A = [float(getattr(x,"F1_A")) for x in data]
    plt.scatter(Z,M,label="Modele")
    plt.scatter(Z,A,label="Automate")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    stress_test_graph("benchmark_data/stress_test_exp2/logs/default/stress.log")