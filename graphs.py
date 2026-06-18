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
            org = float(STs[1])
            Sts = [float(x) / org for x in STs]
            F1s = [float(getattr(x,"F1")) for x in data]
            plt.plot(Sts,F1s)
    plt.xlabel("Nombre de k clusters (en pourcentage du nombre d'état de l'automate originel)")
    plt.ylabel("F1 score")
    plt.title("F1 scores des automates en fonction du nomre de clusters crées")
    plt.yscale("log")
    plt.axis((-1,20,-1,1.5))
    plt.show()

if __name__ == "__main__":
    states_graph("benchmark_data/default/")