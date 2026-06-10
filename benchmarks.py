import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from typing import List
import numpy as np

#Tout les fichiers logs vont être de type VAR1:VAL1, VAR2:VAL2
def load_log_file(filename):
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
    r = load_log_file(filename)
    epochs = [int(getattr(x,"EPOCH")) for x in r]
    loss = [float(getattr(x,"LOSS"))for x in r]
    one_percent = loss[-1] * 1.01
    ten_percent = loss[-1] * 1.1
    fifty_percent = loss[-1] * 1.5
    one_percent_epoch_index = get_index_closer(loss,one_percent)
    ten_percent_epoch_index = get_index_closer(loss,ten_percent)
    fifty_percent_epoch_index = get_index_closer(loss,fifty_percent)

    plt.plot(epochs,loss)
    plt.axvline(one_percent_epoch_index,color="red",label="99% value")
    plt.axvline(ten_percent_epoch_index,color="orange",label="90% value")
    plt.axvline(fifty_percent_epoch_index,color="yellow",label="50% value")
    plt.xlabel("EPOCH")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def simple_graph(data,Field1:str,Field2:str,LabelX:str,LabelY:str):
    plt.cla()
    Y = [float(getattr(x,Field1)) for x in data]
    X = [float(getattr(x,Field2)) for x in data]
    plt.plot(Y,X)
    plt.axis((min(Y),max(Y),0,max(max(X),5)))
    plt.show()

    pass
if __name__ == "__main__":
    epoch_time_graph("logs/default/epoch_loss.log")