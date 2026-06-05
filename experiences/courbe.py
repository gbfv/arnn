import matplotlib.pyplot as plt
import json
import pandas as pd



def multi_auto(data, states):
    """
    data : dict {rnn : value , auto : [values]}
    """

    colors = ["blue","purple", "brown", "green", "orange", "black", "darkblue", "magenta", "goldenrod"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), facecolor="lightgrey")
    keys = list(data.keys())

    all_values = [x for v in data.values() for x in [v['rnn']] + v['automaton']]
    ymin = min(all_values)
    ymax = max(all_values)


    for i, ax in enumerate(axes.flat):
        if i < len(keys):
            file = keys[i]
            ax.plot(states, data[file]['automaton'], marker='o', color=colors[i % len(colors)])
            ax.plot(states, [data[file]['rnn']]*len(states), linestyle='--', color='red')
            ax.set_ylabel('F1 Score (%)')
            ax.set_title(f"{file}")
            ax.set_ylim(ymin-1, ymax+1)
        else:
            fig.delaxes(ax)

    return plt


def kmu500(save=False):
    auto_f1 = [84.5, 91.2, 92.7, 93.4, 93.1, 93.4, 89.1, 90.0, 90.3, 91.4, 91.7]
    fn = [483, 288, 240, 222, 228, 220, 368, 344, 321, 280, 275]
    fp = [65, 37, 36, 27, 32, 29, 28, 31, 35, 40, 34]
    x = [500, 1000, 2000, 5000, 5500, 6000, 6500, 7000, 8000, 10000, 20000]

    g_f1 = [98.9, 98.6, 97.6, 98.5, 97.8, 98.5, 98.5, 95.1, 94.6, 95.9, 98.2,]
    ker_f1 = [89.6, 92.2, 93.2, 93.0, 92.9, 93.1, 92.5, 92.4, 92.7, 92.5, 92.5]
    libcrypto_f1 = [80.7, 81.4, 81.7, 82.0, 81.5, 82.6, 82.3, 82.0, 81.8, 81.6, 82.3]
    ws2_f1 = [89.1, 90.8, 92.2, 91.6, 92.2, 92.2, 92.0, 91.7, 92.0, 90.7, 91.1]
    ieproxy_f1 = [98.0, 97.9, 97.9, 97.6, 97.4, 96.0, 97.4, 97.4, 96.9, 97.1, 94.4]
    crypt32_f1 = [90.5, 91.8, 92.1, 92.2, 92.3, 92.0, 92.0, 92.0, 92.1, 92.5, 92.3]
    fw_f1 = [94.0, 95.4, 95.4, 95.9, 95.6, 95.9, 95.7, 96.0, 95.8, 95.6, 95.4]
    cmd_f1 = [88.0, 93.5, 92.7, 94.7, 94.0, 95.2, 94.4, 93.7, 93.3, 94.4, 94.7]

    plt.figure(figsize=(14, 12), facecolor="lightgrey")
    plt.suptitle("Automate crée sur KMU.500 avec KMU", fontsize=16)

    plt.subplot(331)
    plt.plot(x, auto_f1, marker='o')
    plt.plot(x, [94.4]*len(x), linestyle='--')
    plt.ylabel('F1 Score (%)')
    plt.title('ntdll.dll')

    plt.subplot(332)
    plt.plot(x, g_f1, marker='o', color='purple')
    plt.plot(x, [97.4]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('gdi32.dll')

    plt.subplot(333)
    plt.plot(x, ker_f1, marker='o', color='brown')
    plt.plot(x, [94.5]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('kerberos.dll')

    plt.subplot(334)
    plt.plot(x, libcrypto_f1, marker='o', color='green')
    plt.plot(x, [85.3]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('libcrypto.dll')

    plt.subplot(335)
    plt.plot(x, ws2_f1, marker='o', color='orange')
    plt.plot(x, [93.5]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('ws2_32.dll')

    plt.subplot(336)
    plt.plot(x, ieproxy_f1, marker='o', color='black')
    plt.plot(x, [97.4]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('ieproxy.dll')

    plt.subplot(337)
    plt.plot(x, crypt32_f1, marker='o', color='darkblue')
    plt.plot(x, [93.3]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('crypt32.dll')

    plt.subplot(338)
    plt.plot(x, fw_f1, marker='o', color='magenta')
    plt.plot(x, [96.2]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('firewallAPI.dll')

    plt.subplot(339)
    plt.plot(x, cmd_f1, marker='o', color='goldenrod')
    plt.plot(x, [94.2]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('cmdext.dll')

    if save:
        plt.savefig('kmu500.png', dpi=300)
    else:
        plt.show()



def fgker500(save=False):
    x = [500, 1000, 2000, 5000, 6000, 6500, 7000, 10000]

    ntdll_f1 = [22.7, 22.1, 15.8, 4.4, 6.6, 11.1, 11.5, 20.8]
    g_f1 = [98.6, 99.5, 99.6, 99.6, 99.7, 99.7, 99.2, 99.8]
    ker_f1 = [90.0, 92.3, 93.8, 95.2, 95.5, 95.6, 95.8, 96.5]
    libcrypto_f1 = [72.2, 77.8, 81.0, 81.8, 80.6, 80.8, 80.8, 80.8]
    ws2_f1 = [89.7, 91.6, 92.5, 93.3, 93.4, 93.2, 93.8, 94.1]
    ieproxy_f1 = [96.8, 98.1, 99.2, 98.8, 98.8, 99.2, 99.1, 99.5]
    crypt32_f1 = [90.4, 91.7, 92.7, 93.1, 93.2, 93.1, 93.3, 93.2]
    fw_f1 = [92.2, 94.4, 96.2, 96.8, 96.9, 97.1, 97.0, 97.2]
    cmd_f1 = [91.4, 94.5, 94.8, 96.3, 96.6, 97.0, 96.8, 94.7]

    plt.figure(figsize=(14, 12), facecolor="lightgrey")
    plt.suptitle("Automate crée sur FGKer.500 avec FGKer", fontsize=16)

    plt.subplot(331)
    plt.plot(x, ntdll_f1, marker='o')
    plt.plot(x, [0.9]*len(x), linestyle='--')
    plt.ylabel('F1 Score (%)')
    plt.title('ntdll.dll')

    plt.subplot(332)
    plt.plot(x, g_f1, marker='o', color='purple')
    plt.plot(x, [99.8]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('gdi32.dll', color='red')

    plt.subplot(333)
    plt.plot(x, ker_f1, marker='o', color='brown')
    plt.plot(x, [97.9]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('kerberos.dll', color='red')

    plt.subplot(334)
    plt.plot(x, libcrypto_f1, marker='o', color='green')
    plt.plot(x, [83.1]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('libcrypto.dll')

    plt.subplot(335)
    plt.plot(x, ws2_f1, marker='o', color='orange')
    plt.plot(x, [94.8]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('ws2_32.dll')

    plt.subplot(336)
    plt.plot(x, ieproxy_f1, marker='o', color='black')
    plt.plot(x, [99.0]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('ieproxy.dll')

    plt.subplot(337)
    plt.plot(x, crypt32_f1, marker='o', color='darkblue')
    plt.plot(x, [94.3]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('crypt32.dll')

    plt.subplot(338)
    plt.plot(x, fw_f1, marker='o', color='magenta')
    plt.plot(x, [98.0]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('firewallAPI.dll', color='red')

    plt.subplot(339)
    plt.plot(x, cmd_f1, marker='o', color='goldenrod')
    plt.plot(x, [97.2]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('cmdext.dll')

    if save:
        plt.savefig('fgker500.png', dpi=300)
    else:
        plt.show()



def gru_kmu500(save=False):
    auto_f1 = [88.5, 85.1, 89.2, 87.8, 86.8, 88.3, 89.8, 86.4, 87.7, 87.0]
    x = [1000, 2000, 5000, 5500, 6000, 6500, 7000, 8000, 10000, 20000]

    g_f1 = [97.3, 88.4, 96.5, 96.7, 97.0, 96.9, 97.4, 94.8, 97.1, 88.5]
    ker_f1 = [86.7, 87.6, 88.9, 88.9, 88.9, 89.0, 88.3, 88.9, 89.6, 89.5]
    libcrypto_f1 = [79.6, 81.0, 82.0, 81.5, 81.8, 82.4, 82.7, 83.2, 82.7, 82.2]
    ws2_f1 = [85.7, 86.3, 87.8, 87.8, 88.9, 88.5, 88.0, 88.8, 88.3, 88.2]
    ieproxy_f1 = [96.9, 91.8, 94.9, 94.6, 94.4, 95.1, 96.2, 96.2, 96.0, 95.7]
    crypt32_f1 = [89.1, 90.1, 90.0, 89.9, 90.1, 90.1, 90.1, 90.3, 90.4, 90.5]
    fw_f1 = [92.6, 91.6, 92.9, 93.2, 93.1, 93.1, 93.5, 93.3, 94.1, 92.5]
    cmd_f1 = [85.8, 87.2, 90.8, 90.7, 88.6, 89.4, 86.9, 89.2, 89.0, 89.3]

    plt.figure(figsize=(14, 12), facecolor="lightgrey")
    plt.suptitle("Automate crée sur GRU_kmu.500 avec KMU", fontsize=16)

    plt.subplot(331)
    plt.plot(x, auto_f1, marker='o')
    plt.plot(x, [85.4]*len(x), linestyle='--')
    plt.ylabel('F1 Score (%)')
    plt.title('ntdll.dll')

    plt.subplot(332)
    plt.plot(x, g_f1, marker='o', color='purple')
    plt.plot(x, [98.5]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('gdi32.dll')

    plt.subplot(333)
    plt.plot(x, ker_f1, marker='o', color='brown')
    plt.plot(x, [91.6]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('kerberos.dll')

    plt.subplot(334)
    plt.plot(x, libcrypto_f1, marker='o', color='green')
    plt.plot(x, [84.3]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('libcrypto.dll')

    plt.subplot(335)
    plt.plot(x, ws2_f1, marker='o', color='orange')
    plt.plot(x, [90.9]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('ws2_32.dll')

    plt.subplot(336)
    plt.plot(x, ieproxy_f1, marker='o', color='black')
    plt.plot(x, [97.6]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('ieproxy.dll')

    plt.subplot(337)
    plt.plot(x, crypt32_f1, marker='o', color='darkblue')
    plt.plot(x, [91.8]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('crypt32.dll')

    plt.subplot(338)
    plt.plot(x, fw_f1, marker='o', color='magenta')
    plt.plot(x, [95.2]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('firewallAPI.dll')

    plt.subplot(339)
    plt.plot(x, cmd_f1, marker='o', color='goldenrod')
    plt.plot(x, [92.0]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('cmdext.dll')

    if save:
        plt.savefig('gru_kmu500.png', dpi=300)
    else:
        plt.show()





def bkmu500(save=False):
    auto_f1 = [74.2, 74.3, 74.2, 72.4, 73.2]
    x = [500, 1000, 2000, 6000, 10000]

    g_f1 = [94.9, 94.8, 94.9, 94.5, 94.8]
    ker_f1 = [88.7, 87.5, 87.9, 87.3, 87.2]
    libcrypto_f1 = [85.0, 83.5, 83.7, 84.1, 83.3]
    ws2_f1 = [86.4, 86.2, 86.2, 85.9, 85.4]
    ieproxy_f1 = [95.3, 94.5, 94.4, 92.5, 93.4]
    crypt32_f1 = [86.9, 85.9, 86.7, 85.7, 86.2]
    fw_f1 = [89.5, 88.7, 89.0, 89.0, 88.4]
    cmd_f1 = [86.8, 86.0, 87.8, 85.2, 83.2]

    plt.figure(figsize=(14, 12), facecolor="lightgrey")
    plt.suptitle("Automate crée sur bkmu.500 avec KMU", fontsize=16)

    plt.subplot(331)
    plt.plot(x, auto_f1, marker='o')
    plt.plot(x, [71.8]*len(x), linestyle='--')
    plt.ylabel('F1 Score (%)')
    plt.title('ntdll.dll')

    plt.subplot(332)
    plt.plot(x, g_f1, marker='o', color='purple')
    plt.plot(x, [95.4]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('gdi32.dll')

    plt.subplot(333)
    plt.plot(x, ker_f1, marker='o', color='brown')
    plt.plot(x, [87.2]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('kerberos.dll')

    plt.subplot(334)
    plt.plot(x, libcrypto_f1, marker='o', color='green')
    plt.plot(x, [84.1]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('libcrypto.dll')

    plt.subplot(335)
    plt.plot(x, ws2_f1, marker='o', color='orange')
    plt.plot(x, [85.4]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('ws2_32.dll')

    plt.subplot(336)
    plt.plot(x, ieproxy_f1, marker='o', color='black')
    plt.plot(x, [93.7]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('ieproxy.dll')

    plt.subplot(337)
    plt.plot(x, crypt32_f1, marker='o', color='darkblue')
    plt.plot(x, [86.0]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('crypt32.dll')

    plt.subplot(338)
    plt.plot(x, fw_f1, marker='o', color='magenta')
    plt.plot(x, [88.8]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('firewallAPI.dll')

    plt.subplot(339)
    plt.plot(x, cmd_f1, marker='o', color='goldenrod')
    plt.plot(x, [84.6]*len(x), linestyle='--', color='red')
    plt.ylabel('F1 Score (%)')
    plt.title('cmdext.dll')

    if save:
        plt.savefig('bkmu500.png', dpi=300)
    else:
        plt.show()


def bCelica500(save=False):
    states = [1000, 2000, 6000, 10000]
    data = {
        "ntdll" : {"rnn" : 86.6, "automaton" : [87.0, 86.8, 86.6, 86.7]},
        "gdi32" : {"rnn" : 98.6, "automaton" : [98.8, 98.6, 98.7, 98.5]},
        "kerberos" : {"rnn" : 98.7, "automaton" : [98.9, 98.8, 98.8, 98.7]},
        "libcrypto" : {"rnn" : 90.1, "automaton" : [90.3, 90.1, 90.0, 90.0]},
        "ws2_32" : {"rnn" : 98.5, "automaton" : [98.7, 98.4, 98.4, 98.5]},
        "ieproxy" : {"rnn" : 98.6, "automaton" : [98.6, 98.6, 98.6, 98.6]},
        "crypt32" : {"rnn" : 98.8, "automaton" : [99.0, 99.0, 98.8, 98.8]},
        "firewallAPI" : {"rnn" : 98.7, "automaton" : [99.0, 98.7, 98.8, 98.7]},
        "cmdext" : {"rnn" : 98.4, "automaton" : [98.4, 97.9, 98.2, 97.9]}
    }
    plt = multi_auto(data, states)
    plt.suptitle("Automate bCelica.500", fontsize=16)
    
    if save:
        plt.savefig('bCelica500.png', dpi=300)
    else:
        plt.show()



def BWkmu500(save=False):
    states = [500, 1000, 2000, 6000, 6500, 10000]
    data ={
        "ntdll.log": {"rnn": 85.9, "automaton": [87.5, 87.6, 86.7, 87.2, 86.6, 86.9]},
        "gdi32.log": {"rnn": 97.6, "automaton": [98.3, 98.3, 96.3, 97.7, 98.0, 97.9]}, 
        "kerberos.log": {"rnn": 94.6, "automaton": [98.7, 95.2, 95.8, 95.0, 95.6, 95.3]}, 
        "libcrypto.log": {"rnn": 89.7, "automaton": [94.7, 91.9, 93.3, 88.5, 88.7, 87.0]}, 
        "ws2_32.log": {"rnn": 93.3, "automaton": [98.7, 95.2, 94.8, 94.0, 94.3, 94.2]}, 
        "ieproxy.log": {"rnn": 96.1, "automaton": [98.1, 95.2, 97.9, 95.4, 97.7, 95.4]}, 
        "crypt32.log": {"rnn": 95.5, "automaton": [99.1, 95.2, 94.7, 95.5, 95.9, 95.4]}, 
        "firewallAPI.log": {"rnn": 93.7, "automaton": [98.9, 91.0, 93.9, 92.3, 93.2, 92.4]}, 
        "cmdext.log": {"rnn": 94.3, "automaton": [97.9, 95.8, 97.4, 95.2, 97.2, 94.7]}
    }
    plt = multi_auto(data, states)
    plt.suptitle("Automate BWkmu.500", fontsize=16)
    
    if save:
        plt.savefig('BWkmu500.png', dpi=300)
    else:
        plt.show()



def bgruCelica(save=False):
    states = [500, 1000, 2000, 6000, 6500, 10000]
    data = {
        "kerberos.log": {"rnn": 99.7, "automaton": [98.6, 99.2, 99.2, 99.3, 99.3, 99.5]},
        "ieproxy.log": {"rnn": 99.8, "automaton": [98.5, 98.8, 99.2, 99.2, 99.7, 99.5]}, 
        "crypt32.log": {"rnn": 99.7, "automaton": [98.7, 99.3, 99.1, 99.5, 99.5, 99.6]}, 
        "clp64.log": {"rnn": 90.8, "automaton": [88.0, 89.8, 90.3, 90.5, 92.1, 91.9]}, 
        "energy.log": {"rnn": 99.8, "automaton": [98.9, 99.3, 99.4, 99.6, 99.7, 99.6]}, 
        "basesrv.log": {"rnn": 99.1, "automaton": [97.1, 97.9, 98.5, 98.2, 98.5, 98.8]},
        "ntdll.log": {"rnn": 100.0, "automaton": [97.4, 97.6, 98.1, 98.7, 98.7, 99.1]}, 
        "libcrypto.log": {"rnn": 99.8, "automaton": [97.8, 98.5, 98.5, 99.0, 99.0, 99.2]}, 
        "msvcr100.log": {"rnn": 99.8, "automaton": [99.1, 99.0, 99.2, 99.4, 99.4, 99.4]}
    }
    plt = multi_auto(data, states)
    plt.suptitle("Automate bgruCelica.500", fontsize=16)

    if save:
        plt.savefig('bgruCelica500.png', dpi=300)
    else:
        plt.show()


def bgruCelicaV2(save=False):
    states = [500, 1000, 2000, 6000, 6500, 10000]
    data = {
        "kerberos.log": {"rnn": 99.9, "automaton": [99.1, 99.2, 99.4, 99.6, 99.5, 99.6]},
        "ieproxy.log": {"rnn": 99.8, "automaton": [99.5, 99.7, 99.8, 99.5, 99.7, 99.7]},
        "crypt32.log": {"rnn": 99.8, "automaton": [99.3, 99.2, 99.4, 99.5, 99.4, 99.6]},
        "clp64.log": {"rnn": 89.9, "automaton": [92.0, 90.4, 92.2, 88.2, 88.2, 90.6]},
        "energy.log": {"rnn": 99.7, "automaton": [98.8, 99.1, 99.3, 99.6, 99.5, 99.4]},
        "basesrv.log": {"rnn": 99.7, "automaton": [98.5, 98.2, 99.7, 99.4, 99.1, 99.1]}
    }
    plt = multi_auto(data, states)
    plt.suptitle("Automate bgruCelicaV2.500", fontsize=16)

    if save:
        plt.savefig('bgruCelicaV2_500.png', dpi=300)
    else:
        plt.show()



def loss(file, multi=True, save=False):
    loss = []
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']
    with open(file, "r") as f:
        for line in f:
            l = line.split(";")[:-1]
            loss.append([float(x) for x in l])

    min_y = min([min(l) for l in loss])
    max_y = max([max(l) for l in loss])

    name = file.split(".")[0]

    if multi:
        with open(f"{name}_perfs.json", "r") as f:
            perfs = json.load(f)

        max_perf = max(val for sub1 in perfs for sub2 in sub1 for val in sub2)
        min_perf = min(val for sub1 in perfs for sub2 in sub1 for val in sub2)

        fig, axes = plt.subplots(5, 4, figsize=(20, 14), facecolor="lightgrey")
        fig.suptitle(f"Loss curves from {name.split('/')[-1]}", fontsize=16)

        for i, ax in enumerate(axes.flat):
            if i < len(loss):
                ax.plot(range(1, len(loss[i]) + 1), loss[i], color=colors[i % len(colors)])
                ax.set_yscale('log')
                ax.set_ylim(min_y*0.9, max_y*1.1)
                ax.set_ylabel('Loss')
                ax.set_xlabel('Epochs')


                ax2 = ax.twinx()
                #plot prec, recall, f1 on ax2
                ax2.plot(range(25, 501, 25), perfs[i][0], linestyle='--', color='sienna',  marker='.', label='Precision')
                ax2.plot(range(25, 501, 25), perfs[i][1], linestyle='--', color='darkred', marker='.', label='Recall')
                ax2.plot(range(25, 501, 25), perfs[i][2], linestyle='--', color='orchid',  marker='.', label='F1 Score')
                ax2.set_ylim(min_perf*0.9, max_perf*1.1)
            else:
                fig.delaxes(ax)
    else:
        plt.figure(figsize=(10, 6), facecolor="lightgrey")
        for i in range(len(loss)):
            plt.plot(range(1, len(loss[i]) + 1), loss[i], label=f"Run {i+1}", color=colors[i % len(colors)])
        plt.yscale('log')
        plt.ylim(min_y*0.9, max_y*1.1)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.title(f"Loss curves from {name.split('/')[-1]}")
    
    if save:
        savename = name+".png" if not multi else name+"_multi.png"
        plt.savefig(savename, dpi=300)
    else:
        plt.show()


#Ce qu'il se passe au dessus est un peu daté


def fsm_experiment(save=False):
    states = [500, 1000, 2000, 6000, 6500, 10000]
    data = {
        "ntdll" : {"rnn" : 86.6, "automaton" : [87.0, 86.8, 86.6, 86.7]},
        "gdi32" : {"rnn" : 98.6, "automaton" : [98.8, 98.6, 98.7, 98.5]},
        "kerberos" : {"rnn" : 98.7, "automaton" : [98.9, 98.8, 98.8, 98.7]},
        "libcrypto" : {"rnn" : 90.1, "automaton" : [90.3, 90.1, 90.0, 90.0]},
        "ws2_32" : {"rnn" : 98.5, "automaton" : [98.7, 98.4, 98.4, 98.5]},
        "ieproxy" : {"rnn" : 98.6, "automaton" : [98.6, 98.6, 98.6, 98.6]},
        "crypt32" : {"rnn" : 98.8, "automaton" : [99.0, 99.0, 98.8, 98.8]},
        "firewallAPI" : {"rnn" : 98.7, "automaton" : [99.0, 98.7, 98.8, 98.7]},
        "cmdext" : {"rnn" : 98.4, "automaton" : [98.4, 97.9, 98.2, 97.9]}
    }
    plt = multi_auto(data, states)
    plt.suptitle("Automate bCelica FSM Experiment", fontsize=16)
    
    if save:
        plt.savefig('fsm_experiment.png', dpi=300)
    else:
        plt.show()



if __name__ == "__main__":
    loss("loss/KMU_FS_sqrt.csv", multi=False, save=True)