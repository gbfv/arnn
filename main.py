import torch
import pathlib
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
import json


from rnn import ARNN
from rnn_batch import BRNN
from gru import TGRU
from gru_batch import BGRU
from automaton import Automaton
from logDataset import LogDataset, prepared_data, testing_data




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

def load_model(name):
    model_path = f"models/{name}.pth"
    # Loading model
    if pathlib.Path(model_path).is_file():
        model = torch.load(model_path, weights_only=False).to(DEVICE)
    else:
        raise FileNotFoundError(f"Model file {model_path} not found. Please train the model before testing.")
    return model


def rnn_build(name, training_files, whitelist=True, save=True):
    model_path = f"models/{name}.pth"
    # Loading model
    if pathlib.Path(model_path).is_file():
        return torch.load(model_path, weights_only=False).to(DEVICE)
    
    # Building model
    dataset = LogDataset(files=training_files, whitelist=whitelist)
    dataset.prep_concat(training=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True, collate_fn=pad_batch)
    print(f"Fichier chargé pour l'entraînement : {dataset.used_files}")

    model = ARNN().to(DEVICE)
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




def test_model(model_name, files, whitelist=True, DA=-2, DB=20, DELTA=6): #TODO: utiliser LogDataset pour coller à l'entrainement
    model = load_model(model_name)
    
    # Preparing test data
    X_test, y_test = [], []
    for file in files:
        X, y = prepared_data(f"data/{file}", DA=DA, DB=DB, DELTA=DELTA)
        X_test.extend(X)
        y_test.extend(y)
    print(f"{len(X_test)} items in testing dataset")
    
    X_t, y_t = torch.tensor(X_test).to(DEVICE), torch.tensor(y_test)

    with torch.no_grad():
        predicted = (model.predict(X_t).squeeze() > 0.8).int().detach().cpu().numpy()  # Seuil à 0.5 pour obtenir des 0 et des 1
        print(f'Diff : {sum(abs(y_t - predicted))}')
        print(f"Scores : {model.scores(y_t, predicted)}")



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
    

### Automaton functions ###


def automaton_build(model_name, files, states=1000):
    model = load_model(model_name)

    auto_path = f"automate/{model_name}_{states}.pkl"
    if pathlib.Path(auto_path).is_file():
        with open(auto_path, "rb") as f:
            return pickle.load(f)
   
    #Preparing data
    X_test, y_test = [], []
    for file in files:
        X, y = prepared_data(f"data/{file}", DA=-2, DB=20, DELTA=6)
        X_test.extend(X)
        y_test.extend(y)
    print(f"{len(X_test)} items in testing dataset")
    
    X_t, y_t = torch.tensor(X_test).to(DEVICE), torch.tensor(y_test)

    print("Construction de l'automate...\n")
    A = Automaton(model, list(range(256)), states, X_t, y_t)
    A.emonde()
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



def test_automaton(automaton_name, files, model):
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
        X, y = prepared_data(f"data/{file}", DA=-2, DB=20, DELTA=6)
        X_test.extend(X)
        y_test.extend(y)
    #print(f"{len(X_test)} items in testing dataset")
    
    X_t, y_t = torch.tensor(X_test).to(DEVICE), torch.tensor(y_test)

    predicted = A.predict(X_t)
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


def stats_states(auto_path, files, whitelist=True):
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






##### Main #####


if __name__ == "__main__":
    """ build = ["kernel32.log", "msvcr100.log", "user32.log", "ntdll.log", "libcrypto.log", "firewallAPI.log", "ws2_32.log", "signdrv.log", "cmdext.log", "gdi32.log"]
    test = ["kerberos.log","ieproxy.log","crypt32.log","clp64.log","energy.log","basesrv.log"]
    
    automaton = "bgruCelicaV2.500_"
    states = [500, 1000, 2000, 6000, 6500, 10000]

    big_run(build, test, "bgru_CelicaV2.500", automaton, states) """


    """ stats = stats_states("automate/bgru_CelicaV2.500_1000.pkl", [], whitelist=False)
    
    stats = sorted(stats.items(), key=lambda x: x[1]["frequence"], reverse=True)
    for s, v in stats[:20]:
        print(f"État {s:<3} :  fréquence {v['frequence']*100:>5.1f} | count {v['count']:>5} | pos_1 : {v['1']:>5} | pos_2 : {v['2']:>5} | pos_3 : {v['3']:>5}") """


    """ with open("automate/bgru_CelicaV7.500_1000.pkl", "rb") as f:
        A = pickle.load(f)

    X, y = testing_data("data/kerberos.log", DELTA=0)
    predicted = A.predict_flow(torch.tensor(X), y, window_size=48)

    print(f"nombre de fonctions dans y : {sum(y)} / {len(y)}")
    print(f"nombre de fonctions dans predicted : {sum(predicted)} / {len(predicted)}")
    print(f'\nAutomaton  : {predicted} / True : {y[:20]}')
    print(f'Diff : {sum(abs(y - predicted))}')
    
    model = load_model("bgru_CelicaV7.500")
    prec, recall, f1 = model.scores(y, predicted)
    print(f"Scores : Precision: {prec}, Recall: {recall}, F1-score: {f1}") """



    """ model = load_model("bgru_CelicaV5.500")
    X, y = testing_data("data/kernel32.log", DELTA=0)
    predicted = model.predict_flow(torch.tensor(X), y, window_size=22)

    print(f"nombre de fonctions dans y : {sum(y)} / {len(y)}")
    print(f"nombre de fonctions dans predicted : {sum(predicted)} / {len(predicted)}")
    print(f'\nAutomaton  : {predicted} / True : {y[:20]}')
    print(f'Diff : {sum(abs(y - predicted))}')
    
    model = load_model("bgru_CelicaV5.500")
    prec, recall, f1 = model.scores(y, predicted)
    print(f"Scores : Precision: {prec}, Recall: {recall}, F1-score: {f1}") """
