import sqlite3
import pickle
import torch

from utils import get_device

import os

DB = None
DEVICE = get_device()

def get_cursor():
    global DB
    if DB is None:
        DB = sqlite3.connect("DATABASE.db")
    return DB.cursor()


def setup_db():
    global DB
    curr = get_cursor()
    curr.execute("""CREATE TABLE IF NOT EXISTS Languages(
        id INTEGER  PRIMARY KEY AUTOINCREMENT,
        name STRING,
        size_sigma INTEGER,
        nb_words INTEGER,
        len_words INTEGER,
        size_auto INTEGR,
        reset BOOL,
        data BLOB
    );""")
    curr.execute("""CREATE TABLE IF NOT EXISTS Datasets(
        id INTEGER  PRIMARY KEY AUTOINCREMENT,
        lang INTEGER,
        name STRING,
        label STRING,
        nb_train INTEGER,
        len_train INTEGER,

        nb_val INTEGER,
        len_val INTEGER,

        nb_test INTEGER,
        len_test INTEGER,
        data_train BLOB,
        data_val BLOB,
        data_test BLOB
    );""")
    curr.execute("""CREATE TABLE IF NOT EXISTS Models(
        id INTEGER  PRIMARY KEY AUTOINCREMENT,
        lang INTEGER,
        dataset INTEGER,
        name STRING,
        epochs INTEGER,
        weights INTEGER,
        F1_mean FLOAT,
        data BLOB,
        data_specs_only BLOB
    );""")
    curr.execute("""CREATE TABLE IF NOT EXISTS Autos(
        id INTEGER  PRIMARY KEY AUTOINCREMENT,
        lang INTEGER,
        dataset INTEGER,
        model INTEGER,
        name STRING,
        nb_explore INTEGER,
        nb_clusters INTEGER,
        nb_etats INTEGER,
        F1_mean FLOAT,
        data BLOB
    );""")

    DB.commit()
    curr.close()


def add_entry_lang(
    name,
    size_sigma,
    nb_words,
    len_words,
    size_auto,
    reset,
    data):
    global DB
    curr = get_cursor()
    curr.execute("INSERT INTO Languages (name,size_sigma,nb_words,len_words,size_auto,reset,data) VALUES (?,?,?,?,?,?,?);",(name,size_sigma,nb_words,len_words,size_auto,reset,data))
    DB.commit()

def add_entry_dataset(
    lang,
    name,
    label,
    nb_train,
    len_train,
    nb_val,
    len_val,
    nb_test,
    len_test,
    data_train,
    data_val,
    data_test):
    global DB
    curr = get_cursor()
    curr.execute("INSERT INTO Datasets (lang,name,label,nb_train,len_train,nb_val,len_val,nb_test,len_test,data_train,data_val,data_test ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?);",(lang,name,label,nb_train,len_train,nb_val,len_val,nb_test,len_test,data_train,data_val,data_test))
    DB.commit()
    pass

def add_entry_models(
        lang ,
        dataset ,
        name ,
        epochs ,
        weights ,
        data,
        data_specs):
    global DB
    curr = get_cursor()
    curr.execute("INSERT INTO Models (lang,dataset,name,epochs,weights,data,data_specs_only) VALUES (?,?,?,?,?,?,?);",(lang,dataset,name,epochs,weights,data,data_specs))
    DB.commit()

def add_entry_auto(
    lang ,
    dataset ,
    model ,
    name ,
    nb_explore ,
    nb_clusters ,
    nb_etats ,
    data ):
    global DB
    curr = get_cursor()
    curr.execute("INSERT INTO Autos (lang ,dataset ,model ,name ,nb_explore ,nb_clusters ,nb_etats ,data ) VALUES (?,?,?,?,?,?,?,?);",(lang ,dataset ,model ,name ,nb_explore ,nb_clusters ,nb_etats ,data))
    DB.commit()


def get_data(command):
    global DB
    curr = get_cursor()
    curr.execute(command)
    return curr.fetchall()

def load_language(id:int):
    curr = get_cursor()
    curr.execute("SELECT data FROM Languages WHERE id = ?;",(id,))
    data = curr.fetchall()
    if len(data) != 1:
        print("Problème avec la récupération du Language")
        return None,None,None
    os.makedirs("db_files",exist_ok=True)
    data = data[0][0]
    data = pickle.loads(data)
    automate = data["automate"]
    final_states = data["final_states"]
    params = data["params"]
    print("ID AUTO UTILISE:", id)
    return automate, final_states, params
    
def give_raw_bytes_language(auto,finals,infos):
    pak = {"automate": auto,"final_states": finals, "params": infos}
    return pickle.dumps(pak)


def load_datasets_to_file(id:int):
    curr = get_cursor()
    curr.execute("SELECT data_train,data_val,data_test FROM Datasets WHERE id = ?;",(id,))
    data = curr.fetchall()
    if len(data) != 1:
        print("Problème avec la récupération du Dataset")
        return
    data = data[0]
    open("test/db_files_train.txt","w").write(data[0])
    open("test/db_files_val.txt","w").write(data[1])
    open("test/db_files_test.txt","w").write(data[2])
    return

def capture_datasets():
    d1 = open("test/db_files_train.txt","r").read()
    d2 = open("test/db_files_val.txt","r").read()
    d3 = open("test/db_files_test.txt","r").read()
    return d1,d2,d3
    

def give_raw_bytes_model(M):
    torch.save(M,"test/tmp_model")
    with open("test/tmp_model","rb") as f:
        return f.read()

def give_raw_bytes_model_specs(M):
    torch.save(M.state_dict(),"test/tmp_model_specs")
    with open("test/tmp_model_specs","rb") as f:
        return f.read()

def load_model(id:int):
    curr = get_cursor()
    curr.execute("SELECT data FROM Models WHERE id = ?;",(id,))
    data = curr.fetchall()
    if len(data) != 1:
        print("Problème avec la récupération du Model")
        return
    data = data[0][0]
    f = open("test/tmp_model","wb")
    f.write(data)
    f.close()
    return torch.load("test/tmp_model", weights_only=False).to(DEVICE)

def give_raw_bytes_auto(A):
    return pickle.dumps(A)

def load_auto_from_db(id:int):
    curr = get_cursor()
    curr.execute("SELECT data FROM Autos WHERE id = ?;",(id,))
    data = curr.fetchall()
    if len(data) != 1:
        print("Problème avec la récupération du Model")
        return
    data = data[0][0]
    return pickle.loads(data)



def update_model_score(id_model:int,score:float):
    global DB
    curr = get_cursor()
    curr.execute("UPDATE Models SET F1_mean = ? WHERE id = ?;",(score,id_model))
    DB.commit()

def update_auto_score(id_auto:int,score:float):
    global DB
    curr = get_cursor()
    curr.execute("UPDATE Autos SET F1_mean = ? WHERE id = ?;",(score,id_auto))
    DB.commit()
    

def get_ids_dataset(lang,label,nb_train,len_train,nb_val,len_val,nb_test,len_test):
    curr = get_cursor()
    curr.execute("""SELECT id FROM Datasets WHERE 
        lang = ? AND
        label = ? AND
        nb_train = ? AND
        len_train = ? AND

        nb_val = ? AND
        len_val = ? AND

        nb_test = ? AND
        len_test = ?
        """,(lang,label,nb_train,len_train,nb_val,len_val,nb_test,len_test))
    data = curr.fetchall()
    return data

def get_ids_models(lang,dataset,epochs,weights):
    curr = get_cursor()
    curr.execute("""SELECT id FROM Models WHERE 
        lang = ? AND
        dataset = ? AND
        epochs = ? AND
        weights = ?
        """,(lang,dataset,epochs,weights))
    data = curr.fetchall()
    return data



def get_ids_autos(lang,dataset,model,nb_clusters):
    curr = get_cursor()
    curr.execute("""SELECT id FROM Autos WHERE 
        lang = ? AND
        dataset = ? AND
        model = ? AND
        nb_clusters = ?
        """,(lang,dataset,model,nb_clusters))
    data = curr.fetchall()
    return data



if __name__ == "__main__":
    setup_db()
    print(get_data("SELECT * FROM Models;"))
