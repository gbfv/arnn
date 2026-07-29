import sqlite3
import pickle
import torch
import logging

from utils import get_device

import os

DB = None
DEVICE = get_device()

def get_cursor():
    """
    Donne le curseur de la db
    """
    global DB
    if DB is None:
        DB = sqlite3.connect("DATABASE.db")
    return DB.cursor()


def setup_db():
    """
    Met en place la DB (NE DOIT ETRE PAS ETRE APPELEE A CHAQUE FOIS)
    """
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
        embedding_dim INTEGER,
        hidden_dim INTEGER,
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
    """
    Ajoute une entrée dans la table Languages
    """
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
    """
    Ajoute une entrée dans la table Datasets

    Retourne l'id de la ligne ajoutée
    """
    curr = get_cursor()
    curr.execute("INSERT INTO Datasets (lang,name,label,nb_train,len_train,nb_val,len_val,nb_test,len_test,data_train,data_val,data_test ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?) RETURNING id;",(lang,name,label,nb_train,len_train,nb_val,len_val,nb_test,len_test,data_train,data_val,data_test))
    res = curr.fetchone()[0]
    DB.commit()
    return res

def add_entry_models(
        lang ,
        dataset ,
        name ,
        epochs ,
        weights ,
        embedding_dim,
        hidden_dim,
        data,
        data_specs):
    """
    Ajoute une entrée dans la table Models (sans le F1 score)

    Retourne l'id de la ligne ajoutée
    """
    curr = get_cursor()
    curr.execute("INSERT INTO Models (lang,dataset,name,epochs,weights,embedding_dim,hidden_dim,data,data_specs_only) VALUES (?,?,?,?,?,?,?,?,?) RETURNING id;",(lang,dataset,name,epochs,weights,embedding_dim,hidden_dim,data,data_specs))
    res = curr.fetchone()[0]
    DB.commit()
    return res

def add_entry_auto(
    lang ,
    dataset ,
    model ,
    name ,
    nb_clusters ,
    nb_etats ,
    data ):
    """
    Ajoute une entrée dans la table Autos (sans le F1 score)

    Retourne l'id de la ligne ajoutée
    """
    curr = get_cursor()
    curr.execute("INSERT INTO Autos (lang ,dataset ,model ,name  ,nb_clusters ,nb_etats ,data ) VALUES (?,?,?,?,?,?,?) RETURNING id;",(lang ,dataset ,model ,name ,nb_clusters ,nb_etats ,data))
    res = curr.fetchone()[0]
    DB.commit()
    return res



def load_language(id:int):
    """
    Retourne l'automate du language associé à id depuis la database
    """
    curr = get_cursor()
    curr.execute("SELECT data FROM Languages WHERE id = ?;",(id,))
    data = curr.fetchall()
    if len(data) != 1:
        logging.error("Problème avec la récupération du Language")
        return None,None,None
    data = data[0][0]
    data = pickle.loads(data)
    automate = data["automate"]
    final_states = data["final_states"]
    params = data["params"]
    logging.info("ID AUTO UTILISE: %d", id)
    return automate, final_states, params
    
def give_raw_bytes_language(auto,finals,infos):
    """
    Transforme un Language en blob de donnés
    """
    pak = {"automate": auto,"final_states": finals, "params": infos}
    return pickle.dumps(pak)


def load_datasets_to_file(id:int,dataset_name:str):
    """
    Charge les datasets de id <id> dans test/db_files* depuis la database
    """
    curr = get_cursor()
    curr.execute("SELECT data_train,data_val,data_test FROM Datasets WHERE id = ?;",(id,))
    data = curr.fetchall()
    if len(data) != 1:
        logging.error("Problème avec la récupération du Dataset")
        return
    data = data[0]
    open(f"{dataset_name}_train.txt","w").write(data[0])
    open(f"{dataset_name}_val.txt","w").write(data[1])
    open(f"{dataset_name}_test.txt","w").write(data[2])
    return

def capture_datasets(dataset_name:str):
    """
    Transforme les datasets nommés test/db_files_* en blob de données
    """
    d1 = open(f"{dataset_name}_train.txt","r").read()
    d2 = open(f"{dataset_name}_val.txt","r").read()
    d3 = open(f"{dataset_name}_test.txt","r").read()
    return d1,d2,d3
    

def give_raw_bytes_model(M,random_key):
    """
    Transforme un modèle en blob de données (/!\ Possiblement non-portable, utiliser give_raw_bytes_model_specs si problèmes de compatibilité)
    """
    torch.save(M,f"test/tmp_model{random_key}")
    with open(f"test/tmp_model{random_key}","rb") as f:
        return f.read()

def give_raw_bytes_model_specs(M,random_key):
    """
    Transforme les paramètres d'un modele en blob de données 
    """
    torch.save(M.state_dict(),f"test/tmp_model_specs{random_key}")
    with open(f"test/tmp_model_specs{random_key}","rb") as f:
        return f.read()

def load_model(id:int,random_key):
    """
    Charge le modèle <id> depuis la database
    """
    curr = get_cursor()
    curr.execute("SELECT data FROM Models WHERE id = ?;",(id,))
    data = curr.fetchall()
    if len(data) != 1:
        logging.error("Problème avec la récupération du Model")
        return
    data = data[0][0]
    f = open(f"test/tmp_model{random_key}","wb")
    f.write(data)
    f.close()
    return torch.load(f"test/tmp_model{random_key}", weights_only=False).to(DEVICE)

def give_raw_bytes_auto(A):
    """
    Transforme un automate en blob de données
    """
    return pickle.dumps(A)

def load_auto_from_db(id:int):
    """
    Charge l'automate <id> depuis la database
    """
    curr = get_cursor()
    curr.execute("SELECT data FROM Autos WHERE id = ?;",(id,))
    data = curr.fetchall()
    if len(data) != 1:
        logging.error("Problème avec la récupération du Model")
        return
    data = data[0][0]
    return pickle.loads(data)



def update_model_score(id_model:int,score:float):
    """
    Update le F1 score du modèle id_model
    """
    curr = get_cursor()
    curr.execute("UPDATE Models SET F1_mean = ? WHERE id = ?;",(score,id_model))
    DB.commit()

def update_auto_score(id_auto:int,score:float):
    """
    Update le F1 score de l'automate id_auto
    """
    curr = get_cursor()
    curr.execute("UPDATE Autos SET F1_mean = ? WHERE id = ?;",(score,id_auto))
    DB.commit()
    

def get_ids_dataset(lang,label,nb_train,len_train,nb_val,len_val,nb_test,len_test):
    """
    Récupère tout les ids correspondant au paramètres donnés
    """
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

def get_ids_models(lang,dataset,weights,embedding_dim,hidden_dim):
    """
    Récupère tout les ids correspondant au paramètres donnés
    """
    curr = get_cursor()
    curr.execute("""SELECT id,epochs FROM Models WHERE 
        lang = ? AND
        dataset = ? AND
        weights = ? AND
        embedding_dim = ? AND
        hidden_dim = ?
        ORDER BY epochs DESC;
        """,(lang,dataset,weights,embedding_dim,hidden_dim))
    data = curr.fetchall()
    return data



def get_ids_autos(lang,dataset,model,nb_clusters):
    """
    Récupère tout les ids correspondant au paramètres donnés
    """
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
