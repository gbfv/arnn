import random as rng
from typing import List
import pickle
import benchmarksV2 as bc

import database as db
import numpy as np
import os

class Experiment():
    def __init__(self,id_lang):
        self.id_lang = id_lang
        self.nb_train = rng.randrange(50,2000,100)
        self.len_train = rng.randrange(50,700,20)

        self.nb_val = rng.randrange(50,2000,100)
        self.len_val = rng.randrange(50,700,20)

        self.nb_test = rng.randrange(50,2000,100)
        self.len_test = rng.randrange(50,700,20)

        self.label = rng.choice(["multi-label"])
        self.epochs = 500

        self.weight_id = rng.randint(0,3)

        self.nb_state_discovered = rng.randrange(10_000,1_000_000,50_000)
        self.nb_clusters = rng.randrange(100,800,50)

    def clone(self,chance):
        res = Experiment(self.id_lang)
        if rng.random() > chance:
            res.len_train = self.len_train
        
        if rng.random() > chance:
            res.nb_train = self.nb_train

        if rng.random() > chance:
            res.len_val = self.len_val

        if rng.random() > chance:
            res.nb_val = self.nb_val

        if rng.random() > chance:
            res.len_test = self.len_test
        
        if rng.random() > chance:
            res.nb_test = self.nb_test

        if rng.random() > chance:
            res.label = self.label

        if rng.random() > chance:
            res.poids = self.poids
        
        if rng.random() > chance:
            res.nb_state_discovered = self.nb_state_discovered
            
        if rng.random() > chance:
            res.nb_clusters = self.nb_clusters
        return res

    def __str__(self):
        #Un peu de magie noire
        all_vars = [x for x in dir(self) if not x.startswith("__")]
        return "|".join([f"{x}:{getattr(self,x)}" for x in all_vars if not str(getattr(self,x)).startswith("<")])

def next_gen(list_expes:List[Experiment],scores:List[float]):
    rank_i = np.argsort(scores)
    rank_i = np.flip(rank_i)[:len(list_expes)//2]
    new_expes = []
    for best_i in rank_i:
        new_expes.append(list_expes[best_i].clone(0.99))
        new_expes.append(list_expes[best_i].clone(0.99))
    return new_expes




def get_ids_dataset(lang,label,nb_train,len_train,nb_val,len_val,nb_test,len_test):
    curr = db.get_cursor()
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
    curr = db.get_cursor()
    curr.execute("""SELECT id FROM Models WHERE 
        lang = ? AND
        dataset = ? AND
        epochs = ? AND
        weights = ?
        """,(lang,dataset,epochs,weights))
    data = curr.fetchall()
    return data

def get_ids_autos(lang,dataset,model,nb_clusters):
    curr = db.get_cursor()
    curr.execute("""SELECT id FROM Autos WHERE 
        lang = ? AND
        dataset = ? AND
        model = ? AND
        nb_clusters = ?
        """,(lang,dataset,model,nb_clusters))
    data = curr.fetchall()
    return data




def make_expe_and_log(expe:Experiment):
    auto,finals,infos = db.load_language(expe.id_lang)
    dataset_name = "test/db_files"
    mots = infos["mots"]
    # Get the dataset
    the_id_dataset = -1
    ids_dataset = get_ids_dataset(expe.id_lang,expe.label,expe.nb_train,expe.len_train,expe.nb_val,expe.len_val,expe.nb_test,expe.len_test)
    if len(ids_dataset) == 0:
        print("No dataset found creating it....")
        bc.create_dataset_and_save_it(auto,infos,expe.label,expe.nb_test,expe.nb_train,expe.len_test,expe.len_train,dataset_name)
        d1,d2,d3 = db.capture_datasets()
        db.add_entry_dataset(expe.id_lang,"no_specific_name",expe.label,expe.nb_train,expe.len_train,expe.nb_val,expe.len_val,expe.nb_test,expe.len_test,d1,d2,d3)
        the_id_dataset = get_ids_dataset(expe.id_lang,expe.label,expe.nb_train,expe.len_train,expe.nb_val,expe.len_val,expe.nb_test,expe.len_test)[0][0]
    else:
        the_id_dataset = ids_dataset[0][0]
        print(f"Dataset found (id:{the_id_dataset}), extracting...")
        db.load_datasets_to_file(the_id_dataset)

    the_id_model = -1
    M = None
    ids_models = get_ids_models(expe.id_lang,the_id_dataset,expe.epochs,expe.weight_id)
    if len(ids_models) == 0:
        print("No model found, Training....")
        M = bc.create_model(mots,expe.label,bc.make_weights(expe.weight_id,mots))
        epoch_done = 0
        # A noter il FAUT que l'epoch soit un multiple de 10
        for i in range(50,expe.epochs+50,50):
            M = bc.train_model(M,50,dataset_name)
            epoch_done += 50
            model_bytes = db.give_raw_bytes_model(M)
            print("saving...")
            db.add_entry_models(expe.id_lang,the_id_dataset,"no_specific_name",epoch_done,expe.weight_id,model_bytes)
        the_id_model = get_ids_models(expe.id_lang,the_id_dataset,expe.epochs,expe.weight_id)[0][0]
    else:
        print("Model found, extracting...")
        the_id_model = ids_models[0][0]
        M = db.load_model(the_id_model)
    
    F1 = bc.test_model(M,expe.label,dataset_name)
    the_id_auto = -1
    A = None
    ids_autos = get_ids_autos(expe.id_lang,the_id_dataset,the_id_model,expe.nb_clusters)
    if len(ids_autos) == 0:
        print("No auto found, Creating...")
        A = bc.get_automate_from_model(M,infos,expe.nb_clusters,dataset_name,"pred")
        A.minimize()
        bytes_auto = db.give_raw_bytes_auto(A)
        db.add_entry_auto(expe.id_lang,the_id_dataset,the_id_model,"no_specific_name",-1,expe.nb_clusters,len(A.Q),bytes_auto)
        the_id_auto = get_ids_autos(expe.id_lang,the_id_dataset,the_id_model,expe.nb_clusters)[0][0]
    else:
        print("Auto found, extracting....")
        the_id_auto = ids_autos[0][0]
        A = db.load_auto_from_db(the_id_auto)
    
    init_st = A.find_initial_state()
    F2 = bc.test_automate(A,M,infos,mots,expe.label,init_st,dataset_name)
    return F1, F2
        


    
    
    

    

        
    


if __name__ == "__main__":
    db.setup_db()
    rng.seed(67)
    E = Experiment(1)
    make_expe_and_log(E)