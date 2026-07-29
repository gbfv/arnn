import random as rng
from typing import List
import pickle
import utils_test as utl

import database as db
import numpy as np
import os
import logging

"""
    L'expérience est une classe qui représente un ensemble d'hyperparamètres
"""
class Experiment():
    def __init__(self,id_lang,label):
        self.data:dict= {}
        self.data["id_lang"] = id_lang
        self.data["nb_train"] = rng.randrange(50,2000,100)
        self.data["len_train"] = rng.randrange(50,700,20)

        self.data["nb_val"] = rng.randrange(50,2000,100)
        self.data["len_val"] = rng.randrange(50,700,20)

        self.data["nb_test"] = rng.randrange(50,2000,100)
        self.data["len_test"] = rng.randrange(50,700,20)

        self.data["label"] = label
        self.data["epochs"] = rng.randrange(100,1000,100)

        self.data["weight_id"] = rng.randint(0,2)

        self.data["nb_clusters"] = rng.randrange(100,800,50)

        self.data["embedding_dim"] = rng.randrange(1,5,1)
        self.data["hidden_dim"] = rng.randrange(10,100,10)

    def clone(self,chance):
        """
        Clone l'expérience avec pour chaque hyperparmaètres une chance de ne PAS être modifiée
        """
        res = Experiment(self.data["id_lang"],self.data["label"])

        for key, val in self.data.items():
            if rng.random() > chance:
                res.data[key] = val

        return res

    def export_values(self):
        """
        Exporte les valeurs
        """
        return self.data
    
    def import_values(self,vals):
        """
        Importe les valeurs
        """
        self.data = vals

    def crossover(self,other):
        """
        Reproduit deux expériences.

        Pour chaque hyperparamètre l'enfant à une chance sur 2 de recevoir celui du parent 1 sinon il prend celui du parent 2

        Renvoie 2 enfants qui sont complémantaires
        """
        new_experience_1_vals = {}
        new_experience_2_vals = {}
        for key,vals in self.data.items():
            choice = rng.choice(["self","other"])
            if choice == "self":
                new_experience_1_vals[key] = self.data[key]
                new_experience_2_vals[key] = other.data[key]
            else:
                new_experience_1_vals[key] = other.data[key]
                new_experience_2_vals[key] = self.data[key]
        E1 = Experiment(self.data["id_lang"],self.data["label"])
        E2 = Experiment(self.data["id_lang"],self.data["label"])
        E1.import_values(new_experience_1_vals)
        E2.import_values(new_experience_2_vals)
        return E1,E2

    def __str__(self):
        #Un peu de magie noire
        all_vars = [x for x in dir(self) if not x.startswith("__")]
        return "|".join([f"{x}:{getattr(self,x)}" for x in all_vars if not str(getattr(self,x)).startswith("<")])


    def load_or_create_dataset(self,auto,finals,infos_automate,dataset_name):
        """
        Charge le dataset demandé et s'il n'existe pas, le crée

        Revoie l'id du dataset
        """
        mots = infos_automate["mots"]
        # Get the dataset
        the_id_dataset = -1
        ids_dataset = db.get_ids_dataset(self.data["id_lang"],self.data["label"],self.data["nb_train"],self.data["len_train"],self.data["nb_val"],self.data["len_val"],self.data["nb_test"],self.data["len_test"])
        if len(ids_dataset) == 0:
            logging.info("No dataset found creating it....")
            utl.create_dataset_and_save_it(auto,infos_automate,self.data["label"],self.data["nb_test"],self.data["nb_train"],self.data["len_test"],self.data["len_train"],dataset_name)
            d1,d2,d3 = db.capture_datasets(dataset_name)
            the_id_dataset = db.add_entry_dataset(self.data["id_lang"],"no_specific_name",self.data["label"],self.data["nb_train"],self.data["len_train"],self.data["nb_val"],self.data["len_val"],self.data["nb_test"],self.data["len_test"],d1,d2,d3)
        else:
            the_id_dataset = ids_dataset[0][0]
            logging.info(f"Dataset found (id:{the_id_dataset}), extracting...")
            db.load_datasets_to_file(the_id_dataset,dataset_name)
        return the_id_dataset

    def train_model_and_save(self,M,dataset_name,epochs,total_epochs_done,random_key,id_dataset):
        M = utl.train_model(M,epochs,dataset_name)
        model_bytes = db.give_raw_bytes_model(M,random_key)
        model_specs_bytes = db.give_raw_bytes_model_specs(M,random_key)
        the_id_model = db.add_entry_models(self.data["id_lang"],id_dataset,"no_specific_name",total_epochs_done,self.data["weight_id"],self.data["embedding_dim"],self.data["hidden_dim"],model_bytes,model_specs_bytes)
        #On fait le score et on sauvegarde
        F1 = utl.test_model(M,self.data["label"],dataset_name)
        F1_mean = self.calculate_F1_mean(F1)
        db.update_model_score(the_id_model,F1_mean)
        return M, the_id_model

    def load_or_create_model(self,id_dataset,auto,infos_automate,dataset_name):
        """
        Charge le modèle demandé et s'il n'existe pas, le crée

        Revoie le modèle, son id
        """
        the_id_model = -1
        mots = infos_automate["mots"]
        M = None
        #on utilise une clé aléatoire pour ne pas avoir de problème de collision en cas de multi_treading
        random_key = "".join(rng.choices(list("azertyuiopqsdfghjklmwxcvbn1234567890"),k=8))
        ids_models_and_epochs = db.get_ids_models(self.data["id_lang"],id_dataset,self.data["weight_id"],self.data["embedding_dim"],self.data["hidden_dim"])
        if len(ids_models_and_epochs) == 0:
            logging.info("No model found, Training....")
            M = utl.create_model(mots,self.data["label"],utl.make_weights(self.data["weight_id"],mots),auto,self.data["epochs"],self.data["embedding_dim"],self.data["hidden_dim"])
            # A noter il FAUT que l'epoch soit un multiple de 100
            for i in range(100,self.data["epochs"]+100,100):
                M, the_id_model = self.train_model_and_save(M,dataset_name,100,i,random_key,id_dataset)
        else:
            logging.info("Model found, extracting...")
            #On cherche le plus proche en dessous on SAIT qu'il y en a un car le minimum est 100
            nb_epochs_to_do = -1
            id_to_take = -1
            for id_model,epo in ids_models_and_epochs:
                if epo <= self.data["epochs"] and (nb_epochs_to_do == -1 or self.data["epochs"] - epo < nb_epochs_to_do):
                    id_to_take = id_model
                    nb_epochs_to_do = self.data["epochs"] - epo
            M = None
            #This one SHOULD not happend but we cover it
            if (id_to_take == -1):
                M = utl.create_model(mots,self.data["label"],utl.make_weights(self.data["weight_id"],mots),auto,self.data["epochs"],self.data["embedding_dim"],self.data["hidden_dim"])
                M,the_id_model= self.train_model_and_save(M,dataset_name,self.data["epochs"],self.data["epochs"],random_key,id_dataset)
                return M,the_id_model
            if (nb_epochs_to_do == 0):
                M = db.load_model(id_to_take,random_key)
                the_id_model = id_to_take
            else:
                M = db.load_model(id_to_take,random_key)
                M,the_id_model = self.train_model_and_save(M,dataset_name,nb_epochs_to_do,self.data["epochs"],random_key,id_dataset)
        return M, the_id_model

    def load_or_create_autos(self,model,id_model,id_dataset,infos_automate,dataset_name):
        """
        Charge l'automate demandé et s'il n'existe pas, le crée

        Revoie l'automate et son id
        """
        the_id_auto = -1
        A = None
        mots = infos_automate["mots"]
        ids_autos = db.get_ids_autos(self.data["id_lang"],id_dataset,id_model,self.data["nb_clusters"])
        if len(ids_autos) == 0:
            logging.info("No auto found, Creating...")
            A = utl.get_automate_from_model(model,infos_automate,self.data["nb_clusters"],dataset_name,"pred")
            bytes_auto = db.give_raw_bytes_auto(A)
            the_id_auto = db.add_entry_auto(self.data["id_lang"],id_dataset,id_model,"no_specific_name",self.data["nb_clusters"],len(A.Q),bytes_auto)
            F1 = utl.test_automate(A,model,infos_automate,mots,self.data["label"],-1,dataset_name)
            F1_mean = self.calculate_F1_mean(F1)
            db.update_auto_score(the_id_auto,F1_mean)
        else:
            logging.info("Auto found, extracting....")
            the_id_auto = ids_autos[0][0]
            A = db.load_auto_from_db(the_id_auto)
        return A,the_id_auto


    def calculate_F1_mean(self,F1):
        """
        Transforme le tableau de score F1 en un seul nombre dépandant du label de l'expérience
        """
        match self.data["label"]:
            case "state":
                return np.mean(F1)
            case "multi-classe":
                return F1[-1]
            case "multi-label":
                return np.mean([x[-1] for x in F1])


    def make_expe(self):
        """
        Fais l'expérience et remmplit la base de donnée

        Renvoie le score F1 du modèle et de l'automate
        """
        dataset_name = "test/" + "".join(rng.choices(list("azertyuiopqsdfghjklmwxcvbn1234567890"),k=8)) #On a besoin que se soit aléatoire si multi-thread
        auto,finals,infos = db.load_language(self.data["id_lang"])
        mots = infos["mots"]
        #Le dataset
        id_dataset = self.load_or_create_dataset(auto,finals,infos,dataset_name)
        #Le modèle
        M, id_model= self.load_or_create_model(id_dataset,auto,infos,dataset_name)
        F1_model = utl.test_model(M,self.data["label"],dataset_name)
        F1_model_mean = self.calculate_F1_mean(F1_model)
        #L'automate
        A, the_id_auto = self.load_or_create_autos(M,id_model,id_dataset,infos,dataset_name)
        F1_auto = utl.test_automate(A,M,infos,mots,self.data["label"],-1,dataset_name)
        F1_auto_mean = self.calculate_F1_mean(F1_auto)
        return F1_model_mean, F1_auto_mean





def next_gen(list_expes:List[Experiment],scores:List[float]):
    """
    Crée la prochaine génération

    La première moitiée se duplique

    Le premier quart ce reproduit

    Renvoie la liste de nouvelle expériences
    """
    rank_i = np.argsort(scores)
    logging.debug(rank_i)
    rank_i = np.flip(rank_i)[:len(list_expes)//2]
    
    new_expes = []
    #First half duplicate themselfs
    for best_i in rank_i:
        new_expes.append(list_expes[best_i].clone(0.8))
    
    #First quarter reproduce
    rank_i = rank_i[:len(rank_i) // 2+1]
    for i in range(1,len(rank_i)):
        E1,E2 = list_expes[rank_i[0]].crossover(list_expes[rank_i[i]])
        new_expes.append(E1)
        new_expes.append(E2)
    return new_expes

    
    
def genetic_algorithm(id_language:int,generations:int,nb_tested:int):
    """
    Applique l'algorithme génétique
    """
    label = rng.choice(["multi-label","multi-classe","state"])
    pool = [Experiment(id_language,label) for _ in range(nb_tested)]
    scores = []
    for i in range(generations):
        for e in pool:
            F1_m,F1_a = e.make_expe()
            scores.append(F1_a)
        pool = next_gen(pool,scores)
        scores = []

    

        
    
import sys

if __name__ == "__main__":
    db.setup_db()
    logging.basicConfig(level=logging.DEBUG)
    while True: 
        curr = db.get_cursor()
        curr.execute("SELECT id FROM Languages;")
        ids = curr.fetchall()
        next_id = -1
        if len(ids) == 0:
            next_id = 1
        else:
            ids.sort()
            next_id = ids[-1][0] +1

        auto,finals,infos = utl.create_first_auto("ml")
        by = db.give_raw_bytes_language(auto,finals,infos)
        db.add_entry_lang(
            "no_specific_name",
            len(infos["alphabet"]),
            len(infos["mots"]),
            len(infos["mots"][0]),
            len(auto.states),infos["reset_char"] is None,
            by)
        genetic_algorithm(next_id,10,40)