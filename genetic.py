import random as rng
from typing import List
import pickle
import benchmarksV2 as bc

import db
import numpy as np
import os

class Experiment():
    def __init__(self,id_lang):
        self.id_lang = id_lang
        self.len_train = rng.randrange(50,2000,100)
        self.nb_train = rng.randrange(50,700,20)

        self.len_val = rng.randrange(50,2000,100)
        self.nb_val = rng.randrange(50,700,20)

        self.len_test = rng.randrange(50,2000,100)
        self.nb_test = rng.randrange(50,700,20)

        self.label = rng.choice(["multi-label"])

        self.poids = rng.randint(0,3)

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



def make_expe_and_log(expe:Experiment):
    auto,finals,infos = db.load_language(expe.id_lang)
    dataset_name = "test/db_files"
    # Get the dataset
    ids = get_ids_dataset(expe.id_lang,expe.label,expe.nb_train,expe.len_train,expe.nb_val,expe.len_val,expe.nb_test,expe.len_test)
    if len(ids) == 0:
        print("No dataset found creating it....")
        bc.create_dataset_and_save_it(auto,infos,expe.label,expe.nb_test,expe.nb_train,expe.len_test,expe.len_train,dataset_name)
        
    


if __name__ == "__main__":
    pass