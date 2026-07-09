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
        self.epochs = 1000

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
            res.weight_id = self.weight_id
        
        if rng.random() > chance:
            res.nb_state_discovered = self.nb_state_discovered
            
        if rng.random() > chance:
            res.nb_clusters = self.nb_clusters
        return res

    def export_values(self):
        res = []
        res.append(self.nb_train)
        res.append(self.len_train)
        res.append(self.nb_val)
        res.append(self.len_val)
        res.append(self.nb_test)
        res.append(self.len_test)
        res.append(self.label)
        res.append(self.epochs)
        res.append(self.weight_id)
        res.append(self.nb_state_discovered)
        res.append(self.nb_clusters)
        return res
    
    def import_values(self,vals):
        self.nb_train = vals[0]
        self.len_train = vals[1]
        self.nb_val = vals[2]
        self.len_val = vals[3]
        self.nb_test = vals[4]
        self.len_test = vals[5]
        self.label = vals[6]
        self.epochs = vals[7]
        self.weight_id = vals[8]
        self.nb_state_discovered = vals[9]
        self.nb_clusters = vals[10]

    def reproduce(self,other):
        self_vals = self.export_values()
        other_values = other.export_values()
        new_experience_1_vals = []
        new_experience_2_vals = []
        for i in range(len(self_vals)):
            choice = rng.choice(["self","other"])
            if choice == "self":
                new_experience_1_vals.append(self_vals[i])
                new_experience_2_vals.append(other_values[i])
            else:
                new_experience_1_vals.append(other_values[i])
                new_experience_2_vals.append(self_vals[i])
        E1 = Experiment(self.id_lang)
        E2 = Experiment(self.id_lang)
        E1.import_values(new_experience_1_vals)
        E2.import_values(new_experience_2_vals)
        return E1,E2

    def __str__(self):
        #Un peu de magie noire
        all_vars = [x for x in dir(self) if not x.startswith("__")]
        return "|".join([f"{x}:{getattr(self,x)}" for x in all_vars if not str(getattr(self,x)).startswith("<")])

def next_gen(list_expes:List[Experiment],scores:List[float]):
    rank_i = np.argsort(scores)
    rank_i = np.flip(rank_i)[:len(list_expes)//2]
    print(rank_i)
    
    new_expes = []
    #First half duplicate themselfs
    for best_i in rank_i:
        new_expes.append(list_expes[best_i].clone(0.50))
    
    #First quarter reproduce
    rank_i = rank_i[:len(rank_i) // 2+1]
    for i in range(1,len(rank_i)):
        E1,E2 = list_expes[rank_i[0]].reproduce(list_expes[rank_i[i]])
        new_expes.append(E1)
        new_expes.append(E2)
    return new_expes





def make_expe_and_log(expe:Experiment):
    auto,finals,infos = db.load_language(expe.id_lang)
    dataset_name = "test/db_files"
    mots = infos["mots"]
    # Get the dataset
    the_id_dataset = -1
    ids_dataset = db.get_ids_dataset(expe.id_lang,expe.label,expe.nb_train,expe.len_train,expe.nb_val,expe.len_val,expe.nb_test,expe.len_test)
    if len(ids_dataset) == 0:
        print("No dataset found creating it....")
        bc.create_dataset_and_save_it(auto,infos,expe.label,expe.nb_test,expe.nb_train,expe.len_test,expe.len_train,dataset_name)
        d1,d2,d3 = db.capture_datasets()
        db.add_entry_dataset(expe.id_lang,"no_specific_name",expe.label,expe.nb_train,expe.len_train,expe.nb_val,expe.len_val,expe.nb_test,expe.len_test,d1,d2,d3)
        the_id_dataset = db.get_ids_dataset(expe.id_lang,expe.label,expe.nb_train,expe.len_train,expe.nb_val,expe.len_val,expe.nb_test,expe.len_test)[0][0]
    else:
        the_id_dataset = ids_dataset[0][0]
        print(f"Dataset found (id:{the_id_dataset}), extracting...")
        db.load_datasets_to_file(the_id_dataset)

    the_id_model = -1
    M = None
    ids_models = db.get_ids_models(expe.id_lang,the_id_dataset,expe.epochs,expe.weight_id)
    if len(ids_models) == 0:
        print("No model found, Training....")
        M = bc.create_model(mots,expe.label,bc.make_weights(expe.weight_id,mots))
        epoch_done = 0
        # A noter il FAUT que l'epoch soit un multiple de 10
        for i in range(100,expe.epochs+100,100):
            M = bc.train_model(M,50,dataset_name)
            epoch_done += 50
            model_bytes = db.give_raw_bytes_model(M)
            model_specs_bytes = db.give_raw_bytes_model_specs(M)
            print("saving...")
            db.add_entry_models(expe.id_lang,the_id_dataset,"no_specific_name",epoch_done,expe.weight_id,model_bytes,model_specs_bytes)
        the_id_model = db.get_ids_models(expe.id_lang,the_id_dataset,expe.epochs,expe.weight_id)[0][0]
    else:
        print("Model found, extracting...")
        the_id_model = ids_models[0][0]
        M = db.load_model(the_id_model)
    
    F1 = bc.test_model(M,expe.label,dataset_name)
    if len(ids_models) == 0:
        db.update_model_score(the_id_model,np.mean([x[-1] for x in F1]))
    the_id_auto = -1
    A = None
    ids_autos = db.get_ids_autos(expe.id_lang,the_id_dataset,the_id_model,expe.nb_clusters)
    if len(ids_autos) == 0:
        print("No auto found, Creating...")
        A = bc.get_automate_from_model(M,infos,expe.nb_clusters,dataset_name,"pred")
        bytes_auto = db.give_raw_bytes_auto(A)
        db.add_entry_auto(expe.id_lang,the_id_dataset,the_id_model,"no_specific_name",-1,expe.nb_clusters,len(A.Q),bytes_auto)
        the_id_auto = db.get_ids_autos(expe.id_lang,the_id_dataset,the_id_model,expe.nb_clusters)[0][0]
    else:
        print("Auto found, extracting....")
        the_id_auto = ids_autos[0][0]
        A = db.load_auto_from_db(the_id_auto)
    F2 = bc.test_automate(A,M,infos,mots,expe.label,-1,dataset_name)
    if len(ids_autos) == 0:
        db.update_auto_score(the_id_auto,np.mean([x[-1] for x in F2]))
    return F1, F2
        


    
    
    
def genetic_algorithm(id_language:int,generations:int,nb_tested:int):
    pool = [Experiment(id_language) for _ in range(nb_tested)]
    scores = []
    for i in range(generations):
        for e in pool:
            F1_m,F1_a = make_expe_and_log(e)
            F2_1_mean = np.mean([x[-1] for x in F1_a])
            scores.append(F2_1_mean)
        pool = next_gen(pool,scores)
        scores = []

    

        
    


if __name__ == "__main__":
    db.setup_db()
    while True: 
        curr = db.get_cursor()
        curr.execute("SELECT id FROM Languages;")
        ids = curr.fetchall()
        next_id = -1
        if len(ids) == 0:
            next_id = 1
        else:
            next_id = ids.sort()[-1][0] +1

        auto,finals,infos = bc.create_first_auto("ml",len_words=5,nb_words=4)
        by = db.give_raw_bytes_language(auto,finals,infos)
        db.add_entry_lang(
            "no_specific_name",
            len(infos["alphabet"]),
            len(infos["mots"]),
            len(infos["mots"][0]),
            len(auto.states),infos["reset_char"] is None,
            by)
        genetic_algorithm(next_id,10,10)