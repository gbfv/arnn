import sys
import random
import torch
import queue
import os
import multiprocessing as mp
import subprocess
import time

from genetic import Experiment, make_expe_and_log,next_gen
import database as db
import benchmarksV2 as bc

def init_thread(queue):
    gpu_id = queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage exe <nb_gen> <nb_per_gen>")
    # C'est une copie du main de genetic:

    nb_gen = int(sys.argv[1])
    nb_per_gen = int(sys.argv[2])

    while True: 
        curr = db.get_cursor()
        curr.execute("SELECT id FROM Languages;")
        ids = curr.fetchall()
        next_id = -1
        print(ids)
        if len(ids) == 0:
            next_id = 1
        else:
            ids.sort()
            next_id = ids[-1][0] +1

        auto,finals,infos = bc.create_first_auto("ml",len_words=5,nb_words=4)
        by = db.give_raw_bytes_language(auto,finals,infos)
        db.add_entry_lang(
            "no_specific_name",
            len(infos["alphabet"]),
            len(infos["mots"]),
            len(infos["mots"][0]),
            len(auto.states),infos["reset_char"] is None,
            by)

        num_gpus = torch.cuda.device_count()
        gpus = [x for x in range(num_gpus)]
        print(f"Number of GPU found:{num_gpus}")
        process = [None] * num_gpus
        array_exp = [Experiment(next_id) for _ in range(nb_per_gen)]

        Man = mp.Manager()
        gpu_queue = Man.Queue()
        for g in gpus:
            gpu_queue.put(g)
        
        with mp.Pool(processes=num_gpus,initializer=init_thread,initargs=(gpu_queue,)) as pool:
            for gen_i in range(nb_gen):
                F1_model_arr = []
                F1_auto_arr = []
                scores = pool.map(make_expe_and_log,array_exp)
                for (F1_model,F1_auto) in scores:
                    F1_model_arr.append(F1_model)
                    F1_auto_arr.append(F1_auto)
                array_exp = next_gen(array_exp,F1_auto_arr)



        
        
    

