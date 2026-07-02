import random as rng


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

i = Experiment(3)
print(i)