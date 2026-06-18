from greenery import parse, Charclass
import random
import pickle
import os
import argparse

#Convertie une fsm en version plus légère de notre classe automaton
#Permet de vérifier l'isomorphisme entre une fsm et nos automates
class light_automaton():
    def __init__(self, fsm):
        self.Q = set(fsm.states)
        self.F = set(fsm.finals) 
        self.initial = fsm.initial

        self.Sigma = set()
        for c in fsm.alphabet:
            if "[" in str(c) and "^" not in str(c):
                charclass = str(c).replace("[", "").replace("]", "").split("-")
                #print(f"Charclass : {charclass} / len : {len(charclass)}")
                if len(charclass) == 1: #charclass de forme [abc]
                    for char in charclass[0]:
                        self.Sigma.add(char)
                else: #charclass de forme [a-f] 
                    deb = ord(charclass[0])
                    fin = ord(charclass[1])
                    for i in range(deb, fin+1):
                        self.Sigma.add(chr(i))
            elif "^" not in str(c):
                self.Sigma.add(str(c))


        self.delta = {}
        for state, transitions in fsm.map.items():
            for charclass, next_state in transitions.items():
                if str(charclass) in self.Sigma:
                    self.delta[(state, str(charclass))] = next_state
                elif "[" in str(charclass) and "^" not in str(charclass):
                    charclass = str(charclass).replace("[", "").replace("]", "").split("-")
                    if len(charclass) == 1: #charclass de forme [abc]
                        for char in charclass[0]:
                            self.delta[(state, char)] = next_state
                    else: #charclass de forme [a-f]
                        deb = ord(charclass[0])
                        fin = ord(charclass[1])
                        for i in range(deb, fin+1):
                            self.delta[(state, chr(i))] = next_state

        #normalisation
        self.Sigma = {ord(c)-97 for c in self.Sigma}
        self.delta = {(state, ord(char)-97): next_state for (state, char), next_state in self.delta.items()}


    def __str__(self):
        res = f"Sigma : {self.Sigma}\nQ : {self.Q}\nF : {self.F}\ninitial : {self.initial}\ndelta :\n"
        for (state, char), next_state in self.delta.items():
            res += f"  ({state}, {char}) -> {next_state}\n"
        return res


def is_overlapping(word, language):
    for i in range(1, len(word)):
        part = word[:i]
        for w in language:
            if w.startswith(part) or w.endswith(part):
                return True
    return False

#génère un langage dont les mots ne se recouvrent pas
def no_overlap(alphabet, nbr, length, reset=None):
    """ flex : retourne les mots générés même s'ils n'y en a pas assez (uniquement si reset est None) """
    cpy_alphabet = alphabet.copy()
    combinaisons = len(alphabet)**length if reset is None else len(alphabet)**(length-1)
    if nbr > combinaisons:
        raise ValueError(f"Impossible de générer {nbr} mots de longueur {length} avec l'alphabet donné. Nombre maximum de combinaisons : {combinaisons}")

    attempts = 0
    words = set()

    if reset is not None:
        cpy_alphabet.remove(reset) if reset in cpy_alphabet else None #peut avoir un reset hors de l'alphabet

    while len(words) < nbr and attempts < nbr*10:
        word = ''.join(random.choices(cpy_alphabet, k=length))
        attempts += 1
        if not is_overlapping(word, words):
            words.add(word)

    return list(words), reset



#TODO: Serait cool de passer en json pour plus de compatibilité et de visibilité
def save_fsm(fsm, filename, path=""):
    with open(f"{path}{filename}.pkl", "wb") as f:
        pickle.dump(fsm, f)

def load_fsm(filename, path=""):
    path += "/" if path and not path.endswith("/") else ""
    with open(f"{path}{filename}.pkl", "rb") as f:
        return pickle.load(f)
    


def accept_stream(stream, fsm, motifs, method):
    res = []
    current_state = fsm.initial
    length_motif = max(len(motif) for motif in motifs)
    trad = dict()

    for l in fsm.alphabet:
        str_l = str(l)
        if "-" in str_l and "^" not in str_l:
            deb, fin = str_l.split("-")
            for i in range(ord(deb), ord(fin)+1):
                trad[chr(i)] = l
        elif "[" in str_l and "^" not in str_l:
            for char in str_l.replace("[", "").replace("]", ""):
                trad[char] = l
        elif "^" not in str_l:
            trad[str_l] = l

    for c in stream:
        charclass = trad.get(c, "")
        try:
            current_state = fsm.map[current_state][charclass]
            if method == "state":
                res.append(current_state)
            else:
                if current_state in fsm.finals:
                    res.append(1)
                else:
                    res.append(0)
        except KeyError:
            print(f"Aucune transition entre l'état {current_state} et le symbole {c}")
            return []
        
    if method == "multi-classe":
        hit = [i for i in range(len(res)) if res[i] == 1]
        for index in hit:
            debut = index-(length_motif-1)
            res[debut:index+1] = list(range(1, length_motif+1))

    elif method == "multi-label":
        label = [[0]*len(motifs) for _ in range(len(res))]
        hit = [i for i in range(len(res)) if res[i] == 1]
        for index in hit:
            debut = index-(length_motif-1)
            mot = motifs.index(stream[debut:index+1])
            for j in range(debut, index+1):
                label[j][mot] = j+1-debut
        res = label
    return res



def get_automate(prefix, save=True):
    automate, final_states, params = None, None, None

    #pas très opti de chercher dans les répertoires à chaque fois (créer un singleton pour gérer ça ?)
    if os.path.exists(f"test/fsm/{prefix}.pkl"):
        print(f"FSM {prefix} already exists, loading it...")
        data = load_fsm(prefix, path="test/fsm/")
        automate = data["automate"]
        final_states = data["final_states"]
        params = data["params"]
        print(f"{automate}\n{params}")
        
    else: #TODO: gros travail à faire ici pour générer des automates plus complexes et variés
        print(f"FSM {prefix} does not exist, generating it...")
        #configuration aléatoire de l'automate
        alpha_end = random.randint(100, 103) #Entre [a-c] et [a-f]
        alphabet = [chr(i) for i in range(97, alpha_end)]
        nbr_max = random.randint(3, 5)
        taille = random.randint(3, 6)
        reset = random.choice(alphabet) if random.random() < 0.5 else None
        #genération du langage et de l'automate
        L, reset = no_overlap(alphabet=alphabet, nbr=nbr_max, length=taille, reset=reset)
        regex = f"([{"".join(alphabet)}]|.)*({'|'.join(L)})"
        fsm = parse(regex).to_fsm()
        
        automate = fsm.reduce()
        final_states = set(fsm.finals)
        params = {
            "alphabet" : alphabet,
            "mots" : L,
            "reset_char" : reset
        }
        if save:
            save_fsm({"automate": automate, "final_states": final_states, "params": params}, prefix, path="test/fsm/")
        print(f"Alphabet : {alphabet}\nRegex : {regex}\nReset : {reset}")
        print(automate)

    return automate, final_states, params



def reproduction(nombre, seed=42):
    """ Génère un nombre d'automates aléatoires à partir d'une seed donnée, pour pouvoir les reproduire facilement """
    random.seed(seed)
    for i in range(nombre):
        print(f"[*] Génération de l'automate {i+1} :\n=============================")
        get_automate(f"auto_{i+1}")
        print("\n\n\n") if i < nombre-1 else None
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère un automate")
    parser.add_argument("-a", "--alphabet", type=str, default="", help="Alphabet de l'automate (ex: 'abc' pour {a,b,c})")
    parser.add_argument("-m", "--mots", type=str, default="", help="Mots du langage de l'automate (ex: 'abc,acb' pour {abc, acb})")
    parser.add_argument("-n", "--nom", type=str, default="auto_custom", help="Nom de l'automate généré")
    parser.add_argument("-t", "--test", action="store_true", help="Uniquement pour test, ne sauvegarde pas l'automate dans test/fsm/.")
    args = parser.parse_args()

    if not args.alphabet or not args.mots:
        """ for i in range(5):
            print("[*] Génération d'un automate aléatoire :\n=============================")
            get_automate("", save=False)
            print("\n\n\n") if i < 4 else None

        for _ in range(5):
            L, reset = no_overlap(alphabet=["a", "b", "c"], nbr=4, length=3)
            print(f"Langage : {L}, reset : {reset}") """

        L = ["abcd", "abce", "eeee", "eead"]
        regex = f"([abcdef]|.)*(abcd|abce|eeee|eead)"
        print(f"Regex : {regex}")
        fsm = parse(regex).to_fsm()
        print(fsm)
        print(accept_stream("aaabcbcbcbcbacaaaaab", fsm, L, method="pos"))
        #print(light_automaton(fsm))
    else:
        alphabet = list(args.alphabet)
        mots = args.mots.split(",") if args.mots else []

        if set("".join(mots))-set(alphabet) != set():
            print("Erreur : les mots doivent être composés de caractères de l'alphabet")
            exit(1)

        regex = f"([{"".join(alphabet)}]|.)*({'|'.join(mots)})"
        print(f"Regex : {regex}")
        fsm = parse(regex).to_fsm()
        print(fsm)

        if not args.test:
            params = {
                "alphabet" : alphabet,
                "mots" : mots,
                "reset_char" : None
            }
            save_fsm({"automate": fsm.reduce(), "final_states": set(fsm.finals), "params": params}, args.nom, path="test/fsm/")
