import argparse
import random

def main():
    parser = argparse.ArgumentParser(prog="Createur commandes automates",description="Crée des commmandes qui seront rentrées dans build_auto.py")
    parser.add_argument("-A",type=int,help="Taille de l'alphabet")
    parser.add_argument("-R",action='store_true',default=False,help="Reset caracter(add one caracter to the alphabet)")
    parser.add_argument("-N",type=int,help="Nb de mots")
    parser.add_argument("-n",type=int,help="Taille des mots")
    parser.add_argument("-s",type=int,default=0,help="Seed")
    parser.add_argument("-C",type=int,default=1,help="Nb de commandes a générer")
    parser.add_argument('-b',action='store_true',default=False,help="Fait une version avec reset et une sans")
    args = parser.parse_args()
    if not (args.A  or args.N or args.n):
        parser.print_help()
        return

    alphabet_len = args.A + (1 if args.R else 0)
    alphabet_mots = "".join([chr(ord('a')+ x ) for x in range(alphabet_len-1)])
    alphabet = alphabet_mots + (chr(ord("a")+alphabet_len-1) if args.R else "")
    reset_caracter = chr(ord("a")+alphabet_len-1) if args.R else ""
    if args.s != 0:
        random.seed(args.s)
    for i in range(args.C):
        res = f"python build_auto.py -a " + alphabet
        set_mots_avec_reset = set()
        set_mots_sans_reset = set()
        while len(set_mots) != args.N:
            mot = ""
            mot_sec = ""
            toutes_lettres  = list(alphabet_mots)
            for lettre in range(args.n):
                mot += random.choice(toutes_lettres)
            
            mot = mot + reset_caracter
            set_mots_sans_reset = set_mots.union(set([mot]))
        res += f" -m '{",".join(set_mots)}'"
        print(res)




    
if __name__ == "__main__":
    main()
