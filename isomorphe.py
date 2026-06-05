import pickle
import os
import networkx as nx

def Weisfeiler_Leman(A):
    A_color = {q: 0 for q in A.Q}
    for q in A.F:
        A_color[q] = 1 #on colore les états finaux en 1 et les autres en 0
    alphabet = A.Sigma

    actual_color = 1
    old_color = 0

    while actual_color > old_color:
        new_color = {}
        for q in A.Q:
            signature = []
            for a in alphabet:
                signature.append((a, A_color[A.delta[(q, a)]])) #liste du voisinage de q
            sorted_signature = sorted(signature) #on trie le voisinage pour que les états avec les mêmes voisins aient la même signature
            new_color[q] = hash((A_color[q], tuple(sorted_signature))) #nouvelle couleur de q : (couleur actuelle, voisinage)
            
        A_color = new_color
        old_color = actual_color
        actual_color = len(set(A_color.values())) #nombre de couleurs différentes
    return A_color




def is_isomorphic(A, B, translate=False):
    #filtres simples
    if len(A.Q) != len(B.Q) or len(A.F) != len(B.F):
        print(f"Non isomorphe : nombre d'états ou d'états finaux différent | A : Q={len(A.Q)}, F={len(A.F)} | B : Q={len(B.Q)}, F={len(B.F)}")
        return False, {}
    
    if len(A.Sigma) != len(B.Sigma):
        print(f"Non isomorphe : nombre de symboles différent | A : {len(A.Sigma)}, B : {len(B.Sigma)}")
        return False, {}
    
    if len(A.delta) != len(B.delta):
        print(f"Non isomorphe : nombre de transitions différent | A : {len(A.delta)}, B : {len(B.delta)}")
        return False, {}

    #on applique WL et on compare les couleurs
    color_A = Weisfeiler_Leman(A)
    color_B = Weisfeiler_Leman(B)

    sorted_colors_A = sorted(color_A.values())
    sorted_colors_B = sorted(color_B.values())

    isomorphic = sorted_colors_A == sorted_colors_B

    print(f"Isomorphe : {isomorphic}")
    print("Couleurs de A :")
    for k,v in color_A.items():
        print(f"Etat {k} : couleur {v}")
    print("Couleurs de B :")
    for k,v in color_B.items():
        print(f"Etat {k} : couleur {v}")

    if isomorphic:
        traduction = {}
        if translate:
            for qA, cA in color_A.items():
                for qB, cB in color_B.items():
                    if cA == cB:
                        traduction[qA] = qB
                        break
        return True, traduction
    else:
        return False, {}




def test_isomorphisme(A, path):
    """
    Test si les automates dans le dossier path sont isomorphes à A
    """
    #A.minimize()
    files = os.listdir(path)
    automates = [file for file in files if file.endswith(".pkl")]

    isomorphes = []
    for file in automates:
        with open(os.path.join(path, file), "rb") as f:
            B = pickle.load(f)
        #B.minimize()
        iso, _ = is_isomorphic(A, B)
        if iso:
            isomorphes.append(file)
        """ else:
            B.Q = B.Q-{-1} #retire -1 de B
            for s in B.Sigma:
                B.delta.pop((-1, s), None)
            qiso, _ = is_isomorphic(A, B)
            if qiso:
                quasi_isomorphes.append(file) """



    print(f"Automates isomorphes à A : {len(isomorphes)}")
    for iso in isomorphes:
        print(f"  - {iso}")
    print(f"Automates non isomorphes à A : {len(automates) - len(isomorphes)}")
    for file in automates:
        if file not in isomorphes:
            print(f"  - {file}")
    """ print(f"Automates quasi-isomorphes à A : {len(quasi_isomorphes)}")
    for qiso in quasi_isomorphes:
        print(f"  - {qiso}") """


#Obsolete : ne pas utiliser
def distance_graph(A, B):
    #reconnaissance
    recon = {q: [] for q in A.Q}
    signature_A = {q: [] for q in A.Q}
    signature_B = {q: [] for q in B.Q}

    for (_,j), q in A.delta.items():
        signature_A[q].append(chr(j+97))

    for (_,j), q in B.delta.items():
        signature_B[q].append(chr(j+97))

    print(f"Len A.Q : {len(A.Q)}, Len signA : {len(signature_A)}")
    signature_A = {q: "".join(sorted(j_list)) for q, j_list in signature_A.items()}
    print(f"Len B.Q : {len(B.Q)}, Len signB : {len(signature_B)}")
    signature_B = {q: "".join(sorted(j_list)) for q, j_list in signature_B.items()}
    
    for qA, sigA in signature_A.items():
        print(f"Etat {qA} de A : signature {sigA}")
    for qB, sigB in signature_B.items():
        print(f"Etat {qB} de B : signature {sigB}")
        
    visited = []
    for q in A.Q:
        if q not in visited and signature_A[q] in signature_B.values():
            recon[q].extend([qB for qB, sigB in signature_B.items() if sigB == signature_A[q]])
            if len(recon[q]) == 1: #un seul point d'entrée possibles
                queue = [q]
                while queue:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.append(current)
                        for l in A.Sigma:
                            qA_next = A.delta[(current, l)]
                            qB_next = B.delta[(recon[current][0], l)]
                            #qB_next = B.delta[(recon[current][0], ord(l)-97)]
                            recon[qA_next].append(qB_next)
                            print(f"{recon[current][0]} -- {l} --> {qB_next} : B({qB_next}) est un candidat pour état {qA_next} de A ({current} -- {l} --> {qA_next})")
                            if qA_next not in visited:
                                queue.append(qA_next)

    print("Reconnaissance :")
    for qA, qB_list in recon.items():
        print(f"Etat {qA} de A : recon {qB_list}")

    #divergence
    score = 0

    for qA, qB_list in recon.items():
        for qB in set(qB_list):
            for q, list in recon.items():
                if q != qA and qB in list: #divergence
                    proba1 = qB_list.count(qB) / len(qB_list)
                    proba2 = list.count(qB) / len(list)
                    if proba1 > proba2:
                        list.remove(qB)
                        #rectifier l'arête 
                    else:
                        qB_list.remove(qB)
                        #rectifier l'arête
                    score += 1

    #merge
    set_recon = {qA: set(qB_list) for qA, qB_list in recon.items()}
    print("Set Reconnaissance :", set_recon)
    for qA, qB_set in set_recon.items():
        if len(qB_set) > 1:
            ref = qB_set.pop()
            ref_neighborhood = {l: B.delta[(ref, l)] for l in B.Sigma}
            for qB in qB_set:
                candidate_neighborhood = {l: B.delta[(qB, l)] for l in B.Sigma}
                if ref_neighborhood == candidate_neighborhood:
                    score += 1
                    print(f"Fusion : B({qB}) dans B({ref})")
                    #Fusionner les états dans B



    #remove
    used_B = set()
    for _, qB_list in recon.items():
        for qB in qB_list:
            used_B.add(qB)

    for qB in B.Q:
        if qB not in used_B:
            score += 1
            print(f"Suppression : B({qB})")
    print(f"Distance : {score}")



def ged_nx(A, B):
    #Env 5min pour 11 noeuds
    G_A = nx.DiGraph()
    G_B = nx.DiGraph()

    for q in A.Q:
        G_A.add_node(q, final=q in A.F)
    for (q, a), q_next in A.delta.items():
        G_A.add_edge(q, q_next, label=a)

    for q in B.Q:
        G_B.add_node(q, final=q in B.F)
    for (q, a), q_next in B.delta.items():
        G_B.add_edge(q, q_next, label=a)

    print("Calcul de la distance d'édition avec NetworkX...")
    paths, score = nx.optimal_edit_paths(G_B, G_A)


    chemin_optimal = paths[0]
    node_edits = chemin_optimal[0]
    edge_edits = chemin_optimal[1]

    print("--- CHANGEMENTS SUR LES NOEUDS ---")
    for g1_node, g2_node in node_edits:
        if g1_node is None:
            print(f"[+] Ajout du noeud : {g2_node}")
        elif g2_node is None:
            print(f"[x] Suppression du noeud : {g1_node}")
        else:
            if g1_node != g2_node:
                print(f" [!] Substitution : {g1_node} -> {g2_node}")
            else:
                print(f" [.] Noeud conservé : {g1_node}")

    print("\n--- CHANGEMENTS SUR LES ARÊTES ---")
    for g1_edge, g2_edge in edge_edits:
        if g1_edge is None:
            print(f"[+] Ajout de l'arête : {g2_edge}")
        elif g2_edge is None:
            print(f"[x] Suppression de l'arête : {g1_edge}")
        else:
            if g1_edge != g2_edge:
                print(f" [!] Substitution de l'arête : {g1_edge} -> {g2_edge}")
            else:
                print(f" [.] Arête conservée : {g1_edge}")
    return score




if __name__ == "__main__":
    """ with open("test/TOY_multi_3_100.pkl", "rb") as f:
        A = pickle.load(f)
    test_isomorphisme(A, "test/iso") """

    
    a_path = "test/TOY_safe_1000.pkl"
    b_path = "test/TOY_etat_safe_100.pkl"
    
    print(f"A : {a_path.split('/')[-1].split('.')[0]}")
    print(f"B : {b_path.split('/')[-1].split('.')[0]}")

    with open(a_path, "rb") as f:
        A = pickle.load(f)

    with open(b_path, "rb") as f:
        B = pickle.load(f)

    A.minimize()
    B.minimize()

    """ #retire -1 de B
    B.Q = B.Q-{-1}
    for s in B.Sigma:
        B.delta.pop((-1, s), None) """

    iso, trad = is_isomorphic(A, B, True)
    print(f"Isomorphe : {iso}")
    if trad != {}:
        print("\nTraduction :")
        print("  A | B \n ---|---")
        for k,v in trad.items():
            print(f"{k:>3} | {v:<3}")
