from functools import partial

import numpy as np
import pandas as pd
from multiprocessing import Pool, freeze_support
from numpy import ndarray


class Nodo:
    def __init__(self, root, valore, parent, horizon=3):
        self.root = root
        self.valore = valore
        self.horizon = horizon
        self.parent = parent
        # self.generateSon = self.generateSon()

    '''def __init__(self, root, date, assets_df, parent, assets_return, residui_ritorni, probability, flussi_cassa):
        self.root = root
        self.date = date
        self.assets_df = assets_df
        self.parent = parent
        self.assets_return = parent.getReturn
        self.residui_ritorni = parent.getResidui
        self.varianze = ...     #CALCOLO IN METODO
        self.covarianze = ...  #CALCOLO IN METODO
        self.parent_prob = parent.getProbability
        self.probability = ... #CALCOLO IN METODO
        self.prob_conditionata = probability * parent.getProbCondizionata #CALCOLO IN METODO
        self.flussi_cassa = flussi_cassa
        cash_in = ...
        cash_out = ...
        vett_ribilanciamenti = ...
        generateSon(root)'''

    def getRoot(self):
        return self.root

    def __str__(self):
        return f"{self.valore}"

    def __repr__(self):
        return f"{self.valore}"

    def getValore(self):
        return self.valore

    def metodo(self):
        return self.valore + self.valore

    def generateSon(self, matrice, contatore):
        if (contatore <= self.horizon):
            print(matrice.size)
            if matrice.size == 1:
                print("10 figli")
                # matrice = np.zeros((10,1), dtype=Nodo)
                matrix = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
                son_lis = []
                for i in range(10):
                    nodo = Nodo(False, self.metodo(), matrice.iloc[0])
                    son_lis.append(nodo)
                matrix.loc[len(matrix)] = son_lis
                # matrix.concat(son_lis, axis=0, ignore_index=True)
            else:
                print("3 figli")
                rows, columns = np.shape(matrice)
                matrix = pd.DataFrame(columns=['0', '1', '2'])
                # index=range(rows*columns)
                # matrix = np.zeros((3, 5), dtype=Nodo)

                for row in range(rows):
                    for column in range(columns):
                        son_lis = []
                        for i in range(3):
                            nodo = Nodo(False, self.metodo(), matrice.iloc[row, column])
                            son_lis.append(nodo)
                        matrix.loc[len(matrix)] = son_lis

                '''matrix[0,j] = Nodo(False, self.metodo, matrice[0,j])
                matrix[1, j] = Nodo(False, self.metodo, matrice[1,j])
                matrix[2, j] = Nodo(False, self.metodo, matrice[2,j])'''

            print(matrix)
            self.generateSon(matrix, contatore + 1)

            '''matrix = np.zeros((3,5), dtype=Nodo)
                for j in range(5):
                    for i in range(3):
                        matrix[i,j] = Nodo(False, self.metodo())
                        contatore = contatore + 1
                        self.generateSon(contatore)'''


class NodoAlternativo:
    def __init__(self, root: bool, valore, parent, horizon=12):
        self.root = root
        self.valore = valore
        self.horizon = horizon
        self.parent = parent
        # self.generateSon = self.generateSon()

    '''def __init__(self, root, date, assets_df, parent, assets_return, residui_ritorni, probability, flussi_cassa):
        self.root = root
        self.date = date
        self.assets_df = assets_df
        self.parent = parent
        self.assets_return = parent.getReturn
        self.residui_ritorni = parent.getResidui
        self.varianze = ...     #CALCOLO IN METODO
        self.covarianze = ...  #CALCOLO IN METODO
        self.parent_prob = parent.getProbability
        self.probability = ... #CALCOLO IN METODO
        self.prob_conditionata = probability * parent.getProbCondizionata #CALCOLO IN METODO
        self.flussi_cassa = flussi_cassa
        cash_in = ...
        cash_out = ...
        vett_ribilanciamenti = ...
        generateSon(root)'''

    def getRoot(self):
        return self.root

    def __str__(self):
        return f"{self.valore}"

    def __repr__(self):
        return f"{self.valore}"

    def getValore(self):
        return self.valore

    def metodo(self):
        return self.valore + self.valore

    def sibling_nodes(self, parent, matrix_cols=None):
        """
        qui passiamo come input i dati del parent, come istanza:
        parent = matrice.iloc[row, col].values[0]
        po, procediamo a fare processo 2) [Nodo(False, self.metodo(), matrice.iloc[row, col]) for _ in matrix.columns]
        calcoliamo poi le probabilità come da processo 3) prob_measure_function (DA CREARE)
        teriminiamo ritornando la lista aggiornata 4) [completed_sibling_nodes]
        """
        return [NodoAlternativo(False, self.metodo(), parent) for _ in matrix_cols]

    def generateSon(self, matrice, contatore):
        """
        Ho fatto un test con un orizzonte di 12, il tempo impiegato è di:
        """
        if contatore <= self.horizon:
            print(matrice.size)
            print('contatore: ', contatore)
            if matrice.size == 1:
                row, col = 0, 0
                # PUò ANDARE UGUALMENTE BENE ANCHE IL SEGUENTE CODICE PER IL PERIODO INIZIALE: in entrambi i casi, impiega 15 sec
                print("10 figli")
                # matrice = np.zeros((10,1), dtype=Nodo)
                matrix = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
                """
                CODICE FUTURO:
                ciascun nodo figlio deve prendere dal parent una serie di caratteristiche, e nella fase di creazione
                dei nodi, dovremo eseguire il calcolo delle probabilità. dunque, successivamente credo che il codice
                dovrà essere più di questo tipo:
                1) prendi i dati del parent: matrice.iloc[row, col].values[0] == istanza del parent
                2) genera i nodi passando i vari input, anche dep arent stesso: 
                        [Nodo(False, self.metodo(), matrice.iloc[row, col]) for _ in matrix.columns]
                3) prima di appendere alla matrice, però, dovremo calcolare le probabilità: 
                        prob_measure_function (DA CREARE)
                4) a questo punto, i dati aggiornat dei nodi fratelli si possono inserire nella matrice: 
                        matrix.loc[len(matrix)] = [completed_sibling_nodes]
                        
                Per velocizzare il processo, è bene cercare di creare in parallelo quanto meno i nodi fratelli, se non
                anche cercare, parallelamente, gli stessi nodi parent (quindi multiprocessare tutto o parte del multiplo
                for loop: for row, for col...
                Un'altra cosa, forse la funzione di creazione dei nodi va tirata fuori dalla classe Nodo. Questo perché 
                credo sarebbe molto più facile salvarele matrici nell'albero. Ora infatti le matrici sono pintate ma non 
                vengono salvate in memoria mi sembra. Se la teniamo in Nodo, non saprei come si potrebbe fare a spostare
                la matrice creata periodo per periodo nell'albero senza fermare il codice. Al massimo, si passa come parametro
                una lista comprensiva di matrici dove appendiamo la matrice di ciascun periodo, e se l'orizzonte è raggiunto
                si ritorna la lista completa. Si può provare quindi o in questo modo o delegando ad Albero, o se hai altre idee 
                sei libero di testarle.
                """
                matrix.loc[len(matrix)] = [NodoAlternativo(False, self.metodo(), matrice.iloc[row, col]) for _ in matrix.columns]
            else:
                print("3 figli")
                rows, columns = np.shape(matrice)
                matrix = pd.DataFrame(columns=['1', '2', '3'])
                print(rows, columns)
                # MULTI-TREDARE QUESTA PARTE POTREBBE RISULTAR EUTILE
                for row in range(rows):
                    for col in range(columns):
                        # vettorizzando, non si potrà più usare len(matrix) però, e piuttosto si dovrà dare indicazione precisa della riga di riferimento
                        matrix.loc[len(matrix)] = [Nodo(False, self.metodo(), matrice.iloc[row, col]) for _ in matrix.columns]
                '''matrix[0,j] = Nodo(False, self.metodo, matrice[0,j])
                matrix[1, j] = Nodo(False, self.metodo, matrice[1,j])
                matrix[2, j] = Nodo(False, self.metodo, matrice[2,j])'''

            print(matrix)
            self.generateSon(matrix, contatore + 1)

            '''matrix = np.zeros((3,5), dtype=Nodo)
                for j in range(5):
                    for i in range(3):
                        matrix[i,j] = Nodo(False, self.metodo())
                        contatore = contatore + 1
                        self.generateSon(contatore)'''

    def generateSonMultithreaded(self, matrice, contatore):
        """
        Ho fatto un test con un orizzonte di 12, il tempo impiegato è di:
        """
        if contatore <= self.horizon:
            print(matrice.size)
            print('contatore: ', contatore)
            if matrice.size == 1:
                row, col = 0, 0
                # PUò ANDARE UGUALMENTE BENE ANCHE IL SEGUENTE CODICE PER IL PERIODO INIZIALE: in entrambi i casi, impiega 15 sec
                print("10 figli")
                # matrice = np.zeros((10,1), dtype=Nodo)
                matrix = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
                """
                CODICE FUTURO:
                ciascun nodo figlio deve prendere dal parent una serie di caratteristiche, e nella fase di creazione
                dei nodi, dovremo eseguire il calcolo delle probabilità. dunque, successivamente credo che il codice
                dovrà essere più di questo tipo:
                1) prendi i dati del parent: matrice.iloc[row, col].values[0] == istanza del parent
                2) genera i nodi passando i vari input, anche dep arent stesso: 
                        [Nodo(False, self.metodo(), matrice.iloc[row, col]) for _ in matrix.columns]
                3) prima di appendere alla matrice, però, dovremo calcolare le probabilità: 
                        prob_measure_function (DA CREARE)
                4) a questo punto, i dati aggiornat dei nodi fratelli si possono inserire nella matrice: 
                        matrix.loc[len(matrix)] = [completed_sibling_nodes]
                Per velocizzare il processo, è bene cercare di creare in parallelo quanto meno i nodi fratelli, se non
                anche cercare, parallelamente, gli stessi nodi parent (quindi multiprocessare tutto o parte del multiplo
                for loop: for row, for col...
                Un'altra cosa, forse la funzione di creazione dei nodi va tirata fuori dalla classe Nodo. Questo perché 
                credo sarebbe molto più facile salvarele matrici nell'albero. Ora infatti le matrici sono pintate ma non 
                vengono salvate in memoria mi sembra. Se la teniamo in Nodo, non saprei come si potrebbe fare a spostare
                la matrice creata periodo per periodo nell'albero senza fermare il codice. Al massimo, si passa come parametro
                una lista comprensiva di matrici dove appendiamo la matrice di ciascun periodo, e se l'orizzonte è raggiunto
                si ritorna la lista completa. Si può provare quindi o in questo modo o delegando ad Albero, o se hai altre idee 
                sei libero di testarle.
                """
                matrix.loc[len(matrix)] = [NodoAlternativo(False, self.metodo(), matrice.iloc[row, col]) for _ in
                                           matrix.columns]
            else:
                print("3 figli")
                rows, columns = np.shape(matrice)
                matrix_cols = ['0', '1', '2']
                # esempio di threading:
                parents = []
                for row in range(rows):
                    for col in range(columns):
                        parents.append(matrice.iloc[row, col])
                # print('parents: ', parents)
                with Pool(3) as po:
                    mapped = po.map(partial(self.sibling_nodes, matrix_cols=matrix_cols), parents)
                # print(mapped)
                matrix = pd.DataFrame(mapped)
                '''matrix[0,j] = Nodo(False, self.metodo, matrice[0,j])
                matrix[1, j] = Nodo(False, self.metodo, matrice[1,j])
                matrix[2, j] = Nodo(False, self.metodo, matrice[2,j])'''
            print(matrix)
            self.generateSonMultithreaded(matrix, contatore + 1)
            '''matrix = np.zeros((3,5), dtype=Nodo)
                for j in range(5):
                    for i in range(3):
                        matrix[i,j] = Nodo(False, self.metodo())
                        contatore = contatore + 1
                        self.generateSon(contatore)'''