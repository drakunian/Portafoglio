import numpy as np
import pandas as pd
from numpy import ndarray


class Nodo:
    def __init__(self, root, valore, parent, horizon = 3):
        self.root = root
        self.valore = valore
        self.horizon = horizon
        self.parent = parent
        #self.generateSon = self.generateSon()

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
        if(contatore <= self.horizon):
            print(matrice.size)
            if matrice.size == 1:
                print("10 figli")
                #matrice = np.zeros((10,1), dtype=Nodo)
                matrix = pd.DataFrame(columns=['0','1','2','3','4','5','6','7','8','9'])
                son_lis = []
                for i in range(10):
                    nodo = Nodo(False, self.metodo(), matrice.iloc[0])
                    son_lis.append(nodo)
                matrix.loc[len(matrix)] = son_lis
                #matrix.concat(son_lis, axis=0, ignore_index=True)
            else:
                print("3 figli")
                rows, columns = np.shape(matrice)
                matrix = pd.DataFrame(columns=['0','1','2'])
                #index=range(rows*columns)
                #matrix = np.zeros((3, 5), dtype=Nodo)

                for row in range(rows):
                    for column in range(columns):
                        son_lis = []
                        for i in range(3):
                            nodo = Nodo(False, self.metodo(), matrice.iloc[row,column])
                            son_lis.append(nodo)
                        matrix.loc[len(matrix)] = son_lis



                '''matrix[0,j] = Nodo(False, self.metodo, matrice[0,j])
                matrix[1, j] = Nodo(False, self.metodo, matrice[1,j])
                matrix[2, j] = Nodo(False, self.metodo, matrice[2,j])'''

            print(matrix)
            self.generateSon(matrix, contatore+1)

            '''matrix = np.zeros((3,5), dtype=Nodo)
                for j in range(5):
                    for i in range(3):
                        matrix[i,j] = Nodo(False, self.metodo())
                        contatore = contatore + 1
                        self.generateSon(contatore)'''