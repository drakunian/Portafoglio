from functools import partial
# import pyarrow
import numpy as np
import pandas as pd
from multiprocessing import Pool, freeze_support
from numpy import ndarray


class Nodo:
    """
    A quanto sembra, ci saranno grossi problemi di memoria RAM. Dobbiamo ridurre al minimo l'occupazione di spazio da
    parte dei nodi. dunque, tutti i valori pesanti (come gli asset returns dataframe ecc...) di ogni nodo devono essere
    eliminati una volta passati ai nodi figli, così che, alla fine, solo i nodi foglia avranno tutti i dati completi,
    mentre i nodi precedenti avranno stoccate solo dataframe degli asset, contenente valori specifici del periodo,
    le probabilità ed il vettore decisionale

    L'OBIETTIVO E' QUINDI OTTIMIZZARE L'USO DELLA RAM, CERCARE DI SPOSTARE I PROCESSI NEL CLOUD ED OCCUPARE QUANTA MENO
    MEMORIA POSSIBILE
    """
    def __init__(self, root: bool, valore, parent, horizon=12):
        self.root = root
        self.valore = valore
        self.horizon = horizon
        self.parent = parent
        self.matrix_list = []
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

    def __str__(self):
        return f"{self.valore}"

    def __repr__(self):
        return f"{self.valore}"

    def getValore(self):
        return self.valore

    def metodo(self):
        return self.valore + self.valore

    def sibling_nodes(self, parent, optimization_func: callable = None, matrix_cols=None):
        """
        qui passiamo come input i dati del parent, come istanza:
        parent = matrice.iloc[row, col].values[0]
        poi, procediamo a fare processo 2) [Nodo(False, self.metodo(), matrice.iloc[row, col]) for _ in matrix.columns]
        calcoliamo poi le probabilità come da processo 3) prob_measure_function (DA CREARE)
        teriminiamo ritornando la lista aggiornata 4) [completed_sibling_nodes]

        per funzione di ottimizzazione delle probabilità, di seguito si alla uno schema per la definizione dei constraint:
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # sum of probabilities == 1
        for node in nodes:
            for i in assets:
                {"type": "eq", "fun": lambda x: sum(r_i_s * x) + m1_i_plus - m1_i_minus - a_i}
                {"type": "eq", "fun": lambda x: sum(((r_i_s - a_i)**2) * x) + m2_i_plus - m2_i_minus - sigma_2_i}
                {"type": "eq", "fun": lambda x: sum(((r_i_s - a_i)**3) * x) + m3_i_plus - m3_i_minus - moment_3_i}
                {"type": "eq", "fun": lambda x: sum(((r_i_s - a_i)**4) * x) + m4_i_plus - m4_i_minus - moment_4_i}
            # poi, matrice diagonale, per le covarianze:
            # i valori per così come è scritto il codice si duplicano, possiamo calcolare la metà dei constraint, dobbiamo caipre come
            for i in assets:
                for l in assets:
                    if i != l:
                        {"type": "eq", "fun": lambda x: sum((r_i_s - a_i)*(r_l_s - a_l) * x) - covariance_i_l}
        dove sigma_2_i, covariance_i_l e i momenti sono presi, rispettivamente:
            forecast nodo parent
            albero (definiti costanti)
            albero (definiti costanti)
            forecast nodo parent

        dove la funzione obiettivo sarà:
        assets_data = []
        for i in assets:
            moment_data_i = []
            for moment in moments_i:
                moment_data_i.append(moment_weight_i * (mk_i_plus + mk_i_minus))
            cov_data_i = []
            for weight_cov in weight_covs_il:
                cov_data_i.append(weight_cov * (cil_plus + cil_minus))
            moment_sum_i = sum(moment_data)
            cov_sum_i = sum(cov_data_i)
            assets_data.append(moment_sum_i + cov_sum_i)
        min(sum(assets_data))

        dove, i pesi per ciascun moemnto e per ciascun fattore di covarianza sono definiti,
        per ogni asset, all'inizializzazione dell'albero.
        Serve un modo per connettere failmente ed efficientemente questi valori ai valori dei momenti stessi
        di ciascun asset...

        Per la prossima settimana, dobbiamo aver finito di lavorare a questa parte ed averla integrata definitivamente
        """
        """
        1 -> parent
        2 -> nodi_fratelli= [Nodo(False, self.metodo(), parent) for _ in matrix_cols]
        3 -> optimization_func(dati_del_parent, nodi_fratelli, dati_dell'albero) -> [lista nodi aggiornati]
        """
        return [Nodo(False, self.metodo(), parent) for _ in matrix_cols]

    def generateSonMultithreadedHybrid(self, matrice, contatore):
        if contatore <= self.horizon:
            print(matrice.size)
            print('contatore: ', contatore)
            if matrice.size == 1:
                row, col = 0, 0
                print("10 figli")
                matrix = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
                matrix.loc[len(matrix)] = [
                    Nodo(False, self.metodo(), matrice.iloc[row, col]) for _ in matrix.columns
                ]
            else:
                print("3 figli")
                parents = matrice.to_numpy().flatten()
                matrix_cols = ['0', '1', '2']
                if contatore <= 6:
                    matrix = pd.DataFrame(columns=matrix_cols)
                    for parent in parents:
                        matrix.loc[len(matrix)] = [Nodo(False, self.metodo(), parent) for _ in matrix.columns]
                else:
                    threads = 3 if contatore < 13 else 4
                    print('with threads: ', threads)
                    with Pool(threads) as po:
                        mapped = po.map(partial(
                            # funzione da iterare,             parametri opzionali
                            self.sibling_nodes, optimization_func=None, matrix_cols=matrix_cols
                            # lista di parametri principale
                        ), parents)
                    matrix = pd.DataFrame(mapped)

            print(matrix)
            print(matrix.info())
            # IN POSSIBILI AGGIORNAMENTI FUTURI, SI DOVRA' GESTIRE L'ENORME IMPIEGO DI MEMORIA!
            # matrix.to_parquet(f'matrix_period{contatore}.parquet')
            # self.matrix_list.append(matrix)  # CREA GROSSI PROBLEMI DI RAM STOCCARE LE MATRICI. VANNO SALVATE DIRETTAMENTE IN FILE, NON STOCCATE!
            self.generateSonMultithreadedHybrid(matrix, contatore + 1)

