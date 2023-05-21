import json
import math
import os
import pprint
import random
import re
import timeit
from cmath import pi, log
from multiprocessing import Pool, freeze_support
import numpy as np
import pandas as pd
import itertools
from functools import partial
import scipy.stats as stats
from arch import arch_model
from dateutil.relativedelta import relativedelta
# from docplex.mp.model import Model

# import pyarrow.parquet as pq

from Nodo import Nodo, ScenarioNode, egarch_formula
from dbconn_copy import DBConnection, DBOperations

# from dbconn_copy import DBConnection, DBOperations

pd.options.display.float_format = "{:,.6f}".format


def Load_file_json(assets, cash):
    # 06_ scrittura su file json
    '''results = pd.DataFrame(results)
    results.set_axis(["stock_id"], axis = "columns")
    lista_assets = list(results["stock_id"])   '''         #DOVREI USARE QUESTA RIGA E NON QUELLA DOPO
    #lista_assets = list(df_assets["stock_id"])
   # print(lista_assets)
    frase = ''
    with open("assets.json", "w") as f:
        inizio = '{'
        for i in range(len(assets)):
            stringa = '"' + assets[i] + '": [{"weight":'+ str(0) + ', "n_assets":'+ str(0) + '}], \n'
            frase = frase + stringa


        str_cash = '"cash": {"weight":'+ str(1) + ', "n_assets":'+ str(cash) + '} \n'
        fine = '}'
        frase = inizio + frase + str_cash + fine
        print(frase)
        json_object = json.loads(frase)
        json.dump(json_object, f, indent=2)


class NewTree:
    """
    class-specific stats are good. measurement are done in 0.11382 seconds. All must be put into effort for root node.
    """
    def __init__(
            self,
            assets_df: pd.DataFrame,
            assets_returns_data: pd.DataFrame,
            horizon=12,
            cash_return=.01,
            period='1month',
            cash_currency='EUR'
    ):  # sono variabili prese da input
        self.assets = assets_df
        self.assets.loc['cash', 'currency'] = cash_currency

        self.period = period
        self.horizon = horizon  # Number of time periods (e.g. 12 months)
        self.cash_return = cash_return  # Return on cash asset. Not the rf rate, but the return on YOUR cash asset...

        self.returns_data = assets_returns_data
        self.corr_matrix = self.returns_data.corr()  # Constant correlation matrix
        print(self.corr_matrix)
        self.assets[['omega', 'alpha[1]', 'gamma[1]', 'beta[1]', 'sigma_t']] = self.compute_egarch_params()  # function
        self.compute_moments()
        self.moments_weights = self.compute_moments_weight()
        print('initial assets data: ')
        print(self.assets)
        # print(self.returns_data)
        # End of inputs and starters definition...
        # now, set class parameters of ScenarioNode:
        self.ret_list = [self.returns_data[column].dropna() for column in self.returns_data.columns]

        self.root_node = ScenarioNode(
            root=True, parent=self.assets.dropna(), returns=self.ret_list, cor_matrix=self.corr_matrix
        )
        print('assets data of root node:')
        print(self.root_node.assets_data)
        self.periods = []

    def compute_egarch_params(self) -> pd.DataFrame:
        eam_params_list = []
        for column in self.returns_data:
            rets = self.returns_data[column].dropna() * 100
            eam = arch_model(rets, p=1, q=1, o=1, mean='constant', power=2.0, vol='EGARCH', dist='normal')  # --> ??????????

            eam_fit = eam.fit(disp='off')

            eam_params = eam_fit.params
            last_vol = eam_fit.conditional_volatility.tail(1).values[0]
            # now adjust if value is incredibly high:
            # if last_vol > 1000:
            #     last_vol = last_vol / 1000
            eam_params['sigma_t'] = last_vol / 100  # scaling sigma to decimals

            eam_params.name = column
            eam_params_list.append(eam_params)

        df = pd.concat(eam_params_list, axis=1).T
        df.index.name = 'stock_id'
        self.assets['a_i'] = df['mu'] / 100

        return df[['omega', 'alpha[1]', 'gamma[1]', 'beta[1]', 'sigma_t']]

    def compute_moments(self):
        for i, row in self.assets.dropna().iterrows():
            self.assets.loc[i, 'third_moment'] = stats.moment(
                self.returns_data[i].dropna(), moment=3, nan_policy='propagate'
            )  # skewness
            self.assets.loc[i, 'fourth_moment'] = stats.moment(
                self.returns_data[i].dropna(), moment=4, nan_policy='propagate'
            )  # kurtosis
            # .loc --> used to access a group of rows and columns by label(s) or a boolean array

    def compute_moments_weight(self):
        """
        using principal component analysis (PCA), gets the weights for each moment deviation, same for covariances.

        For now, be naive, 1/4 for each weight in moments so you have them directly in formula.
        same for cov factors weight
        """
        moment_weights = pd.DataFrame(columns=['w1', 'w2', 'w3', 'w4'], index=self.assets.index).fillna(1/4)
        # for node in nodes:
        #     for i in assets:
        #         {"type": "eq", "fun": lambda x: sum(r_i_s * x) + m1_i_plus - m1_i_minus - a_i}
        #         {"type": "eq", "fun": lambda x: sum(((r_i_s - a_i)**2) * x) + m2_i_plus - m2_i_minus - sigma_2_i}
        #         {"type": "eq", "fun": lambda x: sum(((r_i_s - a_i)**3) * x) + m3_i_plus - m3_i_minus - moment_3_i}
        #         {"type": "eq", "fun": lambda x: sum(((r_i_s - a_i)**4) * x) + m4_i_plus - m4_i_minus - moment_4_i}
        #     # poi, matrice diagonale, per le covarianze:
        #     # i valori per così come è scritto il codice si duplicano, possiamo calcolare la metà dei constraint, dobbiamo caipre come
        #     for i in assets:
        #         for l in assets:
        #             if i != l:
        #                 {"type": "eq", "fun": lambda x: sum((r_i_s - a_i)*(r_l_s - a_l) * x) - covariance_i_l}
        # assets_data = []
        # for i in assets:
        #     moment_data_i = []
        #     for moment in moments_i:
        #         moment_data_i.append(moment_weight_i * (mk_i_plus + mk_i_minus))
        #     cov_data_i = []
        #     for weight_cov in weight_covs_il:
        #         cov_data_i.append(weight_cov * (cil_plus + cil_minus))
        # min(sum(assets_data))
        return moment_weights

    def target_function(self, list_of_probabilities):
        value = 'stuff computed using other intermediate functions'
        return value

    def optimization_func(self, dati_del_parent, nodi_fratelli, dati_dell_albero):
        list_of_probabilities = ['dummy startuop list']
        optimized = 'executes cplex code on target_function passing list of probabilities as x'
        return list_of_probabilities

    def sibling_nodes(self, parent, optimization_func: callable = None, matrix_cols=None, date=None) -> list:
        """
        qui passiamo come input i dati del parent, come istanza:
        parent = matrice.iloc[row, col].values[0]
        poi, procediamo a fare processo 2) [Nodo(False, self.metodo(), matrice.iloc[row, col]) for _ in matrix.columns]
        calcoliamo poi le probabilità come da processo 3) prob_measure_function (DA CREARE)
        teriminiamo ritornando la lista aggiornata 4) [completed_sibling_nodes]

        per funzione di ottimizzazione delle probabilità, di seguito si alla uno schema per la definizione dei constraint:
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # sum of probabilities == 1
        x > LB per ogni nodo, la probabilità deve essere maggiore di un certo livello...
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
        per la funzione di ottimizzazione, i valori mk_i sono funzioni delle probabilità. variano al variare delle 
        probabilità, dato appunto dal constraint.
        sarebbe utile scrivere una serie di funzio ni del tipo:
        self.match_mean(x) - a_i, dove x è la lista di probabilità e la funzione somiglia a:
        sum(r_i_s * x) + m1_i_plus - m1_i_minus, ma mk_p e mk_m come li gestiamo?

        mk è impiegato nel calcolo delle deviazioni di cosa? ebbene,
        il set di probabilità è unico, per tutti i constraint, dunque, esisterà una deviazione tra:
        sum(val_s*prob_s) e exp_val, per ogni val. le deviazioni sono quindi il risultato delle probabilità formulate, 
        tali per cui, desideriamo che la sommatoria di tutte le deviazioni dai valori teorici sia minimizzata.
        la funzione obiettivo prende probabilità e ritorna gli obiettivi. la funzione dunque ha una fase intermedia di 
        output delle deviazioni e poi le somma nella effettiva funzione obiettivo...
        Puoi strutturare in questo modo:
        ogni asset prende le probabilità, procede ad outputtare le proprie deviazioni fornendole all'effettiva funzione
        obiettivo...
        """
        return [ScenarioNode(False, parent, self.ret_list, self.corr_matrix, period_date=date) for _ in matrix_cols]

    def generate_tree(self, init_matrix):
        """
        PER ORA, GENERARE 8 PERIODI IN QUESTO MODO RICHIEDE: 5.0 minuti e 37 secondi UTILIZZANDO TUTTI I CORES.
        NON MALE, MA PER GENERARNE 12 DIVENTA ABBASTANZA LENTO.
        COME RISOLVERE? CI SONO MOMENTI IN CUI LA CPU NON E' USATA AL 100%. DOBBIAMO IDENTIFICARE COSA VIENE FATTO IN
        QUEI MOMENTI E FAR SI CHE VENGA USATA APPIENO ANCHE LI
        """
        init_matrix = init_matrix
        counter = 0
        while counter <= self.horizon:
            print(init_matrix.size)
            print('contatore: ', counter)
            if init_matrix.size == 1:
                row, col = 0, 0
                print("10 figli")
                matrix = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
                matrix.loc[len(matrix)] = self.sibling_nodes(
                    init_matrix.iloc[row, col], optimization_func=None, matrix_cols=matrix.columns, date=counter
                )
            else:
                print("3 figli")
                parents = init_matrix.to_numpy().flatten()
                matrix_cols = ['0', '1', '2']
                if counter < 5:
                    matrix = pd.DataFrame(columns=matrix_cols)
                    for parent in parents:
                        matrix.loc[len(matrix)] = self.sibling_nodes(
                            parent, optimization_func=None, matrix_cols=matrix_cols, date=counter
                        )
                else:
                    with Pool() as po:
                        mapped = po.map(partial(
                            self.sibling_nodes,
                            optimization_func=None,
                            matrix_cols=matrix_cols,
                            date=counter
                        ), parents)
                    matrix = pd.DataFrame(mapped)

            init_matrix = matrix  # hide it if return to old way...
            print(matrix)
            print(f'example at time: {counter}')
            print(matrix.loc[0].head(1).values[0].assets_data)
            self.periods.append(counter)
            counter += 1

    def test_node(self):
        init_matrix = pd.DataFrame({self.root_node})
        # save initial matrix somewhere like a parquet file, just make sure that it saves entire instances in the cells
        self.generate_tree(init_matrix)
        print(self.periods)

# %%


def main(): #variabili prese da input
    '''horizon = 24  # default
    while True:
        time_division = int(input("Inserisci la divisione del tempo, (1) settimanale (2) mensile (3) annuale: \t"))
        horizon = int(input("Inserisci l'orizzonte: \t"))
        if ((time_division == 1 and horizon < 104) or (time_division == 2 and horizon < 24) or (
                time_division == 3 and horizon < 2)):
            break

    lb, ub = 0, 1 #di default
    while True:
        lb = input("Inserisci il limite minimo dei pesi (compreso tra 0 e 1): \t")
        ub = input("Inserisci il limite massimo dei pesi (compreso tra 0 e 1): \t")
        if lb > 0 and ub < 1 and lb < ub:
            break


    risk = input("Inserisci quanto sei disposto a rischiare in valore decimale")

    cash_return = float(input("Inserisci il cash return"))
    alpha = 0.05  # livello di confidenza per la misurazione del VaR

    LCVar =  # è un multiplo  di VaR_lis (obiettivi)

    annuaties = None # dataframe, tante righe quante sono i periodi, da prendere in input, di default rimane None
    cf_list = None # da prendere in input una lista lunga quanto i periodi, altrimenti rimane None'''


# ------------------------------------------------------------------------------------------------------------------------

    # 03_chiedo quali stock_id vuole usare
    '''print("Inserisci solo gli indici degli asset che vuoi usare, separati da uno INVIO. Quando hai finito digita STOP.")
    i = 0
    assets_list = []
    while True:
        assets_input = str(input("" + str(i + 1) + " --> "))
        if (assets_input.upper() == "STOP" or assets_input.isalpha()):
            break
        assets_list.append(int(assets_input))
        i += 1

        # chiedo input di cash e lo aggiungo alla assets_list

    while True:
        cash = int(input("Dimmi quanti soldi hai: \t"))
        if cash > 0:
            break

    print(assets_list)
    print(len(assets_list))'''

# ------------------------------------------------------------------------------------------------------------------------

    #prendo i dati dal file data.txt
    data_list = []
    data = open("data.txt", "r")
    for row in data:
        row = data.readline()
        if "/" not in row:
            row = re.sub('[\n]', '', row) # per eliminare i \n alla fine di ogni riga
            data_list.append(row)
            data.readline()   # serve per evitare le righe vuote (non è bellissimo, ma funziona)

    print(data_list)

    horizon, lb, ub, risk, cash_return, alpha, VaR_list, LCVar, annuaties, cf_list, assets_list, cash= data_list[0], data_list[1], data_list[2], data_list[3], data_list[4], data_list[5], data_list[6], data_list[7], data_list[8], data_list[9], data_list[10], data_list[11]
    #print(horizon, lb, ub, risk, cash_return, alpha, VaR_list, LCVar, annuaties, cf_list, assets_list, cash)
    assets_list, VaR_list, LCVar = tuple(assets_list.split(',')), VaR_list.split(','), LCVar.split(',')  # per trasformare le stringhe in liste
    print(type(assets_list))
    print(assets_list)
# ------------------------------------------------------------------------------------------------------------------------
    #collegamento con la query multi_fetch_where
    '''dc = DBConnection()
    do = DBOperations(dc.conn)
    const1 = [f"stock_id IN {assets_list}"]
    const2 = [f"etf_id IN {assets_list}"]
    sd = [['stock_id', 'symbol', 'exchange', 'currency'], 'stock_data', const1]  #sono le *execution della query. Lista delle colonne, nome tabella, lista dei constraint per la query
    ed = [['etf_id', 'symbol', 'exchange', 'currency'], 'etf_data', const2]
    stock_df, etf_df = do.multi_fetch_where(sd, ed)   #chiamata della query su dbconn_copy.py
    #print(stock_df)
    #print(etf_df)
    etf_df = etf_df.rename(columns={'etf_id': 'stock_id'})  #rinominiamo la colonna perchè altrimenti la concatenazione viene sballata
    #print(etf_df)
    assets_df = pd.concat([stock_df, etf_df]).reset_index(drop=True)
    #print(assets_df)
    assets = assets_df['stock_id'].tolist()'''


    # il risultato della query è il dataframe assets_df

    #collegamento con la query multi_value_fetch
    dc = DBConnection()
    do = DBOperations(dc.conn)
    sd = [['stock_id', 'symbol'], 'stock_data']
    ed = [['etf_id', 'symbol'], 'etf_data']
    stock_df, etf_df = do.multi_value_fetch(sd, ed) # restituisce una lista, ma tutto il contenuto della query va a finire nella prima cella
    #stock_df = pd.DataFrame(stock_df[0], columns=['stock_id', 'symbol'])
    etf_df = etf_df.rename(columns={'etf_id':'stock_id'})  #rinominiamo la colonna perchè altrimenti la concatenazione viene sballata
    assets_df = pd.concat([stock_df, etf_df]).reset_index(drop=True)
    print(assets_df)

    # 03_chiedo quali stock_id vuole usare
    print("Inserisci solo gli stock_id degli asset che vuoi usare, separati da uno INVIO. Quando hai finito digita STOP.")
    i = 0
    assets_list = []
    while True:
        assets_input = str(input("" + str(i + 1) + " --> "))
        if (assets_input.upper() == "STOP" or len(assets_input) <= 1):
            break
        assets_list.append(assets_input)

        i += 1

    df_input = assets_df['stock_id'].isin(assets_list) #cerco nella colonna stock_id del dataframe i valori nella lista assets_list
    assets_df = assets_df[df_input]
    stock_df.reset_index(inplace=True, drop=True)
    #print(assets_df)
    assets_df = assets_df['stock_id'].values.tolist() #trasformo la colonna del dataframe in una lista per il json
    print(assets_df) #PROBLEMA DATO CHE GLI ETF SONO UGUALI AGLI ASSETS POTREBBE ESSERE CHE INSERISCO UNO STOCK_ID E TROVO 2 OUTPUT
# ------------------------------------------------------------------------------------------------------------------------

    # 05_controllare se esiste già un file json e riproporlo
    # https://www.scaler.com/topics/seek-function-in-python/
    with open('assets.json', 'r') as file:
        file.seek(0, os.SEEK_END)  # puntatore, si sposta di 0 lettere, partendo dal fondo
        isempty = file.tell() == 0  # returns the current file position in a file stream.
        file.seek(0)  # riavvolgere il file
        print(isempty)
        if isempty == False:
            with open('assets.json', 'r') as f:
                data = f.read()
                json_data = json.loads(data)
                pprint.pprint(
                    json_data)  # pprint --> fornisce la capacità di stampare la rappresentazione formattata dei dati JSON.
    
            risposta = str(input("Esiste un file contenente degli assets già utilizzati. Vuoi usarlo? \t"))
            if risposta.upper() == "NO":
                #faccio una prova con un lista preimpostata
                #results =['AAPL_NASDAQ_USD', 'AMZN_NASDAQ_USD', 'EUE_MTA_EUR', 'JPM_NYSE_USD', 'PRY_MTA_EUR', 'SXRV_XETR_EUR', 'UNIR_MTA_EUR']
                Load_file_json(assets_df, cash) #devo passargli gli assets_list ma per ora non funziona il database
        else:
            #results = ['AAPL_NASDAQ_USD', 'AMZN_NASDAQ_USD', 'EUE_MTA_EUR', 'JPM_NYSE_USD', 'PRY_MTA_EUR', 'SXRV_XETR_EUR', 'UNIR_MTA_EUR']
            Load_file_json(assets_df, cash)

# ------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    start = timeit.default_timer()
    """
    per velocizzare il codice per ora, si leggono ggli input direttamente dai file, poi, si sostituirà questa parte con
    la funzione di raccolta di input che hai scritto
    """

    ast_json = json.loads(open('assets.json', 'r').read())
    assets_df = pd.read_parquet('assets_df.parquet')
    print(assets_df)
    portfolio = pd.DataFrame(ast_json).T  # a sto punto, json file prende anche currency (?)
    assets_df = pd.concat([assets_df[['stock_id', 'currency']].set_index('stock_id'), portfolio], axis=1)
    current_assets_prices = pd.read_parquet('curr_assets_prices.parquet').set_index('stock_id')
    assets_df['close_prices_t'] = current_assets_prices['close'].astype('float64')

    ast_ret = pd.read_parquet('assets_returns.parquet').set_index('datetime')

    tree = NewTree(
        assets_df, ast_ret, horizon=8
    )
    tree.test_node()

    # nodoAlternativo = ScenarioNode(True, 1, None)
    # matrice = pd.DataFrame({nodoAlternativo})
    # print(matrice)
    # contatore = 0
    # nodoAlternativo.generateSonMultithreadedHybrid(matrice, contatore)
    # dataframe = pd.read_json("assets.json")
    stop = timeit.default_timer()
    minutes = (stop - start) / 60
    print('full minutes: ', minutes)
    seconds, minutes = math.modf(minutes)
    print(f'{minutes} minuti e {round(seconds * 60)} secondi')




    '''with open("assets.json", "r") as f:
            data = f.read()
            assets_json = json.loads(data) #assets_json è un dizionario
    #https://ibmdecisionoptimization.github.io/tutorials/html/Beyond_Linear_Programming.html
    #vado a guardare sul link dal punto --> In [21]

    print(type(assets_json))

    w = [4, 2, 5, 4, 5, 1, 3, 5]
    w_df = pd.Series(w)
    v = [10, 5, 18, 12, 15, 1, 2, 8]
    v_df = pd.Series(v)
    C = 15
    N = len(w)


    m = Model("knapsack")
    x = m.binary_var_list(N, name="x")
    m.add_constraint(sum(w[i] * x[i] for i in range(N)) <= C)
    obj_fn = sum(v[i] * x[i] for i in range(N))
    m.set_objective("max", obj_fn)

   # m.print_information()
    sol = m.solve()
    m.print_solution()
    if sol is None:
        print("Infeasible")

    stop = timeit.default_timer()
    print("tempo esecuzione: " + str(stop - start) + ' secondi')'''



# ------------------------------------------------------------------------------------------------------------------------

    #le chiave e i valori del file json sono contenuti in una lista, motivo: la stampa del dataframe del file json viene meglio
    # tutti gli input vanno in una funzione
    # tutte le variabili in input vanno messe in un file txt con dei valori (per evitare di scrivere ogni volta ogni singolo valore)
    #main() # o dalla funzione o dal file txt devo leggere le variabili
    #multi treading, multi processing, libreria arrow come enging di numpy al posto di pandas, libreria randomforest
# ------------------------------------------------------------------------------------------------------------------------

# creare un dataframe contenente la classe Nodo (l'indice delle righe sono le date dei periodi(mese-anno))
# iniziare a creare un albero con celle vuote
# l'albero parte da 10 nodi, ogni nodo ha 3 figli
# il nodo radice viene generato dall'albero
'''nuovo parametro: se il nodo è root genera 10, altrimenti 3
generare figli e calcolare probabilità prima che creino altri figli
posso creare tanti dataframe quanti sono i periodo: nelle righe ci sono il numero di figli (10 se è il root, 3 altrimenti), 
nelle colonne l'identificativo del parent'''
'''
    classe Nodo:
    radice (booleano)
    data del periodo (mese e anno)
    copia del dataframe del dataframe assets_df
    parent (classe)
    assets_retur(ereditato dai parent)
    residui_sui ritorni (ereditati dai parent)
    varianze 
    covarianze (calcolate da un metodo)
    probabilità del parent                  (CPLEX anche per l'ottimizzazione dei pesi)
    probabilità calcolata (del nodo)
    probabilità condizionata = probabi. nodo * prob. condizionata del parent
    lista dei flussi di cassa degli asset del periodo
    valore del cash_in e cash_out del periodo
    vettore dei ribilanciamenti
    metodo ricorsivo per la creazione dei figli (finisce quando arrivo all'ultimo periodo)'''