import json
import math
import os
import pprint
import random
import re
import timeit
from cmath import pi, log
from multiprocessing import Pool, freeze_support

import cplex
import numpy as np
import pandas as pd
from docplex.mp.model import Model
import itertools
from functools import partial
import scipy.stats as stats
from arch import arch_model
from dateutil.relativedelta import relativedelta
# from docplex.mp.model import Model

# import pyarrow.parquet as pq

from Nodo import ScenarioNode, egarch_formula
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
        self.cash_data = self.assets.loc['cash']
        self.assets = self.assets.dropna()

        self.period = period
        self.horizon = horizon  # Number of time periods (e.g. 12 months)
        self.cash_return = cash_return  # Return on cash asset. Not the rf rate, but the return on YOUR cash asset...

        self.returns_data = assets_returns_data
        self.corr_matrix = self.returns_data.corr()  # Constant correlation matrix
        print(self.corr_matrix)
        self.assets[['omega', 'alpha[1]', 'gamma[1]', 'beta[1]', 'sigma_t']] = self.compute_egarch_params()  # function
        self.compute_moments()
        self.moments_weights = self.compute_moments_weight()
        self.ret_list = [self.returns_data[column].dropna() for column in self.returns_data.columns]
        print('initial assets data: ')
        print(self.assets)
        # now, set class parameters of ScenarioNode:
        self.tree = None

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

    def compute_moments_weight(self) -> pd.DataFrame:
        """
        using principal component analysis (PCA), gets weights for each moment deviation, same for covariances.

        For now, be naive, 1/4 for each weight in moments so you have them directly in formula.
        same for cov factors weight
        """
        moment_weights = pd.DataFrame(columns=['w1', 'w2', 'w3', 'w4'], index=self.assets.index).fillna(1/4)
        return moment_weights

    @staticmethod
    def map_covariance_deviations(probability, node):
        deviations = pd.DataFrame(columns=['first_term'])
        # for each stuff unique:
        # velocizza questo for loop
        for i, row in node.conditional_covariances.iterrows():
            l0 = node.conditional_covariances.loc[i, 'level_0']
            l1 = node.conditional_covariances.loc[i, 'level_1']
            deviations.loc[i, 'first_term'] = node.assets_data.loc[l0, 'residuals'] * node.assets_data.loc[l1, 'residuals'] * probability
        return deviations

    def compute_deviations(self, list_of_probabilities, parent, sibling_nodes=[]) -> pd.DataFrame:
        """
        usa questa funzione per generare le deviazioni, che fanno parte della funzione obiettivo. La quale a sua volta
        sarà passata a optimization_func dove verrà eseguito il codice cplex.

        SU QUESTA FUNZIONE CI LAVORERO' IO, ASSICURANDOMI DI FARTI AVERE EFFICIENTEMENTE I DATI DELLE DEVIAZIONI PER IL
        CALCOLO DELLA FUNZIONE OBIETTIVO
        """
        # print([node.assets_data for node in sibling_nodes])
        i = 0
        first_term_1 = []
        first_term_2 = []
        first_term_3 = []
        first_term_4 = []
        cov_dev_matrix = []
        # write this for loop in a better way! (try using map)
        for x in list_of_probabilities:
            used_node = sibling_nodes[i]
            #                                    initial[i] * final[i] for
            first_term_1.append(used_node.assets_data['returns_t'] * x)
            first_term_2.append(((used_node.assets_data['residuals'])**2)*x)
            first_term_3.append(((used_node.assets_data['residuals'])**3)*x)
            first_term_4.append(((used_node.assets_data['residuals'])**4)*x)
            # add deviations for the covariances!
            cov_dev_matrix.append(
                # make function that computes the values, indexing by level_0 and level_1
                self.map_covariance_deviations(x, used_node)
            )
            i += 1

        deviations = pd.concat([
            pd.concat(first_term_1, axis=1).sum(axis=1) - self.assets['a_i'],
            pd.concat(first_term_2, axis=1).sum(axis=1) - parent.assets_data['sigma_t']**2,
            pd.concat(first_term_3, axis=1).sum(axis=1) - self.assets['third_moment'],
            pd.concat(first_term_4, axis=1).sum(axis=1) - self.assets['fourth_moment']
        ], axis=1)
        cov_dev_df = pd.concat(cov_dev_matrix, axis=1).sum(axis=1) - parent.conditional_covariances[0]
        # then compute cov_deviations and return a full dataframe
        return abs(deviations), abs(cov_dev_df)

    def target_function(self, list_of_probabilities: list, parent, sibling_nodes=[]):
        """
        la funzione precedente calcola le deviazioi, qui, hai la funzione obiettivo, che sommatutte le deviazioni pesate
        di ciascun asset.
        IT MUST READ CONSTRAINT IN A WAY THAT IS READABLE BY CPLEX...
        """
        deviations_dataframe, cov_deviations_dataframe = self.compute_deviations(
            list_of_probabilities, parent=parent, sibling_nodes=sibling_nodes
        )
        # moments_value = deviations_dataframe.sum().sum()
        # cov_value = cov_deviations_dataframe.sum()
        return deviations_dataframe.sum().sum() + cov_deviations_dataframe.sum()

    def optimization_func(self, parent, sibling_nodes: list):
        """
        Alla fine, i constraint per il problema cplex sono i seguenti:
            posto -> x = [probabilità_nodo_s per ogni nod in sibling_nodes]
            sum(x) == 1  -> assicurarsi che questo sia il formato adatto per cplex
            x >= LB per ogni x.  -> qui controlla se si può scrivere una cosa abbreviata così o bisogna fare un for loop
            LB si definisce come segue:
                LB = sensibility * 1 / len(x); 0 < sensibility < 1
                dove -> sensibility == .99 per ora, ma potrà essere modificata in futuro

        BISOGNA CONTROLLARE CHE COME SONO SCRITTI SOPRA, I CONSTRAINT SONO IN FORMATO CORRETTO PER CPLEX
        Poi, la questione è la seguente:
            La funzione obiettivo di cplex, ossia il parametro che l'ottimizzatore cplex prende, di che tipo è?
            Il tutto per ora è stato strutturato come se cplex prenda una funzione come parametro, ma va verificato ed
            in caso andranno fuse le varie parti del poblema di ottimizzazione dentro questa funzione
        """
        # NOT GOOD, NOT EVEN FAST, FIND A WAY TO USE CPLEX
        n = len(sibling_nodes)
        list_of_probabilities = [1 / n for _ in sibling_nodes]  # 'dummy startup list'
        # if x is Continuous
        val = self.target_function(list_of_probabilities, parent, sibling_nodes)
        sensibility = 0.5
        LB = sensibility / n

        # m = Model("Optimization function")
        # list_of_probabilities = m.continuous_var_list(n, LB, 1.0)
        # print(list_of_probabilities)
        # m.add_constraint(sum(list_of_probabilities) == 1)
        # m.set_objective("min", self.target_function(list_of_probabilities, parent=parent, sibling_nodes=sibling_nodes))
        # m.print_information()
        # sol = m.solve()
        # print(sol)
        # constraints = ({"type": "eq", "fun": lambda x: sum(x) - 1})
        # bounds = tuple((LB, 1) for _ in range(n))
        # obj = sco.minimize(
        #     self.target_function, np.array(list_of_probabilities),
        #     args=(parent, sibling_nodes), method="SLSQP", bounds=bounds, constraints=constraints
        # )
        return list_of_probabilities  # obj.x

    def sibling_nodes(self, parent, optimization_func: callable = None, matrix_cols=None, date=None) -> list:
        """
        Qui noi chiamiamo la funzione: self.optimization_func, a cui passiamo parent e la lista di sibling_nodes appena
        creata.
        La funzione ritornerà in ordine la lista di probabilità, che verranno poi appese ai nodi fratelli attraverso un
        metodo che creerò in seguito, per ora va bene il for loop scritto sotto
        """
        # vediamo di velocizzare questo for loop
        sibling_nodes = [
            ScenarioNode(
                False, parent=parent, returns=self.ret_list, cor_matrix=self.corr_matrix, period_date=date
            ) for _ in matrix_cols
        ]
        prob_list = optimization_func(parent, sibling_nodes)
        i = 0
        for node in sibling_nodes:
            node.probability = prob_list[i]  # make method to update probability...
            # sibling_nodes[i] = node.to_dict()
            i += 1
        return sibling_nodes

    @staticmethod
    def dictionarize(x):
        return x.apply(lambda y: y.go_to_dict())

    @staticmethod
    def find_coordinates(x):
        for i in range(len(x)):
            x[i].coordinates = [x.name, i]

    def generate_tree(self, init_matrix):
        """
        PER ORA, GENERARE 8 PERIODI IN QUESTO MODO RICHIEDE: 2.0 minuti e 37 secondi UTILIZZANDO TUTTI I CORES.
        IL MASSIMO PRIMA DEL CRASH CON 16 GB RAM E': 9 nodi, in 7 minuti

        CI SONO MOMENTI IN CUI LA CPU NON E' USATA AL 100%. DOBBIAMO IDENTIFICARE COSA VIENE FATTO IN QUEI MOMENTI E FAR
        SI CHE VENGA USATA APPIENO ANCHE LI
        """
        init_matrix = init_matrix
        counter = 0
        while counter <= self.horizon:
            print(init_matrix.size)
            print('contatore: ', counter)
            if init_matrix.size == 1:
                row, col = 0, 0
                root = init_matrix.iloc[row, col]
                print("10 figli")
                matrix = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
                matrix.loc[len(matrix)] = self.sibling_nodes(
                    root, optimization_func=self.optimization_func, matrix_cols=matrix.columns, date=counter
                )
            else:
                print("3 figli")
                parents = init_matrix.to_numpy().flatten()
                matrix_cols = ['0', '1', '2']
                if counter < 3:
                    matrix = pd.DataFrame(columns=matrix_cols)
                    for parent in parents:
                        matrix.loc[len(matrix)] = self.sibling_nodes(
                            parent=parent, optimization_func=self.optimization_func, matrix_cols=matrix_cols, date=counter
                        )
                else:
                    with Pool() as po:
                        mapped = po.map(partial(
                            self.sibling_nodes,
                            optimization_func=self.optimization_func,
                            matrix_cols=matrix_cols,
                            date=counter
                        ), parents)
                    matrix = pd.DataFrame(mapped)
            matrix.apply(lambda x: self.find_coordinates(x), axis=1)
            init_matrix = matrix  # hide it if return to old way...
            # print(matrix)
            print(f'example at time: {counter}')
            print(matrix.loc[0].head(1).values[0].assets_data)
            # replace_matrix with json data here and then create parquet file!
            matrix = matrix.apply(lambda x: self.dictionarize(x), axis=1)
            # print('new matrix: ')
            # print(matrix)
            matrix.to_parquet(f'period_{counter}')
            # table = pa.Table.from_pandas(matrix)
            # pq.write_table(table, f'period_{counter}.parquet')
            counter += 1

    def test_node(self):
        root_node = ScenarioNode(
            root=True, parent=self.assets, returns=self.ret_list, cor_matrix=self.corr_matrix
        )
        print('assets data of root node:')
        print(root_node.assets_data)
        init_matrix = pd.DataFrame({root_node})
        # save initial matrix somewhere like a parquet file, just make sure that it saves entire instances in the cells
        self.generate_tree(init_matrix)

    @staticmethod
    def convert_to_node(x):
        for i in range(len(x)):
            x[i] = ScenarioNode(init_data=x[i])
        # return x.apply(lambda y: ScenarioNode(init_data=y))

    def read_tree(self):
        """
        qui si riconvertono le celle da liste a ScenarioNode. Poi, si svilupperà la logica di iterazione lungo l'albero
        """
        self.tree = [pd.read_parquet(f'period_{x}') for x in range(self.horizon + 1)]  # read parquet of all periods
        for matrix in self.tree:
            matrix.apply(lambda x: self.convert_to_node(x), axis=1)
        print(self.tree[-1].loc[0, 0].conditional_volatilities)

    def clear(self):
        # deletes the tree parquet files
        pass

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
    #print(assets_df)
    portfolio = pd.DataFrame(ast_json).T  # a sto punto, json file prende anche currency (?)
    assets_df = pd.concat([assets_df[['stock_id', 'currency']].set_index('stock_id'), portfolio], axis=1)
    current_assets_prices = pd.read_parquet('curr_assets_prices.parquet').set_index('stock_id')
    assets_df['close_prices_t'] = current_assets_prices['close'].astype('float64')

    ast_ret = pd.read_parquet('asset_returns.parquet')
    print(ast_ret)
    #ast_ret.set_index(['datetime'])

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