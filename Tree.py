#!/usr/bin/python
# -*- coding: utf-8 -*-
import gc
import json
import math
import os
import pprint
import re
import timeit
from multiprocessing import Pool
import numpy as np
import pandas as pd
from functools import partial
import scipy.stats as stats
from arch import arch_model
import pyarrow as pa
import pyarrow.parquet as pq
from docplex.mp.model import Model
from copy import deepcopy
import warnings

import Iteration_tree

warnings.filterwarnings('ignore')

from Nodo import ScenarioNode, egarch_formula

from dbconn_copy import DBConnection, DBOperations

pd.options.display.float_format = "{:,.6f}".format

def Load_file_json(assets, cash):

    frase = ''
    with open("assets.json", "w") as f:
        inizio = '{'
        for i in range(len(assets)):
            stringa = f'"{assets[i]}": [{"weight": {0}, "n_assets": {0} + }], \n'

            frase = frase + stringa

        str_cash = f'"cash": {"weight": {1}, "n_assets": {cash}} \n'
        fine = '}'
        frase = inizio + frase + str_cash + fine

        json_object = json.loads(frase)
        json.dump(json_object, f, indent=2)


def thread_process(func: callable, variable_df: pd.DataFrame):
    with Pool() as po:
        return po.map(func, variable_df.to_numpy())


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
    def map_covariance_deviations(node):
        deviations = pd.DataFrame(columns=['first_term'])
        # for each stuff unique:
        # velocizza questo for loop
        for i, row in node.conditional_covariances.iterrows():
            l0 = node.conditional_covariances.loc[i, 'level_0']
            l1 = node.conditional_covariances.loc[i, 'level_1']
            deviations.loc[i, 'first_term'] = node.assets_data.loc[l0, 'residuals'] * node.assets_data.loc[
                l1, 'residuals']
        return deviations

    def target_function(self, m, list_of_probabilities, parent, sibling_nodes=None):
        """
        la funzione precedente calcola le deviazioi, qui, hai la funzione obiettivo, che sommatutte le deviazioni pesate
        di ciascun asset.
        IT MUST READ CONSTRAINT IN A WAY THAT IS READABLE BY CPLEX...
        """
        x = list_of_probabilities
        r, c = len(x), len(self.assets)

        all_returns = pd.concat([i.assets_data['returns_t'] for i in sibling_nodes], axis=1).T.to_numpy()
        dev1 = np.array(
            [np.array([m.abs(self.assets['a_i'].to_numpy()[l] - all_returns[i][l] * x[i]) for l in range(c)]) for i in
             range(r)]).flatten()

        all_sigmas = pd.concat([i.assets_data['residuals'] ** 2 for i in sibling_nodes], axis=1).T.to_numpy()
        dev2 = np.array([np.array(
            [m.abs((parent.assets_data['sigma_t'] ** 2).to_numpy()[l] - all_sigmas[i][l] * x[i]) for l in range(c)]) for
                         i in range(r)]).flatten()

        all_third = pd.concat([i.assets_data['residuals'] ** 3 for i in sibling_nodes], axis=1).T.to_numpy()
        dev3 = np.array(
            [np.array([m.abs(self.assets['third_moment'].to_numpy()[l] - all_third[i][l] * x[i]) for l in range(c)]) for
             i in range(r)]).flatten()

        all_fourth = pd.concat([i.assets_data['residuals'] ** 4 for i in sibling_nodes], axis=1).T.to_numpy()
        dev4 = np.array(
            [np.array([m.abs(self.assets['fourth_moment'].to_numpy()[l] - all_fourth[i][l] * x[i]) for l in range(c)])
             for i in range(r)]).flatten()

        all_cov = pd.concat([self.map_covariance_deviations(i).T for i in sibling_nodes]).to_numpy()
        cov_dev = np.array(
            [np.array([m.abs(parent.conditional_covariances[0].to_numpy()[l] - all_cov[i][l] * x[i]) for l in range(c)])
             for i in range(r)]).flatten()

        return m.sum(el for el in dev1), m.sum(el for el in dev2), m.sum(el for el in dev3), m.sum(el for el in dev4), m.sum(el for el in cov_dev)

    def optimization_func(self, parent, sibling_nodes):
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
        n = len(sibling_nodes)
        sensibility = 0.5
        LB = sensibility * 1 / n
        # Creazione delle variabili
        with Model(name='esempio_modello') as model:
            variables = [model.continuous_var(lb=LB, ub=1, name=f'x{i}') for i in range(n)]
            print(variables)
            # Vincolo
            model.add_constraint(model.sum(variables[i] for i in range(n)) == 1, ctname='sum_probability')
            model.set_objective(
                "minimize",
                model.sum(
                    obj for obj in self.target_function(model, variables, parent=parent, sibling_nodes=sibling_nodes))
            )
            # Risoluzione del modello
            solution = model.solve()
            print(solution)
        print('Valore ottimo:', solution.get_objective_value())
        # Stampa della soluzione
        '''print('Valore ottimo:', solution.get_objective_value())
        for i, var in enumerate(variables):
            print(f'x{i} = {solution.get_value(var)}')'''
        list_of_probabilities = [solution.get_value(var) for var in variables]
        print(list_of_probabilities)
        # Restituzione della lista delle soluzioni
        return list_of_probabilities  # solution_list

    def sibling_nodes(self, parent, optimization_func=None, matrix_cols=None, date=None):
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
        print(prob_list)
        i = 0
        for node in sibling_nodes:
            node.probability = prob_list[i]  # make method to update probability...
            # sibling_nodes[i] = node.to_dict()
            node.compute_conditional_probability(parent.cond_probability)
            i += 1
        return sibling_nodes

    @staticmethod
    def dictionarize(x):
        return x.apply(lambda y: y.go_to_dict())

    @staticmethod
    def find_coordinates(x):
        for i in range(len(x)):
            x[i].coordinates = [x.name, i]
            if x.name == 0:
                sibling_row = x.name + i
                x[i].son_coordinates = [sibling_row, i]
            elif x.name > 0:
                sibling_row = x.name + i + 2 * x.name
                x[i].son_coordinates = [sibling_row, i]

    @staticmethod
    def dictionarize_thread(row):
        return [el.go_to_dict() for el in row]

    def generate_tree(self, init_matrix):
        """
        PER ORA, GENERARE 8 PERIODI IN QUESTO MODO RICHIEDE: 2.0 minuti e 37 secondi UTILIZZANDO TUTTI I CORES.
        IL MASSIMO PRIMA DEL CRASH CON 16 GB RAM E': 9 nodi, in 7 minuti
        CI SONO MOMENTI IN CUI LA CPU NON E' USATA AL 100%. DOBBIAMO IDENTIFICARE COSA VIENE FATTO IN QUEI MOMENTI E FAR
        SI CHE VENGA USATA APPIENO ANCHE LI
        """
        init_matrix, counter = init_matrix, 1
        while counter <= self.horizon:
            print('contatore: ', counter)
            period_cfs, period_div = 0, 0  # taken from user inputs
            if init_matrix.size == 1:
                row, col = 0, 0
                root = init_matrix.iloc[row, col]
                matrix = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
                matrix.loc[len(matrix)] = self.sibling_nodes(
                    root, optimization_func=self.optimization_func, matrix_cols=matrix.columns, date=counter,
                    #period_cfs=period_cfs, period_div=period_div
                )
            else:
                # print("3 figli")
                parents = init_matrix.to_numpy().flatten()
                init_matrix = None  # reset init matrix to free up ram space...
                matrix_cols = ['0', '1', '2']
                if counter < 4:
                    matrix = pd.DataFrame(columns=matrix_cols)
                    for parent in parents:
                        matrix.loc[len(matrix)] = self.sibling_nodes(
                            parent=parent, optimization_func=self.optimization_func, matrix_cols=matrix_cols,
                            date=counter
                            #, period_cfs=period_cfs, period_div=period_div
                        )
                    # matrix = self.x(
                    #     parents, matrix_cols, counter, period_cfs, period_div, matrix=pd.DataFrame(columns=matrix_cols)
                    # )
                else:
                    with Pool() as po:  # we can't do anything for speed in here... we just need a better processor...
                        mapped = po.map(partial(
                            self.sibling_nodes,
                            optimization_func=self.optimization_func,
                            matrix_cols=matrix_cols,
                            date=counter,
                        ), parents)
                    del parents  # reset parents to free up RAM
                    matrix = pd.DataFrame(mapped, columns=matrix_cols)
                    del mapped  # free up ram...
            init_matrix = matrix  # hide it if return to old way...
            print(f'example at time: {counter}')
            print(matrix.loc[0].head(1).values[0].assets_data)
            print('dimensioni: ', init_matrix.size)
            # matrix = matrix.apply(lambda x: self.dictionarize(x), axis=1)
            if counter < 7:
                matrix = matrix.apply(lambda x: self.dictionarize(x), axis=1)
            else:
                matrix = pd.DataFrame(thread_process(self.dictionarize_thread, matrix), columns=matrix_cols)
            print(matrix)
            matrix.to_parquet(f'period_{counter}')
            #table = pa.Table.from_pandas(matrix)
            #pq.write_table(table, f'period_{counter}')
            del matrix
            gc.collect()
            counter += 1

    def test_node(self):
        root_node = ScenarioNode(
            root=True, parent=self.assets, returns=self.ret_list, cor_matrix=self.corr_matrix
        )
        print('assets data of root node:')
        print(root_node.assets_data)
        init_matrix = pd.DataFrame({root_node}, columns=['0'])
        # save initial matrix somewhere like a parquet file, just make sure that it saves entire instances in the cells
        self.generate_tree(init_matrix)
        print('tree generated!')
        init_matrix.apply(lambda x: self.dictionarize(x), axis=1).to_parquet(f'period_{0}')

    @staticmethod
    def convert_to_node(x):
        for i in range(len(x)):
            x[i] = ScenarioNode(init_data=x[i])
        # return x.apply(lambda y: ScenarioNode(init_data=y))

    def read_tree(self):
        """
        qui si riconvertono le celle da liste a ScenarioNode. Poi, si svilupperà la logica di iterazione lungo l'albero
        """
        self.tree = [pd.read_parquet(f'period_{x}.parquet') for x in range(self.horizon + 1)]  # read parquet of all periods
        for matrix in self.tree:
            matrix.apply(lambda x: self.convert_to_node(x), axis=1)
        #print(self.tree[-1].loc[0, 0].conditional_volatilities)

    def clear(self):
        # deletes the tree parquet files
        pass


def main(exeall = True):  # variabili prese da input
    # prendo i dati dal file data.txt
    data_list = []
    data = open("data.txt", "r")
    for row in data:
        row = data.readline()
        if "/" not in row:
            row = re.sub('[\n]', '', row)  # per eliminare i \n alla fine di ogni riga
            data_list.append(row)
            data.readline()  # serve per evitare le righe vuote (non è bellissimo, ma funziona)

    #print(data_list)

    horizon, lb, ub, risk, cash_return, cash_currency, alpha, VaR_list, LCVar, annuaties, cf_list, assets_list, cash, period = data_list[0], \
        data_list[1], data_list[2], data_list[3], data_list[4], data_list[5], data_list[6], data_list[7], data_list[8], \
        data_list[9], data_list[10], data_list[11], data_list[12], data_list[13]
    # print(horizon, lb, ub, risk, cash_return, alpha, VaR_list, LCVar, annuaties, cf_list, assets_list, cash)
    assets_list, VaR_list, LCVar = tuple(assets_list.split(',')), VaR_list.split(','), LCVar.split(',')
    # per trasformare le stringhe in liste
    # print(type(assets_list))
    # print(assets_list)

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
    # collegamento con la query multi_value_fetch
    '''dc = DBConnection()
    do = DBOperations(dc.conn)
    sd = [['stock_id', 'symbol'], 'stock_data']
    ed = [['etf_id', 'symbol'], 'etf_data']
    stock_df, etf_df = do.multi_value_fetch(sd, ed)  # restituisce una lista, ma tutto il contenuto della query va a finire nella prima cella
    # stock_df = pd.DataFrame(stock_df[0], columns=['stock_id', 'symbol'])
    etf_df = etf_df.rename(
        columns={'etf_id': 'stock_id'})  # rinominiamo la colonna perchè altrimenti la concatenazione viene sballata
    assets_df = pd.concat([stock_df, etf_df]).reset_index(drop=True)
    #print(assets_df)
    # 03_chiedo quali stock_id vuole usare
    #print("Inserisci solo gli stock_id degli asset che vuoi usare, separati da uno INVIO. Quando hai finito digita STOP.")
    i = 0
    assets_list = []
    while True:
        assets_input = str(input("" + str(i + 1) + " --> "))
        if (assets_input.upper() == "STOP" or len(assets_input) <= 1):
            break
        assets_list.append(assets_input)
        i += 1
    df_input = assets_df['stock_id'].isin(assets_list)  # cerco nella colonna stock_id del dataframe i valori nella lista assets_list
    assets_df = assets_df[df_input]
    stock_df.reset_index(inplace=True, drop=True)
    # print(assets_df)
    assets_list = assets_df['stock_id'].values.tolist()  # trasformo la colonna del dataframe in una lista per il json
    #print(assets_df)'''


    '''horizon = 10  # default
    while True:
        period = int(input("Inserisci il period, (1) settimanale (2) mensile (3) annuale: \t"))
        horizon = int(input("Inserisci l'orizzonte: \t"))
        if ((time_division == 1 and horizon < 104) or (time_division == 2 and horizon < 24) or (
                time_division == 3 and horizon < 2)):
            break
    lb, ub = 0, 1 #di default
    while True:
        lb = input("Inserisci il limite minimo dei pesi (compreso tra 0 e 1): \t")
        ub = input("Inserisci il limite massimo dei pesi (compreso tra 0 e 1): \t")
        if 0 < lb < ub < 1:
            break
    risk = input("Inserisci quanto sei disposto a rischiare in valore decimale")
    cash_return = float(input("Inserisci il cash return"))
    cash_currency = input("inserisci la currency")
    alpha = 0.05  # livello di confidenza per la misurazione del VaR
    VaR_list = [0.07, 0.05, 0.1]
    LCVar = [0.077, 0.055, 0.11]  # è un multiplo  di VaR_lis (obiettivi)
    annuaties = None  # dataframe, tante righe quante sono i periodi, da prendere in input, di default rimane None
    cf_list = None  # da prendere in input una lista lunga quanto i periodi, altrimenti rimane None
    assets_list = ['IS0Z_XETR_EUR','IS3N_XETR_EUR','IWLE_XETR_EUR']
    cash = input('Inserisci il cash')'''

    with Model(name='esempio_modello') as model:
        VaR = [model.continuous_var(lb=0, name=f'VaR_{i}') for i in range(len(VaR_list))]
        VaR = [model.continuous_var(lb=0, name=f'VaR_{i}') for i in range(len(VaR_list))]

    # ------------------------------------------------------------------------------------------------------------------------
    # 05_controllare se esiste già un file json e riproporlo
    # https://www.scaler.com/topics/seek-function-in-python/
    with open('assets.json', 'r') as file:
        file.seek(0, os.SEEK_END)  # puntatore, si sposta di 0 lettere, partendo dal fondo
        isempty = file.tell() == 0  # returns the current file position in a file stream.
        file.seek(0)  # riavvolgere il file
        #print(isempty)
        if isempty == False:
            with open('assets.json', 'r') as f:
                data = f.read()
                json_data = json.loads(data)
                pprint.pprint(json_data)  # pprint --> fornisce la capacità di stampare la rappresentazione formattata dei dati JSON.
            risposta = str(input("Esiste un file contenente degli assets già utilizzati. Vuoi usarlo? \t"))
            if risposta.upper() == "NO":
                # faccio una prova con un lista preimpostata
                # results =['AAPL_NASDAQ_USD', 'AMZN_NASDAQ_USD', 'EUE_MTA_EUR', 'JPM_NYSE_USD', 'PRY_MTA_EUR', 'SXRV_XETR_EUR', 'UNIR_MTA_EUR']
                Load_file_json(assets_list, cash)  # devo passargli gli assets_list ma per ora non funziona il database
        else:
            # results = ['AAPL_NASDAQ_USD', 'AMZN_NASDAQ_USD', 'EUE_MTA_EUR', 'JPM_NYSE_USD', 'PRY_MTA_EUR', 'SXRV_XETR_EUR', 'UNIR_MTA_EUR']
            Load_file_json(assets_list, cash)

    if exeall:
        #generazione albero
        tree = NewTree(
            assets_df, ast_ret, horizon=horizon, cash_return=cash_return, period=period, cash_currency=cash_currency
        )
        tree.test_node()

    Iteration_tree.read_tree(lb, ub, alpha, VaR_list, LCVar, cash)


# ------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    start = timeit.default_timer()

    ast_json = json.loads(open('assets.json', 'r').read())
    assets_df = pd.read_parquet('assets_df.parquet')

    portfolio = pd.DataFrame(ast_json).T  # a sto punto, json file prende anche currency (?)
    assets_df = pd.concat([assets_df[['stock_id', 'currency']].set_index('stock_id'), portfolio], axis=1)
    current_assets_prices = pd.read_parquet('curr_assets_prices.parquet')  # .set_index('stock_id')
    assets_df['close_prices_t'] = current_assets_prices['close'].astype('float64')

    ast_ret = pd.read_parquet('assets_returns.parquet')
    #print(ast_ret)
    # ast_ret.set_index(['datetime'])

    #tree = NewTree(
    #    assets_df, ast_ret, horizon=5
    #)
    #tree.test_node()

    stop = timeit.default_timer()
    minutes = (stop - start) / 60
    #print('full minutes: ', minutes)
    seconds, minutes = math.modf(minutes)
    print(f'{minutes} minuti e {round(seconds * 60)} secondi')

    # ------------------------------------------------------------------------------------------------------------------------