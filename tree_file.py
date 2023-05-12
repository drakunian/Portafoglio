import json
import os
import pprint
import random
import re
import timeit
from cmath import pi, log

import numpy as np
import pandas as pd
import itertools

import stochoptim as so  # la libreria degli alberi decisionali
import scipy.stats as stats
from arch import arch_model
from dateutil.relativedelta import relativedelta
from docplex.mp.model import Model

import pyarrow.parquet as pq

from dbconn_copy import DBConnection, DBOperations

#from dbconn_copy import DBConnection, DBOperations

pd.options.display.float_format = "{:,.6f}".format


def egarch_formula(e_i, omega, alpha, gamma_, beta, sigma_t_m_1): #da mettere nella classe tool per le due classi
    """For now, when errors occur giving too high volatilities (as high as 45.000!), we just give back the same
    volatility... Not good but better than the other option for now...
    Then, avoid dividing by 0: if sigma_t_m_1 == 0.0, then divide by 0.000001"""
    if sigma_t_m_1 == 0.0:
        sigma_t_m_1 = 0.00001
    e_val = e_i / sigma_t_m_1
    log_sigma_2 = omega + alpha * (abs(e_val) - np.sqrt(2/pi)) + (gamma_ * e_val) + beta * log(sigma_t_m_1**2)
    sigma_2 = np.exp(log_sigma_2, dtype=np.float64)
    if sigma_2 > 1000:
        sigma_2 = sigma_t_m_1**2
    return np.sqrt(sigma_2) / 100

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
        


class ScenarioNode:
    """collects all data about the given node: the node id, to identify the node on the tree; the realized returns and
    all the other node events (such as cashflows and dividends...).
    On the node will then be collected data about decisions and realizations for the portfolio being optimized..."""
    def __init__(self, is_root=False, ad=None, parent=None, corr_matrix=None, in_ret=None, in_res=None, period_date=None):
        self.root = is_root
        self.assets_data = ad  # get from parent. It will have new column: Close_Prices, that will be renamed here
        self.assets_data['close_prices_t_m_1'] = self.assets_data['close_prices_t']
        print(self.assets_data)

        self.parent = parent
        self.corr_matrix = corr_matrix
        self.inherited_assets_returns = in_ret  # df of realized assets returns, inherited from parent, append new data
        self.inherited_returns_residuals = in_res  # df of residuals from returns, inherited from parent, append new data
        self.date = period_date  # to be used for index of new row of returns


        #computer_returns is a functioin
        self.realized_returns = self.compute_returns() if is_root is False else self.inherited_assets_returns.tail(1)

        self.residuals = self.compute_residuals() if is_root is False else self.inherited_returns_residuals.tail(1)
        # probabilities measurement inputs:
        # diagonal matrix of conditional variances
        self.conditional_variances = self.compute_variances() #compute_variances is a functione che usa il EGARCH
        # covariance matrix
        self.conditional_covariances = self.compute_covariances()

        self.parent_probability = self.parent.probability if is_root is False else None
        self.probability = 1  # of single Node, not conditional to the previous node probability

        # adds the dividends and other cashflows expected on the scenario...

        #da qua in poi ci saranno tutte variabili utili per ottimizzare il portafoglio
        self.rebalancing_vector = []  # vector of shares/value to buy (+) or sell (-) of each asset

    def compute_returns(self):
        """gets a random sample of returns from each assets, then adjusts the returns sampled to get unique values"""
        sampled = [self.inherited_assets_returns[column].dropna().sample().reset_index(drop=True) for column in self.inherited_assets_returns.columns]
        #dropna() --> removes the rows that contains NULL values
        #sample() --> Return a random sample of items from an axis of object.
        #.reset_index() --> allows you reset the index back to the default 0, 1, 2 etc indexes
        #.columns --> has successfully returned all of the column labels of the given dataframe.

        sample_row = pd.concat(sampled, axis=1)  # riga di dataframe con index 0
        #.concat() -->  does all the heavy lifting of performing concatenation operations along with an axis od Pandas objects while performing optional set logic (union or intersection) of the indexes (if any) on the other axes
        sample_row.index = [self.date] # index della riga diventa la data del nodo
        # updates the inherited returns data...
        self.inherited_assets_returns = pd.concat([self.inherited_assets_returns, sample_row])
        self.assets_data['returns_t'] = sample_row.T
        # .T --> used to obtain the transpose of a given array.
        # ... update close_prices:
        self.assets_data['close_prices_t'] = self.assets_data['close_prices_t_m_1'] * (1 + self.assets_data['returns_t'])
        return sample_row

    def compute_residuals(self):
        residuals = (self.realized_returns - self.assets_data['a_i'].T) * 100 #a_i sono i ritorni medi
        # updates the inherited residuals data...
        self.inherited_returns_residuals = pd.concat([self.inherited_returns_residuals, residuals])
        return residuals

    def compute_variances(self):
        if self.root is False:
            # IF YOU SOLVE THE EGARCH PROBLEM, YOU CAN SUBSTITUTE HERE WITH THE EGARCH FORMULA FROM ARCH!
            dummy = self.assets_data
            # rescaling a_i and sigma:
            dummy['sigma_t'] = self.assets_data['sigma_t'] * 100
            # forecasted variances, to be taken also when measuring probabilities of future sibling nodes
            for column in self.inherited_assets_returns:
                e_i = self.inherited_returns_residuals[column].tail(1).values[0]
                omega, alpha, gamma_, beta, sigma_t_m_1 = dummy[['omega', 'alpha[1]', 'gamma[1]', 'beta[1]', 'sigma_t']].loc[column].to_list()
                sigma_2 = egarch_formula(e_i, omega, alpha, gamma_, beta, sigma_t_m_1)
                self.assets_data.loc[column, 'sigma_t'] = sigma_2
        else:
            pass
        return pd.DataFrame(np.diag(self.assets_data['sigma_t']**2), index=self.assets_data.index, columns=self.assets_data.index)
    #.diag(v, k) --> function creates a diagonal matrix or extracts the diagonal elements of a matrix. sulla diagonale ci sono le sigma_2 degli asset
    # The default value of k is 0. Use k>0 for diagonals above the main diagonal, and k<0 for diagonals below the main diagonal.

    def compute_covariances(self):
        # it will me used when measuring probabilities of future sibling nodes
        cov_matrix = np.dot(self.conditional_variances, np.dot(self.corr_matrix, self.conditional_variances))
        #.dop() --> Dot product of two arrays, matrix or scalar
        return pd.DataFrame(cov_matrix, index=self.assets_data.index, columns=self.assets_data.index)
        #.dataFrame() --> is a 2 dimensional data structure, or a table with rows and columns.
        #nameOfTheDataframe.loc[0] --> return the column 0 of the data frame


class NewTree:
    def __init__(self, assets, horizon, cash_return, period='1month'): #sono variabili prese da input
        # Define inputs and starters:
        #self.assents = assets --> sarà il file json
        self.period = period
        self.dt = 1 / horizon
        self.horizon = horizon  # Number of time periods (e.g. 12 months)
        self.cash_return = cash_return  # Return on cash asset. Not the rf rate, but the return on YOUR cash asset...

        self.assets = assets[['stock_id', 'symbol', 'exchange', 'currency']]
        assets_dataframe = pd.read_parquet('assets_dataframe.parquet')
        # self.returns_data = self.assets_returns()  # Monthly returns for n assets
        self.returns_data = pd.read_parquet('assets_returns.parquet')
        print(self.returns_data)

        self.corr_matrix = self.returns_data.corr()  # Constant correlation matrix
        #.corr() --> is used to find the pairwise correlation of all columns in the. (in pratica crea
        # una matrice con diagonale principale = 1, nelle parentesi si può aggiungere un metodo di correlazione)

        self.assets = self.assets.set_index('stock_id')
        #.set_index() --> used to set the DataFrame index using existing columns

        self.garch_params, self.residuals = self.compute_egarch_params() #function
        self.compute_moments() #function
        self.assets[['omega', 'alpha[1]', 'gamma[1]', 'beta[1]', 'sigma_t']] = self.garch_params
        self.assets['close_prices_t'] = 1 #sono normalizzati, andranno presi tramite una funzione
        # CREATES THE SERIES OF NODES:
        self.scenario_nodes = self.scenario_nodes_2(stage_1=50) # numero scelto
        print(self.scenario_nodes)
        # End of inputs and starters definition...
        self.root_node = ScenarioNode(is_root=True, ad=self.assets, corr_matrix=self.corr_matrix, in_ret=self.returns_data,
                                      in_res=self.residuals, period_date=self.returns_data.index[-1])
        # non viene passato il parent in questo ScenarioNode
        self.scenario_period_nodes = [[[self.root_node]]]
        # print(self.corr_matrix)

    def assets_returns(self):
        """Non serve questa funzione ora, gli asset returns sono aperti direttamente dal file parquet"""
        # ast_list = self.assets.reset_index()[['stock_id', 'symbol', 'exchange']].to_numpy()
        # tr = min(20, len(ast_list))
        # assets_returns = assets_returns_matrix(ast_list, log_return=False, is_index=False, eq_length=False, cons=11, threads=tr, period=self.period)
        # assets_returns.to_parquet('assets_returns.parquet')
        # return assets_returns

    def compute_egarch_params(self):
        eam_params_list = []
        residuals_df = []
        for column in self.returns_data:
            rets = self.returns_data[column].dropna() * 100
            eam = arch_model(rets, p=1, q=1, o=1, mean='constant', power=2.0, vol='EGARCH', dist='normal') # --> ??????????
            #arch_model() --> I modelli ARCH sono una classe popolare di modelli di volatilità che utilizzano valori osservati di rendimenti o residui come shock di volatilità

            eam_fit = eam.fit(disp='off')
            #.fit() --> estimate the arch model
            eam_params = eam_fit.params # ritorna A (A = a_i), alfa, beta, gamma, omega
            last_vol = eam_fit.conditional_volatility.tail(1).values[0]  # make sure is volatility and not variance...
            #conditional_volatility --> ??
            # now adjust if value is incredibly high:
            if last_vol > 1000:
                last_vol = last_vol / 1000
            eam_params['sigma_t'] = last_vol / 100  # scaling sigma to decimals
            residual_list = eam.resids(eam_params['mu'], [rets]) #mu è la versione percentuale di a_i
            # A residual is the difference between an observed value and a predicted value in a regression model

            residuals = pd.DataFrame(residual_list.T, columns=[column], index=rets.index)
            # print(eam_fit.conditional_volatility[-1])
            # eam_params['sigma_0'] = eam_fit.conditional_volatility[-1]
            eam_params.name = column
            eam_params_list.append(eam_params)
            residuals_df.append(residuals)

        df = pd.concat(eam_params_list, axis=1).T
        residuals_df = pd.concat(residuals_df, axis=1)
        print(residuals_df)
        df.index.name = 'stock_id'
        self.assets['a_i'] = df['mu'] / 100

        return df[['omega', 'alpha[1]', 'gamma[1]', 'beta[1]', 'sigma_t']], residuals_df

    def compute_moments(self):
        for i, row in self.assets.iterrows():
            self.assets.loc[i, 'third_moment'] = stats.moment(self.returns_data[i].dropna(), moment=3, nan_policy='propagate')  # skewness
            self.assets.loc[i, 'fourth_moment'] = stats.moment(self.returns_data[i].dropna(), moment=4, nan_policy='propagate')  # kurtosis
            # .loc --> used to access a group of rows and columns by label(s) or a boolean array

    def compute_moments_weight(self):
        """
        using principal component analysis (PCA), gets the weights for each moment deviation, same for covariances.
        """
        return 0

    def objective_function(self):
        """
        defined here the objective function
        """
        return 0

    def compute_optimal_probabilities(self):
        """
        Minimizes the objective function, I do not know how...
        But! The optimum is measured on each group of sibling nodes. This means that m+ and m- are computed here, they
        are actually scenario dependent (parent-node dependent...) Find a linear optimizer to use for that purpose...

        start defining constraints (from 24 - 30), and then use minimize function on self.objective_function
        """
        n_assets = len(self.assets)
        constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},  # sum of probabilities == 1
                       {"type": "eq", "fun": lambda x: sum(r_i_s * x) - a_i},  # r_i == a_i for each i
                       {"type": "eq", "fun": lambda x: sum(((r_i_s - a_i)**2) * x) - sigma_2},  # 26 const.
                       )

        bounds = tuple(self.bound for x in range(n_assets))
        return 0

    def scenario_nodes_2(self, stage_1):
        """base node, with index = 0, is always 1; then, from it...
        Returns a dataframe with the nodes for each parent node and the total nodes for each period"""
        periods = self.horizon + 1
        nodes_json = [{'siblings_nodes': 2} for _ in range(periods)] # crea una colonna di 2
        nodes_series = pd.DataFrame(nodes_json).reset_index(drop=True)
        nodes_series.loc[0, 'siblings_nodes'] = 1
        nodes_series.loc[1, 'siblings_nodes'] = stage_1
        return nodes_series

    def sibling_nodes_gen(self, parent, n_siblings):
        """n_siblings is a range of elements to create, parent is an instance of ScenarioNode class"""
        sibling_nodes = []
        for _ in n_siblings:  # make it a thread...
            sibling_node = ScenarioNode(ad=parent.assets_data, parent=parent, corr_matrix=self.corr_matrix,
                                        in_ret=parent.inherited_assets_returns, in_res=parent.inherited_returns_residuals,
                                        period_date=parent.date + relativedelta(months=1))
            sibling_nodes.append(sibling_node)
        # APPLY HERE OPTIMIZATION PROBLEM!
        return sibling_nodes

    def test_node(self):
        # for now, then, try to pass them directly from the init function:
        # STARTS CREATING THE ROOT_NODE'S SONS:
        # WORK IN PROGRESS!!! NEED TO BE SPED UP!
        for i, row in self.scenario_nodes.iterrows():  # speed it up! Vectorized my brother! or apply...
            if i != 0:
                print(len(self.scenario_period_nodes))
                parent_nodes = self.scenario_period_nodes[-1]
                # merge parents in a single list:
                if len(parent_nodes) == 1:
                    parent_nodes = parent_nodes[0]
                else:
                    parent_nodes = list(itertools.chain.from_iterable(parent_nodes))  # merges in single list all parents
                print('number of parents: ', len(parent_nodes))
                print('parents: ', parent_nodes)
                sibling_nodes = range(row[0])
                print('sons per parent: ', sibling_nodes)
                # the process below must be accelerated...
                period_nodes = []
                for parent in parent_nodes:
                    print('the parent: ', parent, i)
                    siblings = self.sibling_nodes_gen(parent, sibling_nodes)
                    period_nodes.append(siblings)
                self.scenario_period_nodes.append(period_nodes)
        # THE TREE IS ULTIMATED, HERE HIS DATA. NOW, HOW TO SAVE IT? HOW TO MANAGE EGARCH VOL ERRORS?
        print(self.scenario_period_nodes)
        print([len(i) for i in self.scenario_period_nodes if i != 0])

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

    main()

    # dataframe = pd.read_json("assets.json")
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
    vattore dei ribilanciamenti
    metodo ricorsivo per la creazione dei figli (finisce quando arrivo all'ultimo periodo)
'''

"""
test
"""