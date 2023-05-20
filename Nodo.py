
# import pyarrow
from math import e

import numpy as np
import pandas as pd
from numpy import ndarray, pi


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


class ScenarioNode:
    """collects all data about the given node: the node id, to identify the node on the tree; the realized returns and
    all the other node events (such as cashflows and dividends...).
    On the node will then be collected data about decisions and realizations for the portfolio being optimized...

    The node by itself generates an executes all its methods in around 0.007422 seconds. So, how we can improve the
    tree generation? Is all in the optimal employment of RAM and Threads/processes at this point"""
    def __init__(
            self, root: bool, parent, returns: pd.DataFrame, cor_matrix: pd.DataFrame, period_date=None
    ):
        self.periods = []

        self.root = root
        self.date = period_date  # to be used for index of new row of returns
        self.parent = None
        self.assets_data = None
        self.init_values(parent)  # modifies data of above parameters

        self.compute_returns(returns)
        # probabilities measurement inputs:
        self.conditional_variances = self.compute_variances()  # diagonal matrix of conditional variances
        self.conditional_covariances = self.compute_covariances(cor_matrix=cor_matrix)  # covariance matrix

        self.parent_probability = parent.probability if root is False else None
        self.probability = 1  # of single Node, not conditional to the previous node probability
        self.cond_probability = self.probability * parent.cond_probability if root is False else self.probability
        # adds the dividends and other cashflows expected on the scenario...
        """
        should also:
        Compute dividend_yield (for return purposes), Append annuities_period to the assets_data 
        then, from cash_flows period subtract/add cash moved by investor to the portfolio and then execute rebalancing
        code
        Add cashflows of dividends as column in assets_data, passing it during each period
        """
        # here we initialize the cash data, to be iterated through the tree:
        self.cash_data = None  # self.assets_data.tail(1)[['currency', 'weight', 'n_assets']]
        # maybe pass those two to all nodes as parameters...
        self.cashflows_data = None
        # self.dividends = None
        self.rebalancing_vector = []  # vector of shares/value to buy (+) or sell (-) of each asset

    def __str__(self):
        return f"node_n_from_period_{self.date}"

    def __repr__(self):
        return f"node_n_from_period_{self.date}"

    def init_values(self, parent):
        if self.root is False:
            self.parent = parent
            self.assets_data = parent.assets_data
        else:
            self.assets_data = parent

    def compute_returns(self, returns: pd.DataFrame):
        """
        gets a random sample of returns from each assets, then adjusts the returns sampled to get unique values
        """
        # delegate dropna and columns iteration to init in tree
        sampled = [ret.sample().reset_index(drop=True) for ret in returns]
        self.assets_data['returns_t'] = 1 + pd.concat(sampled, axis=1).T
        self.assets_data['close_prices_t'] = self.assets_data['close_prices_t'] * self.assets_data['returns_t']
        # now, check on prices, if they are absurd, we need to replace them i think...

    def compute_residuals(self) -> pd.DataFrame:
        residuals = (self.assets_data['returns_t'] - 1 - self.assets_data['a_i']) * 100
        return residuals.T

    def compute_variances(self) -> pd.DataFrame:
        if self.root is False:
            # IF YOU SOLVE THE EGARCH PROBLEM, YOU CAN SUBSTITUTE HERE WITH THE EGARCH FORMULA FROM ARCH!
            # WOULD BE NICE TO REDUCE NUMBER OF PASSAGES, IF IT DIDN'T RESULT IN ABSURD VALUES...
            dummy = self.assets_data
            # here lies the problem with shortening code...
            dummy['sigma_t'] = self.assets_data['sigma_t'] * 100
            dummy['e_val'] = self.compute_residuals() / dummy['sigma_t']
            # self.assets_data['sigma_t'] = dummy[['e_i', 'omega', 'alpha[1]', 'gamma[1]', 'beta[1]', 'sigma_t']].apply(
            #     # WIP!!! ON THAT FUNCTION, SPEED IT UP IN ITS CALCULATIONS...
            #     lambda x: egarch_formula(x[0], x[1], x[2], x[3], x[4], x[5]), axis=1
            # )
            dummy['term_1'] = dummy['alpha[1]'] * (abs(dummy['e_val']) - np.sqrt(2/pi))
            dummy['term_2'] = dummy['gamma[1]'] * dummy['e_val']
            dummy['term_3'] = dummy['beta[1]'] * np.log(dummy['sigma_t']**2)

            dummy['log_sigma_2'] = dummy['omega'] + dummy['term_1'] + dummy['term_2'] + dummy['term_3']

            self.assets_data['sigma_t'] = np.sqrt(e**dummy['log_sigma_2']) / 100
            # WIP!!! THIS IS THE ULTIMATE PROBLEM TO SOLVE! really a dirty solution for now...
            self.assets_data.loc[self.assets_data['sigma_t'] < 0.00001, 'sigma_t'] = 0.00001
            self.assets_data = self.assets_data.drop(['e_val', 'term_1', 'term_2', 'term_3', 'log_sigma_2'], axis=1)
        return pd.DataFrame(
            np.diag(self.assets_data['sigma_t']**2), index=self.assets_data.index, columns=self.assets_data.index
        )

    def compute_covariances(self, cor_matrix: pd.DataFrame) -> pd.DataFrame:
        sqrt_variances = np.sqrt(self.conditional_variances)
        cov_matrix = np.dot(sqrt_variances, np.dot(cor_matrix, sqrt_variances))
        return pd.DataFrame(cov_matrix, index=self.assets_data.index, columns=self.assets_data.index)


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

