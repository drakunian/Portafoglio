from functools import partial
# import pyarrow
from math import e, log
from multiprocessing import Pool

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

    The node by itself generates and executes all its methods in around 0.007422 seconds. So, how we can improve the
    tree generation? Is all in the optimal employment of RAM and Threads/processes at this point"""

    # implement __slots__ to reduce memory usage by class ScenarioNode...
    # __slots__ = (
    #     "root", "parent_coordinates", 'coordinates', 'date', 'assets_data',
    #     'conditional_volatilities', 'conditional_covariances', 'covariances_matrix',
    #     'probability', 'cond_probability', 'cash_data', 'cashflows_data', 'rebalancing_vector'
    # )
    # IT SEEMS NOTHING CHANGES...

    def __init__(
            self,
            root: bool = False,
            parent=None,  # ScenarioNode | pd.Dataframe
            cfs: float = None,  # single value of net in-outflows of the period...
            returns: [pd.DataFrame] = None, div: pd.DataFrame = None, cor_matrix: pd.DataFrame = None, period_date=None,
            init_data: list = None
    ):
        # MODIFICA INIT COSì CHE LEGGE O INPUT PRIMARI, O LA LISTA DALLA CELLA DELLA MATRICE PARQUET
        # HINT: SET ALL PARAMS TO None, Legge quali non sono none ed esegue di conseguenza
        self.root = root
        if init_data is not None:
            self.read_init_data(init_data[0])
        else:
            self.date = period_date  # to be used for index of new row of returns
            # self.parent = None  NON SERVE!
            self.assets_data = self.pass_asset_data(parent)  # modifies data of above parameters

            self.compute_returns(returns, div)
            # probabilities measurement inputs:
            self.conditional_volatilities = self.compute_variances()  # diagonal matrix of conditional variances
            self.conditional_covariances, self.covariances_matrix = self.compute_covariances(cor_matrix=cor_matrix)

            self.probability = 1  # of single Node, not conditional to the previous node probability
            self.cond_probability = 1

        self.cashflows_data = cfs  # inflows ad outflows net of cash in period...
        # self.dividends = None
        self.rebalancing_vector = []  # vector of shares/value to buy (+) or sell (-) of each asset

    def __str__(self):
        return f"node_x_from_period_{self.date}"

    def __repr__(self):
        return f"node_x_from_period_{self.date}"

    def read_init_data(self, init_data):
        self.date = init_data['node_date']
        self.assets_data = pd.DataFrame(json.loads(init_data['assets_data'])).set_index('index')
        self.covariances_matrix = pd.DataFrame(json.loads(init_data['cond_covariances']))
        self.cond_probability = init_data['cond_probability']
        self.cashflows_data = init_data['cashflows_data']

    def pass_asset_data(self, parent):
        # filter assets data from parent!
        if self.root is False:
            # self.parent = parent
            # parent may be an id in the future...
            return parent.assets_data
        else:
            # instead of none, may generate an id in the future...
            return parent

    def compute_returns(self, returns: [pd.DataFrame], dividends: pd.DataFrame):
        """
        gets a random sample of returns from each assets, then adjusts the returns sampled to get unique values
        """
        # delegate dropna and columns iteration to init in tree
        sampled = [ret.sample().reset_index(drop=True) for ret in returns]
        self.assets_data['returns_t'] = pd.concat(sampled, axis=1).T
        self.assets_data['close_prices_t'] = self.assets_data['close_prices_t'] * (1 + self.assets_data['returns_t'])
        self.assets_data['residuals'] = self.assets_data['returns_t'] - self.assets_data['a_i']
        # now, check on prices, if they are absurd, we need to replace them i think...
        # now read dividend data:
        # for now, we take constant dividend yield expected as we use only etfs... then, identifiers by columns mode
        self.assets_data['earned_dividend_yield_t'] = 0  # for now...
        if dividends is not None:
            """
            theoretical_dividend_yield_t accounts for capitalization of dividend expected in the future periods, but
            not yet earned, inside the stock price.  might not be needed as it should be info already contained in a...
            """
            # self.assets_data['theoretical_dividend_yield_t'] = dividends
            self.assets_data['earned_dividend_yield_t'] = dividends  # yield on the open price... net of tax already
        self.assets_data['total_returns'] = self.assets_data['returns_t'] + self.assets_data['earned_dividend_yield_t']

    def compute_variances(self) -> pd.DataFrame:
        """
        inf sigma_t still existing in some nodes. Adjust that!
        """
        if self.root is False:
            dummy = self.assets_data
            # here lies the problem with shortening code...
            dummy['residuals'] = self.assets_data['residuals'] * 100
            dummy['sigma_t'] = self.assets_data['sigma_t'] * 100
            dummy['e_val'] = dummy['residuals'] / dummy['sigma_t']
            dummy['log_sigma_2'] = dummy['omega'] + dummy['alpha[1]']*(abs(dummy['e_val'])-np.sqrt(2/pi)) + \
                                   dummy['gamma[1]']*dummy['e_val'] + dummy['beta[1]']*np.log(dummy['sigma_t']**2)

            self.assets_data['sigma_t'] = np.sqrt(e**dummy['log_sigma_2']) / 100
            del dummy
            # WIP!!! THIS IS THE ULTIMATE PROBLEM TO SOLVE! really a dirty solution for now...
            self.assets_data.loc[self.assets_data['sigma_t'] < 0.00001, 'sigma_t'] = 0.0005  # CHANGE THAT!
            self.assets_data.loc[self.assets_data['sigma_t'] > 1.5, 'sigma_t'] = 0.05  # CHANGE THAT!
            self.assets_data = self.assets_data.drop(['e_val', 'log_sigma_2'], axis=1)  # 'term_1', 'term_2', 'term_3'
            self.assets_data['residuals'] = self.assets_data['residuals'] / 100
        return pd.DataFrame(
            np.diag(self.assets_data['sigma_t']), index=self.assets_data.index, columns=self.assets_data.index
        )

    def compute_covariances(self, cor_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        SIMPLIFY AND SPEED UP THAT ONE HERE!
        """
        cov_matrix = np.dot(self.conditional_volatilities, np.dot(cor_matrix, self.conditional_volatilities))
        cov_matrix = pd.DataFrame(cov_matrix, index=self.assets_data.index, columns=self.assets_data.index)
        dpc = deepcopy(cov_matrix)
        cov_matrix = cov_matrix.where(np.triu(np.ones(cov_matrix.shape)).astype(np.bool_)).stack().reset_index()
        # return both compressed and matrix mode
        return cov_matrix.loc[cov_matrix['level_0'] != cov_matrix['level_1']], dpc

    def compute_conditional_probability(self, parent_cond_prob: float):
        self.cond_probability = self.probability * parent_cond_prob

    def go_to_dict(self):
        """
        converts node into a dictionary to be passed to the dataframe!
        """
        return [{
            'node_date': self.date,
            'assets_data': self.assets_data.reset_index().to_json(orient='records'),  # already has variances&dividends
            'cond_covariances': self.covariances_matrix.to_json(orient='records'),
            'cond_probability': self.cond_probability,
            'cashflows_data': self.cashflows_data
            # pass also cashflows data & dividends of given period
        }]

    # define methods for optimization:
    def compute_final_pf_value(self, parent_pf_data: pd.DataFrame) -> float:
        active_pf_value = parent_pf_data['n_assets'] * self.assets_data['close_prices_t']
        # must sum also dividends earned:
        cash_dividends = sum(
            self.assets_data['close_prices_t'] * self.assets_data['earned_dividend_yield_t'] * parent_pf_data['n_assets']
        )
        cash_value = parent_pf_data.loc['cash', 'n_assets'] + self.cashflows_data + cash_dividends
        return active_pf_value + cash_value

    def cvar_formula(self, weights: pd.DataFrame, dividend_exp: pd.DataFrame) -> float:
        """
        given weights and all the other parameters, it computes the cvar of portfolio. Takes into account effective
        returns (price_change + dividend/price) for each asset, as well as cash as an asset with return == cash_return,
        variance == 0, covariances == 0
        """
        alpha_es = .05
        'std dev of portfolio given weightings...'
        portfolio_risk = np.sqrt(np.dot(
            weights,  # weight will be updated passing it from main function or from parent...
            np.dot(self.covariances_matrix, weights.T)
        ))
        # dividend_exp is a parameter passed to all nodes in period, referring to next-period dividends...
        portfolio_return = sum((self.assets_data['a_i'] + dividend_exp) * self.assets_data['weight'])
        # you may want to pass dividend of same period and of the next when building portfolio...

        return (alpha_es ** -1) * norm.pdf(norm.ppf(alpha_es)) * portfolio_risk - portfolio_return

    def adjust_portfolio(self, weights: pd.DataFrame, dividend_exp: pd.DataFrame, parent_pf_data: pd.DataFrame):
        """
        takes CVaR objective and LCVaR and returns a portfolio with a CVaR in-between adjusting invested_cap/cash ratio
        returns number of shares for each asset, effective weights (against theoretical weights) and value results

        OPTIMIZE THIS FUNCTION!!!!
        """
        # 1) compute CVaR
        cVaR = self.cvar_formula(weights, dividend_exp)
        lCVaR = 0  # pass it globally fom optimization class...
        VaR = 0
        pf_value = self.compute_final_pf_value(parent_pf_data)
        if cVaR > lCVaR:
            delta_var = lCVaR - VaR  # you get it globally
            safety_net = 0.1  # goes from 0 -> 1: if 1, you go directly to VaR level, if 0 you stick to lCVaR, get it globally
            # cash_weight = 1 - ((lCVaR - delta_var*safety_net) / cVaR)
            # now compute rebalanced weights:
            # define the json of assets with rounded shares, effective weights, theoretical weights and cash data
            self.assets_data['weight'] = weights * (1-((lCVaR - delta_var*safety_net) / cVaR))  # theoretical weights
            self.assets_data['n_assets'] = self.assets_data['weight'] * pf_value // self.assets_data['close_prices_t']
            # the one below is not strictly necessary for now...
            # effective weights...
            # self.assets_data['weight'] = self.assets_data['close_prices_t'] * self.assets_data['n_assets'] / pf_value
            cash_amount = pf_value - sum(self.assets_data['n_assets'] * self.assets_data['close_prices_t'])
        else:
            """CVaR is respected and no adjustment to weights should be done..."""
            # create buffer for excess roundings:
            dsp_pf_value = pf_value * .99
            self.assets_data['weight'] = weights
            self.assets_data['n_assets'] = self.assets_data['weight'] * dsp_pf_value // self.assets_data['close_prices_t']
            cash_amount = pf_value - sum(self.assets_data['n_assets'] * self.assets_data['close_prices_t'])
        self.rebalancing_vector = deepcopy(self.assets_data[['weight', 'n_assets']])  # will be passed to siblings
        # self.assets_data['tot_value'] = self.assets_data['n_assets'] * self.assets_data['close_prices_t']
        # see if it works...
        self.rebalancing_vector.loc['cash'] = [
            {'weight': cash_amount/pf_value, 'shares': cash_amount, 'tot_value': cash_amount}
        ]




