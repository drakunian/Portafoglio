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

    The node by itself generates an executes all its methods in around 0.007422 seconds. So, how we can improve the
    tree generation? Is all in the optimal employment of RAM and Threads/processes at this point"""
    def __init__(
            self,
            root: bool,
            parent,  # dict | pd.DataFrame, parent_cond_prob: float,
            returns: [pd.DataFrame], cor_matrix: pd.DataFrame, period_date=None
    ):
        self.root = root

        self.parent_coordinates = parent.coordinates
        self.coordinates: tuple = None

        self.date = period_date  # to be used for index of new row of returns
        # self.parent = None  NON SERVE!
        self.assets_data = self.pass_asset_data(parent)  # modifies data of above parameters

        self.compute_returns(returns)
        # probabilities measurement inputs:
        self.conditional_volatilities = self.compute_variances()  # diagonal matrix of conditional variances
        self.conditional_covariances = self.compute_covariances(cor_matrix=cor_matrix)  # covariance matrix

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

    def pass_asset_data(self, parent):
        # filter assets data from parent!
        if self.root is False:
            # self.parent = parent
            # parent may be an id in the future...
            return parent.assets_data
        else:
            # instead of none, may generate an id in the future...
            return parent

    def compute_returns(self, returns: pd.DataFrame):
        """
        gets a random sample of returns from each assets, then adjusts the returns sampled to get unique values
        """
        # delegate dropna and columns iteration to init in tree
        sampled = [ret.sample().reset_index(drop=True) for ret in returns]
        self.assets_data['returns_t'] = pd.concat(sampled, axis=1).T
        self.assets_data['close_prices_t'] = self.assets_data['close_prices_t'] * (1 + self.assets_data['returns_t'])
        self.assets_data['residuals'] = self.assets_data['returns_t'] - self.assets_data['a_i']
        # now, check on prices, if they are absurd, we need to replace them i think...

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
            # WIP!!! THIS IS THE ULTIMATE PROBLEM TO SOLVE! really a dirty solution for now...
            self.assets_data.loc[self.assets_data['sigma_t'] < 0.00001, 'sigma_t'] = 0.00001  # CHANGE THAT!
            self.assets_data.loc[self.assets_data['sigma_t'] > 1.5, 'sigma_t'] = 0.01  # CHANGE THAT!
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
        cov_matrix = cov_matrix.where(np.triu(np.ones(cov_matrix.shape)).astype(np.bool_)).stack().reset_index()
        return cov_matrix.loc[cov_matrix['level_0'] != cov_matrix['level_1']]  # .set_index(['level_0', 'level_1'])

    def go_to_dict(self):
        """
        converts node into a dictionary to be passed to the dataframe!
        """
        return [{
            'node_coordinates': [self.parent_coordinates, self.coordinates],
            'assets_data': self.assets_data.to_json(orient='records'),
            'cond_covariances': self.conditional_covariances.to_json(orient='records'),
            'cond_probability': self.cond_probability
        }]



