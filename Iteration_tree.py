import os
# print('PYTHONPATH="%s"' % os.environ['PATH'])
import math
import pandas as pd
from Nodo import ScenarioNode
from Tripla import Tripla
import concurrent.futures
import timeit
import numpy as np
from docplex.mp.model import Model
from scipy.stats import norm
import re


def iteration_tree(tree):
    counter = 0
    horizon = 9
    i = 0
    parents = tree[0].to_numpy().flatten()
    while counter <= horizon:
        matrix = tree[counter]

        if counter == 0:
            print(type(matrix.values[0]))
            print(matrix.info())
            optimisation(None, matrix.values[0])
            parents = matrix.to_numpy().flatten()
        else:
            matrix.set_index(pd.Index(parents), inplace=True)

            pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            for index, row in matrix.iterrows():
                pool.submit(optimisation(index, row))

            pool.shutdown(wait=True)

            print(matrix)

            parents = matrix.to_numpy().flatten()

        counter += 1


def convert_to_node(x, model, lCVaR, cVaR, weights, dividend_exp: np.array, value, counter):
    if x.name == 0:
        variable = value
    else:
        variable = [model.continuous_var(lb=0, name=f'parent{i}_{counter-1}') for i in range(4)]
        for i in range(4):
            model.add_constraint(variable[i] == x.name[i])


    for i in range(len(x)):
        print('name')
        name_str = str(i)

        '''if i == 0:
            name_str = str(value)'''
        x[i] = [ScenarioNode(init_data=x[i]).compute_opt_parameters(model, lCVaR, cVaR, weights, dividend_exp, variable)]
        print(f'x[i]: {x[i]}')


        valore_numerico = []
        cplex_variable = []
        for item in x[i]:

            print(type(item))
            for key, value in item.items():
                # print(f'{value}')
                # valore_numerico.append(re.search(r'\d+\.\d+', value.to_string()).group())
                cplex_variable.append(value)
        # x[i] = Tripla(valore_numerico[0], valore_numerico[1], valore_numerico[2])
        x[i] = Tripla(cplex_variable[0], cplex_variable[1], cplex_variable[2], cplex_variable[3])

        print(f'x {x}')


def read_tree():
    """
    qui si riconvertono le celle da liste a ScenarioNode. Poi, si svilupperà la logica di iterazione lungo l'albero
    """
    horizon = 9
    with Model('example') as model:
        weights = [model.continuous_var(lb=0, ub=1, name=f'w_{i}') for i in range(3)]
        lCVaR = 0.11
        cVaR = 0.10
        value = 10000
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

        tree = [pd.read_parquet(f'period_{x}') for x in range(2 + 1)]  # read parquet of all periods
        dividend_exp = np.zeros(shape=(horizon, 3))
        print(dividend_exp)
        parents = None
        counter = 0
        for matrix in tree:

            if counter > 0:
                matrix.index = parents
            counter += 1

            pool.submit(
                matrix.apply(lambda x: convert_to_node(x, model, lCVaR, cVaR, weights, dividend_exp[counter], value, counter),
                             axis=1))
            parents = matrix.to_numpy().flatten()
            print(type(matrix))
            print("------------------------------------------------------------------------")

        pool.shutdown(wait=True)
    return tree


start = timeit.default_timer()
tree = read_tree()
# iteration_tree(tree)

stop = timeit.default_timer()
minutes = (stop - start) / 60
seconds, minutes = math.modf(minutes)
print(f'{minutes} minuti e {round(seconds * 60)} secondi')
