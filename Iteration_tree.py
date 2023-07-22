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

def convert_to_node(x, model, lCVaR, cVaR, weights, dividend_exp: np.array, value, counter, alpha):
    print(counter)
    if x.name == 0:
        variable = value
    else:
        variable = [model.continuous_var(lb=0, name=f'parent{i}_{counter-1}') for i in range(4)]
        for i in range(4):
            model.add_constraint(variable[i] == x.name[i])


    for i in range(len(x)):
        #print('name')
        name_str = str(i)

        horizon = 3
        cond_probability = x[i][0]['cond_probability']
        x[i] = [ScenarioNode(init_data=x[i]).compute_opt_parameters(model, lCVaR, cVaR, weights, dividend_exp, variable, alpha)]



        valore_numerico = []
        cplex_variable = []
        for item in x[i]:
            for key, value in item.items():
                cplex_variable.append(value)
                print(value)


        if counter != horizon:
            x[i] = Tripla(cplex_variable[0], cplex_variable[1], cplex_variable[2], cplex_variable[3])
        else:
            print(x[i])
            #print(Tripla(cplex_variable[0], cplex_variable[1], cplex_variable[2], cplex_variable[3]))
            #x[i] = model.sum([Tripla(cplex_variable[0], cplex_variable[1], cplex_variable[2], cplex_variable[3]), cond_probability])
            x[i] = model.sum(el for el in cplex_variable) * cond_probability

        #print(f'x {x}')


def read_tree(lb, ub, alpha, VaR_list, LCVar, cash):
    """
    qui si riconvertono le celle da liste a ScenarioNode. Poi, si svilupperà la logica di iterazione lungo l'albero
    """
    horizon = 9
    with Model('example') as model:
        weights = [model.continuous_var(lb=lb, ub=ub, name=f'w_{i}') for i in range(3)]
        lCVaR = LCVar
        cVaR = VaR_list
        value = cash
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
                matrix.apply(lambda x: convert_to_node(x, model, lCVaR, cVaR, weights, dividend_exp[counter], value, counter, alpha),axis=1))
            parents = matrix.to_numpy().flatten()
            print("------------------------------------------------------------------------")

        pool.shutdown(wait=True)

        matrix = tree[-1].to_numpy().flatten()

        model.set_objective('max', model.sum(el for el in matrix))
        solution = model.solve()
    print(solution.get_objective_value())

    #return tree

def optimisation(tree):
    print(tree)
    matrix = tree.to_numpy().flatten()

    model



#start = timeit.default_timer()
#read_tree()
##optimisation(tree[-1])
## iteration_tree(tree)
#
#
#stop = timeit.default_timer()
#minutes = (stop - start) / 60
#seconds, minutes = math.modf(minutes)
#print(f'{minutes} minuti e {round(seconds * 60)} secondi')
