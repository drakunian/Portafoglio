import pandas as pd
from Nodo import ScenarioNode
import concurrent.futures
from Tree import thread_process


def optimisation(parent=None, row_son=None):
    if parent is not None:
        for x in row_son:
            x.weight = x.weight + parent.weight
            print(x.coordinates, x.weight)
    else:
        for x in row_son:
            x.weight = x.weight + 100
            print(x.coordinates, x.weight)


def iteration_tree(tree):
    counter = 0
    horizon = 8
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

            pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)
            for index, row in matrix.iterrows():
                pool.submit(optimisation(index, row))

            # wait for all tasks to complete
            pool.shutdown(wait=True)

            # optimisation()
            print(matrix)

            parents = matrix.to_numpy().flatten()
            # parent = matrix.to_numpy().flatten()

        counter += 1

    '''value_list = []
    for el in matrix_T.to_numpy().flattern():
        v = el.rebalancing_vector['tot_value'].sum() * el.conditional_probability
        value_list.append(v)
    target_value = sum(value_list)'''


def convert_to_node(row):
    for i in range(len(row)):
        row[i] = ScenarioNode(init_data=row[i])


def convert_to_node_thread(row):
    return [ScenarioNode(init_data=el) for el in row]


def read_tree():
    """
    qui si riconvertono le celle da liste a ScenarioNode. Poi, si svilupperà la logica di iterazione lungo l'albero
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    tree = [pd.read_parquet(f'period_{x}') for x in range(8 + 1)]  # read parquet of all periods
    for matrix in tree:
        pool.submit(matrix.apply(lambda x: convert_to_node(x), axis=1))
        # matrix.apply(lambda x: convert_to_node(x), axis=1)
        #print(matrix)

    pool.shutdown(wait=True)
    iteration_tree(tree)
    #return tree

'''def read_tree():
    tree = [pd.read_parquet(f'period_{x}') for x in range(3 + 1)]  # read parquet of all periods
    i = 0
    for matrix in tree:
        # matrix.apply(lambda x: self.convert_to_node(x), axis=1)
        if i < 3:
            matrix.apply(lambda x: convert_to_node(x), axis=1)
            if i > 0:
                matrix.index = tree[i - 1].to_numpy().flatten()
        else:
            tree[i] = pd.DataFrame(thread_process(convert_to_node_thread, matrix))
            tree[i].index = tree[i - 1].to_numpy().flatten()
        print(tree[i])
        i+=1'''



tree = read_tree()
#iteration_tree(tree)
