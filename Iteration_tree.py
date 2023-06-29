import pandas as pd
from Nodo import ScenarioNode
import concurrent.futures


def optimisation(parent=None, row_son=None):
    if parent is not None:
        for x in row_son:
            x.weight = x.weight + parent.weight
            print(x.son_coordinates, x.weight)
    else:
        for x in row_son:
            x.weight = x.weight + 100
            print(x.son_coordinates, x.weight)


def iteration_tree(tree):
    counter = 0
    horizon = 3
    parents = tree[0].to_numpy().flatten()
    while counter <= horizon:
        matrix = tree[counter]

        if counter == 0:
            optimisation(None, tree[0].values[0])
            parents = matrix.to_numpy().flatten()
        else:
            matrix.set_index(pd.Index(parents), inplace=True)

            pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            for index, row in matrix.iterrows():
                pool.submit(optimisation(index, row))

            # wait for all tasks to complete
            pool.shutdown(wait=True)

            # optimisation()
            print(matrix)

            parents = matrix.to_numpy().flatten()
            # parent = matrix.to_numpy().flatten()

        counter += 1


def convert_to_node(x):
    for i in range(len(x)):
        x[i] = ScenarioNode(init_data=x[i])


def read_tree():
    """
    qui si riconvertono le celle da liste a ScenarioNode. Poi, si svilupperà la logica di iterazione lungo l'albero
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    tree = [pd.read_parquet(f'period_{x}.parquet') for x in range(3 + 1)]  # read parquet of all periods
    for matrix in tree:
        pool.submit(matrix.apply(lambda x: convert_to_node(x), axis=1))
        # matrix.apply(lambda x: convert_to_node(x), axis=1)
        # print(matrix)

    pool.shutdown(wait=True)
    return tree


tree = read_tree()
iteration_tree(tree)
