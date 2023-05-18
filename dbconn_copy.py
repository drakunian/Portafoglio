import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from psycopg2.extensions import cursor


desired_width = 500
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 500)
np.set_printoptions(linewidth=desired_width)


# database interaction functions:
class DBConnection:
    """Just create the connection to given database, and closes it when requested"""
    def __init__(self, host="70.34.223.212", database='hydroponics', user="postgres", password='moneyproj01'):
        self.conn = psycopg2.connect(host=host, database=database, user=user, password=password)


class DBOperations:
    """all operations like create table, add-update... called from here. It needs the DBConnection and the schema to
    which the tables ar part of"""
    def __init__(self, connection:DBConnection().conn or None=None, schema='public'):
        self.conn = connection
        self.schema = schema

        if self.conn is False:
            print('Connection failed, retry')

    def database_infos(self):
        """returns infos about selected tables of the connected database, giving memory usage, and other data"""
        pass

    def check_table_existence(self, table):
        cur = self.conn.cursor()
        cur.execute(f"""SELECT EXISTS 
        (SELECT FROM pg_tables WHERE schemaname = '{self.schema}' AND tablename = '{table}')""")
        return bool(cur.rowcount)

    def create_table(self, name, columns, tp="TABLE"):
        cur = self.conn.cursor()
        coln = str('[%s]' % ', '.join(map(str, columns)))[1:-1]
        cur.execute(f"""CREATE {tp} {self.schema}.{name} ({coln})""")
        self.conn.commit()

    def add_values(self, table, dataframe):
        """this function adds data from dataframe to database"""
        dataframe = dataframe.replace("'", '_', regex=True).astype(str)
        cur = self.conn.cursor()
        coln = str('[%s]' % ', '.join(map(str, dataframe.columns.tolist())))[1:-1]
        tuple_rows = [tuple(row) for i, row in dataframe.iterrows()]
        str_data = str('[%s]' % ', '.join(map(str, tuple_rows)))[1:-1]
        cur.execute(f"INSERT INTO {self.schema}.{table} ({coln}) VALUES {str_data}")
        self.conn.commit()

    # to keep update functions:
    def update_values_simple(self, table, dataframe, where=""):
        lnt = range(len(dataframe.columns))
        # preparing the new_data to be updated:
        new_values = tuple(dataframe.itertuples(index=False, name=None))
        # preparing the query data:
        columns = str('[%s]' % ', '.join(map(str, dataframe.columns.tolist())))[1:-1]
        ref = [dataframe.columns.tolist()[i] + " = " + 'e.' + dataframe.columns.tolist()[i] for i in lnt if
               dataframe.columns.tolist()[i] != where]
        refactored = str('[%s]' % ', '.join(map(str, ref)))[1:-1]
        update_query = f"""UPDATE {self.schema}.{table} AS t SET {refactored} FROM (VALUES %s) AS e({columns})
                          WHERE e.{where} = t.{where};"""
        # updating columns:
        psycopg2.extras.execute_values(self.conn.cursor(), update_query, new_values, template=None, page_size=200)
        self.conn.commit()

    def multi_value_fetch(self, *executions):
        """return a list of dataframes for each execution requested. The format for the execution is:
        execution = [[list of desired columns], table_name]. With that function, you will just get all values from the
        columns you requested, for each execution"""
        cur = self.conn.cursor()
        panda_list = []
        for execution in executions:
            columns = execution[0]
            coln = str('[%s]' % ', '.join(map(str, columns)))[1:-1]
            cur.execute(f"""SELECT {coln} FROM {self.schema}.{execution[1]};""")
            value = cur.fetchall()
            array = np.empty(shape=(len(value), len(columns)), dtype=object)
            for i in range(len(value)):
                array[i] = value[i]
            pdf = pd.DataFrame(array, columns=columns)
            panda_list.append(pdf)
        return panda_list

    def multi_fetch_where(self, *executions):
        """return values that respond to the same parameter(s); it works for each execution; which format must be:
        [execution = [list of desired columns], table_name, [list of constrains]]. With that function, you will just get
        all values from the columns you requested, for each execution. This is the constrains syntax:
        [column_name = 'stuff' AND column_name = 'stuff' AND ... OR ... etc]; might also use OR instead of AND, but
        don't know how to combine efficiently the two together..."""
        # exe = ["name", "total", "generation"], "pokedex", ["type1 = 'Grass'" + " AND " + "type2 = 'Poison'"]
        cur = self.conn.cursor()
        panda_list = []
        for execution in executions:
            columns, constraints = execution[0], execution[2]
            coln = str('[%s]' % ', '.join(map(str, columns)))[1:-1]
            """
            devi passare una lista di constraints. Poiché negli input avrai execution = [[cols], tab] + const, se const
            è una lista di questo tipo: [f"stock_id IN {assets_list}"], otterrai: [[cols], tab, f"stock_id IN {assets_list}"],
            mentre vorresti [[cols], tab, [f"stock_id IN {assets_list}"]], che puoi ottenero se scrivi i constraint come
            una lista dentro una lista: [[f"stock_id IN {assets_list}"]]. 
            """
            cons = str('[%s]' % ', '.join(map(str, constraints)))[1:-1]  # rende leggibili i constraint per postgresql
            cons = cons.replace('[', '(')  # e qui toglie evetuali parentesi quadre rendendole tonde
            cons = cons.replace(']', ')')
            print(cons)
            cur.execute(f"""SELECT {coln} FROM {self.schema}.{execution[1]} WHERE {cons};""")
            value = cur.fetchall()
            array = np.empty(shape=(len(value), len(columns)), dtype=object)
            for i in range(len(value)):
                array[i] = value[i]
            pdf = pd.DataFrame(array, columns=columns)
            panda_list.append(pdf)
        return panda_list

    def delete_data(self, table, constraints=None):
        """constraints: [column.name = 'value'; col1.name = 'v1' AND ...; column.name IN (list)/ NOT IN (list)]"""
        cur = self.conn.cursor()
        if constraints is not None:
            cons = str('[%s]' % ', '.join(map(str, constraints)))[1:-1]
            cur.execute(f"""DELETE FROM {self.schema}.{table} WHERE {cons};""")
        else:
            cur.execute(f"""DELETE FROM {self.schema}.{table};""")
        self.conn.commit()

    def delete_null(self, table, null_column=None):
        """constraints: [column.name = 'value'; col1.name = 'v1' AND ...; column.name IN (list)/ NOT IN (list)]"""
        cur = self.conn.cursor()
        cur.execute(f"""DELETE FROM {self.schema}.{table} WHERE {null_column} is NULL;""")
        self.conn.commit()