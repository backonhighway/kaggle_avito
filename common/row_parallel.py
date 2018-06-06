import pandas as pd
import numpy as np
from multiprocessing import Pool


class GoldenParallelProcessor:
    def __init__(self, df, function_list, log_func_name_list, num_core, timer):
        self.df = df
        self.function_list = function_list
        self.log_func_name_list = log_func_name_list
        self.timer = timer
        self.num_core = num_core

    def do_process(self):
        for some_func, name in zip(self.function_list, self.log_func_name_list):
            df_list = np.array_split(self.df, self.num_core * 3)
            pool = Pool(self.num_core)
            self.df = pd.concat(pool.map(some_func, df_list))
            pool.close()
            pool.join()

            self.timer.time("done " + name)

