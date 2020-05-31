# -*- coding:utf-8 -*-
# Created by LuoJie at 11/17/19

import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool

# cpu 数量
cores = cpu_count()
# 分块个数
partitions = cores


def parallelize(df, func):
    """
    多核并行处理模块
    :param df: DataFrame数据
    :param func: 预处理函数
    :return: 处理后的数据
    """
    data_split = np.array_split(df, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data
