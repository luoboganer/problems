'''
Filename: 
Author: shifaqiang
Email: 14061115@buaa.edu.cn
Github: https://github.com/luoboganer
Date: 2020-12-18 19:57:02
LastEditors: shifaqiang
LastEditTime: 2020-12-18 20:00:02
Software: Visual Studio Code
Description: 
'''

import math
import time


def solve():
    limit = int(2 * 1e6)
    flags = [True] * limit
    flags[0] = False
    flags[1] = False
    for v in range(2, int(math.sqrt(limit))):
        if flags[v]:
            for t in range(v * v, limit, v):
                flags[t] = False
    ret = 1
    for v in range(0, limit):
        if flags[v]:
            ret += v
    return ret


tic = time.time()
print(solve())
print(f'Time:{time.time() - tic:.6} seconds.')