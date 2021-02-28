'''
Filename: 
Author: shifaqiang
Email: 14061115@buaa.edu.cn
Github: https://github.com/luoboganer
Date: 2020-12-18 19:31:17
LastEditors: shifaqiang
LastEditTime: 2020-12-18 19:39:24
Software: Visual Studio Code
Description: 
    10 001st prime number
'''
import math


def solve():
    limit = int(1e6)
    flags = [True] * limit
    flags[0] = False
    flags[1] = False
    for v in range(2, int(math.sqrt(limit))):
        if flags[v]:
            for t in range(v * v, limit, v):
                flags[t] = False
    ret = []
    for v in range(0, limit):
        if flags[v]:
            ret.append(v)
    # print(ret)
    print(f'{len(ret)} prime numbers')
    print(f'10001st prime number is {ret[10000]}')
    return ret


solve()