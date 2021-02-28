'''
Filename: 
Author: shifaqiang
Email: 14061115@buaa.edu.cn
Github: https://github.com/luoboganer
Date: 2020-12-18 00:21:31
LastEditors: shifaqiang
LastEditTime: 2020-12-18 00:24:22
Software: Visual Studio Code
Description: 
'''


def solve():
    ret = 0
    left, right = 100, 1000
    for a in range(left, right):
        for b in range(left, right):
            v = a * b
            v_str = str(v)
            if v_str == v_str[::-1]:
                ret = max(ret, v)
    return ret


print(solve())