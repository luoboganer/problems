'''
Filename: 
Author: shifaqiang
Email: 14061115@buaa.edu.cn
Github: https://github.com/luoboganer
Date: 2020-12-17 14:05:55
LastEditors: shifaqiang
LastEditTime: 2020-12-17 14:27:36
Software: Visual Studio Code
Description: 
    https://projecteuler.net/ 欧拉工程，数学问题
'''


# 求[1,1000)以内是3和5的倍数的自然数之和
def solve1():
    ret = 0
    for v in range(1000):
        if v % 3 == 0 or v % 5 == 0:
            ret += v
    return ret


def solve2():
    right_bound = 999
    a = right_bound // 3
    b = right_bound // 5
    c = right_bound // 15
    return (1 + a) * a // 2 * 3 + (1 + b) * b // 2 * 5 - (1 + c) * c // 2 * 15


print(solve1())
print(solve2())