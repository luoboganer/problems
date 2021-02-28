'''
Filename: 
Author: shifaqiang
Email: 14061115@buaa.edu.cn
Github: https://github.com/luoboganer
Date: 2020-12-18 00:25:41
LastEditors: shifaqiang
LastEditTime: 2020-12-18 00:28:05
Software: Visual Studio Code
Description: 
'''


def gcd(a, b):
    if a < b:
        a, b = b, a
    while b != 0:
        a, b = b, a % b
    return a


def lcd(a, b):
    return a * b // gcd(a, b)


def solve():
    ret = 1
    for v in range(1, 20 + 1):
        ret = lcd(ret, v)
    return ret


print(solve())
