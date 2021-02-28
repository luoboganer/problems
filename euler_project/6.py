'''
Filename: 
Author: shifaqiang
Email: 14061115@buaa.edu.cn
Github: https://github.com/luoboganer
Date: 2020-12-18 00:29:12
LastEditors: shifaqiang
LastEditTime: 2020-12-18 19:06:01
Software: Visual Studio Code
Description: 
'''


def solve():
    return sum(range(101)) * sum(range(101)) - sum([i * i for i in range(101)])


print(solve())
