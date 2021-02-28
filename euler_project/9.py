'''
Filename: 
Author: shifaqiang
Email: 14061115@buaa.edu.cn
Github: https://github.com/luoboganer
Date: 2020-12-18 19:54:28
LastEditors: shifaqiang
LastEditTime: 2020-12-18 19:55:51
Software: Visual Studio Code
Description: 
'''


def solve():
    for a in range(1, 1000):
        for b in range(1, 1000 - a):
            c = 1000 - a - b
            if a * a + b * b == c * c:
                return a * b * c
    return None


print(solve())