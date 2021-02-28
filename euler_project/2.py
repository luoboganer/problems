'''
Filename: 
Author: shifaqiang
Email: 14061115@buaa.edu.cn
Github: https://github.com/luoboganer
Date: 2020-12-17 16:37:11
LastEditors: shifaqiang
LastEditTime: 2020-12-17 16:52:36
Software: Visual Studio Code
Description: 
    求不超过four million的值为偶数的斐波那契数的和
'''


def solve():
    limit = 4e6
    ret = 0
    a, b = 0, 1
    while b <= limit:
        a, b = b, a + b
        if b % 2 == 0:
            ret += b
    return ret


print(solve())