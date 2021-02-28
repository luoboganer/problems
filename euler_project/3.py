'''
Filename: 
Author: shifaqiang
Email: 14061115@buaa.edu.cn
Github: https://github.com/luoboganer
Date: 2020-12-17 16:53:28
LastEditors: shifaqiang
LastEditTime: 2020-12-18 00:18:31
Software: Visual Studio Code
Description: 

The prime factors of 13195 are 5, 7, 13 and 29.
What is the largest prime factor of the number 600851475143 ?

'''
import math


def isprime(n):
    for v in range(2, int(math.sqrt(n)) + 1):
        if n % v == 0:
            return False
    return True


def solve():
    num = 600851475143
    for v in range(int(math.sqrt(num)), 2, -1):
        if num % v == 0 and isprime(v):
            return v
    return 1


print(solve())
