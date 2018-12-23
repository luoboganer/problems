# some notebook in solve problems in [hanker](https://www.hackerrank.com/)

## about [cpp](https://www.hackerrank.com/domains/cpp)

## about [algorithms](https://www.hackerrank.com/domains/algorithms)

## about [python](https://www.hackerrank.com/domains/python)

- [problem 1-9] exercies about python
- [problem 10] [Triangle Quest](https://www.hackerrank.com/challenges/python-quest-1/problem)

    学到了一个数学公式，从数字"i"变到数字"iiii.."(i出现i次)有公式
    $$pow(10,i)//9*i$$
    例如
    $$\left\{\begin{matrix}
    1 -> 1\\ 
    2 -> 22\\ 
    3 -> 333\\ 
    4 -> 4444\\ 
    5 -> 55555
    \end{matrix}\right.$$
- [list comprehensions](https://www.hackerrank.com/challenges/list-comprehensions/problem)
  
    Python具有的列表解析特性

- [polar coordinates](https://www.hackerrank.com/challenges/polar-coordinates/problem)

    本题通过极坐标的解析引入了Python中的数学包cmath的介绍

- [string tutorial]

    字符串string的一些神奇操作函数

      - 1. 字符串的左右居中对齐，str.rjust(),str.ljust(),str.center()
      - 2. 字符串wrap的textwrap包有很多实用的工具
      - 3. 字符串的分割与联合，str.split()与' '.join(array)

## about [projecteuler](https://www.hackerrank.com/contests/projecteuler/challenges)
- [problem 2] [Even Fibonacci numbers](https://www.hackerrank.com/contests/projecteuler/challenges/euler002)
  
  计算斐波那契数数列中的偶数的和，给定斐波那契数列数列：
    $$F(n)=\left\{\begin{matrix}
    0,n=1\\ 
    1,n=2\\ 
    F(n-1)+F(N-2),n>2
    \end{matrix}\right.$$
  则有数列$0,1,1,2,3,5,8,13,21,34,55,89,144,...$，可以观察到规律每隔三个数出现一个偶数，单独取出其中的偶数组成数列$E(n)$有$0,2,8,34,144,...$，则：$E(n)=4*E(n-1)+E(n-2)$，这个公式也可以在数学上被证明。