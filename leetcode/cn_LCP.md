<!--
 * @Filename: 
 * @Author: shifaqiang
 * @Email: 14061115@buaa.edu.cn
 * @Github: https://github.com/luoboganer
 * @Date: 2020-09-05 11:29:59
 * @LastEditors: shifaqiang
 * @LastEditTime: 2020-10-19 11:30:39
 * @Software: Visual Studio Code
 * @Description: leetcode Cn LCP 部分题目
-->

# 剑指Offer:名企面试官精讲典型编程题

- [LCP 02. 分式化简](https://leetcode-cn.com/problems/deep-dark-fraction/)

    从最后一项开始递归化简，时间复杂度$O(n)$

    ```cpp
    class Solution
    {
    private:
        int gcd(int a, int b)
        {
            if (a < b)
            {
                swap(a, b);
            }
            while (b)
            {
                int r = a % b;
                a = b;
                b = r;
            }
            return a;
        }

    public:
        vector<int> fraction(vector<int> &cont)
        {
            const int n = cont.size();
            cont.emplace_back(1); // for the last item , n= n/1
            for (int i = n - 2; i >= 0; i--)
            {
                int a = cont[i], b = cont[i + 1], c = cont[i + 2];
                a = a * b + c;
                int gcd_ab = gcd(a, b);
                cont[i] = a / gcd_ab, cont[i + 1] = b / gcd_ab;
            }
            return {cont[0], cont[1]};
        }
    };
    ```

- [...](123)
