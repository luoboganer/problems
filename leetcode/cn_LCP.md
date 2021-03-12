<!--
 * @Filename: 
 * @Author: shifaqiang
 * @Email: 14061115@buaa.edu.cn
 * @Github: https://github.com/luoboganer
 * @Date: 2020-09-05 11:29:59
 * @LastEditors: shifaqiang
 * @LastEditTime: 2021-03-10 14:18:01
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

- [LCP 11. 期望个数统计](https://leetcode-cn.com/problems/qi-wang-ge-shu-tong-ji/submissions/)

    数学问题，在任意n个数的所有全排列中，每个位置的值相同的期望为1

    ```cpp
	int expectNumber(vector<int> &scores)
	{
		/**
		 * 数学上可以证明，m个人的全排列A(m,m)中与[1,2,3,...,m]相同的期望为1
		*/
		const int n = scores.size();
		int count = 0;
		if (n > 0)
		{
			sort(scores.begin(), scores.end());
			count = 1;
			for (int i = 1; i < n; i++)
			{
				if (scores[i] != scores[i - 1])
				{
					count++;
				}
			}
		}
		return count;
	}
    ```

- [LCP 17. 速算机器人](https://leetcode-cn.com/problems/nGK0Fy/)

    按给定计算指令模拟，时间复杂度$O(n)$，其中$n=s.length()$

    ```cpp
	int calculate(string s)
	{
		int x = 1, y = 0;
		for (auto &ch : s)
		{
			ch == 'A' ? x = (x << 1) + y : y = (y << 1) + x;
		}
		return x + y;
	}
    ```

- [...](123)
