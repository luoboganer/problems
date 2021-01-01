<!--
 * @Filename: 
 * @Author: shifaqiang
 * @Email: 14061115@buaa.edu.cn
 * @Github: https://github.com/luoboganer
 * @Date: 2020-09-05 11:29:59
 * @LastEditors: shifaqiang
 * @LastEditTime: 2021-01-01 12:05:33
 * @Software: Visual Studio Code
 * @Description: 1701-1800
-->

# 1701-1800

- [1705. 吃苹果的最大数目](https://leetcode-cn.com/problems/maximum-number-of-eaten-apples/)

	用优先队列来模拟每天吃一个苹果的过程

	```cpp
	int eatenApples(vector<int> &apples, vector<int> &days)
	{
		// 优先队列构造小顶堆
		priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> qe;
		int ret = 0, day = 0, n = apples.size();
		for (int i = 0; i < n || !qe.empty(); i++, day++)
		{
			// 今天新成熟的苹果加入优先队列中
			if (apples[i] > 0 && days[i] > 0)
			{
				// pair<int,int> 过期腐烂时间和数量
				qe.push(make_pair(day + days[i], apples[i]));
			}
			// 过期腐烂的扔掉
			while (!qe.empty() && day >= qe.top().first)
			{
				qe.pop();
			}
			// 如果今天有可吃的苹果，则吃掉
			if (!qe.empty() && day < qe.top().first)
			{
				ret++;
				auto x = qe.top();
				qe.pop();
				x.second -= 1;
				if (x.second > 0 && x.first >= day + 1)
				{
					qe.push(x); // 没吃完且明天不会过期的，继续放回去
				}
			}
		}
		return ret;
	}
	```

- [...](123)
