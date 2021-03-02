<!--
 * @Filename: 
 * @Author: shifaqiang
 * @Email: 14061115@buaa.edu.cn
 * @Github: https://github.com/luoboganer
 * @Date: 2020-09-05 11:29:59
 * @LastEditors: shifaqiang
 * @LastEditTime: 2021-03-02 14:15:45
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

- [1722. 执行交换操作后的最小汉明距离](https://leetcode-cn.com/problems/minimize-hamming-distance-after-swap-operations/)

	用并查集来标记所有可以任意交换的节点构成一个个的联通分量，在每一个联通分量中所有值可以任意交换，则其汉明距离即为该联通分量的节点数减去其相同值（匹配数）的个数，时间复杂度$O(m+n)$，其中$m=source.size(),n=allowedSwaps.size()$

	```cpp
	class Solution
	{
	private:
		struct UF
		{
			int count;
			vector<int> uf;
			UF(int n)
			{
				count = n;
				uf.resize(n);
				for (int i = 0; i < n; i++)
				{
					uf[i] = i;
				}
			}
			int find(int x)
			{
				return uf[x] == x ? x : (uf[x] = find(uf[x]));
			}
			bool union_merge(int x, int y)
			{
				x = find(x), y = find(y);
				if (x != y)
				{
					uf[x] = y;
					count--;
					return true;
				}
				return false;
			}
		};

	public:
		int minimumHammingDistance(vector<int> &source, vector<int> &target, vector<vector<int>> &allowedSwaps)
		{
			const int n = source.size();
			UF uf = UF(n);
			for (auto &e : allowedSwaps)
			{
				uf.union_merge(e[0], e[1]);
			}
			unordered_map<int, vector<int>> idToIndexs;
			for (int i = 0; i < n; i++)
			{
				idToIndexs[uf.find(i)].emplace_back(i);
			}
			int ret = 0; // 记录汉明距离
			for (auto &item : idToIndexs)
			{
				int countEqual = 0;
				vector<int> idxs = item.second;
				const int part_n = idxs.size();
				vector<int> part_source(part_n), part_target(part_n);
				for (int i = 0; i < part_n; i++)
				{
					part_source[i] = source[idxs[i]];
					part_target[i] = target[idxs[i]];
				}
				sort(part_source.begin(), part_source.end());
				sort(part_target.begin(), part_target.end());
				int i = 0, j = 0;
				while (i < part_n && j < part_n)
				{
					if (part_source[i] == part_target[j])
					{
						countEqual++;
						i++, j++;
					}
					else if (part_source[i] < part_target[j])
					{
						i++;
					}
					else
					{
						j++;
					}
				}
				ret += part_n - countEqual;
			}
			return ret;
		}
	};
	```

- [1728. 重新排列后的最大子矩阵](https://leetcode-cn.com/problems/largest-submatrix-with-rearrangements/)

	将每个点向上统计连续1的个数，即可转化为[84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)类问题，而每一列可以任意交换，则直接对每一行中所有可能的直方图高度直接从高到低排序即可，时间复杂度$O(rows*cols)$

	```cpp
	int largestSubmatrix(vector<vector<int>> &matrix)
	{
		int ret = 0;
		if (matrix.size() > 0 && matrix[0].size() > 0)
		{
			int rows = matrix.size(), cols = matrix[0].size();
			for (int i = 1; i < rows; i++)
			{
				for (int j = 0; j < cols; j++)
				{
					if (matrix[i][j])
					{
						matrix[i][j] += matrix[i - 1][j];
					}
				}
			}
			for (auto &row : matrix)
			{
				sort(row.rbegin(), row.rend());
				for (int i = 0; i < cols; i++)
				{
					ret = max(ret, (i + 1) * row[i]);
				}
			}
		}
		return ret;
	}
	```

- [](https://leetcode-cn.com/problems/minimum-number-of-operations-to-move-all-balls-to-each-box/)

	用数组前缀和的方法分别计算将第i个位置左侧、右侧的小球全部移动到该位置的花费，左右求和即可，时间复杂度$O(n)$

	```cpp
	vector<int> minOperations(string boxes)
	{
		const int n = boxes.size();
		vector<int> ret(n);
		if (n > 0)
		{
			int left_count = 0, right_count = 0;
			vector<int> left_cost(n + 1, 0), right_cost(n + 1, 0);
			for (int i = 0; i < n; i++)
			{
				left_cost[i + 1] = left_cost[i] + left_count;
				right_cost[n - i - 1] = right_cost[n - i] + right_count;
				boxes[i] == '1' ? left_count++ : 0;
				boxes[n - i - 1] == '1' ? right_count++ : 0;
			}
			for (int i = 0; i < n; i++)
			{
				ret[i] = right_cost[i] + left_cost[i + 1];
			}
		}
		return ret;
	}
	```

- [...](123)
