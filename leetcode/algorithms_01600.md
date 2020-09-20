# 1501-1600

- [1519. Number of Nodes in the Sub-Tree With the Same Label](https://leetcode.com/problems/number-of-nodes-in-the-sub-tree-with-the-same-label/)

	DP + DFS-topdown，时间复杂度$O(n)$

	```cpp
	class Solution
	{
	private:
		void dfs(vector<vector<int>> &graph, vector<vector<int>> &dp, vector<int> &ret, string &labels, int node, int parent)
		{
			int cur_label = static_cast<int>(labels[node] - 'a');
			dp[node][cur_label] += 1;
			if (!graph[node].empty())
			{
				for (auto &&subnode : graph[node])
				{
					if (subnode != parent)
					{
						dfs(graph, dp, ret, labels, subnode, node);
						for (auto i = 0; i < 26; i++)
						{
							dp[node][i] += dp[subnode][i];
						}
					}
				}
			}
			ret[node] = dp[node][cur_label];
		}

	public:
		vector<int> countSubTrees(int n, vector<vector<int>> &edges, string labels)
		{
			// 建立邻接链表
			vector<vector<int>> graph(n);
			for (auto &&e : edges)
			{
				graph[e[0]].push_back(e[1]);
				graph[e[1]].push_back(e[0]);
			}
			// dp[i][ch] 节点i及其子树上的字母ch数量
			vector<vector<int>> dp(n, vector<int>(26, 0));
			vector<int> ret(n, 0);
			dfs(graph, dp, ret, labels, 0, -1);
			return ret;
		}
	};
	```

	- some test case

	```cpp
	7
	[[0,1],[0,2],[1,4],[1,5],[2,3],[2,6]]
	"abaedcd"
	4
	[[0,1],[1,2],[0,3]]
	"bbbb"
	5
	[[0,1],[0,2],[1,3],[0,4]]
	"aabab"
	6
	[[0,1],[0,2],[1,3],[3,4],[4,5]]
	"cbabaa"
	7
	[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6]]
	"aaabaaa"
	4
	[[0,2],[0,3],[1,2]]
	"aeed"
	```

- [1524. Number of Sub-arrays With Odd Sum](https://leetcode.com/problems/number-of-sub-arrays-with-odd-sum/)

	DP，时间复杂度$O(n)$

	```cpp
	int numOfSubarrays(vector<int> &arr)
	{
		const int n = arr.size(), mod = 1e9 + 7;
		for (int i = 0; i < n; i++)
		{
			arr[i] %= 2; // 奇偶问题可以转化为0/1问题
		}
		vector<int> dp_zero(n + 1, 0), dp_one(n + 1, 0);
		for (int i = n - 1; i >= 0; i--)
		{
			if (arr[i] == 1)
			{
				dp_zero[i] = dp_one[i + 1];
				dp_one[i] = (dp_zero[i + 1] + 1) % mod;
			}
			else
			{
				dp_zero[i] = (dp_zero[i + 1] + 1) % mod;
				dp_one[i] = dp_one[i + 1];
			}
		}
		int ret = 0;
		for (auto &&v : dp_one)
		{
			ret = (ret + v) % mod;
		}
		return ret;
	}
	```

- [1553. Minimum Number of Days to Eat N Oranges](https://leetcode.com/problems/minimum-number-of-days-to-eat-n-oranges/)

	- DP，时间复杂度$O(n)$，leetcode评测机$\color{red}{Out of memory}$

	```cpp
	int minDays(int n)
	{
		vector<int> dp(n + 1, 0);
		for (int i = 1; i <= n; i++)
		{
			dp[i] = dp[i - 1] + 1;
			if (i % 2 == 0)
			{
				dp[i] = min(dp[i], dp[i / 2] + 1);
			}
			if (i % 3 == 0)
			{
				dp[i] = min(dp[i], dp[i / 3] + 1);
			}
		}
		return dp.back();
	}
	```

	- hashmap优化，时间复杂度$O(n)$

	```cpp
	int minDays(int n)
	{
		unordered_map<int, int> n2v;
		queue<int> bfs;
		bfs.push(n);
		n2v[n] = 0;
		while (true)
		{
			int cur = bfs.front();
			bfs.pop();
			int a = cur - 1, v = n2v[cur] + 1;
			if (n2v.find(a) == n2v.end())
			{
				n2v[a] = v;
				bfs.push(a);
			}
			if (cur % 2 == 0)
			{
				int b = cur / 2;
				if (n2v.find(b) == n2v.end())
				{
					n2v[b] = v;
					bfs.push(b);
				}
			}
			if (cur % 3 == 0)
			{
				int c = cur / 3;
				if (n2v.find(c) == n2v.end())
				{
					n2v[c] = v;
					bfs.push(c);
				}
			}
			if (n2v.find(0) != n2v.end())
			{
				return n2v[0];
			}
		}
		return 0;
	}
	```

- [1559. Detect Cycles in 2D Grid](https://leetcode.com/problems/detect-cycles-in-2d-grid/)

	- DFS搜索，时间复杂度$O(m*n)$

	```cpp
	class Solution
	{
	private:
		vector<vector<char>> grid;
		vector<vector<bool>> vis;
		int m, n;
		const vector<int> directions{1, 0, -1, 0, 1};
		bool dfs(int r, int c, int x, int y, char ch)
		{
			bool ret = false;
			if (r >= 0 && c >= 0 && r < m && c < n && grid[r][c] == ch)
			{
				if (vis[r][c])
				{
					return true; // 形成了环
				}
				vis[r][c] = true;
				for (auto i = 0; !ret && i < 4; i++)
				{
					int ni = r + directions[i], nj = c + directions[i + 1];
					if (ni != x || nj != y) // 不是aba式的环
					{
						ret = dfs(ni, nj, r, c, ch);
					}
				}
			}
			return ret;
		}

	public:
		bool containsCycle(vector<vector<char>> &grid)
		{
			this->grid = grid;
			bool ret = false;
			if (grid.size() > 0 && grid[0].size() > 0)
			{
				m = grid.size(), n = grid[0].size();
				vis.resize(m, vector<bool>(n, false));
				for (auto i = 0; !ret && i < m; i++)
				{
					for (auto j = 0; !ret && j < n; j++)
					{
						if (!vis[i][j])
						{
							ret = dfs(i, j, -1, -1, grid[i][j]);
						}
					}
				}
			}
			return ret;
		}
	};
	```

- [1560. Most Visited Sector in a Circular Track](https://leetcode.com/problems/most-visited-sector-in-a-circular-track/)

	- 暴力模拟，时间复杂度$O(n^2)$

	```cpp
	vector<int> mostVisited(int n, vector<int> &rounds)
	{
		vector<int> visited(n + 1, 0), ret;
		int m = rounds.size() - 1;
		visited[rounds[0]]++; // 起点
		for (int i = 0; i < m; i++)
		{
			int start = rounds[i], length = ((rounds[i + 1] - start) % n + n) % n;
			for (auto j = 0; j < length; j++)
			{
				visited[(start + j) % n + 1]++;
			}
		}
		int max_values = *max_element(visited.begin() + 1, visited.end());
		for (int i = 1; i <= n; i++)
		{
			if (visited[i] == max_values)
			{
				ret.push_back(i);
			}
		}
		return ret;
	}
	```

	- 数学分析，时间复杂度$O(n)$

	因为是在循环，因此只需要关注start和end，当$start <= end$的时候，结果为$[start,end]$，当$start > end$的时候，结果为$[1,end]+[start,n]$

	```cpp
	vector<int> mostVisited(int n, vector<int> &rounds)
	{
		vector<int> ret;
		int start = rounds[0], end = rounds[rounds.size() - 1];
		if (start <= end)
		{
			for (int i = start; i <= end; i++)
			{
				ret.push_back(i);
			}
		}
		else
		{
			for (int i = 1; i <= end; i++)
			{
				ret.push_back(i);
			}
			for (int i = start; i <= n; i++)
			{
				ret.push_back(i);
			}
		}
		return ret;
	}
	```

- [1566. Detect Pattern of Length M Repeated K or More Times](https://leetcode.com/problems/detect-pattern-of-length-m-repeated-k-or-more-times/)

    - 暴力模拟，时间复杂度$O(n^2)$

	```cpp
	bool containsPattern(vector<int> &arr, int m, int k)
	{
		// 检测长度为m重复次数为k的pattern
		int n = arr.size(), length = m * (k - 1), right_bound = n - k * m;
		for (auto left = 0; left <= right_bound; left++)
		{
			for (auto repeat = 1, i = left + m; i < left + k * m; i += m)
			{
				int j = 0;
				while (j < m && arr[left + j] == arr[i + j])
				{
					j++;
				}
				if (j == m)
				{
					repeat++;
				}
				if (repeat == k)
				{
					return true;
				}
			}
		}
		return false;
	}
	```

    - one pass扫描，通过计数器cnt来记录当前匹配模式的长度，时间复杂度$O(n)$

	```cpp
	bool containsPattern(vector<int> &arr, int m, int k)
	{
		// 检测长度为m重复次数为k的pattern
		int n = arr.size(), cnt = 0, expected_cnt = m * (k - 1), right_bound = n - m;
		for (auto i = 0; i < right_bound; i++)
		{
			arr[i] == arr[i + m] ? cnt++ : cnt = 0;
			if (cnt == expected_cnt)
			{
				return true;
			}
		}
		return false;
	}
	```

- [1579. Remove Max Number of Edges to Keep Graph Fully Traversable](https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/)

	**本题与冗余连接查找的[684. Redundant Connection](https://leetcode.com/problems/redundant-connection/)和[685. Redundant Connection II](https://leetcode.com/problems/redundant-connection-ii/)基本原理相同**

    - 使用并查集来确认Alice和Bob是否能够完全遍历全图（全连通图），leetcode评测机$\color{red}{TLE}$

	```cpp
	class Solution
	{
	private:
		int find(vector<int> &uf, int x)
		{
			return uf[x] == x ? x : (uf[x] = find(uf, uf[x]));
		}
		int removeRedundant_type3(vector<vector<vector<int>>> &graph, int n)
		{
			int ret = 0;
			vector<int> uf(n + 1);
			for (auto i = 0; i <= n; i++)
			{
				uf[i] = i;
			}
			// 使用并查集扫描冗余的边
			for (auto i = 1; i <= n; i++)
			{
				for (auto j = 1; j < i; j++)
				{
					if (graph[i][j][3] >= 1)
					{
						// 重复的边直接删除
						ret += graph[i][j][3] - 1;
						graph[i][j][3] = 1;
						// 检查是否为冗余的边
						int x = find(uf, i), y = find(uf, j);
						if (x == y)
						{
							ret++, graph[i][j][3] = 0;
						}
						else
						{
							uf[x] = y;
						}
					}
				}
			}
			return ret;
		}
		int checkConnected(vector<vector<vector<int>>> &graph, int n, int type)
		{
			int ret = 0;
			vector<int> uf(n + 1);
			for (auto i = 0; i <= n; i++)
			{
				uf[i] = i;
			}
			// 使用并查集扫描冗余的边
			for (auto i = 1; i <= n; i++)
			{
				for (auto j = 1; j < i; j++)
				{
					if (graph[i][j][3])
					{
						// 存在type3时直接删除type1/2
						ret += graph[i][j][type];
						graph[i][j][type] = 0;
						uf[find(uf, i)] = find(uf, j);
					}
				}
			} // 检查专属类型type的边
			for (auto i = 1; i <= n; i++)
			{
				for (auto j = 1; j < i; j++)
				{
					if (graph[i][j][type] > 0)
					{
						// 重复的边直接删除
						ret += graph[i][j][type] - 1;
						graph[i][j][type] = 1;
						// 检查是否为冗余的边
						int x = find(uf, i), y = find(uf, j);
						if (x == y)
						{
							ret++, graph[i][j][type] = 0;
						}
						else
						{
							uf[x] = y;
						}
					}
				}
			}
			// 检查全图连通性
			int group = find(uf, 1);
			for (auto i = 2; i <= n; i++)
			{
				if (group != find(uf, i))
				{
					return -1; // 无法实现全图遍历
				}
			}
			return ret;
		}

	public:
		int maxNumEdgesToRemove(int n, vector<vector<int>> &edges)
		{
			vector<vector<vector<int>>> graph(n + 1, vector<vector<int>>(n + 1, vector<int>(4, 0)));
			// graph[a][b][t]表示节点a到b之间有类型为t的边的条数
			for (auto &&e : edges)
			{
				// 无向图的对称性
				graph[e[1]][e[2]][e[0]]++;
				graph[e[2]][e[1]][e[0]]++;
			}
			int ret = 0;
			/***
			* 1. 首先删除冗余的3
			* 2. 其次删除Alice冗余的1
			* 3. 最后删除Bob冗余的2
			*/
			ret += removeRedundant_type3(graph, n);
			// 在保证Alice和Bob连通性的前提下删除冗余的边
			int Alice = checkConnected(graph, n, 1);
			int Bob = checkConnected(graph, n, 2);
			if (Alice == -1 || Bob == -1)
			{
				return -1;
			}
			else
			{
				ret += Alice + Bob;
			}
			return ret;
		}
	};
	```

    - 使用模板的并查集，对公共的type 3类型通道在前期处理后给Alice和Bob共用

	```cpp
	class Solution
	{
	private:
		struct UF
		{
			vector<int> uf;
			int count;
			UF(int n)
			{
				uf.resize(n);
				for (auto i = 0; i < n; i++)
				{
					uf[i] = i;
				}
				count = n;
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
					uf[x] = y, count--;
					return true;
				}
				return false;
			}
		};

		int checkConnected(vector<vector<int>> &edges, UF uf)
		{
			int ret = 0;
			for (auto &&e : edges)
			{
				ret += (uf.union_merge(e[0], e[1])) ? 0 : 1;
			}
			if (uf.count > 1)
			{
				return -1; // 不是全部连通（完全可遍历）的
			}
			return ret;
		}

	public:
		int maxNumEdgesToRemove(int n, vector<vector<int>> &edges)
		{
			int ret = 0;
			// 建立不同类型边的集合，无需表示整个图结构
			vector<vector<vector<int>>> graph(3);
			for (auto &&e : edges)
			{
				graph[e[0] - 1].push_back({e[1] - 1, e[2] - 1}); // convert base-1 to base-0
			}
			// 建立并查集来表示图的连通性（是否可以完全遍历）
			UF uf(n);
			// 删除冗余的type 3的可以共用的边
			for (auto &&e : graph[2])
			{
				ret += (uf.union_merge(e[0], e[1])) ? 0 : 1;
			}
			// 判断Alice和Bob是否可以完全遍历
			int Alice = checkConnected(graph[0], uf), Bob = checkConnected(graph[1], uf);
			if (Alice == -1 || Bob == -1)
			{
				return -1;
			}
			ret += Alice + Bob;
			return ret;
		}
	};
	```

- [1590. Make Sum Divisible by P](https://leetcode.com/problems/make-sum-divisible-by-p/)

	本题类似于[974. Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/)，用前缀和于哈希表记录同余的位置解决

	```cpp
	int minSubarray(vector<int> &nums, int p)
	{
		int n = nums.size();
		long long total_sum = 0;
		for (auto &&v : nums)
		{
			total_sum += v;
		}
		int remainder = total_sum % p;
		if (remainder == 0)
		{
			return 0; // 恰好可以被整除
		}
		unordered_map<int, int> ids;
		ids[0] = 0;
		int ret = n;
		total_sum = 0;
		for (int j = 1; j <= n; j++)
		{
			total_sum += nums[j - 1];
			int index_j = total_sum % p;
			int index_i = (index_j - remainder + p) % p;
			if (ids.find(index_i) != ids.end())
			{
				ret = min(j - ids[index_i], ret);
			}
			ids[index_j] = j;
		}
		return ret == n ? -1 : ret;
	}
	```

- [1594. Maximum Non Negative Product in a Matrix](https://leetcode.com/problems/maximum-non-negative-product-in-a-matrix/)

	动态规划，与[152. Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)相似，注意在乘法操作中超出int表示范围，要用long long数组来DP

	```cpp
		int maxProductPath(vector<vector<int>> &grid)
	{
		int ret = -1;
		if (grid.size() > 0 && grid[0].size() > 0)
		{
			int rows = grid.size(), cols = grid[0].size();
			int mod = 1e9 + 7;
			// 0 - min  v.s.  1 - max
			vector<vector<vector<long long>>> dp(rows, vector<vector<long long>>(cols, vector<long long>(2)));
			dp[0][0][0] = dp[0][0][1] = grid[0][0];
			for (int i = 1; i < rows; i++)
			{
				long long v = static_cast<int>(grid[i][0]);
				long long a = v * dp[i - 1][0][0], b = v * dp[i - 1][0][1];
				dp[i][0][0] = min(a, b), dp[i][0][1] = max(a, b);
			}
			for (int j = 1; j < cols; j++)
			{
				long long v = static_cast<long long>(grid[0][j]);
				long long a = v * dp[0][j - 1][0], b = v * dp[0][j - 1][1];
				dp[0][j][0] = min(a, b), dp[0][j][1] = max(a, b);
			}
			for (int i = 1; i < rows; i++)
			{
				for (int j = 1; j < cols; j++)
				{
					int v = static_cast<long long>(grid[i][j]);
					long long a = v * dp[i - 1][j][0], b = v * dp[i - 1][j][1], c = v * dp[i][j - 1][0], d = v * dp[i][j - 1][1];
					dp[i][j][0] = min(min(a, b), min(c, d));
					dp[i][j][1] = max(max(a, b), max(c, d));
				}
			}
			ret = dp[rows - 1][cols - 1][1] % mod;
			ret = ret < 0 ? -1 : ret;
		}
		return ret;
	}
	```

- [...](123)