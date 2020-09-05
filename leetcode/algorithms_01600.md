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

- [...](123)