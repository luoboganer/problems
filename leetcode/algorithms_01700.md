<!--
 * @Filename: 
 * @Author: shifaqiang
 * @Email: 14061115@buaa.edu.cn
 * @Github: https://github.com/luoboganer
 * @Date: 2020-09-05 11:29:59
 * @LastEditors: shifaqiang
 * @LastEditTime: 2021-03-08 20:23:01
 * @Software: Visual Studio Code
 * @Description: 1601-1700
-->

# 1601-1700

- [1610. Maximum Number of Visible Points](https://leetcode.com/problems/maximum-number-of-visible-points/)

    以人的位置为新的坐标原点，首先统计和人重合的点，然后将其余的点按照极角的大小排序，再以双指针的形式用滑动窗口扫描给定的角度范围内的最大点数即可，需要注意的点有：
    
    - 因为视角是可以循环的，因此算出所有极角并排序后，再将所有角度加360度后加入序列后面
    - 在计算角度的过程中用arctan函数不需要考虑值域范围最方便

    具体的cpp实现版本如下：

    - first version

    ```cpp
    class Solution
    {
    private:
        double pi = 3.141592653, err = 1e-5;
        double distance(const vector<int> &a, const vector<int> &b)
        {
            return sqrt(((1.0 * a[0] - b[0]) * (a[0] - b[0]) + (1.0 * a[1] - b[1]) * (a[1] - b[1])));
        }
        double compute_angle(const vector<int> &C, const vector<int> &B)
        {
            // B和C在同一条水平线上
            if (B[1] == C[1])
            {
                return C[0] > B[0] ? 180.0 : 0;
            }
            // c 原始的固定点，b是给定点
            double cos_theta = (B[0] - C[0]) / distance(B, C);
            double theta = acos(cos_theta) * 180 / pi;	// [0,2*pi) 并转化为角度值
            return (B[1] < C[1]) ? 360 - theta : theta; // 值域从[0,pi] 转化为[0,2*pi]
        }

    public:
        int visiblePoints(vector<vector<int>> &points, int angle, vector<int> &location)
        {
            int ret = 0, original = 0;
            vector<double> angles;
            for (auto &p : points)
            {
                if (p == location)
                {
                    original++;
                }
                else
                {
                    angles.push_back(compute_angle(location, p));
                }
            }
            sort(angles.begin(), angles.end());
            int left = 0, right = 0, n = angles.size();
            for (int i = 0; i < n; i++)
            {
                angles.emplace_back(angles[i] + 360); // 视角在旋转的过程中可以循环
            }
            n *= 2;
            while (left < n && right < n)
            {
                while (right < n && angles[right] - angles[left] <= angle + err)
                {
                    right++;
                }
                ret = max(right - left, ret);
                if (right < n && angles[right] - angles[left] > angle - err)
                {
                    left++;
                }
            }
            return ret + original;
        }
    };
    ```

    - second version

    ```cpp
	int visiblePoints(vector<vector<int>> &points, int angle, vector<int> &location)
	{
		int ret = 0, original = 0;
		double pi = 3.141592653, eps = 1e-8;
		vector<double> angles;
		double x = location[0], y = location[1];
		for (auto &p : points)
		{
			if (p == location)
			{
				original++;
			}
			else
			{
				angles.emplace_back(atan2(p[1] - y, p[0] - x) * 180 / pi);
			}
		}
		sort(angles.begin(), angles.end());
		int left = 0, right = 0, n = angles.size();
		for (int i = 0; i < n; i++)
		{
			angles.emplace_back(angles[i] + 360); // 视角在旋转的过程中可以循环
		}
		n *= 2;
		while (left < n && right < n)
		{
			while (right < n && angles[right] - angles[left] <= angle + eps)
			{
				right++;
			}
			ret = max(right - left, ret);
			if (right < n && angles[right] - angles[left] > angle - eps)
			{
				left++;
			}
		}
		return ret + original;
	}
    ```

- [1631. 最小体力消耗路径](https://leetcode-cn.com/problems/path-with-minimum-effort/)

    - 使用动态规划从左上角到右下角，BFS宽度优先所有更新从当前节点可以到达周边上下左右四个方向每个节点的最小体力花费

    ```cpp
	int minimumEffortPath(vector<vector<int>> &heights)
	{
		int ret = 0;
		if (heights.size() > 0 && heights[0].size() > 0)
		{
			vector<int> directions{1, 0, -1, 0, 1};
			int rows = heights.size(), cols = heights[0].size();
			// dp[i][j]表示到达点(i,j)的最小体力消耗
			vector<vector<int>> dp(rows, vector<int>(cols, numeric_limits<int>::max()));
			// 从点(0,0)开始
			dp[0][0] = 0;
			queue<pair<int, int>> bfs{{make_pair(0, 0)}};
			while (!bfs.empty())
			{
				// 当前可以到达的点
				int x = bfs.front().first, y = bfs.front().second;
				bfs.pop();
				/**
				 * 更新从当前点出发上下左右可以到达的四个点，对每个点：
				 * 如果到达该点的最小体力消耗值降低了，则其周边的点也有可能降低，将该点加入bfs队列继续搜索
				 * 如果到达该点的最小体力消耗值没变，则不再更新该点的dp值
				*/
				for (int i = 0; i < 4; i++)
				{
					int r = x + directions[i], c = y + directions[i + 1];
					if (r >= 0 && c >= 0 && r < rows && c < cols)
					{
						// v是从(x,y)到周边点(r,c)的体力消耗
						int v = max(dp[x][y], abs(heights[x][y] - heights[r][c]));
						if (v < dp[r][c])
						{
							// 从(x,y)到(r,c)的体力消耗小于从其他路径到(r,c)的体力消耗
							dp[r][c] = v;
							bfs.push(make_pair(r, c));
						}
					}
				}
			}
			ret = dp.back().back();
		}
		return ret;
	}
    ```

    - 将所有格子间的连接看成带权无向图，然后使用并查集从权重最小的边开始合并，直到点$(0,0)$和点$(rows-1,cols-1)$联通，时间复杂度$O(m*n*(log(m,n)+\alpha(m*n)))$，其中$m*n$为单元格总数

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
                uf.resize(n);
                count = n;
                for (int i = 0; i < n; i++)
                {
                    uf[i] = i;
                }
            }
            int find(int x)
            {
                return x == uf[x] ? x : (uf[x] = find(uf[x]));
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
            bool connected(int x, int y)
            {
                return find(x) == find(y);
            }
        };

    public:
        int minimumEffortPath(vector<vector<int>> &heights)
        {
            int ret = 0;
            if (heights.size() > 0 && heights[0].size() > 0)
            {
                /**
                * 1. 将rows*cols个格子看成不同节点，每个格子编号i*cols+j，
                * 相邻格子之间有一条边，权重为体力消耗值
                * 2. 然后将所有边按照权重从小到大排序
                * 3. 将排序后的边逐条加入并查集，直到点(0,0)和(rows-1,cols-1)之间连通
                */
                vector<vector<int>> edges;
                const int rows = heights.size(), cols = heights[0].size();
                // 将所有边及其权重存入数组
                for (int i = 1; i < rows; i++)
                {
                    edges.push_back(vector<int>{i * cols, (i - 1) * cols, abs(heights[i - 1][0] - heights[i][0])});
                }
                for (int j = 1; j < cols; j++)
                {
                    edges.push_back(vector<int>{j - 1, j, abs(heights[0][j - 1] - heights[0][j])});
                }
                for (int i = 1; i < rows; i++)
                {
                    for (int j = 1; j < cols; j++)
                    {
                        edges.push_back(vector<int>{i * cols + j, (i - 1) * cols + j, abs(heights[i][j] - heights[i - 1][j])});
                        edges.push_back(vector<int>{i * cols + j, i * cols + j - 1, abs(heights[i][j] - heights[i][j - 1])});
                    }
                }
                // 对所有边按照权重排序
                sort(edges.begin(), edges.end(), [](const auto &a, const auto &b) -> bool { return a[2] < b[2]; });
                // 用并查集表示所有节点间的连通关系
                UF uf = UF(rows * cols);
                const int start = 0, end = rows * cols - 1;
                for (auto &edge : edges)
                {
                    // 将权重最小的边依次加入并查集，直到起点start和终点end连通
                    uf.union_merge(edge[0], edge[1]);
                    ret = edge[2];
                    if (uf.connected(start, end))
                    {
                        break;
                    }
                }
            }
            return ret;
        }
    };
    ```

    - 将所有格子间的连接看成带权无向图，然后在二分查找的前提下，解决以$x_0$的花费是否可以解决连通点$(0,0)$和点$(rows-1,cols-1)$的判定性问题，时间复杂度$O(m*n*log(C))$，其中$m*n$是格子总数，$C$是格子的最大高度

    ```cpp
	int minimumEffortPath(vector<vector<int>> &heights)
	{
		int ret = 0;
		if (heights.size() > 0 && heights[0].size() > 0)
		{
			/**
			 * 给定格子的最大高度为[1,1e6]，因此可以在这个范围内二分查找可以从(0,0)到达(rows-1,cols-1)的最小体力花费
			*/
			const int rows = heights.size(), cols = heights[0].size();
			int left = 0, right = 1e6;
			while (left <= right)
			{
				int mid = left + ((right - left) >> 1);
				// 判定在消耗体力不超过mid的情况下是否可以满足从(0,0)到达(rows-1,cols-1)
				vector<int> directions{1, 0, -1, 0, 1};
				vector<bool> visited(rows * cols);
				visited[0] = true; // 从(0,0)点开始
				queue<pair<int, int>> bfs{{make_pair(0, 0)}};
				while (!bfs.empty())
				{
					auto [x, y] = bfs.front();
					bfs.pop();
					for (int i = 0; i < 4; i++)
					{
						int r = x + directions[i], c = y + directions[i + 1];
						if (r >= 0 && c >= 0 && r < rows && c < cols && !visited[r * cols + c] && abs(heights[x][y] - heights[r][c]) <= mid)
						{
							bfs.emplace(r, c);
							visited[r * cols + c] = true;
						}
					}
				}
				visited[rows * cols - 1] ? right = mid - 1, ret = mid : left = mid + 1;
			}
		}
		return ret;
	}
    ```

    - some test cases

    ```cpp
    [[1,2,2],[3,8,2],[5,3,5]]
    [[1,2,3],[3,8,4],[5,3,5]]
    [[1,2,1,1,1],[1,2,1,2,1],[1,2,1,2,1],[1,2,1,2,1],[1,1,1,2,1]]
    ```

- [1642. 可以到达的最远建筑](https://leetcode-cn.com/problems/furthest-building-you-can-reach/)

    顺序遍历两个建筑之间的差距，并使用优先队列来存储需要梯子的差值（堆顶为最小值），当梯子不够用的时候在差值最小的地方使用砖块

    ```cpp
	int furthestBuilding(vector<int> &heights, int bricks, int ladders)
	{
		priority_queue<int, vector<int>, greater<int>> qe;
		const int n = heights.size();
		for (int i = 1; i < n; i++)
		{
			int diff = heights[i] - heights[i - 1];
			if (diff <= 0)
			{
				// 可以直接从高到底
				continue;
			}
			qe.push(diff);
			if (qe.size() <= ladders)
			{
				// 可以通过梯子到达
				continue;
			}
			if (qe.top() <= bricks)
			{
				bricks -= qe.top();
				qe.pop();
				// 可以通过垒砖块到达
				continue;
			}
			return i - 1; // 无法到达i
		}
		return n - 1; //可以到达最后一栋建筑
	}
    ```

- [1656. 设计有序流](https://leetcode-cn.com/problems/design-an-ordered-stream/)

    模拟操作，注意元素为字符串的向量初始化后向量元素为空字符串，时间复杂度为$O(n)$

    ```cpp
    class OrderedStream
    {
        vector<string> strs;
        int ptr;
        int n;

    public:
        OrderedStream(int n)
        {
            this->n = n;
            strs.resize(n, "");
            ptr = 1;
        }

        vector<string> insert(int id, string value)
        {
            vector<string> ret;
            strs[id - 1] = value;
            if (id == ptr)
            {
                while (ptr <= n && !strs[ptr - 1].empty())
                {
                    ret.emplace_back(strs[ptr - 1]);
                    ptr++;
                }
            }
            return ret;
        }
    };
    ```

- [1663. 具有给定数值的最小字符串](https://leetcode-cn.com/problems/smallest-string-with-a-given-numeric-value/)

    贪心思想，相同长度的字符串左端字符序越小字符串的字典序越小，因此将给定的数值和尽可能在字符串右端消耗掉，时间复杂度$O(n)$

    ```cpp
	string getSmallestString(int n, int k)
	{
		string ret;
		int base = 26, length = n, cost = k;
		string letters = "0abcdefghijklmnopqrstuvwxyz";
		while (length)
		{
			int idx = min(base, cost - length + 1);
			cost -= idx, length--;
			ret.push_back(letters[idx]);
		}
		reverse(ret.begin(), ret.end());
		return ret;
	}
    ```

- [1664. 生成平衡数组的方案数](https://leetcode-cn.com/problems/ways-to-make-a-fair-array/)

    对于每一个下标$i$，分别计算i的左右两侧奇数下标的数字和与偶数下标的数字和，时间复杂度$O(n)$

    ```cpp
	int waysToMakeFair(vector<int> &nums)
	{
		const int n = nums.size();
		vector<int> odd(n, 0), even(n, 0);
		int odd_base = 0, even_base = 0;
		for (int i = 0; i < n; i++)
		{
			(i & 0x1) ? odd_base += nums[i] : even_base += nums[i];
			odd[i] = odd_base;
			even[i] = even_base;
		}
		int ret = 0;
		for (int i = 0; i < n; i++)
		{
			int left_odd = odd[i] - (i & 0x1 ? nums[i] : 0);
			int left_even = even[i] - (i & 0x1 ? 0 : nums[i]);
			int right_odd = odd_base - odd[i];
			int right_even = even_base - even[i];
			if (left_odd + right_even == left_even + right_odd)
			{
				ret++;
			}
		}
		return ret;
	}
    ```

- [...](123)
