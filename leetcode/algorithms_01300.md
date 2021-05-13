# 1201-1300

- [1201. Ugly Number III](https://leetcode.com/problems/ugly-number-iii/)

	- 从1开始递增的逐个列出所有$v(v\%a==0||v\%b==0||v\%c==0)$，时间复杂度$O(n)$，leetcode评测机$\color{red}{TLE}$

	```cpp
	int nthUglyNumber(int n, int a, int b, int c)
	{
		int ret = 1, a_coef = 1, b_coef = 1, c_coef = 1;
		for (auto i = 0; i < n; ++i)
		{
			int v = min(min(a * a_coef, b * b_coef), c * c_coef);
			if (v % a == 0)
			{
				a_coef++;
			}
			if (v % b == 0)
			{
				b_coef++;
			}
			if (v % c == 0)
			{
				c_coef++;
			}
			ret = v;
		}
		return ret;
	}
	```

	- binary search，时间复杂度$O(log(n))$

	```cpp
	class Solution
	{
	private:
		long long gcd(long long a, long long b)
		{
			if (a < b)
			{
				swap(a, b);
			}
			while (a % b != 0)
			{
				int r = a % b;
				a = b;
				b = r;
			}
			return b;
		}

		long long lcd(long long a, long long b)
		{
			return a * b / gcd(a, b);
		}

	public:
		int nthUglyNumber(int n, int a, int b, int c)
		{
			long long lcd_ab = lcd(a, b), lcd_bc = lcd(b, c), lcd_ca = lcd(c, a);
			long long lcd_abc = lcd(lcd_ab, c);
			long long left = min(min(a, b), c), right = 2 * 1e9;
			while (left < right)
			{
				long long v = left + (right - left) / 2;
				long long count = v / a + v / b + v / c - v / lcd_ab - v / lcd_bc - v / lcd_ca + v / lcd_abc;
				if (count < n)
				{
					left = v + 1;
				}
				else
				{
					right = v;
				}
			}
			return static_cast<int>(left);
		}
	};
	```

- [1202. Smallest String With Swaps](https://leetcode.com/problems/smallest-string-with-swaps/)

	并查集的建立与使用，时间复杂度$O(nlog(n)))$

	```cpp
	class Solution
	{
	private:
		int find(vector<int> &father, int x)
		{
			// 包含路径压缩的并查集查找
			return father[x] == x ? x : (father[x] = find(father, father[x]));
		}
		void union_insert(vector<int> &father, int x, int y)
		{
			father[find(father, x)] = father[find(father, y)];
		}

	public:
		string smallestStringWithSwaps(string s, vector<vector<int>> &pairs)
		{
			/*
			* 1. 在可交换的pair基础上使用并查集合并所有的连通index
			* 2. 单个连通的index内所有字符直接排序，按字典序排序后放回该连通域内的所有下标位置
			* 3. 返回构造的结果字符串
			*/
			int n = s.length();
			vector<int> father(n);
			// initialization of union find
			for (auto i = 0; i < n; i++)
			{
				father[i] = i;
			}
			// update of the union find
			for (auto &&p : pairs)
			{
				union_insert(father, p[0], p[1]);
			}
			// 单个连通域内排序
			vector<char> chars(n);
			unordered_map<int, vector<int>> map;
			for (auto i = 0; i < n; i++)
			{
				map[find(father, i)].push_back(i); // 标记每个字符的直接代表
			}
			for (auto &&item : map)
			{
				string cur;
				vector<int> indexs = item.second;
				for (auto &&index : indexs)
				{
					cur.push_back(s[index]);
				}
				sort(cur.begin(), cur.end());
				sort(indexs.begin(), indexs.end());
				for (auto i = 0; i < cur.length(); i++)
				{
					chars[indexs[i]] = cur[i];
				}
			}
			// 构造结果串
			string ret;
			for (auto &&ch : chars)
			{
				ret.push_back(ch);
			}
			return ret;
		}
	};
	```

- [1203. 项目管理](https://leetcode-cn.com/problems/sort-items-by-groups-respecting-dependencies/)

    拓扑排序问题，首先对所有小组排序，然后每个小组组内进行排序，时间复杂度$O(n+m)$

    ```cpp
    class Solution
    {
    private:
        vector<int> topSort(vector<int> &indegrees, vector<vector<int>> &graph, vector<int> &items)
        {
            vector<int> bfs;
            for (auto &item : items)
            {
                if (indegrees[item] == 0)
                {
                    // 入度为0的节点直接安排
                    bfs.emplace_back(item);
                }
            }
            for (int i = 0; i < bfs.size(); i++)
            {
                for (auto node : graph[bfs[i]])
                {
                    if (--indegrees[node] == 0)
                    {
                        bfs.emplace_back(node);
                    }
                }
            }
            return bfs.size() == items.size() ? bfs : vector<int>{};
        }

    public:
        vector<int> sortItems(int n, int m, vector<int> &group, vector<vector<int>> &beforeItems)
        {
            // 每组可能的项目(无人接手的从m开始递增编号，因为已有的组编号是从[0,m-1])
            vector<vector<int>> groupToItems(n + m);
            // 组间和组内依赖图，全部可能n+m个组编号，n个项目
            vector<vector<int>> groupGraph(n + m), itemGraph(n);
            // 组间和组内入度数组
            vector<int> groupInDegrees(n + m, 0), itemInDegrees(n, 0);
            vector<int> id(n + m);
            for (int i = 0; i < n + m; i++)
            {
                id[i] = i;
            }
            // 无人接手的项目分配ID
            for (int leftID = m, i = 0; i < n; i++)
            {
                if (group[i] == -1)
                {
                    group[i] = leftID++;
                }
                groupToItems[group[i]].emplace_back(i);
            }
            // 前后依赖关系建立邻接链表的图表示
            for (int i = 0; i < n; i++)
            {
                int curGroupID = group[i];
                for (auto &item : beforeItems[i])
                {
                    int beforeGroupId = group[item];
                    if (beforeGroupId == curGroupID)
                    {
                        // 项目item和i在同一组且item为i的前置条件
                        itemInDegrees[i] += 1;
                        itemGraph[item].emplace_back(i);
                    }
                    else
                    {
                        // 项目item和i不在同一组，则其所在的beforeGroupID为curGroupID前置条件
                        groupInDegrees[curGroupID]++;
                        groupGraph[beforeGroupId].emplace_back(curGroupID);
                    }
                }
            }
            // 组间拓扑排序
            vector<int> groupTopSort = topSort(groupInDegrees, groupGraph, id);
            if (groupTopSort.size() == 0)
            {
                return vector<int>{}; // 组间排序失败
            }
            // 组内拓扑排序
            vector<int> ans;
            for (auto &curGroupID : groupTopSort)
            {
                int size = groupToItems[curGroupID].size();
                if (size > 0)
                {
                    vector<int> ret = topSort(itemInDegrees, itemGraph, groupToItems[curGroupID]);
                    if (ret.size() == 0)
                    {
                        return vector<int>{}; // 这一组组内无法排序
                    }
                    for (auto &item : ret)
                    {
                        ans.emplace_back(item);
                    }
                }
            }
            return ans;
        }
    };
    ```

- [1208. 尽可能使字符串相等](https://leetcode-cn.com/problems/get-equal-substrings-within-budget/)

    使用双指针实现滑动窗口，时间复杂度$O(n)$

    ```cpp
	int equalSubstring(string s, string t, int maxCost)
	{
		// 给定s与t长度相同
		int ret = 0, left = 0, right = 0, cost = 0, n = s.length();
		// 预处理s和t相同位置的字符之间的替换开销
		vector<int> costs(n);
		for (int i = 0; i < n; i++)
		{
			costs[i] = abs(s[i] - t[i]);
		}
		while (right < n)
		{
			cost += costs[right++]; // 右指针右移
			if (cost <= maxCost)
			{
				// 当前s[left,rigth)在最大maxCost花费下可以完全相同
				ret = max(ret, right - left);
			}
			while (left <= right && cost > maxCost)
			{
				// 左指针右移
				cost -= costs[left++];
			}
		}
		return ret;
	}
    ```

- [1209. Remove All Adjacent Duplicates in String II](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/)

    删除一个字符串s中重复次数为k的串，可以使用栈的方式来存储每个字符及其出现的次数，每个字符至多被访问两次，时间复杂度$O(n)$

    ```cpp
    string removeDuplicates(string s, int k)
    {
        string st;
        stack<int> count;
        for (auto &&ch : s)
        {
            if (st.length() == 0)
            {
                st.push_back(ch);
                count.push(1);
            }
            else
            {
                if (st.back() == ch)
                {
                    count.push(count.top() + 1);
                }else{
                    count.push(1);
                }
                st.push_back(ch);
                if (count.top() == k)
                {
                    for (int i = 0; i < k; i++)
                    {
                        st.pop_back(), count.pop();
                    }
                }
            }
        }
        return st;
    }
    ```

- [1217](https://leetcode.com/problems/play-with-chips/)

    将指定位置的筹码chip集中移动到任何位置，移动一格的花费为1、移动两格的花费为0，因此偶数位置多时全部集中到0位置，奇数位置多时全部集中到1位置，即统计位置的奇偶数数量，返回最小值即可

    ```cpp
    int minCostToMoveChips(vector<int>& chips) {
        int odd=0,even=0;
        for(auto &&v:chips){
            if(v&0x1){
                odd++;
            }else{
                even++;
            }
        }
        return min(odd,even);
    }
    ```

- [1218](https://leetcode.com/problems/longest-arithmetic-subsequence-of-given-difference/)

    求给定数组中，公差为给定difference的等差子序列的最大长度，显然为动态规划类型

    - 尝试用$dp[i]$表示包含数$arr[i]$的最长等差子序列长度，则$dp[i]=dp[j]+1$，其中$arr[j]$为$arr[i]$左侧第一个值为$arr[i]-difference$的数，需要遍历$i$左侧的数组查找，因此时间复杂度为$O(n^2)$，可以得出正确结果，但是$\color{red}{[TLE](Time Limit Exceeded)}$

    ```cpp
    int longestSubsequence(vector<int>& arr, int difference) {
        const int count=arr.size();
        vector<int> dp(count,1);
        int ret=1;
        for(int i=0;i<count;i++){
            int target=arr[i]-difference;
            for(int j=i-1;j>=0;j--){
                if(arr[j]==target){
                    dp[i]=dp[j]+1;
                    break;
                }
            }
            ret=max(ret,dp[i]);
        }
        return ret;
    }
    ```

    - 注意到$arr[i]$的限定范围为$[-1e4,1e4]$，因此可以直接用$dp[v]$表示last element为$v$的最长等差子序列长度，这样$dp[v]=dp[v-difference]+1$，其中$dp[v-difference]$可以直接索引到，因此时间复杂度为$O(n)$

    ```cpp
    int longestSubsequence(vector<int>& arr, int difference) {
        const int count=arr.size(),length=1e4*2+1;
        vector<int> dp(length,0);
        int ret=0;
        for(int i=0;i<count;i++){
            int index=arr[i]-difference+1e4;
            dp[arr[i]+1e4]=(index>=0 && index<length)?dp[index]+1:1;
            ret=max(ret,dp[arr[i]+1e4]);
        }
        return ret;
    }
    ```

- [1219](https://leetcode.com/problems/path-with-maximum-gold/)

    在给定矩阵中求出可以一笔连起来的非零数字和最大即可，根据给定的数据规模来看是典型的DFS应用，这里特别注意每次标记一个$visited[i][j]$之后，在完成与该点相关的计算之后，要释放该标记，以便回溯

    ```cpp
    int dfs_helper(vector<vector<int>> &grid, int i, int j, vector<vector<bool>> &visited)
    {
        int ret = 0;
        if (!(i < 0 || j < 0 || i >= grid.size() || j >= grid[0].size() || grid[i][j] == 0 || visited[i][j] == 1))
        {
            visited[i][j] = true;
            ret = max(ret, dfs_helper(grid, i - 1, j, visited));
            ret = max(ret, dfs_helper(grid, i + 1, j, visited));
            ret = max(ret, dfs_helper(grid, i, j - 1, visited));
            ret = max(ret, dfs_helper(grid, i, j + 1, visited));
            ret += grid[i][j];
            visited[i][j] = false;  // for backtracing
        }
        return ret;
    }
    int getMaximumGold(vector<vector<int>> &grid)
    {
        int ret = 0;
        if (grid.size() > 0 && grid[0].size() > 0)
        {
            const int m = grid.size(), n = grid[0].size();
            vector<vector<bool>> visited(m, vector<bool>(n, false));
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    ret = max(ret, dfs_helper(grid, i, j, visited));
                }
            }
        }
        return ret;
    }
    ```

- [1222. 可以攻击国王的皇后](https://leetcode-cn.com/problems/queens-that-can-attack-the-king/)

    在“米”字型的八个方向上找到第一个可以攻击king的queen位置即可，固定的$8*8$棋盘，因此时间复杂度$O(1)$

    ```cpp
	vector<vector<int>> queensAttacktheKing(vector<vector<int>> &queens, vector<int> &king)
	{
		unordered_set<int> visited;
		const int n = 8, base = 10;
		vector<vector<int>> grid(n, vector<int>(n, 0));
		for (auto &p : queens)
		{
			grid[p[0]][p[1]] = 1;
		}
		// up
		for (int i = king[0] - 1, j = king[1]; i >= 0; i--)
		{
			if (grid[i][j] == 1)
			{
				visited.insert(i * base + j);
				break;
			}
		}
		// bottom
		for (int i = king[0] + 1, j = king[1]; i < n; i++)
		{
			if (grid[i][j] == 1)
			{
				visited.insert(i * base + j);
				break;
			}
		}
		// left
		for (int i = king[0], j = king[1] - 1; j >= 0; j--)
		{
			if (grid[i][j] == 1)
			{
				visited.insert(i * base + j);
				break;
			}
		}
		// right
		for (int i = king[0], j = king[1] + 1; j < n; j++)
		{
			if (grid[i][j] == 1)
			{
				visited.insert(i * base + j);
				break;
			}
		}
		// left up
		for (int i = king[0] - 1, j = king[1] - 1; i >= 0 && j >= 0; i--, j--)
		{
			if (grid[i][j] == 1)
			{
				visited.insert(i * base + j);
				break;
			}
		}
		// left bottom
		for (int i = king[0] + 1, j = king[1] - 1; i < n && j >= 0; i++, j--)
		{
			if (grid[i][j] == 1)
			{
				visited.insert(i * base + j);
				break;
			}
		}
		// right up
		for (int i = king[0] - 1, j = king[1] + 1; i >= 0 && j < n; i--, j++)
		{
			if (grid[i][j] == 1)
			{
				visited.insert(i * base + j);
				break;
			}
		}
		// right bottom
		for (int i = king[0] + 1, j = king[1] + 1; i < n && j < n; i++, j++)
		{
			if (grid[i][j] == 1)
			{
				visited.insert(i * base + j);
				break;
			}
		}
		// 检查可以攻击的queue
		vector<vector<int>> ret;
		for (auto queue : queens)
		{
			if (visited.find(queue[0] * base + queue[1]) != visited.end())
			{
				ret.emplace_back(queue);
			}
		}
		return ret;
	}
    ```

- [1227](https://leetcode.com/problems/airplane-seat-assignment-probability/)

    本题中，前n-1个人占据前n-1个座位的情况有$A_{n-1}^{n-1}$种，此时第n个人一定可以匹配到自己的位置，当前n-1个人在前n-1个座位中只占据了n-2个并占用第n个座位的可能情况有$A_{n-1}^{n-2}$种，此时第n个人只能坐在前n-1个座位中空出来的那个，因此第n个人坐在第n个座位的概率为
    $$Prob=\left\{\begin{matrix} \frac{A_{n-1}^{n-1}}{A_{n-1}^{n-1}+A_{n-1}^{n-2}} = 0.5, \quad n \ge 2\\  1.0 \quad n=1 \end{matrix}\right.$$

    ```cpp
    double nthPersonGetsNthSeat(int n) {
        double ans=1.0;
        if(n>1){
            ans=0.5;
        }
        return ans;
    }
    ```

- [1230](https://leetcode.com/problems/toss-strange-coins/)

    有n个硬币，第i个随机扔起正面朝上的概率为$prob_i$，求将n个硬币全部随机扔起之后有target个正面朝上的概率。本题属于典型的动态规划DP类型，递推公式为$$dp_{ij}=dp_{i-1,j-1}*prob_i+dp_{i-1,j}*(1-prob_i)$$，其中$dp_{ij}$表示随机扔起i个硬币后有j个朝上的概率，即随机扔起i个后有j个朝上有两种情况，一是随机扔起$i-1$个有$j-1$个朝上并且第j个也朝上，二是随机扔起$i-1$个有$j$个朝上并且第j个朝下。

    ```cpp
    double probabilityOfHeads(vector<double>& prob, int target) {
        int const count=prob.size();
        vector<double> dp(target+1,0);
        dp[0]=1.0;
        for (int i = 0; i < count; i++)
        {
            for (int j = min(i+1,target); j > 0; j--)
            {
                dp[j]=dp[j-1]*prob[i]+dp[j]*(1-prob[i]);
            }
            dp[0]*=1-prob[i];
        }
        return dp.back();
    }
    ```

- [1237. 找出给定方程的正整数解](https://leetcode-cn.com/problems/find-positive-integer-solution-for-a-given-equation/)

    遍历给定的x区间[1,1000]，二分搜索可能的y值，时间复杂度$O(nlog(n))$，其中$n=1000$

    ```cpp
	vector<vector<int>> findSolution(CustomFunction &customfunction, int z)
	{
		vector<vector<int>> ret;
		// binary search in range [1,1000]
		for (int x = 1; x <= 1000; x++)
		{
			int left = 1, right = 1000;
			while (left <= right)
			{
				int mid = left + ((right - left) >> 1);
				int v = customfunction.f(x, mid);
				if (v == z)
				{
					ret.emplace_back(vector<int>{x, mid});
                    break;
				}
				else if (v > z)
				{
					right = mid - 1;
				}
				else
				{
					left = mid + 1;
				}
			}
		}
		return ret;
	}
    ```

- [1245](https://leetcode.com/problems/tree-diameter/)

    给定一颗树，即N叉树，求树的直径，即树中任意两各节点之间的最远距离，通过两遍BFS来求解，首先BFS从任意节点start出发求得最远节点end，然后第二遍BFS从节点end出发求得最远节点last，end到last之间的距离即为树的直径。

    ```cpp
    int depth_BFS(int start, vector<vector<int>> &from_i_to_node, int &end)
    {
        int depth = 0;
        vector<int> cur_level{start};
        vector<bool> visited(from_i_to_node.size(), false);
        end = start;
        while (true)
        {
            vector<int> next_level;
            for (auto &&node : cur_level)
            {
                if (!visited[node])
                {
                    next_level.insert(next_level.end(), from_i_to_node[node].begin(), from_i_to_node[node].end());
                    if (!from_i_to_node[node].empty())
                    {
                        end = from_i_to_node[node].back();
                    }
                    visited[node] = true;
                }
            }
            if (next_level.empty())
            {
                break;
            }
            else
            {
                depth++;
                cur_level = next_level;
            }
        }
        return depth;
    }
    int treeDiameter(vector<vector<int>> &edges)
    {
        int ans = 0;
        const int count = edges.size();
        if (count > 0)
        {
            // 第一次bfs，从任意节点start出发找到最远节点end
            int start = 0, end = 0;
            // sort(edges.begin(), edges.end(), [](vector<int> &a, vector<int> &b) -> bool { return a[0] < b[0]; });
            vector<vector<int>> from_i_to_node(count + 1, vector<int>{});
            for (int i = 0; i < count; i++)
            {
                from_i_to_node[edges[i][0]].push_back(edges[i][1]);
                from_i_to_node[edges[i][1]].push_back(edges[i][0]);
            }
            int depth = depth_BFS(start, from_i_to_node, end);
            // 第二次bfs，从end出发找到最远节点last，end到last之间的距离即为树的直径 diameter
            ans = depth_BFS(end, from_i_to_node, end);
        }
        return ans;
    }
    ```

- [1249](https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/)

    在一个包含英文字母和左右小括号的字符串中，移除最少数量的左右括号，使得字符串中的左右括号相匹配，使用栈来实现即可。

    ```cpp
    string minRemoveToMakeValid(string s)
    {
        stack<int> st; // record index of left parenthese
        vector<int> removed_index;
        for (int i = 0; i < s.length(); i++)
        {
            if(s[i]=='('){
                st.push(i);
            }else if(s[i]==')'){
                if(st.empty()){
                    removed_index.push_back(i);
                }else{
                    st.pop();
                }
            }
        }
        while(!st.empty()){
            removed_index.push_back(st.top());
            st.pop();
        }
        sort(removed_index.begin(), removed_index.end());
        string ans;
        for (int i = 0, j = 0; i < s.length(); i++)
        {
            if(j<removed_index.size() && i==removed_index[j]){
                j++;
            }else{
                ans.push_back(s[i]);
            }
        }
        return ans;
    }
    ```

- [1250](https://leetcode.com/problems/check-if-it-is-a-good-array/)

    数论中存在一个裴蜀定理[wikipad](https://www.wikidata.org/wiki/Q513028)，简单讲就是两个整数$x,y$互质的充要条件为存在整数$a,b$使得$a*x+b*y=1$，因此本题给定一维数组$x$，如果它们的最大公约数（GCD）为1，则它们可以采用合适的权重$a$实现$\sum_{i=1}^{n}a_i*x_i=1$，即只需验证给定数组中是否存在一组互质的数即可。

    ```cpp
    int gcd(int a,int b){
        if(a<b){
            swap(a, b);
        }
        while(b){
            int r = a % b;
            a = b;
            b = r;
        }
        return a;
    }
    bool isGoodArray(vector<int> &nums)
    {
        bool ans = false;
        if(nums.size()>0){
            int v = nums.front();
            for (int i = 1; i < nums.size(); i++)
            {
                v = gcd(v, nums[i]);
            }
            if(v==1){
                ans = true;
            }
        }
        return ans;
    }
    ```

- [1254](https://leetcode.com/problems/number-of-closed-islands/)

    求被水域完全包围的陆地（岛屿）的数量，DFS遍历所有岛屿，与常规统计岛屿数量[200](https://leetcode.com/problems/number-of-islands/)多的一点是，放弃所有与边界相连的岛屿

    ```cpp
    bool is_closed(vector<vector<int>> &grid, int i, int j)
    {
        bool ans = true;
        if (i < 0 || j < 0 || i >= grid.size() || j >= grid[0].size())
        {
            ans = false;
            // 到达边界
        }
        else
        {
            if (grid[i][j] == 1)
            {
                // 水域
                ans = true;
            }
            else
            {
                // 陆地
                grid[i][j] = 1; // flag for visited
                ans &= is_closed(grid, i - 1, j);
                ans &= is_closed(grid, i + 1, j);
                ans &= is_closed(grid, i, j - 1);
                ans &= is_closed(grid, i, j + 1);
            }
        }
        return ans;
    }
    int closedIsland(vector<vector<int>> &grid)
    {
        int ans = 0;
        if (grid.size() > 0 && grid[0].size() > 0)
        {
            const int rows = grid.size(), cols = grid[0].size();
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (grid[i][j] == 0 && is_closed(grid, i, j))
                    {
                        ans++;
                    }
                }
            }
        }
        return ans;
    }
    ```

- [1257](https://leetcode.com/problems/smallest-common-region/)

    按照层次大小给定一些区域，大区域包含小区域，求包含两个给定区域的最小区域，按照区域的包含关系建立层次化的树结构，父节点为大区域，子节点为其包含的所有子区域，然后求两个给定区域的最低公共父节点即可

    ```cpp
    class Solution {
        public:
            struct TreeNodeString
            {
                /* data */
                string region;
                TreeNodeString *parent;
                int level;
                vector<TreeNodeString *> children;
                TreeNodeString(string x, int v, TreeNodeString *p) : region(x), level(v), parent(p) {}
            };
            string findSmallestRegion(vector<vector<string>> &regions, string region1, string region2)
            {
                unordered_map<string, TreeNodeString *> queries;
                TreeNodeString *root = new TreeNodeString("", 0, nullptr);
                for (auto &&region : regions)
                {
                    string common_region = region[0];
                    TreeNodeString *cur;
                    // 确定父节点
                    if (queries.find(common_region) != queries.end())
                    {
                        cur = queries[common_region];
                    }
                    else
                    {
                        cur = new TreeNodeString(common_region, 1, root);
                        root->children.push_back(cur);
                        queries[common_region] = cur;
                    }
                    for (int i = 1; i < region.size(); i++)
                    {
                        TreeNodeString *node = new TreeNodeString(region[i], cur->level + 1, cur);
                        cur->children.push_back(node);
                        queries[region[i]] = node;
                    }
                }
                // 寻找给定的两个区域的最低公共父节点
                TreeNodeString *r1 = queries[region1], *r2 = queries[region2];
                while (r1->level > r2->level)
                {
                    r1 = r1->parent;
                }
                while (r1->level < r2->level)
                {
                    r2 = r2->parent;
                }
                while (r1 != r2)
                {
                    r1 = r1->parent;
                    r2 = r2->parent;
                }
                return r1->region;
            }
    };
    ```

    一些测试案例

    ```cpp
    [["Earth","North America","South America"],["North America","United States","Canada"],["United States","New York","Boston"],["Canada","Ontario","Quebec"],["South America","Brazil"]]
    "Quebec"
    "New York"
    [["Earth", "North America", "South America"],["North America", "United States", "Canada"],["United States", "New York", "Boston"],["Canada", "Ontario", "Quebec"],["South America", "Brazil"]]
    "Canada"
    "South America"
    [["Earth", "North America", "South America"],["North America", "United States", "Canada"],["United States", "New York", "Boston"],["Canada", "Ontario", "Quebec"],["South America", "Brazil"]]
    "Canada"
    "Quebec"
    ```

- [1260. 二维网格迁移](https://leetcode-cn.com/problems/shift-2d-grid/)

    将二维数组拉平flatten之后即转化为数组向右平移k位问题，可以通过三次翻转在$O(n)$时间内解决

    ```cpp
	vector<vector<int>> shiftGrid(vector<vector<int>> &grid, int k)
	{
		if (grid.size() > 0 && grid[0].size() > 0 && k > 0)
		{
			int rows = grid.size(), cols = grid[0].size();
			int n = rows * cols;
			k %= n; // 防止向右平移位置数超过数组本身的大小
			if (k != 0)
			{
				// first conversion, [0,n-k-1]
				for (int left = 0, right = n - k - 1; left < right; left++, right--)
				{
					swap(grid[left / cols][left % cols], grid[right / cols][right % cols]);
				}
				// second conversion, [n-k,n-1]
				for (int left = n - k, right = n - 1; left < right; left++, right--)
				{
					swap(grid[left / cols][left % cols], grid[right / cols][right % cols]);
				}
				// third conversion, [0,n-1]
				for (int left = 0, right = n - 1; left < right; left++, right--)
				{
					swap(grid[left / cols][left % cols], grid[right / cols][right % cols]);
				}
			}
		}
		return grid;
	}
    ```

- [1267. 统计参与通信的服务器](https://leetcode-cn.com/problems/count-servers-that-communicate/)

    统计每一行和每一列的节点数，然后判断每个节点，如果所在行和所在列还有其他节点，则可以和其他节点通信，时间复杂度$O(m*n)$

    ```cpp
	int countServers(vector<vector<int>> &grid)
	{
		int ret = 0;
		if (grid.size() > 0 && grid[0].size() > 0)
		{
			const int rows = grid.size(), cols = grid[0].size();
			vector<int> count_row(rows, 0), count_col(cols, 0);
			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < cols; j++)
				{
					if (grid[i][j])
					{
						count_row[i]++, count_col[j]++;
					}
				}
			}
			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < cols; j++)
				{
					if (grid[i][j] && (count_row[i] > 1 || count_col[j] > 1))
					{
						ret++;
					}
				}
			}
		}
		return ret;
	}
    ```

- [1268. Search Suggestions System](https://leetcode.com/problems/search-suggestions-system/)

    做出类似于现代编辑器中代码提示、自动补全的功能，即在给定的所有字符串中，搜索符合当前已经键入的字符的串，按字典序排序只要前3个

    - [Trie]字典树方法，这里不知道为什么我的字典树写法会超时(TLE)，讨论区有些人的字典树方法是可以AC的，很纠结

    ```cpp
    struct TrieNode
    {
        // 字典树节点的定义
        bool isWord;
        TrieNode *next[26];
        TrieNode()
        {
            isWord = false;
            for (int i = 0; i < 26; i++)
            {
                next[i] = nullptr;
            }
        }
    };
    class Solution
    {
    private:
        void insertWordToDictory(TrieNode *root, string s)
        {
            for (int i = 0; i < s.length(); i++)
            {
                int index = (int)(s[i] - 'a');
                if (root->next[index] == nullptr)
                {
                    root->next[index] = new TrieNode();
                }
                root = root->next[index];
            }
            root->isWord = true;
        }
        void dfs_query(vector<string> &ans, string s, TrieNode *root)
        {
            if (root && ans.size() < 3)
            {
                if (root->isWord)
                {
                    ans.push_back(s);
                }
                for (int i = 0; i < 26; i++)
                {
                    dfs_query(ans, s + (char)('a' + i), root->next[i]);
                }
            }
        }
        vector<string> query(TrieNode *root, string s)
        {
            vector<string> ans;
            if (root && s.length() > 0)
            {
                // 首先保证目前已经键入的字母完全匹配
                for (int i = 0; root && i < s.length(); i++)
                {
                    root = root->next[(int)(s[i] - 'a')];
                }
                dfs_query(ans, s, root);
            }
            return ans;
        }

    public:
        vector<vector<string>> suggestedProducts(vector<string> &products, string searchWord)
        {
            TrieNode *dictionary = new TrieNode();
            for (auto &&word : products)
            {
                insertWordToDictory(dictionary, word);
            }
            vector<vector<string>> ans;
            string target;
            for (auto &&ch : searchWord)
            {
                target.push_back(ch);
                ans.push_back(query(dictionary, target));
            }
            return ans;
        }
    };
    ```

    - 统计公共前缀长度方法

    ```cpp
    vector<vector<string>> suggestedProducts(vector<string> &products, string searchWord)
    {
        const int nubmer_products = products.size(), length_word = searchWord.length();
        vector<int> cnt(nubmer_products, 0);
        sort(products.begin(), products.end());
        for (int i = 0; i < nubmer_products; i++)
        {
            int j = 0;
            while (j < length_word && j < products[i].length() && searchWord[j] == products[i][j])
            {
                j++;
            }
            cnt[i] = j;
        }
        vector<vector<string>> ans;
        for (int i = 0; i < length_word; i++)
        {
            vector<string> cur;
            int j = 0;
            while (cur.size() < 3 && j < nubmer_products)
            {
                if (cnt[j] > i)
                {
                    cur.push_back(products[j]);
                }
                j++;
            }
            ans.push_back(cur);
        }
        return ans;
    }
    ```

- [1269. 停在原地的方案数](https://leetcode-cn.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/)

    动态规划，注意可能走到的最右侧位置不会超过总步长的一半，否则没有足够的步数返回，时间复杂度$O(arrLen^2)$

    ```cpp
	int numWays(int steps, int arrLen)
	{
		arrLen = min(arrLen, (steps >> 1) + 1);
		// 因为最远不会超过总步数的一半，否则就走不回来了
		vector<int> prev(arrLen + 2, 0), next(arrLen + 2, 0);
		prev[1] = 1; // 初始位置
		long long v, mode = 1e9 + 7;
		for (int s = 1; s <= steps; s++)
		{
			for (int p = 1; p <= arrLen; p++)
			{
				v = static_cast<long long>(prev[p - 1]) + prev[p] + prev[p + 1];
				next[p] = v % mode;
			}
			prev = next;
		}
		return prev[1];
	}
    ```

- [1277. Count Square Submatrices with All Ones](https://leetcode.com/problems/count-square-submatrices-with-all-ones/)

    求一个0和1组成的矩阵中由1表示的正方形的个数，动态规划

    ```cpp
    int countSquares(vector<vector<int>> &matrix)
    {
        int ans = 0;
        if (matrix.size() > 0 && matrix[0].size() > 0)
        {
            int row = matrix.size(), col = matrix[0].size();
            for (int i = 1; i < row; i++)
            {
                for (int j = 1; j < col; j++)
                {
                    if (matrix[i][j] == 1)
                    {
                        int edge = min(matrix[i - 1][j], matrix[i][j - 1]);
                        if (matrix[i - edge][j - edge] > 0)
                        {
                            matrix[i][j] = edge + 1;
                        }
                        else
                        {
                            matrix[i][j] = edge;
                        }
                    }
                }
            }
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    ans += matrix[i][j];
                }
            }
        }
        return ans;
    }
    ```

- [1287. Element Appearing More Than 25% In Sorted Array](https://leetcode.com/problems/element-appearing-more-than-25-in-sorted-array/)

    寻找升序排序数组中出现次数超过$\frac{1}{4}$的元素

    - 数组有序，则遍历数组统计所有元素出现的次数，直到该次数大于数组元素总数的$\frac{1}{4}$，时间复杂度$O(n)$

    ```cpp
    int findSpecialInteger(vector<int> &arr)
    {
        const int length = arr.size();
        int value = -1, count = 0;
        for (int i = 0; i < length; i++)
        {
            if (arr[i] == value)
            {
                count++;
            }
            else
            {
                count = 1, value = arr[i];
            }
            if (count * 4 > length)
            {
                break;
            }
        }
        return value;
    }
    ```

    - 出现次数超出元素总数的$\frac{1}{4}$，则该数一定会在$\frac{1}{4}$、$\frac{1}{2}$、$\frac{3}{4}$三个位置中的一个出现，统计这三个数的数量是否超出$\frac{1}{4}$即可，在此过程中还可以使用二叉搜索该数在数组中的位置，将时间复杂度降低到$O(log(n))$

    ```cpp
    int findSpecialInteger(vector<int> &arr)
    {
        const int length = arr.size();
        vector<int> indexs{length / 4, length / 2, length * 3 / 4};
        int value = -1, count = 0;
        for (auto &&index : indexs)
        {
            value = arr[index], count = 1;
            int i = index - 1;
            while (i >= 0 && arr[i] == value)
            {
                count++, i--;
            }
            i = index + 1;
            while (i < length && arr[i] == value)
            {
                count++, i++;
            }
            if (count * 4 > length)
            {
                break;
            }
        }
        return value;
    }
    ```

    - 方法二的binary search版本

    ```cpp
    int binary_search(vector<int> &arr, int index, bool left_end)
    {
        int target = arr[index];
        if (left_end)
        {
            // find left-most index so that arr[index]=target
            int left = 0, right = index;
            while (left < right)
            {
                int mid = left + ((right - left - 1) >> 1);
                if (arr[mid] < target)
                {
                    left = mid + 1;
                }
                else
                {
                    right = mid;
                }
            }
            index = left;
        }
        else
        {
            // find right-most index so that arr[index]=target
            int left = index, right = arr.size() - 1;
            while (left < right)
            {
                int mid = left + ((right - left + 1) >> 1);
                if (arr[mid] > target)
                {
                    right = mid - 1;
                }
                else
                {
                    left = mid;
                }
            }
            index = left;
        }
        return index;
    }
    int findSpecialInteger(vector<int> &arr)
    {
        const int length = arr.size();
        vector<int> indexs{length / 4, length / 2, length * 3 / 4};
        int value = -1, count = 0;
        for (auto &&index : indexs)
        {
            int left = binary_search(arr, index, true);
            int right = binary_search(arr, index, false);
            if ((right - left + 1) * 4 > length)
            {
                value = arr[index];
                break;
            }
        }
        return value;
    }
    ```

- [1288. Remove Covered Intervals](https://leetcode.com/problems/remove-covered-intervals/)

    给定n个时间段，移除被其他时间段完全覆盖的时间段，统计还剩下的时间段

    - 顺序扫描所有的时间段interval，如果interval还未被移除，则和用除自己之外的所有其它时间段intervals[j]比较，如果intervals[j]被覆盖，则移除interv[j]，时间复杂度$O(n^2)$

    ```cpp
    int removeCoveredIntervals(vector<vector<int>> &intervals)
    {
        int count = intervals.size();
        vector<bool> removed(count, false);
        for (int i = 0; i < count; i++)
        {
            if (!removed[i])
            {
                int start = intervals[i][0], end = intervals[i][1];
                int j = 0;
                while (j < i)
                {
                    if (!removed[j] && intervals[j][0] >= start && intervals[j][1] <= end)
                    {
                        removed[j] = true;
                    }
                    j++;
                }
                j++; // avoiding to remove intervals[i] (itself)
                while (j < count)
                {
                    if (!removed[j] && intervals[j][0] >= start && intervals[j][1] <= end)
                    {
                        removed[j] = true;
                    }
                    j++;
                }
            }
        }
        for (auto &&remove : removed)
        {
            if (remove)
            {
                count--;
            }
        }
        return count;
    }
    ```

    - 首先在$O(nlog(n))$时间内对intervals按照每个时间段的开时时间即intervals[i][0]排序，则当$j>i$时不可能存在$intervals[i][0]>intervals[j][0]$的情况，即此时intervals[i]不可能覆盖intervals[j]，可以将$O(n^2)$时间内检查的内存循环操作数减少一半

    ```cpp
    int removeCoveredIntervals(vector<vector<int>> &intervals)
    {
        int const count = intervals.size();
        int ret = count;
        sort(intervals.begin(), intervals.end(), [](vector<int> a, vector<int> b) -> bool { return a[0] < b[0]; });
        vector<bool> removed(count, false);
        for (int i = 0; i < count; i++)
        {
            if (!removed[i])
            {
                const int end_bound = intervals[i][1];
                for (int j = i + 1; j < count; j++)
                {
                    if (!removed[j] && intervals[j][1] <= end_bound)
                    {
                        removed[j] = true, ret--;
                    }
                }
            }
        }
        return ret;
    }
    ```

    - 进一步，在排序时首先按照interval[0]升序排列，interval[0]相同的时间区间则按interval[1]升序排列，之后一次遍历intervals时标记当前右界right，如果$intervals[i][1]<right$则interval[i]一定可以被他之前的某个interval覆盖，时间复杂度直接降低到排序时间$O(nlog(n))$

    ```cpp
    int removeCoveredIntervals(vector<vector<int>> &intervals)
    {
        int const count = intervals.size();
        int ret = 0, right_bound = 0;
        sort(intervals.begin(), intervals.end(), [](vector<int> a, vector<int> b) -> bool { return (a[0] != b[0]) ? (a[0] < b[0]) : (a[1] < b[1]); });
        for (auto &&interval : intervals)
        {
            if (interval[1] > right_bound)
            {
                ret++, right_bound = interval[1];
            }
        }
        return ret;
    }
    ```

- [1289. Minimum Falling Path Sum II](https://leetcode.com/problems/minimum-falling-path-sum-ii/)

    和[931. Minimum Falling Path Sum](https://leetcode.com/problems/minimum-falling-path-sum/)不同的是，本题限定如果第i行选在第j列，则第i+1行不能选择第j列
    
    - 动态规划，用dp[i][j]表示以arr[i][j]为结尾的Falling Path Sum，状态转移方程为$dp[i][j]=arr[i][j]+min(dp[i-1][k]),k \neq j$，然后求$min(dp.back())$即可，时间复杂度$O(n^3)$，其中n为方阵arr的边长

    ```cpp
    int minFallingPathSum(vector<vector<int>> &arr)
    {
        int ans = 0;
        if (arr.size() > 0 && arr[0].size() > 0)
        {
            int rows = arr.size(), cols = arr[0].size();
            for (int i = 1; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    int temp = numeric_limits<int>::max(), k = 0;
                    while (k < j)
                    {
                        temp = min(temp, arr[i - 1][k++]);
                    }
                    k++;
                    while (k < cols)
                    {
                        temp = min(temp, arr[i - 1][k++]);
                    }
                    arr[i][j] += temp;
                }
            }
            ans = numeric_limits<int>::max();
            for (auto &&v : arr.back())
            {
                ans = min(ans, v);
            }
        }
        return ans;
    }
    ```
    
    - 进一步优化，可以将求$min(dp[i-1][k]),k \neq j$的过程转化为求dp[i-1]中的最小值$dp[index_0]$和第二小值$dp[index_1]$，则$dp[i][j]=arr[i][j]+\left\{\begin{matrix} dp[i-1][index_0], index_0 \neq j\\  dp[i-1][index_1],index_0 = j \end{matrix}\right.$然后求$min(dp.back())$即可，时间复杂度降低到$O(n^2)$，其中n为方阵arr的边长

    ```cpp
    int minFallingPathSum(vector<vector<int>> &arr)
    {
        int ans = 0;
        if (arr.size() > 0 && arr[0].size() > 0)
        {
            int rows = arr.size(), cols = arr[0].size();
            for (int i = 1; i < rows; i++)
            {
                int first_min = numeric_limits<int>::max(), second_min = first_min;
                int first = -1;
                for (int j = 0; j < cols; j++)
                {
                    if (arr[i - 1][j] < first_min)
                    {
                        second_min = first_min, first_min = arr[i - 1][j], first = j;
                    }
                    else if (arr[i - 1][j] < second_min)
                    {
                        second_min = arr[i - 1][j];
                    }
                }
                for (int j = 0; j < cols; j++)
                {
                    arr[i][j] += (first == j) ? second_min : first_min;
                }
            }
            ans = *min_element(arr.back().begin(), arr.back().end());
        }
        return ans;
    }
    ```

- [1291. Sequential Digits](https://leetcode.com/problems/sequential-digits/)

    - 递归构造所有符合条件的数

    ```cpp
    string next(string s)
    {
        string ret;
        if (!s.empty() && s.back() < '9')
        {
            s.push_back(s.back() + 1);
            ret = s.substr(1, s.length() - 1);
        }
        return ret;
    }
    vector<int> sequentialDigits(int low, int high)
    {
        vector<int> ans;
        string a = to_string(low), b = to_string(high), start;
        char start_digit = '0';
        int width_a = a.length(), width_b = b.length();
        for (int i = 0; i < width_a; i++)
        {
            start.push_back(++start_digit);
        }
        for (int width = width_a; width <= width_b; width++)
        {
            string cur = start;
            start.push_back(++start_digit);
            while (!cur.empty() && cur.length() == width_a && cur.compare(a) < 0)
            {
                cur = next(cur);
            }
            while (!cur.empty() && (cur.length() < width_b || (cur.length() == width_b && cur.compare(b) <= 0)))
            {
                ans.push_back(stoi(cur));
                cur = next(cur);
            }
        }
        return ans;
    }
    ```

    - 打表

    ```cpp
    vector<int> sequentialDigits(int low, int high)
    {
        vector<int> ans, table{12, 23, 34, 45, 56, 67, 78, 89, 123, 234, 345, 456, 567, 678, 789, 1234, 2345, 3456, 4567, 5678, 6789, 12345, 23456, 34567, 45678, 56789, 123456, 234567, 345678, 456789, 1234567, 2345678, 3456789, 12345678, 23456789, 123456789};
        int i = 0, length = table.size();
        while (i < length && table[i] < low)
        {
            i++;
        }
        while (i < length && table[i] <= high)
        {
            ans.push_back(table[i++]);
        }
        return ans;
    }
    ```
