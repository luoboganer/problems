# 201-300

- [201. Bitwise AND of Numbers Range](https://leetcode.com/problems/bitwise-and-of-numbers-range/)

    - 顺序遍历，时间复杂度$O(n)$，借助0与任何数按位与之后均为0剪枝（没有剪枝会TLE）

    ```cpp
    int rangeBitwiseAnd(int m, int n)
    {
        long long ret = 0xffffffff;
        for (long long v = m; ret != 0 && v <= n; v++)
        {
            ret &= v;
        }
        return (int)ret;
    }
    ```

    - 可以观察并验证[m,n]中所有数字按位与的结果是[m,n]区间内所有数字二进制表示的左侧相同的部分，因此从一个32bits全1的mash开始向左移动即可，时间复杂度$O(1)$

    ```cpp
    int rangeBitwiseAnd(int m, int n)
    {
        int width = 0;
        while (m != n)
        {
            width++;
            m >>= 1, n >>= 1;
        }
        return m << width;
    }
    ```

- [207. Course Schedule](https://leetcode.com/problems/course-schedule/)

    **相关题目**

    - [210. Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)
    - [630. Course Schedule III](https://leetcode.com/problems/course-schedule-iii/)
    - [1462. Course Schedule IV](https://leetcode.com/problems/course-schedule-iv/)

        **拓扑排序问题-BFS/DFS**
    
    所有课程的前后依赖关系构成了directed graph/tree，递归/迭代式的检查所有入度为0（没有先修依赖）的课程，然后首先安排这些课程，然后将其子节点的入度减1，最后比较安排了的课程数和总的课程数是否相同即可

    - BFS邻接矩阵写法，时间复杂度$O(V*E)$

	```cpp
	bool canFinish(int numCourses, vector<vector<int>> &prerequisites)
	{
		/*
			1. 邻接矩阵会导致后期遍历的时间复杂度变为O(V*E)
			2. 使用邻接链表会将后期遍历的时间复杂度降为O(V+E)
		*/
		// 建立邻接矩阵，统计每个node的入度
		vector<vector<int>> graph(numCourses, vector<int>(numCourses, 0)); // 1 for edge
		vector<int> indegrees(numCourses, 0);
		for (auto &&edge : prerequisites)
		{
			graph[edge[0]][edge[1]] = 1; // a edge from start to end (directed graph/tree)
			indegrees[edge[1]]++;
		}
		// 当前入度为0的节点即可安排，其父节点（prerequisite）被安排时该节点变为入度0
		queue<int> qe;
		for (auto node = 0; node < numCourses; node++)
		{
			if (indegrees[node] == 0)
			{
				qe.push(node);
				indegrees[node] = -1; //  -1 标识该node/course已经安排
			}
		}
		int count = 0;
		while (!qe.empty())
		{
			int cur_node = qe.front();
			qe.pop(), count++; // node代表的课程可以安排
			for (auto node = 0; node < numCourses; node++)
			{
				/* cur_node的后续课程的入度均减1，因为cur_node已经安排 */
				if (graph[cur_node][node] == 1)
				{
					indegrees[node]--;
				}
				if (indegrees[node] == 0)
				{
					qe.push(node);
					indegrees[node] = -1;
				}
			}
		}
		// 当可安排的课程数量等于总的课程数量时安排完毕
		return count == numCourses;
	}
	```

    - BFS邻接链表写法，时间复杂度$O(V+E)$

	```cpp
	bool canFinish(int numCourses, vector<vector<int>> &prerequisites)
	{
		// 建立邻接矩阵，统计每个node的入度
		vector<vector<int>> graph(numCourses); // 1 for edge
		vector<int> indegrees(numCourses, 0), bfs;
		for (auto &&edge : prerequisites)
		{
			graph[edge[0]].push_back(edge[1]); // a edge from start to end (directed graph/tree)
			indegrees[edge[1]]++;
		}
		// 当前入度为0的节点即可安排，其父节点（prerequisite）被安排时该节点变为入度0
		for (auto node = 0; node < numCourses; node++)
		{
			if (indegrees[node] == 0)
			{
				bfs.push_back(node);
			}
		}
		for (auto i = 0; i < bfs.size(); i++)
		{
			for (auto &&j : graph[bfs[i]])
			{
				if (--indegrees[j] == 0)
				{
					bfs.push_back(j);
				}
			}
		}
		// 当可安排的课程数量等于总的课程数量时安排完毕
		return bfs.size() == numCourses;
	}
	```

	-  DFS，时间复杂度$O(V+E)$

	```cpp
    class Solution
    {
    private:
        bool hasCycle(vector<vector<int>> &graph, vector<int> &visited, vector<int> &onPath, int cur_node)
        {
            if (!visited[cur_node])
            {
                visited[cur_node] = true, onPath[cur_node] = true;
                for (auto &&node : graph[cur_node])
                {
                    if (onPath[node] || hasCycle(graph, visited, onPath, node))
                    {
                        return true;
                    }
                }
            }
            onPath[cur_node] = false; // 回溯思想，该数组是用来判断cur_node在DFS过程中所有从节点node出发的路径中是否重复出现（有环）
            return false;
        }

    public:
        bool canFinish(int numCourses, vector<vector<int>> &prerequisites)
        {
            // 建立邻接链表
            vector<vector<int>> graph(numCourses); // 1 for edge
            for (auto &&edge : prerequisites)
            {
                graph[edge[0]].push_back(edge[1]); // a edge from start to end (directed graph/tree)
            }
            // DFS检查是否图中存在环路
            vector<int> visited(numCourses, 0), onPath(numCourses, 0);
            for (auto i = 0; i < numCourses; i++)
            {
                if (!visited[i] && hasCycle(graph, visited, onPath, i))
                {
                    return false;
                }
            }
            return true;
        }
    };
	```

- [208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)

    建立字典树，有插入单词和查询单词函数

    ```cpp
    class Trie
    {
    private:
        struct TrieNode
        {
            /* data */
            bool is_word;
            TrieNode *next[26];
            TrieNode()
            {
                is_word = false;
                for (int i = 0; i < 26; i++)
                {
                    next[i] = nullptr;
                }
            }
        };

        TrieNode *dictionary = nullptr;

    public:
        /** Initialize your data structure here. */
        Trie()
        {
            dictionary = new TrieNode();
        }

        /** Inserts a word into the trie. */
        void insert(string word)
        {
            TrieNode *root = dictionary;
            for (int i = 0; i < word.length(); i++)
            {
                int index = (int)(word[i] - 'a');
                if (!root->next[index])
                {
                    root->next[index] = new TrieNode();
                }
                root = root->next[index];
            }
            root->is_word = true;
        }

        /** Returns if the word is in the trie. */
        bool search(string word)
        {
            TrieNode *root = dictionary;
            for (auto &&ch : word)
            {
                int index = (int)(ch - 'a');
                if (root->next[index])
                {
                    root = root->next[index];
                }
                else
                {
                    return false;
                }
            }
            return root->is_word;
        }

        /** Returns if there is any word in the trie that starts with the given prefix. */
        bool startsWith(string prefix)
        {
            TrieNode *root = dictionary;
            for (auto &&ch : prefix)
            {
                int index = (int)(ch - 'a');
                if (root->next[index])
                {
                    root = root->next[index];
                }
                else
                {
                    return false;
                }
            }
            return true;
        }
    };
    ```

- [209](https://leetcode.com/problems/minimum-size-subarray-sum/)

    用左右双指针left、right设置滑动窗口capacity来满足sum和的要求，求滑动窗口可能的最小值即可
    
    ```cpp
    int minSubArrayLen(int s, vector<int> &nums)
    {
        int left = 0, right = 0, capacity = 0, count = nums.size(), ret = numeric_limits<int>::max();
        while (right < count)
        {
            // while (right < count && capacity < s)
            // {
            // 	capacity += nums.at(right++);
            // }
            // while (left < count && capacity >= s)
            // {
            // 	ret = min(right - left, ret);
            // 	capacity -= nums.at(left++);
            // }

            capacity += nums[right++];
            while (capacity >= s)
            {
                ret = min(ret, right - left);
                capacity -= nums[left--];
            }
        }
        ret = (ret == numeric_limits<int>::max()) ? 0 : ret;
        return ret;
    }
    ```

- [210. 课程表 II](https://leetcode-cn.com/problems/course-schedule-ii/)

    类似于[207. Course Schedule](https://leetcode.com/problems/course-schedule/)，输出拓扑排序的结果即可

    ```cpp
	vector<int> findOrder(int numCourses, vector<vector<int>> &prerequisites)
	{
		vector<int> indegrees(numCourses, 0); // 统计每个节点的入度
		vector<vector<int>> graph(numCourses);
		for (auto &e : prerequisites)
		{
			indegrees[e[0]]++;
			graph[e[1]].emplace_back(e[0]);
		}
		vector<int> bfs;
		for (int i = 0; i < numCourses; i++)
		{
			if (indegrees[i] == 0)
			{
				//没有先决条件的课程
				bfs.emplace_back(i);
			}
		}
		for (int i = 0; i < bfs.size(); i++)
		{
			for (auto &node : graph[bfs[i]])
			{
				if (indegrees[node] > 0)
				{
					if (--indegrees[node] == 0)
					{
						bfs.emplace_back(node);
					}
				}
			}
		}
		return bfs.size() == numCourses ? bfs : vector<int>{};
	}
    ```

- [211. Add and Search Word - Data structure design](https://leetcode.com/problems/add-and-search-word-data-structure-design/)

    字典树的建立与使用，比[208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)多了查询单词时满足通配符的要求，通配符的查询通过递归调用来实现

    ```cpp
    struct TrieNode
    {
        bool is_word;
        TrieNode *next[26];
        TrieNode()
        {
            is_word = false;
            for (int i = 0; i < 26; i++)
            {
                next[i] = nullptr;
            }
        }
    };

    class WordDictionary
    {
    private:
        TrieNode *dictionary = nullptr;

    public:
        /** Initialize your data structure here. */
        WordDictionary()
        {
            dictionary = new TrieNode();
        }

        /** Adds a word into the data structure. */
        void addWord(string word)
        {
            TrieNode *root = dictionary;
            for (int i = 0; i < word.length(); i++)
            {
                int index = (int)(word[i] - 'a');
                if (!root->next[index])
                {
                    root->next[index] = new TrieNode();
                }
                root = root->next[index];
            }
            root->is_word = true;
        }

        /** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
        bool search_helper(string word, TrieNode *root)
        {
            for (int i = 0; i < word.length(); i++)
            {
                char ch = word[i];
                if (ch == '.')
                {
                    bool ret = false;
                    for (int j = 0; !ret && j < 26; j++)
                    {
                        if (root->next[j])
                        {
                            ret = search_helper(word.substr(i + 1, word.length() - i - 1), root->next[j]);
                        }
                    }
                    return ret;
                }
                else if (isalpha(ch))
                {
                    int index = (int)(ch - 'a');
                    if (root->next[index])
                    {
                        root = root->next[index];
                    }
                    else
                    {
                        return false;
                    }
                }
                else
                {
                    return false;
                }
            }
            if (root->is_word)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        bool search(string word)
        {
            return search_helper(word, dictionary);
        }
    };
    ```

- [212. Word Search II](https://leetcode.com/problems/word-search-ii/)

    与[79. Word Search](https://leetcode.com/problems/word-search/)类似，在给定的二维字符表格中搜索可能的单词，当单词数量太多时采用字典树（Trie）数据结构来存储所有单词，DFS方式遍历整个字符表格

    ```cpp
    class Solution
    {
    private:
        struct TrieNode
        {
            bool isword;
            TrieNode *next[26];
            TrieNode()
            {
                isword = false;
                for (int i = 0; i < 26; i++)
                {
                    next[i] = nullptr;
                }
            }
        };
        void dfs_helper(TrieNode *root, vector<vector<char>> &board, int i, int j, string &cur, vector<string> &ans)
        {
            if (i >= 0 && j >= 0 && i < board.size() && j < board[0].size() && board[i][j] != '#')
            {
                // 搜索范围没有超出board且当前字符未被使用
                char ch = board[i][j];
                int index = (int)(ch - 'a');
                if (root->next[index])
                {
                    board[i][j] = '#';
                    cur.push_back(ch);
                    if (root->next[index]->isword)
                    {
                        ans.push_back(cur);                // find a word
                        root->next[index]->isword = false; // remove the word to avoiding duplication
                    }
                    // 递归查找
                    vector<int> directions{1, 0, -1, 0, 1};
                    root = root->next[index];
                    for (int k = 0; k < 4; k++)
                    {
                        dfs_helper(root, board, i + directions[k], j + directions[k + 1], cur, ans);
                    }
                    cur.pop_back();
                    board[i][j] = ch;
                }
            }
        }

    public:
        vector<string> findWords(vector<vector<char>> &board, vector<string> &words)
        {
            // 建立字典树
            TrieNode *dictionary = new TrieNode();
            for (auto &&word : words)
            {
                TrieNode *root = dictionary;
                for (auto &&ch : word)
                {
                    int index = (int)(ch - 'a');
                    if (!root->next[index])
                    {
                        root->next[index] = new TrieNode();
                    }
                    root = root->next[index];
                }
                root->isword = true;
            }
            // 在二维字母表中dfs搜索字典树中的每个单词
            vector<string> ans;
            if (board.size() > 0 && board[0].size() > 0)
            {
                int m = board.size(), n = board[0].size();
                string cur;
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        dfs_helper(dictionary, board, i, j, cur, ans);
                    }
                }
            }
            return ans;
        }
    };
    ```

- [215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)

    - 排序，默认quick sort时间复杂度$O(nlog(n))$

    ```cpp
    int findKthLargest(vector<int> &nums, int k)
    {
        sort(nums.begin(), nums.end());
        return nums[nums.size() - k];
    }
    ```

    - 维护一个大顶堆，时间复杂度$O(n)$

    ```cpp
    int findKthLargest(vector<int> &nums, int k)
    {
        int ret;
        make_heap(nums.begin(), nums.end()); // 大顶堆
        for (int i = 0; i < k; i++)
        {
            pop_heap(nums.begin(), nums.end());
            ret = nums.back();
            nums.pop_back();
        }
        return ret;
    }
    ```

- [216](https://leetcode.com/problems/combination-sum-iii/)

    从$[1,2,3,4,5,6,7,8,9]$这9个数字中选择k个使其和为n，则k和n的取值范围为$1 \le k \le 9, \frac{k(k+1)}{2} \le n \le \frac{k(19-k)}{2}$，在此有效范围内，$f(k,n)$可以在选择任一个数$x$后递归到$f(k-1,n-x)$

    ```cpp
    void dfs_helper(vector<vector<int>> &ans, vector<int> cur, int n, int k, int start_value)
    {
        if (n == 0 && k == 0)
        {
            ans.push_back(cur);
        }
        else if (n > 0 && k > 0)
        {
            for (int v = start_value; v < 10; v++)
            {
                cur.push_back(v);
                dfs_helper(ans, cur, n - v, k - 1, v + 1);
                cur.pop_back();
            }
        }
    }
    vector<vector<int>> combinationSum3(int k, int n)
    {
        vector<vector<int>> ans;
        if (k >= 1 && k <= 9 && n >= k * (k + 1) / 2 && n <= k * (19 - k) / 2)
        {
            vector<int> cur;
            dfs_helper(ans, cur, n, k, 1);
        }
        return ans;
    }
    ```

- [217](https://leetcode.com/problems/contains-duplicate/submissions/)

    在给定数组中查找是否有重复值，典型的集合的应用

    - 先排序然后一遍扫描，O(nlog(n))

    ```cpp
    bool containsDuplicate(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int count = nums.size() - 1;
        bool res = false;
        for (int i = 0; i < count; i++)
        {
        	if (nums.at(i) == nums.at(i + 1))
        	{
        		res = true;
        		break;
        	}
        }
        return res;
    }
    ```

    - hash set，近似O(n)

    ```cpp
    bool containsDuplicate(vector<int> &nums)
    {
        unordered_set<int> nums_set;
        int count = nums.size();
        bool res = false;
        for (int i = 0; i < count; i++)
        {
            if (nums_set.find(nums.at(i)) != nums_set.end())
            {
                res = true;
                break;
            }
            nums_set.insert(nums.at(i));
        }
        return res;
    }
    ```

    在n较小的情况下，hash查询的效率较高，在n较大的情况下，快速排序扫描的效率较高

- [219](https://leetcode.com/problems/contains-duplicate-ii/)

    在给定数组nums中下标i和j的间距在给定值k以内的滑动窗口内($abs(i-j) \le k$)是否存在重复值
    
    - 在线性扫描的过程中用hash map记录已经出现过的数下标，然后检查该数是否在之前已经出现并检查下标差距是否小于k（在滑动窗口内）即可

    ```cpp
    bool containsNearbyDuplicate(vector<int>& nums, int k) {
        bool ret=false;
		if(nums.size()>0){
			unordered_map<int,int> record;
			for(int i=0;i<nums.size();i++){
				if(record.find(nums[i])==record.end()){
					record[nums[i]]=i;
				}else{
					if(i-record[nums[i]]<=k){
						ret=true;
						break;
					}else{
						record[nums[i]]=i;
					}
				}
			}
		}
		return ret;
    }
    ```

    - 在线性扫描给定数组的过程中，用一个hash set记录长度为k的滑动窗口内出现过的数以备查询

    ```cpp
    bool containsNearbyDuplicate(vector<int> &nums, int k)
    {
        bool ret = false;
        if (nums.size() > 0)
        {
            unordered_set<int> record;
            for (int i = 0; i < nums.size(); i++)
            {
                if (record.find(nums[i]) != record.end())
                {
                    ret = true;
                    break;
                }
                else
                {
                    record.insert(nums[i]);
                    // 窗口向右滑动，把右侧下一个数包含进来
                    if (i >= k)
                    {
                        record.erase(nums[i - k]);
                        // slide window 向右滑动，删除窗口外左侧已经滑出的一个数
                    }
                }
            }
        }
        return ret;
    }
    ```

- [220](https://leetcode.com/problems/contains-duplicate-iii/)

    - 题意：求给定数组nums中是否存在两个数nums[i]和nums[j]，下标差小于等于k（即在长度为k的滑动窗口内，$abs(i-j) \le k$）且两数差的绝对值小于等于t（$abs(nums[i]-nums[j]) \le t$）
    
    - 对比：题目要求即是在[219](https://leetcode.com/problems/contains-duplicate-ii/)的基础上添加了两数差的绝对值小于等于t（$abs(nums[i]-nums[j]) \le t$）的要求

    - 思路：线性遍历数组nums的过程中用一个hash set来保存长度为k的滑动窗口内的数，并检验当前遍历到的nums[i]是否满足与hash set内的任何一个数差值小于等于限定值t即可

    - 代码实现：

    ```cpp
    bool containsNearbyAlmostDuplicate(vector<int> &nums, int k, int t)
    {
        bool ret = false;
        long long threshold = (long long)(t);
        if (nums.size() > 0 && k > 0 && t >= 0)
        {
            unordered_set<int> record;
            for (int i = 0; !ret && i < nums.size(); i++)
            {
                int cur_v = (long long)(nums[i]);
                // 防止int型减法溢出
                for (auto &&v : record)
                {
                    if (abs(cur_v - (long long)(v)) <= threshold)
                    {
                        ret = true;
                        break;
                    }
                }
                record.insert(nums[i]);
                // 窗口向右滑动，把右侧下一个数包含进来
                if (i >= k)
                {
                    record.erase(nums[i - k]);
                    // slide window 向右滑动，删除窗口外左侧已经滑出的一个数
                }
            }
        }
        return ret;
    }
    ```

    该方法可以通过所有测试用例，但是会TLE(Time Limit Exceeded)，分析代码发现外层循环遍历每个数是必须的，无法降低复杂度，因此在内层循环hash set的遍历处着手，将其改为set，然后利用排序set的lower bound特性。另一方面将 $abs(nums[i]-nums[j]) \le t$ 分解为
    $$nums[j] \ge nums[i] - t \quad and \quad nums[j] \le nums[i] + t$$
    分别进行判断，从而降低遍历set的时间复杂度，代码如下：

    参考[blog](https://leetcode.com/problems/contains-duplicate-iii/discuss/61641/C%2B%2B-using-set-(less-10-lines)-with-simple-explanation.)

    ```cpp
    bool containsNearbyAlmostDuplicate(vector<int> &nums, int k, int t)
    {
        bool ret = false;
        long long threshold = (long long)(t);
        if (nums.size() > 0 && k > 0 && t >= 0)
        {
            set<long long> record;
            for (int i = 0; i < nums.size(); i++)
            {
                if (i > k)
                {
                    record.erase(nums[i - k - 1]);
                    // slide window 向右滑动，删除窗口外左侧已经滑出的一个数
                }
                auto it = record.lower_bound((long long)(nums[i]) - (long long)t);
                if ((it != record.end()) && (*it <= ((long long)(nums[i]) + (long long)(t))))
                {
                    ret = true;
                    break;
                }
                record.insert(nums[i]);
            }
        }
        return ret;
    }
    ```

- [221. Maximal Square](https://leetcode.com/problems/maximal-square/)

    动态规划，时间复杂度$O(m*n)$

    ```cpp
    int maximalSquare(vector<vector<char>> &matrix)
    {
        /*
            动态规划，dp[i][j]标识到当前位置所能组成的最大正方形的边长
        */
        int area = 0, max_side_length = 0;
        if (matrix.size() > 0 && matrix[0].size() > 0)
        {
            int rows = matrix.size(), cols = matrix[0].size();
            vector<vector<int>> dp(rows, vector<int>(cols, 0));
            for (auto i = 0; i < rows; i++)
            {
                for (auto j = 0; j < cols; j++)
                {
                    if (i == 0 || j == 0)
                    {
                        dp[i][j] = (int)(matrix[i][j] - '0');
                    }
                    else
                    {
                        if (matrix[i][j] == '1')
                        {
                            dp[i][j] = min(min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                        }
                    }
                    max_side_length = max(dp[i][j], max_side_length);
                }
            }
            area = max_side_length * max_side_length;
        }
        return area;
    }
    ```

- [222. Count Complete Tree Nodes](https://leetcode.com/problems/count-complete-tree-nodes/)

    统计一颗完全二叉树的节点

    - $O(n)$时间遍历所有节点

    ```cpp
    int countNodes(TreeNode *root)
    {
        int ans = 0;
        if (root)
        {
            ans++;
            ans += countNodes(root->left);
            ans += countNodes(root->right);
        }
        return ans;
    }
    ```

    - $O((log(n))^2)$时间内通过计算完全树的高度来计算其节点数量

    ```cpp
    int height(TreeNode *root)
    {
        int ans = -1;
        while (root)
        {
            ans++;
            root = root->left;
        }
        return ans;
    }
    int countNodes(TreeNode *root)
    {
        int ans = 0;
        if (root)
        {
            int h = height(root);
            if (h == height(root->right) + 1)
            {
                ans = (1 << h) + countNodes(root->right);
            }
            else
            {
                ans = (1 << (h - 1)) + countNodes(root->left);
            }
        }
        return ans;
    }
    ```

- [227](https://leetcode.com/problems/basic-calculator-ii/)

    计算只包含加减乘除和正整数的表达式的值

    - 方法一，给加减和乘除定义不同的优先级，然后使用栈将其转化为逆波兰表达式，最终求值的

    ```cpp
    int calculate(string s) {
        vector<string> items;
        unordered_map<string,int> priority;
        priority["+"]=0;
        priority["-"]=0;
        priority["*"]=1;
        priority["/"]=1;
        string item;
        stack<string> ops;
        for (int i = 0; i < s.length(); i++)
        {
            if(s[i]==' '){
                continue;
            }else if(isdigit(s[i])){
                item+=s[i];
            }else{
                if(item.length()>0){
                    items.push_back(item);
                    item.clear();
                }
                item=s[i];
                while(!ops.empty() && priority[item]<=priority[ops.top()]){
                    items.push_back(ops.top());
                    ops.pop();
                }
                ops.push(item);
                item.clear();
            }
        }
        if(item.length()>0){
            items.push_back(item);
        }
        while (!ops.empty())
        {
            items.push_back(ops.top());
            ops.pop();
        }
        // for calculate
        stack<int> nums;
        for (const string &x:items)
        {
            if(isdigit(x[0])){
                nums.push(stoi(x));
            }else
            {
                char c=x[0];
                int b=nums.top();
                nums.pop();
                int a=nums.top();
                nums.pop();
                switch (c)
                {
                case '+':
                    /* code */
                    nums.push(a+b);
                    break;
                case '-':
                    /* code */
                    nums.push(a-b);
                    break;
                case '*':
                    /* code */
                    nums.push(a*b);
                    break;
                case '/':
                    /* code */
                    nums.push(a/b);
                    break;
                default:
                    break;
                }
            }
        }
        return nums.top();
    }
    ```

    - 方法二，因为没有使用括号，因此可以在 one pass 扫描的时候直接计算高优先级的乘除的值，并将减法转化为相反数的加法，最后将所有的数直接求和即可

    ```cpp
    int calculate(string s)
    {
        int ans = 0;
        if (s.length() > 0)
        {
            stack<int> nums;
            string ops = "+-*/";
            nums.push(0);
            char op = '+';
            int cur_value = 0;
            string::iterator it = s.begin();
            while (true)
            {
                if (isdigit(*it))
                {
                    cur_value = cur_value * 10 + (int)(*it - '0');
                }
                if (it + 1 == s.end() || (ops.find(*it) != ops.npos))
                {
                    if (op == '+')
                    {
                        nums.push(cur_value);
                    }
                    else if (op == '-')
                    {
                        nums.push(-cur_value);
                    }
                    else if (op == '*')
                    {
                        int temp = nums.top();
                        nums.pop();
                        nums.push(temp * cur_value);
                    }
                    else if (op == '/')
                    {
                        int temp = nums.top();
                        nums.pop();
                        nums.push(temp / cur_value);
                    }
                    cur_value = 0;
                    op = *it;
                }
                if (it + 1 == s.end())
                {
                    break;
                }
                else
                {
                    it++;
                }
            }
            while (!nums.empty())
            {
                ans += nums.top();
                nums.pop();
            }
        }
        return ans;
    }
    ```

- [229](https://leetcode.com/problems/majority-element-ii/)

    在给定数组中寻找出现次数超过$\frac{1}{3}$的数字，是[169](https://leetcode.com/problems/majority-element/)题的升级版，同样用hashmap进行统计，排序，摩尔投票等方法解决。

    ```cpp
    vector<int> majorityElement(vector<int> &nums)
    {
        // method 1, hash map counting

        // unordered_map<int, int> count;
        // for (auto x : nums)
        // {
        // 	if (count.find(x) != count.end())
        // 	{
        // 		count[x]++;
        // 	}
        // 	else
        // 	{
        // 		count[x] = 1;
        // 	}
        // }
        // int length_one_of_third = nums.size() / 3;
        // vector<int> ret;
        // for (const auto &pair : count)
        // {
        // 	if (pair.second > length_one_of_third)
        // 	{
        // 		ret.push_back(pair.first);
        // 	}
        // }
        // return ret;

        // method 2, mooer voting
        vector<int> ret;
        if (nums.size() > 0)
        {
            int cur_majority_A = nums[0], cur_majority_B = nums[0];
            int cur_votes_A = 0, cur_votes_B = 0;
            for (auto x : nums)
            {
                // 投票过程
                if (x == cur_majority_A)
                {
                    cur_votes_A++;
                    continue;
                }
                if (x == cur_majority_B)
                {
                    cur_votes_B++;
                    continue;
                }
                if (cur_votes_A == 0)
                {
                    cur_majority_A = x;
                    cur_votes_A = 1;
                    continue;
                }
                if (cur_votes_B == 0)
                {
                    cur_majority_B = x;
                    cur_votes_B = 1;
                    continue;
                }

                // 此时AB均为被投票且他们的得票数都大于0，因此都减一分
                cur_votes_A--;
                cur_votes_B--;
            }
            /*
            投票结束，题目并未保证给定数组中一定有两个出现次数超过1/3的数字，
            所以还要检查这两个数出现的次数是否真的超过1/3,
            这是因为摩尔投票法仅能找出出现次数超过一半的数而不能找出出现次数最多的众数
        	*/
            cur_votes_A = 0, cur_votes_B = 0;
            for (auto x : nums)
            {
                if (x == cur_majority_A)
                {
                    cur_votes_A++;
                }
                else if (x == cur_majority_B)
                {
                    cur_votes_B++;
                }
            }
            if (cur_votes_A > nums.size() / 3)
            {
                ret.push_back(cur_majority_A);
            }
            if (cur_votes_B > nums.size() / 3)
            {
                ret.push_back(cur_majority_B);
            }
        }
        return ret;
    }
    ```

- [230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

    在给定二叉搜索树中找到第K小的数，二叉搜索树中序遍历即为升序数列，则迭代式中序遍历这棵树，遍历到第K个输出值时，返回结果

    ```cpp
    int kthSmallest(TreeNode *root, int k)
    {
        int count = 0, ans;
        stack<TreeNode *> st;
        TreeNode *cur = root;
        while (cur || !st.empty())
        {
            if (cur)
            {
                st.push(cur), cur = cur->left;
            }
            else
            {
                count++;
                if (count == k)
                {
                    ans = st.top()->val;
                    break;
                }
                else
                {
                    cur = st.top()->right;
                    st.pop();
                }
            }
        }
        return ans;
    }
    ```

- [232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/)
    
    - 使用两个队列来模拟栈的先进后出操作，其中每一次的peek操作时间复杂度均为O(n)，其它操作时间复杂度为$O(1)$，因此总的时间复杂度是$O(n^2)$

	```cpp
	class MyQueue {
	public:
		/** Initialize your data structure here. */
		stack<int> st1,st2;
		MyQueue() {
			while(!st1.empty()){
				st1.pop();
			}
			while(!st2.empty()){
				st2.pop();
			}
		}

		/** Push element x to the back of queue. */
		void push(int x) {
			st1.push(x);
		}

		/** Removes the element from in front of queue and returns that element. */
		int pop() {
			while(!st1.empty()){
				st2.push(st1.top());
				st1.pop();
			}
			int val=st2.top();
			st2.pop();
			while(!st2.empty()){
				st1.push(st2.top());
				st2.pop();
			}
			return val;
		}

		/** Get the front element. */
		int peek() {
			while(!st1.empty()){
				st2.push(st1.top());
				st1.pop();
			}
			int val=st2.top();
			while(!st2.empty()){
				st1.push(st2.top());
				st2.pop();
			}
			return val;
		}

		/** Returns whether the queue is empty. */
		bool empty() {
			return st1.empty();
		}
	};
	```

    - 降低peek操作时间复杂度到$O(1)$，其它操作时间复杂度为$O(1)$，因此总的时间复杂度是$O(n)$

	```cpp
	class MyQueue
	{
	public:
		/** Initialize your data structure here. */
		stack<int> st1, st2;
		int count;
		MyQueue()
		{
			while (!st1.empty())
			{
				st1.pop();
			}
			while (!st2.empty())
			{
				st2.pop();
			}
			count = 0;
		}

		/** Push element x to the back of queue. */
		void push(int x)
		{
			st1.push(x);
			count++;
		}

		/** Removes the element from in front of queue and returns that element. */
		int pop()
		{
			int ret = peek();
			count--;
			st2.pop();
			return ret;
		}

		/** Get the front element. */
		int peek()
		{
			// 这里看起来是O(n)，实际上每个元素在st2种最多push一次，pop一次，因此peek的时间复杂度为O(1)
			if (st2.empty())
			{
				while (!st1.empty())
				{
					st2.push(st1.top());
					st1.pop();
				}
			}
			return st2.top();
		}

		/** Returns whether the queue is empty. */
		bool empty()
		{
			return count == 0;
		}
	};
	```

- [234](https://leetcode.com/problems/palindrome-linked-list/)

    判断一个链表是否回文
    
    - 没有空间使用限制时，可以one pass将链表值存入vector<int>，然后two pass对比是否相同

    ```cpp
    bool isPalindrome(ListNode *head)
    {
        bool ret = true;
        if (head)
        {
            vector<int> nums;
            ListNode *cur = head;
            while (cur)
            {
                nums.push_back(cur->val);
                cur = cur->next;
            }
            cur = head;
            for (int i = nums.size() - 1; i >= 0; i--)
            {
                if (nums[i] != cur->val)
                {
                    ret = false;
                    break;
                }
                else
                {
                    cur = cur->next;
                }
            }
        }
        return ret;
    }
    ```

    - 在$O(n)$时间和$O(1)$空间下，使用slow和fast两个指针，第一次遍历找到链表中点，然后翻转链表后半部分并和前半部分进行比较

    ```cpp
    bool isPalindrome(ListNode *head)
    {
        bool ret = true;
        if (head)
        {
            ListNode *slow = head, *fast = head;
            while (fast && fast->next)
            {
                slow = slow->next;
                fast = fast->next->next;
            }
            if (fast)
            {
                slow = slow->next;
            }
            // this time, pointer slow is the begin of the last half of the linked list
            // reverse the last half of the linked list
            ListNode *pre = nullptr, *next = nullptr;
            while (slow)
            {
                next = slow->next;
                slow->next = pre;
                pre = slow;
                slow = next;
            }
            slow = pre;  // negative
            next = head; // positive
            while (slow)
            {
                if (slow->val != next->val)
                {
                    ret = false;
                    break;
                }
                else
                {
                    slow = slow->next, next = next->next;
                }
            }
        }
        return ret;
    }
    ```

- [235. Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

    - 利用BST的性质，root左子树的值均小于root->val，root右子树的值均大于root->val，可快速定位p和q的LCA最低公共祖先，时间复杂度$O(n)$

        - 递归写法

        ```cpp
        int findMinFibonacciNumbers(int k)
        {
            vector<int> fibonacci{1, 2};
            int last = 1;
            while (fibonacci[last] < k)
            {
                fibonacci.push_back(fibonacci[last - 1] + fibonacci[last]);
                last++;
            }
            vector<int> dp(k + 1, 0);
            for (auto i = 0; i <= k; i++)
            {
                dp[i] = i; // 全部用1
            }
            for (auto &&v : fibonacci)
            {
                for (auto i = v; i <= k; i++)
                {
                    dp[i] = min(dp[i - v] + 1, dp[i]);
                }
            }
            return dp.back();
        }
        ```

        - 迭代式写法

        ```cpp
		TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q)
		{
			TreeNode *ret = nullptr;
			if (p->val > q->val)
			{
				swap(p, q);
			}
			bool notfound = true;
			while (notfound && root)
			{
				if (root->val == p->val || root->val == q->val || (q->val > root->val && p->val < root->val))
				{
					ret = root;
					notfound = false;
				}
				root = p->val > root->val ? root->right : root->left;
			}
			return ret;
		}
        ```


    - 按照一般的二叉树求LCA问题，通过DFS后序遍历分别求出从root到p和q的两条路径，两条路径的最后一个相同节点即为LCA节点，时间复杂度$O(n)$

    ```cpp
	class Solution
	{
	private:
		TreeNode *ret = nullptr;
		bool helper(TreeNode *root, TreeNode *p, TreeNode *q)
		{
			bool ret_root = false;
			if (root)
			{
				int left = helper(root->left, p, q) ? 1 : 0;
				int right = helper(root->right, p, q) ? 1 : 0;
				int mid = (root == p || root == q) ? 1 : 0;
				if (left + right + mid >= 2)
				{
					ret = root;
				}
				ret_root = mid + left + right > 0;
			}
			return ret_root;
		}

	public:
		TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q)
		{
			helper(root, p, q);
			return ret;
		}
	};
    ```

- [237](https://leetcode.com/problems/delete-node-in-a-linked-list/)

    给定链表中某个节点，删除该节点，难点在于该节点的前继节点未知。如果该节点有后继，则用后继节点来代替该节点，如果没有后继，则该节点指向空即可。

    ```cpp
    void deleteNode(ListNode* node) {
        if(node->next){
            node->val=node->next->val;
            node->next=node->next->next;
        }else{
            node=nullptr;
        }
    }
    ```

- [238](https://leetcode.com/problems/product-of-array-except-self/)

    数组双向遍历的典型应用

    ```cpp
    vector<int> productExceptSelf(vector<int> &nums)
    {
        int const n = nums.size();
        vector<int> ret(n, 1);
        for (int i = 1; i < n; i++)
        {
            ret[i] = ret[i - 1] * nums[i - 1];
        }
        int base = 1;
        for (int i = n - 1; i > 0; i--)
        {
            ret[i] *= base;
            base *= nums[i];
        }
        ret[0] *= base;
        return ret;
    }
    ```

- [239](https://leetcode.com/problems/sliding-window-maximum/)

    - 根据求最大值自然想到优先队列，在优先队列的每个节点存储数字和下标以确保及时删除左侧滑出窗口的部分，时间复杂度$O(nlog(n))$，空间复杂度$O(k)$

    ```cpp
	vector<int> maxSlidingWindow(vector<int> &nums, int k)
	{
		const int n = nums.size();
		priority_queue<pair<int, int>> qe; // 默认为大顶堆
		vector<int> ret;
		// 首先将前k-1个元素放入优先队列
		for (int i = 0; i < k - 1; i++)
		{
			qe.push(make_pair(nums[i], i));
		}
		for (int i = k - 1; i < n; i++)
		{
			while (!qe.empty() && qe.top().second + k <= i)
			{
				qe.pop();
			}
			qe.push(make_pair(nums[i], i));
			ret.emplace_back(qe.top().first);
		}
		return ret;
	}
    ```

    - 给定数组nums和窗口大小k，求数组在窗口滑动过程中的最大值，这里主要是双端队列的使用，即以一个双端队列来实现一个递减的单调队列，时间复杂度与空间复杂度均为$O(n)$。

    ```cpp
	vector<int> maxSlidingWindow(vector<int> &nums, int k)
	{
		const int n = nums.size();
		vector<int> ret;
		deque<int> dq; // 双端队列
		for (int i = 0; i < n; i++)
		{
			// 保证单调递减的队列，从而保证队首下标对应的元素一定是滑动窗口内最大的
			while (!dq.empty() && nums[dq.back()] <= nums[i])
			{
				dq.pop_back();
			}
			dq.push_back(i);
			// 从队首扔掉超出滑动窗口左端的部分
			while (!dq.empty() && dq.front() + k <= i)
			{
				dq.pop_front();
			}
			if (i >= k - 1)
			{
				// 每一个滑动窗口放入当前最大值
				ret.emplace_back(nums[dq.front()]);
			}
		}
		return ret;
	}
    ```

- [240](https://leetcode.com/problems/search-a-2d-matrix-ii/)

    和[74](https://leetcode.com/problems/search-a-2d-matrix/)不同的是，本题限定矩阵中每一行为升序、每一列为升序
    
    - 以中心点将矩阵分为四个部分，中心点值大于target时，目标区间缩小为左上、左下、右上三个子区间，中心点值小于target时，目标区间定位到右上、右下、左下三个子区间，即每次比较可以排除四分之一区域，时间复杂度$O(log_{\frac{4}{3}}{(m*n)})$，但是递归写法效率较低

    ```cpp
    bool searchMatrix(vector<vector<int>> &matrix, int target, int i, int j, int m, int n)
    {
        bool ret = false;
        if (!(i > m || j > n))
        {
            // to ensure the matrix is not empty
            int mid_i = i + ((m - i) >> 1), mid_j = j + ((n - j) >> 1);
            if (matrix[mid_i][mid_j] == target)
            {
                ret = true;
            }
            else if (matrix[mid_i][mid_j] > target)
            {
                ret = searchMatrix(matrix, target, i, mid_j, mid_i - 1, n) || searchMatrix(matrix, target, i, j, m, mid_j - 1);
            }
            else
            {
                ret = searchMatrix(matrix, target, i, mid_j + 1, mid_i, n) || searchMatrix(matrix, target, mid_i + 1, j, m, n);
                // right || bottom
            }
        }
        return ret;
    }
    bool searchMatrix(vector<vector<int>> &matrix, int target)
    {
        bool ret = false;
        if (matrix.size() > 0 && matrix[0].size() > 0)
        {

            ret = searchMatrix(matrix, target, 0, 0, matrix.size() - 1, matrix[0].size() - 1);
        }
        return ret;
    }
    ```

    - 从左下向右上方向搜索，或者反向从右上到左下搜索，时间复杂度$O(m+n)$，迭代式写法效率较高

    ```cpp
    bool searchMatrix(vector<vector<int>> &matrix, int target)
    {
        bool ret = false;
        if (matrix.size() > 0 && matrix[0].size() > 0)
        {
            int m = matrix.size(), n = matrix[0].size();
            int i = m - 1, j = 0;
            while (i >= 0 && j < n)
            {
                if (target == matrix[i][j])
                {
                    ret = true;
                    break;
                }
                else if (target < matrix[i][j])
                {
                    i--;
                }
                else
                {
                    j++;
                }
            }
        }
        return ret;
    }
    ```

- [258](https://leetcode.com/problems/add-digits/)
    
    对一个数字求各个数位的和，递归直到这个数是个位数

    - 循环或者递归的方法

    ```cpp
    int addDigitals(int n)
    {
        while (n > 9)
        {
            int base = n;
            n = 0;
            while (base)
            {
                n += base % 10;
                base /= 10;
            }
        }
        return n;
    }
    ```

    - O(1)方法，可以[数学证明](https://leetcode.com/problems/add-digits/discuss/241551/Explanation-of-O(1)-Solution-(modular-arithmetic))

    ```cpp
	return (n - 1) % 9 + 1
    ```

- [264](https://leetcode.com/problems/ugly-number-ii/)

    定义：素因子只有2,3,5的数正整数成为ugly数，1是特殊的ugly数。

    问题：寻找第n个ugly数

    方法：
    - 暴力：从1开始逐个检查所有正整数序列是否为ugly数直到统计量count达到n
    - dynamic program：除1外，下一个ugly数必然是2,3,5的倍数(倍率分别从1开始，每使用一次倍率增长一次)中较小的一个

- [268](https://leetcode.com/problems/missing-number/)
    - Gauss formual
    
    从0到n的和是$S_0=\frac{n(n+1)}{2}$，missing一个数x以后的和是$S=\frac{n(n+1)}{2}-x$，则丢失的数是$x=S_0-S$。

    - bit XOR

    下标是0到n-1，补充一个n作为初始值，然后这些数字是0-n且missing一个数，相当于从0-n除了missing的这个数只出现一次之外其他数字都出现了两次，因此可以用XOR操作找到这个只出现了一次的数即可。

- [274. H-Index](https://leetcode.com/problems/h-index/)

    给定一个科学家所有论文的引用次数，求其H指数
    
    - 先对其引用次数排序（快排），然后当H小于等于当前引用次数时H持续自增，时间复杂度$O(nlog(n))$

    ```cpp
    int hIndex(vector<int> &citations)
    {
        sort(citations.begin(), citations.end());
        int h = 0;
        for (int i = citations.size() - 1; i >= 0; i--)
        {
            if (h + 1 <= citations[i])
            {
                h++;
            }
        }
        return h;
    }
    ```

    - 利用桶排序实现，two pass the citations，时间复杂度$O(n)$

    ```cpp
    int hIndex(vector<int> &citations)
    {
        const int count = citations.size(); // number of papers
        vector<int> buckets(count + 1, 0);
        for (auto &&v : citations)
        {
            if (v > count)
            {
                buckets[count]++;
            }
            else
            {
                buckets[v]++;
            }
        }
        int ans = 0, h_papers = 0;
        for (int i = count; i >= 0; i--)
        {
            h_papers += buckets[i];
            if (h_papers >= i)
            {
                ans = i;
                break;
            }
        }
        return ans;
    }
    ```

- [275. H-Index II](https://leetcode.com/problems/h-index-ii/)

    与[274. H-Index](https://leetcode.com/problems/h-index/)相比本题的citations数组保证升序，但是需要在对数时间复杂度呢解决求H指数问题

    这里对数时间复杂度且为排序数组，则必然要用到二分法，则可以想到用right标记所有满足H指数的定义中最小的下标值，即$min_{right}{(citations[right] \ge citations.size() - right）}$

    ```cpp
    int hIndex(vector<int> &citations)
    {
        int size = citations.size();
        int left = 0, right = size - 1;
        while (left < right)
        {
            int mid = left + ((right - left) >> 1);
            if (citations[mid] >= size - mid)
            {
                right = mid;
            }
            else
            {
                left = mid + 1;
            }
        }
        int h_index = 0;
        if (right >= 0 && citations[right] >= size - right)
        {
            h_index = size - right;
        }
        return h_index;
    }
    ```

- [278](https://leetcode.com/problems/first-bad-version/)

    在形如$(1,2,3,4,5,...,n,n+1,n+1,n+1,n+1,n+1,n+1)$这样的数组中中寻找第一个$n+1$的下标位置，这是二叉搜索的另一种形式，即每次命中$n+1$的右侧都是$n+1$，而没有命中的左侧都不是。相似的题目还有寻找有重复的排序数组中某个元素出现的下标区间[leetcode 34](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)等。

    ```cpp
    int firstBadVersion(int n) {
        unsigned int lo=1,hi=n;
        while(lo<hi){
            unsigned int mid=(lo+hi)>>1;
            if(isBadVersion(mid)){
                hi=mid;
            }else{
                lo=mid+1;
            }
        }
        return lo;
    }
    ```

- [279. Perfect Squares](https://leetcode.com/problems/perfect-squares/)

	- BFS搜索

	```cpp
	int numSquares(int n)
	{
		int ret = 0;
		unordered_set<int> bfs;
		bfs.insert(n);
		while (true)
		{
			unordered_set<int> cur;
			for (auto &&v : bfs)
			{
				if (v == 0)
				{
					return ret;
				}
				else
				{
					int q = 1;
					while (q * q <= v)
					{
						cur.insert(v - q * q);
						q++;
					}
				}
			}
			bfs = cur, ret++;
		}
		return ret;
	}
	```

	- 动态规划，时间复杂度$O(n*\sqrt{n})$

	```cpp
	int numSquares(int n)
	{
		vector<int> dp(n + 1, n);
		dp[0] = 0;
		for (int v = 1; v <= n; v++)
		{
			int max_q = floor(sqrt(v));
			for (int q = 1; q <= max_q; q++)
			{
				dp[v] = min(dp[v], 1 + dp[v - q * q]);
			}
		}
		return dp.back();
	}
	```

	- 使用static关键字，使得动态规划的结果在多个case之间复用

	```cpp
	int numSquares(int n)
	{
		static vector<int> dp{0};
		for (int v = dp.size(); v <= n; v++)
		{
			int cnt = numeric_limits<int>::max();
			for (int q = 1; q * q <= v; q++)
			{
				cnt = min(cnt, 1 + dp[v - q * q]);
			}
			dp.push_back(cnt);
		}
		return dp[n];
	}
	```

- [283](https://leetcode.com/problems/move-zeroes/)

    将一个数列中的0元素全部移动到数列尾部，O(1)空间复杂度和O(n)时间复杂度，保持原数列的稳定性

    ```cpp
    void moveZeroes(vector<int> &nums)
    {
        // int count = nums.size(), i = 0, k = 0;
        // for (int i = 0; i < count; i++)
        // {
        // 	if (nums[i])
        // 	{
        // 		nums[k++] = nums[i];
        // 	}
        // }
        // for (; k < count; k++)
        // {
        // 	nums[k++] = 0;
        // }

        // optimization
        int count = nums.size(), k = 0, i = 0;
        for (; i < count; i++)
        {
            if (nums[i])
            {
                swap(nums[i], nums[k++]);
            }
        }
    }
    ```

- [287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)

    - 线性扫描，使用bool数组标记出现过得所有数，时间复杂度$O(n)$、空间复杂度$O(n)$

    ```cpp
    int findDuplicate(vector<int> &nums)
    {
        int ret = -1;
        vector<int> flag(nums.size() + 1, 0);
        for (auto &&v : nums)
        {
            if (flag[v] == 1)
            {
                ret = v;
                break;
            }
            else
            {
                flag[v] = 1;
            }
        }
        return ret;
    }
    ```

    - 更改数组，以下标作为该数出现与否的标记，时间复杂度$O(n)$，空间复杂度$O(1)$

    ```cpp
    int findDuplicate(vector<int> &nums)
    {
        int ret = -1;
        for (int i = 0; i < nums.size(); i++)
        {
            int index = abs(nums[i]) - 1;
            if (nums[index] > 0)
            {
                nums[index] = -nums[index];
            }
            else
            {
                ret = index + 1;
                break;
            }
        }
        return ret;
    }
    ```

    - 按照题目要求，原数组只读不可写，空间复杂度$O(1)$，时间复杂度$O(n)$，Floyd cycle detection algorithm[wikipad](https://en.wikipedia.org/wiki/Cycle_detection)，与题目[142. Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)方法相同

    ```cpp
    int findDuplicate(vector<int> &nums)
    {
        int slow = nums[0], fast = nums[0];
        do
        {
            slow = nums[slow], fast = nums[nums[fast]];
        } while (slow != fast);
        int ret = nums[0];
        while (ret != slow)
        {
            slow = nums[slow], ret = nums[ret];
        }
        return ret;
    }
    ```

- [289. Game of Life](https://leetcode.com/problems/game-of-life/)

    - 仿真模拟，统计每个cell周边的live数，时间复杂度$O(m*n)$，空间复杂度$O(m*n)$

    ```cpp
    void gameOfLife(vector<vector<int>> &board)
    {
        if (board.size() > 0 && board[0].size() > 0)
        {
            int m = board.size(), n = board[0].size();
            vector<vector<int>> auxiliary(m, vector<int>(n, 0));
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    auxiliary[i][j] = board[i][j];
                }
            }
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    int live = -auxiliary[i][j]; // 下面的9宫格求和会计算中心点自己，提前除去
                    for (int r = -1; r <= 1; r++)
                    {
                        for (int c = -1; c <= 1; c++)
                        {
                            int a = i + r, b = j + c;
                            if (a >= 0 && a < m && b >= 0 && b < n)
                            {
                                live += auxiliary[a][b];
                            }
                        }
                    }
                    if (board[i][j] == 1)
                    {
                        board[i][j] = (live == 2 || live == 3) ? 1 : 0;
                    }
                    else if (board[i][j] == 0 && live == 3)
                    {
                        board[i][j] = 1;
                    }
                }
            }
        }
    }
    ```

    - 用值的正负状态来表示live和dead，实现inplace的update，时间复杂度$O(m*n)$，空间复杂度$O(1)$

    ```cpp
    void gameOfLife(vector<vector<int>> &board)
    {
        if (board.size() > 0 && board[0].size() > 0)
        {
            int m = board.size(), n = board[0].size();
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    int live = -board[i][j]; // 下面的9宫格求和会计算中心点自己，提前出去
                    for (int r = -1; r <= 1; r++)
                    {
                        for (int c = -1; c <= 1; c++)
                        {
                            int a = i + r, b = j + c;
                            if (a >= 0 && a < m && b >= 0 && b < n)
                            {
                                live += abs(board[a][b]) == 1 ? 1 : 0;
                            }
                        }
                    }
                    if (board[i][j] == 1)
                    {
                        board[i][j] = (live == 2 || live == 3) ? 1 : -1;
                    }
                    else if (board[i][j] == 0 && live == 3)
                    {
                        board[i][j] = 2;
                    }
                }
            }
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    board[i][j] = board[i][j] > 0 ? 1 : 0;
                }
            }
        }
    }
    ```

- [297](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)

    序列化或者反序列化一颗树

    ```cpp
    class Codec {
    public:
        // Encodes a tree to a single string.
        string serialize(TreeNode *root)
        {
            vector<TreeNode *> serialized, current_level, next_level;
            if (root)
            {
                // serialize a tree
                current_level.push_back(root);
                while (!current_level.empty())
                {
                    // add current level to serialized
                    for (int i = 0; i < current_level.size(); i++)
                    {
                        serialized.push_back(current_level[i]);
                        if (current_level[i])
                        {
                            next_level.push_back(current_level[i]->left);
                            next_level.push_back(current_level[i]->right);
                        }
                    }
                    current_level = next_level;
                    next_level.clear();
                }
                // remove nullptr in the end
                while (!current_level.empty())
                {
                    if (current_level.back() == nullptr)
                    {
                        current_level.pop_back();
                    }
                    else
                    {
                        break;
                    }
                }
            }
            // convert serialized to string
            string ans;
            for (auto &&node : serialized)
            {
                if (node)
                {
                    ans += to_string(node->val);
                }
                else
                {
                    ans += "NULL";
                }
                ans.push_back(',');
            }
            if (ans.length() > 0)
            {
                ans.pop_back();
            }
            ans = '[' + ans + ']';
            return ans;
        }
        // Decodes your encoded data to tree.
        TreeNode *deserialize(string data)
        {
            TreeNode *root = nullptr;
            data = data.substr(1, data.length() - 2);
            if (data.length() > 0)
            {
                stringstream ss;
                string item;
                ss.str(data);
                getline(ss, item, ',');
                root = new TreeNode(stoi(item));
                queue<TreeNode *> qe;
                qe.push(root);
                while (true)
                {
                    TreeNode *cur_node = qe.front();
                    qe.pop();
                    if (!getline(ss, item, ','))
                    {
                        break; // end of the string
                    }
                    if (item.compare("NULL") != 0)
                    {
                        TreeNode *left = new TreeNode(stoi(item));
                        cur_node->left = left;
                        qe.push(left);
                    }
                    if (!getline(ss, item, ','))
                    {
                        break; // end of the string
                    }
                    if (item.compare("NULL") != 0)
                    {
                        TreeNode *right = new TreeNode(stoi(item));
                        cur_node->right = right;
                        qe.push(right);
                    }
                }
            }
            return root;
        }

    };

    // test case
    // [1,2,3,null,null,4,5]
    // [3,5,1,6,2,0,8,null,null,7,4,null,9,null,null,null,8,5]
    // [3,5,1,6,2,0,8,null,null,7,4,null,9,null,8,5]

    // Your Codec object will be instantiated and called as such:
    // Codec codec;
    // codec.deserialize(codec.serialize(root));
    ```

- [300](https://leetcode.com/problems/longest-increasing-subsequence/)

    求给定无序顺序的最长升序子序列

    - 思路一，dynamic plan，时间复杂度$O(n^2)$

    ```cpp
    int lengthOfLIS(vector<int> &nums)
    {
        const int count = nums.size();
        vector<int> length(count, 1);
        int ans = 0;
        for (int i = 0; i < count; i++)
        {
            int j = 0;
            while (j < i)
            {
                if (nums[j] < nums[i])
                {
                    length[i] = max(length[j] + 1, length[i]);
                }
                j++;
            }
            ans = max(ans, length[i]);
        }
        return ans;
    }
    ```

    - 思路二，dynamic + binary search，时间复杂度$O(nlog(n))$

    ```cpp
    int lengthOfLIS(vector<int> &nums)
    {
        const int count = nums.size();
        int ans = 0;
        if (count < 2)
        {
            ans = count;
        }
        else
        {
            vector<int> longestIncreasingSeries;
            longestIncreasingSeries.push_back(nums[0]);
            for (int i = 1; i < count; i++)
            {
                if (nums[i] > longestIncreasingSeries.back())
                {
                    longestIncreasingSeries.push_back(nums[i]);
                }
                int left = 0, right = longestIncreasingSeries.size() - 1;
                // 在当前 longIncreasingSeries 中用 nums[i] 替换比其大的最小值
                while (left < right)
                {
                    int mid = left + ((right - left) >> 1);
                    if (nums[i] > longestIncreasingSeries[mid])
                    {
                        left = mid + 1;
                    }
                    else
                    {
                        right = mid;
                    }
                }
                longestIncreasingSeries[left] = nums[i];
            }
            ans = longestIncreasingSeries.size();
        }
        return ans;
    }
    ```