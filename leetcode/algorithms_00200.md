# 101-200

- [101](https://leetcode.com/problems/symmetric-tree/)

    分析一棵树对称问题的本质，是：

    - 空树是对称的
    - 没有左右子树的叶节点是对称的
    - 非空的非叶子节点对称的条件有：
        - 左右子树同时存在且左右子树的值相同（对称）
        - 左子树的左子树和右子树的右子树递归对称，左子树的右子树和右子树的左子树递归对称

- [104](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

    计算一颗二叉树的最大深度，这是典型的使用递归解决tree类问题的模板，按照top-down和bottom-up两种思路，参见[article](https://leetcode.com/explore/learn/card/data-structure-tree/17/solve-problems-recursively/534/)。
    - top-down

    ```cpp
    int ans = 0;
    void updateDepth(TreeNode *root, int depth){
        if (root){
            ans = max(ans, depth);
            updateDepth(root->left, depth + 1);
            updateDepth(root->right, depth + 1);
        }
    }
    int maxDepth(TreeNode *root){
        if (root){
            updateDepth(root, 1);
        }
        return ans;
    }
    ```

    - bottom-up

    ```cpp
    int maxDepth(TreeNode* root) {
        if(!root){
            return 0;
        }else{
            int left=maxDepth(root->left);
            int right=maxDepth(root->right);
            return 1+max(left,right);
        }
    }
    ```

- [110](https://leetcode.com/problems/balanced-binary-tree/)

    判断二叉树是否是平衡二叉树（任何节点左右子树的高度差小于等于1），在递归求二叉树最大深度的过程中维护一个全局变量balanced，随时比较任意节点的左右子树高度差即可。

- [114. Flatten Binary Tree to Linked List](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)

    将二叉树转换为左子树永远为空的单向链表

    - 先序遍历将所有节点存储到一个数组中，然后顺序连接，时间复杂度$O(n)$，空间复杂度$O(n)$

    ```cpp
	void flatten(TreeNode *root)
	{
		vector<TreeNode *> nodes;
		TreeNode *cur = root;
		stack<TreeNode *> st;
		while (cur || !st.empty())
		{
			if (cur)
			{
				nodes.push_back(cur);
				if (cur->right)
				{
					st.push(cur->right);
				}
				cur = cur->left;
			}
			else
			{
				cur = st.top();
				st.pop();
			}
		}
		cur = new TreeNode(0);
		root = cur;
		for (auto node : nodes)
		{
			cur->right = node;
			node->left = nullptr;
			cur = cur->right;
		}
		root = root->right;
	}
    ```

    - 在先序遍历的过程中直接转换二叉树节点间的链接关系，将空间复杂度优化到$O(1)$

    ```cpp
	void flatten(TreeNode *root)
	{
		TreeNode *cur = root;
		stack<TreeNode *> st;
		while (cur)
		{
			if (cur->left)
			{
				if (cur->right)
				{
					st.push(cur->right);
				}
				cur->right = cur->left;
				cur->left = nullptr;
				cur = cur->right;
			}
			else
			{
				if (cur->right)
				{
					cur = cur->right;
				}
				else if (!st.empty())
				{
					cur->right = st.top();
					st.pop();
					cur = cur->right;
				}
				else
				{
					cur = nullptr;
				}
			}
		}
	}
    ```

    - 递归写法

    ```cpp
    TreeNode* flattenHelper(TreeNode *root)
    {
        // root must not be NULL, flatten root and return its right most not NULL node
        if(!root->left && !root->right){
            return root;
        }else if(!root->left && root->right){
            return flattenHelper(root->right);
        }else if(root->left && !root->right){
            root->right = root->left;
            root->left = nullptr;
            return flattenHelper(root->right);
        }else{
            TreeNode* temp = flattenHelper(root->left);
            temp->right = root->right;
            root->right = root->left;
            root->left = nullptr;
            return flattenHelper(temp->right);
        }
    }
    void flatten(TreeNode* root){
        // O(N), one pass for all nodes
        if(root){
            TreeNode *temp = root;
            temp = flattenHelper(temp);
        }
    }
    ```

- [116](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)

    更一般化的问题是如题[117](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/)所示的条件，给定一个二叉树，将每一个节点的next指针指向他的同深度的右侧兄弟节点，简单BFS(Breadth-First-Search)，即层序遍历然后将同层的节点扫描一遍将每个节点的next指针指向同层下一个节点即可。

    ```cpp
    Node *connect(Node *root)
    {
        if(root){
            vector<Node *> level{root}, next_level;
            while (!level.empty()){
                for (auto &&node : level){
                    if(node->left){
                        next_level.push_back(node->left);
                    }
                    if(node->right){
                        next_level.push_back(node->right);
                    }
                }
                int i = 0, count = next_level.size() - 1;
                while(i < count){
                    next_level[i]->next = next_level[i + 1];
                    i++;
                }
                level = next_level;
                next_level.clear();
            }
        }
        return root;
    }
    ```

    当将给定二叉树限定为本题所示的Perfect Binary Tree的时候，可以用递归的方式来完成而无需BFS层序遍历的庞大空间开销，需要注意递归到子节点时需要利用父节点的next指针信息。

    ```cpp
    Node *connect(Node *root)
    {
        if(root && root->left){
            root->left->next = root->right;
            if(root->next){
                root->right->next = root->next->left;
            }
            connect(root->left);
            connect(root->right);
        }
        return root;
    }
    ```

- [117. 填充每个节点的下一个右侧节点指针 II](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/)

    在$O(n)$时间和$O(1)$空间下的非递归实现

    ```cpp
    class Solution
    {
    private:
        void process(Node *&last, Node *&p, Node *&nextStart)
        {
            if (last)
            {
                // 当前last非空，则last的next指针指向p
                last->next = p;
            }
            if (!nextStart)
            {
                nextStart = p;
            }
            last = p; // 当前p为新的本层最后一个节点
        }

    public:
        Node *connect(Node *root)
        {
            if (root)
            {
                Node *start = root; // 根节点只有一个节点，从该节点开始遍历
                while (start)
                {
                    /**
                     * 维护当前的末尾last和下一层可能的最左端开始节点nextStart两个指针对
                     * 然后遍历当前层通过next指针形成的链表
                     */
                    Node *last = nullptr, *nextStart = nullptr;
                    for (Node *p = start; p != nullptr; p = p->next)
                    {
                        // 遍历当前层，为下一层标记next指针
                        if (p->left)
                        {
                            process(last, p->left, nextStart);
                        }
                        if (p->right)
                        {
                            process(last, p->right, nextStart);
                        }
                    }
                    start = nextStart;
                }
            }
            return root;
        }
    };
    ```

- [122](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

    不限交易次数的股票交易

    - 我的方法

    ```cpp
    int maxProfit(vector<int>& prices) {
        int ans=0,cur_value=0;
        for (int i = prices.size()-1; i >= 0; i--)
        {
            if(cur_value<=prices[i]){
                cur_value=prices[i];
            }else{
                while(i>0 && prices[i-1]<=prices[i]){
                    i--;
                }
                ans+=cur_value-prices[i];
                cur_value=0;
            }
        }
        return ans;
    }
    ```

    - Solution方法

    其实就是(c-b)+(b-a)=c-a的基本原理，greedy algorithm，时间复杂度$O(n)$

    ```cpp
    int maxProfit(vector<int> &prices)
    {
        int ans = 0;
        for (int i = 1; i < prices.size(); i++)
        {
            ans += max(prices[i] - prices[i - 1], 0);
        }
        return ans;
    }
    ```

    - dynamic plan，时间复杂度$O(n)$

    ```cpp
	int maxProfit(vector<int> &prices)
	{
		/**
		 * dynamic plan algorithm
		*/
		int n = prices.size(), inf = numeric_limits<int>::max();
		vector<vector<int>> dp(n + 1, vector<int>(2));
		dp[0][0] = -inf;
		dp[0][1] = 0;
		for (auto i = 0; i < n; i++)
		{
			dp[i + 1][0] = max(dp[i][0], dp[i][1] - prices[i]);
			dp[i + 1][1] = max(dp[i][1], dp[i][0] + prices[i]);
		}
		return dp[n][1];
	}
    ```

- [123. Best Time to Buy and Sell Stock III](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/)

    - 动态规划，时间复杂度$O(n)$

    ```cpp
	int maxProfit(vector<int> &prices)
	{
		/**
		 * dynamic plan algorithm
		*/
		int n = prices.size(), inf = numeric_limits<int>::max();
		vector<vector<int>> dp(n + 1, vector<int>(4));
		dp[0][0] = dp[0][2] = -inf; // 初始为买入状态，收益为负无穷
		dp[0][1] = dp[0][3] = 0;	// 初始为出售（不再持有）状态，收益为0
		for (auto i = 0; i < n; i++)
		{
			dp[i + 1][0] = max(dp[i][0], -prices[i]);
			dp[i + 1][1] = max(dp[i][1], dp[i][0] + prices[i]);
			dp[i + 1][2] = max(dp[i][2], dp[i][1] - prices[i]);
			dp[i + 1][3] = max(dp[i][3], dp[i][2] + prices[i]);
		}
		return max(dp[n][1], dp[n][3]); // 只有最终为售出状态才有可能收益最大
	}
	```
	
	- 空间复杂度为$O(1)$优化

	```cpp
	int maxProfit(vector<int> &prices)
	{
		/**
		 * dynamic plan algorithm
		 *********************************
		 * 空间复杂度从O(n)优化到O(1)
		 * 
		 * 初始化条件：
		 * 		初始状态为get时收益为负无穷，初始状态为清仓售空时收益为0
		 * 返回值：
		 * 		只有最终状态为售出时才有可能获得最大收益
		*/
		int n = prices.size(), inf = numeric_limits<int>::max();
		int get1 = -inf, out1 = 0, get2 = -inf, out2 = 0;
		for (auto &&v : prices)
		{
			get1 = max(get1, -v);
			out1 = max(out1, get1 + v);
			get2 = max(get2, out1 - v);
			out2 = max(out2, get2 + v);
		}
		return max(out1, out2);
	}
	```

- [124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

    递归，时间复杂度$O(n)$，其中n是给定二叉树的所有节点数

    ```cpp
    class Solution
    {
    private:
        int ret;
        int PathSum(TreeNode *root)
        {
            int max_path_sum = 0;
            if (root)
            {
                int left = max(0, PathSum(root->left));
                int right = max(0, PathSum(root->right));
                ret = max(ret, left + right + root->val);
                max_path_sum = max(left, right) + root->val;
            }
            return max_path_sum;
        }

    public:
        int maxPathSum(TreeNode *root)
        {
            ret = numeric_limits<int>::min();
            PathSum(root);
            return (int)ret;
        }
    };
    ```

    - some testcases

    ```cpp
    [2,-1]
    [1,2,3]
    [-10,9,20,null,null,15,7]
    [5,4,8,11,null,13,4,7,2,null,null,null,1]
    [-2,-1]
    [-3]
    [0]
    [-3,0]
    ```

- [128. Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)

	在集合中查找一个数v的下一个连续值v+1是否存在，时间复杂度$O(n)$

	```cpp
	int longestConsecutive(vector<int> &nums)
	{
		unordered_set<int> nums_set;
		for (auto &&v : nums)
		{
			nums_set.insert(v);
		}
		int ret = 0, cur_length = 0;
		for (auto &v : nums_set)
		{
			if (nums_set.find(v - 1) == nums_set.end())
			{
				// 当 v-1在集合中的时候v必然被统计过了
				cur_length = 1;
				while (nums_set.find(v + cur_length) != nums_set.end())
				{
					cur_length++;
				}
				ret = max(ret, cur_length);
			}
		}
		return ret;
	}
	```

- [130. Surrounded Regions](https://leetcode.com/problems/surrounded-regions/)

    时间复杂度$O(m*n)$，其中$m*n$为矩阵中所有元素的个数，具体算法流程：
    - 遍历四个边界，寻找所有边界上的'O'，然后DFS该'O'及所有相连的'O'全部标记为'#'
    - 遍历board，所有节点中'O'标记为'X'，'#'改回'O'

    ```cpp
    void dfs_marker(vector<vector<char>> &board, int r, int c)
    {
        if (r >= 0 && c >= 0 && r < board.size() && c < board[0].size() && board[r][c] == 'O')
        {
            board[r][c] = '#';
            vector<int> directions{1, 0, -1, 0, 1};
            for (int k = 0; k < 4; k++)
            {
                dfs_marker(board, r + directions[k], c + directions[k + 1]);
            }
        }
    }
    void solve(vector<vector<char>> &board)
    {
        if (board.size() > 0 && board[0].size() > 0)
        {
            const int m = board.size(), n = board[0].size();
            for (int j = 0; j < n; j++)
            {
                if (board[0][j] == 'O')
                {
                    dfs_marker(board, 0, j);
                }
                if (board[m - 1][j] == 'O')
                {
                    dfs_marker(board, m - 1, j);
                }
            }
            for (int j = 0; j < m; j++)
            {
                if (board[j][0] == 'O')
                {
                    dfs_marker(board, j, 0);
                }
                if (board[j][n - 1] == 'O')
                {
                    dfs_marker(board, j, n - 1);
                }
            }
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    board[i][j] = (board[i][j] == '#') ? 'O' : 'X';
                }
            }
        }
    }
    ```

- [131. 分割回文串](https://leetcode-cn.com/problems/palindrome-partitioning/)

    基本思路：搜索回溯+动态规划预处理
    具体方法：
    - 使用搜索回溯的方法，假设已经处理完毕$s[0,i-1]$的部分并划分为合适的回文段存入ans中，则从字符$s_i$开始搜索到$s_j$的$s[i,j]$序列找到一个回文段，将该段加入ans中，直到$i$到达字符序列的最右端，得到一个可能的分割序列加入答案ret中
    - 在判断$s[i,j]$是否回文序列的过程中，会出现大量的重复计算，提前使用动态规划的思路预处理得到dp数组，$dp_{ij}$表示$s[i,j]$段是否为回文序列
    - 时间复杂度$O(n*2^n)$

    ```cpp
    class Solution
    {
    private:
        void dfs(vector<vector<string>> &ret, vector<string> &ans, vector<vector<bool>> &dp, const string &s, int start, const int n)
        {
            if (start == n)
            {
                // 搜索到了最后一个字符
                ret.emplace_back(ans);
                return;
            }
            for (int j = start; j < n; j++)
            {
                if (dp[start][j])
                {
                    ans.emplace_back(s.substr(start, j - start + 1));
                    dfs(ret, ans, dp, s, j + 1, n);
                    ans.pop_back();
                }
            }
        }

    public:
        vector<vector<string>> partition(string s)
        {
            const int n = s.length();
            vector<vector<string>> ret;							// 保存所有可能的分割结果
            vector<string> ans;									// 一种可能的分割结果
            vector <vector<bool>> dp(n, vector<bool>(n, true)); // 动态规划预处理s[i...j]是否回文
            for (int i = n - 1; i >= 0; i--)
            {
                for (int j = i + 1; j < n; j++)
                {
                    dp[i][j] = dp[i + 1][j - 1] && (s[i] == s[j]);
                }
            }
            dfs(ret, ans, dp, s, 0, n); // 从s的第一个字符开始搜索回溯
            return ret;
        }
    };
    ```

- [132. 分割回文串 II](https://leetcode-cn.com/problems/palindrome-partitioning-ii/submissions/)

    动态规划，时间复杂度$O(n^2)$
    
    - 动态规划预处理计算字符串s中任意两点间s[i,j]段子串是否为回文序列 
    - 动态规划计算将s[0,i]段划分为回文的子串需要的最少次数

    ```cpp
	int minCut(string s)
	{
		const int n = s.length();
		vector<int> ret(n, n - 1); // 最多分割n-1次的单个字符，全部回文
		vector<vector<bool>> dp(n, vector<bool>(n, true));
		for (int i = n - 1; i >= 0; i--)
		{
			for (int j = i + 1; j < n; j++)
			{
				dp[i][j] = dp[i + 1][j - 1] && (s[i] == s[j]);
			}
		}
		for (int i = 0; i < n; i++)
		{
			if (dp[0][i])
			{
				ret[i] = 0;
			}
			else
			{
				for (int j = 0; j < i; j++)
				{
					if (dp[j + 1][i])
					{
						ret[i] = min(ret[i], ret[j] + 1);
					}
				}
			}
		}
		return ret[n - 1];
	}
    ```

- [133. 克隆图](https://leetcode-cn.com/problems/clone-graph/)

    从给定的根节点开始，复制每个节点node后，bfs搜索复制所有与node连接的节点，在这个过程中国为一个hashmap存储已经复制过的节点，与已经复制过的节点则直接在node的neighbors中插入节点指针即可

    ```cpp
	Node *cloneGraph(Node *node)
	{
		if (node)
		{
			Node *ret = new Node(node->val);
			if (!node->neighbors.empty())
			{
				unordered_map<int, Node *> valToNode;
				valToNode[ret->val] = ret;
				queue<pair<Node *, Node *>> bfs{{make_pair(ret, node)}}; // copyed,old
				while (!bfs.empty())
				{
					Node *cur_copyed = bfs.front().first, *cur_old = bfs.front().second;
					bfs.pop();
					for (auto &next_node : cur_old->neighbors)
					{
						auto it = valToNode.find(next_node->val);
						if (it != valToNode.end())
						{
							cur_copyed->neighbors.emplace_back(it->second);
						}
						else
						{
							Node *temp = new Node(next_node->val);
							valToNode[next_node->val] = temp;
							cur_copyed->neighbors.emplace_back(temp);
							bfs.push(make_pair(temp, next_node));
						}
					}
				}
			}
			return ret;
		}
		return nullptr; // 空节点仍然返回空
	}
    ```

- [134. Gas Station](https://leetcode.com/problems/gas-station/)

    - 遍历每个station检测其是否可以到达终点，即从该station出发到达每一个站点时拥有的油料值cur($cur=cur+gas[i]-cost[i]$)都足够到达下一个站点($cur>=0$)，时间复杂度$O(n^2)$

    ```cpp
    int canCompleteCircuit(vector<int> &gas, vector<int> &cost)
	{
		int n = gas.size(), ret = -1;
		if (n > 0)
		{
			for (auto station = 0; station < n; station++)
			{
				int cur = 0;
				for (auto i = 0; cur >= 0 && i < n; i++)
				{
					int index = (i + station) % n;
					cur += (gas[index] - cost[index]);
				}
				if (cur >= 0)
				{
					ret = station;
					break;
				}
			}
		}
		return ret;
	}
    ```

    - 贪心原则，时间复杂度$O(n)$

    ```cpp
	int canCompleteCircuit(vector<int> &gas, vector<int> &cost)
	{
		int n = gas.size(), total_tank = 0, cur_tank = 0, start_station = 0;
		for (auto i = 0; i < n; i++)
		{
			total_tank += gas[i] - cost[i];
			cur_tank += gas[i] - cost[i];
			if (cur_tank < 0)
			{
				start_station = i + 1;
				/**
				 * 从start_station出发无法到达i+1(具体是从i无法到达i+1)，则从[start_staion,i]任一站点出发都无法到达i+1
				 * 则以i+1为起点重新开始
				*/
				cur_tank = 0;
			}
		}
		return total_tank >= 0 ? start_station : -1;
	}
    ```

- [136](https://leetcode.com/problems/single-number/)

    一个数组中只有一个数落单、其他数均成对出现，采用异或的按位操作一遍扫描找到那个落单的数。
    
    进一步[260](https://leetcode.com/problems/single-number-iii/)中寻找两个落单的数，一遍异或操作可以得到这两个落单数的异或结果，然后其他数按照这个结果二进制表示中第一个非零位是否为1分为两组异或一遍即可。
    
    还有[137](https://leetcode.com/problems/single-number-ii/)中除了一个落单的数字、其他数字均出现了三次，可以统计int的32位表示中每一位为1的个数，然后这个统计结果为不能被3整除的那些bit位为1的二进制表示结果即为所求的落单的那个数。

- [139. Word Break](https://leetcode.com/problems/word-break/)

	- 对wordDict中的每个单词，遍历在s中是否可以找到，找到之后在以该单词截断s，递归检查s的剩余部分，leetcode评测机$\color{red}{TLE}$

	```cpp
	class Solution
	{
		int findSubstring(string s, string word)
		{
			int ret = -1, s_length = s.length(), word_length = word.length();
			if (s_length >= word_length)
			{
				int count = s_length - word_length + 1;
				for (int i = 0; i < count; i++)
				{
					bool found = true;
					for (int j = 0; found && j < word_length; j++)
					{
						if (word[j] != s[i + j])
						{
							found = false;
						}
					}
					if (found)
					{
						ret = i;
						break;
					}
				}
			}
			return ret;
		}

	public:
		bool wordBreak(string s, vector<string> &wordDict)
		{
			if (s.empty())
			{
				return true;
			}
			for (auto word : wordDict)
			{
				int start = findSubstring(s, word);
				if (start != -1 && wordBreak(s.substr(0, start), wordDict) && wordBreak(s.substr(start + word.size()), wordDict))
				{
					return true;
				}
			}
			return false;
		}
	};
	```

    - 动态规划，dp[i]表示s中前i个字符可以有wordDict中的单词形成，时间复杂度$O(n^2)$，其中$n=s.length$

    ```cpp
	bool wordBreak(string s, vector<string> &wordDict)
	{
		int n = s.length(), max_word_length = 0;
		unordered_set<string> words;
		for (auto &&word : wordDict)
		{
			words.insert(word);
			max_word_length = max(max_word_length, (int)word.length());
		}
		vector<bool> dp(n + 1, false);
		/**
		 * 1. dp[i]表示字符串s.substr(0,i)是否可以有wordDict中的单词组成
		 * 		Runtime: 24 ms, faster than 54.32% of C++ online submissions for Word Break.
		 * 2. 优化点：在内层循环中j = max(0, i - max_word_length)，这样可以使得内层循环直接从wordDict中长度最长的单词处开始所有，减小内层循环的长度
		 * 		Runtime: 4 ms, faster than 99.02% of C++ online submissions for Word Break.
		*/
		dp[0] = true;
		for (auto i = 1; i <= n; i++)
		{
			for (auto j = max(0, i - max_word_length); !dp[i] && j < i; j++)
			{
				if (dp[j] && words.find(s.substr(j, i - j)) != words.end())
				{
					dp[i] = true;
				}
			}
		}
		return dp[n];
	}
    ```

- [141](https://leetcode.com/problems/linked-list-cycle/)

    检查单向链表中是否存在cycle，两种方法，O(n)时间复杂度
    - hastset 存储节点，不停的检查set中是否存在cur->next
    - 快慢双指针遍历，当快慢指针相同的时候出现cycle

    ```cpp
    bool hasCycle(ListNode *head)
    {
        // method 1, hashset
        // unordered_set<ListNode *> record;
        // bool ret = false;
        // while (head)
        // {
        // 	if (record.find(head) != record.end())
        // 	{
        // 		ret = true;
        // 		break;
        // 	}
        // 	else
        // 	{
        // 		record.insert(head);
        // 	}
        // 	head = head->next;
        // }
        // return ret;

        // method 2, two pointer (slower and faster)
        bool ret = false;
        if (head != NULL && head->next != NULL)
        {
            ret = true;
            ListNode *slow = head, *fast = head->next;
            while (slow != fast)
            {
                if (slow == NULL || fast == NULL)
                {
                    ret = false;
                    break;
                }
                else
                {
                    slow = slow->next;
                    fast = fast->next->next;
                }
            }
        }
        return ret;
    }
    ```

- [142. Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)

    在 141判断链表中是否有cycle的基础上找出cycle的入口entry pointer，[detail algorithm](https://leetcode.com/problems/linked-list-cycle-ii/discuss/44781/Concise-O(n)-solution-by-using-C%2B%2B-with-Detailed-Alogrithm-Description)。

- [143](https://leetcode.com/problems/reorder-list/)

    **链表、树、图等指针操作千千万万要注意空指针甚至是输入根节点为空的情**

    首先发现链表head的中点，将链表分为前后两段a和b，翻转后半段b，而后在a、b中交替取一个连接起来，时间复杂度$O(n)$

    ```cpp
    class Solution
    {
        ListNode *reverse_list(ListNode *head)
        {
            ListNode *pre = nullptr, *cur = head, *temp = nullptr;
            while (cur)
            {
                temp = cur->next;
                cur->next = pre;
                pre = cur;
                cur = temp;
            }
            return pre;
        }

    public:
        void reorderList(ListNode *head)
        {
            ListNode *slow = head, *fast = head;
            while (fast && fast->next)
            {
                slow = slow->next, fast = fast->next->next;
            }
            ListNode *auxiliary = new ListNode(0), *a = head, *b = reverse_list(slow);
            ListNode *cur = auxiliary;
            while (a != slow)
            {
                cur->next = a;
                a = a->next;
                cur = cur->next;
                if (b)
                {
                    cur->next = b;
                    b = b->next;
                    cur = cur->next;
                }
            }
            head = auxiliary->next;
        }
    };
    ```
    

- [144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

    二叉树的前序遍历

    - 递归写法

    ```cpp
    class Solution
    {
    private:
        void dfs(TreeNode *root, vector<int> &ret)
        {
            if (root)
            {
                ret.push_back(root->val);
                dfs(root->left, ret);
                dfs(root->right, ret);
            }
        }

    public:
        vector<int> preorderTraversal(TreeNode *root)
        {
            vector<int> ret;
            dfs(root, ret);
            return ret;
        }
    };
    ```

    - 迭代写法

    ```cpp
	vector<int> preorderTraversal(TreeNode *root)
	{
		vector<int> ret;
		if (root)
		{
			stack<TreeNode *> st;
			TreeNode *cur = root;
			while (cur || !st.empty())
			{
				if (cur)
				{
					ret.push_back(cur->val);
					st.push(cur->right);
					cur = cur->left;
				}
				else
				{
					cur = st.top();
					st.pop();
				}
			}
		}
		return ret;
	}
    ```

- [145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)

    二叉树的后序遍历

    - 递归写法

    ```cpp
    class Solution
    {
    private:
        void dfs(TreeNode *root, vector<int> &ret)
        {
            if (root)
            {
                dfs(root->left, ret);
                dfs(root->right, ret);
                ret.push_back(root->val);
            }
        }

    public:
        vector<int> postorderTraversal(TreeNode* root)
        {
            vector<int> ret;
            dfs(root, ret);
            return ret;
        }
    };
    ```

    - 迭代写法

    ```cpp
	vector<int> postorderTraversal(TreeNode *root)
	{
		vector<int> ret;
		if (root)
		{
			TreeNode *cur = root;
			stack<TreeNode *> st;
			while (cur || !st.empty())
			{
				// 首先找到未访问部分的最左下子节点
				while (cur)
				{
					st.push(cur);
					cur = cur->left ? cur->left : cur->right;
				}
				// 当前栈顶即为下一个需要访问的节点
				cur = st.top();
				st.pop();
				ret.push_back(cur->val);
				if (!st.empty() && cur == st.top()->left)
				{
					cur = st.top()->right;
				}
				else
				{
					cur = nullptr; // 强制弹栈回到上一层
				}
			}
		}
		return ret;
	}
    ```

- [146. LRU Cache](https://leetcode.com/problems/lru-cache/)

    - hashmap + double linked list，get与put操作时间复杂度均为$O(1)$

    ```cpp
    class LRUCache
    {
    private:
        struct DoubleListNode
        {
            /* data */
            int key, val;
            DoubleListNode *prev, *next;
            DoubleListNode(int _key, int _val) : key(_key), val(_val), prev(NULL), next(NULL) {}
        };
        unordered_map<int, DoubleListNode *> cache; // 存储key-node pointer对
        int cache_capacity, current_capacity;
        // 按照队列原则管理，head->next是最长时间未被使用的节点，每次刚刚cache到使用的节点/刚刚insert的节点移动至队尾tail->prev
        DoubleListNode *head, *tail;
        bool isEmptyCache()
        {
            return current_capacity == 0;
        }
        bool isFullCache()
        {
            return current_capacity == cache_capacity;
        }
        DoubleListNode *removeFromHead()
        {
            /*
            1、 head->next是最长时间未被使用的节点，capacity满时需要删除最长时间未被使用的节点head->next来说释放空间
            2、 成功删除头结点返回被删除的节点指针，否则返回nullptr
            */
            DoubleListNode *ret = nullptr;
            if (!isEmptyCache())
            {
                current_capacity--;
                ret = head->next;
                head->next = head->next->next;
                head->next->prev = head;
            }
            return ret;
        }
        DoubleListNode *addToTail(DoubleListNode *current_node)
        {
            /*
            1、 每次刚刚cache到使用的节点/刚刚insert的节点移动至队尾tail->prev，capacity满时需要删除
            2、 如果需要释放空间删除了某个节点，则返回被删除节点指针，否则返回nullptr
            */
            DoubleListNode *removed_node = nullptr;
            if (isFullCache())
            {
                removed_node = removeFromHead(); // cache满时抛弃最长时间未被使用节点，释放空间
            }
            DoubleListNode *temp = tail->prev;
            temp->next = current_node, current_node->prev = temp;
            current_node->next = tail, tail->prev = current_node;
            current_capacity++;
            return removed_node;
        }
        bool removeFromNode(DoubleListNode *current_node)
        {
            bool ret = false;
            if (current_node != nullptr && !isEmptyCache())
            {
                DoubleListNode *a = current_node->prev, *b = current_node->next;
                a->next = b, b->prev = a;
                current_capacity--;
                ret = true;
            }
            return ret;
        }

    public:
        LRUCache(int capacity)
        {
            cache_capacity = capacity, current_capacity = 0;
            head = new DoubleListNode(-1, -1), tail = new DoubleListNode(-1, -1);
            head->next = tail, tail->prev = head; // 构建双向链表
        }

        int get(int key)
        {
            int ret = -1;
            auto it = cache.find(key);
            if (it != cache.end())
            {
                // cache 命中
                removeFromNode(it->second);
                addToTail(it->second);
                ret = it->second->val;
            }
            return ret;
        }

        void put(int key, int value)
        {
            auto it = cache.find(key);
            DoubleListNode *current_node = nullptr;
            if (it != cache.end())
            {
                // key已经存在，重新set value
                current_node = it->second;
                current_node->val = value;
                // 然后将该节点放到末尾
                removeFromNode(current_node);
            }
            else
            {
                // 插入该key-value对
                current_node = new DoubleListNode(key, value);
                cache[key] = current_node;
            }
            auto *removed_node = addToTail(current_node);
            if (removed_node)
            {
                cache.erase(removed_node->key);
            }
        }
    };

    /**
    * Your LRUCache object will be instantiated and called as such:
    * LRUCache* obj = new LRUCache(capacity);
    * int param_1 = obj->get(key);
    * obj->put(key,value);
    */
    ```

- [147. Insertion Sort List](https://leetcode.com/problems/insertion-sort-list/)

    链表存储的数组，插入排序操作

    ```cpp
    ListNode *insertionSortList(ListNode *head)
    {
        ListNode *new_head = new ListNode(0), *root = new_head, *cur = head, *pre_of_cur = new_head;
        new_head->next = head;
        while (cur)
        {
            root = new_head;
            while (root->next != cur && root->next->val < cur->val)
            {
                // 为cur寻找合适的位置
                root = root->next;
            }
            if (root->next != cur)
            {
                // cur不在合适的位置，需要移动
                pre_of_cur->next = cur->next, cur->next = root->next, root->next = cur;
                cur = pre_of_cur->next; // 将cur移动到下一个未排序节点
            }
            else
            {
                // 当前cur已经在正确位置，直接将cur移动到下一个未排序位置
                pre_of_cur = cur, cur = cur->next;
            }
        }
        return new_head->next;
    }
    ```

- [148. Sort List](https://leetcode.com/problems/sort-list/)

    归并排序（merge sort），时间复杂度$O(nlog(n))$

    ```cpp
    ListNode *merge(ListNode *a, ListNode *b)
    {
        ListNode *auxiliary_head = new ListNode(0), *cur = auxiliary_head;
        while (a && b)
        {
            if (a->val < b->val)
            {
                cur->next = a, cur = cur->next, a = a->next;
            }
            else
            {
                cur->next = b, cur = cur->next, b = b->next;
            }
        }
        while (a)
        {
            cur->next = a, cur = cur->next, a = a->next;
        }
        while (b)
        {
            cur->next = b, cur = cur->next, b = b->next;
        }
        return auxiliary_head->next;
    }
    ListNode *sortList(ListNode *head)
    {
        ListNode *ret = nullptr;
        if (!head || !head->next)
        {
            ret = head;
        }
        else
        {
            ListNode *slow = head, *fast = head;
            while (fast->next && fast->next->next)
            {
                slow = slow->next, fast = fast->next->next;
            }
            fast = slow->next;
            slow->next = nullptr;
            head = sortList(head), fast = sortList(fast);
            ret = merge(head, fast);
        }
        return ret;
    }
    ```

- [152. Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)

	**当nums中的元素均为正数的时候，本题可以通过取对数转化为[53. Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)求连续子数组的最大和问题**

    - dynamic plan，时间复杂度$O(n)，空间复杂度$O(n)$

	```cpp
	int maxProduct(vector<int> &nums)
	{
		/**
		 * 朴素的DP解法，给定dp_min[i]和dp_max[i]表示以当前nums[i]结尾的subarray中乘积最小值和最大值
		 * 状态转移方程：
		 * 		dp_min[i] = min(dp_min[i-1]*nums[i],dp_max[i-1]*nums[i],nums[i])
		 * 		dp_max[i] = max(dp_min[i-1]*nums[i],dp_max[i-1]*nums[i],nums[i])
		 * 时间复杂度：O(n)
		 * 空间复杂度：O(n)
		*/
		int ret = 0;
		if (nums.size() > 0)
		{
			int n = nums.size();
			vector<int> dp_min(n), dp_max(n);
			dp_min[0] = nums[0], dp_max[0] = nums[0], ret = nums[0];
			for (int i = 1; i < n; i++)
			{
				int a = dp_min[i - 1] * nums[i], b = dp_max[i - 1] * nums[i];
				dp_min[i] = min(min(a, b), nums[i]), dp_max[i] = max(max(a, b), nums[i]);
				ret = max(ret, dp_max[i]);
			}
		}
		return ret;
	}
	```

    - dynamic plan，时间复杂度$O(n)$，空间复杂度优化到$O(1)$

	```cpp
	int maxProduct(vector<int> &nums)
	{
		/**
		 * 朴素的DP解法，给定dp_min[i]和dp_max[i]表示以当前nums[i]结尾的subarray中乘积最小值和最大值
		 * 状态转移方程：
		 * 		dp_min[i] = min(dp_min[i-1]*nums[i],dp_max[i-1]*nums[i],nums[i])
		 * 		dp_max[i] = max(dp_min[i-1]*nums[i],dp_max[i-1]*nums[i],nums[i])
		 * 时间复杂度：O(n)
		 * 空间复杂度：O(n)
		 *
		 * 由于dynamic plan过程中的dp[i]只和上一个状态dp[i-1]有关，因此空间复杂度可以优化到O(1)
		 *
		*/
		int ret = 0;
		if (nums.size() > 0)
		{
			int n = nums.size(), dp_min = nums[0], dp_max = nums[0];
			ret = nums[0];
			for (int i = 1; i < n; i++)
			{
				int a = dp_min * nums[i], b = dp_max * nums[i];
				dp_min = min(min(a, b), nums[i]), dp_max = max(max(a, b), nums[i]);
				ret = max(ret, dp_max);
			}
		}
		return ret;
	}
	```

- [153. Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)

	寻找排序数组(**没有重复值，unique**)旋转后的基准点(最小值)，binary search，时间复杂度$O(log(n))$

	```cpp
	int findMin(vector<int> &nums)
	{
		int left = 0, right = nums.size() - 1;
		while (left < right)
		{
			int mid = left + ((right - left) >> 1);
			if (nums[mid] > nums[right])
			{
				left = mid + 1;
			}
			else
			{
				right = mid;
			}
		}
		return nums[left];
	}
	```

- [154. Find Minimum in Rotated Sorted Array II](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/)

	寻找排序数组(**有重复值，duplicated**)旋转后的基准点(最小值)，binary search，平均时间复杂度$O(log(n))$，在最坏的情况下(整个数组所有数字全部相同)，会退化到$O(nlog(n))$，

	```cpp
	class Solution
	{
	private:
		int binary_search(vector<int> &nums, int left, int right)
		{
			int ret = nums[left];
			if (left < right)
			{
				int mid = left + ((right - left) >> 1);
				if (nums[mid] > nums[right])
				{
					ret = binary_search(nums, mid + 1, right);
				}
				else if (nums[mid] == nums[right])
				{
					ret = min(binary_search(nums, left, mid), binary_search(nums, mid + 1, right));
				}
				else
				{
					ret = binary_search(nums, left, mid);
				}
			}
			return ret;
		}

	public:
		int findMin(vector<int> &nums)
		{
			return binary_search(nums, 0, nums.size() - 1);
		}
	};
	```

    - some test cases

	```cpp
	[1,3,5]
	[3,3,1,3]
	[2,2,2,0,1]
	[4,5,6,7,7,7,0,0,1,1,2,2,3,3,3]
	```

- [155](https://leetcode.com/problems/min-stack/)

    设计实现一个最小栈，即除正常的压栈、弹栈操作外，可以随时返回栈中元素的最小值，因此使用一个int型数组st来保存元素，在数组尾部操作实现压栈与弹栈操作，另外使用一个int型数组min_index，在每次压栈时用min_index[i]到记录st中从栈底到栈顶(st[0]到st[i])的最小值元素位置下标，即可随时返回栈中最小值元素，这样压栈、弹栈、返回最小值的操作均为$O(1)$时间

    ```cpp
    class MinStack {
        private:
            vector<int> st;
            vector<int> min_indexs;
        public:
            /** initialize your data structure here. */
            MinStack() {}
            void push(int x) {
                if(!st.empty()){
                    int index=x<st[min_indexs.back()]?st.size():min_indexs.back();
                    min_indexs.push_back(index);
                }else{
                    min_indexs.push_back(0);
                }
                st.push_back(x);
            }
            void pop() {
                st.pop_back();
                min_indexs.pop_back();
            }
            int top() {
                return st.back();
            }
            int getMin() {
                return st[min_indexs.back()];
            }
    };
    ```

- [162](https://leetcode.com/problems/find-peak-element/)

    寻找peak element即是寻找局部最大值，因此可以二分搜索在$O(log(n))$时间内实现，如果是全局最大值则至少要遍历一次，需要$O(n)$时间来实现

    ```cpp
    int findPeakElement(vector<int>& nums) {
        int left=0,right=nums.size()-1;
        while(left<right){
            int mid=(left+right)>>1;
            if(nums[mid]>nums[mid+1]){
                right=mid;
            }else{
                left=mid+1;
            }
        }
        return left;
    }
    ```

    注意peak element只需要比自己左右两侧的值大就可以了，没必要比最左端和最右端的值大

- [164. Maximum Gap](https://leetcode.com/problems/maximum-gap/)

    - 排序后求相邻两个数之间的gap，时间复杂度$O(nlog(n))$

    ```cpp
    int maximumGap(vector<int> &nums)
    {
        int ret = 0;
        if(nums.size()>=2){
            sort(nums.begin(), nums.end());
            for (int i = 1; i < nums.size(); i++)
            {
                ret = max(ret, nums[i] - nums[i - 1]);
            }
        }
        return ret;
    }
    ```

    - **桶排序(bucket sort)**，时间复杂度$O(n+m) \approx O(n)$，其中m为桶的大小

    ```cpp
    int maximumGap(vector<int> &nums)
    {
        int ret = 0;
        if (nums.size() >= 2)
        {
            // bucket sort
            int min_value = *min_element(nums.begin(), nums.end());
            int max_value = *max_element(nums.begin(), nums.end());
            int bucketSize = max(1, (max_value - min_value) / (int)(nums.size() - 1));
            int bucketCount = (int)(ceil((max_value - min_value + 1) * 1.0 / bucketSize));
            vector<vector<int>> buckets(bucketCount);
            for (auto &&v : nums)
            {
                buckets[(v - min_value) / bucketSize].push_back(v);
            }
            nums.clear();
            for (auto &&bucket : buckets)
            {
                if (!bucket.empty())
                {
                    sort(bucket.begin(), bucket.end());
                    nums.insert(nums.end(), bucket.begin(), bucket.end());
                }
            }
            // get the max gap
            for (int i = 1; i < nums.size(); i++)
            {
                ret = max(ret, nums[i] - nums[i - 1]);
            }
        }
        return ret;
    }
    ```

    - 使用桶排序(bucket sort)的概念而不排序，直接在桶的内部比较，时间复杂度$O(n+m) \approx O(n)$

    ```cpp
    struct Bucket
    {
        bool used = false;
        int low = numeric_limits<int>::max();
        int high = numeric_limits<int>::min();
    };

    int maximumGap(vector<int> &nums)
    {
        int ret = 0;
        if (nums.size() >= 2)
        {
            // bucket sort
            int min_value = *min_element(nums.begin(), nums.end());
            int max_value = *max_element(nums.begin(), nums.end());
            int bucketSize = max(1, (max_value - min_value) / (int)(nums.size() - 1));
            int bucketCount = (int)(ceil((max_value - min_value + 1) * 1.0 / bucketSize));
            vector<Bucket> buckets(bucketCount);
            for (auto &&v : nums)
            {
                int index = (v - min_value) / bucketSize;
                buckets[index].used = true;
                buckets[index].low = min(buckets[index].low, v);
                buckets[index].high = max(buckets[index].high, v);
            }
            // get the max gap
            int preBucketMax = min_value;
            for (auto &&bucket : buckets)
            {
                if (bucket.used)
                {
                    ret = max(ret, bucket.low - preBucketMax);
                    preBucketMax = bucket.high;
                }
            }
        }
        return ret;
    }
    ```

- [165. Compare Version Numbers](https://leetcode.com/problems/compare-version-numbers/)

	将两个版本号转化为同级位置下标相同的两个int型向量，用默认的小版本号0补齐两个向量到一样长，对比两个向量即可，时间复杂度$O(n)$

	```cpp
	class Solution
	{
	private:
		vector<int> stringVersion2Vector(string s)
		{
			vector<int> ret;
			s.push_back('.');
			int v = 0;
			for (auto &&ch : s)
			{
				if (ch == '.')
				{
					ret.push_back(v);
					v = 0;
				}
				else
				{
					v = v * 10 + (int)(ch - '0');
				}
			}
			return ret;
		}

	public:
		int compareVersion(string version1, string version2)
		{
			vector<int> v1 = stringVersion2Vector(version1);
			vector<int> v2 = stringVersion2Vector(version2);
			int length = max(v1.size(), v2.size());
			while (v1.size() < length)
			{
				v1.push_back(0); // 默认的小版本号为0
			}
			while (v2.size() < length)
			{
				v2.push_back(0);
			}
			for (auto i = 0; i < length; i++)
			{
				if (v1[i] > v2[i])
				{
					return 1;
				}
				else if (v1[i] < v2[i])
				{
					return -1;
				}
			}
			return 0;
		}
	};
	```

    - some test cases

	```cpp
	"0.1"
	"1.1"
	"1.0.1"
	"1"
	"7.5.2.4"
	"7.5.3"
	"1.01"
	"1.001"
	"1.0"
	"1.0.0"
	```

- [166. Fraction to Recurring Decimal](https://leetcode.com/problems/fraction-to-recurring-decimal/)

    基本思想是在小数部分除不尽的情况下用hashmap来记录出现过的所有余数，余数r重复出现时则从r第一次出现的位置到当前位置构成无限循环小数，本题的难点在于测试数据中的各种边界点，需要注意点有：
    - 对负数的处理，可以转化为正数计算，然后再将可能有的负号添加到结果中，此时一定注意-2147483648转化为正数会导致int溢出，一定要用long long类型
    - 除数为0没有意义，直接返回空串作为结果
    - cpp STL中有内置的gcd()函数求最大公约数，可以不用自己写
    - 结果为0是前面不能添加负号(eg, 0 / (-2) = 0)

    ```cpp
    string fractionToDecimal(int numerator, int denominator)
    {
        string ret, remainder;
        long long a = numerator, b = denominator; // 防止-2147483648变为正数溢出
        // 首先保证除数不为0
        if (b != 0)
        {
            // 处理正负数的情况
            int sign = 1;
            if (a < 0)
            {
                a = -a, sign *= -1;
            }
            if (b < 0)
            {
                b = -b, sign *= -1;
            }
            // 计算整数部分
            long long factor = gcd(a, b);
            a /= factor, b /= factor;
            long long q = a / b, r = a % b;
            if (sign < 0 && a != 0)
            {
                ret.push_back('-');
            }
            ret += to_string(q);
            unordered_map<long long, int> remainder2index;
            int startPointOfRepeating = 0;
            // 计算小数部分
            while (r != 0)
            {
                remainder2index[r] = startPointOfRepeating++;
                r *= 10;
                q = r / b, r = r % b;
                remainder.push_back((char)(q + '0'));
                if (remainder2index.find(r) != remainder2index.end())
                {
                    // 如果余数出现重复，则是无限循环小数
                    remainder.insert(remainder.begin() + remainder2index[r], '(');
                    remainder.push_back(')');
                    break;
                }
            }
            if (remainder.length() > 0)
            {
                ret += '.' + remainder;
            }
        }
        return ret;
    }
    ```

- [167](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)

	在给定的有序数组中寻找两个数使其和为给定的target，O(n)时间复杂度

  	```cpp
    vector<int> twoSum(vector<int> &numbers, int target)
    {
		int left = 0, right = numbers.size() - 1;
		while (left < right)
		{
			if (numbers.at(left) + numbers.at(right) == target)
			{
				break;
			}
			else if (numbers.at(left) + numbers.at(right) > target)
			{
				right--;
			}
			else if (numbers.at(left) + numbers.at(right) < target)
			{
				left++;
			}
		}
		return vector<int>{left + 1, right + 1};
    }
  	```

- [169](https://leetcode.com/problems/majority-element/)

    找出给定数组中出现次数超过一半的数，后续还有升级版[229](https://leetcode.com/problems/majority-element-ii/)寻找给定数组中出现次数超过$\frac{1}{3}$的数。

    - 首先遍历数组用hashmap统计每个数出现的次数，然后遍历hashmap来比较是否有出现次数超过一半，假定hashmap的$O(1)$存取效率，可以实现$O(n)$的限行时间复杂度
    - 首先对数组排序，从而使得相同数字全部相邻，然后从左到右遍历是否有出现次数超过一半的数字，时间复杂度来自排序算法，一般为$O(nlog(n))$
    - MOOER voting 算法，可以实现真$O(n)$的线性时间和$O(1)$的常量空间，[here](https://www.hijerry.cn/p/45987.html)and[here](https://blog.csdn.net/tinyjian/article/details/79110473)有相关技术博客。
    
        > 摩尔投票法的基本原理是数学上可以证明数组中出现次数超过数组长度一半的元素最多仅有一个。在从左向右遍历数组的过程中维护两个变量，当前大多数cur_majority和当前大多数的投票得分cur_votes，在遍历数组的过程中遇到投票vote：
        - 如果vote和当前大多数相同，则得分cur_votes++
        - 如果vote和当前大多数不同
            - 此时如果当前大多数的得分已经为0，则用vote更新当前大多数，且得分更新为1
            - 此时如果当前大多数的得分尚大于0，则得分cur_vote--
        > 最终的当前大多数即是得票过半的 majority element。
    
    ```cpp
    int majorityElement(vector<int> &nums)
    {
        // method 1, sorting and search

        // sort(nums.begin(), nums.end());
        // int cur_count = 1, count = nums.size(), half = count / 2, pivot = nums[0];
        // for (int i = 1; i < count; i++)
        // {
        // 	if (cur_count > half)
        // 	{
        // 		break;
        // 	}
        // 	if (nums[i] == pivot)
        // 	{
        // 		cur_count++;
        // 	}
        // 	else
        // 	{
        // 		cur_count = 1;
        // 		pivot = nums[i];
        // 	}
        // }
        // return pivot;

        // method 2, mooer voting

        int cur_majority = nums[0], cur_votes = 1;
        for (int i = 1; i < nums.size(); i++)
        {
            if (nums[i] == cur_majority)
            {
                cur_votes++;
            }
            else if (cur_votes > 0)
            {
                cur_votes--;
            }
            else
            {
                cur_majority = nums[i];
                cur_votes = 1;
            }
        }
        return cur_majority;
    }
    ```

- [172](https://leetcode.com/problems/factorial-trailing-zeroes/)

    计算给定数n的阶乘($n!$)的结果中0的个数，理论上结果中0只能由5的偶数倍得来，因此计算从1到n的所有数中因子5的个数即可，时间复杂度$O(log(n))$。

    ```cpp
    int trailingZeroes(int n)
    {
        int ans = 0, power = 1;
        long base = 5;
        while (base <= n)
        {
            ans += n / base;
            base *= 5;
            power++;
        }
        return ans;
    }
    ```

- [173](https://leetcode.com/problems/binary-search-tree-iterator/)

    用$O(n)$的时间复杂度和$O(h)$的空间复杂度实现一个二叉搜索树（BST）的迭代器，n和h分别为二叉搜索树的节点数和高度，这里用一个栈来存储所有的左子节点即可实现$O(h)$的空间复杂度，同时保证栈顶即为当前最左子节点，即最小值，每次next请求返回栈顶元素，同时将栈顶的右子节点压栈即可，这样每个节点至多访问一次，实现$O(n)$的时间复杂度，同时判断栈是否为空即可确定hasNext

    ```cpp
    class BSTIterator
    {
    public:
        stack<TreeNode *> st;
        void push_stack(TreeNode *root)
        {
            while (root)
            {
                st.push(root);
                root = root->left;
            }
        }
        BSTIterator(TreeNode *root)
        {
            push_stack(root);
        }

        /** @return the next smallest number */
        int next()
        {
            TreeNode *cur = st.top();
            st.pop();
            int ans = cur->val;
            push_stack(cur->right);
            return ans;
        }

        /** @return whether we have a next smallest number */
        bool hasNext()
        {
            return !st.empty();
        }
    };
    ```

- [174. Dungeon Game](https://leetcode.com/problems/dungeon-game/)

	动态规划，时间复杂度$O(m*n)$

	```cpp
	int calculateMinimumHP(vector<vector<int>> &dungeon)
	{
		int ret = 0;
		if (dungeon.size() > 0 && dungeon[0].size() > 0)
		{
			int rows = dungeon.size(), cols = dungeon[0].size();
			// K到达P的生命值必须为正数，又要血量最少，则为1
			dungeon[rows - 1][cols - 1] = max(1, 1 - dungeon[rows - 1][cols - 1]);
			for (auto i = rows - 2; i >= 0; i--)
			{
				dungeon[i][cols - 1] = max(1, dungeon[i + 1][cols - 1] - dungeon[i][cols - 1]);
			}
			for (auto j = cols - 2; j >= 0; j--)
			{
				dungeon[rows - 1][j] = max(1, dungeon[rows - 1][j + 1] - dungeon[rows - 1][j]);
			}
			for (auto i = rows - 2; i >= 0; i--)
			{
				for (auto j = cols - 2; j >= 0; j--)
				{
					dungeon[i][j] = max(1, min(dungeon[i][j + 1], dungeon[i + 1][j]) - dungeon[i][j]);
				}
			}
			return dungeon[0][0];
		}
		return ret;
	}
	```

- [179. Largest Number](https://leetcode.com/problems/largest-number/)

    将给定的一个数组中所有的数重新排列串联起来构成一个新的整数，使得其值最大，遵循越接近9的数字尽量靠前排就可以，注意潜在的前导0的处理

    ```cpp
    string largestNumber(vector<int> &nums)
    {
        vector<string> nums_str;
        int count_zero = 0;
        for (auto &&v : nums)
        {
            if (v != 0)
            {
                nums_str.push_back(to_string(v));
            }
            else
            {
                count_zero++;
            }
        }
        sort(nums_str.begin(), nums_str.end(), [](string a, string b) {
            while (true)
            {
                int i = 0, length_a = a.length(), length_b = b.length(), length = min(length_a, length_b);
                if (length == 0)
                {
                    return true; // two equal string
                }
                while (i < length)
                {
                    if (a[i] > b[i])
                    {
                        return true;
                    }
                    else if (a[i] < b[i])
                    {
                        return false;
                    }
                    else
                    {
                        i++;
                    }
                }
                if (i < length_a && i == length_b)
                {
                    a = a.substr(i, length_a - i);
                }
                else if (i == length_a && i < length_b)
                {
                    b = b.substr(i, length_b - i);
                }
                else
                {
                    return true; // two equal string
                }
            }
        });
        string ans;
        for (auto &&s : nums_str)
        {
            ans += s;
        }
        if (count_zero != 0)
        {
            // for potential leading zero
            if (ans.empty())
            {
                ans = "0";
            }
            else
            {
                for (int i = 0; i < count_zero; i++)
                {
                    ans.push_back('0');
                }
            }
        }
        return ans;
    }
    ```

- [187. Repeated DNA Sequences](https://leetcode.com/problems/repeated-dna-sequences/)

	在给定字符串中查找固定长度的重复出现的字符串，rolling hash，时间复杂度$O(n)$

	```cpp
	class Solution
	{
	private:
		bool compare(vector<int> &nums, int a, int b, int fixed_length)
		{
			for (auto i = 0; i < fixed_length; i++)
			{
				if (nums[a + i] != nums[b + i])
				{
					return false;
				}
			}
			return true;
		}

	public:
		vector<string> findRepeatedDnaSequences(string s)
		{
			vector<string> ret;
			if (s.length() > 10)
			{
				unordered_set<int> founds;
				unordered_map<int, vector<int>> map; // rolling hash value -> start index of substring with length 10
				long long hash_value = 0, base = 26, mod = 1e9 + 7;
				int n = s.length(), fixed_length = 10;
				vector<int> nums(n);
				vector<long long> powers(fixed_length, 1);
				for (int i = 1; i < fixed_length; i++)
				{
					powers[i] = (base * powers[i - 1]) % mod;
				}
				for (auto i = 0; i < n; i++)
				{
					nums[i] = static_cast<int>(s.at(i) - 'A');
				}
				for (auto i = 0; i < fixed_length; i++)
				{
					hash_value = (hash_value * base + nums[i]) % mod;
				}
				map[hash_value] = vector<int>{0};
				for (auto i = fixed_length; i < n; i++)
				{
					int current_start = i - fixed_length + 1;
					hash_value = ((hash_value - nums[i - fixed_length] * powers[fixed_length - 1]) % mod + mod) % mod;
					hash_value = (hash_value * base + nums[i]) % mod;
					if (map.find(hash_value) != map.end())
					{
						for (auto &&start_index : map[hash_value])
						{
							if (compare(nums, start_index, current_start, fixed_length))
							{
								founds.insert(start_index);
								break;
							}
						}
						map[hash_value].push_back(current_start);
					}
					else
					{
						map[hash_value] = vector<int>{current_start};
					}
				}
				for (auto &&item : founds)
				{
					ret.push_back(s.substr(item, fixed_length));
				}
			}
			return ret;
		}
	};
	```

- [188. Best Time to Buy and Sell Stock IV](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/)

    - 动态规划，时间复杂度$O(n*k)$，空间复杂度$O(k)

	```cpp
	int maxProfit(int k, vector<int> &prices)
	{
		/**
		 * dynamic plan algorithm
		 *********************************
		 * 空间复杂度从O(n)优化到O(1)
		 * 
		 * 初始化条件：
		 * 		初始状态为get时收益为负无穷，初始状态为清仓售空时收益为0
		 * 返回值：
		 * 		只有最终状态为售出时才有可能获得最大收益
		*/
		int ret = 0, n = prices.size(), inf = numeric_limits<int>::max();
		if (k * 2 < n)
		{
			vector<int> get(k + 1, -inf), out(k + 1, 0);
			for (auto &&v : prices)
			{
				for (auto i = k; i >= 1; i--)
				{
					out[i] = max(out[i], get[i] + v);
					get[i] = max(get[i], out[i - 1] - v);
				}
			}
			ret = *max_element(out.begin(), out.end());
		}
		else
		{
			for (auto i = 1; i < n; i++)
			{
				ret += max(0, prices[i] - prices[i - 1]);
			}
		}
		return ret;
	}
	```

- [189. 旋转数组](https://leetcode-cn.com/problems/rotate-array/)

    - 建立新数组ret，按照循环右移$k\%n$后的顺序填充新数组，时间复杂度$O(n)$，空间复杂度$O(1)$

    ```cpp
	void rotate(vector<int> &nums, int k)
	{
		const int n = nums.size();
		if (k > 0)
		{
			k %= n;
			vector<int> ret(n);
			for (int i = 0; i < n; i++)
			{
				ret[(k + i + 1) % n] = nums[i];
			}
			nums = ret;
		}
	}
    ```

    - 通过三次翻转在$O(n)$时间和$O(1)$时间内实现

    ```cpp
	void rotate(vector<int> &nums, int k)
	{
		const int n = nums.size();
		if (k > 0)
		{
			k %= n;
			int left = 0, right = n - k - 1;
			while (left < right)
			{
				swap(nums[left++], nums[right--]);
			}
			left = n - k, right = n - 1;
			while (left < right)
			{
				swap(nums[left++], nums[right--]);
			}
			left = 0, right = n - 1;
			while (left < right)
			{
				swap(nums[left++], nums[right--]);
			}
		}
	}
    ```

- [200](https://leetcode.com/problems/number-of-islands/submissions/)

    统计岛屿数量，典型的DFS应用，每当发现一个unvisited陆地标记('1')的时候，岛屿数量加1，递归地将其四个方向的陆地('1')全部标记为visited即可。

    ```cpp
    void marker(vector<vector<char>> &grid, int i, int j)
    {
        if (i > 0 && grid[i - 1][j] == '1')
        {
            grid[i - 1][j] = '2'; // visited
            marker(grid, i - 1, j);
        }
        if (i < grid.size() - 1 && grid[i + 1][j] == '1')
        {
            grid[i + 1][j] = '2'; // visited
            marker(grid, i + 1, j);
        }
        if (j > 0 && grid[i][j - 1] == '1')
        {
            grid[i][j - 1] = '2'; // visited
            marker(grid, i, j - 1);
        }
        if (j < grid[0].size() - 1 && grid[i][j + 1] == '1')
        {
            grid[i][j + 1] = '2'; // visited
            marker(grid, i, j + 1);
        }
    }
    int numIslands(vector<vector<char>> &grid)
    {
        int ans = 0;
        if (grid.size() > 0 && grid[0].size() > 0)
        {
            int rows = grid.size(), cols = grid[0].size();
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (grid[i][j] == '1')
                    {
                        // 发现陆地
                        marker(grid, i, j);
                        ans++;
                    }
                }
            }
        }
        return ans;
    }
    ```