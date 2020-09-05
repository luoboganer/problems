# 1301-1400

- [1301. Number of Paths with Max Score](https://leetcode.com/problems/number-of-paths-with-max-score/)

    典型的动态规划问题，用dp[i][j]表示当前状态，其中dp[i][j][0]表示到当前的max score，dp[i][j][1]表示到当前的number of paths with max score，时间复杂度$O(n^2)$

    ```cpp
    vector<int> pathsWithMaxScore(vector<string> &board)
    {
        const int n = board.size(), mode = 1e9 + 7;
        vector<int> ret{0, 0};
        if (n > 0)
        {
            board.back().back() = '0';
            vector<vector<int>> directions{ {-1, -1}, {-1, 0}, {0, -1} };
            vector<vector<vector<int>>> dp(n, vector<vector<int>>(n, vector<int>(2, 0)));
            dp[0][0][1] = 1; // 出发点是永远可以到达的，且距离为0路径数为1
            long long auxiliary = 0;
            // first row and first col
            for (int i = 1; i < n; i++)
            {
                if (board[0][i] == 'X')
                {
                    break;
                }
                else
                {
                    dp[0][i][0] = dp[0][i - 1][0] + (int)(board[0][i] - '0');
                    dp[0][i][1] = 1; // 只有board[0][i-1]到board[0][i]一条路径
                }
            }
            for (int i = 1; i < n; i++)
            {
                if (board[i][0] == 'X')
                {
                    break;
                }
                else
                {
                    dp[i][0][0] = dp[i - 1][0][0] + (int)(board[i][0] - '0');
                    dp[i][0][1] = 1; // 只有board[i-1][0]到board[i][0]一条路径
                }
            }
            for (int i = 1; i < n; i++)
            {
                for (int j = 1; j < n; j++)
                {
                    if (board[i][j] != 'X')
                    {
                        for (auto &&item : directions)
                        {
                            if (dp[i + item[0]][j + item[1]][1] > 0)
                            {
                                if (dp[i + item[0]][j + item[1]][0] > dp[i][j][0])
                                {
                                    dp[i][j][0] = dp[i + item[0]][j + item[1]][0]; // 可到达的路径
                                    dp[i][j][1] = dp[i + item[0]][j + item[1]][1];
                                }
                                else if (dp[i + item[0]][j + item[1]][0] == dp[i][j][0])
                                {
                                    dp[i][j][1] = (auxiliary + dp[i][j][1] + dp[i + item[0]][j + item[1]][1]) % mode;
                                }
                            }
                        }
                        dp[i][j][0] += (int)(board[i][j] - '0');
                    }
                }
            }
            if (dp.back().back()[1] != 0)
            {
                ret = dp.back().back();
            }
        }
        return ret;
    }
    ```

- [1312. Minimum Insertion Steps to Make a String Palindrome](https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/)

    - 递归方法，$\color{red}{TLE}$

    ```cpp
    int findMinInsertions(string &s,int left,int right){
        int ret = 0;
        if(left>right){
            ret = numeric_limits<int>::max();
        }else if(left==right){
            ret = 0;
        }else if(left==right-1){
            ret = s[left] == s[right] ? 0 : 1;
        }else{
            ret = (s[left] == s[right]) ? (findMinInsertions(s, left + 1, right - 1)) : (min(findMinInsertions(s, left + 1, right), findMinInsertions(s, left, right - 1)) + 1);
        }
        return ret;
    }
    int minInsertions(string s)
    {
        return findMinInsertions(s, 0, s.length() - 1);
    }
    ```

    - DP方法直接求需要插入字符的次数，时间复杂度$O(n^2)$

    ```cpp
    int minInsertions(string s)
    {
        int n = s.length();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        for (int gap = 1; gap < n; gap++)
        {
            for (int left = 0, right = gap; right < n; left++, right++)
            {
                dp[left][right] = (s[left] == s[right]) ? (dp[left + 1][right - 1]) : (min(dp[left + 1][right], dp[left][right - 1]) + 1);
            }
        }
        return dp[0].back();
    }
    ```

    - DP求最长公共子串，将剩下的部分补齐即可构成回文串，时间复杂度$O(n^2)$

    ```cpp
    int minInsertions(string s)
    {
        const int n = s.length();
        string t(s.rbegin(), s.rend());
        vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0));
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= n; j++)
            {
                dp[i][j] = (s[i - 1] == t[j - 1]) ? (dp[i - 1][j - 1] + 1) : max(dp[i][j - 1], dp[i - 1][j]);
            }
        }
        return n - dp.back().back();
    }
    ```

    几个典型testcase

    ```cpp
    "zzazz"
    "mbadm"
    "leetcode"
    "g"
    "no"
    "vsrgaxxpgfiqdnwvrlpddcz"
    "tldjbqjdogipebqsohdypcxjqkrqltpgviqtqz"
    ```

- [1314. Matrix Block Sum](https://leetcode.com/problems/matrix-block-sum/)

    - 暴力求和，时间复杂度$O(n^2*k^2)$，LeetCode时间效率$\color{red}{872 ms, 14.37\%}$

    ```cpp
    vector<vector<int>> matrixBlockSum(vector<vector<int>> &mat, int K)
    {
        const int m = mat.size(), n = mat[0].size();
        vector<vector<int>> ret(m, vector<int>(n, 0));
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int r = i, c = j;
                for (int x = r - K; x <= r + K; x++)
                {
                    for (int y = c - K; y <= c + K; y++)
                    {
                        if (x >= 0 && y >= 0 && x < m && y < n)
                        {
                            ret[r][c] += mat[x][y];
                        }
                    }
                }
            }
        }
        return ret;
    }
    ```

    - 对原数组从左上至右下累加求和（前缀和），时间复杂度$O(n^2)$，LeetCode时间效率$\color{red}{24 ms, 96.44\%}$

    ```cpp
    vector<vector<int>> matrixBlockSum(vector<vector<int>> &mat, int K)
    {
        const int m = mat.size(), n = mat[0].size();
        vector<vector<int>> prefixSum(m + 1, vector<int>(n + 1, 0));
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                prefixSum[i + 1][j + 1] += mat[i][j] - prefixSum[i][j] + prefixSum[i][j + 1] + prefixSum[i + 1][j];
            }
        }
        vector<vector<int>> ret(m, vector<int>(n, 0));
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int r1 = max(0, i - K), c1 = max(0, j - K), r2 = min(i + 1 + K, m), c2 = min(j + 1 + K, n);
                ret[i][j] = prefixSum[r2][c2] + prefixSum[r1][c1] - prefixSum[r1][c2] - prefixSum[r2][c1];
            }
        }
        return ret;
    }
    ```

- [1316. Distinct Echo Substrings](https://leetcode.com/problems/distinct-echo-substrings/)

    - 用一个hashset来存储所有符合条件的子串，统计数量，时间复杂度$O(n^2),n=text.length$，LeetCode时间效率$\color{red}{184 ms, 76.06\%}$

    ```cpp
    int distinctEchoSubstrings(string text)
    {
        unordered_set<string> substrings;
        const int n = text.length() / 2;
        for (int i = 1; i <= n; i++)
        {
            // 搜索所有被重复子串长度为i的子串substring
            int left = 0, right = i, c = 0;
            while (right < text.length())
            {
                if (text[left++] == text[right++])
                {
                    c++;
                }
                else
                {
                    c = 0;
                }
                if (c >= i && c < i * 2)
                {
                    substrings.insert(text.substr(left, i));
                }
            }
        }
        return substrings.size();
    }
    ```

    - 抛弃hashset，两次遍历，避免重复，时间复杂度$O(n^2),n=text.length$，LeetCode时间效率$\color{red}{80 ms, 88.09\%}$

    ```cpp
    int distinctEchoSubstrings(string text)
    {
        const int n = text.length();
        vector<int> copy_n(n + 1, 0);
        for (int i = 1; i < n; i++)
        {
            int left = 0, right = i, c = 0;
            while (right < n)
            {
                if (text[left++] == text[right++])
                {
                    c++;
                }
                else
                {
                    c = 0;
                }
                copy_n[right] = max(copy_n[right], c);
            }
        }
        int ret = 0;
        for (int i = 1; i <= n / 2; i++)
        {
            int left = 0, right = i, c = 0;
            while (right < n)
            {
                if (text[left++] == text[right++])
                {
                    c++;
                }
                else
                {
                    c = 0;
                }
                if (c >= i && copy_n[right] < i * 2)
                {
                    ret++;
                }
            }
        }
        return ret;
    }
    ```

- [1325. Delete Leaves With a Given Value](https://leetcode.com/problems/delete-leaves-with-a-given-value/)

    - 迭代式post-order遍历二叉树，将符合条件的节点值标记为-1，然后递归式遍历所有节点，删除值为-1的节点，时间复杂度$O(n)$，其中n为二叉树所有节点的数量

    ```cpp
    TreeNode *dfs_remove(TreeNode *root)
    {
        if (root)
        {
            if (root->val == -1)
            {
                root = nullptr;
            }
            else
            {
                root->left = dfs_remove(root->left);
                root->right = dfs_remove(root->right);
            }
        }
        return root;
    }
    TreeNode *removeLeafNodes(TreeNode *root, int target)
    {
        if (root)
        {
            TreeNode *cur = root;
            stack<TreeNode *> st;
            unordered_set<TreeNode *> visited;
            while (cur || !st.empty())
            {
                if (cur)
                {
                    st.push(cur);
                    cur = cur->left;
                }
                else
                {
                    if (visited.find(st.top()->right) != visited.end())
                    {
                        cur = st.top(), st.pop();
                        if (cur->val == target && (!cur->left || cur->left->val == -1) && (!cur->right || cur->right->val == -1))
                        {
                            cur->val = -1;
                        }
                        cur = nullptr;
                    }
                    else
                    {
                        cur = st.top()->right;
                        visited.insert(cur);
                    }
                }
            }
            root = dfs_remove(root);
        }
        return root;
    }
    ```

    - 将迭代式post-order遍历改为递归式

    ```cpp
    TreeNode *dfs_remove(TreeNode *root)
    {
        if (root)
        {
            if (root->val == -1)
            {
                root = nullptr;
            }
            else
            {
                root->left = dfs_remove(root->left);
                root->right = dfs_remove(root->right);
            }
        }
        return root;
    }
    void dfs_mark(TreeNode *root, int target)
    {
        if (root)
        {
            if (root->left)
            {
                dfs_mark(root->left, target);
            }
            if (root->right)
            {
                dfs_mark(root->right, target);
            }
            if (root->val == target && (!root->left || root->left->val == -1) && (!root->right || root->right->val == -1))
            {
                root->val = -1;
            }
        }
    }
    TreeNode *removeLeafNodes(TreeNode *root, int target)
    {
        dfs_mark(root, target);
        root = dfs_remove(root);
        return root;
    }
    ```

    部分测试数据

    ```cpp
    [1,2,3,2,null,2,4]
    2
    [1,3,3,3,2]
    3
    [1,2,null,2,null,2]
    2
    [1,1,1]
    1
    [1,2,3]
    1
    ```

- [1346. Check If N and Its Double Exist](https://leetcode.com/problems/check-if-n-and-its-double-exist/)

    - 快速排序后从前向后遍历arr[i]和arr[j]是否存在二倍关系，时间复杂度$O(n^2)$

    ```cpp
    bool checkIfExist(vector<int> &arr)
    {
        int const count = arr.size();
        sort(arr.begin(), arr.end());
        bool ans = true;
        for (int i = 0; ans && i < count; i++)
        {
            for (int j = i + 1; ans && j < count; j++)
            {
                if (arr[i] * 2 == arr[j] || arr[i] == arr[j] * 2)
                {
                    ans = false;
                }
            }
        }
        return !ans;
    }
    ```

    - 快排后对任意arr[i]二分查找是否存在arr[j]与其存在二倍关系切$i!=j$，时间复杂度$O(n)$

    ```cpp
    int binary_search(vector<int> &arr, int v)
    {
        int left = 0, right = arr.size() - 1, ans = -1;
        while (left <= right)
        {
            int mid = left + (right - left) / 2;
            if (arr[mid] == v)
            {
                ans = mid;
                break;
            }
            else if (arr[mid] > v)
            {
                right = mid - 1;
            }
            else
            {
                left = mid + 1;
            }
        }
        return ans;
    }
    bool checkIfExist(vector<int> &arr)
    {
        int const count = arr.size();
        sort(arr.begin(), arr.end());
        bool ans = false;
        for (int i = 0; !ans & i < count; i++)
        {
            int j = binary_search(arr, arr[i] * 2);
            if (j != -1 && i != j)
            {
                ans = true;
            }
        }
        return ans;
    }
    ```

- [1361. Validate Binary Tree Nodes](https://leetcode.com/problems/validate-binary-tree-nodes/)

	检查所有节点的入度，其中root有且仅有一个入度为0，其它节点入度必须为1，在此基础上保证root代表的二叉树包含了全部节点节即可，时间复杂度$O(n)$

	```cpp
	class Solution
	{
	private:
		int countNodes(vector<int> &leftChild, vector<int> &rightChild, vector<int> &visited, int root)
		{
			int ret = 1;
			visited[root] = 1;
			if (leftChild[root] != -1 && !visited[leftChild[root]])
			{
				ret += countNodes(leftChild, rightChild, visited, leftChild[root]);
			}
			if (rightChild[root] != -1 && !visited[rightChild[root]])
			{
				ret += countNodes(leftChild, rightChild, visited, rightChild[root]);
			}
			return ret;
		}

	public:
		bool validateBinaryTreeNodes(int n, vector<int> &leftChild, vector<int> &rightChild)
		{
			vector<int> in_degree(n, 0), visited(n, 0);
			for (int i = 0; i < n; i++)
			{
				if (leftChild[i] != -1)
				{
					in_degree[leftChild[i]]++;
					if (in_degree[leftChild[i]] > 1)
					{
						return false;
					}
				}
				if (rightChild[i] != -1)
				{
					in_degree[rightChild[i]]++;
					if (in_degree[rightChild[i]] > 1)
					{
						return false; // 二叉树中节点的入度为0(root)或者1(nonRoot)
					}
				}
			}
			int root = -1;
			for (int i = 0; i < n; i++)
			{
				if (in_degree[i] == 0)
				{
					// 入度为0即为root
					if (root == -1)
					{
						root = i;
					}
					else
					{
						return false; // root多于一个
					}
				}
			}
			// 统计root下面的节点数量，保证n个节点全部纳入root下面，避免漏掉某些点或者形成环
			return root != -1 && n == countNodes(leftChild, rightChild, visited, root);
		}
	};

- [1379. Find a Corresponding Node of a Binary Tree in a Clone of That Tree](https://leetcode.com/problems/find-a-corresponding-node-of-a-binary-tree-in-a-clone-of-that-tree/)

    - 递归方法，DFS，时间效率$\color{red}{688 ms, 53.45\%}$

    ```cpp
    TreeNode *findTargetNode(TreeNode *node, TreeNode *target)
    {
        TreeNode *ret = nullptr;
        if (node != nullptr)
        {
            if (node->val == target->val)
            {
                ret = node;
            }
            else
            {
                TreeNode *left = findTargetNode(node->left, target);
                if (left != nullptr)
                {
                    ret = left;
                }
                else
                {
                    ret = findTargetNode(node->right, target);
                }
            }
        }
        return ret;
    }
    TreeNode *getTargetCopy(TreeNode *original, TreeNode *cloned, TreeNode *target)
    {
        return findTargetNode(cloned, target);
    }
    ```

    - 队列实现，BFS，时间效率$\color{red}{732 ms, 20.92\%}$

    ```cpp
    TreeNode *getTargetCopy(TreeNode *original, TreeNode *cloned, TreeNode *target)
    {
        queue<TreeNode *> qe;
        qe.push(cloned);
        TreeNode *ret = nullptr, *current_node = nullptr;
        while (ret == nullptr && !qe.empty())
        {
            current_node = qe.front();
            qe.pop();
            if (current_node->val == target->val)
            {
                ret = current_node;
            }
            else
            {
                if (current_node->left)
                {
                    qe.push(current_node->left);
                }
                if (current_node->right)
                {
                    qe.push(current_node->right);
                }
            }
        }
        return ret;
    }
    ```

- [1381. Design a Stack With Increment Operation](https://leetcode.com/problems/design-a-stack-with-increment-operation/)

    - 每次inc操作都给所有的数字加val值，时间复杂度$O(k)$

    ```cpp
    class CustomStack
    {
    private:
        int top, capacity;
        vector<int> st;

    public:
        CustomStack(int maxSize)
        {
            st = vector<int>(maxSize, 0);
            top = -1;
            capacity = maxSize;
        }

        void push(int x)
        {
            if (top != capacity - 1)
            {
                st[++top] = x;
            }
        }

        int pop()
        {
            int val = -1;
            if (top >= 0)
            {
                val = st[top--];
            }
            return val;
        }

        void increment(int k, int val)
        {
            for (auto i = 0; i < k && i <= top; i++)
            {
                st[i] += val;
            }
        }
    };
    ```

    - lazy increment，空间换时间，用一个额外的inc数组来记录增量值，其中inc[i]标识下标从0到i的每个元素增量值，时间复杂度$O(1)$

    ```cpp
    class CustomStack
    {
    private:
        int top, capacity;
        vector<int> st, inc;

    public:
        CustomStack(int maxSize)
        {
            st = vector<int>(maxSize, 0);
            inc = vector<int>(maxSize, 0);
            top = -1;
            capacity = maxSize;
        }

        void push(int x)
        {
            if (top != capacity - 1)
            {
                st[++top] = x;
            }
        }

        int pop()
        {
            int val = -1;
            if (top >= 0)
            {
                val = st[top];
                val += inc[top];
                if (top > 0)
                {
                    inc[top - 1] += inc[top];
                }
                inc[top] = 0; // 清除对栈顶之后元素的increment操作
                top--;
            }
            return val;
        }

        void increment(int k, int val)
        {
            int index = min(k - 1, top);
            if (index >= 0)
            {
                inc[index] += val;
            }
        }
    };
    ```

- [1387. Sort Integers by The Power Value](https://leetcode.com/problems/sort-integers-by-the-power-value/)

    - lo和hi数据范围未知情况下每个数字单独计算power，时间效率$\color{red}{92 ms, 46.65\%}$

    ```cpp
    int getKth(int lo, int hi, int k)
    {
        const int count = hi - lo + 1;
        vector<vector<int>> powers(count, vector<int>(2, -1));
        for (int v = lo; v <= hi; v++)
        {
            int power = 0, temp_v = v;
            while (temp_v != 1)
            {
                if (lo <= temp_v && temp_v <= hi && powers[temp_v - lo][1] != -1)
                {
                    power += powers[temp_v - lo][1];
                    temp_v = 1;
                }
                else
                {
                    power++;
                    temp_v = (temp_v & 0x1) ? (3 * temp_v + 1) : (temp_v >> 1);
                }
            }
            powers[v - lo][0] = v, powers[v - lo][1] = power;
        }
        sort(powers.begin(), powers.end(), [](auto &a, auto &b) -> bool { return a[1] < b[1] || (a[1] == b[1] && a[0] < b[0]); });
        return powers[k - 1][0];
    }
    ```

    - 给定lo和hi的数据范围，固定DP或者recursion，时间效率$\color{red}{100 ms, 45.50\%}$

    ```cpp
    class Solution
    {
    private:
        int getPower(int v, vector<int> &powers)
        {
            int ret = 0;
            if (v <= 1000)
            {
                if (powers[v] == -1)
                {
                    powers[v] = 1 + ((v & 0x1) ? getPower(v * 3 + 1, powers) : getPower(v / 2, powers));
                }
                ret = powers[v];
            }
            else
            {
                ret = 1 + ((v & 0x1) ? getPower(v * 3 + 1, powers) : getPower(v / 2, powers));
            }
            return ret;
        }

    public:
        int getKth(int lo, int hi, int k)
        {
            const int count = hi - lo + 1, max_hi = 3001;
            vector<int> powers_maxHi(max_hi, -1);
            powers_maxHi[1] = 0;
            vector<vector<int>> powers(count, vector<int>(2, -1));
            for (int v = lo; v <= hi; v++)
            {
                powers[v - lo][0] = v, powers[v - lo][1] = getPower(v, powers_maxHi);
            }
            sort(powers.begin(), powers.end(), [](auto &a, auto &b) -> bool { return a[1] < b[1] || (a[1] == b[1] && a[0] < b[0]); });
            return powers[k - 1][0];
        }
    };
    ```

- [1395. Count Number of Teams](https://leetcode.com/problems/count-number-of-teams/)

    - 给定i,j,k三个下标暴力遍历，时间复杂度$O(n^3)$

    ```cpp
    int numTeams(vector<int> &rating)
    {
        int ret = 0, count = rating.size();
        for (auto i = 0; i < count; i++)
        {
            for (auto j = i + 1; j < count; j++)
            {
                for (auto k = j + 1; k < count; k++)
                {
                    if ((rating[i] < rating[j] && rating[j] < rating[k]) || (rating[i] > rating[j] && rating[j] > rating[k]))
                    {
                        ret++;
                    }
                }
            }
        }
        return ret;
    }
    ```

    - 二重遍历记录每个数v两侧大于v的数个数和右侧小于v的数个数，然后计算即可，时间复杂度$O(n^2)$

    ```cpp
    int numTeams(vector<int> &rating)
    {
        int ret = 0, count = rating.size();
        for (auto i = 1; i < count - 1; i++)
        {
            int leftLess = 0, rightLess = 0, leftGreater = 0, rightGreater = 0;
            for (auto j = 0; j < i; j++)
            {
                rating[j] < rating[i] ? leftLess++ : leftGreater++;
            }
            for (auto j = i + 1; j < count; j++)
            {
                rating[i] < rating[j] ? rightGreater++ : rightLess++;
            }
            ret += leftLess * rightGreater + leftGreater * rightLess;
        }
        return ret;
    }
    ```
