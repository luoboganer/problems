# 501-600

- [501. Find Mode in Binary Search Tree](https://leetcode.com/problems/find-mode-in-binary-search-tree/)

    搜索二叉树的中序遍历，在中序遍历的过程中维护cur_val,cur_count,max_count,ret三个变量的值即可

    ```cpp
    class Solution
    {
    private:
        void dfs(TreeNode *root, int *cur_val, int *cur_freq, int *max_freq, vector<int> &ret)
        {
            // recursive search for left/right subtree
            if (root->left)
            {
                dfs(root->left, cur_val, cur_freq, max_freq, ret);
            }
            // update frequency of the current value
            if (root->val == *cur_val)
            {
                (*cur_freq)++;
            }
            else
            {
                (*cur_freq) = 1;
            }
            if (*cur_freq > *max_freq)
            {
                *max_freq = *cur_freq;
                ret.clear();
                ret.push_back(root->val);
            }
            else if (*cur_freq == *max_freq)
            {
                ret.push_back(root->val);
            }
            *cur_val = root->val;
            // recursive search for left/right subtree
            if (root->right)
            {
                dfs(root->right, cur_val, cur_freq, max_freq, ret);
            }
        }

    public:
        vector<int> findMode(TreeNode *root)
        {
            vector<int> ret;
            int cur_val = 0, cur_freq = 0, max_freq = 0;
            if (root)
            {
                dfs(root, &cur_val, &cur_freq, &max_freq, ret);
            }
            return ret;
        }
    };
    ```

- [503. 下一个更大元素 II](https://leetcode-cn.com/problems/next-greater-element-ii/)

    - 从右向左遍历数组元素，维护一个单调不降栈（栈顶元素最大）来实现寻找每个元素的下一个最大值，时间复杂度$O(n)$

        **Tricks:通过在数组nums后接一个相同的nums来实现循环搜索**

    ```cpp
	vector<int> nextGreaterElements(vector<int> &nums)
	{
		stack<int> st;
		int n = nums.size();
		vector<int> ret(n);
		for (int i = 0; i < n; i++)
		{
			nums.emplace_back(nums[i]);
		}
		for (int i = 2 * n - 1; i >= n; i--)
		{
			while (!st.empty() && st.top() <= nums[i])
			{
				st.pop();
			}
			st.push(nums[i]);
		}
		for (int i = n - 1; i >= 0; i--)
		{
			while (!st.empty() && st.top() <= nums[i])
			{
				st.pop();
			}
			ret[i] = st.empty() ? -1 : st.top();
			st.push(nums[i]);
		}
		return ret;
	}
    ```

    - 通过数组下标的循环实现空间优化

    ```cpp
    vector<int> nextGreaterElements(vector<int> &nums)
	{
		stack<int> st;
		int n = nums.size();
		vector<int> ret(n);
		for (int i = n - 1; i >= 0; i--)
		{
			while (!st.empty() && st.top() <= nums[i])
			{
				st.pop();
			}
			st.push(nums[i]);
		}
		for (int i = n - 1; i >= 0; i--)
		{
			while (!st.empty() && st.top() <= nums[i])
			{
				st.pop();
			}
			ret[i] = st.empty() ? -1 : st.top();
			st.push(nums[i]);
		}
		return ret;
	}
    ```    

- [508. 出现次数最多的子树元素和](https://leetcode-cn.com/problems/most-frequent-subtree-sum/)

    递归地或者非递归自底向上的求所有节点的和，然后用hashmap统计每个值出现的频率，求频率最大的值

    ```cpp
    class Solution
    {
    private:
        int dfs(TreeNode *root, unordered_map<int, int> &count)
        {
            int val = 0;
            if (root)
            {
                val = root->val + dfs(root->left, count) + dfs(root->right, count);
                count[val]++;
            }
            return val;
        }

    public:
        vector<int> findFrequentTreeSum(TreeNode *root)
        {
            unordered_map<int, int> count;
            count[dfs(root, count)]++;
            vector<int> ret;
            int max_freq = 0;
            for (auto &[v, freq] : count)
            {
                if (freq > max_freq)
                {
                    max_freq = freq;
                    ret.clear();
                    ret.emplace_back(v);
                }
                else if (freq == max_freq)
                {
                    ret.emplace_back(v);
                }
            }
            return ret;
        }
    };
    ```

- [509](https://leetcode.com/problems/fibonacci-number/)
    
    斐波那契数列
    - 递归法
    - 非递归循环 faster

- [516](https://leetcode.com/problems/longest-palindromic-subsequence/)

    求给定字符串s中的最长回文子串长度，将s翻转后形成字符串t，用动态规划求s和t的最长公共子序列LCS长度即可‘时间复杂度$O(log(n))$

    ```cpp
    int longestPalindromeSubseq(string s)
    {
        string t = s;
        reverse(s.begin(), s.end());
        const int length = s.length();
        vector<vector<int>> dp(length + 1, vector<int>(length + 1, 0));
        for (int i = 1; i <= length; i++)
        {
            for (int j = 1; j <= length; j++)
            {
                if (s[i - 1] == t[j - 1])
                {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }
                else
                {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp.back().back();
    }
    ```

- [518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)

    动态规划，时间复杂度$O(n*amount)$，其中n为给定硬币种类数，amount是需要兑换的钱数总量

    ```cpp
	int change(int amount, vector<int> &coins)
	{
		const int n = coins.size();
		vector<int> dp(amount + 1, 0);
        dp[0] = 1;
		for (auto &v : coins)
		{
			for (int j = v; j <= amount; j++)
			{
				dp[j] += dp[j - v];
			}
		}
		return dp.back();
	}
    ```

- [521](https://leetcode.com/problems/longest-uncommon-subsequence-i/)

    注意理解最长非公共子串儿的正确含义，即只有两个字符串完全相同时才构成公共子串，否则最长非公共子串就是两个字符串中较长的一个。

- [523. Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/)

	- 计算数组nums的前缀和，然后双重遍历计算其超过两个数的连续和是否是k的倍数（包括k的0倍，其k本身可能为0），时间复杂度$O(n^2)$

	```cpp
    bool checkSubarraySum(vector<int> &nums, int k)
    {
        nums.insert(nums.begin(), 0);
        int count = nums.size();
        for (auto i = 1; i < count; i++)
        {
            nums[i] += nums[i - 1];
        }
        for (auto i = 0; i < count; i++)
        {
            for (auto j = i + 2; j < count; j++)
            {
                int temp = nums[j] - nums[i];
                if (temp == k || (k != 0 && temp % k == 0))
                {
                    return true;
                }
            }
        }
        return false;
    }
	```

	- 计算nums前缀和，然后每个值取%k的结果，当某个值出现两次的时候，即$a\%k==b\%k$，则$b-a=n*k$，符合题意，用hashmap来存储出现过得数字，时间复杂度$O(n)$

	```cpp
    bool checkSubarraySum(vector<int> &nums, int k)
    {
        nums.insert(nums.begin(), 0);
        int count = nums.size();
        unordered_map<int, int> prefix_sum;
        prefix_sum[0] = 0; // nums的第一个数是0
        for (auto i = 1; i < count; i++)
        {
            // 防止除0错误
            nums[i] = (k == 0) ? (nums[i] + nums[i - 1]) : ((nums[i] + nums[i - 1]) % k);
            if (prefix_sum.find(nums[i]) != prefix_sum.end())
            {
                // 保证该值以及出现过且下标差大于1（连续子数组中至少包括两个数）
                if (i - prefix_sum[nums[i]] > 1)
                {
                    return true;
                }
            }
            else
            {
                // 第一次遇到值nums[i]出现时，记录其出现的index
                prefix_sum[nums[i]] = i;
            }
        }
        return false;
    }
	```

- [524. Longest Word in Dictionary through Deleting](https://leetcode.com/problems/longest-word-in-dictionary-through-deleting/)

	- 对字典d排序后一次比较s是否可以形成字典d中的某个字符串，时间复杂度$O(nlog(n)*x+n*x)$，其中$x=max(s.length,d_{i}.length)$

	```cpp
	class Solution
	{
		bool sFormedt(string s, string t)
		{
			if (s.length() < t.length())
			{
				return false;
			}
			int i = 0, j = 0;
			while (i < s.length() && j < t.length())
			{
				if (s[i] == t[j])
				{
					i++, j++;
				}
				else
				{
					i++;
				}
			}
			return j == t.length();
		}

	public:
		string findLongestWord(string s, vector<string> &d)
		{
			sort(d.begin(), d.end(), [](const string &a, const string &b) { return a.length() > b.length() || (a.length() == b.length() && a.compare(b) < 0); });
			for (auto &&t : d)
			{
				if (sFormedt(s, t))
				{
					return t;
				}
			}
			return "";
		}
	};
	```

    - 不排序依次直接比较s是否可以形成字典d中的某个字符串，时间复杂度$O(n*x)$，其中$x=max(s.length,d_{i}.length)$

    ```cpp
	class Solution
	{
		bool sFormedt(string s, string t)
		{
			if (s.length() < t.length())
			{
				return false;
			}
			int i = 0, j = 0;
			while (i < s.length() && j < t.length())
			{
				if (s[i] == t[j])
				{
					i++, j++;
				}
				else
				{
					i++;
				}
			}
			return j == t.length();
		}
		bool compareGreater(const string &a, const string &b)
		{
			return a.length() > b.length() || (a.length() == b.length() && a.compare(b) < 0);
		}

	public:
		string findLongestWord(string s, vector<string> &d)
		{
			string ret;
			for (auto &&t : d)
			{
				if (sFormedt(s, t) && compareGreater(t, ret))
				{
					ret = t;
				}
			}
			return ret;
		}
	};
    ```

- [525. Contiguous Array](https://leetcode.com/problems/contiguous-array/)

    - 统计数组截止到nums[i]位置0和1的数量zeros[i]和ones[i]，然后比较zeros[j] - zeros[i] == ones[j] - ones[i]$，时间复杂度$O(n^2)$，$LeetCode评测机$\color{red}{TLE}$

    ```cpp
    int findMaxLength(vector<int> &nums)
    {
        int ret = 0;
        const int count = nums.size();
        vector<int> zeros(count + 1, 0), ones(count + 1, 0);
        int zero = 0, one = 0;
        for (auto i = 0; i < count; i++)
        {
            nums[i] == 0 ? zero++ : one++;
            zeros[i + 1] = zero, ones[i + 1] = one;
        }
        for (int i = 0; i <= count; i++)
        {
            for (int j = i + 1; j <= count; j++)
            {
                if (zeros[j] - zeros[i] == ones[j] - ones[i])
                {
                    ret = max(ret, ones[j] - ones[i]);
                }
            }
        }
        return ret * 2;
    }
    ```

    - 用一个count变量来统计0和1出现的次数，出现1自增，出现0自减，则count在两个位置i和j出现同一个数的时候意味着i和j之间0和1的数量相同，寻找$max(j-i)$即可，时间复杂度$O(n)$，参考[solution](https://leetcode.com/problems/contiguous-array/solution/)，时间效率$\color{red}{ 96 ms, 90.38\%}$

    ```cpp
    int findMaxLength(vector<int> &nums)
    {
        int ret = 0, count = 0, n = nums.size();
        vector<int> count2index(2 * n + 1, -2);
        count2index[n] = -1;
        for (auto i = 0; i < n; i++)
        {
            nums[i] == 1 ? count++ : count--;
            if (count2index[count + n] == -2)
            {
                count2index[count + n] = i;
            }
            else
            {
                ret = max(ret, i - count2index[count + n]);
            }
        }
        return ret;
    }
    ```

- [526. 优美的排列](https://leetcode-cn.com/problems/beautiful-arrangement/)

    递归搜索所有可能的排列，然后利用优美排列的性质剪枝

    ```cpp
    class Solution
    {
    private:
        int dfs(vector<int> &nums, int start, const int n)
        {
            int ret = 0;
            if (start > n)
            {
                ret = 1;
            }
            else
            {
                for (int i = start - 1; i < n; i++)
                {
                    if (nums[i] % start == 0 || start % nums[i] == 0)
                    {
                        swap(nums[i], nums[start - 1]);
                        ret += dfs(nums, start + 1, n);
                        swap(nums[i], nums[start - 1]);
                    }
                }
            }
            return ret;
        }

    public:
        int countArrangement(int n)
        {
            vector<int> nums(n);
            for (int i = 0; i < n; i++)
            {
                nums[i] = i + 1;
            }
            return dfs(nums, 1, n);
        }
    };
    ```

- [528. 按权重随机选择](https://leetcode-cn.com/problems/random-pick-with-weight/submissions/)

    将权重映射到等长的区间范围内采样，时间复杂度$O(log(n))$，其中$n=sum(weights)$

    ```cpp
    class Solution
    {
    private:
        vector<int> weights;

    public:
        Solution(vector<int> &w)
        {
            weights = w;
            const int n = weights.size();
            for (int i = 1; i < n; i++)
            {
                weights[i] += weights[i - 1];
            }
        }

        int pickIndex()
        {
            int weight = rand() % weights.back();
            return upper_bound(weights.begin(), weights.end(), weight) - weights.begin();
        }
    };
    ```

- [530. 二叉搜索树的最小绝对差](https://leetcode-cn.com/problems/minimum-absolute-difference-in-bst/)

    本题与[783. 二叉搜索树节点最小距离](https://leetcode-cn.com/problems/minimum-distance-between-bst-nodes/)完全相同，对二叉搜索树BST进行中序遍历即可

    ```cpp
    int getMinimumDifference(TreeNode* root) {
		int ret = numeric_limits<int>::max();
		if (root)
		{
			// 中序遍历BST，然后计算前后两个数的差值的绝对值，取最小值
			int pre_val = numeric_limits<int>::min();
			TreeNode *cur = root;
			stack<TreeNode *> st;
			while (cur || !st.empty())
			{
				if (cur)
				{
					st.push(cur);
					cur = cur->left;
				}
				else
				{
					cur = st.top();
					st.pop();
					if (pre_val == numeric_limits<int>::min())
					{
						pre_val = cur->val;
					}
					else
					{
						ret = min(ret, cur->val - pre_val);
						pre_val = cur->val;
					}
					cur = cur->right;
				}
			}
		}
		return ret;
    }
    ```

- [535](https://leetcode.com/problems/encode-and-decode-tinyurl/)

    tinyURL的encode与decode算法

    整体思路：采用hashmap或者字符串数组存储<key,value>对，key是个全局唯一的ID，value是其longURL。而shortURL则是key的64进制表示，这64个字符一般是[0-9a-zA-Z+-]。这里之所以是64进制是因为64为2的6次幂，进制转换效率高。

- [537. Complex Number Multiplication](https://leetcode-cn.com/problems/complex-number-multiplication/)

    模拟竖式相乘即可，重点在输入输出的格式处理(字符串处理)

    ```cpp
    class Solution
    {
        vector<int> stringToComplex(string s)
        {
            int pos = s.find_first_of('+');
            string s1 = s.substr(0, pos + 1), s2 = s.substr(pos + 1, s.length() - 2 - pos);
            return {stoi(s1), stoi(s2)};
        }

    public:
        string complexNumberMultiply(string a, string b)
        {
            vector<int> a_value = stringToComplex(a), b_value = stringToComplex(b);
            int x = a_value[0] * b_value[0] - a_value[1] * b_value[1];
            int y = a_value[0] * b_value[1] + a_value[1] * b_value[0];
            return to_string(x) + "+" + to_string(y) + "i";
        }
    };
    ```

- [539](https://leetcode.com/problems/minimum-time-difference/)

    在24时制的"hh:mm"格式字符串表示的时间序列中寻找最近的两个时间段差值，将每个时间用转换为分钟数后排序，在任意相邻的两个时间的差值中取最小值即可，时间复杂度$O(nlog(n))$，特别注意排序之后的时间序列第一个值和最后一个值之间的差值也要考虑在内。

    ```cpp
    int findMinDifference(vector<string> &timePoints)
    {
        vector<int> minutes;
        for (auto &&point : timePoints)
        {
            minutes.push_back(((point[0] - '0') * 10 + (point[1] - '0')) * 60 + ((point[3] - '0') * 10 + (point[4] - '0')));
        }
        sort(minutes.begin(), minutes.end());
        minutes.push_back(minutes[0] + 1440);
        int diff = numeric_limits<int>::max();
        for (int i = 1; i < minutes.size(); i++)
        {
            diff = min(diff, min(minutes[i] - minutes[i - 1], minutes[i - 1] + 1440 - minutes[i]));
        }
        return diff;
    }
    ```

- [540. 有序数组中的单一元素](https://leetcode-cn.com/problems/single-element-in-a-sorted-array/)

    - 用逐个异或的办法消除重复的数字，时间复杂度$O(n)$

    ```cpp
	int singleNonDuplicate(vector<int> &nums)
	{
		int ret = 0;
		for (auto &v : nums)
		{
			ret ^= v;
		}
		return ret;
	}
    ```

    - 二分法查找只出现一次的那个数字，时间复杂度$O(log(n))$

    ```cpp
	int singleNonDuplicate(vector<int> &nums)
	{
		int left = 0, right = nums.size() - 1;
		while (left < right - 1)
		{
			// 此时至少有三个数字
			int mid = left + ((right - left) >> 1);
			if (nums[mid] == nums[mid - 1] && nums[mid] < nums[mid + 1])
			{
				(mid - left) % 2 == 0 ? right = mid : left = mid + 1;
			}
			else if (nums[mid] > nums[mid - 1] && nums[mid] == nums[mid + 1])
			{
				(right - mid) % 2 == 0 ? left = mid : right = mid - 1;
			}
			else
			{
				return nums[mid];
			}
		}
		return nums[left]; // 理论上不会执行到这里，除非数组只有一个数
	}
    ```

- [542. 01 矩阵](https://leetcode-cn.com/problems/01-matrix/)

    给定一个0/1矩阵，寻找每个1位置距离最近的0的距离，两种思路，一是以每个1为中心BFS遍历，第一次出现0的层深即为距离，二是典型的DP思维，每个1到0的最近距离是它的四个邻居（上下左右）到最近的0的距离的最小值加一。

    DP的实现代码

    ```cpp
    vector<vector<int>> updateMatrix(vector<vector<int>> &matrix)
    {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> dist(m, vector<int>(n, 1e4 + 1)); // 1e4是矩阵中元素的最大数量，题目给定
        // 左上角到右下角
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (matrix[i][j] == 1)
                {
                    if (i > 0)
                    {
                        dist[i][j] = min(dist[i][j], dist[i - 1][j] + 1);
                    }
                    if (j > 0)
                    {
                        dist[i][j] = min(dist[i][j], dist[i][j - 1] + 1);
                    }
                }
                else
                {
                    dist[i][j] = 0;
                }
            }
        }
        // 从右下角到左上角
        for (int i = m - 1; i >= 0; i--)
        {
            for (int j = n - 1; j >= 0; j--)
            {
                if (matrix[i][j] == 1)
                {
                    if (i < m - 1)
                    {
                        dist[i][j] = min(dist[i][j], dist[i + 1][j] + 1);
                    }
                    if (j < n - 1)
                    {
                        dist[i][j] = min(dist[i][j], dist[i][j + 1] + 1);
                    }
                }
                else
                {
                    dist[i][j] = 0;
                }
            }
        }

        return dist;
    }
    ```

- [543. Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)

    - 递归方法，DFS，时间复杂度$O(n)$

    ```cpp
    int ans = 1;
    int depth(TreeNode *node)
    {
        int ret = 0;
        if (node)
        {
            int left = depth(node->left);
            int right = depth(node->right);
            ans = max(left + right + 1, ans);
            ret = max(left, right) + 1;
        }
        return ret;
    }
    int diameterOfBinaryTree(TreeNode *root)
    {
        depth(root);
        return ans - 1;
    }
    ```

    - 递归的不同实现，不使用函数以外的数据，时间复杂度$O(n)$

    ```cpp
    class Solution
    {
    private:
        vector<int> dfs(TreeNode *root)
        {
            // 返回root的最大深度和直径
            vector<int> ret{0, 0};
            if (root)
            {
                vector<int> left = dfs(root->left);
                vector<int> right = dfs(root->right);
                // root的最大深度是左子树的深度和右子树的深度的最大值加1（1是当前节点root）
                ret[0] = max(left[0], right[0]) + 1;
                // root的直径是max(左子树直径，右子树直径，经过当前节点root的一条最长路径)
                ret[1] = max(max(left[1], right[1]), left[0] + right[0]);
            }
            return ret;
        }

    public:
        int diameterOfBinaryTree(TreeNode *root)
        {
            return dfs(root)[1];
        }
    };
    ```

- [547. Friend Circles](https://leetcode.com/problems/friend-circles/)

	并查集的建立与使用，时间复杂度$O(n^2log(n)))$

	```cpp
	class Solution
	{
	private:
		vector<int> father;
		int find(int x)
		{
			return father[x] == x ? x : (father[x] = find(father[x]));
		}

	public:
		int findCircleNum(vector<vector<int>> &M)
		{
			// initialization of the union find
			int n = M.size(), count = n;
			father.reserve(n);
			for (auto i = 0; i < n; i++)
			{
				father[i] = i;
			}
			// update of the union find
			for (auto i = 0; i < n; i++)
			{
				for (auto j = 0; j < i; j++)
				{
					if (M[i][j])
					{
						int fi = find(i), fj = find(j);
						if (fi != fj)
						{
							count--, father[fi] = fj;
						}
					}
				}
			}
			return count;
		}
	};
	```

- [554. 砖墙](https://leetcode-cn.com/problems/brick-wall/)

    使用HashMap统计每一块砖的右边缘（缝隙处）的长度数量即可，时间复杂度$O(n*m)$，其中$n*m$是所有砖块的总量

    ```cpp
	int leastBricks(vector<vector<int>> &wall)
	{
		unordered_map<int, int> count;
		for (auto &row : wall)
		{
			const int n = row.size() - 1;
			for (int i = 0, base = 0; i < n; i++)
			{
				base += row[i];
				count[base]++;
			}
		}
		int max_count = 0;
		for (auto &[r, freq] : count)
		{
			max_count = max(max_count, freq);
		}
		return wall.size() - max_count;
	}
    ```

- [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)

    在给定数组nums中，求其和为K的子数组的数量，如果数组中不含有负数，可以使用滑动窗口计算

    - 计算数组nums的前缀和prefix，即$prefix[i]=\sum_{j=0}^{i}nums_j$，然后从0到j遍历，每次从prefix[i]中减掉前面的一个数，即可在$O(n^2)$时间内实现

    ```cpp
    int subarraySum(vector<int> &nums, int k)
    {
        int ret = 0, prefix_sum = 0;
        for (int i = 0; i < nums.size(); i++)
        {
            prefix_sum += nums[i];
            int base = prefix_sum;
            for (int j = 0; j <= i; j++)
            {
                if (base == k)
                {
                    ret++;
                }
                base -= nums[j];
            }
        }
        return ret;
    }
    ```

    - 结合hashmap的$O(1)$存取效率与前缀和方式实现$O(n)$时间复杂度，其基本原理为如果前缀和$prefix[j]-prefix[i]=K$，则$\sum_{r=i+1}^{j}nums[r]=K$

    ```cpp
    int subarraySum(vector<int> &nums, int k)
    {
        int ret = 0, prefix_sum = 0;
        unordered_map<int, int> prefix_count;
        prefix_count[0] = 1;
        for (int i = 0; i < nums.size(); i++)
        {
            prefix_sum += nums[i];
            if (prefix_count.find(prefix_sum - k) != prefix_count.end())
            {
                ret += prefix_count[prefix_sum - k];
            }
            if (prefix_count.find(prefix_sum) != prefix_count.end())
            {
                prefix_count[prefix_sum]++;
            }
            else
            {
                prefix_count[prefix_sum] = 1;
            }
        }
        return ret;
    }
    ```

- [561](https://leetcode.com/problems/array-partition-i/)
    
    2n个给定范围的数据划分成n组使得每组最小值求和最大，基本思路是对所有数排序后对奇数位置上的数求和即可，这里类似于NMS非极大值抑制的思路，主要的时间复杂度在排序上。
    - 基本想法是quick sort，时间复杂度$O(nlog(n))$
    - 本题给出了数据范围，可以bucket sort，时间复杂度$O(n)$，但是需要$O(N)$的额外空间

- [567. 字符串的排列](https://leetcode-cn.com/problems/permutation-in-string/)

    滑动窗口，以s1的长度为窗口大小在s2上滑动，比较s2在窗口内的部分是否为s1的全排列即可，时间复杂度$O(n)$，其中$n=s2.length()$

    ```cpp
	bool checkInclusion(string s1, string s2)
	{
		const int length = 26, m = s1.length(), n = s2.length();
		if (m <= n)
		{
			vector<int> count_s1(length, 0), count_s2(length, 0);
			for (auto &ch : s1)
			{
				count_s1[static_cast<int>(ch - 'a')]++;
			}
			for (int i = 0; i < m - 1; i++)
			{
				count_s2[static_cast<int>(s2[i] - 'a')]++;
			}
			for (int i = m - 1; i < n; i++)
			{
				count_s2[static_cast<int>(s2[i] - 'a')]++;
				bool anagram = true;
				for (int i = 0; i < length; i++)
				{
					if (count_s1[i] != count_s2[i])
					{
						anagram = false;
						break;
					}
				}
				if (anagram)
				{
					return true;
				}
				count_s2[static_cast<int>(s2[i - m + 1] - 'a')]--;
			}
		}
		return false;
	}
    ```

- [572. Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/)

    判断二叉树t是否是二叉树s的子树，递归地，s和t完全相同或者t是s->left的子树或者t是s->right的子树

    ```cpp
    bool same(TreeNode *s, TreeNode *t)
    {
        bool ret = false;
        if (!s && !t)
        {
            ret = true;
        }
        else if (s && t && s->val == t->val)
        {
            ret = same(s->left, t->left) && same(s->right, t->right);
        }
        return ret;
    }
    bool isSubtree(TreeNode *s, TreeNode *t)
    {
        bool ret = false;
        if (!t)
        {
            ret = true; // 空树是任何树的子树
        }
        else
        {
            if (!s)
            {
                ret = false; // t非空而s空，t不可能是s的子树
            }
            else
            {
                ret = same(s, t) || isSubtree(s->left, t) || isSubtree(s->right, t);
            }
        }
        return ret;
    }
    ```

- [581. Shortest Unsorted Continuous Subarray](https://leetcode.com/problems/shortest-unsorted-continuous-subarray/)

    - 将nums排序后比较与原数组不同的部分长度即可，时间复杂度$O(nlog(n))$

    ```cpp
    int findUnsortedSubarray(vector<int> &nums)
    {
        const int count = nums.size();
        vector<int> sorted_nums = nums;
        sort(sorted_nums.begin(), sorted_nums.end());
        int left = 0, right = count - 1;
        while (left <= right && sorted_nums[left] == nums[left])
        {
            left++;
        }
        while (left <= right && sorted_nums[right] == nums[right])
        {
            right--;
        }
        return right - left + 1;
    }
    ```

    - 寻找左侧已排序的部分长度和右侧已排序部分长度，剩余的中间长度即为最长未排序部分，时间复杂度$O(n)$

    ```cpp
    int findUnsortedSubarray(vector<int> &nums)
    {
        const int count = nums.size();
        vector<int> left_max = nums, right_min = nums;
        for (auto i = 1; i < count; i++)
        {
            left_max[i] = max(left_max[i - 1], left_max[i]), right_min[count - i - 1] = min(right_min[count - i], right_min[count - i - 1]);
        }
        int left = 0, right = count - 1;
        while (left <= right && right_min[left] == nums[left])
        {
            left++;
        }
        while (left <= right && left_max[right] == nums[right])
        {
            right--;
        }
        return right - left + 1;
    }
    ```

- [583. 两个字符串的删除操作](https://leetcode-cn.com/problems/delete-operation-for-two-strings/)

    删除最少数量的字符使得两个字符串相等即可，因为删除完成之后要保证相等，因此保留下来的是最长公共子串(LCS)，递归求解LCS会TLE，需要二维DP或者一维DP，时间复杂度$O(m*n)$

    ```cpp
    int minDistance(string word1, string word2)
	{
		const int m = word1.length(), n = word2.length();
		vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				dp[i + 1][j + 1] = word1[i] == word2[j] ? dp[i][j] + 1 : max(dp[i + 1][j], dp[i][j + 1]);
			}
		}
		return m + n - dp[m][n] * 2;
	}
    ```

- [590. N-ary Tree Postorder Traversal](https://leetcode.com/problems/n-ary-tree-postorder-traversal/)

    N叉树的后续遍历(post-order traversal)

    - 递归写法

    ```cpp
    class Solution
    {
    private:
        void recursive_postorder(Node *root, vector<int> &ret)
        {
            if (root)
            {
                for (auto &&node : root->children)
                {
                    recursive_postorder(node, ret);
                }
                ret.push_back(root->val);
            }
        }

    public:
        vector<int> postorder(Node *root)
        {
            vector<int> ret;
            recursive_postorder(root, ret);
            return ret;
        }
    };
    ```

    - 迭代写法

    ```cpp
	vector<int> postorder(Node *root)
	{
		vector<int> ret;
		stack<Node *> st{{root}};
		Node *cur;
		while (!st.empty())
		{
			cur = st.top();
			st.pop();
			if (cur)
			{
				ret.push_back(cur->val);
				for (auto &node : cur->children)
				{
					st.push(node);
				}
			}
		}
		reverse(ret.begin(), ret.end());
		return ret;
	}
    ```

- [593](https://leetcode.com/problems/valid-square/)

    验证给定的任意四个点是否可以组成一个正方形，思路如下：
    - [1] 任意选定一个点p1，在剩余三个点中选择距离p1最远的点作为p2，另外两个点作为p3和p4
    - [2] 首先验证连线(对角线)p1p2和p3p4垂直，这样可以组成筝形或者菱形
    - [3] 验证对角线等长，这样排除菱形
    - [4] 验证筝形的相邻两条边相等，即p1p2和p2p3相等，则构成正方形

    ```cpp
    long long edge_length_square(vector<int> &p1, vector<int> &p2)
    {
        return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]);
    }
    bool validSquare(vector<int> &p1, vector<int> &p2, vector<int> &p3, vector<int> &p4)
    {
        bool ret = true;
        // 知道距离p1最远的点构成一条线，另外两个点构成一条线
        long long dist_12 = edge_length_square(p1, p2), dist_13 = edge_length_square(p1, p3), dist_14 = edge_length_square(p1, p4);
        vector<int> indexs;
        if (dist_12 > dist_13 && dist_12 > dist_14)
        {
            indexs = vector<int>{1, 2, 3, 4};
        }
        else if (dist_13 > dist_12 && dist_13 > dist_14)
        {
            indexs = vector<int>{1, 3, 2, 4};
        }
        else if (dist_14 > dist_12 && dist_14 > dist_13)
        {
            indexs = vector<int>{1, 4, 2, 3};
        }
        else
        {
            ret = false;
        }
        if (ret)
        {
            // 验证两条线等长且互相垂直
            vector<vector<int>> nodes{p1, p2, p3, p4};
            p1 = nodes[indexs[0] - 1], p2 = nodes[indexs[1] - 1], p3 = nodes[indexs[2] - 1], p4 = nodes[indexs[3] - 1];
            ret = ret && (edge_length_square(p1, p2) == edge_length_square(p3, p4));
            ret = ret && ((p4[0] - p3[0]) * (p2[0] - p1[0]) + (p4[1] - p3[1]) * (p2[1] - p1[1]) == 0);
            // 验证相邻两条边相等，排除筝形
            ret = ret && (edge_length_square(p1, p3) == edge_length_square(p2, p3));
        }
        return ret;
    }
    ```

    四个经典的测试样例

    ```cpp
    [1,1]
    [5,3]
    [3,5]
    [7,7]
    [2,1]
    [1,2]
    [0,0]
    [2,0]
    [1,0]
    [0,1]
    [0,-1]
    [-1,0]
    [0,0]
    [1,1]
    [1,0]
    [0,1]
    ```

- [599. 两个列表的最小索引总和](https://leetcode-cn.com/problems/minimum-index-sum-of-two-lists/)

    - 暴利遍历两个数组，时间复杂度$O(m*n)$

    ```cpp
	vector<string> findRestaurant(vector<string> &list1, vector<string> &list2)
	{
		vector<string> ret;
		int m = list1.size(), n = list2.size();
		int min_idx_sum = m + n;
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				if (i + j <= min_idx_sum && list1[i].compare(list2[j]) == 0)
				{
					if (i + j < min_idx_sum)
					{
						min_idx_sum = i + j;
						ret.clear();
					}
					ret.emplace_back(list1[i]);
				}
			}
		}
		return ret;
	}
    ```

    - hashmap记录第二个数组中元素的下标并查询，时间复杂度$O(m+n)$

    ```cpp
	vector<string> findRestaurant(vector<string> &list1, vector<string> &list2)
	{
		unordered_map<string, int> stringToIdx;
		vector<string> ret;
		int m = list1.size(), n = list2.size();
		for (int i = 0; i < m; i++)
		{
			stringToIdx[list1[i]] = i;
		}
		int min_idx_sum = m + n;
		for (int i = 0; i < n; i++)
		{
			auto it = stringToIdx.find(list2[i]);
			if (it != stringToIdx.end())
			{
				if (it->second + i < min_idx_sum)
				{
					min_idx_sum = it->second + i;
					ret.clear();
					ret.emplace_back(list2[i]);
				}
				else if (it->second + i == min_idx_sum)
				{
					ret.emplace_back(list2[i]);
				}
			}
		}
		return ret;
	}
    ```

    - 对两个数组排序后寻找相同的值，时间复杂度$O(max(m,n)log(max(m,n)))$

    ```cpp
	vector<string> findRestaurant(vector<string> &list1, vector<string> &list2)
	{
		vector<string> ret;
		int m = list1.size(), n = list2.size();
		int min_idx_sum = m + n;
		vector<pair<string, int>> list1_pair(m), list2_pair(n);
		for (int i = 0; i < m; i++)
		{
			list1_pair[i] = make_pair(list1[i], i);
		}
		for (int i = 0; i < n; i++)
		{
			list2_pair[i] = make_pair(list2[i], i);
		}
		sort(list1_pair.begin(), list1_pair.end());
		sort(list2_pair.begin(), list2_pair.end());
		int i = 0, j = 0;
		while (i < m && j < n)
		{
			int x = list1_pair[i].first.compare(list2_pair[j].first);
			if (x == 0)
			{
				if (list1_pair[i].second + list2_pair[j].second < min_idx_sum)
				{
					min_idx_sum = list1_pair[i].second + list2_pair[j].second;
					ret.clear();
					ret.emplace_back(list1_pair[i].first);
				}
				else if (list1_pair[i].second + list2_pair[j].second == min_idx_sum)
				{
					ret.emplace_back(list1_pair[i].first);
				}
                i++, j++;
			}
			else
			{
				x < 0 ? i++ : j++;
			}
		}
		return ret;
	}
    ```
