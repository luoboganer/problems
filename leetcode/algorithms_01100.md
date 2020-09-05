# 1001-1100

- [1002](https://leetcode.com/problems/find-common-characters/)

    用两个长度为26的数组来统计两个单词中每个字母出现的次数，然后共同字母的次数为两个单词分别统计结果中的较小值，如果有一方为0则表示没有共同出现过，时间复杂度$O(n)$，其中n是所有单词长度的和，即所有字母的个数。

    ```cpp
    vector<string> commonChars(vector<string> &A)
    {
        // string s = A[0];
        // sort(s.begin(), s.end());
        // for (int k = 1; k < A.size(); k++)
        // {
        // 	string x, t = A[k];
        // 	sort(t.begin(), t.end());
        // 	int i = 0, j = 0;
        // 	while (i < s.length() && j < t.length())
        // 	{
        // 		if (s[i] == t[j])
        // 		{
        // 			x += s[i];
        // 			i++;
        // 			j++;
        // 		}
        // 		else if (s[i] < t[j])
        // 		{
        // 			i++;
        // 		}
        // 		else
        // 		{
        // 			j++;
        // 		}
        // 	}
        // 	s = x;
        // }
        // vector<string> ret;
        // for (string::iterator it = s.begin(); it != s.end(); it++)
        // {
        // 	string t;
        // 	t += *it;
        // 	ret.push_back(t);
        // }
        // return ret;

        // method 2, O(n)
        const int length_of_lowercases = 26;
        vector<int> count(length_of_lowercases, numeric_limits<int>::max());
        for (auto word : A)
        {
            vector<int> cnt(length_of_lowercases, 0);
            for (auto c : word)
            {
                cnt[c - 'a']++;
            }
            for (int i = 0; i < length_of_lowercases; i++)
            {
                count[i] = min(count[i], cnt[i]);
            }
        }
        vector<string> ret;
        for (int i = 0; i < length_of_lowercases; i++)
        {
            for (int j = 0; j < count[i]; j++)
            {
                ret.push_back(string(1, i + 'a'));
            }
        }
        return ret;
    }
    ```

- [1008](https://leetcode.com/problems/construct-binary-search-tree-from-preorder-traversal/)

    从二叉搜索树BST的先序遍历preorder开始重建BST

    - 以preorder[0]为分界线将后面preorder后面的部分划分成两部份，左半部分递归重建左子树，右半部分递归重建右子树，时间复杂度$O(nlog(n))$

    ```cpp
    TreeNode *bstFromPreorder(vector<int> &preorder)
    {
        if (preorder.size() > 0)
        {
            TreeNode *root = new TreeNode(preorder[0]);
            int index = 1;
            while (index < preorder.size() && preorder[0] > preorder[index])
            {
                index++;
            }
            root->left = bstFromPreorder(vector<int>&(preorder.begin() + 1, preorder.begin() + index));
            root->right = bstFromPreorder(vector<int>&(preorder.begin() + index, preorder.end()));
            return root;
        }
        else
        {
            return NULL;
        }
    }
    ```

    - 重写函数为bstFromPreorder(vector<int>& preorder,int boundary)，设定一个界boundary，从preorder[i]开始用下标i遍历preorder，小于界boundary的递归建立左子树，大于界的递归建立右子树，可以实现线性时间复杂度$O(n)$

    ```cpp
    int i = 0;
    TreeNode *bstFromPreorder(vector<int> &preorder)
    {
        return bstFromPreorder(preorder, numeric_limits<int>::max());
    }
    TreeNode *bstFromPreorder(vector<int> &preorder, int boundary)
    {
        if (!(i < preorder.size()) || preorder[i] > boundary)
        {
            return NULL;
        }
        else
        {
            TreeNode *root = new TreeNode(preorder[i++]);
            root->left = bstFromPreorder(preorder, root->val);
            root->right = bstFromPreorder(preorder, boundary);
            return root;
        }
    }
    ```

    - 栈实现，时间复杂度$O(n)$，遍历数组中的每个元素item，找到第一个比item小的数p，然后把item挂到p的右孩子，因此需要维护一个从栈底到栈顶递减的栈序列，从而为数组迭代过程中的每个数item找到第一个比它小的数p，即栈顶元素；如果栈中不存在p比item小，则item当前最小，成为栈顶元素的左孩子。

    ```cpp
    TreeNode *bstFromPreorder(vector<int> &preorder)
    {
        TreeNode *cur_root = new TreeNode(numeric_limits<int>::max());
        stack<TreeNode *> st;
        st.push(cur_root);
        for (auto item : preorder)
        {
            TreeNode *cur = new TreeNode(item), *p = nullptr;
            while (st.top()->val < item)
            {
                p = st.top();
                st.pop();
            }
            if (p)
            {
                p->right = cur;
            }
            else
            {
                st.top()->left = cur;
            }
            st.push(cur);
        }
        return cur_root->left;
    }
    ```

- [1014. Best Sightseeing Pair](https://leetcode.com/problems/best-sightseeing-pair/)

    对于给定数组中的所有数对$A[i],A[j]$求$max(A[i]+A[j]+i-j)$

    - 朴素的遍历所有可能的数对，时间复杂度$O(n^2)$，结果正确但是$\color{red}{TLE}$

    ```cpp
    int maxScoreSightseeingPair(vector<int> &A)
    {
        int ret = 0;
        for (int i = 0; i < A.size(); i++)
        {
            for (int j = i + 1; j < A.size(); j++)
            {
                ret = max(ret, A[i] + A[j] + i - j);
            }
        }
        return ret;
    }
    ```

    - one pass scan，时间复杂度$O(n)$，参考[ref](https://leetcode.com/problems/best-sightseeing-pair/discuss/260850/JavaC%2B%2BPython-One-Pass)

    ```cpp
    int maxScoreSightseeingPair(vector<int> &A)
    {
        int ret = 0, cur = 0;
        for (int i = 0; i < A.size(); i++)
        {
            ret = max(ret, cur + A[i]);
            cur = max(cur, A[i]) - 1;
        }
        return ret;
    }
    ```

- [1015. Smallest Integer Divisible by K](https://leetcode.com/problems/smallest-integer-divisible-by-k/)

    求全部有1组成的、可以被K整除的十进制数字的位宽，因为是位宽而无需求出这个数字，因此可以在$ans=ans*10+1$的过程中不断对K取模以限制ans在int的范围内

    ```cpp
    int smallestRepunitDivByK(int K)
    {
        unsigned int ans = 0;
        for (int i = 0; i < K; i++)
        {
            ans = (ans * 10 + 1) % K;
            if (ans % K == 0)
            {
                return i + 1;
            }
        }
        return -1;
    }
    ```

- [1019](https://leetcode.com/problems/next-greater-node-in-linked-list/)

    用栈可以实现$O(n)$时间复杂度，即对数组从右往左遍历的过程中保持栈顶$st[i]<st[i-1]$，从栈底到栈顶是严格递增的顺序

    ```cpp
    vector<int> nextLargerNodes(ListNode *head)
    {
        // method 1, O(n^2)

        // vector<int> ret;
        // while (head)
        // {
        // 	ListNode *cur = head->next;
        // 	int base = 0;
        // 	while (cur->next)
        // 	{
        // 		if (head->val < cur->val)
        // 		{
        // 			base = cur->val;
        // 			break;
        // 		}
        // 		else
        // 		{
        // 			cur = cur->next;
        // 		}
        // 	}
        // 	ret.push_back(base);
        // 	head = head->next;
        // }
        // return ret;

        // method 2, O(n), using stack

        vector<int> ret, st;
        while (head)
        {
            // convert linked list to array
            ret.push_back(head->val);
            head = head->next;
        }
        for (int i = ret.size() - 1; i >= 0; i--)
        {
            // maintain the stack is decreasing from st.top() to st.bottom()
            while (!st.empty() && st.back() <= ret[i])
            {
                st.pop_back();
            }
            st.push_back(ret[i]);
            // st[i-1] is the next greater number of st[i]
            ret[i] = st.size() > 1 ? st[st.size() - 2] : 0;
        }
        return ret;
    }
    ```

- [1024](https://leetcode.com/problems/video-stitching/)

    一开始设定当前右端点$cur_right$，然后按照贪心策略寻找左端点小于等于当前右端点(保证可以连接而没有断点)且右端点最远(贪心原则，以便使用最少的视频片段)的没有用过的视频片段，直到所有视频片段被用完或者当前右端点$cur_right$超过了总时间长度要求$T$。

- [1026. Maximum Difference Between Node and Ancestor](https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/)

    给定一颗树，自顶向下的找到全局最小值min和全局最大值max，返回$max-min$即可

    ```cpp
    int helper(TreeNode *root, int max_value, int min_value)
    {
        int ret = max_value - min_value;
        if (root)
        {
            max_value = max(max_value, root->val), min_value = min(min_value, root->val);
            ret = max(ret, max(helper(root->left, max_value, min_value), helper(root->right, max_value, min_value)));
        }
        return ret;
    }
    int maxAncestorDiff(TreeNode *root)
    {
        return helper(root, root->val, root->val);
    }
    ```

    几组典型测试数据

    ```cpp
    [8,3,10,1,6,null,14,null,null,4,7,13]
    [2,5,0,null,null,4,null,null,6,1,null,3]
    [1,null,2,null,0,3]
    [2,null,0,1]
    ```

- [1028. Recover a Tree From Preorder Traversal](https://leetcode.com/problems/recover-a-tree-from-preorder-traversal/)

    使用一个辅助栈来存储所有当前节点cur的父/祖节点，则栈的大小即是当前节点的深度，对新的数字构成节点后如果cur左子树为空，则优先挂到左子树，否则挂到右子树，时间复杂度$O(n),n=S.length$

    ```cpp
    TreeNode *recoverFromPreorder(string S)
    {
        TreeNode *auxiliary_root = new TreeNode(0), *cur = auxiliary_root;
        if (S.length() > 0)
        {
            stack<TreeNode *> st;
            int depth = 0;
            int i = 0, length = S.length();
            while (i < length)
            {
                if (S[i] == '-')
                {
                    depth = 0;
                    while (i < length && S[i] == '-')
                    {
                        depth++, i++;
                    }
                }
                else
                {
                    int base = 0;
                    while (i < length && S[i] != '-')
                    {
                        base = base * 10 + (int)(S[i++] - '0');
                    }
                    TreeNode *temp = new TreeNode(base);
                    while (st.size() != depth)
                    {
                        cur = st.top(), st.pop();
                    }
                    if (cur->left)
                    {
                        cur->right = temp, st.push(cur);
                        cur = cur->right;
                    }
                    else
                    {
                        cur->left = temp, st.push(cur);
                        cur = cur->left;
                    }
                }
            }
        }
        return auxiliary_root->left;
    }
    ```

- [1031. Maximum Sum of Two Non-Overlapping Subarrays](https://leetcode.com/problems/maximum-sum-of-two-non-overlapping-subarrays/)

    在给定数组A中选择两个长度分别为L和M的连续、无重叠子数组，使这两个子数组的和最大。用两个辅助数组subArraySumL、subArraySumM分别表示当前下标左侧L、M个数的和，即表示题目要求的两个长度分别为L和M的连续子数组的元素之和，然后在针对subArraySumL中每个值（下标i），在subArraySumM中找到下标i - M - 1及其左侧的最大值left（构成长度为M的数组）、下标i + L及其右侧的最大值right（构成长度为L的子数组）,这样$ret=max(ret,subArraySumL[i]+(max(let,right))$即为所求的最终值，时间复杂度$O(A.length)$

    ```cpp
    vector<int> getSubArraySum(vector<int> &nums, int length)
    {
        vector<int> ret(nums.size(), -1);
        int i = 0, base = 0;
        while (i < length)
        {
            base += nums[i++];
        }
        ret[i - 1] = base;
        while (i < nums.size())
        {
            base += nums[i] - nums[i - length];
            ret[i++] = base;
        }
        return ret;
    }
    int maxSumTwoNoOverlap(vector<int> &A, int L, int M)
    {
        const int length = A.size();
        vector<int> subArraySumL = getSubArraySum(A, L), subArraySumM = getSubArraySum(A, M);
        vector<int> leftMostL = subArraySumL, rightMostL = subArraySumL;
        for (int i = 0; i < length - 1; i++)
        {
            leftMostL[i + 1] = (leftMostL[i + 1] == -1) ? leftMostL[i + 1] : max(leftMostL[i], leftMostL[i + 1]);
            rightMostL[length - i - 2] = (rightMostL[length - i - 2] == -1) ? rightMostL[length - i - 2] : max(rightMostL[length - i - 2], rightMostL[length - i - 1]);
        }
        int ret = 0;
        for (int i = M - 1; i < length; i++)
        {
            int left = (i - M - 1 >= 0) ? leftMostL[i - M - 1] : 0, right = (i + L < length) ? rightMostL[i + L] : 0;
            ret = max(ret, subArraySumM[i] + max(left, right));
        }
        return ret;
    }
    ```

- [1032](https://leetcode.com/problems/stream-of-characters/)

    本题主要练习字典树的的构建与查询，tire tree是一种高效的单词存储与查询数据结构，比如可以完成IDE的代码自动提示语补全功能，[here](https://blog.csdn.net/v_july_v/article/details/6897097)有相关博客。

- [1035](https://leetcode.com/problems/uncrossed-lines/)

    类似于LCS最长公共子序列的问题，二维dp或者一维dp均可，注意问题中隐藏的dp思维。

- [1042](https://leetcode.com/problems/flower-planting-with-no-adjacent/)

    贪心思想，时间复杂度$O(n)$：经典的染色问题，在给定四种颜色且保证有满足条件的答案时可以确保每个节点的连通节点小于等于三个，因此外层循环遍历每个节点i，内层循环遍历该节点链接的所有节点j，使用j尚未使用的颜色染给i即可。

    ```cpp
    vector<int> gardenNoAdj(int N, vector<vector<int>>& paths) {
        vector<vector<int>> connections(N);
        vector<int> ans(N,0);
        for (int i = 0; i < paths.size(); i++)
        {
            connections[paths[i][0]-1].push_back(paths[i][1]-1);
            connections[paths[i][1]-1].push_back(paths[i][0]-1);
        }
        for (int i = 0; i < N; i++)
        {
            int colors[5]={0};
            for (int j : connections[i])
            {
                // 与i相连的节点j最多有3个
                colors[ans[j]]=1;
                // 表示颜色colors[ans[j]]已经被用了
            }
            for (int color = 1; color <= 4; color++)
            {
                if(!colors[color]){
                    // 颜色color尚未被与i相连的节点使用
                    ans[i]=color;
                    break;
                }
            }
        }
        return ans;
    }
    ```

- [1043. Partition Array for Maximum Sum](https://leetcode.com/problems/partition-array-for-maximum-sum/)

    动态规划，dp[i]表示可以从dp[0,1,...,i]得到的最大和，则$dp[i]=\max_{1 \le size \le K }(dp[i-size]+max_{i-szie \le j \le i}A[j])$，时间复杂度$O(K*A.length)$

    ```cpp
    int maxSumAfterPartitioning(vector<int> &A, int K)
    {
        const int n = A.size();
        vector<int> dp(n + 1, 0);
        for (int i = 0; i < n; i++)
        {
            int cur_max_value = A[i];
            for (int size = 1; size <= i + 1 && size <= K; size++)
            {
                cur_max_value = max(cur_max_value, A[i - size + 1]);
                dp[i + 1] = max(dp[i + 1], dp[i - size + 1] + cur_max_value * size);
            }
        }
        return dp.back();
    }
    ```

- [1044. Longest Duplicate Substring](https://leetcode.com/problems/longest-duplicate-substring/)

    - binary search + rabin karp (rolling hash)，时间复杂度$O(nlog(n))$，空间复杂度$O(n)$
        - binary search，如果有长度为n的子串duplicated，那么一定有长度为k($1<k<n$)的子串duplicated，因此可以二分搜索这个可能的长度
        - rabin karp，通过rolling hash的办法快速比较滑动窗口内的字符串是否相同

	```cpp
	class Solution
	{
	private:
		bool compare_equal(vector<int> &nums, int p1, int p2, int length)
		{
			for (auto i = 0; i < length; i++)
			{
				if (nums[i + p1] != nums[i + p2])
				{
					return false;
				}
			}
			return true;
		}

	public:
		string longestDupSubstring(string S)
		{
			const int n = S.length(), mod = 1e9 + 7;
			long long base = 26;
			vector<long long> powers(n, 1);
			// 提前计算好可能用到的base的幂
			for (auto i = 1; i < n; i++)
			{
				powers[i] = (powers[i - 1] * base) % mod;
			}
			vector<int> nums(n);
			for (int i = 0; i < n; i++)
			{
				nums[i] = (int)(S[i] - 'a');
			}
			vector<int> result{-1, 0};
			int min_length = 1, max_length = n;
			while (min_length < max_length)
			{
				int length = min_length + (max_length - min_length) / 2;
				bool flag = false;
				// 验证是否有长度为length的重复子串
				unordered_map<int, vector<int>> map;
				long long current = 0;
				for (auto i = 0; i < length; i++)
				{
					current = (current * base + nums[i]) % mod;
				}
				map[current] = vector<int>{0};
				/**
				* 这里是长度为length的以i为结尾下标的字符串s[i-length+1,i]是否存在重复，一旦发现重复则无需继续遍历i
				* !flag的作用是在flag为true是退出循环，达到剪枝的目的
				*/
				for (auto i = length; !flag && i < n; i++)
				{
					int cur_start_index = i - length + 1;
					current = ((current - nums[i - length] * powers[length - 1]) % mod + mod) % mod;
					current = (current * base + nums[i]) % mod;
					auto it = map.find(current);
					if (it == map.end())
					{
						map[current] = vector<int>{i - length + 1};
						// i - length + 1 是当前rolling hash值为current的长度为length的子串的起点下标index
					}
					else
					{
						// 发现了hash值的重复
						for (auto &&start : map[current])
						{
							if (compare_equal(nums, cur_start_index, start, length) == true)
							{
								// 发现了重复字符串
								if (length > result[1])
								{
									// 重复字符串的长度比当前值高
									result[0] = cur_start_index;
									result[1] = length;
								}
								flag = true;
								break; // 剪枝，已经发现了长度为length的重复字符串，后面hash值相同长度相同的子串也就没必要再去检测
							}
						}
						map[current].push_back(cur_start_index);
					}
				}
				flag ? min_length = length + 1 : max_length = length;
			}
			return result[0] == -1 ? "" : S.substr(result[0], result[1]);
		}
	};
	```

- [1049](https://leetcode.com/problems/last-stone-weight-ii/)

    本题需要把数组stones分为两部分$A,B$使得$min(abs(sum(A)-sum(B)))$，是经典的$0,1$背包问题。

    ```cpp
    int lastStoneWeightII(vector<int>& stones) {
        bitset<1501> dp{1};
        int sum_stones=0,ans=numeric_limits<int>::max();
        // dp[x]表示可以在stones中选择一部分数使其和为x
        for(auto const &it:stones){
            for (int i = 1500; i >= it; i--)
            {
                dp[i]=dp[i]|dp[i-it];
            }
            sum_stones+=it;
        }
        for (int i = 0; i <= 1500; i++)
        {
            ans=min(ans,abs(sum_stones-dp[i]*2*i));
        }
        return ans;
    }
    ```

- [1072. Flip Columns For Maximum Number of Equal Rows](https://leetcode.com/problems/flip-columns-for-maximum-number-of-equal-rows/)

	- 假设$ith row$在flip之后与$jth row$完全一样，在flip之前也是完全一样的，假设$ith row$在flip之后与$kth row$完全相反，则在flip之前也是完全相反的，因此问题转化为寻找与$ith row$完全相反或完全相同的行的数量，时间复杂度$O(m^2*n)$，leetcode评测机$\color{red}{TLE}$

	```cpp
	class Solution
	{
	private:
		bool is_same(vector<int> &a, vector<int> &b, int length)
		{
			for (auto i = 0; i < length; ++i)
			{
				if (a[i] != b[i])
				{
					return false;
				}
			}
			return true;
		}

	public:
		int maxEqualRowsAfterFlips(vector<vector<int>> &matrix)
		{
			int ret = 0;
			if (matrix.size() > 0 && matrix[0].size() > 0)
			{
				int m = matrix.size(), n = matrix[0].size();
				for (auto i = 0; i < m; i++)
				{
					int cnt = 0;
					// 寻找与第i行完全相同或者完全不同的行的数量
					for (auto k = 0; k < m; k++)
					{
						vector<int> flip(n);
						for (int j = 0; j < n; j++)
						{
							flip[j] = 1 - matrix[k][j];
						}
						if (is_same(matrix[i], matrix[k], n) || is_same(matrix[i], flip, n))
						{
							cnt++;
						}
					}
					ret = max(cnt, ret);
				}
			}
			return ret;
		}
	};
	```

	- hashmap优化，时间复杂度$O(m*n)$

	```cpp
	int maxEqualRowsAfterFlips(vector<vector<int>> &matrix)
	{
		int ret = 0;
		if (matrix.size() > 0 && matrix[0].size() > 0)
		{
			int m = matrix.size(), n = matrix[0].size();
			unordered_map<string, int> count;
			for (auto i = 0; i < m; i++)
			{
				string cur;
				int top = matrix[i][0];
				for (auto j = 1; j < n; j++)
				{
					(matrix[i][j] == top) ? cur.push_back('1') : cur.push_back('0');
				}
				count[cur]++;
			}
			for (auto &&[item, val] : count)
			{
				ret = max(ret, val);
			}
		}
		return ret;
	}
	```

- [1079. Letter Tile Possibilities](https://leetcode.com/problems/letter-tile-possibilities/)

    可以用递归的DFS实现，模拟所有可能被组成的字符串

    - 引用传值实现

    ```cpp
    void dfs(int &ret, vector<int> &count)
    {
        for (int i = 0; i < 26; i++)
        {
            if (count[i] != 0)
            {
                ret++, count[i]--;
                dfs(ret, count);
                count[i]++;
            }
        }
    }
    int numTilePossibilities(string tiles)
    {
        vector<int> count(26, 0);
        for (auto &&ch : tiles)
        {
            count[(int)(ch - 'A')]++;
        }
        int ret = 0;
        dfs(ret, count);
        return ret;
    }
    ```

    - 函数返回值实现

    ```cpp
    int dfs(vector<int> &count)
    {
        int ret = 0;
        for (int i = 0; i < 26; i++)
        {
            if (count[i] != 0)
            {
                ret++, count[i]--;
                ret += dfs(count);
                count[i]++;
            }
        }
        return ret;
    }
    int numTilePossibilities(string tiles)
    {
        vector<int> count(26, 0);
        for (auto &&ch : tiles)
        {
            count[(int)(ch - 'A')]++;
        }
        return dfs(count);
    }
    ```

- [1081. Smallest Subsequence of Distinct Characters](https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/)

    与[316. Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/)完全相同，在给定字符串s中选择一个子串使其每个字符最多出现一次且逻辑序最小，可以用栈的思想来实现，即在顺序扫描字符串的过程中遇到任何当前还未使用的字符ch，则必须入栈（保证每个字符至少出现一次且仅仅出现一次），然后将当前栈顶比ch逻辑序大且在后面仍然有备用（统计每个字符的数量，每当ch使用一次则减少一次其可用量，直到减为0时不可用）的字符弹栈，之后将当前字符ch入栈即可，在此过程中s的每个字符有最多进栈一次弹栈一次操作两次，加上前期统计每个字符总的可用数量操作一次，每个字符最多操作三次，时间复杂度$O(n)$

    ```cpp
    string smallestSubsequence(string text)
    {
        string ret;
        vector<int> count(26, 0), unused(26, 1);
        for (auto &&ch : text)
        {
            count[(int)(ch - 'a')]++;
        }
        for (auto &&ch : text)
        {
            if (unused[(int)(ch - 'a')] == 1)
            {
                while (!ret.empty() && ret.back() > ch && count[(int)(ret.back() - 'a')] > 0)
                {
                    unused[(int)(ret.back() - 'a')] = 1;
                    ret.pop_back();
                }
                unused[(int)(ch - 'a')] = 0, ret.push_back(ch);
            }
            count[(int)(ch - 'a')]--;
        }
        return ret;
    }
    ```

- [1090. Largest Values From Labels](https://leetcode.com/problems/largest-values-from-labels/)

    贪心算法，将所有值values从大到小排序，然后在不超出限制的情况下优先选择更大的value，在此过程中用hashmap记录每个label的使用情况以备查询是否超出限制，时间复杂度$O(nlog(n))$
    
    ```cpp
    void quick_sort(vector<int> &values, vector<int> &labels, int left, int right)
    {
        if (left < right)
        {
            int p = left + ((right - left) >> 1);
            int i = left, j = right;
            swap(values[i], values[p]), swap(labels[i], labels[p]);
            int value = values[i], label = labels[i];
            while (i < j)
            {
                while (i < j && values[j] <= value)
                {
                    j--;
                }
                values[i] = values[j], labels[i] = labels[j];
                while (i < j && values[i] >= value)
                {
                    i++;
                }
                values[j] = values[i], labels[j] = labels[i];
            }
            values[i] = value, labels[i] = label;
            quick_sort(values, labels, left, i - 1);
            quick_sort(values, labels, i + 1, right);
        }
    }
    int largestValsFromLabels(vector<int> &values, vector<int> &labels, int num_wanted, int use_limit)
    {
        int ret = 0;
        // 手写快速排序
        quick_sort(values, labels, 0, values.size() - 1);
        // cout << integerVectorToString(values) << endl;
        // cout << integerVectorToString(labels) << endl;
        unordered_map<int, int> count;
        for (auto &&label : labels)
        {
            if (count.find(label) == count.end())
            {
                count[label] = use_limit;
            }
        }
        for (int i = 0; num_wanted > 0 && i < values.size(); i++)
        {
            if (count[labels[i]] > 0)
            {
                ret += values[i], count[labels[i]]--, num_wanted--;
            }
        }
        return ret;
    
    ```

- [1091. Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/)

    BFS+Queue

    ```cpp
    int shortestPathBinaryMatrix(vector<vector<int>> &grid)
    {
        int ret = -1;
        if (grid.size() > 0 && grid[0].size() > 0 && grid[0][0] == 0)
        {
            int rows = grid.size(), cols = grid[0].size();
            vector<int> shifts{-1, 0, 1};
            queue<vector<int>> cords;
            cords.push({0, 0});
            ret = 0;
            bool notfound = true;
            while (notfound && !cords.empty())
            {
                int count = cords.size();
                while (count)
                {
                    int r0 = cords.front()[0], c0 = cords.front()[1];
                    cords.pop();
                    if (r0 == rows - 1 && c0 == cols - 1)
                    {
                        notfound = false;
                        break;
                    }
                    for (int i = 0; i < 3; i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            int r = r0 + shifts[i], c = c0 + shifts[j];
                            if (r >= 0 && c >= 0 && r < rows && c < cols && grid[r][c] == 0)
                            {
                                grid[r][c] = 1;
                                cords.push({r, c});
                            }
                        }
                    }
                    count--;
                }
                ret++;
            }
            ret = notfound ? -1 : ret;
        }
        return ret;
    }
    ```
