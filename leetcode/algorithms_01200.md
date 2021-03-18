# 1101-1200

- [1109. Corporate Flight Bookings](https://leetcode.com/problems/corporate-flight-bookings/)

    - 最直接的想法是用一个数组记录每个飞机的定位数，时间复杂度$O(n*bookings.length)$，但是会$\color{red}{TLE}$

    ```cpp
    vector<int> corpFlightBookings(vector<vector<int>>& bookings, int n) {
        vector<int> ans(n+1,0);
        for (int i = 0; i < bookings.size(); i++)
        {
            for (int j = bookings[i][0]; j <= bookings[i][1]; j++)
            {
                ans[j]+=bookings[i][2];
            }
        }
        ans.erase(ans.begin());
        return ans;
    }
    ```

    - 使用扫描线（running scan）的概念，可以记录一个定位区间的首架飞机（定位数，正值）和最后一架飞机的下一架（负值），然后从左向右累加，时间复杂度$O(max(n,bookings.length))$，参考[BLOG](https://blog.csdn.net/sinat_27953939/article/details/95119024)

    ```cpp
    vector<int> corpFlightBookings(vector<vector<int>> &bookings, int n)
    {
        vector<int> count(n + 1, 0);
        for (auto &&item : bookings)
        {
            count[item[0] - 1] += item[2], count[item[1]] -= item[2];
        }
        for (int i = 1; i < n; i++)
        {
            count[i] += count[i - 1];
        }
        count.pop_back(); // throw the last auxiliary value
        return count;
    }
    ```

- [1110. Delete Nodes And Return Forest](https://leetcode.com/problems/delete-nodes-and-return-forest/)

	- 使用一个辅助节点auxiliary来标记每个节点的父节点，当父节点的子节点需要删除时，父节点指向需要删除子节点的指针为空NULL，然后递归处理每个子节点，因为每个节点最多访问一次，因此时间复杂度$O(n)$

	```cpp
	class Solution
	{
	private:
		void dfs(TreeNode *parent, vector<TreeNode *> &ret, unordered_set<int> &nums, bool need_delete, bool parent_deleted)
		{
			if (parent)
			{
				if (parent_deleted && !need_delete)
				{
					// 父节点已经被删除，但是本身该节点不需要删除
					ret.push_back(parent);
				}
				if (parent->left)
				{
					TreeNode *cur = parent->left;
					bool deleted_left = false;
					if (nums.find(cur->val) != nums.end())
					{
						parent->left = nullptr;
						deleted_left = true;
					}
					dfs(cur, ret, nums, deleted_left, need_delete);
				}
				if (parent->right)
				{

					TreeNode *cur = parent->right;
					bool deleted_right = false;
					if (nums.find(cur->val) != nums.end())
					{
						parent->right = nullptr;
						deleted_right = true;
					}
					dfs(cur, ret, nums, deleted_right, need_delete);
				}
			}
		}

	public:
		vector<TreeNode *> delNodes(TreeNode *root, vector<int> &to_delete)
		{
			TreeNode *auxiliary = new TreeNode(-1);
			auxiliary->left = root;
			unordered_set<int> nums;
			for (auto &&v : to_delete)
			{
				nums.insert(v);
			}
			vector<TreeNode *> ret;
			dfs(auxiliary, ret, nums, true, false);
			return ret;
		}
	};
	```

	- cpp的优化实现

	```cpp
	class Solution
	{
	private:
		unordered_set<int> nums;
		vector<TreeNode *> ret;
		TreeNode *dfs(TreeNode *root, bool is_root)
		{
			if (root)
			{
				bool deleted = nums.find(root->val) != nums.end();
				if (is_root && !deleted)
				{
					ret.push_back(root); // root节点且不需要delete
				}
				root->left = dfs(root->left, deleted);
				root->right = dfs(root->right, deleted);
				if (deleted)
				{
					root = nullptr;
				}
			}
			return root;
		}

	public:
		vector<TreeNode *> delNodes(TreeNode *root, vector<int> &to_delete)
		{
			nums.clear(), ret.clear();
			for (auto &&v : to_delete)
			{
				nums.insert(v);
			}
			dfs(root, true);
			return ret;
		}
	};
	```

- [1111. Maximum Nesting Depth of Two Valid Parentheses Strings](https://leetcode.com/problems/maximum-nesting-depth-of-two-valid-parentheses-strings/)

	- 并列的两个vps不增加其中任何一个的depth，因此只需要将嵌套的vps尽可能分开到A和B即可，时间复杂度$O(n)$

	```cpp
	vector<int> maxDepthAfterSplit(string seq)
	{
		int n = seq.size();
		vector<int> ret(n);
		int left = 0, right = 0, k = 0;
		for (auto &&ch : seq)
		{
			if (ch == '(')
			{
				ret[k++] = left;
				left = 1 - left;
			}
			else
			{
				ret[k++] = right;
				right = 1 - right;
			}
		}
		return ret;
	}
	```

- [1114](https://leetcode.com/problems/print-in-order/)

    注意cpp中的多线程与线程锁机制

- [1115](https://leetcode.com/problems/print-foobar-alternately/)

    cpp中的多线程机制，线程锁的使用，mutex包和mutex，基本格式如下，两个线程A和B交替使用，则：

    ```cpp
    mutex a,b;
    b.lock();
    A{
        a.lock();
        something;
        b.unlock();
    }
    B{
        b.lock();
        something;
        a.unlock();
    }
    ```

    本题解法如下：

    ```cpp
    class FooBar {
        private:
            int n;
            mutex m1,m2;

        public:
            FooBar(int n) {
                this->n = n;
                m2.lock();
            }

            void foo(function<void()> printFoo) {
                for (int i = 0; i < n; i++) {
                    // printFoo() outputs "foo". Do not change or remove this line.
                    m1.lock();
                    printFoo();
                    m2.unlock();
                }
            }

            void bar(function<void()> printBar) {
                for (int i = 0; i < n; i++)
                    // printBar() outputs "bar". Do not change or remove this line.
                    m2.lock();
                    printBar();
                    m1.unlock();
                }
            }
    };
    ```

- [1122. 数组的相对排序](https://leetcode-cn.com/problems/relative-sort-array/)

    以arr2的数字相对顺序为标准对arr1快速排序即可，时间复杂度$O(nlog(n))$

    ```cpp
	vector<int> relativeSortArray(vector<int> &arr1, vector<int> &arr2)
	{
		const int n = arr2.size();
		unordered_map<int, int> order;
		for (int i = 0; i < n; i++)
		{
			order[arr2[i]] = i;
		}
		sort(arr1.begin(), arr1.end(), [&order](const auto &a, const auto &b) -> bool {
			bool ret;
			auto it_a = order.find(a), it_b = order.find(b);
			if (it_a != order.end() && it_b != order.end())
			{
				ret = it_a->second < it_b->second;
			}
			else if (it_a != order.end() && it_b == order.end())
			{
				ret = true;
			}
			else if (it_a == order.end() && it_b != order.end())
			{
				ret = false;
			}
			else
			{
				ret = a < b;
			}
			return ret;
		});
        return arr1;
	}
    ```

- [1128](https://leetcode.com/problems/number-of-equivalent-domino-pairs/)

    - brute force[$O(n^2)$]

    ```cpp
    int numEquivDominoPairs(vector<vector<int>>& dominoes) {
        int const count=dominoes.size();
        for (int i = 0; i < count; i++)
        {
            if(dominoes[i][0]>dominoes[i][1]){
                swap(dominoes[i][0],dominoes[i][1]);
            }
        }
        sort(dominoes.begin(),dominoes.end(),[](vector<int>& x,vector<int>&y)->bool{return x[0]<y[0];});
        int ans=0;
        for (int i = 0; i < count; i++)
        {
            int j=i+1;
            while(j<count && dominoes[i][0]==dominoes[j][0]){
                if(dominoes[i][1]==dominoes[j][1]){
                    ans++;
                }
                j++;
            }
        }
        return ans;
    }
    ```

    - encoding method[$O(n)$]

    ```cpp
    int numEquivDominoPairs(vector<vector<int>>& dominoes) {
        int ans=0;
        // 1 <= dominoes[i][j] <= 9
        vector<int> count(100,0);
        for (int i = 0; i < dominoes.size(); i++)
        {
            count[10*max(dominoes[i][0],dominoes[i][1])+min(dominoes[i][0],dominoes[i][1])]++;
        }
        for (int i = 0; i < count.size(); i++)
        {
            if(count[i]!=0){
                ans+=count[i]*(count[i]-1)/2;
            }
        }
        return ans;
    }
    ```

- [1138. Alphabet Board Path](https://leetcode.com/problems/alphabet-board-path/)

    难点在于涉及到z的移动，需要控制移动方向不要超出board的范围

    ```cpp
    string move(int src, int dst, bool horizontal)
    {
        vector<char> chars{'L', 'R', 'U', 'D'};
        int shift = horizontal ? 0 : 2;
        string ans;
        if (dst > src)
        {
            while (src < dst)
            {
                src++, ans += chars[1 + shift];
            }
        }
        else
        {
            while (src > dst)
            {
                src--, ans += chars[0 + shift];
            }
        }
        return ans;
    }
    string alphabetBoardPath(string target)
    {
        int r0 = 0, c0 = 0;
        string ans;
        for (auto &&ch : target)
        {
            int r = (int)(ch - 'a') / 5, c = (int)(ch - 'a') % 5;
            int mid_r = r0, mid_c = c, v = mid_r * 5 + mid_c;
            if (v < 26)
            {
                // 先水平移动，后垂直移动
                ans += move(c0, c, true);
                ans += move(r0, r, false);
                ans.push_back('!');
            }
            else
            {
                // 如果失败，先垂直移动，后水平移动
                ans += move(r0, r, false);
                ans += move(c0, c, true);
                ans.push_back('!');
            }
            r0 = r, c0 = c;
        }
        return ans;
    }
    ```

- [1140. Stone Game II](https://leetcode.com/problems/stone-game-ii/)

	DP，时间复杂度$O(n^3)$

	```cpp
	int stoneGameII(vector<int> &piles)
	{
		/**
		 * Dynamic Plan
		 * dp[i][j] 表示 M = j 时 Alex 从 piles[i] 开始到结束时可以获得的最多石头数字
		 * dp[i][j] = max(suffix_sum[i] - dp[i+X][max(j,X)]), 1 <= X <= 2*j
		 *            这里去max(j,X)是因为alex去完之后到lee，则M=max(M,X)
		 * */
		int n = piles.size();
		vector<vector<int>> dp(n + 1, vector<int>(n + 1));
		vector<int> suffix_sum(n + 1, 0);
		for (int i = n - 1; i >= 0; i--)
		{
			suffix_sum[i] = suffix_sum[i + 1] + piles[i];
		}
		for (auto i = 0; i <= n; i++)
		{
			dp[i][n] = suffix_sum[i]; // M足够大时Alex可以一次拿完全部stones
		}
		for (auto i = n - 1; i >= 0; i--)
		{
			for (auto j = n - 1; j >= 1; j--)
			{
				for (auto x = 1; x <= 2 * j && i + x <= n; x++)
				{
					dp[i][j] = max(dp[i][j], suffix_sum[i] - dp[i + x][max(j, x)]);
				}
			}
		}
		return dp[0][1]; // initially, M = 1
	}
	```

- [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

    动态规划，时间复杂度与空间复杂度均为$O(m*n)$

    ```cpp
	int longestCommonSubsequence(string text1, string text2)
	{
		const int m = text1.length(), n = text2.length();
		vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				dp[i + 1][j + 1] = text1[i] == text2[j] ? dp[i][j] + 1 : max(dp[i + 1][j], dp[i][j + 1]);
			}
		}
		return dp[m][n];
	}
    ```

- [1169](https://leetcode.com/problems/invalid-transactions/)

    在一个transaction列表中通过两条规则过滤掉invalid的transaction，规则如下：
    - amount > 1000
    - time[i]-time[j]<=60且names[i]=name[j],city[i]!=city[j]
    解析存储所有的transactions之后，规则1线性时间过滤一遍就好，规则2需要平方时间验证所有相同姓名下不同城市时间差在60以内的交易

    ```cpp
    vector<string> invalidTransactions(vector<string> &transactions)
    {
        const int count = transactions.size();
        vector<string> names, cities;
        vector<int> times(count, 0), amounts(count, 0);
        vector<bool> valid(count, true);

        // 解析记录
        for (int i = 0; i < count; i++)
        {
            string item;
            stringstream ss(transactions[i]);
            getline(ss, item, ',');
            names.push_back(item);
            getline(ss, item, ',');
            times[i] = stoi(item);
            getline(ss, item, ',');
            amounts[i] = stoi(item);
            getline(ss, item, ',');
            cities.push_back(item);
            if (amounts[i] > 1000)
            {
                valid[i] = false;
            }
        }
        // n^2时间遍历所有相同人名的不同交易
        for (int i = 0; i < count; i++)
        {
            for (int j = i + 1; j < count; j++)
            {
                if (names[i].compare(names[j]) == 0 && abs(times[i] - times[j]) <= 60 && cities[i].compare(cities[j]) != 0)
                {
                    valid[i] = false;
                    valid[j] = false;
                }
            }
        }

        // 输出结果
        vector<string> ans;
        for (int i = 0; i < count; i++)
        {
            if (valid[i] == false)
            {
                ans.push_back(transactions[i]);
            }
        }
        return ans;
    }
    ```

- [1171. Remove Zero Sum Consecutive Nodes from Linked List](https://leetcode.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/)

    给定链表存储的数组，删除其中和为0的连续子数组。将其视为array处理，首先求出前缀和，前缀和为0的数及之前的部分全部删除，然后在剩余的前缀和中，如果$prefix[j]-prefix[i]=0$，则数组中$nums_{i+1},...,nums_j$的连续和为0，删除这部分，时间复杂度$O(n^2)$

    ```cpp
    ListNode *removeZeroSumSublists(ListNode *head)
    {
        vector<int> nums, ret;
        int prefixSum = 0;
        ListNode *cur = head;
        while (cur)
        {
            prefixSum += cur->val;
            if (prefixSum)
            {
                nums.push_back(prefixSum);
            }
            else
            {
                nums.clear(), prefixSum = 0;
            }
            cur = cur->next;
        }
        for (int i = 0; i < nums.size(); i++)
        {
            if (nums[i] != 0)
            {
                for (int j = nums.size() - 1; j > i; j--)
                {
                    if (nums[j] - nums[i] == 0)
                    {
                        for (int k = i + 1; k <= j; k++)
                        {
                            nums[k] = 0;
                        }
                        i = j - 1; // with i++ in the outer for circle
                        break;
                    }
                }
            }
        }
        ListNode *auxiliary_head = new ListNode(0);
        cur = auxiliary_head;
        int i = 0, prev_non_zero = 0;
        for (int i = 0; i < nums.size(); i++)
        {
            if (nums[i] != 0)
            {
                cur->next = new ListNode(nums[i] - prev_non_zero);
                cur = cur->next;
                prev_non_zero = nums[i];
            }
        }
        return auxiliary_head->next;
    }
    ```

- [1186. Maximum Subarray Sum with One Deletion](https://leetcode.com/problems/maximum-subarray-sum-with-one-deletion/)

	动态规划，时间复杂度$O(n)$，空间复杂度$O(1)$

	```cpp
	int maximumSum(vector<int> &arr)
	{
		/**
		 * withDelete 表示包含当前元素结尾且尚未删除任何一个元素的最大子数组和
		 * withoutDelete 表示包含当前元素结尾且已经删除了一个元素的最大子数组和
		*/
		const int n = arr.size(), negative_inf = numeric_limits<int>::min();
		long long ret = negative_inf, withoutDelete = negative_inf, withDelete = negative_inf;
		for (int i = 0; i < n; i++)
		{
			withDelete = max(withoutDelete, withDelete + arr[i]);
			withoutDelete = max((long long)(arr[i]), withoutDelete + arr[i]);
			ret = max(ret, max(withoutDelete, withDelete));
		}
		return (int)(ret);
	}
	```

- [1189. “气球” 的最大数量](https://leetcode-cn.com/problems/maximum-number-of-balloons/)

    统计$b,a,l,o,n$五个字符出现的数量即可，时间复杂度$O(n)$，其中$n=text.length()$

    ```cpp
    int maxNumberOfBalloons(string text)
	{
		const int length = 26;
		vector<int> count(length, 0);
		for (auto &ch : text)
		{
			count[static_cast<int>(ch - 'a')]++;
		}
		vector<int> ret_balloon;
		ret_balloon.emplace_back(count[static_cast<int>('b' - 'a')]);
		ret_balloon.emplace_back(count[static_cast<int>('a' - 'a')]);
		ret_balloon.emplace_back(count[static_cast<int>('l' - 'a')] / 2);
		ret_balloon.emplace_back(count[static_cast<int>('o' - 'a')] / 2);
		ret_balloon.emplace_back(count[static_cast<int>('n' - 'a')]);
		return *min_element(ret_balloon.begin(), ret_balloon.end());
	}
    ```
