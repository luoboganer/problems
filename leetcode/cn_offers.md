<!--
 * @Filename: 
 * @Author: shifaqiang
 * @Email: 14061115@buaa.edu.cn
 * @Github: https://github.com/luoboganer
 * @Date: 2020-09-05 11:29:59
 * @LastEditors: shifaqiang
 * @LastEditTime: 2021-04-01 11:26:56
 * @Software: Visual Studio Code
 * @Description: 剑指Offer:名企面试官精讲典型编程题
-->

# 剑指Offer:名企面试官精讲典型编程题

- [剑指 Offer 12. 矩阵中的路径](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/)

    经典的DFS搜索+递归回溯问题

    ```cpp
    class Solution
    {
        bool dfs(vector<vector<char>> &board, string &word, int r, int c, int k)
        {
            if (k == word.size())
            {
                return true;
            }
            else if (r >= 0 && c >= 0 && r < board.size() && c < board[0].size() && board[r][c] == word[k])
            {
                board[r][c] = '#'; // flag for visited
                vector<int> directions{1, 0, -1, 0, 1};
                for (int i = 0; i < 4; i++)
                {
                    if (dfs(board, word, r + directions[i], c + directions[i + 1], k + 1))
                    {
                        return true;
                    }
                }
                board[r][c] = word[k]; // backtracking
            }
            return false;
        }

    public:
        bool exist(vector<vector<char>> &board, string word)
        {
            if (word.length() == 0)
            {
                return true;
            }
            else if (board.size() > 0 && board[0].size() > 0)
            {
                const int rows = board.size(), cols = board[0].size();
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        if (dfs(board, word, i, j, 0))
                        {
                            return true;
                        }
                    }
                }
            }
            return false;
        }
    };
    ```

- [剑指 Offer 13. 机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

    BFS搜索从$(0,0)$出发所有能够到达的点，时间复杂度与空间复杂度均为$O(m*n)$

    ```cpp
    class Solution
    {
    private:
        int bitSum(int v)
        {
            int ret = 0;
            while (v)
            {
                ret += v % 10;
                v /= 10;
            }
            return ret;
        }

    public:
        int movingCount(int m, int n, int k)
        {
            vector<int> directions{1, 0, -1, 0, 1};
            const int length = 4;
            vector<int> bitSumX(m), bitSumY(n);
            for (int i = 0; i < m; i++)
            {
                bitSumX[i] = bitSum(i);
            }
            for (int i = 0; i < n; i++)
            {
                bitSumY[i] = bitSum(i);
            }
            vector<vector<bool>> unvisited(m, vector<bool>(n, true));
            int count = 1; // 初始位于格子(0,0)
            queue<pair<int, int>> qe{{make_pair(0, 0)}};
            unvisited[0][0] = false;
            while (!qe.empty())
            {
                auto [x, y] = qe.front();
                qe.pop();
                for (int i = 0; i < length; i++)
                {
                    int r = x + directions[i], c = y + directions[i + 1];
                    if (r >= 0 && r < m && c >= 0 && c < n && unvisited[r][c])
                    {
                        unvisited[r][c] = false;
                        if (bitSumX[r] + bitSumY[c] <= k)
                        {
                            count++;
                            qe.push(make_pair(r, c));
                        }
                    }
                }
            }
            return count;
        }
    };
    ```

- [剑指 Offer 19. 正则表达式匹配](https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/)

    **与主站[10. Regular Expression Matching](https://leetcode.com/problems/regular-expression-matching/)完全相同**

    - 递归写法

    ```cpp
	bool isMatch(string s, string p)
	{
		bool ret;
		if (p.empty())
		{
			ret = s.empty();
		}
		else
		{
			bool first_match = (!s.empty()) && (s[0] == p[0] || p[0] == '.'); // s非空且第一个字符匹配
			// 在第一个字符匹配的情况下递归检查后面的字符是否匹配
			if (p.length() >= 2 && p[1] == '*')
			{
				ret = (first_match && isMatch(s.substr(1), p)) || isMatch(s, p.substr(2));
			}
			else
			{
				ret = first_match && isMatch(s.substr(1), p.substr(1));
			}
		}
		return ret;
	}
    ```

    - 动态规划，时间复杂度$O(s.length*p.length)$

    ```cpp
	bool isMatch(string s, string p)
	{
		const int m = s.length(), n = p.length();
		vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
		dp[0][0] = true; // s和p均为空的时候默认为匹配
		for (int j = 2; j <= n; j++)
		{
			dp[0][j] = dp[0][j - 2] && (p[j - 1] == '*'); // 模式串为a*时匹配任意空串(*前的字符可以有0个)
		}
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				if (p[j] == '*')
				{
					// p[j]为*时，s[i]匹配*之前的字符p[j-1]或者 p[j-1]*匹配一个空串
					dp[i + 1][j + 1] = dp[i + 1][j - 1] || (dp[i][j + 1] && (s[i] == p[j - 1] || p[j - 1] == '.'));
				}
				else
				{
					dp[i + 1][j + 1] = dp[i][j] && ((s[i] == p[j]) || (p[j] == '.'));
				}
			}
		}
		return dp.back().back();
	}
    ```

- [剑指 Offer 26. 树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)

    递归解决，首先判定B为空时不是任何树的子结构，当B为非空时递归盘点B是A的一部分/B是A的左子树的子结构/B是A的右子树的子结构

    **判定B是以A为根节点的一部分时满足以下条件之一即可**
    - A和B均为空节点
    - A和B均为非空节点且节点值相同，且：
        - B有左子树时A的左子树与B的左子树递归相同
        - B有右子树时A的右子树与B的右子树递归相同

    ```cpp
    class Solution
    {
    private:
        bool subTreeNodeA(TreeNode *a, TreeNode *b)
        {
            return (!a && !b) || (a && b && a->val == b->val && ((!b->left && !b->right) || (subTreeNodeA(a->left, b->left) && !b->right) || (!b->left && subTreeNodeA(a->right, b->right)) || (b->left && b->right && subTreeNodeA(a->left, b->left) && subTreeNodeA(a->right, b->right))));
        }

    public:
        bool isSubStructure(TreeNode *A, TreeNode *B)
        {
            return (A && B) && (subTreeNodeA(A, B) || isSubStructure(A->left, B) || isSubStructure(A->right, B));
        }
    };
    ```

- [剑指 Offer 30. 包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)

    使用单独的数组来记录当前最小值的idx即可实现全$O(1)$的操作

    ```cpp
    class MinStack
    {
    private:
        vector<int> st, min_index;

    public:
        /** initialize your data structure here. */
        MinStack()
        {
            st.clear(), min_index.clear();
        }

        void push(int x)
        {
            if (!min_index.empty() && st[min_index.back()] < x)
            {
                min_index.emplace_back(min_index.back());
            }
            else
            {
                min_index.emplace_back(st.size());
            }
            st.emplace_back(x);
        }

        void pop()
        {
            min_index.pop_back(), st.pop_back();
        }

        int top()
        {
            return st.back();
        }

        int min()
        {
            return st[min_index.back()];
        }
    };
    ```

- [剑指 Offer 31. 栈的压入、弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

    模拟压栈和弹栈的操作直到可以按给定顺序弹栈且栈为空，时间复杂度$O(n)$

    ```cpp
	bool validateStackSequences(vector<int> &pushed, vector<int> &popped)
	{
		const int n = pushed.size();
		bool ret = true;
		if (n > 0)
		{
			int i = 0, j = 0;
			stack<int> st;
			while (j < n)
			{
				if (!st.empty() && st.top() == popped[j])
				{
					j++;
					st.pop();
				}
				else if (i < n)
				{
					st.push(pushed[i++]);
				}
				else
				{
					ret = false;
					break;
				}
			}
		}
		return ret;
	}
    ```

- [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/submissions/)

    递归检查root左右子树是否符合要求

    ```cpp
    class Solution
    {
    private:
        bool dfs(vector<int> &nums, int left, int right, int lower, int upper)
        {
            if (left <= right)
            {
                if (nums[right] < lower || nums[right] > upper)
                {
                    return false;
                }
                int idx = right - 1;
                while (idx >= left && nums[idx] > nums[right])
                {
                    idx--;
                }
                return dfs(nums, left, idx, lower, nums[right]) && dfs(nums, idx + 1, right - 1, nums[right], upper);
            }
            return true;
        }

    public:
        bool verifyPostorder(vector<int> &postorder)
        {
            return dfs(postorder, 0, postorder.size() - 1, numeric_limits<int>::min(), numeric_limits<int>::max());
        }
    };
    ```

- [剑指 Offer 37. 序列化二叉树](https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/)

    二叉树的序列化与反序列化，主要是队列(Queue)的运用

    ```cpp
    class Codec
    {
    public:
        // Encodes a tree to a single string.
        string serialize(TreeNode *root)
        {
            string ret;
            // 按照层序遍历的原则输出每个节点，叶子节点的两个子节点输出null
            if (root)
            {
                queue<TreeNode *> bfs{{root}};
                vector<TreeNode *> nodes;
                while (!bfs.empty())
                {
                    TreeNode *cur = bfs.front();
                    bfs.pop();
                    nodes.emplace_back(cur);
                    if (cur)
                    {
                        bfs.push(cur->left);
                        bfs.push(cur->right);
                    }
                }
                while (!nodes.empty() && nodes.back() == nullptr)
                {
                    nodes.pop_back();
                }
                for (auto &node : nodes)
                {
                    ret += node ? to_string(node->val) : "null";
                    ret += ',';
                }
            }
            return '[' + ret + ']';
        }

        // Decodes your encoded data to tree.
        TreeNode *deserialize(string data)
        {
            TreeNode *auxiliary = new TreeNode(0);
            data = data.substr(1, data.length() - 2);
            if (data.length() > 0)
            {
                vector<string> tokens;
                string token;
                for (auto ch : data)
                {
                    if (ch == ',')
                    {
                        tokens.emplace_back(token);
                        token.clear();
                    }
                    else
                    {
                        token += ch;
                    }
                }
                queue<TreeNode *> bfs{{auxiliary}};
                bool left = false;
                for (auto t : tokens)
                {
                    TreeNode *cur = bfs.front();
                    if (!left)
                    {
                        bfs.pop(); // 当前填充cur的右子树，则cur可以出队
                    }
                    if (t.compare("null") != 0)
                    {
                        if (left)
                        {
                            cur->left = new TreeNode(stoi(t));
                            bfs.push(cur->left);
                        }
                        else
                        {
                            cur->right = new TreeNode(stoi(t));
                            bfs.push(cur->right);
                        }
                    }
                    left = !left;
                }
            }
            return auxiliary->right;
        }
    };
    ```

- [剑指 Offer 38. 字符串的排列](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/)

    有重复字符串的全排列(permutation)，与[47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)相同

    ```cpp
    class Solution
    {
    private:
        void dfs(vector<string> &ret, string &s, int start, const int end)
        {
            if (start == end)
            {
                ret.emplace_back(s);
            }
            else
            {
                for (int i = start; i <= end; i++)
                {
                    bool repeat = false;
                    for (int j = start; !repeat && j < i; j++)
                    {
                        repeat = s[i] == s[j];
                    }
                    if (!repeat)
                    {
                        swap(s[i], s[start]);
                        dfs(ret, s, start + 1, end);
                        swap(s[i], s[start]);
                    }
                }
            }
        }

    public:
        vector<string> permutation(string s)
        {
            vector<string> ret;
            dfs(ret, s, 0, s.length() - 1);
            return ret;
        }
    };
    ```

- [剑指 Offer 40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

    cpp中优先队列的使用

    ```cpp
	vector<int> getLeastNumbers(vector<int> &arr, int k)
	{
        /**
         * priority_queue<int, vector<int>, greater<int>>
         * priority_queue<int, vector<int>, less<int>>
        */
        priority_queue<int, vector<int>, greater<int>> qe(arr.begin(), arr.end());
		vector<int> ret(k);
		for (int i = 0; i < k; i++)
		{
			ret[i] = qe.top();
			qe.pop();
		}
		return ret;
	}
    ```

- [剑指 Offer 44. 数字序列中某一位的数字](https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/)

    与主站[400](https://leetcode.com/problems/nth-digit/)完全相同，统计位宽为$k$的数字数量即可

    ```cpp
	int findNthDigit(int n)
	{
		long long base = 9, width = 1, count = n;
		while (true)
		{
			long long base_width = base * width;
			if (base_width >= count)
			{
				break;
			}
			count -= base_width;
			base *= 10, width++;
		}
		base /= 9;
		return static_cast<int>(to_string(base + (count - 1) / width)[(count - 1) % width] - '0');
	}
    ```

- [剑指 Offer 45. 把数组排成最小的数](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

    注意字符串排序的规则，xy最小的标准是字典序$xy<yx$

    ```cpp
    string minNumber(vector<int> &nums)
	{
		vector<string> items;
		for (auto &v : nums)
		{
			items.emplace_back(to_string(v));
		}
		sort(items.begin(), items.end(), [](const auto &a, const auto &b) -> bool {
			// 字符串排序规则
			string xy = a + b, yx = b + a;
			return xy.compare(yx) < 0;
		});
		string ret;
		for (auto &item : items)
		{
			ret += item;
		}
		return ret;
	}
    ```

- [剑指 Offer 46. 把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

    一维动态规划，从数字的最低位开始每次表示一位或者两位，时间复杂度$O(n)$

    ```cpp
	int translateNum(int num)
	{
		vector<int> reversed_bits;
		do
		{
			reversed_bits.emplace_back(num % 10);
			num /= 10;
		} while (num);
		const int n = reversed_bits.size();
		vector<int> dp(n + 1, 0);
		for (int i = 1; i < n; i++)
		{
			int temp = reversed_bits[i] * 10 + reversed_bits[i - 1];
			(temp >= 10 && temp <= 25) ? dp[i + 1] = dp[i] + dp[i - 1] : dp[i + 1] = dp[i];
		}
		return dp.back();
	}
    ```

- [剑指 Offer 52. 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

    寻找两个链表的第一个公共节点，与检测一个链表中是否有环/寻找环节点类似，双指针遍历即可

    ```cpp
	ListNode *getIntersectionNode(ListNode *headA, ListNode *headB)
	{
		ListNode *a = headA, *b = headB;
		while (a != b)
		{
			a = a ? a->next : headB;
			b = b ? b->next : headA;
		}
		return a;
	}
    ```

- [剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

    - 使用**lower_bound()/upper_bound()**库函数

    ```cpp
	int search(vector<int> &nums, int target)
	{
		auto left = lower_bound(nums.begin(), nums.end(), target);
		auto right = upper_bound(nums.begin(), nums.end(), target);
		return right - left;
	}
    ```

    - 手动实现有限制条件（最左/最右）的二分查找

    ```cpp
    class Solution
    {
    private:
        int binary_search(vector<int> &nums, int target, int mode)
        {
            /**
            * mode = 0 'most_left', lower
            * mode = 1 'most_right', upper
            */
            int left = 0, right = nums.size();
            while (left < right)
            {
                int mid = left + (right - left) / 2;
                if (nums[mid] > target || (mode == 0 && nums[mid] == target))
                {
                    right = mid;
                }
                else
                {
                    left = mid + 1;
                }
            }
            return left;
        }

    public:
        int search(vector<int> &nums, int target)
        {
            int left = binary_search(nums, target, 0);
            int right = binary_search(nums, target, 1);
            return right - left;
        }
    };
    ```

- [剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

    中序遍历将二叉搜索树转化为排序树组，按照下标找出第k大的节点
    
    ```cpp
    class Solution
    {
    private:
        vector<int> inorder_traversal_iter(TreeNode *root)
        {
            vector<int> ret;
            if (root)
            {
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
                        ret.push_back(cur->val);
                        cur = cur->right;
                    }
                }
            }
            return ret;
        }

    public:
        int kthLargest(TreeNode *root, int k)
        {
            vector<int> nums = inorder_traversal_iter(root);
            return nums[nums.size() - k];
        }
    };
    ```

- [剑指 Offer 57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

    由题意每个可能的序列为[a,a+1,a+2,...,a+b]且其和为target，按照高斯求和的方式确定两个端点[a,b]即可

    ```cpp
	vector<vector<int>> findContinuousSequence(int target)
	{
		vector<vector<int>> ret;
		if (target >= 2)
		{
			// [a,a+1,a+2,a+3,...,a+b]
			int max_b = floor(sqrt(target * 2));
			for (int b = 1; b <= max_b; b++)
			{
				int a = (target * 2 / (b + 1) - b) / 2;
				if ((b + 1) * (2 * a + b) == 2 * target)
				{
					vector<int> cur(b + 1);
					for (int i = 0; i <= b; i++)
					{
						cur[i] = a + i;
					}
					ret.push_back(cur);
				}
			}
		}
		return ret;
	}
    ```

- [剑指 Offer 60. n个骰子的点数](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/)

    从骰子数为1开始递归/迭代计算骰子数为n的情况，注意double数的精度问题

    ```cpp
    	vector<double> twoSum(int n)
	{
		if (n < 1)
		{
			return {};
		}
		else if (n == 1)
		{
			return {1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6};
		}
		vector<double> ret{1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6};
		vector<double> base{1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6};
		for (int v = 2; v <= n; v++)
		{
			const int count = 5 * v + 1, ret_length = ret.size();
			vector<double> cur(count, 0.0);
			for (int i = 0; i < ret.size(); i++)
			{
				for (int j = 0; j < 6; j++)
				{
					cur[i + j] += ret[i] * base[j];
				}
			}
			ret = cur;
		}
		return ret;
	}
    ```

- [剑指 Offer 61. 扑克牌中的顺子](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

    从最小牌面（非大小王，即非0）开始凑够连续五张牌即可，模拟

    ```cpp
	bool isStraight(vector<int> &nums)
	{
		vector<int> count(16, 0); // 统计0-13每张牌的数量
		int min_value = 14;		  // 牌面最大为13
		for (auto &v : nums)
		{
			if (v != 0)
			{
				min_value = min(min_value, v);
			}
			count[v]++;
		}
		for (int i = 1; i < 5; i++)
		{
			if (count[i + min_value] == 0)
			{
				if (count[0] > 0)
				{
					count[0]--;
				}
				else
				{
					return false; // 缺少牌面值i+min_value
				}
			}
		}
		return true;
	}
    ```

- [剑指 Offer 67. 把字符串转换成整数](https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/)

    与[8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/)相同，注意超出int型表示范围的边界条件处理

    ```cpp
	int strToInt(string str)
	{
		long long ret = 0, lower = numeric_limits<int>::min(), upper = numeric_limits<int>::max();
		int i = 0, n = str.length();
		bool negative = false;
		// 判断可能存在的空白字符串
		while (i < n && isspace(str[i]))
		{
			i++;
		}
		// 判断可能存在的正负号
		if (i < n && str[i] == '+')
		{
			i++;
		}
		else if (i < n && str[i] == '-')
		{
			i++, negative = true;
		}
		// 从第一个非空白、非正负号的字符开始转换
		while (i < n && isdigit(str[i]))
		{
			ret = ret * 10 + static_cast<int>(str[i] - '0');
			if (negative && (-ret <= lower))
			{
				ret = -lower; // 剪枝处理
				break;
			}
			if (ret >= upper)
			{
				ret = upper; // 剪枝处理
                break;
			}
			i++;
		}
		return static_cast<int>(negative ? -ret : ret);
	}
    ```

- [剑指 Offer 68 - II. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

    深度优先遍历DFS，递归实现，时间复杂度$O(n)$，其中$n$是给定树的节点总数

    ```cpp
    class Solution
    {
        pair<int, TreeNode *> dfs(TreeNode *root, TreeNode *p, TreeNode *q)
        {
            if (root)
            {
                auto left = dfs(root->left, p, q);
                if (left.second != nullptr)
                {
                    return make_pair(0, left.second);
                }
                auto right = dfs(root->right, p, q);
                if (right.second != nullptr)
                {
                    return make_pair(0, right.second);
                }
                int count = left.first + right.first;
                TreeNode *parent = nullptr;
                if (p->val == root->val)
                {
                    count++;
                }
                if (q->val == root->val)
                {
                    count++;
                }
                if (count >= 2)
                {
                    parent = root;
                }
                return make_pair(count, parent);
            }
            return make_pair(0, nullptr);
        }

    public:
        TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q)
        {
            return dfs(root, p, q).second;
        }
    };
    ```

- [...](123)
