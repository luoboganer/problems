<!--
 * @Filename: 
 * @Author: shifaqiang
 * @Email: 14061115@buaa.edu.cn
 * @Github: https://github.com/luoboganer
 * @Date: 2020-09-05 11:29:59
 * @LastEditors: shifaqiang
 * @LastEditTime: 2020-10-19 12:00:41
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

- [...](123)
