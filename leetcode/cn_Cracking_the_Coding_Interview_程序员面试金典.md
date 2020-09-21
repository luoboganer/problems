<!--
 * @Filename: 
 * @Author: shifaqiang
 * @Email: 14061115@buaa.edu.cn
 * @Github: https://github.com/luoboganer
 * @Date: 2020-09-05 11:29:59
 * @LastEditors: shifaqiang
 * @LastEditTime: 2020-09-21 17:04:56
 * @Software: Visual Studio Code
 * @Description: 程序员面试金典
-->

# 程序员面试金典

- [面试题 01.05. 一次编辑](https://leetcode-cn.com/problems/one-away-lcci/)

    - 动态规划dynamic plan计算编辑距离（edit distance），判断是否$edit_distance \le 1$即可，时间复杂度$O(n^2)$

	```cpp
	bool oneEditAway(string first, string second)
	{
		int m = first.size(), n = second.size();
		vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
		for (auto i = 1; i <= m; i++)
		{
			dp[i][0] = i;
		}
		for (auto j = 1; j <= n; j++)
		{
			dp[0][j] = j;
		}
		for (auto i = 0; i < m; i++)
		{
			for (auto j = 0; j < n; j++)
			{
				if (first[i] == second[j])
				{
					dp[i + 1][j + 1] = dp[i][j];
				}
				else
				{
					dp[i + 1][j + 1] = min(min(dp[i][j + 1], dp[i + 1][j]), dp[i][j]) + 1;
				}
			}
		}
		return dp.back().back() <= 1;
	}
	```

    - 事实上不需要完全计算出编辑距离，只要双指针从两端向中间扫描，判断编辑距离和1之间的大小即可，时间复杂度$O(n)$

	```cpp
	bool oneEditAway(string first, string second)
	{
		/**
		 * 编辑距离为0，则两个字符串完全相同
		 * 编辑距离为1，则：
		 * 			1. 两个字符串长度想同且只有一个字符不同
		 * 			2. 两个字符串长度相差为1且剩余字符串完全相同
		*/
		if (first.compare(second) == 0)
		{
			return true;
		}
		int length_first = first.length(), length_second = second.length();
		if (abs(length_first - length_second) > 1)
		{
			return false;
		}
		int i = 0, j = length_first - 1, k = length_second - 1;
		while (i < length_first && i < length_second && first[i] == second[i])
		{
			i++;
		}
		while (j >= 0 && k >= 0 && first[j] == second[k])
		{
			j--, k--;
		}
		return j - i < 1 && k - i < 1;
	}
	```

- [面试题 01.06. 字符串压缩](https://leetcode-cn.com/problems/compress-string-lcci/)

    ```cpp
	string compressString(string S)
	{
		string ret;
		int count = 0;
		for (auto &&ch : S)
		{
			if (ret.empty() || ch != ret.back())
			{
				if (count > 0)
				{
					ret += to_string(count);
				}
				ret.push_back(ch);
				count = 1;
			}
			else
			{
				count++;
			}
		}
		if (count > 0)
		{
			ret += to_string(count);
		}
		return ret.length() < S.length() ? ret : S;
	}
    ```

- [面试题 01.07. 旋转矩阵](https://leetcode-cn.com/problems/rotate-matrix-lcci/)

	首先沿着主对角线交换，然后左右交换即可实现右旋90度，时间复杂度$O(n^2)$

	```cpp
	void rotate(vector<vector<int>> &matrix)
	{
		int n = matrix.size();
		if (n > 0)
		{
			for (auto i = 0; i < n; i++)
			{
				for (auto j = 0; j < i; j++)
				{
					swap(matrix[i][j], matrix[j][i]);
				}
			}
			for (auto i = 0; i < n; i++)
			{
				int left = 0, right = n - 1;
				while (left < right)
				{
					swap(matrix[i][left++], matrix[i][right--]);
				}
			}
		}
	}
	```

- [面试题 02.03. 删除中间节点](https://leetcode-cn.com/problems/delete-middle-node-lcci/)

    - 当node存在下一个节点时，node的值等于下一个节点的值，然后又下下个节点，择node等于下一个节点继续删除，否则node的下一个节点指向空即可，时间复杂度$O(n)$

	```cpp
	void deleteNode(ListNode *node)
	{
		while (node->next)
		{
			node->val = node->next->val;
			if (node->next->next)
			{
				node = node->next;
			}
			else
			{
				node->next = nullptr;
			}
		}
	}
	```

    - 因为node不是最后一个节点，因此node->next必然存在，只需要将node->next的值保存到当前节点，然后删除node->next这个节点即可，时间复杂度$(O(1))$

	```cpp
	void deleteNode(ListNode *node)
	{
		node->val = node->next->val;
		node->next = node->next->next;
	}
	```

    - some test cases

	```cpp
	[4,5,1,9]
	5
	[4,5,1,9]
	1
	[4,3,5,1,9]
	5
	[4,3,5,1,9]
	3
	```

- [面试题 02.04. 分割链表](https://leetcode-cn.com/problems/partition-list-lcci/)

    - 归并排序的思想，先分割再合并，时间复杂度$O(n)$

	```cpp
	ListNode *partition(ListNode *head, int x)
	{
		ListNode *first = new ListNode(0), *second = new ListNode(0);
		ListNode *cur_first = first, *cur_second = second;
		while (head)
		{
			if (head->val < x)
			{
				cur_first->next = head;
				cur_first = head;
			}
			else
			{
				cur_second->next = head;
				cur_second = head;
			}
			head = head->next;
		}
		cur_first->next = second->next, cur_second->next = nullptr;
		return first->next;
	}
	```

    - 逐个扫描，遇到小于基准的节点放到头结点去即可，时间复杂度$(O(n))$

	```cpp
	ListNode *partition(ListNode *head, int x)
	{
		ListNode *auxiliary = new ListNode(0);
		auxiliary->next = head;
		ListNode *cur = auxiliary;
		while (cur->next)
		{
			if (cur->next->val < x)
			{
				ListNode *temp = cur->next;
				cur->next = cur->next->next;
				temp->next = auxiliary->next;
				auxiliary->next = temp;
				if (cur == auxiliary)
				{
					cur = cur->next; // 避免在头结点值小于基准x的时候死循环
				}
			}
			else
			{
				cur = cur->next;
			}
		}
		return auxiliary->next;
	}
	```

- [面试题 02.05. 链表求和](https://leetcode-cn.com/problems/sum-lists-lcci/)

    模拟竖式求和，时间复杂度$O(n)$

    ```cpp
	ListNode *addTwoNumbers(ListNode *l1, ListNode *l2)
	{
		int carry = 0;
		ListNode *cur_l1 = new ListNode(0), *cur_l2 = new ListNode(0);
		cur_l1->next = l1, cur_l2->next = l2;
		while (cur_l1->next && cur_l2->next)
		{
			int v = carry + cur_l1->next->val + cur_l2->next->val;
			cur_l1->next->val = v % 10;
			carry = v / 10;
			cur_l1 = cur_l1->next, cur_l2 = cur_l2->next;
		}
		if (cur_l2->next)
		{
			cur_l1->next = cur_l2->next; // l2剩余的尾部
		}
		while (carry)
		{
			if (cur_l1->next)
			{
				int v = cur_l1->next->val + carry;
				cur_l1->next->val = v % 10;
				carry = v / 10;
			}
			else
			{
				cur_l1->next = new ListNode(carry % 10);
				carry = carry / 10;
			}
			cur_l1 = cur_l1->next;
		}
		return l1;
	}
    ```

- [面试题 02.06. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list-lcci/)

    - 转化为数组或者使用栈，然后检查是否回味，时间复杂度$O(n)$，空间复杂度$O(n)$

    ```cpp
	bool isPalindrome(ListNode *head)
	{
		vector<int> nums;
		while (head)
		{
			nums.push_back(head->val);
			head = head->next;
		}
		int left = 0, right = nums.size() - 1;
		while (left < right)
		{
			if (nums[left++] != nums[right--])
			{
				return false;
			}
		}
		return true;
	}
    ```

    - 使用快慢双指针找到链表的中间节点，然后翻转后半部分链表，然后对比前半部分和翻转后的后半部分是否完全一致，时间复杂度$O(n)$，空间复杂度优化到$O(1)$

    ```cpp
    class Solution
    {
    private:
        ListNode *find_middle_node(ListNode *head)
        {
            if (head)
            {
                ListNode *slow = head, *fast = head;
                while (fast && fast->next)
                {
                    slow = slow->next, fast = fast->next->next;
                }
                return fast ? slow->next : slow;
            }
            return head;
        }
        ListNode *reverse_list(ListNode *head)
        {
            if (head)
            {
                ListNode *pre = nullptr, *cur = head;
                while (cur)
                {
                    ListNode *temp = cur->next;
                    cur->next = pre;
                    pre = cur;
                    cur = temp;
                }
                return pre;
            }
            return head;
        }

    public:
        bool isPalindrome(ListNode *head)
        {
            bool ret = true;
            if (head)
            {
                ListNode *first = head, *second = reverse_list(find_middle_node(head));
                while (first && second)
                {
                    if (first->val != second->val)
                    {
                        ret = false;
                        break;
                    }
                    first = first->next, second = second->next;
                }
            }
            return ret;
        }
    };
    ```

    - some test cases

    ```cpp
    [1,2]
    [1,2,3,2,1]
    [1,2,3,3,2,1]
    [1,2,3,4,2,1]
    []
    [1]
    ```

- [面试题 04.04. 检查平衡性](https://leetcode-cn.com/problems/check-balance-lcci/)
    
    递归的检查左右子树是否平衡，如果不平衡返回-1，平衡则返回子树的深度，然后当左右子树均平衡且深度差不超过1时该树是平衡的

	```cpp
	class Solution
	{
	private:
		int dfs(TreeNode *root)
		{
			/**
			* balance : return depth
			* not balance : return -1
			*/
			int depth = 0;
			if (root)
			{
				int left = dfs(root->left), right = dfs(root->right);
				if (left != -1 && right != -1 && abs(left - right) <= 1)
				{
					depth = max(left, right) + 1;
				}
				else
				{
					depth = -1;
				}
			}
			return depth;
		}

	public:
		bool isBalanced(TreeNode *root)
		{
			return dfs(root) != -1;
		}
	};
	```

- [面试题 04.05. 合法二叉搜索树](https://leetcode-cn.com/problems/legal-binary-search-tree-lcci/)

    判断一棵树是否为合法的二叉搜索树（binary search tree，BST）

    ```cpp
    class Solution
    {
    private:
        bool dfs(TreeNode *root, long long min_value, long long max_value)
        {
            bool ret = true;
            if (root)
            {
                long long v = root->val;
                ret = v < max_value && v > min_value && dfs(root->left, min_value, v) && dfs(root->right, v, max_value);
            }
            return ret;
        }

    public:
        bool isValidBST(TreeNode *root)
        {
            return dfs(root, numeric_limits<long long>::min(), numeric_limits<long long>::max());
        }
    };
    ```

- [面试题 08.03. 魔术索引](https://leetcode-cn.com/problems/magic-index-lcci/)

	难点：给定的**有序**数组中存在**重复的数值(duplicated)**，当没有重复数值的时候可以直接在$O(log(n))$复杂度下进行二分查找即可

    - 线性扫描，时间复杂度$O(n)$

	```cpp
	int findMagicIndex(vector<int> &nums)
	{
		int count = nums.size();
		for (auto i = 0; i < count; i++)
		{
			if (nums[i] == i)
			{
				return i;
			}
		}
		return -1;
	}
	```

    - 二分查找剪枝，平均时间复杂度$O(log(n))$，在最坏的情况下时间复杂度会退化到$O(n)$

	```cpp
	class Solution
	{
	private:
		int divided_conqer(vector<int> &nums, int left, int right)
		{
			int ret = -1;
			if (left <= right)
			{
				int mid = left + ((right - left) >> 1);
				int leftAnswer = divided_conqer(nums, left, mid - 1);
				if (leftAnswer != -1)
				{
					ret = leftAnswer;
				}
				else if (nums[mid] == mid)
				{
					ret = mid;
				}
				else
				{
					ret = divided_conqer(nums, mid + 1, right);
				}
			}
			return ret;
		}

	public:
		int findMagicIndex(vector<int> &nums)
		{
			return divided_conqer(nums, 0, nums.size() - 1);
		}
	};
	```

- [面试题 17.01. 不用加号的加法](https://leetcode-cn.com/problems/add-without-plus-lcci/)

	使用位运算求和，注意求进位在左移的过程中一定使用无符号整数防止溢出

	```cpp
	int add(int a, int b)
	{
		while (b)
		{
			auto carry = (unsigned int)(a & b) << 1;
			a ^= b;
			b = carry;
		}
		return a;
	}
	```

- [...](123)
