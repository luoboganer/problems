<!--
 * @Filename: 
 * @Author: shifaqiang
 * @Email: 14061115@buaa.edu.cn
 * @Github: https://github.com/luoboganer
 * @Date: 2020-09-05 11:29:59
 * @LastEditors: shifaqiang
 * @LastEditTime: 2021-04-02 09:35:52
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

- [面试题 01.09. 字符串轮转](https://leetcode-cn.com/problems/string-rotation-lcci/)

	注意s1和s2中有空串时的边界条件处理，而当两个字符串均为非空时，利用s1旋转会形成的周期性，在s1+s1中检查是否存在子串s2即可

	```cpp
	bool isFlipedString(string s1, string s2)
	{
		s1 += s1;
		return (s1.empty() && s2.empty()) || (!s2.empty() && s1.find(s2) != string::npos);
	}
	```

- [面试题 02.01. 移除重复节点](https://leetcode-cn.com/problems/remove-duplicate-node-lcci/)

    - 哈希表记录出现过的数值，时间复杂度$O(n)$，空间复杂度$O(n)$

	```cpp
	ListNode *removeDuplicateNodes(ListNode *head)
	{
		unordered_set<int> seen_number;
		ListNode *auxiliary = new ListNode(0);
		auxiliary->next = head;
		ListNode *cur = auxiliary;
		while (cur->next)
		{
			if (seen_number.find(cur->next->val) != seen_number.end())
			{
				cur->next = cur->next->next;
			}
			else
			{
				seen_number.insert(cur->next->val);
				cur = cur->next;
			}
		}
		return auxiliary->next;
	}
	```

    - 在当前节点之前的节点中查找当前节点的值是否出现过， 时间复杂度$O(n^2)$，空间复杂度$O(1)$

	```cpp
	ListNode *removeDuplicateNodes(ListNode *head)
	{
		unordered_set<int> seen_number;
		ListNode *auxiliary = new ListNode(0);
		auxiliary->next = head;
		ListNode *cur = auxiliary;
		while (cur->next)
		{
			ListNode *prev = auxiliary;
			bool found_repeat = false;
			while (!found_repeat && prev->next != cur->next)
			{
				if (prev->next->val == cur->next->val)
				{
					cur->next = cur->next->next;
					found_repeat = true;
				}
				else
				{
					prev = prev->next;
				}
			}
			if (!found_repeat)
			{
				cur = cur->next;
			}
		}
		return auxiliary->next;
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

- [面试题 03.01. 三合一](https://leetcode-cn.com/problems/three-in-one-lcci/)

	一个数组分为三段模拟三个栈，每一段左端为栈底右端为栈顶即可

	```cpp
	class TripleInOne
	{
	private:
		vector<int> st;
		vector<int> sp;
		int size;

	public:
		TripleInOne(int stackSize)
		{
			size = stackSize;
			st.resize(stackSize * 3);
			// 三个栈在一个数组中，区间范围为[0,size-1] [size,2*szie-1] [2*size,3*szie-1]
			sp.resize(3);
			sp[0] = -1;
			sp[1] = stackSize - 1;
			sp[2] = 2 * stackSize - 1;
		}

		void push(int stackNum, int value)
		{
			if (sp[stackNum] != (1 + stackNum) * size - 1)
			{
				st[++sp[stackNum]] = value;
			}
		}

		int pop(int stackNum)
		{
			if (sp[stackNum] == stackNum * size - 1)
			{
				return -1;
			}
			return st[sp[stackNum]--];
		}

		int peek(int stackNum)
		{
			if (sp[stackNum] == stackNum * size - 1)
			{
				return -1;
			}
			return st[sp[stackNum]];
		}

		bool isEmpty(int stackNum)
		{
			return sp[stackNum] == stackNum * size - 1;
		}
	};
	```

- [面试题 03.06. 动物收容所](https://leetcode-cn.com/problems/animal-shelter-lcci/)

	使用队列模拟动物的先进先出原则

	```cpp
	class AnimalShelf
	{
	private:
		int idx; // idx代表动物的入园序号
		queue<vector<int>> cat, dog;

	public:
		AnimalShelf()
		{
			idx = 0;
			while (!cat.empty())
			{
				cat.pop();
			}
			while (!dog.empty())
			{
				dog.pop();
			}
		}

		void enqueue(vector<int> animal)
		{
			animal.emplace_back(idx++);
			animal[1] == 0 ? cat.push(animal) : dog.push(animal);
		}

		vector<int> dequeueAny()
		{
			vector<int> ret{-1, -1, -1};
			if (!dog.empty() && !cat.empty())
			{
				if (dog.front()[2] < cat.front()[2])
				{
					ret = dog.front();
					dog.pop();
				}
				else
				{
					ret = cat.front();
					cat.pop();
				}
			}
			else if (!dog.empty())
			{
				ret = dog.front();
				dog.pop();
			}
			else if (!cat.empty())
			{
				ret = cat.front();
				cat.pop();
			}
			ret.pop_back();
			return ret;
		}

		vector<int> dequeueDog()
		{
			vector<int> ret{-1, -1};
			if (!dog.empty())
			{
				ret = dog.front();
				ret.pop_back(); // 删除入园序号
				dog.pop();
			}
			return ret;
		}

		vector<int> dequeueCat()
		{
			vector<int> ret{-1, -1};
			if (!cat.empty())
			{
				ret = cat.front();
				ret.pop_back(); // 删除入园序号
				cat.pop();
			}
			return ret;
		}
	};
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

- [面试题 04.08. 首个共同祖先](https://leetcode-cn.com/problems/first-common-ancestor-lcci/)

	递归处理，注意函数返回多个值时的处理，cpp语言实现可以采用pair/tuple等数据结构

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
					return left;
				}
				auto right = dfs(root->right, p, q);
				if (right.second != nullptr)
				{
					return right;
				}
				int count = left.first + right.first;
				if (root->val == p->val)
				{
					count++;
				}
				if (root->val == q->val)
				{
					count++;
				}
				if (count == 2)
				{
					return make_pair(count, root);
				}
				return make_pair(count, nullptr);
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

- [面试题 04.10. 检查子树](https://leetcode-cn.com/problems/check-subtree-lcci/)

	递归实现，检查t1和t2完全相同，或者t2是t1的左子树的子树，或者t2是t1的右子树的子树

	```cpp
	class Solution
	{
	private:
		bool sameTree(TreeNode *a, TreeNode *b)
		{
			return (a == nullptr && b == nullptr) || (a != nullptr && b != nullptr && a->val == b->val && sameTree(a->left, b->left) && sameTree(a->right, b->right));
		}

	public:
		bool checkSubTree(TreeNode *t1, TreeNode *t2)
		{
			return t2 == nullptr || (t1 != nullptr && (sameTree(t1, t2) || checkSubTree(t1->left, t2) || checkSubTree(t1->right, t2)));
		}
	};
	```

- [面试题 04.12. 求和路径](https://leetcode-cn.com/problems/paths-with-sum-lcci/)

    - 对于给定非空节点root，首先计算以当前节点root为起点的满足向下路径和为target的路径数量，然后递归的计算以root的左右子树为起点的路径数量求和，时间复杂度$O(n^2)$，其中$n$为给定树root的节点总数

	```cpp
	class Solution
	{
	private:
		int countCurrentNode(TreeNode *root, int cur, int target)
		{
			/**
			* 计算以当前节点root为开始节点的可能的路径和
			*/
			int ret = 0;
			if (root)
			{
				cur += root->val;
				if (cur == target)
				{
					ret += 1;
				}
				ret += countCurrentNode(root->left, cur, target) + countCurrentNode(root->right, cur, target);
			}
			return ret;
		}
		int dfsAllNodes(TreeNode *root, int target)
		{
			return root ? countCurrentNode(root, 0, target) + dfsAllNodes(root->left, target) + dfsAllNodes(root->right, target) : 0;
		}

	public:
		int pathSum(TreeNode *root, int sum)
		{
			return dfsAllNodes(root, sum);
		}
	};
	```

    - 非递归写法，DFS深度优先搜索

	```cpp
	class Solution
	{
	private:
		int countCurrentNode(TreeNode *root, int target)
		{
			/**
			* 计算以当前节点root为开始节点的可能的路径和
			*/
			int ret = 0, cur_sum = 0;
			stack<TreeNode *> st;
			TreeNode *cur = root;
			unordered_set<TreeNode *> visited;
			while (!st.empty() || cur)
			{
				if (cur)
				{
					st.push(cur);
					visited.insert(cur);
					cur_sum += cur->val;
					if (cur_sum == target)
					{
						ret++;
					}
					cur = cur->left;
				}
				else
				{
					if (st.top()->right && visited.find(st.top()->right) == visited.end())
					{
						cur = st.top()->right;
					}
					else
					{
						cur_sum -= st.top()->val;
						st.pop();
					}
				}
			}
			return ret;
		}

	public:
		int pathSum(TreeNode *root, int sum)
		{
			return root ? countCurrentNode(root, sum) + pathSum(root->left, sum) + pathSum(root->right, sum) : 0;
		}
	};
	```

- [面试题 05.01. 插入](https://leetcode-cn.com/problems/insert-into-bits-lcci/)

	题目的本质即是将M左移i位后覆盖N的第i到j位，位操作在$O(1)$时间内实现

	**注意对unsigned int取反的操作，!v是逻辑取反，~v是按位取反**

	```cpp
	int insertBits(int N, int M, int i, int j)
	{
		unsigned int mask = 0, m = M, n = N;
		for (int r = i; r <= j; r++)
		{
			mask += (1 << r);
		}
		return static_cast<int>((m << i)) + static_cast<int>(n & (~mask));
	}
	```

- [面试题 05.02. 二进制数转字符串](https://leetcode-cn.com/problems/bianry-number-to-string-lcci/)

	整数的二进制$a_na_{n-1}...a_2a_1a_0$表示$a_n*2^n+a_{n-1}*2^{n-1}+...+a_2*2^2+a_1*2^1+a_0*2^0$，则小数的二进制$0.a_1a_2...a_{n-1}a_n$表示$a_1*0.5^1+a_2*0.5^2+a_3*0.5^3+...+a_{n-1}*0.5^{n-1}+a_n*0.5^n$

	```cpp
	string printBin(double num)
	{
		string ret = "0.";
		double base = 0.5;
		while (num)
		{
			int bit = num / base;
			ret.push_back(static_cast<char>(bit + '0'));
			if (ret.length() > 32)
			{
				ret = "ERROR";
				break; // 超出32bit位宽限制
			}
			if (bit == 1)
			{
				num -= base;
			}
			base /= 2;
		}
		return ret;
	}
	```

- [面试题 05.03. 翻转数位](https://leetcode-cn.com/problems/reverse-bits-lcci/)

	经典动态规划问题，需要注意：
	1. 给定num为负数的情况下防止右移溢出（转化为unsigned int）
	2. 32 bits的int型整数不足32 bits高位要补0补够32 bits

	```cpp
	int reverseBits(int num)
	{
		// 列出num的二进制表示
		int ret = 0;
		unsigned int v = num; // 防止负数溢出
		// 统计至多含有一个0的连续1的最长序列长度, 动态规划
		int withZero = 0, withoutZero = 0;
		while (v)
		{
			if (v & 0x1)
			{
				withZero++;
				withoutZero++;
			}
			else
			{
				withZero = withoutZero + 1;
				withoutZero = 0;
			}
			ret = max(ret, withZero);
			v >>= 1;
		}
		if (ret < 32)
		{
			// 32bit的int型数据，小于32位时可以在最高位补0
			ret = max(ret, withoutZero + 1);
		}
		return ret;
	}
	```

- [面试题 05.04. 下一个数](https://leetcode-cn.com/problems/closed-number-lcci/submissions/)

	使用位操作实现，具体方法见注释，时间复杂度$O(1)$

	```cpp
	class Solution
	{
	private:
		int next_number(vector<int> &bits, int a, int b)
		{
			/**
			* 1. 从右向左遍历每一个二进制位，找到第一组连续的10位置idx并交换这两个位置的0和1，
			* 		保证新数字大于num，然后将idx右侧的1全部集中到最低位，从而使在大于numd的数字中最小，
			* 		得到的二进制表示即为下一个更大的数字
			* 2. 从右向左遍历每一个二进制位，找到第一组连续的01位置idx并交换这两个位置的0和1，
			* 		保证新数字小于num，然后将idx右侧的1全部集中到最高位，从而使在小于num的数字中最大，
			* 		得到的二进制表示即为下一个更小的数字
			*/
			const int width = 32;
			vector<int> copy_bits(bits.begin(), bits.end()); // 不改变化输入原则
			int idx = width - 1;
			while (idx > 0 && !(bits[idx] == a && bits[idx - 1] == b))
			{
				idx--;
			}
			if (idx > 0)
			{
				// 交换0/1
				swap(copy_bits[idx], copy_bits[idx - 1]);
				// 集中有效bit到最低位或者最高位
				if (a == 1)
				{
					// 求下一个更大数字，此时将idx右侧的1全部集中到最低位
					int i = width - 1, r = width - 1;
					while (i > idx)
					{
						if (copy_bits[i] == 1)
						{
							copy_bits[i] = 0;
							copy_bits[r--] = 1;
						}
						i--;
					}
				}
				else
				{
					// 求下一个更小数字，此时将idx右侧的1全部集中到最高位
					int i = idx + 1, r = idx + 1;
					while (i < width)
					{
						if (copy_bits[i] == 1)
						{
							copy_bits[i] = 0;
							copy_bits[r++] = 1;
						}
						i++;
					}
				}
				unsigned int ret = 0;
				for (auto &bit : copy_bits)
				{
					ret = (ret << 1) + bit;
				}
				return static_cast<int>(ret);
			}
			return -1; // 没有发现正确解
		}

	public:
		vector<int> findClosedNumbers(int num)
		{
			unsigned int v = num;
			const int width = 32;
			vector<int> bits(width, 0);
			for (int i = width - 1; i >= 0; i--)
			{
				bits[i] = v & 0x1;
				v >>= 1;
			}
			return {next_number(bits, 1, 0), next_number(bits, 0, 1)};
		}
	};
	```

- [面试题 05.07. 配对交换](https://leetcode-cn.com/problems/exchange-lcci/)
    
    - 对32bits的int型数据，生成32个位置的bit值(0/1)，然后逐个交换奇数位和偶数位

	```cpp
	int exchangeBits(int num)
	{
		int ret = 0, v = num, n = 32;
		vector<int> bits(n, 0);
		int i = n - 1;
		while (v && i >= 0)
		{
			bits[i--] = v & 0x1;
			v >>= 1;
		}
		for (i = 0; i < n; i += 2)
		{
			swap(bits[i], bits[i + 1]);
		}
		for (auto bit : bits)
		{
			ret = (ret << 1) + bit;
		}
		return ret;
	}
	```

    - 数学方法，用按位与的方法，保留偶数位并右移一位，保留奇数位左移一位，求和

	```cpp
	int exchangeBits(int num)
	{
		return ((num & 0xaaaaaaaa) >> 1) + ((num & 0x55555555) << 1);
	}
	```

- [面试题 08.02. 迷路的机器人](https://leetcode-cn.com/problems/robot-in-a-grid-lcci/)

	DFS搜索可行的路径，当出现超时问题时采用剪枝策略，即当遇到不可行位置时将该位置标记为障碍以剪枝对该节点的重复访问

	```cpp
	class Solution
	{
	private:
		bool dfs(vector<vector<int>> &ret, vector<vector<int>> &obstacleGrid, vector<int> &cur, vector<int> &target)
		{
			if (cur[0] < 0 || cur[0] > target[0] || cur[1] < 0 || cur[1] > target[1] || obstacleGrid[cur[0]][cur[1]] == 1)
			{
				// 超出各自区域或者该节点为障碍物
				return false;
			}
			// 当前点可以进入，没有障碍物
			ret.emplace_back(cur);
			if (cur == target)
			{
				return true;
				// 到达目标位置
			}
			// 向右走
			cur[0]++;
			if (dfs(ret, obstacleGrid, cur, target))
			{
				return true;
			}
			cur[0]--;
			// 向下走
			cur[1]++;
			if (dfs(ret, obstacleGrid, cur, target))
			{
				return true;
			}
			cur[1]--;
			// 当前位置可以进入，但是从当前位置出发继续走没有可行路径
			ret.pop_back();					  //回溯
			obstacleGrid[cur[0]][cur[1]] = 1; // 剪枝：将当前点设为障碍
			return false;
		}

	public:
		vector<vector<int>> pathWithObstacles(vector<vector<int>> &obstacleGrid)
		{
			vector<vector<int>> ret;
			if (obstacleGrid.size() > 0 && obstacleGrid[0].size() > 0)
			{
				vector<int> cur{0, 0}, target{(int)obstacleGrid.size() - 1, (int)obstacleGrid[0].size() - 1};
				dfs(ret, obstacleGrid, cur, target);
			}
			return ret;
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

- [面试题 08.06. 汉诺塔问题](https://leetcode-cn.com/problems/hanota-lcci/)

	经典的递归问题

	```cpp
	class Solution
	{
	private:
		void recursive_move(vector<int> &A, int count, vector<int> &B, vector<int> &C)
		{
			if (count == 1)
			{
				C.push_back(A.back());
				A.pop_back();
			}
			else if (count > 1)
			{
				recursive_move(A, count - 1, C, B);
				C.push_back(A.back());
				A.pop_back();
				recursive_move(B, count - 1, A, C);
			}
		}

	public:
		void hanota(vector<int> &A, vector<int> &B, vector<int> &C)
		{
			recursive_move(A, A.size(), B, C);
		}
	};
	```

- [面试题 08.07. 无重复字符串的排列组合](https://leetcode-cn.com/problems/permutation-i-lcci/)

	- DFS递归实现

	```cpp
	class Solution
	{
	private:
		void dfs(vector<string> &ret, string &ans, string const &S, vector<bool> &unused, const int n)
		{
			if (ans.length() == n)
			{
				ret.emplace_back(ans);
				return;
			}
			for (int i = 0; i < n; i++)
			{
				if (unused[i])
				{
					unused[i] = false;
					ans += S[i];
					dfs(ret, ans, S, unused, n);
					ans.pop_back();
					unused[i] = true;
				}
			}
		}

	public:
		vector<string> permutation(string S)
		{
			vector<string> ret;
			string ans;
			const int n = S.length();
			vector<bool> unused(n, true);
			dfs(ret, ans, S, unused, n);
			return ret;
		}
	};
	```

	- 字典序递归写法

	```cpp
	class Solution
	{
	private:
		void dfs(vector<string> &ret, string &S, int start, const int end)
		{
			if (start == end)
			{
				ret.emplace_back(S);
				return;
			}
			for (int i = start; i <= end; i++)
			{
				swap(S[start], S[i]);
				dfs(ret, S, start + 1, end); // 递归
				swap(S[start], S[i]);			  // 回溯
			}
		}

	public:
		vector<string> permutation(string S)
		{
			vector<string> ret;
			dfs(ret, S, 0, S.length() - 1);
			return ret;
		}
	};
	```

- [面试题 08.08. 有重复字符串的排列组合](https://leetcode-cn.com/problems/permutation-ii-lcci/)

	DFS搜索所有位置交换的可能，注意与[面试题 08.07. 无重复字符串的排列组合](https://leetcode-cn.com/problems/permutation-i-lcci/)无重复状态不同的是，在待交换位置字符相同时交换的结果是一样的，因此避免即可

	```cpp
	class Solution
	{
	private:
		void dfs(vector<string> &ret, string &s, int start, int const end)
		{
			if (start == end)
			{
				ret.emplace_back(s);
			}
			for (int i = start; i <= end; i++)
			{
				bool repeated = false;
				for (int j = start; !repeated && j < i; j++)
				{
					// 判断在s[start]这个位置是否已经有重复字符被使用
					// 该步骤可以用unordered_set实现
					if (s[i] == s[j])
					{
						repeated = true;
					}
				}
				if (!repeated)
				{
					swap(s[i], s[start]);
					dfs(ret, s, start + 1, end);
					swap(s[i], s[start]);
				}
			}
		}

	public:
		vector<string> permutation(string S)
		{
			vector<string> ret;
			dfs(ret, S, 0, S.length() - 1);
			return ret;
		}
	};
	```

- [面试题 08.11. 硬币](https://leetcode-cn.com/problems/coin-lcci/)

	动态规划，时间复杂度$O(m*n)$，其中m是兑换的钱数、n是给定硬币类型的数量

	```cpp
	int waysToChange(int n)
	{
		vector<long long> dp(n + 1, 1);
		long long mode = 1e9 + 7;
		dp[0] = 0;
		vector<int> coins{1, 2, 10, 25};
		for (int i = 1; i < 4; i++)
		{
			int v = coins[i];
			for (int j = v; j <= n; j++)
			{
				dp[j] = (dp[j] + dp[j - v]) % mode;
			}
		}
		return static_cast<int>(dp.back());
	}
	```

- [面试题 10.02. 变位词组](https://leetcode-cn.com/problems/group-anagrams-lcci/)

	对每个字符串排序，然后将相同的字符串排列在一起即可

	```cpp
	vector<vector<string>> groupAnagrams(vector<string> &strs)
	{
		const int n = strs.size();
		vector<vector<string>> strs_sorted(n, vector<string>(2));
		for (int i = 0; i < n; i++)
		{
			strs_sorted[i][0] = strs[i];
			sort(strs_sorted[i][0].begin(), strs_sorted[i][0].end());
			strs_sorted[i][1] = strs[i];
		}
		sort(strs_sorted.begin(), strs_sorted.end());
		vector<vector<string>> ret;
		int i = 0, j = 0;
		while (i < n)
		{
			vector<string> cur;
			while (j < n && strs_sorted[j][0].compare(strs_sorted[i][0]) == 0)
			{
				cur.emplace_back(strs_sorted[j++][1]);
			}
			ret.emplace_back(cur);
			i = j;
		}
		return ret;
	}
	```

- [面试题 10.10. 数字流的秩](https://leetcode-cn.com/problems/rank-from-stream-lcci/)

    - 核心是二分查找算法，两个函数实现的时间复杂度均为$O(log(n))$，可以用STL中的lower_bound和upper_bound两个函数实现

	```cpp
	class StreamRank
	{
	private:
		vector<int> nums;

	public:
		StreamRank()
		{
			nums.clear();
		}

		void track(int x)
		{
			auto position = lower_bound(nums.begin(), nums.end(), x);
			nums.insert(position, x);
		}

		int getRankOfNumber(int x)
		{
			return upper_bound(nums.begin(), nums.end(), x) - nums.begin();
		}
	};
	```

    - 二叉搜索树实现，时间复杂度$O(log(n))$

	```cpp
	class StreamRank
	{
	private:
		struct BSTNode
		{
			int count; //统计左子树及自身的节点数，默认为1
			int val;
			BSTNode *left, *right;
			BSTNode(int _val)
			{
				count = 1;
				val = _val;
				left = nullptr, right = nullptr;
			}
		};

		struct BST
		{
			BSTNode *root;
			BST()
			{
				root = nullptr;
			}

			void insert(int x)
			{
				if (root)
				{
					BSTNode *cur = root;
					while (cur)
					{
						if (cur->val == x)
						{
							cur->count++;
							break;
						}
						else if (cur->val < x)
						{
							// 大于当前节点值
							if (cur->right)
							{
								cur = cur->right; // 插入右子树
							}
							else
							{
								cur->right = new BSTNode(x);
								cur = nullptr; // 相当于break
							}
						}
						else
						{
							// 小于当前节点值
							cur->count++; // 当前节点的左子树节点数统计值加一
							if (cur->left)
							{
								cur = cur->left;
							}
							else
							{
								cur->left = new BSTNode(x);
								cur = nullptr; // 相当于break
							}
						}
					}
				}
				else
				{
					root = new BSTNode(x);
				}
			}

			int find(int x)
			{
				BSTNode *cur = root;
				int ret = 0;
				while (cur)
				{
					if (cur->val == x)
					{
						ret += cur->count;
						cur = nullptr;
					}
					else if (cur->val < x)
					{
						ret += cur->count;
						cur = cur->right;
					}
					else
					{
						cur = cur->left;
					}
				}
				return ret;
			}
		};

		BST *bst;

	public:
		StreamRank()
		{
			bst = new BST();
		}

		void track(int x)
		{
			bst->insert(x);
		}

		int getRankOfNumber(int x)
		{
			return bst->find(x);
		}
	};
	```

- [面试题 16.01. 交换数字](https://leetcode-cn.com/problems/swap-numbers-lcci/)

	- 加减计算，注意int型数据的表示范围，防止溢出

	```cpp
	vector<int> swapNumbers(vector<int> &numbers)
	{
		long long a = numbers[0], b = numbers[1];
		a = a + b;
		b = a - b;
		a = a - b;
		return {static_cast<int>(a), static_cast<int>(b)};
	}
	```

	- 异或计算

	```cpp
	vector<int> swapNumbers(vector<int> &numbers)
	{
		numbers[0] ^= numbers[1];
		numbers[1] ^= numbers[0];
		numbers[0] ^= numbers[1];
		return numbers;
	}
	```

- [面试题 16.02. 单词频率](https://leetcode-cn.com/problems/words-frequency-lcci/)

    - hashmap统计单词数量

	```cpp
	class WordsFrequency
	{
	private:
		unordered_map<string, int> count;

	public:
		WordsFrequency(vector<string> &book)
		{
			count.clear();
			for (auto &s : book)
			{
				count[s]++;
			}
		}

		int get(string word)
		{
			auto it = count.find(word);
			return it == count.end() ? 0 : it->second;
		}
	};
	```

    - TrieTree字典树实现

	```cpp
	class WordsFrequency
	{
	private:
		struct TrieNode
		{
			int is_word;
			TrieNode *next[26];
			TrieNode()
			{
				is_word = 0;
				for (int i = 0; i < 26; i++)
				{
					next[i] = nullptr;
				}
			}
		};
		TrieNode *root = new TrieNode();

	public:
		WordsFrequency(vector<string> &book)
		{
			for (auto &s : book)
			{
				TrieNode *cur = root;
				for (auto &ch : s)
				{
					int index = static_cast<int>(ch - 'a');
					if (cur->next[index] == nullptr)
					{
						cur->next[index] = new TrieNode();
					}
					cur = cur->next[index];
				}
				cur->is_word++;
			}
		}

		int get(string word)
		{
			TrieNode *cur = root;
			for (auto &ch : word)
			{
				int index = static_cast<int>(ch - 'a');
				if (cur->next[index])
				{
					cur = cur->next[index];
				}
				else
				{
					return 0;
				}
			}
			return cur->is_word;
		}
	};
	```

- [面试题 16.03. 交点](https://leetcode-cn.com/problems/intersection-lcci/)

	通过数学方法计算两条直线的交点，然后判断交点是否在给定的线段上，时间复杂度$O(1)$，特别注意两条直线平行（平行与重合）与两条直线垂直于X轴等边界情况

	```cpp
	vector<double> intersection(vector<int> &start1, vector<int> &end1, vector<int> &start2, vector<int> &end2)
	{
		vector<double> ret;
		int x1 = start1[0], y1 = start1[1], x2 = end1[0], y2 = end1[1], x3 = start2[0], y3 = start2[1], x4 = end2[0], y4 = end2[1];
		if ((y2 - y1) * (x4 - x3) == (y4 - y3) * (x2 - x1))
		{
			// 平行或者重合
			if (x1 == x2)
			{
				// 两段直线均垂直于x轴
				if (x1 == x3)
				{
					int a = min(y1, y2), b = max(y1, y2);
					int c = min(y3, y4), d = max(y3, y4);
					if (c <= b && c >= a)
					{
						ret = {static_cast<double>(x1), static_cast<double>(c)};
					}
					else if (a >= c && a <= d)
					{
						ret = {static_cast<double>(x1), static_cast<double>(a)};
					}
				}
			}
			else
			{
				// 两条直线斜率均存在且相同
				if ((y2 - y1) * (x1 - x3) == (y1 - y3) * (x2 - x1))
				{
					// 截距相同即重合
					int a = min(x1, x2), b = max(x1, x2);
					int c = min(x3, x4), d = max(x3, x4);
					if (c <= b && c >= a)
					{
						ret.emplace_back(c);
						ret.emplace_back(c == x3 ? y3 : y4);
					}
					else if (a >= c && a <= d)
					{
						ret.emplace_back(a);
						ret.emplace_back(a == x1 ? y1 : y2);
					}
				}
			}
		}
		else
		{
			// 两条直线交叉，必然有一个交点
			double x_a = (x4 - x3) * (x2 - x1) * (y3 - y1) - (y4 - y3) * (x2 - x1) * x3 + (y2 - y1) * (x4 - x3) * x1;
			double y_a = (y2 - y1) * (x4 * (y3 - y1) + x3 * (y1 - y4) + x1 * (y4 - y3));
			double b = (y2 - y1) * (x4 - x3) - (y4 - y3) * (x2 - x1);
			double x = x_a / b, y = y_a / b + y1;
			if (x >= min(x1, x2) && x <= max(x1, x2) && y >= min(y1, y2) && y <= max(y1, y2) && x >= min(x3, x4) && x <= max(x3, x4) && y >= min(y3, y4) && y <= max(y3, y4))
			{
				// 交点在线段上
				ret = {x, y};
			}
		}
		return ret;
	}
	```

- [面试题 16.04. 井字游戏](https://leetcode-cn.com/problems/tic-tac-toe-lcci/)

	逐行逐列以及正副对角线统计$'X','O','\ '$三种字符的数量，然后根据给定规则判断结果即可，时间复杂度$O(n^2)$

	```cpp
	class Solution
	{
	private:
		int charToIdx(char ch)
		{
			int idx;
			if (ch == 'X')
			{
				idx = 0;
			}
			else if (ch == 'O')
			{
				idx = 1;
			}
			else
			{
				idx = 2;
			}
			return idx;
		}

	public:
		string tictactoe(vector<string> &board)
		{
			const int n = board.size();
			vector<vector<int>> row_count(n, vector<int>(3, 0)); // 'X','O',' '
			vector<vector<int>> col_count(n, vector<int>(3, 0));
			vector<vector<int>> diagonal_count(2, vector<int>(3, 0));
			for (int i = 0; i < n; i++)
			{
				// 统计行列的情况
				for (int j = 0; j < n; j++)
				{
					row_count[i][charToIdx(board[i][j])]++;
					col_count[j][charToIdx(board[i][j])]++;
				}
				// 统计两条对角线的情况
				diagonal_count[0][charToIdx(board[i][i])]++;		 // 主对角线
				diagonal_count[1][charToIdx(board[i][n - i - 1])]++; // 副对角线
			}
			int space_count = 0;
			for (int i = 0; i < n; i++)
			{
				space_count += row_count[i][2];
				if (row_count[i][0] == n || col_count[i][0] == n)
				{
					return "X";
				}
				if (row_count[i][1] == n || col_count[i][1] == n)
				{
					return "O";
				}
			}
			if (diagonal_count[0][0] == n || diagonal_count[1][0] == n)
			{
				return "X";
			}
			else if (diagonal_count[0][1] == n || diagonal_count[1][1] == n)
			{
				return "O";
			}
			if (space_count == 0)
			{
				return "Draw";
			}
			return "Pending";
		}
	};
	```

- [面试题 16.05. 阶乘尾数](https://leetcode-cn.com/problems/factorial-zeros-lcci/)

	阶乘尾数中的0均为10，即有2和5相乘构成，因此计算从1到n中因子2和5的个数即可，而2远多于5，因此只需要统计5的因子的个数，时间复杂度$O(log(n))$

	```cpp
	int trailingZeroes(int n)
	{
		int ret = 0;
		while (n)
		{
			n /= 5;
			ret += n;
		}
		return ret;
	}
	```

- [面试题 16.10. 生存人数](https://leetcode-cn.com/problems/living-people-lcci/)

	区间统计问题，使用扫描线实现，时间复杂度$O(m+n)$，其中$m=max\_year-min\_year,n=birth.size()$

	```cpp
	int maxAliveYear(vector<int> &birth, vector<int> &death)
	{
		const int m = 2000 - 1900 + 2, n = birth.size();
		vector<int> count(m, 0);
		for (int i = 0; i < n; i++)
		{
			count[birth[i] - 1900]++;	  // 出生
			count[death[i] - 1900 + 1]--; // 死亡的第二年不再统计
		}
		int ret_idx = -1, ret_count = 0, base = 0;
		for (int i = 0; i < m; i++)
		{
			base += count[i];
			if (base > ret_count)
			{
				ret_count = base;
				ret_idx = i;
			}
		}
		return ret_idx + 1900;
	}
	```

- [面试题 16.26. 计算器](https://leetcode-cn.com/problems/calculator-lcci/)

	首先将字符串形式的中缀表达式转换为逆波兰表达式(后缀表达式)，然后再使用栈计算结果，时间复杂度$O(n)$

	```cpp
	class Solution
	{
	private:
		int priority(char op)
		{
			int ret;
			if (op == '+' || op == '-')
			{
				ret = 0;
			}
			else if (op == '*' || op == '/')
			{
				ret = 1;
			}
			return ret;
		}

	public:
		int calculate(string s)
		{
			// 中缀表达式转为后缀表达式（逆波兰表达式）
			vector<string> items;
			stack<char> ops;
			const int n = s.length();
			s.push_back('#'); // 哨兵位，保证最后的操作数被输出
			for (int i = 0; i < n; i++)
			{
				char ch = s[i];
				if (ch == '+' || ch == '-'||ch == '*' || ch == '/')
				{
					// 输出栈顶优先级高于或者等于当前ch的操作符（保证当前栈顶操作符优先级最高）
					while (!ops.empty() && priority(ops.top())>=priority(ch))
					{
						string op;
						op += ops.top();
						items.emplace_back(op);
						ops.pop();
					}
					ops.push(ch);
				}
				else if (ch <= '9' && ch >= '0')
				{
					string num;
					while (i <= n && s[i] >= '0' && s[i] <= '9')
					{
						num += s[i++];
					}
					items.emplace_back(num);
					i -= 1; //避免与for循环的i++重复
				}
			}
			while (!ops.empty())
			{
				string op;
				op += ops.top();
				items.emplace_back(op);
				ops.pop();
			}
			// 计算后缀表达式
			stack<int> st;
			for (auto &item : items)
			{
				if (item[0] >= '0' && item[0] <= '9')
				{
					st.push(stoi(item));
				}
				else
				{
					int b = st.top();
					st.pop();
					int a = st.top();
					st.pop();
					if (item == "+")
					{
						st.push(a + b);
					}
					else if (item == "-")
					{
						st.push(a - b);
					}
					else if (item == "*")
					{
						st.push(a * b);
					}
					else if (item == "/")
					{
						st.push(a / b);
					}
				}
			}
			return st.top();
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

- [面试题 17.07. 婴儿名字](https://leetcode-cn.com/problems/baby-names-lcci/)

	名字之间的同质关系可以用并查集（Union Find）来表示，合并并查集内的每一联通分量的统计频率即可，时间复杂度$O(m+nlog(n))$，其中$m=names.size(),n=synonyms.size()$

	```cpp
	class Solution
	{
	private:
		struct UF
		{
			int count;
			vector<int> uf;

			UF(int n)
			{
				count = n;
				uf.resize(n);
				for (int i = 0; i < n; i++)
				{
					uf[i] = i;
				}
			}

			int find(int x)
			{
				return uf[x] == x ? x : (uf[x] = find(uf[x]));
			}

			bool union_merge(int x, int y)
			{
				x = find(x), y = find(y);
				if (x != y)
				{
					uf[x] = y;
					count--;
					return true;
				}
				return false;
			}
		};

	public:
		vector<string> trulyMostPopular(vector<string> &names, vector<string> &synonyms)
		{
			// 解析每个名字和它的出现频率
			const int n = names.size();
			vector<string> nameIdentifier(n);
			vector<int> nameFrequency(n);
			unordered_map<string, int> identifierToIdx;
			for (int i = 0; i < n; i++)
			{
				auto item = names[i];
				auto pos = item.find('(');
				string name = item.substr(0, pos);
				nameIdentifier[i] = name;
				identifierToIdx[name] = i;
				nameFrequency[i] = stoi(item.substr(pos + 1, item.length() - pos - 2));
			}
			// 并查集表示名字间的同名关系
			UF uf = UF(n);
			for (auto &item : synonyms)
			{
				auto pos = item.find(',');
				auto nameA = item.substr(1, pos - 1), nameB = item.substr(pos + 1, item.length() - pos - 2);
				uf.union_merge(identifierToIdx[nameA], identifierToIdx[nameB]);
			}
			// 合并所有同名的名字和频率
			vector<vector<string>> groupToNames(n);
			vector<int> groupToFrequencySum(n, 0);
			for (int i = 0; i < n; i++)
			{
				int group = uf.find(i);
				groupToNames[group].emplace_back(nameIdentifier[i]);
				groupToFrequencySum[group] += nameFrequency[i];
			}
			vector<string> ret(uf.count);
			for (int i = 0, k = 0; i < n; i++)
			{
				if (groupToFrequencySum[i] != 0)
				{
					auto name = *min_element(groupToNames[i].begin(), groupToNames[i].end());
					ret[k++] = name + '(' + to_string(groupToFrequencySum[i]) + ')';
				}
			}
			return ret;
		}
	};
	```

- [面试题 17.10. 主要元素](https://leetcode-cn.com/problems/find-majority-element-lcci/)

	摩尔投票法则（赞成票与否定票互相抵消的思想），寻找占比超过一半的元素

	```cpp
	int majorityElement(vector<int> &nums)
	{
		const int n = nums.size();
		if (n > 0)
		{
			int count = 0, key = nums[0];
			for (auto &v : nums)
			{
				if (key == v)
				{
					count++;
				}
				else
				{
					if (count > 0)
					{
						count--;
					}
					else
					{
						key = v, count = 1;
					}
				}
			}
			return count > 0 ? key : -1;
		}
		return -1;
	}
	```

- [面试题 17.11. 单词距离](https://leetcode-cn.com/problems/find-closest-lcci/)

	- hashmap记录所有单词出现的idx，然后寻找单词word1和word2的idx之间最小差值

	```CPP
	int findClosest(vector<string> &words, string word1, string word2)
	{
		unordered_map<string, vector<int>> wordToIdxs;
		const int n = words.size();
		for (int i = 0; i < n; i++)
		{
			wordToIdxs[words[i]].emplace_back(i);
		}
		int ret = numeric_limits<int>::max();
		for (auto a : wordToIdxs[word1])
		{
			for (auto b : wordToIdxs[word2])
			{
				ret = min(ret, abs(a - b));
			}
		}
		return ret;
	}
	```

	- 双指针分别指向当前扫描过程中遇到的单词word1和word2

	```cpp
	int findClosest(vector<string> &words, string word1, string word2)
	{
		int a = -1, b = -1;
		const int n = words.size();
		int ret = numeric_limits<int>::max();
		for (int i = 0; i < n; i++)
		{
			if (words[i] == word1)
			{
				a = i;
			}
			else if (words[i] == word2)
			{
				b = i;
			}
			if (a != -1 && b != -1)
			{
				ret = min(ret, abs(a - b));
			}
		}
		return ret;
	}
	```

- [面试题 17.21. 直方图的水量](https://leetcode-cn.com/problems/volume-of-histogram-lcci/)

	扫描各个柱子高度的过程中维护一个单调降序栈，时间复杂度$O(n)$

	```cpp
	int trap(vector<int> &height)
	{
		const int n = height.size();
		stack<int> st; // 维护一个单调递减栈
		int ret = 0;
		for (int i = 0; i < n; ++i)
		{
			while (!st.empty() && height[i] > height[st.top()])
			{
				int top = st.top();
				st.pop();
				if (!st.empty())
				{
					ret += (min(height[i], height[st.top()]) - height[top]) * (i - st.top() - 1);
				}
			}
			st.push(i); // 当前值入栈
		}
		return ret;
	}
	```

- [...](123)
