<!--
 * @Filename: 
 * @Author: shifaqiang
 * @Email: 14061115@buaa.edu.cn
 * @Github: https://github.com/luoboganer
 * @Date: 2020-09-05 11:29:59
 * @LastEditors: shifaqiang
 * @LastEditTime: 2021-01-18 16:45:06
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

- [...](123)
