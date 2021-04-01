# 601-700

- [605. 种花问题](https://leetcode-cn.com/problems/can-place-flowers/)
    
    - 逐个下标位置检查是否可以种花，条件为本身位置没有花且左右相邻位置也没有花或相邻位置为边界

	```cpp
	bool canPlaceFlowers(vector<int> &flowerbed, int n)
	{
		const int size = flowerbed.size();
		for (int i = 0; i < size && n > 0; i++)
		{
			if (flowerbed[i] == 0 && (i == 0 || flowerbed[i - 1] == 0) && (i + 1 == size || flowerbed[i + 1] == 0))
			{
				n--;
				flowerbed[i] = 1;
			}
		}
		return n == 0;
	}
	```

    - 【优化思路】对于当前位置已经有花儿的，下一个位置也不可能种花，i下标直接+2，当前位置为0且无法种花的，才跳到下一个位置检查

    ```cpp
	bool canPlaceFlowers(vector<int> &flowerbed, int n)
	{
		const int size = flowerbed.size();
		for (int i = 0; i < size && n > 0; i++)
		{
			if (flowerbed[i] == 0)
			{
				if ((i == 0 || flowerbed[i - 1] == 0) && (i + 1 == size || flowerbed[i + 1] == 0))
				{
					n--;
					flowerbed[i] = 1;
					i += 2;
				}
				else
				{
					i++;
				}
			}
			else
			{
				i += 2;
			}
		}
		return n == 0;
	}
    ```

- [606](https://leetcode.com/problems/construct-string-from-binary-tree/)

    重点是二叉树的非递归先序遍历 preorder

    ```cpp
    string tree2str(TreeNode *t)
    {
        // method 1, recursive

        // string s;
        // if (t)
        // {
        // 	s += t->val + '0';
        // 	if (t->right)
        // 	{
        // 		s = s + '(' + tree2str(t->left) + ")(" + tree2str(t->right) + ')';
        // 	}
        // 	else
        // 	{
        // 		if (t->left)
        // 		{
        // 			s = s + '(' + tree2str(t->left) + ')';
        // 		}
        // 	}
        // }
        // return s;

        //  method 2, non-recursive with stack

        string s;
        if (t)
        {
            stack<TreeNode *> st;
            st.push(t);						   // st.top() is the current node
            unordered_set<TreeNode *> visited; // set visited contain all visited nodes
            while (!st.empty())
            {
                t = st.top();
                if (visited.find(t) != visited.end())
                {
                    // node t has been processed
                    s += ')';
                    st.pop();
                }
                else
                {
                    // node t hasn't been processed
                    s += '(' + to_string(t->val);
                    visited.insert(t);
                    if(t->right){
                        st.push(t->right);
                    }
                    if(t->left){
                        st.push(t->left);
                    }
                    else if(t->right){
                        s+="()";
                    }
                }
            }
            s = s.substr(1, s.length() - 2);
        }
        return s;
    }
    ```

- [621. Task Scheduler](https://leetcode.com/problems/task-scheduler/)
    
    - 两个相同任务之间的冷却时间为n，则每一轮排入n+1个不同的任务即可(剩余任务不足n+1个时用待命状态来代替)，同时为了保证总的时间最少，优先排入重复次数最多的任务（贪心思想），时间复杂度$O(time)$
    
        - 排序实现贪心思想

    	```cpp
    	int leastInterval(vector<char> &tasks, int n)
    	{
    		int const length = 26;
    		vector<int> count(length, 0);
    		int total_time = 0, total_task = 0;
    		for (auto &&ch : tasks)
    		{
    			total_task++;
    			count[static_cast<int>(ch - 'A')]++;
    		}
    		while (total_task)
    		{
    			// 每一轮选择n+1个任务安排
    			sort(count.rbegin(), count.rend());
    			int i = 0;
    			while (i <= n && total_task > 0)
    			{
    				if (i < length && count[i] > 0)
    				{
    					count[i]--, total_task--; // 安排一个任务
    				}
    				total_time++, i++; // 无论是否安排任务，这一个时间片都要消耗掉
    			}
    		}
    		return total_time;
    	}
    	```

        - 优先队列实现贪心思想

    	```cpp
		int leastInterval(vector<char> &tasks, int n)
		{
			int const length = 26;
			vector<int> count(length, 0);
			for (auto &&ch : tasks)
			{
				count[static_cast<int>(ch - 'A')]++;
			}
			priority_queue<int> qe; // 大顶推
			for (auto v : count)
			{
				if (v > 0)
				{
					qe.push(v);
				}
			}
			int total_time = 0;
			while (!qe.empty())
			{
				// 每一轮选择n+1个任务安排
				int i = 0;
				vector<int> temp;
				while (i <= n && (!qe.empty() || !temp.empty()))
				{
					if (!qe.empty())
					{
						int v = qe.top() - 1;
						qe.pop();
						if (v > 0)
						{
							temp.push_back(v);
						}
						// 某个任务安排一个时间片之后，还有未安排次数，重新加入优先队列
					}
					total_time++, i++;
				}
				for (auto v : temp)
				{
					qe.push(v); // 现在才重新将未安排完的任务加入优先度列，是防止在一个eopch(n+1)内安排重复的同一个任务，破坏冷却时间的约定
				}
			}
			return total_time;
		}
    	```

    - 首先安排重复次数最多的任务，然后在产生的空闲时间内安排其他任务，时间复杂度$O(n)$

	```cpp
	int leastInterval(vector<char> &tasks, int n)
	{
		int const length = 26;
		vector<int> count(length, 0);
		for (auto &&ch : tasks)
		{
			count[static_cast<int>(ch - 'A')]++;
		}
		sort(count.rbegin(), count.rend());
		int max_value = count[0] - 1, idle_slots = max_value * n;
		for (auto i = 1; i < length && count[i] > 0; i++)
		{
			idle_slots -= min(count[i], max_value);
		}
		return tasks.size() + max(0, idle_slots);
	}
	```

- [622](https://leetcode.com/problems/design-circular-queue/)

    设计实现一个队列类，主要是队列为空或者队列为满的判断，设置front和tail两个指针，实现入队、出队、判满、判空等操作

    ```cpp
    class MyCircularQueue
    {
    private:
        vector<int> values;
        int front = 0, tail = -1, count = 0, size = 0;
        // front 指向队首元素，tail指向队尾元素

    public:
        /** Initialize your data structure here. Set the size of the queue to be k. */
        MyCircularQueue(int k)
        {
            values = vector<int>(k, 0);
            front = 0, tail = -1, count = 0, size = k;
        }

        /** Insert an element into the circular queue. Return true if the operation is successful. */
        bool enQueue(int value)
        {
            if (count < size)
            {
                tail = (tail + 1) % size;
                values[tail] = value;
                count++;
                return true;
            }
            else
            {
                return false;
            }
        }

        /** Delete an element from the circular queue. Return true if the operation is successful. */
        bool deQueue()
        {
            if (count > 0)
            {
                front = (front + 1) % size;
                count--;
                return true;
            }
            else
            {
                return false;
            }
        }

        /** Get the front item from the queue. */
        int Front()
        {
            return count > 0 ? values[front] : -1;
        }

        /** Get the last item from the queue. */
        int Rear()
        {
            return count > 0 ? values[tail] : -1;
        }

        /** Checks whether the circular queue is empty or not. */
        bool isEmpty()
        {
            return count == 0;
        }

        /** Checks whether the circular queue is full or not. */
        bool isFull()
        {
            return count == size;
        }
    };
    ```

- [639. Decode Ways II](https://leetcode.com/problems/decode-ways-ii/)

    与[91. Decode Ways](https://leetcode.com/problems/decode-ways/)不同的是，本题增加了一个可以表示1-9的通配符*

    ```cpp
    int numDecodings(string s)
    {
        const int mode = 1e9 + 7, count = s.length();
        vector<long long> dp(count + 1, 0);
        if (count > 0)
        {
            dp[0] = 1;
            dp[1] = (s[0] == '*') ? 9 : (s[0] == '0' ? 0 : 1);
            for (int i = 1; i < count; i++)
            {
                // dp[i], s[i] is a letter
                dp[i + 1] = (s[i] == '*') ? ((dp[i] * 9) % mode) : (s[i] == '0' ? 0 : dp[i]);
                // dp[i-1], s[i-1]s[i] is a letter
                if (s[i - 1] == '1')
                {
                    dp[i + 1] = (dp[i + 1] + dp[i - 1] * (s[i] == '*' ? 9 : 1)) % mode;
                }
                else if (s[i - 1] == '2')
                {
                    dp[i + 1] = (dp[i + 1] + dp[i - 1] * (s[i] == '*' ? 6 : (s[i] <= '6' && s[i] >= '0') ? 1 : 0)) % mode;
                }
                else if (s[i - 1] == '*')
                {
                    dp[i + 1] = (dp[i + 1] + dp[i - 1] * (s[i] == '*' ? 9 : 1) + dp[i - 1] * (s[i] == '*' ? 6 : (s[i] <= '6' && s[i] >= '0') ? 1 : 0)) % mode;
                }
            }
        }
        return (int)dp.back();
    }
    ```

- [643. 子数组最大平均数 I](https://leetcode-cn.com/problems/maximum-average-subarray-i/)

    简单的滑动窗口问题(slide window)，时间复杂度$O(n)$

    **注意初始化的ret值不能为0而是第一个窗口内数组的平均数，因为存在窗口内子数组的平均数为负数的情况**

    ```cpp
	double findMaxAverage(vector<int> &nums, int k)
	{
		double ret = 0, sumSubNums = 0;
		const int n = nums.size();
		for (int i = 0; i < k; i++)
		{
			sumSubNums += nums[i];
		}
		ret = sumSubNums / k; //第一个窗口内的平均数
		for (int i = k; i < n; i++)
		{
			sumSubNums += nums[i] - nums[i - k];
			ret = max(ret, sumSubNums / k);
		}
		return ret;
	}
    ```

- [648. Replace Words](https://leetcode.com/problems/replace-words/)

    给定一串单词前缀root，构建字典树，然后对给定句子中的每个单词，查询字典树，如果有前缀，则用前缀代替该单词，然后输出该句子

    ```cpp
    class Solution {
    private:
        struct TrieNode
        {
            /* data */
            bool is_word;
            TrieNode *next[26];
            TrieNode()
            {
                is_word = false;
                for (int i = 0; i < 26; i++)
                {
                    next[i] = nullptr;
                }
            }
        };
    public:
        string replaceWords(vector<string> &dict, string sentence)
        {
            // build a Trie dictionary
            TrieNode *dictionary = new TrieNode();
            for (auto &&word : dict)
            {
                TrieNode *root = dictionary;
                for (int i = 0; i < word.length(); i++)
                {
                    /* code */
                    int index = (int)(word[i] - 'a');
                    if (!root->next[index])
                    {
                        root->next[index] = new TrieNode();
                    }
                    root = root->next[index];
                }
                root->is_word = true;
            }
            // query dictionary for every word in the sentense
            string ans, item;
            stringstream ss(sentence);
            while (getline(ss, item, ' '))
            {
                TrieNode *root = dictionary;
                int root_length = -1;
                for (int i = 0; i < item.length(); i++)
                {
                    /* code */
                    if (root)
                    {
                        if (root->is_word)
                        {
                            root_length = i;
                            break;
                        }
                        else
                        {
                            int index = (int)(item[i] - 'a');
                            root = root->next[index];
                        }
                    }
                }
                if (root_length != -1)
                {
                    item = item.substr(0, root_length);
                }
                ans += item + ' ';
            }
            // return the answer
            if (ans.length() > 0)
            {
                ans = ans.substr(0, ans.length() - 1); // delete the last space
            }
            return ans;
        }
    };
    ```

- [649. Dota2 参议院](https://leetcode-cn.com/problems/dota2-senate/)

    用两个队列分别存储R和D的位置下标（投票顺序），然后从队首取值，值小的禁值大的权利，然后进入下一轮投票，直到有一方团灭

    ```cpp
	string predictPartyVictory(string senate)
	{
		queue<int> r_qe, d_qe;
		int length = senate.length();
		for (int i = 0; i < length; i++)
		{
			senate[i] == 'R' ? r_qe.push(i) : d_qe.push(i);
		}
		while (!r_qe.empty() && !d_qe.empty())
		{
			int r = r_qe.front(), d = d_qe.front();
			r_qe.pop(), d_qe.pop();
			r < d ? r_qe.push(r + length) : d_qe.push(d + length);
		}
		return r_qe.empty() ? "Dire" : "Radiant";
	}
    ```

- [654. 最大二叉树](https://leetcode-cn.com/problems/maximum-binary-tree/)

    寻找序列中的最大值，建立根节点，递归建立左子树和右子树，时间复杂度$O(n^2)$

    ```cpp
    class Solution
    {
    private:
        TreeNode *recursive_search(vector<int> &nums, int left, int right)
        {
            if (left <= right)
            {
                int max_index = max_element(nums.begin() + left, nums.begin() + right + 1) - nums.begin();
                TreeNode *root = new TreeNode(nums[max_index]);
                root->left = recursive_search(nums, left, max_index - 1);
                root->right = recursive_search(nums, max_index + 1, right);
                return root;
            }
            return nullptr;
        }

    public:
        TreeNode *constructMaximumBinaryTree(vector<int> &nums)
        {
            return recursive_search(nums, 0, nums.size() - 1);
        }
    };
    ```

- [662. Maximum Width of Binary Tree](https://leetcode.com/problems/maximum-width-of-binary-tree/)

    - BFS遍历所有非空节点，在cur_level去除两端的nullptr后计算长度，时间复杂度$O(n)$，其中n为给定树的节点数量，LeetCode提交$\color{red}{48ms,5.22\%}$

    ```cpp
    int widthOfBinaryTree(TreeNode *root)
    {
        int ret = 0;
        if (root)
        {
            vector<TreeNode *> cur_level{root};
            while (!cur_level.empty())
            {
                vector<TreeNode *> next_level;
                int left = 0, right = cur_level.size() - 1, count = cur_level.size();
                while (left < count && cur_level[left] == nullptr)
                {
                    left++;
                }
                while (right >= left && cur_level[right] == nullptr)
                {
                    right--;
                }
                if (left <= right)
                {
                    ret = max(right - left + 1, ret);
                    while (left <= right)
                    {
                        if (cur_level[left])
                        {
                            next_level.push_back(cur_level[left]->left);
                            next_level.push_back(cur_level[left]->right);
                        }
                        else
                        {
                            next_level.push_back(nullptr), next_level.push_back(nullptr);
                        }
                        left++;
                    }
                    cur_level = next_level;
                }else{
                    break;
                }
            }
        }
        return ret;
    }
    ```

    - 给任意节点指定$parent\_id$，则其左子树$left\_id=parent\_id*2$,且右子树$right\_id=parent\_id*2+1$，在DFS过程中用数组lefts记录最左子树的ID，则包含当前节点的最大width即为当前节点id与当前深度下最左子树的id只差，即$width=cur\_id+1-left[depth]$，然后在DFS过程中对width去最大值即可，时间复杂度$O(n)$，其中n为所欲节点数量，LeetCode提交效率$\color{red}{0ms,100\%}$
        - 在此过程中所有id数用unsigned int防止int型溢出

    ```cpp
    unsigned int dfs(TreeNode *root, unsigned int node_id, unsigned int depth, vector<unsigned int> &leftIDs)
    {
        unsigned int ret = 0;
        if (root)
        {
            if (leftIDs.size() <= depth)
            {
                leftIDs.push_back(node_id);
            }
            ret = max({node_id + 1 - leftIDs[depth], dfs(root->left, node_id * 2, depth + 1, leftIDs), dfs(root->right, node_id * 2 + 1, depth + 1, leftIDs)});
        }
        return ret;
    }
    int widthOfBinaryTree(TreeNode *root)
    {
        vector<unsigned int> leftIDs;
        return (int)(dfs(root, 1, 0, leftIDs));
    }
    ```

    几个测试用例

    ```cpp
    [1,3,2,5,3,null,9]
    [1,3,null,5,3]
    [1,3,2,5]
    [1,3,2,9,null,null,9,6,null,null,9]
    [0,null,0,null,0,null,***] repeat 10000
    ```

- [665. Non-decreasing Array](https://leetcode.com/problems/non-decreasing-array/)

    - 朴素方法，判断在最多修改一个数字的条件下是否给定数组可以成为非严格升序的，时间复杂度$O(n)$，注意在第一次遇到$nums[i]>nums[i+1]$的情况下，判断是否可以通过修改$nums[i]$来保证数组非严格升序，如果可以则按照符合条件的修改办法更新数组，第二次遇到$nums[i]>nums[i+1]$则直接返回$False$

    ```cpp
    bool checkPossibility(vector<int> &nums)
    {
        bool cached = false, ret = true;
        int const count = nums.size();
        for (int i = 0; i < count - 1; i++)
        {
            if (nums[i] > nums[i + 1])
            {
                if (cached)
                {
                    ret = false;
                    break;
                }
                else
                {
                    cached = true;
                    if (i == 0)
                    {
                        nums[0] = nums[1]; // nums[i]是第一个数
                    }
                    else if (i == count - 2)
                    {
                        nums[i + 1] = nums[i]; // nums[i]是倒数第二个数
                    }
                    else if (nums[i - 1] <= nums[i + 1])
                    {
                        nums[i] = nums[i - 1];
                        // nums[i]在数组中间，即 0 < i < count-1
                        // 把nums[i]降低到和nums[i-1]一样小
                    }
                    else if (nums[i] <= nums[i + 2])
                    {
                        nums[i] = nums[i + 1];
                        // nums[i]在数组中间，即 0 < i < count-2
                        // 把nums[i]提高到和nums[i+1]一样大
                    }
                    else
                    {
                        ret = false;
                        break;
                    }
                }
            }
        }
        return ret;
    }
    ```

    - 在修改的过程中只用一次遍历，实现$x>y$的情况下将x改小或者y改大，在这两种情况下只要有一种情形可以构造非递减序列即可返回true，时间复杂度$O(n)$

    ```cpp
    bool checkPossibility(vector<int> &nums)
	{
		const int n = nums.size();
		int cnt = 0;
		for (int i = 0; i < n - 1; i++)
		{
			int x = nums[i], y = nums[i + 1];
			if (x > y)
			{
				cnt++;
				if (cnt > 1)
				{
					return false; // 遇到nums[i]>nums[i]递减的情况两次及以上
				}
				if (i > 0 && y < nums[i - 1])
				{
					nums[i + 1] = x; // 修改第二个数，使用给定的一次修改机会
				}
			}
		}
		return true;
	}
    ```

    - 左右两个指针的滑动窗口实现，时间复杂度$O(n)$

    ```cpp
	bool checkPossibility(vector<int> &nums)
	{
		const int n = nums.size();
		int left = 0, right = n - 1;
		while (left < n - 1 && nums[left] <= nums[left + 1])
		{
			left++; // 过滤左侧非递减区间
		}
		if (left == n - 1)
		{
			return true; // 已经是连续非递减区间
		}
		while (right > 0 && nums[right - 1] <= nums[right])
		{
			right--; // 过滤右侧非递减区间
		}
		if (right - left >= 2)
		{
			return false; // 存在两个以上的递减区间，无法通过一次修改变成非递减区间
		}
		if (left == 0 || right == n - 1)
		{
			return true; // 唯一递减区间在左端点或者右端点，一定可以修改成非递减区间
		}
		if (nums[right + 1] >= nums[left] || nums[left - 1] <= nums[right])
		{
			// 中间存在一个递减区间且可以修改为非递减区间
			return true;
		}
		return false; // 理论上不会执行到这里
	}
    ```

    - 几个典型测试数据

    ```cpp
    [4,2,3]
    [4,2,1]
    [2,3,3,2,4]
    [3,4,2,3]
    ```

- [669](https://leetcode.com/problems/trim-a-binary-search-tree/)

    binary search tree (BST)中减掉值小于L的节点和值大于R的节点。分两步先减掉小于L的节点，第二步减掉大于R的节点。[$\color{red}{分治思想}$]

    ```cpp
    TreeNode* trimLeftBST(TreeNode* root, int value) {
        if(root){
            if(root->val < value){
                root=trimLeftBST(root->right,value);
            }else{
                root->left=trimLeftBST(root->left,value);
            }
        }
        return root;
    }
    TreeNode* trimRightBST(TreeNode* root, int value) {
        if(root){
            if(root->val > value){
                root=trimRightBST(root->left,value);
            }else{
                root->right=trimBST(root->right,value);
            }
        }
        return root;
    }
    TreeNode* trimBST(TreeNode* root, int L, int R) {
        root=trimLeftBST(root,L);
        root=trimRightBST(root,R);
        return root;
    }
    ```

- [670. Maximum Swap](https://leetcode.com/problems/maximum-swap/)

    将数字num作为字符串，从右向左记录当前最大字符的下标max_digit_index，然后由左向右找到第一个与当前最大字符不同的位置（不是当前最大字符的最高位），交换该位置与当前最大字符即可，时间复杂度为$O(n),n=num.length$，而题目限制$num \in [0,1e8]$，则$n \le 8$，时间复杂度为$O(1)$

    ```cpp
    int maximumSwap(int num)
    {
        string s = to_string(num);
        const int length = s.length();
        vector<int> max_digit_index(length, length - 1);
        for (int i = length - 2; i >= 0; i--)
        {
            max_digit_index[i] = (s[i] > s[max_digit_index[i + 1]]) ? i : (max_digit_index[i + 1]);
        }
        for (int i = 0; i < length; i++)
        {
            if (s[i] != s[max_digit_index[i]])
            {
                swap(s[i], s[max_digit_index[i]]);
                break;
            }
        }
        int ret = stoi(s);
        return ret;
    }
    ```

- [673. Number of Longest Increasing Subsequence](https://leetcode.com/problems/number-of-longest-increasing-subsequence/)

    使用动态规划，dp[i]表示数组截止到nums[i]的最长升序子序列，count[i]表示数组截止到nums[i]的最长升序子序列数量，然后找出dp中最大值，再累计该最大值在count中的数量即可

    ```cpp
    int findNumberOfLIS(vector<int> &nums)
    {
        const int n = nums.size();
        int ret = 0, max_length = 0;
        if (n > 1)
        {
            vector<int> dp(n, 0), count(n, 1);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    if (nums[i] > nums[j])
                    {
                        if (dp[i] <= dp[j])
                        {
                            dp[i] = dp[j] + 1, count[i] = count[j];
                        }
                        else if (dp[i] == dp[j] + 1)
                        {
                            count[i] += count[j];
                        }
                    }
                }
            }
            for (int i = 0; i < n; i++)
            {
                if (dp[i] > max_length)
                {
                    max_length = dp[i], ret = count[i];
                }
                else if (dp[i] == max_length)
                {
                    ret += count[i];
                }
            }
        }
        else
        {
            ret = n;
        }
        return ret;
    }
    ```

- [674. Longest Continuous Increasing Subsequence](https://leetcode.com/problems/longest-continuous-increasing-subsequence/)

    使用滑动窗口框出所有严格递增子数组，取滑动窗口最大值即可，时间复杂度$O(n)$

    ```cpp
    int findLengthOfLCIS(vector<int> &nums)
    {
        int ans = 0, cur_length = 1, count = nums.size();
        if (count > 0)
        {
            for (int i = 1; i < nums.size(); i++)
            {
                if (nums[i] > nums[i - 1])
                {
                    cur_length++;
                }
                else
                {
                    cur_length = 1;
                    ans = max(ans, cur_length);
                }
            }
            ans = max(ans, cur_length); // for first and last item
        }
        return ans;
    }
    ```

- [676. Implement Magic Dictionary](https://leetcode.com/problems/implement-magic-dictionary/)

	- 根据单词长度来基数哈希，时间复杂度$O(n*k)$，其中$k=max(word.length)$

	```cpp
	class MagicDictionary
	{
	private:
		unordered_map<int, vector<string>> magicMap;

	public:
		/** Initialize your data structure here. */
		MagicDictionary()
		{
			magicMap.clear();
		}

		/** Build a dictionary through a list of words */
		void buildDict(vector<string> dict)
		{
			for (auto &&s : dict)
			{
				auto it = magicMap.find(s.length());
				if (it != magicMap.end())
				{
					(*it).second.push_back(s);
				}
				else
				{
					magicMap[s.length()] = vector<string>{s};
				}
			}
		}

		/** Returns if there is any word in the trie that equals to the given word after modifying exactly one character */
		bool search(string word)
		{
			int length = word.length();
			for (auto &&item : magicMap[length])
			{
				int error = 0;
				for (int i = 0; i < length; ++i)
				{
					if (word[i] != item[i])
					{
						error++;
					}
				}
				if (error == 1)
				{
					return true;
				}
			}
			return false;
		}
	};
	```

    - Trie字典树实现

    ```cpp
    class MagicDictionary
    {
    private:
        struct TrieNode
        {
            bool is_word;
            TrieNode *next[26];

            TrieNode()
            {
                is_word = false;
                for (int i = 0; i < 26; i++)
                {
                    next[i] = nullptr;
                }
            }
        };

        class TrieTree
        {
        private:
            TrieNode *dictionary;

        public:
            TrieTree()
            {
                dictionary = new TrieNode();
            }
            void insert(string &word)
            {
                TrieNode *cur = dictionary;
                for (auto &ch : word)
                {
                    int idx = static_cast<int>(ch - 'a');
                    if (!cur->next[idx])
                    {
                        cur->next[idx] = new TrieNode();
                    }
                    cur = cur->next[idx];
                }
                cur->is_word = true;
            }
            bool search(string word)
            {
                TrieNode *cur = dictionary;
                const int n = word.length();
                for (int i = 0; i < n; i++)
                {
                    int idx = static_cast<int>(word[i] - 'a');
                    if (cur->next[idx] == nullptr)
                    {
                        return false;
                    }
                    cur = cur->next[idx];
                }
                return cur->is_word;
            }
        };

        TrieTree *dictionary;

    public:
        /** Initialize your data structure here. */
        MagicDictionary()
        {
            this->dictionary = new TrieTree();
        }

        void buildDict(vector<string> dictionary)
        {
            for (auto &word : dictionary)
            {
                this->dictionary->insert(word);
            }
        }

        bool search(string searchWord)
        {
            const int length = 26, n = searchWord.length();
            for (int i = 0; i < n; i++)
            {
                char ch = searchWord[i];
                for (char c = 'a'; c <= 'z'; c++)
                {
                    searchWord[i] = c;
                    if (c != ch && this->dictionary->search(searchWord))
                    {
                        return true;
                    }
                }
                searchWord[i] = ch; // 回溯
            }
            return false;
        }
    };
    ```

- [679. 24 Game](https://leetcode.com/problems/24-game/)

	DFS深度优先搜索 + 回溯

	```cpp
	class Solution
	{
	private:
		vector<double> compute(double a, double b)
		{
			// a和b两个数之间采用加减乘除有六种可能的结果
			return {a + b, a - b, b - a, a * b, a / b, b / a};
		}
		bool dfs_backtrack(vector<double> &nums)
		{
			const int n = nums.size();
			if (n == 1 && abs(nums[0] - 24) < 1e-9)
			{
				return true;
				// 递归回溯的结束条件		}
			}
			// 选择两个数进行计算，得到结果后与剩余的数接着进行递归计算
			for (auto i = 0; i < n; i++)
			{
				for (auto j = i + 1; j < n; j++)
				{
					vector<double> result = compute(nums[i], nums[j]);
					vector<double> cur_nums(n - 1);
					for (auto r = 0, k = 0; r < n; r++)
					{
						if (r != i && r != j)
						{
							cur_nums[k++] = nums[r];
						}
					}
					for (auto &&v : result)
					{
						cur_nums[n - 2] = v;
						if(dfs_backtrack(cur_nums)){
							return true;
						}
					}
				}
			}
			return false;
		}

	public:
		bool judgePoint24(vector<int> &nums)
		{
			vector<double> nums_double(nums.size());
			for (int i = 0; i < nums.size(); i++)
			{
				nums_double[i] = 1.0 * nums[i];
			}
			return dfs_backtrack(nums_double);
		}
	};
	```

- [684. Redundant Connection](https://leetcode.com/problems/redundant-connection/)

	冗余连接查找的[684. Redundant Connection](https://leetcode.com/problems/redundant-connection/)和[685. Redundant Connection II](https://leetcode.com/problems/redundant-connection-ii/)基本原理相同，其中前者为无向图，后者为有向图，使用并查集来查找冗余的边即可

	```cpp
	class Solution
	{
	private:
		struct UF
		{
			vector<int> uf;
			int count;
			UF(int n)
			{
				uf.resize(n);
				for (auto i = 0; i < n; i++)
				{
					uf[i] = i;
				}
				count = n;
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
		vector<int> findRedundantConnection(vector<vector<int>> &edges)
		{
			int n = edges.size();
			UF uf(n);
			vector<int> ret;
			for (auto &&e : edges)
			{
				if (!uf.union_merge(e[0] - 1, e[1] - 1))
				{
					ret = e;
				}
			}
			return ret;
		}
	};
	```

- [685. Redundant Connection II](https://leetcode.com/problems/redundant-connection-ii/)

	在有向图中，N个节点形成树结构需要N-1条边，在冗余一条边的情况下，会存在两种情况：

    - 所有节点的入度和出度均为1，此时退化为与无向图条件下[684. Redundant Connection](https://leetcode.com/problems/redundant-connection/)相同的问题
    - 存在一个根节点入度为0，存在一个节点入度为2（**冗余出现了**），此时对这个入度为2的节点node又有两种情况：
        - node的出度为0，则指向该节点的两条边任意删除一条即可
        - node的出度为1，则需要在指向该节点的两条边中选择一条删除使其剩余的边构成一颗数，任意删除可能会造成全图不连通（**连通域的个数大于1**）的情况

	冗余连接查找的[684. Redundant Connection](https://leetcode.com/problems/redundant-connection/)和[685. Redundant Connection II](https://leetcode.com/problems/redundant-connection-ii/)基本原理相同，其中前者为无向图，后者为有向图，使用并查集来查找冗余的边即可

	```cpp
	class Solution
	{
	private:
		struct UF
		{
			vector<int> uf;
			int count;
			UF(int n)
			{
				uf.resize(n);
				for (auto i = 0; i < n; i++)
				{
					uf[i] = i;
				}
				count = n;
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
		vector<int> findRedundantDirectedConnection(vector<vector<int>> &edges)
		{
			// 统计所有节点的入度与出度信息
			int n = edges.size(), end_point = -1;
			vector<int> indegrees(n, 0), outdegrees(n, 0);
			for (auto &&e : edges)
			{
				outdegrees[e[0] - 1]++, indegrees[e[1] - 1]++;
				if (indegrees[e[1] - 1] == 2)
				{
					end_point = e[1];
				}
			}
			// initialization of UF(union find)
			UF uf(n);
			vector<int> ret;
			if (end_point == -1)
			{
				// 没有入度为2的节点，此时全图成为一个环，删除任意一条边使得全图成为一棵树即可
				// 并查集的应用
				for (auto &&e : edges)
				{
					if (!uf.union_merge(e[0] - 1, e[1] - 1))
					{
						ret = e;
					}
				}
			}
			else
			{
				// 记录指向入度为2的节点end_point的两条边
				vector<int> first_edge, second_edge;
				/**
				* 在这两条边中只合并第一条，检查最终全图是否联通
				* 	1. 如果连通，则删除第二条即可
				* 	2. 如果不连通，则需要删除第一条，此时一定是合并第二条的情况下全图连通
				*/
				bool is_first_edge = true;
				for (auto &&e : edges)
				{
					if (e[1] == end_point)
					{
						if (is_first_edge)
						{
							first_edge = e, is_first_edge = false;
							uf.union_merge(e[0] - 1, e[1] - 1);
						}
						else
						{
							second_edge = e;
						}
					}
					else
					{
						// 其它的全部边均需合并
						uf.union_merge(e[0] - 1, e[1] - 1);
					}
				}
				ret = uf.count == 1 ? second_edge : first_edge;
			}
			return ret;
		}
	};
	```
    
    - some test cases

	```cpp
	[[1,2],[1,3],[2,3]]
	[[1,2], [2,3], [3,4], [4,1], [1,5]]
	[[2,1],[3,1],[4,2],[1,4]]
	```

- [687. Longest Univalue Path](https://leetcode.com/problems/longest-univalue-path/)

    求给定二叉树中节点值相同的连续路径最大长度

    ```cpp
    int ret;
    int count(TreeNode *root)
    {
        int ans = 0;
        if (root)
        {
            int left = count(root->left), right = count(root->right);
            int countLeft = 0, countRight = 0;
            if (root->left && root->val == root->left->val)
            {
                countLeft += left + 1;
            }
            if (root->right && root->val == root->right->val)
            {
                countRight += right + 1;
            }
            ret = max(ret, countLeft + countRight);
            ans = max(countLeft, countRight);
        }
        return ans;
    }
    int longestUnivaluePath(TreeNode *root)
    {
        ret = 0;
        count(root);
        return ret;
    }
    ```

- [692. Top K Frequent Words](https://leetcode.com/problems/top-k-frequent-words/)

    hashmap统计+buckte sort，时间复杂度$O(n)$

    ```cpp
    vector<string> topKFrequent(vector<string> &words, int k)
    {
        unordered_map<string, int> counts;
        for (auto &&word : words)
        {
            counts[word]++;
        }
        int n = words.size();
        vector<vector<string>> buckets(n, vector<string>{});
        for (auto &&item : counts)
        {
            buckets[item.second].push_back(item.first);
        }
        reverse(buckets.begin(), buckets.end());
        auto it = buckets.begin();
        vector<string> ret(k);
        int index = 0;
        for (auto &&item : buckets)
        {
            sort(item.begin(), item.end());
            for (int i = 0; i < item.size() && index < k; i++)
            {
                ret[index++] = item[i];
            }
            if (index == k)
            {
                break;
            }
        }
        return ret;
    }
    ```

- [696](https://leetcode.com/problems/count-binary-substrings/)

    首先对连续的0和1进行分组并统计个数，然后相邻的一组a个0和b个1，可以构成$min(a,b)$个符合条件的子串，然后求和即可。

    ```cpp
    int countBinarySubstrings(string s)
    {
        int ans=0,count=1;
        int length=s.length();
        if(length>0){
            vector<int> groups;
            for (int i = 1; i < length; i++)
            {
                if(s[i-1]==s[i]){
                    count++;
                }else{
                    groups.push_back(count);
                    count=1;
                }
            }
            groups.push_back(count);
            if(groups.size()>1){
                for (int i = 1; i < groups.size(); i++)
                {
                    ans+=min(groups[i-1],groups[i]);
                }
            }
        }
        return ans;
    }
    ```

- [697](https://leetcode.com/problems/degree-of-an-array/)
    
    给定一个非空非负数组$nums$，该数组中出现次数最多的数字出现的次数（即最高频数）称之为该数组的$degree$，求该数组的一个连续子段$nums[i]-nums[j]$使得该子段长度最小（$min(j-i+1)$）且与原数组有相同的$degree$。
    
    因为给定数组元素$nums[i] \in [0,50000]$，因此可以统计$[0,50000]$范围内每个数在原数组中出现的频率以及最左位置和最右位置，然后对所有频率最高的数中，按照端点位置计算子段长度并取较小值即可，时间复杂度$O(max(n,50000))$

    ```cpp
    int findShortestSubArray(vector<int>& nums) {
        const int length=1e5;
        vector<vector<int>> count(length,vector<int>(3,0));
        for (int i = 0; i < nums.size(); i++)
        {
            if(count[nums[i]][0]==0){
                count[nums[i]][0]=1;
                count[nums[i]][1]=i;
                count[nums[i]][2]=i;
            }else{
                count[nums[i]][0]+=1;
                count[nums[i]][2]=i;
            }
        }
        int cur_degree=0,ans=nums.size();
        for (int i = 0; i < length; i++)
        {
            if(count[i][0]>cur_degree){
                ans=count[i][2]-count[i][1]+1;
                cur_degree=count[i][0];
            }else if(count[i][0]==cur_degree){
                ans=min(ans,count[i][2]-count[i][1]+1);
            }
        }
        return ans;
    }
    ```
