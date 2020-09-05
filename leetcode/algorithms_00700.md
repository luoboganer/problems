# 601-700

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

    判断在最多修改一个数字的条件下是否给定数组可以成为非严格升序的，时间复杂度$O(n)$，注意在第一次遇到$nums[i]>nums[i+1]$的情况下，判断是否可以通过修改$nums[i]$来保证数组非严格升序，如果可以则按照符合条件的修改办法更新数组，第二次遇到$nums[i]>nums[i+1]$则直接返回$False$

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

    几个典型测试数据

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
