# 701-800

- [706. 设计哈希映射](https://leetcode-cn.com/problems/design-hashmap/)

    开放链地址法

    ```cpp
    class MyHashMap
    {
    private:
        vector<vector<pair<int, int>>> data;
        static const int factor = 9973; // 10000以内最大的质数
        static int hash(int key)
        {
            return key % factor;
        }

    public:
        /** Initialize your data structure here. */
        MyHashMap()
        {
            data.clear();
            data.resize(factor);
        }

        /** value will always be non-negative. */
        void put(int key, int value)
        {
            int idx = hash(key);
            for (auto it = data[idx].begin(); it != data[idx].end(); it++)
            {
                if ((*it).first == key)
                {
                    (*it).second = value;
                    return;
                }
            }
            data[idx].emplace_back(make_pair(key, value));
        }

        /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
        int get(int key)
        {
            int idx = hash(key);
            for (auto it = data[idx].begin(); it != data[idx].end(); it++)
            {
                if ((*it).first == key)
                {
                    return (*it).second;
                }
            }
            return -1;
        }

        /** Removes the mapping of the specified value key if this map contains a mapping for the key */
        void remove(int key)
        {
            int idx = hash(key);
            for (auto it = data[idx].begin(); it != data[idx].end(); it++)
            {
                if ((*it).first == key)
                {
                    data[idx].erase(it);
                    return;
                }
            }
        }
    };
    ```

- [710. 黑名单中的随机数](https://leetcode-cn.com/problems/random-pick-with-blacklist/)

    - 黑名单拒绝采样，LeetCode评测机$TLE$

    ```cpp
    class Solution
    {
    private:
        vector<int> blacklist;
        int N;
        bool exists(int v)
        {
            int left = 0, right = blacklist.size() - 1;
            while (left <= right)
            {
                int mid = left + ((right - left) >> 1);
                if (blacklist[mid] > v)
                {
                    right = mid - 1;
                }
                else if (blacklist[mid] < v)
                {
                    left = mid + 1;
                }
                else
                {
                    return true;
                }
            }
            return false;
        }

    public:
        Solution(int N, vector<int> &blacklist)
        {
            this->N = N;
            this->blacklist = blacklist;
            sort(this->blacklist.begin(), this->blacklist.end());
        }

        int pick()
        {
            int v;
            do
            {
                v = static_cast<int>(static_cast<double>(rand()) / RAND_MAX * N);
            } while (exists(v));
            return v;
        }
    };
    ```

    - 给定区间$[0,N)$共有N个数，其中黑名单M个，则剩余可取的数为$N-M$个，将黑名单中在$rand(N-M)$范围内的数字映射到大于$N-M$且不在黑名单中即可

    ```cpp
    class Solution
    {
    private:
        unordered_map<int, int> blacklistToWhite;
        int mode;
        bool exists(vector<int> &nums, int v)
        {
            int left = 0, right = nums.size() - 1;
            while (left <= right)
            {
                int mid = left + ((right - left) >> 1);
                if (nums[mid] > v)
                {
                    right = mid - 1;
                }
                else if (nums[mid] < v)
                {
                    left = mid + 1;
                }
                else
                {
                    return true;
                }
            }
            return false;
        }

    public:
        Solution(int N, vector<int> &blacklist)
        {
            int M = blacklist.size();
            mode = N - M;
            int k = N - M;
            sort(blacklist.begin(), blacklist.end());
            for (auto &v : blacklist)
            {
                if (v < mode)
                {
                    while (exists(blacklist, k))
                    {
                        k++;
                    }
                    blacklistToWhite[v] = k++;
                }
            }
        }

        int pick()
        {
            int v = rand() % mode;
            auto it = blacklistToWhite.find(v);
            if (it != blacklistToWhite.end())
            {
                v = it->second;
            }
            return v;
        }
    };
    ```

    - 二分查找白名单中的第k个数，其中$k=rand(N-M)$

    ```cpp
    class Solution
    {
    private:
        vector<int> blacklist;
        int mode;

    public:
        Solution(int N, vector<int> &blacklist)
        {
            int M = blacklist.size();
            mode = N - M;
            this->blacklist = blacklist;
            sort(this->blacklist.begin(), this->blacklist.end());
        }

        int pick()
        {
            int k = rand() % mode;
            int left = 0, right = blacklist.size() - 1;
            while (left < right)
            {
                int mid = left + ((right - left + 1) >> 1);
                blacklist[mid] - mid > k ? right = mid - 1 : left = mid;
            }
            return left == right && blacklist[left] - left <= k ? k + left + 1 : k;
        }
    };
    ```

- [718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/)

    最长公共子序列（LCS）问题，二维动态规划，时间复杂度$O(mn)$

    ```cpp
	int findLength(vector<int> &A, vector<int> &B)
	{
		const int m = A.size(), n = B.size();
		int ret = 0;
		vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				dp[i + 1][j + 1] = A[i] == B[j] ? dp[i][j] + 1 : 0;
				ret = max(ret, dp[i + 1][j + 1]);
			}
		}
		return ret;
	}
    ```

- [719. Find K-th Smallest Pair Distance](https://leetcode.com/problems/find-k-th-smallest-pair-distance/)

	- 暴力算出所有可能的distance，排序然后取第k小的值，时间复杂度$O(n^2*log(n))$，leetcode评测机$\color{red}{TLE}$

	```cpp
	int smallestDistancePair(vector<int> &nums, int k)
	{
		vector<int> distances;
		int n = nums.size();
		for (int i = 0; i < n; i++)
		{
			for (int j = i + 1; j < n; j++)
			{
				distances.push_back(abs(nums[j] - nums[i]));
			}
		}
		sort(distances.begin(), distances.end());
		return distances[k - 1];
	}
	```

	- binary search + prefix，时间复杂度$O(n+w+nlog(n)+nlog(w))),w=max(nums)-min(nums)$

	```cpp
	int smallestDistancePair(vector<int> &nums, int k)
	{
		// O(nlog(n))
		sort(nums.begin(), nums.end());
		int n = nums.size();
		int w = nums[n - 1];
		/**
		 * 可能的答案在[0,w]之间，可以进行二分搜索
		 * prefix[v] 表示nums中 nums[i] <= v 的i的数量
		 * multiplicity[j] 表示 i<j 且 nums[i]==nums[j] 的i的个数，即nums中nums[j]前面和nums[j]值相等的元素个数
		 * 
		*/
		vector<int> prefix(w + 1, 0), multiplicity(n, 0);
		// O(n)
		for (auto i = 1; i < n; i++)
		{
			if (nums[i] == nums[i - 1])
			{
				multiplicity[i] = multiplicity[i - 1] + 1;
			}
		}
		// O(n+w)
		for (auto left = 0, v = 0; v <= w; v++)
		{
			while (left < n && nums[left] == v)
			{
				left++;
			}
			prefix[v] = left;
		}
		// O(nlog(w))
		int left = 0, right = w;
		while (left < right)
		{
			int mid = left + (right - left) / 2;
			int guess_mid = 0;
			for (int i = 0; i < n; i++)
			{
				// x=nums[i] ， 则统计在x的右侧且 0 <= x+v <= mid(guess) 的v的个数
				guess_mid += prefix[min(w, nums[i] + mid)] - prefix[nums[i]] + multiplicity[i];
			}
			if (guess_mid >= k) // binary search
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
	```

    - binary search + slide window (two pointer)，时间复杂度$O(nlog(n)+nlog(w))),w=max(nums)-min(nums)$

    ```cpp
	int smallestDistancePair(vector<int> &nums, int k)
	{
		// O(nlog(n))
		sort(nums.begin(), nums.end());
		int n = nums.size();
		int w = nums[n - 1] - nums[0];
		/**
		 * 可能的答案在[0,w]之间，可以进行二分搜索
		 * 使用滑动窗口来计算所有小于等于guess(mid)的距离对的数目
		 * 
		*/
		// O(nlog(w))
		int left = 0, right = w;
		while (left < right)
		{
			int mid = left + (right - left) / 2;
			int guess_mid = 0, slow = 0;
			for (auto fast = 0; fast < n; fast++)
			{
				// 统计 nums[fast] - nums[slow] <= mid 的数目
				while (nums[fast] - nums[slow] > mid)
				{
					slow++;
				}
				guess_mid += fast - slow;
			}
			if (guess_mid >= k) // binary search
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
    ```

- [720. Longest Word in Dictionary](https://leetcode.com/problems/longest-word-in-dictionary/)

    Trie + BFS

    ```cpp
    class Solution
    {
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
        string find(TrieNode *root)
        {
            string ans;
            if (root && root->is_word)
            {
                for (int i = 0; i < 26; i++)
                {
                    string temp = (char)('a' + i) + find(root->next[i]);
                    if (temp.length() > ans.length())
                    {
                        ans = temp;
                    }
                }
            }
            return ans;
        }

    public:
        string longestWord(vector<string> &words)
        {
            // build the Trie dictionary
            TrieNode *dictionary = new TrieNode();
            dictionary->is_word = true; // the word is "", empty
            for (auto &&word : words)
            {
                TrieNode *root = dictionary;
                for (auto &&ch : word)
                {
                    int index = (int)(ch - 'a');
                    if (!root->next[index])
                    {
                        root->next[index] = new TrieNode();
                    }
                    root = root->next[index];
                }
                root->is_word = true;
            }
            // BFS查询
            string ans = find(dictionary);
            return ans.substr(0, ans.length() - 1); // for the last excess 'a'
        }
    };
    ```

- [721. 账户合并](https://leetcode-cn.com/problems/accounts-merge/)

    并查集与hashmap的使用，时间复杂度$O(m*n)$

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
        vector<vector<string>> accountsMerge(vector<vector<string>> &accounts)
        {
            const int n = accounts.size();
            UF uf = UF(n);
            // 用并查集合并同一个用户，通过是否有相同email账号判断
            unordered_map<string, int> accountToIdx;
            for (int i = 0; i < n; i++)
            {
                for (int j = 1; j < accounts[i].size(); j++)
                {
                    if (accountToIdx.find(accounts[i][j]) != accountToIdx.end())
                    {
                        uf.union_merge(i, accountToIdx[accounts[i][j]]);
                    }
                    else
                    {
                        accountToIdx[accounts[i][j]] = i;
                    }
                }
            }
            // 将并查集内的同一个连通分量的全部email地址合并
            vector<unordered_set<string>> idxToAccounts(n);
            vector<string> idxToName(n);
            for (int i = 0; i < n; i++)
            {
                int idx = uf.find(i);
                idxToName[idx] = accounts[i][0];
                for (int j = 1; j < accounts[i].size(); j++)
                {
                    idxToAccounts[idx].insert(accounts[i][j]);
                }
            }
            vector<vector<string>> ret(uf.count);
            for (int i = 0, r = 0; i < n; i++)
            {
                if (idxToAccounts[i].size() > 0)
                {
                    ret[r].emplace_back(idxToName[i]);
                    for (auto &account : idxToAccounts[i])
                    {
                        ret[r].emplace_back(account);
                    }
                    r++;
                }
            }
            // email地址排序
            for (int i = 0; i < uf.count; i++)
            {
                sort(ret[i].begin() + 1, ret[i].end());
            }
            return ret;
        }
    };
    ```

- [722. Remove Comments](https://leetcode.com/problems/remove-comments/)

    删除c/c++风格的代码注释，分为行注释和块注释两种类型

    ```cpp
	vector<string> removeComments(vector<string> &source)
	{
		vector<string> ret;
		string s;
		for (auto &&line : source)
		{
			s += line;
			s.push_back('\n'); // 标记换行  
		}
		int i = 0, count = s.length();
		string line;
		while (i < count)
		{
			if (s[i] == '/')
			{
				if (i + 1 < count && s[i + 1] == '/')
				{
					// 行注释
					while (i < count && s[i] != '\n')
					{
						i++; // i走到行注释之后的第一个字符
					}
				}
				else if (i + 1 < count && s[i + 1] == '*')
				{
					// 块注释开始
					i += 2;
					while (i + 1 < count && !(s[i] == '*' && s[i + 1] == '/'))
					{
						i++;
					}
					i += 2; // i走到块注释之后的第一个字符
				}
				else
				{
					// 非注释项，单独出现了一个正常的 '/' 字符
					line.push_back(s[i++]);
				}
			}
			else if (s[i] == '\n')
			{
				if (!line.empty())
				{
					ret.push_back(line);
					line.clear();
				}
				i++; // i走到下一行的第一个字符
			}
			else
			{
				line.push_back(s[i++]);
			}
		}
		return ret;
	}
	```

	- some test case

	```cpp
    ["/*Test program */", "int main()", "{ ", "  // variable declaration ", "int a, b, c;", "/* This is a test", "   multiline  ", "   comment for ", "   testing */", "a = b + c;", "}"]
    ["a/*comment", "line", "more_comment*/b"]
    ["void func(int k) {", "// this function does nothing /*", "   k = k*2/4;", "   k = k/2;*/", "}"]
	```

- [724. 寻找数组的中心索引](https://leetcode-cn.com/problems/find-pivot-index/)

    runing_sum方法左右分别计算累加和，从左到右找出第一个两侧累加和相同的地方即可，时间复杂度$O(n)$

    ```cpp
	int pivotIndex(vector<int> &nums)
	{
		const int n = nums.size();
		vector<int> left_sum(n), right_sum(n);
		int left_base = 0, right_base = 0;
		for (int i = 0; i < n; i++)
		{
			left_sum[i] = left_base;
			left_base += nums[i];
			right_sum[n - 1 - i] = right_base;
			right_base += nums[n - 1 - i];
		}
		for (int i = 0; i < n; i++)
		{
			if (left_sum[i] == right_sum[i])
			{
				return i;
			}
		}
		return -1;
	}
    ```

- [725. Split Linked List in Parts](https://leetcode.com/problems/split-linked-list-in-parts/)

    将给定的单向链表截断为长度尽量相等的k组，链表节点数小于k的用空链表补足k组

    ```cpp
    vector<ListNode *> splitListToParts(ListNode *root, int k)
    {
        int count = 0;
        ListNode *cur = root;
        while (cur)
        {
            count++, cur = cur->next;
        }
        int part_id = 0, length_of_part = count / k, number_of_parts_with_additional = count % k;
        vector<ListNode *> ret;
        cur = root;
        while (part_id < k)
        {
            ListNode *head = cur, *pre = nullptr;
            int length = length_of_part;
            if (head)
            {
                while (head && length > 0)
                {
                    pre = head, head = head->next, length--;
                }
                if (head && part_id < number_of_parts_with_additional)
                {
                    pre = head, head = head->next;
                }
                pre->next = nullptr;
                ret.push_back(cur);
            }
            else
            {
                ret.push_back(nullptr);
            }
            cur = head;
            part_id++; // 分出一组到ret
        }
        return ret;
    }
    ```

- [733](https://leetcode.com/problems/flood-fill/)
    
    类似于图像处理中区域增长的方式，采用DFS递归写法或者用栈stack实现

- [738. Monotone Increasing Digits](https://leetcode.com/problems/monotone-increasing-digits/)

    将给定数字N当做字符串s，从右向左遍历s并维护一个字符的非严格降序栈（逆转之后则为非严格升序栈，符合题目要求的结果），如果当前字符小于或等于栈顶则入栈，大于则$st=string(st.size, '9')$，并将当前字符自减（类比减法中的借位思想），最后将非严格降序的字符栈翻转并转化为int型数即可，时间复杂度$O(n),n=s.length$，其中n是给定数字N的字符宽度，当限定为int型数字时$n<10$，则时间复杂度为$O(1)$

    ```cpp
    int monotoneIncreasingDigits(int N)
    {
        string s = to_string(N);
        string monotone_increasing;
        for (int i = s.length() - 1; i >= 0; i--)
        {
            char cur = s[i];
            if (!monotone_increasing.empty() && cur > monotone_increasing.back())
            {
                cur--;
                monotone_increasing = string(monotone_increasing.length(), '9');
            }
            monotone_increasing.push_back(cur);
        }
        reverse(monotone_increasing.begin(), monotone_increasing.end());
        int ret = stoi(monotone_increasing);
        return ret;
    }
    ```

- [739](https://leetcode.com/problems/daily-temperatures/)

    在给定的一段时间的温度记录中，寻找下一个比当前温度高的日期与当前日期的间隔

    - 从后往前遍历所有日期，对于每个日期，从次日向后遍历找到第一个比当前日期温度高的日期即可

    ```cpp
    vector<int> dailyTemperatures(vector<int>& T) {
		int count = T.size();
		vector<int> res(count, 0);
		for (int i = count - 2; i >= 0; i--)
		{
			int j = i + 1;
			while (T.at(i) >= T.at(j) && j < count)
			{
				if (res[j] != 0)
				{
					j += res[j];
				}
				else
				{
					j = i;
					break;
				}
			}
			res[i] = j - i;
		}
		return res;
    }
    ```

    - 用栈的思想，从后往前遍历所有日期，维护一个温度降序的日期下标栈，这样每次遍历到一个日期，只需比较其与栈顶日期的温度，即可发现其右侧第一个比当前温度高的日期，这里注意因为时降序栈，因此栈顶的温度必然小于栈底，则在栈顶第一次发现比当前日期小的温度时，必然是其右侧出现的第一个比当前温度高所有值中最小的，这样每个日期下标最多在栈中进出一次，时间复杂度为$O(n)$

    ```cpp
    vector<int> dailyTemperatures(vector<int> &T)
    {
        int const count = T.size();
        vector<int> ret(count, 0);
        stack<int> st;
        for (int i = count - 1; i >= 0; i--)
        {
            while (!st.empty() && T[i] >= T[st.top()])
            {
                st.pop();
            }
            ret[i] = st.empty() ? 0 : st.top() - i;
            st.push(i);
        }
        return ret;
    }
    ```

- [744](https://leetcode.com/problems/find-smallest-letter-greater-than-target/)

    在给定的有序（升序）小写字母序列letters中寻找第一个大于target的字母
    - 顺序扫描[$O(n)$]

    ```cpp
    char nextGreatestLetter(vector<char>& letters, char target) {
        vector<bool> count(26,false);
        for(auto &ch:letters){
            count[ch-'a']=true;
        }
        char ans='a';
        int start=target-'a'+1;
        int end=start+26;
        for (int i = start; i < end; i++)
        {
            if(count[i%26]){
                ans+=i%26;
                break;
            }
        }
        return ans;
    }
    ```

    - 二分查找[$O(log(n))$]

    ```cpp
    char nextGreatestLetter(vector<char>& letters, char target) {
        int low=0,high=letters.size();
        while(low<high){
            int mid=low+((high-low)>>1);
            if(letters[mid]<=target){
                low=mid+1;
            }else{
                high=mid;
            }
        }
        return letters[low%letters.size()];
    }
    ```

- [745. Prefix and Suffix Search](https://leetcode.com/problems/prefix-and-suffix-search/)

	对于给定的words构造前缀字典树（TrieTree），然后查询其中符合prefix和suffix限制的单词，可将每个单词word所有可能的后缀直接编码到word的前部插入TrieTree，用'{'和word分割，然后直接查询即可，时间复杂度$O(NK^2+QK)$，空间复杂度$O(NK^2)$，其中N是word的个数，K是word的最大长度，Q是查询的次数

	```cpp
	class WordFilter
	{
	private:
		struct TrieNode
		{
			int is_leaf;
			TrieNode *next[27];
			TrieNode()
			{
				is_leaf = -1;
				for (auto i = 0; i < 27; i++)
				{
					next[i] = nullptr;
				}
			}
		};
		TrieNode *root; // 按照前缀构造
		void insert(TrieNode *trie_root, string word, int weight)
		{
			TrieNode *cur = trie_root;
			for (auto &&ch : word)
			{
				int index = (int)(ch - 'a');
				if (cur->next[index] == nullptr)
				{
					cur->next[index] = new TrieNode();
				}
				cur = cur->next[index];
			}
			cur->is_leaf = weight;
		}
		int find(TrieNode *cur, string prefix, int weight)
		{
			if (cur)
			{
				if (cur->is_leaf > weight)
				{
					weight = cur->is_leaf; // 当前单词即符合查询条件
				}
				for (auto i = 0; i < 27; i++)
				{
					if (cur->next[i])
					{
						weight = find(cur->next[i], prefix, weight);
					}
				}
			}
			return weight;
		}

	public:
		WordFilter(vector<string> &words)
		{
			int count = words.size();
			root = new TrieNode();
			for (auto i = 0; i < count; i++)
			{
				int weight = i, length = words[i].length();
				// 按照可能的后缀 + # + word 构造TrieTree
				for (auto k = 0; k <= length; k++)
				{
					string cur_word = words[i].substr(length - k, k) + '{' + words[i];
					insert(root, cur_word, weight);
					// cout << cur_word << endl;
				}
			}
		}

		int f(string prefix, string suffix)
		{
			// 查找符合前缀的所有单词
			TrieNode *cur = root;
			prefix = suffix + '{' + prefix;
			for (auto &&ch : prefix)
			{
				int index = (int)(ch - 'a');
				if (cur->next[index] == nullptr)
				{
					return -1; // 没有符合该prefix的word
				}
				else
				{
					cur = cur->next[index];
				}
			}
			return find(cur, prefix, -1);
		}
	};
	```

- [747. 至少是其他数字两倍的最大数](https://leetcode-cn.com/problems/largest-number-at-least-twice-of-others/)

    寻找数组中相异的两个最大值和次大值，判断倍数关系即可，时间复杂度$O(n)$，注意数组中数字全部相同、只有一个数字等特殊情况的处理

    ```cpp
	int dominantIndex(vector<int> &nums)
	{
		int ret = -1;
		const int n = nums.size();
		if (n > 0)
		{
			int first = 0, second = 0;
			while (second < n && nums[first] == nums[second])
			{
				second++;
			}
			if (second == n)
			{
				return 0; // 数组中所有数字全部相同
			}
			if (nums[second] > nums[first])
			{
				swap(first, second);
			}
			for (int i = second + 1; i < n; i++)
			{
				if (nums[i] > nums[first])
				{
					second = first;
					first = i;
				}
				else if (nums[i] < nums[first] && nums[i] > nums[second])
				{
					second = i;
				}
			}
			if (nums[first] < 2 * nums[second])
			{
				return -1; // 最大值低于次大值的两倍
			}
			ret = first;
		}
		return ret;
	}
    ```

- [748. Shortest Completing Word](https://leetcode.com/problems/shortest-completing-word/)

    - letter count，时间复杂度$O(\sum{word.length()})$，LeetCode时间效率$\color{red}{16ms,91.56\%}$

    ```cpp
    string shortestCompletingWord(string licensePlate, vector<string> &words)
    {
        vector<int> cnt_license(26, 0);
        for (auto &&ch : licensePlate)
        {
            if (islower(ch))
            {
                cnt_license[(int)(ch - 'a')]++;
            }
            else if (isupper(ch))
            {
                cnt_license[(int)(ch - 'A')]++;
            }
        }
        string ret;
        int length = numeric_limits<int>::max();
        for (auto &&word : words)
        {
            if (word.length() < length)
            {
                vector<int> cnt_word(26, 0);
                for (auto &&ch : word)
                {
                    cnt_word[(int)(ch - 'a')]++;
                }
                int i = 0;
                while (i < 26 && cnt_word[i] >= cnt_license[i])
                {
                    i++;
                }
                if(i==26){
                    length = word.length(), ret = word;
                }
            }
        }
        return ret;
    }
    ```

    - encode by prime numbers

    ```cpp
    uint64_t getCharProduct(string s, uint64_t mode)
    {
        // the first 26 prime numbers
        vector<int> primes{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101};
        uint64_t ret = 1;
        for (auto &&ch : s)
        {
            ret = (ret * primes[(int)(ch - 'a')]) % mode;
        }
        return ret;
    }
    string shortestCompletingWord(string licensePlate, vector<string> &words)
    {
        string processed_licensePlate;
        for (auto &&ch : licensePlate)
        {
            if (isupper(ch))
            {
                processed_licensePlate += ch - 'A' + 'a';
            }
            else if (islower(ch))
            {
                processed_licensePlate += ch;
            }
        }
        uint64_t charProduct = getCharProduct(processed_licensePlate, UINT64_MAX);
        string ret = "0123456789012345";
        for (auto &&word : words)
        {
            if (word.length() < ret.length() && (getCharProduct(word, charProduct) % charProduct == 0))
            {
                ret = word;
            }
        }
        return ret;
    }
    ```

- [754](https://leetcode.com/problems/reach-a-number/)

    从$k=1$开始累加直到超过target，根据奇偶性校验的结果来修正最终值

    ```cpp
    int reachNumber(int target) {
        int k = 0;
        target = abs(target);
        while (target > 0)
        {
            target -= ++k;
        }
        return (target & 1) ? k + 1 + k % 2 : k;
    }
    ```

- [763. Partition Labels](https://leetcode.com/problems/partition-labels/submissions/)

    下标cur从前往后遍历字符串S，在当前段内每个字符在整个字符串的最后出现位置更新为last，当cur等于last时截断即可

    ```cpp
    vector<int> partitionLabels(string S)
    {
        vector<int> last(26, -1);
        for (int i = 0; i < S.length(); i++)
        {
            int index = (int)(S[i] - 'a');
            last[index] = max(i, last[index]);
        }
        int start = -1, end = 0;
        vector<int> ans;
        for (int i = 0; i < S.length(); i++)
        {
            int index = (int)(S[i] - 'a');
            if (last[index] > end)
            {
                end = last[index];
            }
            if (i == end)
            {
                ans.push_back(end - start);
                start = end;
            }
        }
        return ans;
    }
    ```

- [775. Global and Local Inversions](https://leetcode.com/problems/global-and-local-inversions/)

    注意这是一个返回bool值得判定题目，并不需要真的计算出global inversions和local inversions

    - 理论上有local inversion一定是global inversion，但是global inversion不一定是local inversion，所以只需要保证所有global inversion全部是local inversion即可返回true，时间复杂度$O(n)$

    ```cpp
    bool isIdealPermutation(vector<int> &A)
    {
        const int count = A.size() - 2;
        int cur_max = numeric_limits<int>::min();
        for (auto i = 0; i < count; i++)
        {
            cur_max = max(cur_max, A[i]);
            if (cur_max > A[i + 2])
            {
                return false;
            }
        }
        return true;
    }
    ```

    - 数学上可以证明global inversions和local inversions相同的条件是$abs(A[i] - i) \le 1$，时间复杂度$O(n)$

    ```cpp
    bool isIdealPermutation(vector<int> &A)
    {
        const int count = A.size();
        for (auto i = 0; i < count; i++)
        {
            if (abs(A[i] - i) > 1)
            {
                return false;
            }
        }
        return true;
    }
    ```

- [783. 二叉搜索树节点最小距离](https://leetcode-cn.com/problems/minimum-distance-between-bst-nodes/)

    本题与[530. 二叉搜索树的最小绝对差](https://leetcode-cn.com/problems/minimum-absolute-difference-in-bst/)完全相同，对二叉搜索树BST进行中序遍历即可

    ```cpp
	int minDiffInBST(TreeNode *root)
	{
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

- [784](https://leetcode.com/problems/letter-case-permutation/)

    对给定的字符串中的字母做大小写的全排列
    - BFS

    ```cpp
    vector<string> letterCasePermutation(string S) {
        string init;
        for (int i = 0; i < S.length(); i++)
        {
            if(isdigit(S.at(i))){
                init.push_back(S.at(i));
            }else{
                init.push_back(tolower(S.at(i)));
            }
        }
        vector<string> ans;
        ans.push_back(init);
        for (int level = 0; level < S.length(); level++)
        {
            if(isdigit(S.at(level))){
                continue;
            }else{
                int count=ans.size();
                for (int i = 0; i < count; i++)
                {
                    string next=ans[i];
                    next[level]+='A'-'a';
                    ans.push_back(next);
                }
            }
        }
        return ans;
    }

    ```

    - DFS/backtracking

    ```cpp
    vector<string> letterCasePermutation(string S) {
        vector<string> ans;
        adder(ans,S,0);
        return ans;    
    }
    void adder(vector<string>& ans,string s,int pos){
        if(pos<s.length()){
            if(isdigit(s[pos])){
                adder(ans,s,pos+1);
            }else{
                adder(ans,s,pos+1);
                if(islower(s[pos])){
                    s[pos]=toupper(s[pos]);
                }else{
                    s[pos]=tolower(s[pos]);
                }
                adder(ans,s,pos+1);
            }
        }else{
            ans.push_back(s);
        }
    }
    ```

- [785](https://leetcode.com/problems/is-graph-bipartite/)

    验证通过邻接矩阵给定的无向图是否为二部图，使用两种颜色对图中所有节点染色即可，若染色成功则，任何两个相邻节点颜色不同，则为true，否则为false

    - 递归写法

    ```cpp
    class Solution
    {
    private:
        bool dfs_helper(vector<vector<int>> &g, vector<int> &colors, int cur_node, int cur_color)
        {
            if (!g[cur_node].empty())
            {
                colors[cur_node] = cur_color;
                // 与当前节点相连的节点均需染与cur_color不同的颜色
                for (auto &&node : g[cur_node])
                {
                    if (colors[node] == cur_color)
                    {
                        return false; // 颜色冲突
                    }
                    if (colors[node] == 0 && !dfs_helper(g, colors, node, -cur_color))
                    {
                        return false; // 染色失败
                    }
                }
            }
            return true;
        }

    public:
        bool isBipartite(vector<vector<int>> &graph)
        {
            const int count = graph.size(); //  number of nodes
            vector<int> colors(count, 0);	// 0 unknown, 1 to A, -1 to B
            for (auto i = 0; i < count; i++)
            {
                if (colors[i] == 0 && !dfs_helper(graph, colors, i, 1))
                {
                    return false;
                }
            }
            return true;
        }
    };
    ```
    - 非递归/迭代式写法

    ```cpp
	bool isBipartite(vector<vector<int>> &graph)
	{
		const int count = graph.size(); //  number of nodes
		vector<int> colors(count, 0);	// 0 unknown, 1 to A, -1 to B
		for (auto i = 0; i < count; i++)
		{
			if (colors[i] == 0)
			{
				// node i 需要开始染色
				colors[i] = 1;
				queue<int> qe{{i}};
				while (!qe.empty())
				{
					int cur_node = qe.front();
					qe.pop();
					for (auto &&node : graph[cur_node])
					{
						if (colors[node] == colors[cur_node])
						{
							return false;
						}
						if (colors[node] == 0)
						{
							colors[node] = -colors[cur_node];
							qe.push(node);
						}
					}
				}
			}
		}
		return true;
	}
    ```

- [787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/)

    - 求图中从src到dst的最小花费，Bellman-Ford(DP)，时间复杂度$O(E*K)$，其中E是边数，n是节点数

    ```cpp
	int findCheapestPrice(int n, vector<vector<int>> &flights, int src, int dst, int K)
	{
		const int inf = numeric_limits<int>::max();
		const long long auxiliary = 0;
		/**
		 * dp[i][j]表示最多飞i次航班到达j位置时的最少价格，
		*/
		vector<vector<int>> dp(K + 2, vector<int>(n, inf));
		dp[0][src] = 0; // 一开始就在src位置
		for (auto i = 1; i <= K + 1; i++)
		{
			dp[i][src] = 0;
			for (auto &&x : flights)
			{
				dp[i][x[1]] = static_cast<int>(min(auxiliary + dp[i - 1][x[0]] + x[2], auxiliary + dp[i][x[1]]));
			}
		}
		return dp.back()[dst] == inf ? -1 : dp.back()[dst];
	}
	```

- [788](https://leetcode.com/problems/rotated-digits/)

    判断一个整数的每个位置上数字旋转180度之后是否可以形成一个不等于原来值的有效数字，暴力搜索$O(nlog(n))$，动态规划$O(log(n))$，[here](http://www.frankmadrid.com/ALudicFallacy/2018/02/28/rotated-digits-leet-code-788/)算法描述。

    - [暴力搜索]

    ```cpp
    int rotatedDigits(int N)
    {
        const int count = 10;
        int *rotated = new int[count];
        rotated[0] = 0;
        rotated[1] = 1;
        rotated[2] = 5;
        rotated[3] = -1;
        rotated[4] = -1;
        rotated[5] = 2;
        rotated[6] = 9;
        rotated[7] = -1;
        rotated[8] = 8;
        rotated[9] = 6;

        int count_of_good = 0;
        for (int i = 1; i <= N; i++)
        {
            int base = i;
            vector<int> digits;
            bool flag = true;
            while (base)
            {
                if (rotated[base % 10] != -1)
                {
                    digits.push_back(rotated[base % 10]);
                    base /= 10;
                }
                else
                {
                    flag = false;
                    break;
                }
            }
            if (flag)
            {
                base = 0;
                while (!digits.empty())
                {
                    base = base * 10 + digits.back();
                    digits.pop_back();
                }
                if (base != i)
                {
                    count_of_good++;
                }
            }
        }
        return count_of_good;
    }
	```

    这里同样的思路换种写法，将ans参数从引用传递改为值传递（这里体现为全局变量），在LeetCode的在线提交中时间效率可以从战胜80%提高到战胜100%，这说明cpp中值传递的效率要高于引用传递

    ```cpp
    class Solution {
    public:
        vector<string> ans;
        void dfs_helper_generateParenthesis(string cur, int left, int right)
        {
            if (left == 0 && right == 0)
            {
                ans.push_back(cur);
            }
            else
            {
                if (left > 0)
                {
                    dfs_helper_generateParenthesis(cur + '(', left - 1, right);
                }
                if (left < right && right > 0)
                {
                    dfs_helper_generateParenthesis(cur + ')', left, right - 1);
                }
            }
        }
        vector<string> generateParenthesis(int n)
        {
            dfs_helper_generateParenthesis("", n, n);
            return ans;
        }
    };
    ```

    - [dynamic plan]

	```cpp
    int myrotatedDigits(string str, bool new_number)
    {
        // some defination of number
        int type[10] = {0, 0, 1, 2, 2, 1, 1, 2, 0, 1};
        int differentRotation[10] = {0, 0, 1, 1, 1, 2, 3, 3, 3, 4}; // cumulative new number, 2 5 6 9
        int validRotation[10] = {1, 2, 3, 3, 3, 4, 5, 5, 6, 7};		// cumulative valid number, 0,1,2,5,6,8,9
        int sameRotation[10] = {1, 2, 2, 2, 2, 2, 2, 2, 3, 3};		// cumulative same number, 0, 1, 8

        // count rotations
        int digit = str[0] - '0';
        int ret = 0;

        // 1-digitals
        if (str.length() == 1)
        {
            ret = new_number ? validRotation[digit] : differentRotation[digit];
        }
        else
        {
            if (digit != 0)
            {
                ret += validRotation[digit - 1] * pow(7, str.length() - 1);
                if (!new_number)
                {
                    ret -= sameRotation[digit - 1] * pow(3, str.length() - 1);
                }
            }

            if (type[digit] == 1)
            {
                new_number = true;
            }
            if (type[digit] != 2)
            {
                ret += myrotatedDigits(str.substr(1, string::npos), new_number);
            }
        }
        return ret;
    }

    int rotatedDigits(int N)
    {
        return myrotatedDigits(to_string(N), false);
    }
    ```

- [791. 自定义字符串排序](https://leetcode-cn.com/problems/custom-sort-string/)

    将T中字符按照S中给定顺序排序即可，时间复杂度$O(nlog(n)),n=T.length()$

    ```cpp
	string customSortString(string S, string T)
	{
		const int length = 26, n = S.length();
		vector<int> order(length, length);
		for (int i = 0; i < n; i++)
		{
			order[static_cast<int>(S[i] - 'a')] = i;
		}
		sort(T.begin(), T.end(), [&order](const auto a, const auto b) -> bool { return order[static_cast<int>(a - 'a')] < order[static_cast<int>(b - 'a')]; });
		return T;
	}
    ```

- [797](https://leetcode.com/problems/all-paths-from-source-to-target/)

    本题是通过可达矩阵给出一个有向图，然后求出从0节点到达最后一个节点的所有路径，理论上DFS或者BFS遍历所有节点即可，重点在于代码的实现方式和效率，是其它图节点遍历问题的模板。

    - 递归式的深度优先遍历

    ```cpp
    void helper(vector<vector<int>> &graph, vector<vector<int>> &ans, vector<int> &cur)
    {
        if (cur.back() == graph.size() - 1)
        {
            ans.push_back(cur);
        }
        else
        {
            for (auto &&next_node : graph[cur.back()])
            {
                cur.push_back(next_node);
                helper(graph, ans, cur);
                cur.pop_back();
            }
        }
    }
    vector<vector<int>> allPathsSourceTarget(vector<vector<int>> &graph)
    {
        vector<vector<int>> ans;
        if (graph.size() > 0)
        {
            vector<int> cur{0};
            helper(graph, ans, cur);
        }
        return ans;
    }
    ```
