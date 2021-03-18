# 0-100

- [1](https://leetcode.com/problems/two-sum/)

    题目要求在一个数组中寻找两个和为给定值的数，暴力枚举时两层遍历的时间复杂度为$O(n^2)$，因此使用**unorder_map**的接近$O(1)$的查询效率来实现$O(n)$一遍扫描的做法，这里注意cpp中STL template **unorder_map** 的用法

    ```cpp
    vector<int> twoSum(vector<int>& nums, int target) {
		vector<int> res(2, 0);
		unordered_map<int, int> tmp;
		int count = nums.size();
		for (int i = 0; i < count; i++)
		{
			int diff = target - nums[i];
			unordered_map<int, int>::iterator got = tmp.find(diff);
			if (got != tmp.end())
			{
				res[0] = got->second;
				res[1] = i;
				break;
			}
			else
			{
				tmp.insert(pair<int, int>{nums[i], i});
			}
		}
		return res;
    }
    ```

- [4](https://leetcode.com/problems/median-of-two-sorted-arrays/)

    给定两个长度分别为m和n的有序数组，返回两个数组合并后新数组的中位数

    - 思路一：两个有序数组归并排序后返回下标中间值一个数即可，时间复杂度$O(log(m+n))$

    ```cpp
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        vector<int> nums;
        int i=0,j=0,length_a=nums1.size(),length_b=nums2.size();
        while(i<length_a && j<length_b){
            if(nums1[i]<nums2[j]){
                nums.push_back(nums1[i++]);
            }else{
                nums.push_back(nums2[j++]);
            }
        }
        while(i<length_a){
            nums.push_back(nums1[i++]);
        }
        while (j<length_b)
        {
            nums.push_back(nums2[j++]);
        }
        int length=nums.size();
        double ans=0.0;
        if(length&1){
            ans=nums[length>>1];
        }else{
            ans=(nums[length>>1]+nums[(length>>1)+1])/2.0;
        }
        return ans;
    }
    ```

    - 思路二：因为两个给定数组都是有序的，因此可以通过二分查找来实现寻找中位数，这样可以将时间复杂度降到$O(log(min(m,n)))$

    ```cpp
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int m=nums1.size(),n=nums2.size();
        double ans=0;
        if(m>n){
            // m<n时交换两个数组，保证第一个二分搜索的数组长度不小于第二个
            vector<int> temp=nums1;
            nums1=nums2;
            nums2=temp;
            swap(m,n);
        }
        int start=0,end=m,mid=(m+n+1)/2;
        while(start<=end){
            int i=(start+end)/2;
            int j=mid-i;
            // 此时可以保证i左侧和j左侧的所有数字数量，i右侧与j右侧的所有数字的数量相同
            // 接下来通过二分查找保证 max(nums1[i-1],nums2[j-1])<=min(nums1[i],nums2[j])
            if(i<end && nums2[j-1]>nums1[i]){
                // i is lower, move right
                start=i+1;
            }else if(i>start && nums1[i-1]>nums2[j]){
                // i is upper, move left
                end=i-1;
            }else{
                // i刚好合适，但也有特殊情况是i到达数组nums1的边界
                int max_left=0; // 左半部分最大值
                if(i==0){
                    max_left=nums2[j-1];
                }else if(j==0){
                    max_left=nums1[i-1];
                }else{
                    max_left=max(nums1[i-1],nums2[j-1]);
                }
                if((m+n)%2==1){
                    ans=max_left;
                    break;
                }else{
                    int min_right;
                    if(i==m){
                        min_right=nums2[j];
                    }else if(j==n)
                    {
                        min_right=nums1[i];
                    }else{
                        min_right=min(nums1[i],nums2[j]);
                    }
                    ans=(max_left+min_right)/2.0;
                    break;
                }
            }
        }
        return ans;
    }
    ```

- [5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)

    在给定字符串s中求其最长回文子串

    - 动态规划（DP），时间复杂度$O(n^2)$，dp[i][j]表示$s_i,...,s_j$是回文子串，则状态转移方程为：
    $$dp[i][j]=\left\{\begin{matrix}True, dp[i+1][j-1] \ and \ s_i==s_j\\ False, otherwise \end{matrix}\right.$$

    ```cpp
    string longestPalindrome(string s)
    {
        string ret;
        const int length = s.length();
        if (length > 0)
        {
            vector<vector<int>> dp(length, vector<int>(length, 0));
            int start = 0, substring_length = 1;
            for (int i = length - 1; i >= 0; i--)
            {
                dp[i][i] = 1;
                for (int j = i + 1; j < length; j++)
                {
                    dp[i][j] = ((dp[i + 1][j - 1] || (j - i < 3)) && s[i] == s[j]) ? 1 : 0;
                    if (dp[i][j] && (j - i + 1 > substring_length))
                    {
                        start = i, substring_length = j - i + 1;
                    }
                }
            }
            ret = s.substr(start, substring_length);
        }
        return ret;
    }
    ```

    - 以s中任何一字符为中心，向两边扩展求其回文子序列，时间复杂度$O(n^2)$

        - 基本实现

        ```cpp
        string longestPalindrome(string s)
        {
            string ret;
            const int length = s.length();
            // 奇数长度的回文子串
            for (int i = 0; i < length; i++)
            {
                int left = i, right = i, valid_start = i, valid_end = i;
                while (left >= 0 && right < length && s[left] == s[right])
                {
                    valid_start = left, valid_end = right;
                    left--, right++;
                }
                if (valid_end - valid_start + 1 > ret.length())
                {
                    ret = s.substr(valid_start, valid_end - valid_start + 1);
                }
            }
            // 偶数长度的回文子串
            for (int i = 0; i < length; i++)
            {
                int left = i, right = i + 1, valid_start = i, valid_end = i;
                while (left >= 0 && right < length && s[left] == s[right])
                {
                    valid_start = left, valid_end = right;
                    left--, right++;
                }
                if (valid_end - valid_start + 1 > ret.length())
                {
                    ret = s.substr(valid_start, valid_end - valid_start + 1);
                }
            }
            return ret;
        }
        ```

        - 一种改进的实现

        ```cpp
        vector<int> lengthOfExporedSubstring(string s, int left, int right)
        {
            int valid_left = left, valid_right = left, length = s.length();
            while (left >= 0 && right < length && s[left] == s[right])
            {
                valid_left = left, valid_right = right;
                left--, right++;
            }
            return {valid_left, valid_right};
        }
        string longestPalindrome(string s)
        {
            int start = 0, end = 0;
            for (int i = 0; i < s.length(); i++)
            {
                vector<int> explored_odd = lengthOfExporedSubstring(s, i, i);
                if (explored_odd[1] - explored_odd[0] > end - start)
                {
                    start = explored_odd[0], end = explored_odd[1];
                }
                vector<int> explored_even = lengthOfExporedSubstring(s, i, i + 1);
                if (explored_even[1] - explored_even[0] > end - start)
                {
                    start = explored_even[0], end = explored_even[1];
                }
            }
            return s.substr(start, end - start + 1);
        }
        ```

    - 马拉车算法（Manacher's algorithm）[简书](https://www.jianshu.com/p/392172762e55)、[知乎专栏](https://zhuanlan.zhihu.com/p/62351445)、[Wikipad](https://en.wikipedia.org/wiki/Longest_palindromic_substring#Manacher%27s_algorithm)，时间复杂度$O(n)$

    ```cpp
    string longestPalindrome(string s)
    {
        string ret;
        if (s.length() < 2)
        {
            ret = s;
        }
        else
        {
            // 预处理字符串
            string t = "#";
            for (auto &&ch : s)
            {
                t.push_back(ch), t.push_back('#');
            }
            t.push_back('%');
            // 计算表示回文半径的p数组
            int length = t.length(), mx = 0, id = 0, max_length = -1, index = 0;
            vector<int> p(length, 0);
            for (int i = 1; i < length - 1; i++)
            {
                p[i] = (mx > i) ? min(p[2 * id - i], mx - i) : 1;
                while (i + p[i] < length && i - p[i] >= 0 && t.at(i + p[i]) == t.at(i - p[i]))
                {
                    p[i]++;
                }
                // 如果回文子串的右边界超过了mx，则需要更新mx和id的值
                if (mx < p[i] + i)
                {
                    id = i, mx = p[i] + i;
                }
                // 如果回文子串的长度大于maxLength，则更新maxLength和index的值
                if (p[i] - 1 > max_length)
                {
                    max_length = p[i] - 1;
                    index = i;
                }
            }
            int start = (index - max_length) / 2;
            ret = s.substr(start, max_length);
        }
        return ret;
    }
    ```

- [6. Z 字形变换](https://leetcode-cn.com/problems/zigzag-conversion/)

    - 按照Z字形的一竖一拐为一组，将字符串分组后填充到矩阵中，然后逐行读取组成新的结果字符串，时间效率$\color{red}{5.4\%,124ms}$
    
        **注意字符总数小于给定行数或者自由一行的特殊边界情况处理**

    ```cpp
	string convert(string s, int numRows)
	{
		const int n = s.length();
		if (n <= numRows || numRows == 1)
		{
			return s;
		}
		int sizeOfCharPerGroup = 2 * numRows - 2;
		int numOfGroups = (int)ceil(s.length() * 1.0 / sizeOfCharPerGroup);
		int numCols = numOfGroups * (numRows - 1);
		vector<vector<char>> matrix(numRows, vector<char>(numCols, ' '));
		for (int group = 0, r = 0; group < numOfGroups; group++)
		{
			int i = 0, j = group * (numRows - 1);
			while (r < n && i < numRows)
			{
				matrix[i++][j] = s[r++];
			}
			i -= 2, j += 1;
			while (r < n && i > 0)
			{
				matrix[i--][j++] = s[r++];
			}
		}
		string ret;
		for (int i = 0; i < numRows; i++)
		{
			for (int j = 0; j < numCols; j++)
			{
				if (matrix[i][j] != ' ')
				{
					ret.push_back(matrix[i][j]);
				}
			}
		}
		return ret;
	}
    ```

    - 直接确定新字符串中每个位置该是原字符串中的下标，时间效率$\color{red}{96\%,8ms}$

    ```cpp
	string convert(string s, int numRows)
	{
		const int n = s.length();
		string ret;
        ret.resize(n);
		if (n <= numRows || numRows == 1)
		{
			ret = s;
		}
		else
		{
			int sizeOfPerGroup = 2 * numRows - 2;
			int numGroups = (int)(ceil(1.0 * n / sizeOfPerGroup));
			int r = 0;
			// 第一行
			for (int idx = 0; idx < n; idx += sizeOfPerGroup)
			{
				ret[r++] = s[idx];
			}
			// 中间行
			for (int row = 1; row <= numRows - 2; row++)
			{
				for (int idx = row; idx < n; idx += sizeOfPerGroup)
				{
					ret[r++] = s[idx];
					int next_idx = 2 * (numRows - row - 1) + idx;
					if (next_idx < n)
					{
						ret[r++] = s[next_idx];
					}
				}
			}
			// 最后一行
			for (int idx = numRows - 1; idx < n; idx += sizeOfPerGroup)
			{
				ret[r++] = s[idx];
			}
		}
		return ret;
	}
    ```

- [8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/)

    字符串处理问题，时间复杂度$O(n)$，主要是各种边界问题的处理

    ```cpp
	int myAtoi(string s)
	{
		long long ret = 0;
		int i = 0, n = s.size();
		bool negative = false;
		while (s[i] == ' ')
		{
			i++;
		}
		if (s[i] == '-' || s[i] == '+' || (s[i] >= '0' && s[i] <= '9'))
		{
			if (s[i] == '-')
			{
				i++;
				negative = true;
			}
			else if (s[i] == '+')
			{
				i++;
			}
			while (s[i] == '0')
			{
				i++;
				// 过滤所有可能存在的前导0
			}
			// 开始转换连续剩余数字
			int width = 0; // 10进制宽度达到11位必然超出signed int类型32bit的表示范围
			while (width <= 11 && i < n && s[i] >= '0' && s[i] <= '9')
			{
				ret = ret * 10 + static_cast<int>(s[i++] - '0');
				width++;
			}
			if (negative && (-ret) <= numeric_limits<int>::min())
			{
				return numeric_limits<int>::min();
			}
			else if (!negative && ret >= numeric_limits<int>::max())
			{
				return numeric_limits<int>::max();
			}
		}
		// 其它情况无法转换直接返回0
		return static_cast<int>(negative ? -ret : ret);
	}
    ```

- [10. Regular Expression Matching](https://leetcode.com/problems/regular-expression-matching/)

	- 递归回溯

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
			// 首先查看第一个元素是否匹配
			bool first_match = (!s.empty()) && (s[0] == p[0] || p[0] == '.');
			if (p.size() >= 2 && p[1] == '*')
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
		int s_length = s.length(), p_length = p.length();
		vector<vector<int>> dp(s_length + 1, vector<int>(p_length + 1, 0)); // default dp[i][j]=false
		dp[0][0] = true;													// s和p均为空的时候matched
		for (auto j = 2; j <= p_length; j++)
		{
			dp[0][j] = p[j - 1] == '*' && dp[0][j - 2];
		}
		for (auto i = 0; i < s_length; i++)
		{
			for (auto j = 0; j < p_length; j++)
			{
				if (p[j] == '*')
				{
					dp[i + 1][j + 1] = dp[i + 1][j - 1] || (dp[i][j + 1] && (s[i] == p[j - 1] || p[j - 1] == '.'));
				}
				else
				{
					dp[i + 1][j + 1] = dp[i][j] && (s[i] == p[j] || p[j] == '.');
				}
			}
		}
		return dp.back().back();
	}
	```

	- some test cases

	```cpp
	"aa"
	"a"
	"aa"
	"a*"
	"ab"
	".*"
	"aab"
	"c*a*b"
	"mississippi"
	"mis*is*p*."
	```

- [11](https://leetcode.com/problems/container-with-most-water/)

    在一组通过数组给定高度的立柱中选择两根立柱，使得中间的空间可以装水最多，即在具有高度$\{a_0,a_1,...,a_n\}$的这些柱子中选择两根$a_i,a_j$使得$min(a_i,a_j)*(j-i)$最大。

    ```cpp
    int maxArea(vector<int>& height) {
        int ans=0;
        int i=0,j=height.size()-1;
        while(i<j){
            ans=max(ans,min(height[i],height[j])*(j-i));
            if(height[i]<height[j]){
                i++;
            }else{
                j--;
            }
        }
        return ans;
    }
    ```

- [13](https://leetcode.com/problems/roman-to-integer/)

    罗马数字转阿拉伯数字，主要是思想是对罗马数字字符序列从右到左扫描，注意IXC的位置和表示的数字有关即可。

- [18. 四数之和](https://leetcode-cn.com/problems/4sum/)

    - 两两组（在此过程中用hashmap记录两个数的idx），规约到两数之和为target的问题（注意去重操作），时间复杂度为$O(n^2)$，但是涉及到诸多hash操作，实际运行效率低

    ```cpp
	vector<vector<int>> fourSum(vector<int> &nums, int target)
	{
		unordered_map<int, vector<vector<int>>> twoSum2elements;
		const int n = nums.size();
		sort(nums.begin(), nums.end());
		for (int i = 0; i < n; i++)
		{
			for (int j = i + 1; j < n; j++)
			{
				twoSum2elements[nums[i] + nums[j]].push_back({i, j});
			}
		}
		vector<int> twoSum;
		for (auto &item : twoSum2elements)
		{
			twoSum.emplace_back(item.first);
		}
		sort(twoSum.begin(), twoSum.end());
		int i = 0, j = twoSum.size() - 1;
		set<vector<int>> ret_set;
		vector<vector<int>> ret;
		while (i <= j)
		{
			int v = twoSum[i] + twoSum[j];
			if (v < target)
			{
				i++;
			}
			else if (v > target)
			{
				j--;
			}
			else
			{
				for (auto &a : twoSum2elements[twoSum[i]])
				{
					for (auto &b : twoSum2elements[twoSum[j]])
					{
						unordered_set<int> idx{a[0], a[1], b[0], b[1]};
						if (idx.size() == 4)
						{
							vector<int> cur{nums[a[0]], nums[a[1]], nums[b[0]], nums[b[1]]};
							sort(cur.begin(), cur.end());
							ret_set.insert(cur);
						}
					}
				}
				i++, j--;
			}
		}
		for (auto &v : ret_set)
		{
			ret.emplace_back(v);
		}
		return ret;
	}
    ```

    - 排序加双指针，时间复杂度$O(n^3)$

    ```cpp
    vector<vector<int>> fourSum(vector<int> &nums, int target)
	{
		vector<vector<int>> ret;
		sort(nums.begin(), nums.end());
		const int n = nums.size();
		for (int a = 0; a < n; a++)
		{
			if (a > 0 && nums[a] == nums[a - 1])
			{
				continue; // 去重
			}
			for (int b = a + 1; b < n; b++)
			{
				if (b > a + 1 && nums[b] == nums[b - 1])
				{
					continue;
				}
				int c = b + 1, d = n - 1;
				while (c < d)
				{
					// 排序+双指针减少一重循环
					int v = nums[a] + nums[b] + nums[c] + nums[d];
					if (v == target)
					{
						vector<int> cur{nums[a], nums[b], nums[c], nums[d]};
						ret.emplace_back(cur);
						c++, d--;
						while (c < d && nums[c] == nums[c - 1])
						{
							c++;
						}
						while (c < d && nums[d] == nums[d + 1])
						{
							d--;
						}
					}
					else if (v < target)
					{
						c++;
					}
					else if (v > target)
					{
						d--;
					}
				}
			}
		}
		return ret;
	}
    ```

- [19. 删除链表的倒数第N个节点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

    - 首先遍历链表计算总的节点数count，然后第二趟从head开始前进count-n步即为倒数第n个节点

    ```cpp
    ListNode *removeNthFromEnd(ListNode *head, int n)
	{
		int count = 0;
		ListNode *cur = head;
		while (cur)
		{
			count++, cur = cur->next;
		}
		if (n == count)
		{
			head = nullptr;
		}
		else if (n < count)
		{
			cur = head;
			int step = count - n - 1;
			while (step--)
			{
				cur = cur->next;
			}
			cur->next = cur->next->next;
		}
		return head;
	}
    ```

    - 双指针方法，first前进n步，然后second从头结点开始同时前进，则first到尾结点时second指向倒数第二个节点

    ```cpp
	ListNode *removeNthFromEnd(ListNode *head, int n)
	{
		ListNode *auxiliary = new ListNode(0);
		auxiliary->next = head;
		ListNode *slow = auxiliary, *fast = auxiliary;
		while (n--)
		{
			fast = fast->next;
		}
		while (fast->next)
		{
			slow = slow->next, fast = fast->next;
		}
		slow->next = slow->next->next;
		return auxiliary->next;
	}
    ```

- [22](https://leetcode.com/problems/generate-parentheses/)

    本题给定左右括号对数量n，生成所有符合条件的括号数

    - 方法一，回溯+剪枝，回溯的时间复杂度高达$O(4^n)$，但是剪枝策略可以有效降低实际运行时间

    ```cpp
    void dfs_helper_generateParenthesis(string cur, int left, int right, vector<string> &ans)
    {
        if (left == 0 && right == 0)
        {
            ans.push_back(cur);
        }
        else
        {
            if (left > 0)
            {
                dfs_helper_generateParenthesis(cur + '(', left - 1, right, ans);
            }
            if (left < right && right > 0)
            {
                dfs_helper_generateParenthesis(cur + ')', left, right - 1, ans);
            }
        }
    }
    vector<string> generateParenthesis(int n)
    {
        vector<string> ans;
        dfs_helper_generateParenthesis("", n, n, ans);
        return ans;
    }
    ```
    
    同样的思路，这里将参数ans的传递方式从引用传递改为值传递（这里体现为全局变量）的时候，在LeetCode在线提交的时间效率从80%提高到beat 100%，这说明在cpp中值传递的效率高于引用传递。

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

    - 方法二，dynamic plan

    用$dp[i]$表示含有i对括号的结果，则$dp[i]$可以从$dp[i-1]$的结果添加一对括号即可，在$i-1$对括号的基础上添加第$i$对括号，将左括号放在最坐标，然后在剩下的位置选择合适位置插入右括号即可，状态转移方程为:
    $$dp[i]='('+dp[j]+')'+dp[i-j-1], j \in \{0,1,2,3,...,i-1\}$$

    ```cpp
    vector<string> generateParenthesis(int n)
    {
        vector<vector<string>> dp;
        dp.push_back(vector<string>{""});
        if (n > 0)
        {
            for (int i = 1; i <= n; i++)
            {
                vector<string> cur;
                // dp[i]= ( + dp[j] + ) +dp[i-j-1]
                for (int j = 0; j < i; j++)
                {
                    for (auto &&item1 : dp[j])
                    {
                        for (auto &&item2 : dp[i - j - 1])
                        {
                            cur.push_back('(' + item1 + ')' + item2);
                        }
                    }
                }
                dp.push_back(cur);
            }
        }
        return dp.back();
    }
    ```

- [23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

    - 两两合并，时间复杂度$O(n^2)$

    ```cpp
    class Solution
    {
    private:
        ListNode *mergerTwoLists(ListNode *a, ListNode *b)
        {
            ListNode *auxiliary = new ListNode(0);
            ListNode *cur = auxiliary;
            while (a && b)
            {
                if (a->val < b->val)
                {
                    cur->next = a;
                    a = a->next;
                }
                else
                {
                    cur->next = b;
                    b = b->next;
                }
                cur = cur->next;
            }
            while (a)
            {
                cur->next = a;
                cur = cur->next;
                a = a->next;
            }
            while (b)
            {
                cur->next = b;
                cur = cur->next;
                b = b->next;
            }
            return auxiliary->next;
        }

    public:
        ListNode *mergeKLists(vector<ListNode *> &lists)
        {
            queue<ListNode *> qe;
            for (auto &head : lists)
            {
                qe.push(head);
            }
            while (qe.size() > 0)
            {
                if (qe.size() >= 2)
                {
                    ListNode *a = qe.front();
                    qe.pop();
                    ListNode *b = qe.front();
                    qe.pop();
                    qe.push(mergerTwoLists(a, b));
                }
                else if (qe.size() == 1)
                {
                    return qe.front();
                }
            }
            return nullptr;
        }
    };
    ```

    - 优先队列（小顶堆）实现一次合并，时间复杂度$O(nlog(n))$

    ```cpp

    ```

- [25. Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)

    翻转链表的进化版，考察链表操作的基本功

    ```cpp
    ListNode *reverseKGroup(ListNode *head, int k)
    {
        ListNode *start = head, *tail = head, *new_head = new ListNode(0), *new_prev = new_head;
        new_head->next = head;
        while (tail)
        {
            int count = k;
            while (tail && count > 0)
            {
                tail = tail->next, count--;
            }
            if (count == 0)
            {
                // 翻转k个 [start,tail)节点，左闭右开
                ListNode *cur = start, *pre = nullptr;
                while (cur != tail)
                {
                    // 翻转当前段的k个节点
                    ListNode *next = cur->next;
                    cur->next = pre;
                    pre = cur;
                    cur = next;
                }
                new_prev->next = pre; //上一段的结尾(new_prev->next)连接到当前段的开始(pre)
                new_prev = start;     //当前段的结尾(start是当前段的开始，翻转之后变成当前段的结尾)
                start = tail;         // 起始指针指向下一段
                // cout << listNodeToString(pre) << endl;
            }
            else
            {
                // 到了当前段(结尾)不够k个节点，直接从上一段结尾连接到当前段的开始
                new_prev->next = start;
                break;
            }
        }
        return new_head->next;
    }
    ```

- [29. Divide Two Integers](https://leetcode.com/problems/divide-two-integers/)

    - 本题特别注意边界条件特别是超出signed int的处理
    - 不用内置符号的两数之间的算术运算，加法用按位异或、按位与运算，减法转化为加法，除法用左右位移操作

    ```cpp
    int divide(int dividend, int divisor)
    {
        long long ans = 0, coef = 1, a = abs((long long)dividend), b = abs((long long)divisor);
        if (dividend < 0)
        {
            coef *= -1;
        }
        if (divisor < 0)
        {
            coef *= -1;
        }
        while (a >= b)
        {
            int width = 0;
            while (a >= (b << width))
            {
                width++;
            }
            ans += ((long long)1 << (width - 1));
            a -= (b << (width - 1));
        }
        ans *= coef;
        ans = max(ans, (long long)numeric_limits<int>::min());
        ans = min(ans, (long long)numeric_limits<int>::max());
        return (int)ans;
    }
    ```

- [30. Substring with Concatenation of All Words](https://leetcode.com/problems/substring-with-concatenation-of-all-words/)

	- 在s中暴力搜索所有长度符合要求的substring是否符合要求（恰好可以有words中所有单词concatenation构成），时间复杂度$O(n*words.size)$

	```cpp
	vector<int> findSubstring(string s, vector<string> &words)
	{
		vector<int> ret;
		if (words.size() > 0 && s.length() >= words.size() * words[0].length())
		{
			unordered_map<string, int> count;
			for (auto &&word : words)
			{
				count[word]++;
			}
			int n = s.length(), number_words = words.size(), length_word = words[0].length();
			int length_substring = number_words * length_word;
			int right_most_index = n - length_substring;
			for (auto i = 0; i <= right_most_index; i++)
			{
				unordered_map<string, int> count_temp = count;
				bool flag = true;
				for (auto j = 0; j < length_substring; j += length_word)
				{
					string cur = s.substr(i + j, length_word);
					auto it = count_temp.find(cur);
					if ((it != count_temp.end()) && (it->second > 0))
					{
						it->second--;
					}
					else
					{
						// 保证words中的word在组成substring时够用
						flag = false;
						break;
					}
				}
				if (flag)
				{
					for (auto &&item : count_temp)
					{
						if (item.second != 0)
						{
							// 保证words中word在组成substring时刚好全部用完
							flag = false;
							break;
						}
					}
				}
				if (flag)
				{
					ret.push_back(i);
				}
			}
		}
		return ret;
	}
	```

	- sliding window (two pointer)，时间复杂度$O(n*words[0].length)$

	```cpp
	vector<int> findSubstring(string s, vector<string> &words)
	{
		vector<int> ret;
		if (words.size() > 0 && s.length() >= words.size() * words[0].length())
		{
			unordered_map<string, int> count;
			for (auto &&word : words)
			{
				count[word]++;
			}
			int n = s.length(), number_words = words.size(), length_word = words[0].length();
			int length_substring = number_words * length_word;
			for (auto k = 0; k < length_word; k++)
			{
				unordered_map<string, int> slide_window;
				int cnt = 0; // 统计slide_window中有多少单词是给定的count里面的
				for (auto i = k; i + length_word <= n; i += length_word)
				{
					if (i >= length_substring)
					{
						// slide windows的左断点向右滑动length_word长度，去掉一个单词
						string left_end = s.substr(i - length_substring, length_word);
						slide_window[left_end]--;
						if (slide_window[left_end] < count[left_end])
						{
							cnt--;
						}
					}
					// slide_window右端点向右滑动新加入一个单词
					string cur = s.substr(i, length_word);
					slide_window[cur]++;
					if (slide_window[cur] <= count[cur])
					{
						cnt++;
					}
					if (cnt == number_words)
					{
						ret.push_back(i + length_word - length_substring);
					}
				}
			}
		}
		return ret;
	}
	```

- [32. Longest Valid Parentheses](https://leetcode.com/problems/longest-valid-parentheses/)

    在'('和')'组成的字符串中求出最长有效（左右括号匹配）括号串的长度

    - 用栈st来存储当前有效'('的下标，然后遇到匹配的')'则当前有效串长为下标$index-st.top()$，时间复杂度$O(n)$，空间复杂度$O(n)$

    ```cpp
    int longestValidParentheses(string s)
    {
        int ans = 0;
        stack<int> st;
        st.push(-1);
        for (int i = 0; i < s.length(); i++)
        {
            if (s[i] == '(')
            {
                st.push(i);
            }
            else
            {
                st.pop();
                if (st.empty())
                {
                    st.push(i);
                }
                else
                {
                    ans = max(ans, i - st.top());
                }
            }
        }
        return ans;
    }
    ```

    - 用left、right两个变量来统计当前有效地'('和')'数量，时间复杂度$O(n)$，空间复杂度$O(1)$

    ```cpp
    int singleDirection(string s, char ch)
    {
        int ans = 0, left = 0, right = 0;
        for (int i = 0; i < s.length(); i++)
        {
            if (s[i] == ch)
            {
                left++;
            }
            else
            {
                right++;
            }
            if (left == right)
            {
                ans = max(ans, right * 2);
            }
            else if (left < right)
            {
                left = 0, right = 0;
            }
        }
        return ans;
    }
    int longestValidParentheses(string s)
    {
        int ans = singleDirection(s,'(');
        reverse(s.begin(), s.end());
        ans = max(ans, singleDirection(s,')'));
        return ans;
    }
    ```

- [33](https://leetcode.com/problems/search-in-rotated-sorted-array/)

    在有序的旋转数组$(eg,{4,5,6,7,0,1,2,3})$中查找一个数target，首先二分查找确定数组的起点pivot，然后第二次二分查找确定target的index。

    ```cpp
    int search(vector<int>& nums, int target) {
        int const n=nums.size();
        int left=0,right=n-1;
        while(left<right){
            int mid=(left+right)>>1;
            if(nums[mid]<nums[right]){
                right=mid;
            }else{
                left=mid+1;
            }
        }
        int pivot=left;
        left=0,right=n-1;
        while(left<=right){
            int mid=(left+right)>>1;
            int index=(mid+pivot)%n;
            if(target<nums[index]){
                right=mid-1;
            }else if(target>nums[index]){
                left=mid+1;
            }else{
                return index;
            }
        }
        return -1;
    }
    ```

- [34. Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
    
    - 使用库函数**lower_bound()/upper_bound()**

	```cpp
	vector<int> searchRange(vector<int> &nums, int target)
	{
		int left = lower_bound(nums.begin(), nums.end(), target) - nums.begin();
		int right = upper_bound(nums.begin(), nums.end(), target) - nums.begin();
		if (left != right)
		{
			return {left, right - 1};
		}
		return {-1, -1};
	}
	```

    - 手写binary_search

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
		vector<int> searchRange(vector<int> &nums, int target)
		{
			int left = binary_search(nums, target, 0);
			int right = binary_search(nums, target, 1);
			if (left != right)
			{
				return {left, right - 1};
			}
			return {-1, -1};
		}
	};
	```

- [37. Sudoku Solver](https://leetcode.com/problems/sudoku-solver/)

    填满一个数独表格，每一个格子有9中可能，用DFS的方式尝试、回溯即可，时间复杂度$O(9^k),k \le 81$，其中$k$是给定数独表中待填充空格的数量

    ```cpp
    bool sudoku_dfs(vector<vector<int>> &sudoku, int index, vector<vector<bool>> &row, vector<vector<bool>> &col, vector<vector<bool>> &cell)
    {
        bool ret = false;
        if (index < 81)
        {
            int i = index / 9, j = index % 9;
            if (sudoku[i][j] == 0)
            {
                for (int v = 1; v <= 9; v++)
                {
                    if (row[i][v] && col[j][v] && cell[i / 3 * 3 + j / 3][v])
                    {
                        row[i][v] = false, col[j][v] = false, cell[i / 3 * 3 + j / 3][v] = false;
                        sudoku[i][j] = v;
                        if (sudoku_dfs(sudoku, index + 1, row, col, cell))
                        {
                            ret = true;
                            break;
                        }
                        else
                        {
                            // backtracking
                            row[i][v] = true, col[j][v] = true, cell[i / 3 * 3 + j / 3][v] = true;
                            sudoku[i][j] = 0;
                        }
                    }
                }
            }
            else
            {
                ret = sudoku_dfs(sudoku, index + 1, row, col, cell);
            }
        }
        else
        {
            ret = true; // 9*9=81个格子全部填满了
        }
        return ret;
    }
    void solveSudoku(vector<vector<char>> &board)
    {
        const int size = 9;
        vector<vector<int>> sudoku(size, vector<int>(size, 0));
        vector<vector<bool>> row(size, vector<bool>(size + 1, true)), col(size, vector<bool>(size + 1, true)),
            cell(size, vector<bool>(size + 1, true));
        // build the matrix
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                if (board[i][j] != '.')
                {
                    int v = (int)(board[i][j] - '0');
                    sudoku[i][j] = v;
                    row[i][v] = false, col[j][v] = false, cell[i / 3 * 3 + j / 3][v] = false;
                }
            }
        }
        sudoku_dfs(sudoku, 0, row, col, cell);
        // convert the matrix to sudoku board
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                board[i][j] = (char)('0' + sudoku[i][j]);
            }
        }
    }
    ```

- [39](https://leetcode.com/problems/combination-sum/)

    给定一个数组candidates，没有重复数字，从中选出一些数字（可以重复使用）使其和为给定的target，找出所有可能的组合，这里用递归的方式解决，即遍历candidates中的数字，对于任意的$candidates[i]$递归求解满足和为$target-candidates[i]$的所有可能值即可，递归结束条件为$target=0$

    ```cpp
    class Solution
    {
    private:
        void dfs(vector<vector<int>> &ret, vector<int> &cur, vector<int> &candidates, int target, int idx)
        {
            if (target == 0)
            {
                ret.emplace_back(cur);
            }
            else if (target > 0)
            {
                const int n = candidates.size();
                for (int i = idx; i < n; i++)
                {
                    cur.push_back(candidates[i]);
                    dfs(ret, cur, candidates, target - candidates[i], i);
                    cur.pop_back();
                }
            }
        }

    public:
        vector<vector<int>> combinationSum(vector<int> &candidates, int target)
        {
            vector<vector<int>> ret;
            vector<int> cur;
            dfs(ret, cur, candidates, target, 0);
            return ret;
        }
    };
    ```

- [40](https://leetcode.com/problems/combination-sum-ii/)

    本题与[39](https://leetcode.com/problems/combination-sum/)的区别有两点，一是给定数字有重复，二是所有给定数字只允许使用一次，因此需要特别注意去重

    ```cpp
    void dfs_helper(vector<int> &candidates, int target, int index, vector<int> &cur, vector<vector<int>> &ans)
    {
        if (target == 0)
        {
            ans.push_back(cur);
        }
        else if (target > 0)
        {
            for (int i = index; i < candidates.size(); i++)
            {
                if (i > index && candidates[i] == candidates[i - 1])
                {
                    continue; // skip duplications
                }
                cur.push_back(candidates[i]);
                dfs_helper(candidates, target - candidates[i], i + 1, cur, ans);
                cur.pop_back();
            }
        }
    }
    vector<vector<int>> combinationSum2(vector<int> &candidates, int target)
    {
        vector<int> cur;
        vector<vector<int>> ans;
        sort(candidates.begin(), candidates.end());
        dfs_helper(candidates, target, 0, cur, ans);
        return ans;
    }
    ```

- [41](https://leetcode.com/problems/first-missing-positive/)

    寻找给定数组nums中第一个缺失的正整数，首先初始化一个长度相同的全false数组flags，然后线性扫描nums中每一个数，如果是正整数x且在1到nums.length()范围内，则在flags[x]中标记为true，然后线性扫描flags找到第一个false即可，如果限定在O(1)空间内，则想到用nums数组本身的下标就是0到n-1可以作为标记，线性扫描nums将正整数nums[i]交换到nums[nums[i]-1]即可，然后第二遍扫描确定第一个i+1和nums[i]不匹配的值即可，特别注意空数组等边界测试条件

    ```cpp
    int firstMissingPositive(vector<int> &nums)
    {
        const int count = nums.size(); 
        for (int i = 0; i < count; i++)
        {
            if(i+1!=nums[i]){
                while(nums[i]>=1 && nums[i]<=count && nums[i]!=nums[nums[i]-1]){
                    swap(nums[i], nums[nums[i] - 1]);
                }
            }
        }
        int ans = 1; // for empty numbers, the 1 is the correct answer
        for (int i = 0; i < count; i++)
        {
            if(nums[i]!=i+1){
                ans = i + 1;
                break;
            }else{
                ans = count + 1;
            }
        }
        return ans;
    }
    ```

    这是典型的测试用例

    ```cpp
    [1,2,0]
    [3,4,-1,1]
    [7,8,9,11,12]
    []
    [1]
    ```

- [42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)

	维护一个单调递减栈，栈中存储的是单调递减的下标index，时间复杂度$O(n)$

	```cpp
	int trap(vector<int> &height)
	{
		const int n = height.size();
		stack<int> st;
		int curIndex = 0, ret = 0;
		while (curIndex < n)
		{
			// 维护一个单调递降栈，栈中存储单调递降的index
			while (!st.empty() && height[curIndex] > height[st.top()])
			{
				int top = st.top();
				st.pop();
				if (!st.empty())
				{
					ret += (min(height[curIndex], height[st.top()]) - height[top]) * (curIndex - st.top() - 1);
				}
			}
			st.push(curIndex);
		}
		return ret;
	}
	```

- [44. Wildcard Matching](https://leetcode.com/problems/wildcard-matching/)

	- 动态规划，根据字符匹配的规则合理设定状态转移方程即可，时间复杂度$O(s.length*p.length)$

	```cpp
	bool isMatch(string s, string p)
	{
		int m = s.length(), n = p.length();
		vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
		dp[0][0] = true;
		for (auto j = 0; j < n; j++)
		{
			/**
			 * 1. s为空且模式串p不为空的情况
			 * 2. p为空且s不为空的情况一定为False 
			 * */
			dp[0][j + 1] = dp[0][j] && p[j] == '*';
		}
		vector<bool> wild_match(dp[0]); // wild_match[j]表示p[0,j-1]和s[0,i]是否匹配
		for (auto i = 0; i < m; i++)
		{
			for (auto j = 0; j < n; j++)
			{
				dp[i + 1][j + 1] = ((p[j] == '?' || p[j] == s[i]) && dp[i][j]) || (p[j] == '*' && wild_match[j]);
				wild_match[j + 1] = wild_match[j + 1] || dp[i + 1][j + 1];
			}
		}
		return dp[m][n];
	}
	```

	- greedy algorithm，时间复杂度$O(max(s.length,p.length))$

	```cpp
	bool isMatch(string s, string p)
	{
		int m = s.length(), n = p.length();
		int i = 0, j = 0, matched = 0, starIdx = -1;
		while (i < m)
		{
			// * was found, only advancing pattern pointer
			if (j < n && p[j] == '*')
			{
				starIdx = j++;
				matched = i;
			}
			// advancing both pointer
			else if (j < n && (p[j] == '?' || p[j] == s[i]))
			{
				i++, j++;
			}
			// last pattern pointer was *, advancing string pattern
			else if (starIdx != -1)
			{
				// 当前字符不匹配时，将string中的s[i]匹配给pattern中发现的上一个*
				i = ++matched;
				j = starIdx + 1;
			}
			else
			{
				// 不匹配且没有通配符*
				return false;
			}
		}
		// 检查pattern中剩余的字符，这是s已经为空，则p剩余部分为空或者全为*
		while (j < n && p[j] == '*')
		{
			j++;
		}
		return j == n;
	}
	```

- [45](https://leetcode.com/problems/jump-game-ii/)

    Jump Game([55](https://leetcode.com/problems/jump-game/))判断是否可以到达右侧终点，本题演化为求到达右端点的最小代价（步数），同样是贪心的思维（DP动态规划会TLE），从左到右扫描一遍即可，时间复杂度$O(n)$，也是一种隐式的BFS（宽度优先搜索），$i==curEnd$即表示扫描了当前level，而curFurthest是当前level的size，即当前level可以到达的最右侧端点。

    ```cpp
    int jump(vector<int>& nums) {
        const int count=nums.size()-1;
        int steps=0,curEnd=0,curFurthest=0;
        for (int i = 0; i < count; i++)
        {
            curFurthest=max(curFurthest,i+nums[i]);
            if(i==curEnd){
                steps++;
                curEnd=curFurthest;
            }
        }
        return steps;
    }
    ```

- [46](https://leetcode.com/problems/permutations/)

    - 注意全排列的实现，递归的和非递归的，字典序的和非字典序的
    - cpp的STL中有*next_permutation*和*prev_permutation*两个函数，注意他们的实现方式

    ```cpp
    class Solution
    {
    private:
        void dfs(vector<vector<int>> &ret, vector<int> &nums, int start, const int end)
        {
            if (start == end)
            {
                ret.emplace_back(nums);
            }
            else
            {
                for (int i = start; i <= end; i++)
                {
                    swap(nums[i], nums[start]);
                    dfs(ret, nums, start + 1, end);
                    swap(nums[i], nums[start]);
                }
            }
        }

    public:
        vector<vector<int>> permute(vector<int> &nums)
        {
            vector<vector<int>> ret;
            dfs(ret, nums, 0, nums.size() - 1);
            return ret;
        }
    };
    ```

- [47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

    与[46](https://leetcode.com/problems/permutations/)不同的是，注意消除重复数字的交换

    ```cpp
    class Solution
    {
    private:
        void dfs(vector<vector<int>> &ret, vector<int> &nums, int start, const int end)
        {
            if (start == end)
            {
                ret.emplace_back(nums);
            }
            for (int i = start; i <= end; i++)
            {
                bool repeated = false;
                for (int j = start; !repeated && j < i; j++)
                {
                    if (nums[i] == nums[j])
                    {
                        repeated = true;
                    }
                }
                if (!repeated)
                {
                    swap(nums[i], nums[start]);
                    dfs(ret, nums, start + 1, end);
                    swap(nums[i], nums[start]);
                }
            }
        }

    public:
        vector<vector<int>> permuteUnique(vector<int> &nums)
        {
            vector<vector<int>> ret;
            dfs(ret, nums, 0, nums.size() - 1);
            return ret;
        }
    };
    ```

- [48](https://leetcode.com/problems/rotate-image/)

    Rotate Image，旋转图片90度，即将一个二维数组原地旋转90度。
    
    - 我的蠢办法，冥思苦想半小时，Debug又是半小时，仔细设计每次的坐标变换即可，按照每次旋转一个环（一圈），圈内从左到右一层一层旋转，每次基本的旋转单元只有四个数

    ```cpp
    void rotate(vector<vector<int>>& matrix) {
        const int n=matrix.size();
        int circles=n/2;
        for (int circle = 0; circle < circles; circle++)
        {
            int levels=n-circle*2-1; // 内层循环次数
            levels+=circle; // circle是内层循环开始点
            for (int level = circle; level < levels; level++)
            {
                int top=circle,left=level;
                int tmp=matrix[circle][level];
                matrix[circle][level]=matrix[n-1-level][circle];
                matrix[n-1-level][circle]=matrix[n-1-circle][n-1-level];
                matrix[n-1-circle][n-1-level]=matrix[level][n-1-circle];
                matrix[level][n-1-circle]=tmp;
            }
        }
    }
    ```

    - 大神的方案，令人虎躯一震，数学好真的是可以为所欲为 [Rotate Image Solution](https://leetcode.com/problems/rotate-image/discuss/18872/A-common-method-to-rotate-the-image)

    ```cpp
    void rotate(vector<vector<int>> &matrix)
    {
        reverse(matrix.begin(), matrix.end());
        for (int i = 0; i < matrix.size(); i++)
        {
            for (int j = i + 1; j < matrix[i].size(); j++)
            {
                swap(matrix[i][j], matrix[j][i]);
            }
        }
    }
    ```

- [49. Group Anagrams](https://leetcode.com/problems/group-anagrams/)

    将给定的一串单词按照组成字符分组

    - 统计每个单词中每个字母的数量并逐个对比，$\color{red}{TLE}$

    ```cpp
	bool check(vector<int> &a, vector<int> &b)
	{
		bool ret = true;
		if (a.size() != b.size())
		{
			ret = false;
		}
		else
		{
			for (int i = 0; i < a.size(); i++)
			{
				if (a[i] != b[i])
				{
					ret = false;
					break;
				}
			}
		}
		return ret;
	}
	vector<vector<string>> groupAnagrams(vector<string> &strs)
	{
		vector<vector<string>> ret;
		int length_of_strings = strs.size(), number_of_lowercase = 26;
		vector<vector<int>> count(length_of_strings, vector<int>(number_of_lowercase, 0));
		vector<bool> used(length_of_strings, false);
		// 统计每个字符串中每个小写字母出现的次数
		for (int i = 0; i < length_of_strings; i++)
		{
			for (auto x : strs[i])
			{
				count[i][int(x - 'a')]++;
			}
		}
		// grouping
		for (int i = 0; i < length_of_strings; i++)
		{
			if (!used[i])
			{
				used[i] = true;
                vector<string> anagrams_of_cur_string{strs[i]};
				for (int j = i + 1; j < length_of_strings; j++)
				{
					if (!used[j])
					{
						if (check(count[i], count[j]))
						{
							used[j] = true;
                            anagrams_of_cur_string.push_back(strs[j]);
						}
					}
				}
				ret.push_back(anagrams_of_cur_string);
			}
		}
		return ret;
	}
    ```

    - 将每个单词中所有字母排序并用哈希表存储，时间复杂度$O(nklogk)$，其中$n$是给定单词数量，$k$是单词的最大长度

    ```cpp
    vector<vector<string>> groupAnagrams(vector<string> &strs)
    {
        unordered_map<string, vector<string>> records;
        vector<vector<string>> ret;
        for (auto &&s : strs)
        {
            string temp = s;
            sort(temp.begin(), temp.end());
            records[temp].push_back(s);
        }
        for (auto &&item : records)
        {
            ret.push_back(item.second);
        }
        return ret;
    }
    ```

- [51. N-Queens](https://leetcode.com/problems/n-queens/)

    给定n，求所有可能的N皇后排列，N皇后的规则为N个皇后放入N*N的棋盘中使其不能互相攻击：
    - 同行会互相攻击
    - 同列会互相攻击
    - 同斜线会互相攻击
    解决该问题的基本思路是DFS+backtracking

    - 基本的回溯法，并在每次尝试时检查是否会有互相攻击的情况（$\color{red}{52ms}$）
    
    ```cpp
    bool check(vector<string> cur, int r, int c, int n)
    {
        bool ret = true;
        // 同一列
        for (int i = 0; ret && i < n; i++)
        {
            if (cur[i][c] == 'Q')
            {
                ret = false;
            }
        }
        // 主对角线
        int i = r - 1, j = c - 1;
        while (ret && i >= 0 && j >= 0)
        {
            if (cur[i][j] == 'Q')
            {
                ret = false;
            }
            i--, j--;
        }
        i = r + 1, j = c + 1;
        while (ret && i < n && j < n)
        {
            if (cur[i][j] == 'Q')
            {
                ret = false;
            }
            i++, j++;
        }
        // 副对角线
        i = r - 1, j = c + 1;
        while (ret && i >= 0 && j < n)
        {
            if (cur[i][j] == 'Q')
            {
                ret = false;
            }
            i--, j++;
        }
        i = r + 1, j = c - 1;
        while (ret && i < n && j >= 0)
        {
            if (cur[i][j] == 'Q')
            {
                ret = false;
            }
            i++, j--;
        }
        return ret;
    }
    void dfs_NQueens(vector<vector<string>> &ans, vector<string> &cur, int row, int n)
    {
        if (row < n)
        {
            // setting cur[i][j]='Q'
            for (int j = 0; j < n; j++)
            {
                if (check(cur, row, j, n))
                {
                    cur[row][j] = 'Q';
                    dfs_NQueens(ans, cur, row + 1, n);
                    cur[row][j] = '.'; // backtracking
                }
            }
        }
        else
        {
            ans.push_back(cur);
        }
    }
    vector<vector<string>> solveNQueens(int n)
    {
        vector<vector<string>> ans;
        string s;
        for (int i = 0; i < n; i++)
        {
            s.push_back('.');
        }
        vector<string> cur(n, s);
        dfs_NQueens(ans, cur, 0, n);
        return ans;
    }
    ```

    - 提前记录当前已经放在放置的皇后位置，bitmask方法，快速检查冲突，提高时间效率（$\color{red}{4ms}$），参见[BLOG](https://leetcode.com/problems/n-queens/discuss/19808/Accepted-4ms-c%2B%2B-solution-use-backtracking-and-bitmask-easy-understand.)

    ```cpp
    void dfs_NQueens(vector<vector<string>> &ans, vector<string> &cur, int row, int n, vector<int> &flag)
    {
        if (row < n)
        {
            // setting cur[i][j]='Q'
            for (int j = 0; j < n; j++)
            {
                if (flag[j] && flag[n + row + j] && flag[4 * n - row - 2 + j])
                {
                    flag[j] = flag[n + row + j] = flag[4 * n - row - 2 + j] = 0;
                    cur[row][j] = 'Q';
                    dfs_NQueens(ans, cur, row + 1, n, flag);
                    cur[row][j] = '.'; // backtracking
                    flag[j] = flag[n + row + j] = flag[4 * n - row - 2 + j] = 1;
                }
            }
        }
        else
        {
            ans.push_back(cur);
        }
    }
    vector<vector<string>> solveNQueens(int n)
    {
        vector<vector<string>> ans;
        vector<string> cur(n, string(n, '.'));
        vector<int> flag(5 * n - 2, 1);
        // [col(n), row+col(2n-1), n-row-1+col(2n-1)], 累计有[col,n+row+col,4n+col-row-1](5n-2)
        dfs_NQueens(ans, cur, 0, n, flag);
        return ans;
    }
    ```

- [52. N-Queens II](https://leetcode.com/problems/n-queens-ii/)

    与[51. N-Queens](https://leetcode.com/problems/n-queens/)相同的N皇后问题，本题要求输出符合要求的排列个数即可，无需输出全部的排列，在回溯法与DFS搜索的基础上，可以利用对称性减少一半的计算量

    ```cpp
    void dfs_NQueens(int &ans, vector<string> &cur, int row, int n, vector<int> &flag)
    {
        if (row < n)
        {
            // setting cur[i][j]='Q'
            for (int j = 0; j < n; j++)
            {
                if (flag[j] && flag[n + row + j] && flag[4 * n - row - 2 + j])
                {
                    flag[j] = flag[n + row + j] = flag[4 * n - row - 2 + j] = 0;
                    cur[row][j] = 'Q';
                    dfs_NQueens(ans, cur, row + 1, n, flag);
                    cur[row][j] = '.'; // backtracking
                    flag[j] = flag[n + row + j] = flag[4 * n - row - 2 + j] = 1;
                }
            }
        }
        else
        {
            ans++;
        }
    }
    int totalNQueens(int n)
    {
        int ans = 0;
        vector<string> cur(n, string(n, '.'));
        vector<int> flag(5 * n - 2, 1);
        // [col(n), row+col(2n-1), n-row-1+col(2n-1)], 累计有[col,n+row+col,4n+col-row-1](5n-2)
        dfs_NQueens(ans, cur, 0, n, flag);
        return ans;
    }
    ```

- [53](https://leetcode.com/problems/maximum-subarray/)

    一维dp(dynamic plan)

- [54](https://leetcode.com/problems/spiral-matrix/)

    spiral matrix，按照蛇形回环遍历一个二维数组，主要难点在数组的下标控制。类似的问题有[59](https://leetcode.com/problems/spiral-matrix-ii/)，将$1-n^2$这$n^2$个数按照蛇形规则填充到一个$n*n$的二维数组中；还有[885](https://leetcode.com/problems/spiral-matrix-iii/)，将指定的数字按照蛇序列从指定位置$(i,j)$开始填充到一个二维数组中。

    ```cpp
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
		vector<int> ans;
		if(!(matrix.size()==0 || matrix[0].size()==0)){
			int up=0,down=matrix.size()-1,left=0,right=matrix[0].size()-1;
			while(true){
				for(int i = left; i <= right; i++)
				{
					// from left to right
					ans.push_back(matrix[up][i]);
				}
				if(++up>down){
					break;
				}
				for(int i = up; i <= down; i++)
				{
					// from up to down
					ans.push_back(matrix[i][right]);
				}
				if(--right<left){
					break;
				}
				for(int i = right; i >= left; i--)
				{
					// for right to left
					ans.push_back(matrix[down][i]);
				}
				if(--down<up){
					break;
				}
				for(int i = down; i >= up; i--)
				{
					// from down to up
					ans.push_back(matrix[i][left]);
				}
                if(++left>right){
					break;
				}
			}
		}
		return ans;
    }
    ```

    另外一种思路，将每一次外层的环抽象成四段相等的线段，只要确定了起始点和方向、长度，就可以确定这条线上的每个坐标，然后输出该坐标下的值，时间复杂度$O(n)$，其中$n$为矩阵中数字的数量
    
    ```cpp
    vector<int> spiralOrder(vector<vector<int>> &matrix)
    {
        vector<int> ans;
        if (matrix.size() > 0 && matrix[0].size() > 0)
        {
            int rows = matrix.size(), cols = matrix[0].size();
            int total_number = rows * cols, cur_number = 0;
            vector<int> directionX{0, 1, 0, -1};
            vector<int> directionY{1, 0, -1, 0};
            int direction = 0, step_length = rows;
            int x = 0, y = -1;
            while (cur_number < total_number)
            {
                if (direction == 0 || direction == 2)
                {
                    step_length = cols;
                    rows--;
                }
                else
                {
                    step_length = rows;
                    cols--;
                }
                for (int i = 0; i < step_length; i++)
                {
                    x += directionX[direction];
                    y += directionY[direction];
                    ans.push_back(matrix[x][y]);
                    cur_number++;
                }
                direction = (++direction) % 4;
            }
        }
        return ans;
    }
    ```

- [55](https://leetcode.com/problems/jump-game/)

    给定一个int形数组nums，初始位置在0，$nums[i]$表示从位置i出发向右可以跳跃的最大步数，即从i出发可以到达的步数范围为$[i,i+nums_i]$，判断对于给定的数组能否到达最后一个位置$nums_{nums.size()-1}$

    - 最朴素的想法是用一个bool数组来标记从0出发是否可以到达当前位置i，首先将位置0设置为可达，然后从左到右遍历每一个位置position，如过位置position是可达的，则将从position出发可达的位置区间$[i,i+nums_i]$全部标记为可达，然后检查最后一个位置是否被标记为可达。

    ```cpp
    bool canJump(vector<int> &nums)
    {
        const int count = nums.size();
        if (count > 0)
        {
            vector<bool> reachable(count, false);
            reachable[0] = true;
            for (int i = 0; i < count; i++)
            {
                if (reachable[i] && nums[i] > 0)
                {
                    for (int j = 1; j <= nums[i] && i + j < count; j++)
                    {
                        reachable[i + j] = true;
                    }
                }
                if (reachable.back())
                {
                    return true;
                }
            }
            return false;
        }
        else
        {
            return true;
        }
    }
    ```

    $\color{red}{时间复杂度O(n^2),Time\  Limit\  Exceeded}$
    - Dynamic Programming Top-down，自顶向下的动态规划，递归思想，辅以memorization，即用一个memo数组来标记从位置$nums_i$出发是否可达终点，首先将所有位置均标注为unknown，然后从0开始询问每个位置是否可达终点，如果可达则返回true，如果不可达则返回false，如果unknow，则从当前位置开始递归地依次询问从当前位置可达的所有右侧位置是否可达，如果是，则将改点标记为可达，并返回true，如以上询问失败，则将该点标记为不可达并返回false。如此最后可递归询问到目标结果。

    ```cpp
    bool canJumpFromPosition(int position,vector<int>& nums,vector<int>& memo){
        if(memo[position]!=0){
            return (memo[position]==1)?true:false;
        }else{
            int furthest=min((int)(nums.size()-1),position+nums[position]); // 从当前位置出发可以抵达的最远位置
            for (int nextPosition = position+1; nextPosition <= furthest; nextPosition++)
            {
                if(canJumpFromPosition(nextPosition,nums,memo)){
                    memo[position]=1;
                    return true;
                }
            }
            memo[position]=2;
            return false;
        }
    }
    bool canJump(vector<int> &nums)
    {
        vector<int> memo(nums.size(),0); // 0-unknown, 1-reachable, 2-unreachable
        memo[nums.size()-1]=1; // final position is reachable from itself
        return canJumpFromPosition(0,nums,memo);
    }
    ```

    $\color{red}{时间复杂度O(n^2),Time\  Limit\  Exceeded}$
    - Dynamic Programming Bottom-up，自底向上的动态规划，从top-down到bottom-up的转换通常用来消除递归过程

    ```cpp
    bool canJump(vector<int> &nums)
    {
        int const count=nums.size();
        vector<int> memo(count,0); // 0-unknown, 1-reachable, 2-unreachable
        memo[count-1]=1; // final position is reachable from itself
        for (int i = count-2; i >= 0; i--)
        {
            int furthest=min(count-1,i+nums[i]);
            for (int j = i+1; j <= furthest; j++)
            {
                if(memo[j]==1){
                    memo[i]=1;
                    break;
                }
            }
        }
        return memo[0]==1; // 从位置0是否可达右端点
    }
    ```

    $\color{green}{时间复杂度O(n^2),Accepted,faster\  than\  13.04\%}$
    - greedy algorithm，贪心算法，通过观察自底向上的动态规划方法可以看出，我们只需要当前节点可以到达现在已经确定的可以到达的最左端，就可以确保从当前点出发可以到达最右端点，即初始化$curPosition=nums_{length}-1$，即当前可以到达的最左点，然后从右到左遍历所有点i时$nums_i+i>=curPosition$即可确定位置i可达，可以更新curPosition为i，最后判断curPosition是否为出发的左端点0即可。

    ```cpp
    bool canJump(vector<int> &nums)
    {
        int const count=nums.size();
        int curPosition=count-1;
        for (int i = count-2; i >= 0; i--)
        {
            if(i+nums[i]>=curPosition){
                curPosition=i;
            }
        }
        return curPosition==0;
    }
    ```

    $\color{green}{时间复杂度O(n),Accepted,faster\  than\  99.96\%}$

- [57. Insert Interval](https://leetcode.com/problems/insert-interval/)

    将一个给定的区间[a,b]插入一个给定的区间序列，并合并重复区间

    ```cpp
    vector<vector<int>> insert(vector<vector<int>> &intervals, vector<int> &newInterval)
    {
        vector<vector<int>> ans;
        int i = 0, count = intervals.size();
        while (i < count && intervals[i][1] < newInterval[0])
        {
            ans.push_back(intervals[i++]);
        }
        while (i < count && intervals[i][0] <= newInterval[1])
        {
            newInterval[0] = min(newInterval[0], intervals[i][0]);
            newInterval[1] = max(newInterval[1], intervals[i][1]);
            i++; // merge two intervals with overlay
        }
        ans.push_back(newInterval);
        while (i < count)
        {
            ans.push_back(intervals[i++]);
        }
        return ans;
    }
    ```

- [59. 螺旋矩阵 II](https://leetcode-cn.com/problems/spiral-matrix-ii/)

    与[54](https://leetcode.com/problems/spiral-matrix/)相似，控制数组下标反向回填即可，时间复杂度$O(n^2)$

    ```cpp
	vector<vector<int>> generateMatrix(int n)
	{
		vector<vector<int>> matrix(n, vector<int>(n));
		int left = 0, up = 0, right = n - 1, bottom = n - 1, k = 1;
		while (true)
		{
			for (int i = left; i <= right; i++)
			{
				matrix[up][i] = k++;
			}
			if (++up > bottom)
			{
				break;
			}
			for (int i = up; i <= bottom; i++)
			{
				matrix[i][right] = k++;
			}
			if (--right < left)
			{
				break;
			}
			for (int i = right; i >= left; i--)
			{
				matrix[bottom][i] = k++;
			}
			if (--bottom < up)
			{
				break;
			}
			for (int i = bottom; i >= up; i--)
			{
				matrix[i][left] = k++;
			}
			if (++left > right)
			{
				break;
			}
		}
		return matrix;
	}
    ```

- [60](https://leetcode.com/problems/permutation-sequence/)

    求$"1234...n"$形成的第$k$个全排列，数学上可以计算第$k$个全排列的第$i$个字符。

    ```cpp
    string getPermutation(int n, int k) {
		string s,ans;
		int factory=1;
		for (int i = 1; i <= n; i++)
		{
			s+=i+'0';
			factory*=i;
		}
		if(n<1 || n>9){
			return ans;
		}else{
			k--;
			for (int i = 0; i < n; i++)
			{
				factory/=n-i;
				ans+=s[i+k/factory];
				s.erase(s.begin()+i+k/factory);
				k%=factory;
			}
			return ans;
		}
    }    
    ```

- [62](https://leetcode.com/problems/unique-paths/)

    一维dp(dynamic plan)
    $$dp[m,n]=dp[m-1,n]+dp[m,n-1],\left\{\begin{matrix} dp[0,0]=0\\  dp[0,1]=1\\ dp[1,0]=1 \end{matrix}\right.$$

- [63](https://leetcode.com/problems/unique-paths-ii/)

    在[62-unique-path](https://leetcode.com/problems/unique-paths/)的基础上增加了障碍点，因此需要考虑初始化条件，即第一行、第一列有障碍的问题，同时咱有障碍的点路径数为0。

    另外需要注意由于障碍点的0路径导致最终结果在int表示范围内，但是计算过程中可能会出现超出int表示范围的数字，需要用long long来表示并取模(mod INT_MAX)。

- [65. Valid Number](https://leetcode.com/problems/valid-number/)

    验证给定的字符串是否是一个符合科学计数法的十进制浮点数

    ```cpp
    bool isNumber(string s)
    {
        bool ret = false;
        if (s.length() > 0)
        {
            int index_of_e = -1, index_of_dot = -1;
            int start = 0;
            // 首先去除首位的空字符
            while (start < s.length() && s[start] == ' ')
            {
                start++;
            }
            while (s.back() == ' ')
            {
                s.pop_back();
            }
            if (start < s.length() && (s[start] == '+' || s[start] == '-'))
            {
                start++; // 过滤可能的空字符
            }
            if (start >= s.length())
            {
                return false;
            }
            // 检查此后所有字符必须为 0-9|e|.
            // 其中e不出现或仅出现一次
            // 小数点不出现或仅出现一次
            // 如果小数点和e同时出现，小数点必须在e前面
            for (int j = start; j < s.length(); j++)
            {
                if (isdigit(s[j]))
                {
                    continue;
                }
                else if (s[j] == 'e')
                {
                    if (index_of_e == -1)
                    {
                        index_of_e = j;
                        if (j == start)
                        {
                            return false; // e前面必须有数
                        }
                        if (j + 1 < s.length() && (s[j + 1] == '+' || s[j + 1] == '-'))
                        {
                            j++; // e后面的指数部分必须是整数，但可以是负数或者0
                        }
                        if (j == s.length() - 1)
                        {
                            return false; // 指数部分必须有数
                        }
                    }
                    else
                    {
                        return false;
                    }
                }
                else if (s[j] == '.')
                {
                    if (index_of_dot == -1)
                    {
                        index_of_dot = j;
                        if (!((j > start && isdigit(s[j - 1])) || (j + 1 < s.length() && isdigit(s[j + 1]))))
                        {
                            return false; // 小数点前面或者后面至少有一个数字，即小数点不能单独出现
                        }
                    }
                    else
                    {
                        return false;
                    }
                }
                else
                {
                    return false;
                }
            }
            if (index_of_dot != -1 && index_of_e != -1 && index_of_dot >= index_of_e)
            {
                return false;
            }
            ret = true;
        }
        return ret;
    }
    ```

- [68. Text Justification](https://leetcode.com/problems/text-justification/)

    将给定字符串根据固定行宽进行对齐操作

    ```cpp
    vector<string> fullJustify(vector<string> &words, int maxWidth)
    {
        vector<string> ans;
        const int count = words.size();
        int i = 0, j = 0, k = 0;
        while (i < count)
        {
            string line = words[i];
            int length = words[i].length();
            j = i + 1;
            while (j < count && length + words[j].length() + 1 <= maxWidth)
            {
                length += words[j++].length() + 1;
            }
            int number_spaces = j - i - 1, total_spaces = maxWidth - length + number_spaces;
            if (number_spaces > 0)
            {
                // two or more word in a line
                k = i + 1;
                if (j < count)
                {
                    //this is not the last line, full-justified
                    int single_spaces = total_spaces / number_spaces, remainder = total_spaces % number_spaces;
                    while (k < j)
                    {
                        for (int r = 0; r < single_spaces; r++)
                        {
                            line += ' ';
                        }
                        if (k - i - 1 < remainder)
                        {
                            line += ' ';
                        }
                        line += words[k++];
                    }
                }
                else
                {
                    // the last line, left-justified
                    while (k < j)
                    {
                        line += ' ' + words[k++];
                    }
                    while (line.length() < maxWidth)
                    {
                        line += ' ';
                    }
                }
            }
            else
            {
                // only one word in a line, left-justified
                while (line.length() < maxWidth)
                {
                    line += ' ';
                }
            }
            ans.push_back(line);
            i = j;
        }
        return ans;
    }
    ```

- [69](https://leetcode.com/problems/sqrtx/)

    牛顿迭代法

- [71](https://leetcode.com/problems/simplify-path/)

    将一个给定的unix形式的文件路径改为标准的canonical path形式，将有效地目录结构自顶向下存储为string数组，然后使用/字符将string数组链接起来即可，注意首尾边界形式的处理和函数结构的设计

    ```cpp
    void helper(vector<string>& paths,string s){
        if(!s.empty()){
            if(s.compare("..")){
                if(!paths.empty()){
                    paths.pop_back();
                }
            }else if(s.compare(".")!=0){
                paths.push_back(s);
            }
        }
    }
    string simplifyPath(string path)
    {
        vector<string> paths;
        string s;
        for (auto &&ch : path)
        {
            if(ch=='/'){
                helper(paths, s);
                s.clear();
            }else{
                s.push_back(ch);
            }
        }
        helper(paths, s); // for last segmentation of the path
        string ans;
        for (auto &&simple_path : paths)
        {
            ans.push_back('/');
            ans += simple_path;
        }
        if (ans.empty())
        {
            // for empth path
            ans.push_back('/');
        }
        return ans;
    }
    ```

- [72](https://leetcode.com/problems/edit-distance/)

    经典的编辑距离问题，即给定两个字符串a和b，求从a转化到b的最少单字符操作次数，单字符操作指一个字符的增加、删除、改变三种操作中的一种，这是一个经典的动态规划问题，和[1035](https://leetcode.com/problems/uncrossed-lines/)的解法很像，定义状态$dp_{i,j}$表示字符串a的前$i$个字符到字符串b的前$j$个字符之间的编辑距离，则其状态转移方程如下：
    $$dp_{i,j}=\left\{\begin{matrix}
        min(i,j) = 0 : max(i,j)\\
        min(i,j) \neq 0 : \left\{\begin{matrix}
        a_i = b_j : dp_{i-1,j-1}\\
        a_i \neq b_j : min \left\{\begin{matrix}
        dp_{i-1,j}+1 \\
        dp_{i-1,j-1}+1 \\
        dp_{i,j-1}+1 \\
        \end{matrix}\right.
        \end{matrix}\right.
        \end{matrix}\right.$$

    **注意数学公式中0-index和代码中1-index的转换**

    ```cpp
    int minDistance(string word1, string word2)
    {
        int ans = 0;
        int length1 = word1.length(), length2 = word2.length();
        if (min(length1, length2) == 0)
        {
            ans = max(length1, length2);
        }
        else
        {
            vector<vector<int>> dp(length1 + 1, vector<int>(length2 + 1, 0));
            for (int i = 1; i <= length1; i++)
            {
                dp[i][0] = i;
            }
            for (int i = 1; i <= length2; i++)
            {
                dp[0][i] = i;
            }
            for (int i = 1; i <= length1; i++)
            {
                for (int j = 1; j <= length2; j++)
                {
                    if (word1[i-1] == word2[j-1])
                    {
                        dp[i][j] = dp[i - 1][j - 1];
                    }
                    else
                    {
                        dp[i][j] = min(min(dp[i - 1][j], dp[i - 1][j - 1]), dp[i][j - 1]) + 1;
                    }
                }
            }
            ans = dp.back().back();
        }
        return ans;
    }
    ```

    为了节省内存空间，可以从二维数组实现的DP压缩到一维数组实现

    ```cpp
    int minDistance(string word1, string word2)
    {
        int ans = 0;
        int length1 = word1.length(), length2 = word2.length();
        if (min(length1, length2) == 0)
        {
            ans = max(length1, length2);
        }
        else
        {
            vector<int> dp(length2 + 1, 0);
            for (int i = 0; i <= length2; i++)
            {
                dp[i] = i;
            }
            for (int i = 1; i <= length1; i++)
            {
                int cur = i, temp = 0;
                for (int j = 1; j<=length2; j++)
                {
                    if(word1[i-1]==word2[j-1]){
                        temp = dp[j-1];
                    }else{
                        temp = min(cur, min(dp[j], dp[j - 1])) + 1;
                    }
                    dp[j - 1] = cur;
                    cur = temp;
                }
                dp.back() = cur;
            }
            ans = dp.back();
        }
        return ans;
    }
    ```

- [73](https://leetcode.com/problems/set-matrix-zeroes/)

    将一个矩阵中有0的行和列全部set为0

    - 方法1：pass one记录所有的值为0的行和列坐标，pass two按记录的坐标将这些行和列全部set为0，时间复杂度$O(m*n)$，空间复杂度$O(m*n)$，缺点在于空间复杂度高

    ```cpp
    void setZeroes(vector<vector<int>> &matrix)
    {
        unordered_set<int> indexs;
        int m = matrix.size(), n = matrix[0].size();
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (matrix[i][j] == 0)
                {
                    indexs.insert((i + 1));
                    indexs.insert(-(j + 1));
                }
            }
        }
        for (auto &&index : indexs)
        {
            if (index > 0)
            {
                for (int i = 0; i < n; i++)
                {
                    matrix[index - 1][i] = 0;
                }
            }
            else
            {
                for (int i = 0; i < m; i++)
                {
                    matrix[i][-index - 1] = 0;
                }
            }
        }
    }
    ```

    - 方法2：pass one将有0所在的行首和列首set为0，pass two将首值为0的行和列全部set为0，时间复杂度$O(m*n)$，空间复杂度$O(1)$

    ```cpp
    void setZeroes(vector<vector<int>> &matrix)
    {
        int m = matrix.size(), n = matrix[0].size();
        bool firstCol = false, firstRow = false;
        if(matrix[0][0]==0){
            firstCol = true;
            firstRow = true;
        }else{
            for (int i = 0; i < m; i++)
            {
                if (matrix[i][0] == 0)
                {
                    firstCol = true;
                }
            }
            for (int j = 0; j < n; j++)
            {
                if (matrix[0][j] == 0)
                {
                    firstRow = true;
                }
            }
        }
        // one pass
        for (int i = 1; i < m; i++)
        {
            for (int j = 1; j < n; j++)
            {
                if (matrix[i][j] == 0)
                {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        // two pass
        for (int i = 1; i < m; i++)
        {
            for (int j = 1; j < n; j++)
            {
                if (matrix[i][0]==0 ||matrix[0][j]==0){
                    matrix[i][j] = 0;
                }
            }
        }
        // specifial for first column and first row
        if(firstCol){
            for (int i = 0; i < m; i++)
            {
                matrix[i][0] = 0;
            }
        }
        if(firstRow){
            for (int j = 0; j < n; j++)
            {
                matrix[0][j] = 0;
            }
        }
    }
    ```

- [74](https://leetcode.com/problems/search-a-2d-matrix/)

    在一个排序矩阵中搜索某个数存在与否，$m*n$矩阵每一行都是升序的且每一行第一个元素都比前一行最后一个元素大，则reshape成$(m*n,1)$的形状后为升序的，则可binary search，时间复杂度$O(m*n)$

    ```cpp
    bool searchMatrix(vector<vector<int>> &matrix, int target)
    {
        bool ret = false;
        if (matrix.size() > 0 && matrix[0].size() > 0)
        {
            int m = matrix.size(), n = matrix[0].size();
            int left = 0, right = m * n - 1;
            while (left <= right)
            {
                int mid = left + ((right - left) >> 1);
                if (matrix[mid / n][mid % n] == target)
                {
                    ret = true;
                    break;
                }
                else if (matrix[mid / n][mid % n] < target)
                {
                    left = mid + 1;
                }
                else
                {
                    right = mid - 1;
                }
            }
        }
        return ret;
    }
    ```

- [75](https://leetcode.com/problems/sort-colors/)

    即荷兰国旗问题，三色排序，快速排序的基本思想练习

    - two pass，count and set

    ```cpp
    void sortColors(vector<int> &nums)
    {
        vector<int> count(3, 0);
        for (auto &&v : nums)
        {
            count[v]++;
        }
        int start = 0, end = 0;
        for (int v = 0; v < 3; v++)
        {
            start = end, end += count[v];
            for (int i = start; i < end; i++)
            {
                nums[i] = v;
            }
        }
    }
    ```

    - one pass，method like quick sort

    ```cpp
    void sortColors(vector<int> &nums)
    {
        int left = -1, right = nums.size(), cur = 0;
        while (cur < right)
        {
            if (nums[cur] == 0)
            {
                swap(nums[++left], nums[cur++]);
            }
            else if (nums[cur] == 2)
            {
                swap(nums[--right], nums[cur]);
            }
            else
            {
                cur++;
            }
        }
    }
    ```

    - one pass , [0,i) - 0   [i,j) - 1    [j,k) - 2

    ```cpp
    void sortColors(vector<int> &nums)
    {
        int i = 0, j = 0, tmp = 0;
        for (int k = 0; k < nums.size(); k++)
        {
            tmp = nums[k];
            nums[k] = 2;
            if (tmp < 2)
            {
                nums[j++] = 1;
            }
            if (tmp == 0)
            {
                nums[i++] = 0;
            }
        }
    }
    ```

- [77](https://leetcode.com/problems/combinations/)

    求组合数C(n,k)所有可能形式，一般为DFS的递归写法
    
    ```cpp
    vector<vector<int>> ans;
    vector<int> combination;
    int target;
    void dfs_helper(int n, int k)
    {
        for (int i = n; i >= k; i--)
        {
            combination[k - 1] = i;
            if(k>1){
                dfs_helper(i - 1, k - 1);
            }else{
                ans.push_back(combination);
            }
        }
    }
    vector<vector<int>> combine(int n, int k)
    {
        target = k;
        combination = vector<int>(k, 0);
        dfs_helper(n, k);
        return ans;
    }
    ```

- [78](https://leetcode.com/problems/subsets/)

    给定一个没有重复数字的数组，求该数组所有可能的子集，即该集合的超集，首先求出n-1个元素的超集，然后将该超集中的每个集合添加第n个元素后添加入超集即可

    ```cpp
    vector<vector<int>> subsets(vector<int> &nums)
    {
        vector<vector<int>> ans;
        ans.push_back({});
        for (auto &&v : nums)
        {
            const int count = ans.size();
            for (int i = 0; i < count; i++)
            {
                vector<int> temp = ans[i];
                temp.push_back(v);
                ans.push_back(temp);
            }
        }
        return ans;
    }
    ```

- [79. Word Search](https://leetcode.com/problems/word-search/)

    在一个给定的字符矩阵grid中搜索是否有连续的字符串组成单词word，典型的DFS深度优先搜索应用

    ```cpp
    bool dfs_helper(vector<vector<char>> &board, string word, int i, int j, int index)
    {
        bool ret = false;
        if (index == word.length())
        {
            ret = true;
        }
        else if (!(i < 0 || j < 0 || i >= board.size() || j >= board[0].size() || board[i][j] == '#' || board[i][j] != word[index]))
        {
            // 数组越界、当前字符已被占用、当前字符不符合word当前index位置的值，直接返回false
            board[i][j] = '#'; // flag for visited node
            vector<int> direction{1, 0, -1, 0, 1};
            for (int k = 0; !ret && k < 4; k++)
            {
                ret = dfs_helper(board, word, i + direction[k], j + direction[k + 1], index + 1);
            }
            board[i][j] = word[index]; // backtracking for visited node, mark it as unvisited
        }
        return ret;
    }
    bool exist(vector<vector<char>> &board, string word)
    {
        bool ret = false;
        if (board.size() > 0 && board[0].size() > 0 && word.length() > 0)
        {
            const int m = board.size(), n = board[0].size();
            for (int i = 0; !ret && i < m; i++)
            {
                for (int j = 0; !ret && j < n; j++)
                {
                    ret = dfs_helper(board, word, i, j, 0);
                }
            }
        }
        return ret;
    }
    ```

- [80. 删除排序数组中的重复项 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii/)

    使用双指针idx和i，其中idx存储重复次数两次以内的数字，i从左向右遍历给定数组nums，如果当前$nums[idx]$之前已经有两个和当前数字$nums[i]$相同的数字，则忽略/删除当前数字，时间复杂度$O(n)$

    ```cpp
	int removeDuplicates(vector<int> &nums)
	{
		int idx = 0, i = 0, n = nums.size();
		while (i < n)
		{
			if (!(idx >= 2 && nums[idx - 2] == nums[i] && nums[idx - 1] == nums[i]))
			{
				nums[idx++] = nums[i];
			}
			i++;
		}
		return idx;
	}
    ```

- [81](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/)

    在一个升序旋转后的数组中寻找是否存在一个值，与题目[33](https://leetcode.com/problems/search-in-rotated-sorted-array/)不同的是本题中增加条件，数组中可能有duplicated的值，这时仍然用二分搜索，平均时间复杂度$O(log(n))$，但是由于需要在二分查找是判断mid和left、right的值都相同的情况，所以最坏情况下会退化到$O(n)$

    - [33]没有duplicated的情况

    ```cpp
    int search(vector<int>& nums, int target) {
        int const n=nums.size();
        int left=0,right=n-1;
        while(left<right){
            int mid=(left+right)>>1;
            if(nums[mid]<nums[right]){
                right=mid;
            }else{
                left=mid+1;
            }
        }
        int pivot=left;
        left=0,right=n-1;
        while(left<=right){
            int mid=(left+right)>>1;
            int index=(mid+pivot)%n;
            if(target<nums[index]){
                right=mid-1;
            }else if(target>nums[index]){
                left=mid+1;
            }else{
                return index;
            }
        }
        return -1;
    }
    ```

    - 本题有duplicated的情况
    可以参考以下两篇artical，[ref_1](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/discuss/28218/My-8ms-C%2B%2B-solution-(o(logn)-on-average-o(n)-worst-case))，[ref_2](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/discuss/28194/C%2B%2B-concise-log(n)-solution)

    ```cpp
    bool search(vector<int> &nums, int target)
    {
        int left = 0, right = nums.size() - 1;
        int ans = false;
        while (left <= right)
        {
            int mid = left + ((right - left) / 2);
            if (nums[mid] == target)
            {
                ans = true;
                break;
            }
            else if (nums[mid] == nums[left] && nums[mid] == nums[right])
            {
                left++, right--;
            }
            else if (nums[left] <= nums[mid])
            {
                if (nums[left] <= target && nums[mid] > target)
                {
                    right = mid - 1;
                }
                else
                {
                    left = mid + 1;
                }
            }
            else
            {
                // nums[left]>nums[mid]
                if (nums[mid] < target && target <= nums[right])
                {
                    left = mid + 1;
                }
                else
                {
                    right = mid - 1;
                }
            }
        }
        return ans;
    }
    ```

- [82. 删除排序链表中的重复元素 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)

    使用三个指针同时向后扫描，遇到相同元素全部删除，时间复杂度$O(n)$

    ```cpp
	ListNode *deleteDuplicates(ListNode *head)
	{
		if (head)
		{
			ListNode *auxiliary = new ListNode(0);
			auxiliary->next = head;
			ListNode *pre = auxiliary, *cur = head, *next = head->next;
			while (next)
			{
				if (cur->val == next->val)
				{
					// 发现重复值，删除该值的全部节点
					int value = cur->val;
					while (cur && cur->val == value)
					{
						cur = cur->next;
					}
					pre->next = cur;
					next = (cur ? cur->next : nullptr);
				}
				else
				{
					pre = cur;
					cur = next;
					next = cur->next;
				}
			}
            head = auxiliary->next;
		}
		return head;
	}
    ```

- [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
    
    - 通过双指针i和j枚举所有可能矩形宽度$j-i+1$，在此宽度下计算可能的矩形面积，时间复杂度$O(n^2)$，leetcode评测机$\color{red}{TLE}$

	```cpp
	int largestRectangleArea(vector<int> &heights)
	{
		int ret = 0, n = heights.size();
		for (auto i = 0; i < n; i++)
		{
			int min_height = heights[i];
			for (auto j = i; j < n; j++)
			{
				min_height = min(min_height, heights[j]);
				ret = max(ret, min_height * (j - i + 1));
			}
		}
		return ret;
	}
	```
    
    - 枚举所有可能的矩形高度$heights[i]$，然后找到i左侧第一个小于heights[i]的柱子left和右侧第一个小于heights[i]的柱子right，则该矩形面积为$heights[i]*(right-left-1)$，时间复杂度$O(n^2)$，leetcode评测机$\color{red}{TLE}$

	```cpp
	int largestRectangleArea(vector<int> &heights)
	{
		int ret = 0, n = heights.size();
		for (auto i = 0; i < n; i++)
		{
			int min_height = heights[i];
			for (auto j = i; j < n; j++)
			{
				int left = i, right = i;
				while (left >= 0 && heights[left] >= heights[i])
				{
					left--;
				}
				while (right < n && heights[right] >= heights[i])
				{
					right++;
				}
				ret = max(ret, heights[i] * (right - left - 1));
			}
		}
		return ret;
	}
	```

    - 单调栈，时间复杂度$O(n)$

    ```cpp
	int largestRectangleArea(vector<int> &heights)
	{
		int ret = 0, n = heights.size();
		stack<int> st;
		vector<int> left(n, 0), right(n, n);
		for (auto i = 0; i < n; i++)
		{
			while (!st.empty() && heights[st.top()] >= heights[i])
			{
				st.pop();
			}
			left[i] = st.empty() ? -1 : st.top();
			st.push(i);
		}
		st = stack<int>();
		for (auto i = n - 1; i >= 0; i--)
		{
			while (!st.empty() && heights[st.top()] >= heights[i])
			{
				st.pop();
			}
			right[i] = st.empty() ? n : st.top();
			st.push(i);
		}
		for (int i = 0; i < n; i++)
		{
			ret = max(ret, heights[i] * (right[i] - left[i] - 1));
		}
		return ret;
	}
    ```
    
    - 单调栈 + 常数优化

	```cpp
	int largestRectangleArea(vector<int> &heights)
	{
		int ret = 0, n = heights.size();
		stack<int> st;
		vector<int> left(n, 0), right(n, n);
		for (auto i = 0; i < n; i++)
		{
			while (!st.empty() && heights[st.top()] >= heights[i])
			{
				right[st.top()]=i;
				st.pop();
			}
			left[i] = st.empty() ? -1 : st.top();
			st.push(i);
		}
		for (int i = 0; i < n; i++)
		{
			ret = max(ret, heights[i] * (right[i] - left[i] - 1));
		}
		return ret;
	}
	```
    
    - 单调栈的一次遍历写法

        **单调栈的本质是寻找当前heights[i]的左侧第一个比其低的柱子**

	```cpp
	int largestRectangleArea(vector<int> &heights)
	{
		int ret = 0, n = heights.size();
		heights.push_back(-1); // 哨兵点位
		stack<int> st;
		for (int i = 0; i <= n; i++)
		{
			while (!st.empty() && heights[st.top()] >= heights[i])
			{
				int h = heights[st.top()];
				st.pop();
				ret = max(ret, h * (st.empty() ? i : (i - st.top() - 1)));
			}
			st.push(i);
		}
		return ret;
	}
	```

- [85. 最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/)

    在矩阵中的每一个位置matrix[i][j]计算其上侧连续1的数量，即可形成一个柱状图，转化为[84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)问题解决，时间复杂度$O(rows*cols)$$

    ```cpp
	int maximalRectangle(vector<vector<char>> &matrix)
	{
		int ret = 0;
		if (matrix.size() > 0 && matrix[0].size() > 0)
		{
			const int rows = matrix.size(), cols = matrix[0].size();
			vector<vector<int>> heights(rows, vector<int>(cols + 1)); // 每一行最后一列加入一个哨兵点位
			for (int j = 0; j < cols; j++)
			{
				heights[0][j] = matrix[0][j] == '1' ? 1 : 0;
			}
			heights[0][cols] = -1;
			for (int i = 1; i < rows; i++)
			{
				for (int j = 0; j < cols; j++)
				{
					heights[i][j] = matrix[i][j] == '1' ? 1 + heights[i - 1][j] : 0;
				}
				heights[i][cols] = -1;
			}
			// 计算每一行的柱状图形成的最大矩形
			for (int i = 0; i < rows; i++)
			{
				stack<int> st;
				for (int j = 0; j <= cols; j++)
				{
					while (!st.empty() && heights[i][st.top()] >= heights[i][j])
					{
						int h = heights[i][st.top()];
						st.pop();
						ret = max(ret, h * (st.empty() ? j : j - st.top() - 1));
					}
					st.push(j);
				}
			}
		}
		return ret;
	}
    ```

- [88. Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)

    - 借助额外的空间合并排序数组，归并排序的思想，时间复杂度$O(n)$

    ```cpp
    void merge(vector<int> &nums1, int m, vector<int> &nums2, int n)
    {
        int count = m + n;
        vector<int> ret(count, 0);
        int i = 0, j = 0, k = 0;
        while (i < m && j < n)
        {
            if (nums1[i] < nums2[j])
            {
                ret[k] = nums1[i++];
            }
            else
            {
                ret[k] = nums2[j++];
            }
            k++;
        }
        while (i < m)
        {
            ret[k++] = nums1[i++];
        }
        while (j < n)
        {
            ret[k++] = nums2[j++];
        }
        nums1 = ret;
    }
    ```

    - 在nums1的原地合并数组，时间复杂度$O(n)$，和空间复杂度为$O(1)$

    ```cpp
    void merge(vector<int> &nums1, int m, vector<int> &nums2, int n)
    {
        int count = m + n;
        for (auto i = count - 1; i >= n; i--)
        {
            nums1[i] = nums1[i - n]; // 腾出nums1的前半部分空间
        }
        int i = n, j = 0, k = 0;
        while (i < count && j < n)
        {
            if (nums1[i] < nums2[j])
            {
                nums1[k++] = nums1[i++];
            }
            else
            {
                nums1[k++] = nums2[j++];
            }
        }
        while (j < n)
        {
            nums1[k++] = nums2[j++];
        }
    }
    ```

- [89](https://leetcode.com/problems/gray-code/)

    给定一个位宽值n，求$range(0,2^{n}-1)$的一个全排列，要求排列中每两个相邻的数字的二进制表示仅仅有一个bit位不同，类似的还有[1238](https://leetcode.com/problems/circular-permutation-in-binary-representation/)，比本题多了一个要求是该排列要从指定数字k开始。

    本题的规律在于，从位宽为n-1即$range(0,2^{n-1}-1)$的全排列，逆转之后附加上最高位bit(前一半为0，后一半为0)，即可实现位宽为n即$range(0,2^{n}-1)$之间的全排列。

    ```cpp
    vector<int> grayCode(int n)
    {
        if(n==0){
            return {0};
        }
        vector<int> ans{0, 1};
        for (int i = 1; i < n; i++)
        {
            int added = (1 << i);
            int count = ans.size() - 1;
            for (int j = count; j >= 0; j--)
            {
                ans.push_back(ans[j] + added);
            }
        }
        return ans;
    }
    ```

- [90](https://leetcode.com/problems/subsets-ii/)

    在[78](https://leetcode.com/problems/subsets/)的基础上，本题加入了给定数组中可能存在重复数字的条件

    ```cpp
    vector<vector<int>> subsetsWithDup(vector<int>& nums)
    {
        vector<vector<int>> ans;
        vector<int> indexs;
        ans.push_back({});
        sort(nums.begin(), nums.end());
        int last_count = 0;
        for (int i = 0; i < nums.size(); i++)
        {
            const int count = ans.size();
            int j = 0;
            if (i > 0 && nums[i] == nums[i - 1])
            {
                j = last_count;
            }
            for (; j < count; j++)
            {
                vector<int> temp = ans[j];
                temp.push_back(nums[i]);
                ans.push_back(temp);
            }
            last_count = count;
        }
        return ans;
    }
    ```

- [91. Decode Ways](https://leetcode.com/problems/decode-ways/)

    对于给定字符串s用dp[i+1]表示s的的前缀s[0]...s[i]可以构成的decode ways数，则
    $$
    dp[i]+=\left\{\begin{matrix}
    dp[i-1], 1 \le s[i] \le 9 \quad and \\
    dp[i-2], s[i-2]=1, 0 \le s[i] \le 9 \quad or \quad s[i-2]=2, 0 \le s[i] \le 6
    \end{matrix}\right.
    $$

    ```cpp
    int numDecodings(string s)
    {
        int ans = 0;
        int const count = s.length();
        if (count > 0)
        {
            vector<int> dp(count + 1, 0);
            dp[0] = 1;
            if (s[0] != '0')
            {
                dp[1] = 1;
            }
            for (int i = 1; i < count; i++)
            {
                if (s[i] != '0')
                {
                    dp[i + 1] += dp[i];
                }
                if (s[i - 1] == '1' || (s[i - 1] == '2' && (s[i] >= '0' && s[i] <= '6')))
                {
                    dp[i + 1] += dp[i - 1];
                }
            }
            ans = dp.back();
        }
        return ans;
    }
    ```

- [92. 反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/)

    链表常规操作，时间复杂度$O(n)$

    ```cpp
	ListNode *reverseBetween(ListNode *head, int left, int right)
	{
		ListNode *auxiliary = new ListNode(0);
		auxiliary->next = head;
		if (right - left)
		{
			ListNode *pre = auxiliary;
			for (int i = 1; i < left; i++)
			{
				pre = pre->next;
			}
			ListNode *pre_cur = pre->next, *cur = pre->next->next;
			for (int i = left + 1; i <= right; i++)
			{
				pre_cur->next = cur->next;
				cur->next = pre->next;
				pre->next = cur;
				cur = pre_cur->next;
			}
		}
		return auxiliary->next;
	}
    ```

- [93. Restore IP Addresses](https://leetcode.com/problems/restore-ip-addresses/)

    每次从剩余字符串的开头取出1到3位作为一个整型数，递归得处理剩余字符串，看是否构成四个0到255范围内的整型数时恰好用完字符串，其中注意0只能由字符串'0'构成，stoi('00')=0不符合要求

    ```cpp
    void helper(string &s, int i, int length, vector<int> &cur, vector<string> &ans)
    {
        if (i + length <= s.length())
        {
            int v = stoi(s.substr(i, length));
            if (to_string(v).length() != length)
            {
                return; // for avoiding stoi(00)=0
            }
            if (v < 256 && v >= 0)
            {
                cur.push_back(v);
                if (cur.size() < 4 && i + length < s.length())
                {
                    for (int k = 1; k <= 3; k++)
                    {
                        helper(s, i + length, k, cur, ans);
                    }
                }
                else if (cur.size() == 4 && i + length == s.length())
                {
                    string ip;
                    for (int i = 0; i < 4; i++)
                    {
                        ip += to_string(cur[i]);
                        ip.push_back('.');
                    }
                    ans.push_back(ip.substr(0, ip.length() - 1));
                }
                cur.pop_back();
            }
        }
    }
    vector<string> restoreIpAddresses(string s)
    {
        vector<string> ans;
        vector<int> cur;
        helper(s, 0, 1, cur, ans);
        helper(s, 0, 2, cur, ans);
        helper(s, 0, 3, cur, ans);
        return ans;
    }
    ```

- [94. Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)

    **二叉树的中序遍历**

    - 递归写法

    ```cpp
    class Solution
    {
    private:
        void dfs(TreeNode *root, vector<int> &ret)
        {
            if (root)
            {
                dfs(root->left, ret);
                ret.push_back(root->val);
                dfs(root->right, ret);
            }
        }

    public:
        vector<int> inorderTraversal(TreeNode *root)
        {
            vector<int> ret;
            dfs(root, ret);
            return ret;
        }
    };
    ```

    - 迭代写法

    ```cpp
	vector<int> inorderTraversal(TreeNode *root)
	{
		vector<int> ret;
		if (root)
		{
			stack<TreeNode *> st;
			TreeNode *cur = root;
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
    ```

- [95. Unique Binary Search Trees II](https://leetcode.com/problems/unique-binary-search-trees-ii/)

    - 递归生成从1到n的二叉搜索树

    ```cpp
    vector<TreeNode *> generate_recursive(int left, int right)
    {
        vector<TreeNode *> ret;
        if (left <= right)
        {
            for (int k = left; k <= right; k++)
            {
                vector<TreeNode *> left_children = generate_recursive(left, k - 1);
                vector<TreeNode *> right_children = generate_recursive(k + 1, right);
                for (auto &&left_node : left_children)
                {
                    for (auto &&right_node : right_children)
                    {
                        TreeNode *root = new TreeNode(k);
                        root->left = left_node;
                        root->right = right_node;
                        ret.push_back(root);
                    }
                }
            }
        }
        else
        {
            ret.push_back(nullptr);
        }
        return ret;
    }
    vector<TreeNode *> generateTrees(int n)
    {
        vector<TreeNode *> ret;
        if (n > 0)
        {
            ret = generate_recursive(1, n);
        }
        return ret;
    }
    ```

    - 减少递归次数，提高时间效率的实现

    ```cpp
    vector<TreeNode *> generate_recursive(int left, int right)
    {
        vector<TreeNode *> ret;
        if (left < right)
        {
            for (int k = left; k <= right; k++)
            {
                vector<TreeNode *> left_children = generate_recursive(left, k - 1);
                vector<TreeNode *> right_children = generate_recursive(k + 1, right);
                for (auto &&left_node : left_children)
                {
                    for (auto &&right_node : right_children)
                    {
                        TreeNode *root = new TreeNode(k);
                        root->left = left_node;
                        root->right = right_node;
                        ret.push_back(root);
                    }
                }
            }
        }
        else if (left == right)
        {
            TreeNode *node = new TreeNode(left);
            ret.push_back(node);
        }
        else
        {
            ret.push_back(nullptr);
        }
        return ret;
    }
    vector<TreeNode *> generateTrees(int n)
    {
        vector<TreeNode *> ret;
        if (n > 0)
        {
            ret = generate_recursive(1, n);
        }
        return ret;
    }
    ```

- [98](https://leetcode.com/problems/validate-binary-search-tree/)

    验证给定的二叉树是否为二叉搜索树(BST)

    - 思路一，通过迭代或递归的方式获取二叉树的中序遍历数组inorder，然后线性扫描验证inorder数组是否为升序，时间复杂度$O(n)$，其中n为二叉树节点数量
    
    ```cpp
    bool isValidBST(TreeNode* root) {
    	bool ret = true;
		if (root)
		{
			vector<int> inorder;
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
					inorder.push_back(cur->val);
					if (cur->right)
					{
						cur = cur->right;
					}
					else
					{
						cur = nullptr;
					}
				}
			}
			for (int i = 1; i < inorder.size(); i++)
			{
				if (inorder[i] <= inorder[i - 1])
				{
					ret = false;
					break;
				}
			}
		}
		return ret;
    }
    ```

    - 思路二，递归式验证非空二叉树的左子树为BST且左子树所有节点值小于当前节点，右子树为BST且右子树所有节点值大于当前节点，则为BST

    ```cpp
    bool helper(TreeNode *root, long long lower, long long upper)
    {
        bool ret = true;
        if (root)
        {
            int v = root->val;
            if ((v <= lower) || (v >= upper))
            {
                ret = false;
            }
            else
            {
                ret = helper(root->left, lower, v) && helper(root->right, v, upper);
            }
        }
        return ret;
    }
    bool isValidBST(TreeNode *root)
    {
        return helper(root, numeric_limits<long long>::min(), numeric_limits<long long>::max());
    }
    ```