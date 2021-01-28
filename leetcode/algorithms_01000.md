# 901-1000

- [901. Online Stock Span](https://leetcode.com/problems/online-stock-span/)

    寻找prices数组中从右往左第一个大于当前price的价格下标，最坏情况下时间复杂度$O(n^2)$

	```cpp
    class StockSpanner
    {
    private:
        // prices[i]是当前ith天的price，prev[i]是第一个大于prices[i]的price的下标
        vector<int> prices, prev;

    public:
        StockSpanner() {}

        int next(int price)
        {
            prices.push_back(price);
            int i = prices.size() - 2; // 从当前值得前一个开始search
            while (!prev.empty() && i >= 0 && prices[i] <= price)
            {
                i = prev[i];
            }
            prev.push_back(i);
            int span = prices.size() - 1 - i;
            return span;
        }
    };
	```

- [905](https://leetcode.com/problems/sort-array-by-parity/)
    cpp的两个标准库函数，用于将vector按照一定的条件划分，例如将一个int类型数组按照奇数偶数划分
    - [partition()](https://en.cppreference.com/w/cpp/algorithm/partition)
    - [stable_partition()](https://en.cppreference.com/w/cpp/algorithm/stable_partition)
    
    eg:

    ```cpp
    std::vector<int> v = {0,1,2,3,4,5,6,7,8,9};
    auto it = std::partition(v.begin(), v.end(), [](int i){return i % 2 == 0;});
    ```

- [906](https://leetcode.com/problems/super-palindromes/)

    使用数位回溯或者数位dp甚至是暴力模拟构造从$1-10^9$范围内所有可能的回文数，计算其平方并检查平方结果是否为回文数即可。

- [908](https://leetcode.com/problems/smallest-range-i/)

    给定数组A和阈值k，在$\pm k$范围内调节数组中的每一个数字，使得A中的最大值和最小值之间的差距尽可能的小。理论上尽量用$\pm k$范围内的数去抹平原数组中最大值和最小值之间的差距就可以了。

    ```cpp
    int smallestRangeI(vector<int> &A, int K)
    {
        const int count = A.size();
        int i = 0;
        int max_value = A[0], min_value = A[0];
        for (i = 1; i < count; i++)
        {
            if (A[i] > max_value)
            {
                max_value = A[i];
            }
            else if (A[i] < min_value)
            {
                min_value = A[i];
            }
        }
        return max(0, max_value - min_value - (K << 1));
    }
    ```

    [here-910](https://leetcode.com/problems/smallest-range-ii/)限定这个题的调节量只能是$-k$和$+k$，而不能是$\pm k$范围内的任意数。

    ```cpp
    int smallestRangeII(vector<int>& A, int K) {
        const int count=A.size();
		sort(A.begin(),A.end());
		int ans=A[count-1]-A[0];
		for(int i = 0; i < count-1; i++)
		{
			int a=A[i],b=A[i+1];
			int high=max(A[count-1]-K,a+K);
			int low=min(A[0]+K,b-K);
			ans=min(ans,high-low);
		}
		return ans;
	}
    ```

- [914. X of a Kind in a Deck of Cards](https://leetcode.com/problems/x-of-a-kind-in-a-deck-of-cards/)

    当给定数组deck长度小于2时明显不可能，当长度大于等于2时，统计每个不同的数出现的次数（桶统计或者hashmap），然后求所有次数的最大公约数（X）大约等于2即可

    ```cpp
    int gcd(int a, int b)
    {
        if (a < b){
            swap(a, b);
        }
        while (b != 0){
            int r = a % b;
            a = b;
            b = r;
        }
        return a;
    }
    bool hasGroupsSizeX(vector<int> &deck)
    {
        const int count = deck.size();
        bool ret = false;
        if (count >= 2)
        {
            unordered_map<int, int> number_of_unique_deck;
            for (auto &&v : deck)
            {
                if (number_of_unique_deck.find(v) != number_of_unique_deck.end())
                {
                    number_of_unique_deck[v]++;
                }
                else
                {
                    number_of_unique_deck[v] = 1;
                }
            }
            vector<int> values;
            for (auto &&item : number_of_unique_deck)
            {
                values.push_back(item.second);
            }
            int x = values[0]; // deck非空时values至少有一个值
            for (int i = 1; i < values.size(); i++)
            {
                x = gcd(x, values[i]);
            }
            ret = x >= 2;
        }
        return ret;
    }
    ```

- [915. Partition Array into Disjoint Intervals](https://leetcode.com/problems/partition-array-into-disjoint-intervals/)

    设left长度为mid，则在满足$max_{i=0}^{mid-1}A[i] \le min_{i=mid}^{n-1}A[i]$的条件下求$min(mid)$，时间复杂度、空间复杂度均为$O(n)$
    
    ```cpp
    int partitionDisjoint(vector<int> &A)
    {
        const int count = A.size();
        vector<int> rightMin(count, 0);
        rightMin[count - 1] = A[count - 1];
        int leftMax = numeric_limits<int>::min();
        for (int i = count - 2; i >= 0; i--)
        {
            rightMin[i] = min(A[i], rightMin[i + 1]);
        }
        int ret = 0;
        for (int i = 0; i < count - 1; i++)
        {
            leftMax = max(leftMax, A[i]);
            if (leftMax <= rightMin[i + 1])
            {
                ret = i + 1;
                break;
            }
        }
        return ret;
    }
    ```

- [916. Word Subsets](https://leetcode.com/problems/word-subsets/)

    - 要求A中universal的word，针对A中的每个word，如果是universal的需要满足$letters\_count\_s[j] \ge letters\_count\_B[i][j]$，其中i遍历B中的全部单词，j遍历26个字母，$letters\_count\_s[j]$表示A中单词word的组成字母统计，$letters\_count\_B[i][j]$表示B中每个单词的组成字母统计，时间复杂度$O(A.length*B.length)$，答案正确，显然会$\color{red}{TLE}$

    ```cpp
    vector<string> wordSubsets(vector<string> &A, vector<string> &B)
    {
        const int count_B = B.size();
        vector<vector<int>> letters_count_B(count_B, vector<int>(26, 0));
        for (int i = 0; i < count_B; i++)
        {
            for (auto &&ch : B[i])
            {
                letters_count_B[i][(int)(ch - 'a')]++;
            }
        }
        vector<string> ans;
        for (auto &&s : A)
        {
            vector<int> letters_count_s(26, 0);
            for (auto &&ch : s)
            {
                letters_count_s[(int)(ch - 'a')]++;
            }
            bool flag = true;
            for (int i = 0; flag && i < count_B; i++)
            {
                for (int j = 0; flag && j < 26; j++)
                {
                    if (letters_count_s[j] < letters_count_B[i][j])
                    {
                        flag = false;
                    }
                }
            }
            if (flag)
            {
                ans.push_back(s);
            }
        }
        return ans;
    }
    ```

    - 上述方法中的$letters\_count\_s[j] \ge letters\_count\_B[i][j]$可以压缩为$letters\_count\_s[j] \ge max_{i=0}^{B.length-1}letters\_count\_B[i][j]$，时间复杂度$O(A.length+B.length)$，$\color{green}{Accepted}$

    ```cpp
    vector<string> wordSubsets(vector<string> &A, vector<string> &B)
    {
        const int letters = 26;
        vector<int> letters_count_max(26, 0);
        for (auto &&s : B)
        {
            vector<int> letters_count_temp(26, 0);
            for (auto &&ch : s)
            {
                letters_count_temp[(int)(ch - 'a')]++;
            }
            for (int i = 0; i < letters; i++)
            {
                letters_count_max[i] = max(letters_count_max[i], letters_count_temp[i]);
            }
        }
        vector<string> ans;
        for (auto &&s : A)
        {
            vector<int> letters_count_s(letters, 0);
            for (auto &&ch : s)
            {
                letters_count_s[(int)(ch - 'a')]++;
            }
            bool flag = true;
            for (int j = 0; flag && j < letters; j++)
            {
                if (letters_count_s[j] < letters_count_max[j])
                {
                    flag = false;
                }
            }
            if (flag)
            {
                ans.push_back(s);
            }
        }
        return ans;
    }
    ```

- [918. Maximum Sum Circular Subarray](https://leetcode.com/problems/maximum-sum-circular-subarray/)

    通常在给定一个array的情况下，求子数组subarray和最大的方法为动态规划DP，以array[j]结尾的subarray和最大为$dp_{j+1} = array_j+max(0,dp_j),dp_0=array_0$

	- [Approach 1] Next array

	```cpp
    int maxSubarraySumCircular(vector<int> &A)
    {
        // 从A[i]到A[A.length-1]之间没有circle可能构成的subarray最大和，DP
        int ans = A[0], cur = A[0], length = A.size();
        // cur dp[j+1]=max(dp[j],0)+A[j+1] , then ans= max(cur)
        for (int i = 1; i < length; i++)
        {
            cur = A[i] + max(0, cur);
            ans = max(ans, cur);
        }
        // leftsum[i] = A[0]+A[1]+...+A[i]
        // rightsum[i] = A[i]+A[i+1]+...+A[A.length-1]
        vector<int> leftsum = A, rightsum = A;
        for (auto i = 1; i < length; i++)
        {
            leftsum[i] += leftsum[i - 1];
            rightsum[length - i - 1] += rightsum[length - i];
        }
        vector<int> max_rightsum = rightsum;
        for (auto i = length - 2; i >= 0; i--)
        {
            max_rightsum[i] = max(max_rightsum[i], max_rightsum[i + 1]);
        }
        for (auto i = 0; i < length - 2; i++)
        {
            ans = max(ans, leftsum[i] + max_rightsum[i + 2]);
        }
        return ans;
    }
	```

	- [Approach 2] Prefix Sums + Monoqueue，双端队列的应用

	```cpp
    int maxSubarraySumCircular(vector<int> &A)
    {
        int length = A.size();
        vector<int> leftsum(2 * length + 1, 0);
        for (int i = 0; i < 2 * length; i++)
        {
            leftsum[i + 1] = leftsum[i] + A[i % length];
        }
        // for each j, largest leftsum[j] - leftsum[i]
        // for each j, find a smallest leftsum[i] (i<j and j-i<=length)
        int ans = A[0];
        deque<int> qe;
        qe.push_back(0); // 双端队列
        for (int j = 1; j <= 2 * length; j++)
        {
            if (qe.front() < j - length)
            {
                qe.pop_front();
            }
            ans = max(ans, leftsum[j] - leftsum[qe.front()]);
            while (!qe.empty() && leftsum[j] <= leftsum[qe.back()])
            {
                qe.pop_back();
            }
            qe.push_back(j);
        }
        return ans;
    }
	```

	- [Approach 3] Kadane's (Sign Variant)

	```cpp
    class Solution
    {
    private:
        long long dp_helper(vector<int> &A, int left, int right, int sign)
        {
            // dp[i+1] = max(dp[i],0)+A[i+1]
            long long zero = 0, ret = numeric_limits<int>::min(), cur = numeric_limits<int>::min();
            for (int i = left; i <= right; i++)
            {
                cur = max(cur, zero) + sign * A[i];
                ret = max(ret, cur);
            }
            return ret;
        }

    public:
        int maxSubarraySumCircular(vector<int> &A)
        {
            long long sum_A = 0;
            for (auto &&v : A)
            {
                sum_A += v;
            }
            /*
                max(A[0]+A[1]+..+A[i] + A[j]+A[j+1]+...+A[length-1])
                = sum_A - min(A[i+1]+...+A[j-1])
                = sum_A + kadane(-A)
                ****
                为了保证[0,i],[j,length-1]两个区间非空，kadane(-A)需要在[0,length-2]和[1,length-1]上执行两次
            */
            const int length = A.size();
            long long ans_kadane = dp_helper(A, 0, length - 1, 1);
            long long ans_negative_1 = sum_A + dp_helper(A, 0, length - 2, -1);
            long long ans_negative_2 = sum_A + dp_helper(A, 1, length - 1, -1);
            vector<long long> ans{ans_kadane, ans_negative_1, ans_negative_2};
            return (int)*max_element(ans.begin(), ans.end());
        }
    };
	```

	- [Approach 4] Kadane's (Min Variant)

	```cpp
    class Solution
    {
    private:
        long long dp_helper(vector<int> &A, int left, int right, bool max_min)
        {
            // max_min as true for max, dp[i+1] = max(dp[i],0)+A[i+1]
            // max_min as false for min, dp[i+1] = min(dp[i],0)+A[i+1]
            long long zero = 0, ret, cur;
            if (max_min)
            {
                ret = numeric_limits<int>::min(), cur = numeric_limits<int>::min();
                for (int i = left; i <= right; i++)
                {
                    cur = max(cur, zero) + A[i];
                    ret = max(ret, cur);
                }
            }
            else
            {
                ret = numeric_limits<int>::max(), cur = numeric_limits<int>::max();
                for (int i = left; i <= right; i++)
                {
                    cur = min(cur, zero) + A[i];
                    ret = min(ret, cur);
                }
            }
            return ret;
        }

    public:
        int maxSubarraySumCircular(vector<int> &A)
        {
            long long sum_A = 0;
            for (auto &&v : A)
            {
                sum_A += v;
            }
            /*
                max(A[0]+A[1]+..+A[i] + A[j]+A[j+1]+...+A[length-1])
                = sum_A - min(A[i+1]+...+A[j-1])
                ****
                为了保证[0,i],[j,length-1]两个区间非空，min(A[i+1]+...+A[j-1])需要在[0,length-2]和[1,length-1]上执行两次
            */
            const int length = A.size();
            long long ans_kadane = dp_helper(A, 0, length - 1, true);
            long long ans_negative_1 = sum_A - dp_helper(A, 0, length - 2, false);
            long long ans_negative_2 = sum_A - dp_helper(A, 1, length - 1, false);
            vector<long long> ans{ans_kadane, ans_negative_1, ans_negative_2};
            return (int)*max_element(ans.begin(), ans.end());
        }
    };
	```

- [924. Minimize Malware Spread](https://leetcode.com/problems/minimize-malware-spread/)

    - 并查集的建立与使用

	```cpp
	class Solution
	{
	private:
		int find(vector<int> &uf, int x)
		{
			// 含有路径压缩的并查集
			return uf[x] == x ? x : (uf[x] = find(uf, uf[x]));
		}

	public:
		int minMalwareSpread(vector<vector<int>> &graph, vector<int> &initial)
		{
			// union find initialization for n nodes in the network
			const int n = graph.size();
			vector<int> uf(n);
			for (auto i = 0; i < n; i++)
			{
				uf[i] = i;
			}
			// update of the un
			for (auto i = 0; i < n; i++)
			{
				for (auto j = 0; j < i; j++)
				{
					if (graph[i][j])
					{
						uf[find(uf, uf[i])] = uf[find(uf, uf[j])];
					}
				}
			}
			// count for every inital infected node
			unordered_map<int, int> groupID2nodeNum, groupID2cachedNum;
			for (auto i = 0; i < n; i++)
			{
				groupID2nodeNum[find(uf, i)]++;
			}
			sort(initial.begin(), initial.end());
			int count = initial.size();
			for (auto i = 0; i < count; i++)
			{
				int group = find(uf, initial[i]);
				groupID2cachedNum[group]++;
			}
			int ret = initial[0];
			int cur_max_node_removed = (groupID2cachedNum[find(uf, ret)] == 1) ? groupID2nodeNum[find(uf, ret)] : 0;
			for (auto i = 0; i < count; i++)
			{
				int node = initial[i], group = find(uf, node);
				if (groupID2nodeNum[group] > cur_max_node_removed && groupID2cachedNum[group] == 1)
				{
					ret = node, cur_max_node_removed = groupID2nodeNum[group];
				}
			}
			return ret;
		}
	};
	```

- [929](https://leetcode.com/problems/unique-email-addresses/)
    两个考察点
    - cpp STL中的string操作，子串、查找、替换等
    - cpp STL中集合(set)的使用，或者自己实现一个set

- [930](https://leetcode.com/problems/binary-subarrays-with-sum/)

    首先统计数组中所以1的下标位置，注意最左端插入-1、最右端插入数组大小size，帮助处理边界情况，然后对target的值分0和非0两种情况处理
    - 0只能由连续的一段0构成，则连续的$n$个0可以构成$\frac{n*(n-1)}{2}$个连续子数组，其和均为0
    - 非0值x只能由x个1的和构成，这些1的下标的长度应为x，然后左右两侧可以包含任意长度的0，则当其左右两侧0的个数为left和right时，符合条件的连续子数组数量为$(left+1)*(right+1)$

    ```cpp
    int numSubarraysWithSum(vector<int> &A, int S)
    {
        int count = A.size(), ans = 0;
        vector<int> index_of_one;
        index_of_one.push_back(-1); // for the first number is 0
        for (int i = 0; i < count; i++)
        {
            if (A[i] == 1)
            {
                index_of_one.push_back(i);
            }
        }
        index_of_one.push_back(count); // for the last number is 0
        if (S == 0)
        {
            // special case
            const int length = index_of_one.size();
            for (int i = 1; i < length; i++)
            {
                int n = index_of_one[i] - index_of_one[i - 1] - 1;
                ans += n * (n + 1) / 2;
            }
        }
        else
        {
            // S > 0
            const int right_end = index_of_one.size() - S;
            for (int i = 1; i < right_end; i++)
            {
                int left = index_of_one[i] - index_of_one[i - 1];
                int right = index_of_one[i + S] - index_of_one[i + S - 1];
                ans += left * right;
            }
        }
        return ans;
    }
    ```    

- [931. Minimum Falling Path Sum](https://leetcode.com/problems/minimum-falling-path-sum/)
    
    在给定的矩阵中每行选择一个数，如果第i行选择了第j列，则第i+1行只能在第j-1、j、j+1三列中选择，使得选择的所有数的和最小。典型的动态规划问题，用dp[i][j]表示以A[i][j]为终点的Minimum Falling Path Sum，则$dp[i][j]=A[i][j]+min(dp[i-1][j-1]+dp[i-1][j]+dp[i-1][j+1])$，然后在最后一行的值中取最小值即可。

    ```cpp
    int minFallingPathSum(vector<vector<int>>& A) {
        int ans=0;
		if(!((A.size()==0)||(A[0].size()==0))){
			int rows=A.size(),cols=A[0].size();
			if(cols<2){
				for (int i = 0; i < rows; i++)
				{
					ans+=A[i][0];
				}
			}else{
				for (size_t i = 1; i < rows; i++)
				{
					A[i][0]+=min(A[i-1][0],A[i-1][1]);
					for (int j = 1; j < cols-1; j++)
					{
						A[i][j]+=min(min(A[i-1][j-1],A[i-1][j]),A[i-1][j+1]);
					}
					A[i][cols-1]+=min(A[i-1][cols-2],A[i-1][cols-1]);
				}
				ans=numeric_limits<int>::max();
				for (int i = 0; i < rows; i++)
				{
					ans=min(ans,A[rows-1][i]);
				}
			}
		}
		return ans;
    }
    ```

- [932. Beautiful Array](https://leetcode.com/problems/beautiful-array/)

	- 分治法，时间复杂度$O(nlog(n))$

	```cpp
	class Solution
	{
	private:
		unordered_map<int, vector<int>> ans;
		vector<int> helper(int n)
		{
			if (ans.find(n) == ans.end())
			{
				vector<int> ret(n);
				if (n == 1)
				{
					ret[0] = 1;
				}
				else
				{
					int k = 0;
					for (auto &&x : helper((n + 1) / 2))
					{
						ret[k++] = 2 * x - 1;
					}
					for (auto &&x : helper(n / 2))
					{
						ret[k++] = 2 * x;
					}
				}
				ans[n] = ret;
			}
			return ans[n];
		}

	public:
		vector<int> beautifulArray(int N)
		{
			return helper(N);
		}
	};
	```

- [937. Reorder Data in Log Files](https://leetcode.com/problems/reorder-data-in-log-files/)

    本题即为简单的字符串排序，但是题目意思要求的排序规则，英文很难理解，应该时:
    - 首先letters-logs在前，digit-logs在后
    - letters-logs部分按照identifier后面log的字典序排列，log相同的再按identifier的字典序
    - digit-logs按照输入顺序排列不变

    - 先记录digit-logs，对letters-logs排序后加入digit-logs

    ```cpp
    vector<string> reorderLogFiles(vector<string> &logs)
    {
        vector<int> digital_logs_index;
        vector<string> ans;
        for (int i = 0; i < logs.size(); i++)
        {
            int index = logs[i].find_first_of(' ');
            if (isdigit(logs[i][index + 1]))
            {
                digital_logs_index.push_back(i);
            }
            else
            {
                ans.push_back(logs[i]);
            }
        }
        sort(ans.begin(), ans.end(), [](string a, string b) -> bool {
            int pos_a = a.find_first_of(' '), pos_b = b.find_first_of(' ');
            string x = a.substr(pos_a + 1, a.length() - pos_a - 1), y = b.substr(pos_b + 1, b.length() - pos_b - 1);
            int ret = x.compare(y);
            return (ret == 0) ? (a.compare(b) < 0) : (ret < 0);
        });
        for (auto &&index : digital_logs_index)
        {
            ans.push_back(logs[index]);
        }
        return ans;
    }
    ```

    - 直接对全体logs稳定排序

    ```cpp
    vector<string> reorderLogFiles(vector<string> &logs)
    {
        stable_sort(logs.begin(), logs.end(), [](string a, string b) -> bool {
            int pos_a = a.find_first_of(' '), pos_b = b.find_first_of(' ');
            bool ret = false, digital_a = isdigit(a[pos_a + 1]), digital_b = isdigit(b[pos_b + 1]);
            if (!digital_a && digital_b)
            {
                ret = true;
            }
            else if (!digital_a && !digital_b)
            {
                int cmp = a.substr(pos_a + 1, a.length() - pos_a - 1).compare(b.substr(pos_b + 1, b.length() - pos_b - 1));
                ret = (cmp == 0) ? (a.compare(b) < 0) : (cmp < 0);
            }
            return ret;
        });
        return logs;
    }
    ```

- [941](https://leetcode.com/problems/valid-mountain-array/)

    重点在于要有increasing的过程，也要有decreasing的过程。

    ```cpp
    bool validMountainArray(vector<int>& A) {
        const int count=A.size()-1;
		if(count<2){
			return false;
		}else{
			int i=0;
			bool increasing=false,decreasing=false;
			while(i<count && A[i]<A[i+1]){
				i++;
				increasing=true;
			}
			while(i<count && A[i]>A[i+1]){
				i++;
				decreasing=true;
			}
			return (i==count)&&increasing&&decreasing;
		}
    }
    ```

- [945. Minimum Increment to Make Array Unique](https://leetcode.com/problems/minimum-increment-to-make-array-unique/)

    - 排序，时间复杂度$O(nlog(n))$

    ```cpp
    int minIncrementForUnique(vector<int> &A)
    {
        int count = A.size(), moves = 0;
        if (count > 0)
        {
            sort(A.begin(), A.end());
            int cur_min_value = *A.begin();
            for (auto i = 1; i < count; i++)
            {
                moves += A[i] <= cur_min_value ? cur_min_value - A[i] + 1 : 0; // self increment by step=1
                cur_min_value = max(cur_min_value + 1, A[i]);
            }
        }
        return moves;
    }
    ```

    - 并查集UnionFind，时间复杂度$O(n)$

    ```cpp
    class Solution
    {
    private:
        unordered_map<int, int> count; // 值 - 所属的连通分量
        int find(int x)
        {
            return count[x] = count.count(x) ? find(count[x] + 1) : x;
        }

    public:
        int minIncrementForUnique(vector<int> &A)
        {
            int moves = 0;
            for (auto v : A)
            {
                moves += find(v) - v;
            }
            return moves;
        }
    };
    ```

- [947. Most Stones Removed with Same Row or Column](https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/)

    - 暴力二重遍历所有节点，使用并查集标注联通分量

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
                uf.resize(n);
                count = n;
                for (int i = 0; i < n; i++)
                {
                    uf[i] = i;
                }
            }
            int find(int x)
            {
                return x == uf[x] ? x : (uf[x] = find(uf[x]));
            }
            bool union_merge(int x, int y)
            {
                x = find(x), y = find(y);
                if (x != y)
                {
                    uf[x] = y, count--;
                    return true;
                }
                return false;
            }
        };

    public:
        int removeStones(vector<vector<int>> &stones)
        {
            int n = stones.size();
            UF uf = UF(n);
            // 用并查集来计算所有stone作为节点表示的图中联通分量的个数
            // 每个联通分量都可以按照规则删除到只剩一个节点
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (stones[i][0] == stones[j][0] || stones[i][1] == stones[j][1])
                    {
                        uf.union_merge(i, j); // 横纵坐标有一个相同则可以合并到同一个连通分量中
                    }
                }
            }
            return uf.count;
        }
    };
    ```
    
    - 并查集的建立与使用，时间复杂度$O(nlog(n)))$

	```cpp
	class Solution
	{
	private:
		int find(vector<int> &father, int x)
		{
			// 包含路径压缩的并查集查找
			return father[x] == x ? x : (father[x] = find(father, father[x]));
		}
		void union_insert(vector<int> &father, int x, int y)
		{
			father[find(father, x)] = father[find(father, y)];
		}

	public:
		int removeStones(vector<vector<int>> &stones)
		{
			// 总的石头数量
			int total_stones = stones.size();
			/***
			* 1. 使用并查集将所有的石头分类为一个个的连通域，每个连通域可以move删除操作直到最后一个石头
			* 2. 所以可以移除的石头数量 ret = total_stones - n_groups (连通域的数量)
			*/
			unordered_map<int, vector<int>> x2points, y2points;
			vector<int> father(total_stones);
			for (auto i = 0; i < total_stones; i++)
			{
				father[i] = i;
				x2points[stones[i][0]].push_back(i);
				y2points[stones[i][1]].push_back(i);
			}
			for (auto &&item : x2points)
			{
				vector<int> points = item.second;
				int n = points.size();
				for (auto i = 1; i < n; i++)
				{
					union_insert(father, points[0], points[i]);
				}
			}
			for (auto &&item : y2points)
			{
				vector<int> points = item.second;
				int n = points.size();
				for (auto i = 1; i < n; i++)
				{
					union_insert(father, points[0], points[i]);
				}
			}
			unordered_set<int> groups;
			for (auto i = 0; i < total_stones; i++)
			{
				groups.insert(find(father, i));
			}
			return total_stones - groups.size();
		}
	};
	```

	- 优化的并查集

	```cpp
	class Solution
	{
	private:
		unordered_map<int, int> uf; // 每个点的横坐标、纵坐标映射到其代表（并查集）
		int island;
		int find(int x)
		{
			if (!uf.count(x))
			{
				island++;
				uf[x] = x;
			}
			if (uf[x] != x)
			{
				uf[x] = find(uf[x]);
			}
			return uf[x];
		}
		void union_insert(int x, int y)
		{
			x = find(x), y = find(y);
			if (x != y)
			{
				uf[x] = y, island--;
			}
		}

	public:
		int removeStones(vector<vector<int>> &stones)
		{
			uf.clear();
			island = 0;
			for (auto &&p : stones)
			{
				union_insert(p[0], ~p[1]);
			}
			return stones.size() - island;
		}
	};
	```

- [950](https://leetcode.com/problems/reveal-cards-in-increasing-order/)

    按照揭牌的顺序，反向操作模拟
    题目给定的揭牌操作：
    - 揭开最上面一张牌
    - 把下一张牌移到最下面

    最终得到升序的序列

    因此反向操作，首先对给定数组升序排序，然后
    - 牌堆最后一张移到最上面
    - 有序数列的最后一个（当前最大值）放到牌堆最上面盖住

    直到给定的数列完全被放到牌堆里

    时间复杂度$O(nlog(n))$

- [951. Flip Equivalent Binary Trees](https://leetcode.com/problems/flip-equivalent-binary-trees/)

    判断两颗二叉树是否相同或经过翻转后是否相同，递归判断

    ```cpp
    bool flipEquiv(TreeNode *root1, TreeNode *root2)
    {
        bool ret = false;
        if (!root1 && !root2)
        {
            ret = true;
        }
        else if (root1 && root2 && root1->val == root2->val)
        {
            ret = (flipEquiv(root1->left, root2->left) && flipEquiv(root1->right, root2->right)) || (flipEquiv(root1->left, root2->right) && flipEquiv(root1->right, root2->left));
        }
        return ret;
    }
    ```

- [952. Largest Component Size by Common Factor](https://leetcode.com/problems/largest-component-size-by-common-factor/)

	并查集的建立与使用，时间复杂度$O(n*sqrt(max_value(A)))$

	```cpp
	class Solution
	{
	private:
		int find(vector<int> &father, int x)
		{
			return father[x] == x ? x : (father[x] = find(father, father[x]));
		}

	public:
		int largestComponentSize(vector<int> &A)
		{
			int ret = 0, max_value = *max_element(A.begin(), A.end());
			// 并查集初始化
			vector<int> father(max_value + 1, 0);
			for (auto i = 0; i <= max_value; i++)
			{
				father[i] = i;
			}
			for (auto &&v : A)
			{
				// 遍历A中的每一个值v
				for (int d = sqrt(v); d >= 2; d--)
				{
					// 遍历v所有可能的大于1的factor
					if (v % d == 0)
					{
						// 添加可能的边（v和d，v和v/d）
						father[find(father, v)] = father[find(father, d)];
						father[find(father, v)] = father[find(father, v / d)];
					}
				}
			}
			// 用hashmap来统计每个连通区域的大小
			unordered_map<int, int> map;
			for (auto &&v : A)
			{
				// 对于A中的每个v，将其所在联通区域的节点个数加1，并在此过程中更新节点数统计的最大值ret
				ret = max(ret, ++map[find(father, v)]);
			}
			return ret;
		}
	};
	```

- [955. Delete Columns to Make Sorted II](https://leetcode.com/problems/delete-columns-to-make-sorted-ii/)

	贪心算法，遍历每一列是否会删除，如果不删除标记将会造成那些行符合字典序要求（标记的这些行将不参与之后列的判断），时间复杂度$O(rows*cols)$

	```cpp
	int minDeletionSize(vector<string> &A)
	{
		int ret = 0, rows = A.size(), cols = A[0].size();
		vector<bool> sorted(rows - 1, false);
		for (auto j = 0; j < cols; j++)
		{
			/* 检查每一列是否为字典序 */
			bool deleted_col = false;
			vector<int> sorted_rows;
			for (auto i = 1; i < rows; i++)
			{
				if (sorted[i - 1] == false)
				{
					if (A[i - 1][j] < A[i][j])
					{
						sorted_rows.push_back(i - 1);
					}
					else if (A[i - 1][j] > A[i][j])
					{
						deleted_col = true;
						break;
					}
				}
			}
			// 一列遍历完成以后，处理这一列的情况，是否删除，如果不删除将造成那些行成为字典序的
			if (deleted_col)
			{
				ret++;
			}
			else
			{
				for (auto &&index : sorted_rows)
				{
					sorted[index] = true;
				}
			}
		}
		return ret;
	}
	```

	- some test cases

	```cpp
	["ca","bb","ac"]
	["xc","yb","za"]
	["zyx","wvu","tsr"]
	["xga","xfb","yfa"]
	["abx","agz","bgc","bfc"]
	["doeeqiy","yabhbqe","twckqte"]
	```

- [959. 由斜杠划分区域](https://leetcode-cn.com/problems/regions-cut-by-slashes/)

    使用并查集实现，最后没有被斜杠或者反斜杠分开的区域视为两个联通的区域合并，最终连通区域的个数即为被划分的区域数量，时间复杂度$O(n^2*\alpha(4*n^2))$，其中$n=grid.size()$，$\alpha(4*n^2)$是路径压缩实现下并查集的单次查找时间

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
                uf.resize(n);
                count = n;
                for (int i = 0; i < n; i++)
                {
                    uf[i] = i;
                }
            }
            int find(int x)
            {
                return x == uf[x] ? x : (uf[x] = find(uf[x]));
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
        int regionsBySlashes(vector<string> &grid)
        {
            /**
            * 假设每个格子内都有斜杠和反斜杠，则可以将格子分割为四个小格子，
            * 从最上面的一个开始逆时针分别编号为0123
            * 1. 没有斜杠，则01联通，23联通
            * 2. 没有反斜杠，则03联通，12联通
            * 3. 默认情况下左右两个格子的13是联通的，上下两个格子的20是联通的
            * 使用并查集记录这个联通过程，则最后的连通域个数即是划分的区域数
            * 
            */
            int ret = 0;
            if (grid.size() > 0 && grid[0].size() > 0)
            {
                const int n = grid.size(); // 给定的是n*n的方阵
                const int size = 4 * n * n;
                UF uf = UF(size);
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        int d0 = 4 * (i * n + j);
                        int d1 = d0 + 1, d2 = d0 + 2, d3 = d0 + 3;
                        if (grid[i][j] == '/')
                        {
                            uf.union_merge(d0, d3);
                            uf.union_merge(d1, d2);
                        }
                        else if (grid[i][j] == '\\')
                        {
                            uf.union_merge(d0, d1);
                            uf.union_merge(d2, d3);
                        }
                        else
                        {
                            uf.union_merge(d0, d1);
                            uf.union_merge(d1, d2);
                            uf.union_merge(d2, d3);
                        }
                        if (i > 0)
                        {
                            // 0与上面一格的2相连
                            uf.union_merge(d0, 4 * ((i - 1) * n + j) + 2);
                        }
                        if (j > 0)
                        {
                            // 3与左面一格的1相连
                            uf.union_merge(d3, 4 * (i * n + j - 1) + 1);
                        }
                    }
                }
                ret = uf.count;
            }
            return ret;
        }
    };
    ```

- [961](https://leetcode.com/problems/n-repeated-element-in-size-2n-array/)
    In a array A of size 2N, there are N+1 unique elements, and exactly one of these elements is repeated N time, find and return this element.
    - HashTable 通过hash统计找到个数不是1的那个数，时间复杂度为O(N)，空间复杂度O(1)
    - 根据排列组合，相当于把N个不同的数插入到N个重复的数中，因此无论怎么插入连续的两个数作为一组中一定有一个数是那个重复的major element，因此两组、即连续的四个数中一定有两个major element，它们是相等的，因此按照步长为1,2,3,4搜索相等的两个数即可，时间复杂度与hash法相同为O(N)，空间复杂度O(1)

    ```cpp
    for (int i=0; i<A.size(); i+=2) {
        if (A[i] == A[i+1]) return A[i];
    }
    return A[0] == A[2] || A[0] == A[3] ? A[0] : A[1];
    ```

- [962. Maximum Width Ramp](https://leetcode.com/problems/maximum-width-ramp/)

    - 暴力搜索，对当前A[i]，从A的右侧开始找到第一个$A[j]>=A[i]$，则$ret=max(ret,j-i)$，时间复杂度$O(n^2)$，LeetCode时间效率$\color{red}{TLE}$

    ```cpp
    int maxWidthRamp(vector<int> &A)
    {
        const int length = A.size();
        int ret = 0;
        for (int i = 0; i < length; i++)
        {
            for (int j = length-1; j>i; j--)
            {
                if(A[i]<=A[j]){
                    ret = max(ret, j - i);
                    break;
                }
            }
        }
        return ret;
    }
    ```

    - 根据数组A的大小对其下标进行排序(稳定排序)，时间复杂度$O(nlog(n))$

    ```cpp
    int maxWidthRamp(vector<int> &A)
    {
        vector<vector<int>> values;
        for (int i = 0; i < A.size(); i++)
        {
            values.push_back({i, A[i]});
        }
        stable_sort(values.begin(), values.end(), [](const vector<int> &a, const vector<int> &b) -> bool { return a[1] < b[1]; });
        int ramp = 0, pre_index = A.size();
        for (auto &&item : values)
        {
            int i = item[0];
            ramp = max(ramp, i - pre_index); // 通过排序保证A[pre_index]<=A[i]，即A[pre_index]是比A[i]小的数
            pre_index = min(pre_index, i);   // 保证pre_index最小，从而i-pre_index是当前最大，即A[pre_index]是A中比A[i]小的数中最靠左的
        }
        return ramp;
    }
    ```

    - 维护一个降序栈，时间复杂度$O(n)$

    ```cpp
    int maxWidthRamp(vector<int> &A)
    {
        stack<int> st;
        for (int i = 0; i < A.size(); i++)
        {
            if (st.empty() || A[st.top()] > A[i])
            {
                st.push(i);
            }
        }
        int ramp = 0;
        for (int i = A.size() - 1; i > ramp; i--)
        {
            while (!st.empty() && A[i] >= A[st.top()])
            {
                ramp = max(ramp, i - st.top()), st.pop();
            }
        }
        return ramp;
    }
    ```

- [966](https://leetcode.com/problems/vowel-spellchecker/)

    给定一个字典wordlist，给定查询单词word：
    - word在wordlist中有完全匹配的单词s，返回s
    - 忽略大小写之后word在wordlist中有匹配的单词s，返回s
    - 忽略大小写并将元音字母全部替换为任意的其它元音字母之后word在wordlist中存在匹配的单词s，返回s
    - 以上三种情况均不符合在返回空字符串

    ```cpp
    vector<string> spellchecker(vector<string>& wordlist, vector<string>& queries) {
		/*
			three pass scan:
			first pass for completely matched
			second pass for match up to capitalization
			three pass for match with vowel error
			brute force scan will get a TLE, so we need store wordlist with hashset and hashmap
		*/
		unordered_set<string> wordlist_correct;
		unordered_map<string,string> wordlist_lowercase;
		unordered_map<string,string> wordlist_without_vowel;
		string vowels="aoeiu";
		// re-struct the wordlist
		for (int i = wordlist.size()-1; i>=0; i--)
		{
			wordlist_correct.insert(wordlist[i]);
			string lowercase,no_vewel;
			for(const auto c:wordlist[i]){
				char lowercase_c=tolower(c);
				lowercase+=lowercase_c;
				if(vowels.find(lowercase_c)==string::npos){
					no_vewel+=lowercase_c;
				}else{
					no_vewel+='*';
					// to guarantee length of word is equal
				}
			}
			wordlist_lowercase[lowercase]=wordlist[i];
			wordlist_without_vowel[no_vewel]=wordlist[i];
		}
		// queries
		vector<string> ans;
		for (auto const &s:queries)
		{
			if(wordlist_correct.find(s)!=wordlist_correct.end()){
				ans.push_back(s);
				break;
			}
			string lowercase,no_vewel;
			for(const auto c:s){
				char lowercase_c=tolower(c);
				lowercase+=lowercase_c;
				if(vowels.find(lowercase_c)==string::npos){
					no_vewel+=lowercase_c;
				}else{
					no_vewel+='*';
					// to guarantee length of word is equal
				}
			}
			if(wordlist_lowercase.find(lowercase)!=wordlist_lowercase.end()){
				ans.push_back(wordlist_lowercase[lowercase]);
				break;
			}
			if(wordlist_without_vowel.find(no_vewel)!=wordlist_without_vowel.end()){
				ans.push_back(wordlist_without_vowel[no_vewel]);
				break;
			}
			// no match
			ans.push_back("");
		}
		return ans;
    }
    ```

    本题充分体现了hash存储的优势

- [969. Pancake Sorting](https://leetcode.com/problems/pancake-sorting/)

    采用插入排序的思想，通过每两次flip将当前最大数置换到当前数组的最后位置，时间复杂度$O(n^2)$

    ```cpp
    vector<int> pancakeSort(vector<int> &A)
    {
        int cur = 0;
        vector<int> ret;
        for (int last = A.size() - 1; last >= 0; last--)
        {
            if (A[last] != last + 1)
            {
                int i = 0;
                while (A[i] != last + 1)
                {
                    i++;
                }
                ret.push_back(i + 1);
                reverse(A.begin(), A.begin() + i + 1);
                if (A[last] != last + 1)
                {
                    ret.push_back(last + 1);
                    reverse(A.begin(), A.begin() + last + 1);
                }
            }
        }
        return ret;
    }
    ```

- [973. 最接近原点的 K 个点](https://leetcode-cn.com/problems/k-closest-points-to-origin/)

    特别注意优先队列实现的语法，即优先队列中元素的构造(定义，构造方法，比较运算符的重载)，时间复杂度$O(nlog(n))$

    ```cpp
    class Solution
    {
    private:
        struct Point
        {
            int x, y;
            Point(vector<int> p)
            {
                x = p[0], y = p[1];
            }
            bool operator<(const Point &t) const
            {
                return x *x + y *y > t.x *t.x + t.y *t.y;
            }
        };

    public:
        vector<vector<int>> kClosest(vector<vector<int>> &points, int K)
        {
            priority_queue<Point> qe;
            for (auto &&v : points)
            {
                Point cur = Point(v);
                qe.push(cur);
            }
            vector<vector<int>> ret(K);
            for (int i = 0; i < K; i++)
            {
                Point cur = qe.top();
                ret[i] = {cur.x, cur.y};
                qe.pop();
            }
            return ret;
        }
    };
    ```

- [974. Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/)

    此类连续子数组和差之类的问题，大多使用prefix sum的方式解决

    ```cpp
    int subarraysDivByK(vector<int> &A, int K)
    {
        int count = A.size(), prefix_sum = 0, ans = 0;
        vector<int> remainder_count(K, 0);
        remainder_count[0] = 1; // for the first prefix sum with value K
        for (int i = 0; i < count; i++)
        {
            prefix_sum += A[i];
            remainder_count[(prefix_sum % K + K) % K]++; // avoiding negative remainder
        }
        for (auto &&v : remainder_count)
        {
            ans += ((v * (v - 1)) >> 1);
        }
        return ans;
    }
    ```

- [976](https://leetcode.com/problems/largest-perimeter-triangle/)

    给定一个$size>3$数组array，求数组中数可以组成的周长最长的三角形，这本质上是个数学题。首先岁数组中所有的数按降序排列有
    $$a_0>a_1>a_2>...>a_{n-1}>a_n$$
    则对任意满足
    $$a_i-a_{i+1}<a_{i+2}$$
    的三个数可以由数组的降序来保证
    $$a_i+a_{i+1}>a_{i+2}$$
    使得这三个数可以构成有效三角形，
    此时由于数组的降序性有$i$最小时三角形周长$C=a_i+a_{i+1}+a_{i+2}$获得最大值。

    ```cpp
    int largestPerimeter(vector<int>& A) {
	    int ans=0;
        if(A.size()>=3){
			sort(A.begin(),A.end(),[](const int a,const int b)->bool{return a>b;});
			for(int i = 2; i < A.size(); i++)
			{
				if(A[i-2]-A[i-1]<A[i]){
					ans=A[i]+A[i-1]+A[i-2];
					break;
				}
			}
		}
		return ans;
    }
    ```

    但是对于[812](https://leetcode.com/problems/largest-triangle-area/)这样根据给定的一群点来求面积最大三角形的优化问题则只能是用三重循环暴力搜索，即使可以使用凸包(convex hull)优化，效果不明显，最坏情况下仍然是$O(n^3)$复杂度。另外一个问题，根据给定的点$A,B,C$求三角形面积的公式有：
    - 海伦公式

        令$p=\frac{a+b+c}{2}$,则$S=\sqrt{p(p-a)(p-b)(p-c)}$,其中$a,b,c$为三条边边长
    - [向量外积](https://en.wikipedia.org/wiki/Cross_product)

        $$
        \begin{array}{c}{\text { Area }=\frac{1}{2}|\vec{AB} \times \vec{AC}|} \\
        {\text {Area}=\frac{1}{2} |\left(x_{b}-x_{a}, y_{b}-y_{a}\right) ) \times\left(x_{c}-x_{a}, y_{c}-y_{a}\right) ) |} \\
        {\text {Area}=\frac{1}{2} |\left(x_{b}-x_{a}\right)\left(y_{c}-y_{a}\right)-\left(x_{c}-x_{a}, y_{b}-y_{a}\right) ) |} \\
        {\text {Area}=\frac{1}{2}\left|x_{a} y_{b}+x_{b} y_{c}+x_{c} y_{a}-x_{a} y_{c}-x_{c} y_{b}-x_{b} y_{a}\right|}\end{array}$$

- [979. Distribute Coins in Binary Tree](https://leetcode.com/problems/distribute-coins-in-binary-tree/)

    DFS遍历每个节点，时间复杂度$O(n)$，其中n为给定树root中的节点数

    ```cpp
    int dfs_distributeCoins(TreeNode *root, int &ret)
    {
        int ans = 0;
        if (root)
        {
            int left = dfs_distributeCoins(root->left, ret);
            int right = dfs_distributeCoins(root->right, ret);
            root->val += left + right;
            ans = root->val - 1;
            ret += abs(ans);
        }
        return ans;
    }
    int distributeCoins(TreeNode *root)
    {
        int ret = 0;
        dfs_distributeCoins(root, ret);
        return ret;
    }
    ```

- [980](https://leetcode.com/problems/unique-paths-iii/)

    [62](https://leetcode.com/problems/unique-paths/)和[63](https://leetcode.com/problems/unique-paths-ii/)的进阶版，除了指定起点和终点之外，增加了不可通过的障碍点，求从起点到终点并经过所有无障碍点的路径数量，使用DFS进行递归遍历，在此过程中维护一个已经访问过的点的记录矩阵visited、可经过点的数量count_0，从而保证不循环访问已经访问过的点，同时当可经过的点count_0为0且到达指定终点时，发现一条符合条件的路径，路径计数器ans自增加一

    ```cpp
    void dfs_helper(vector<vector<int>> &grid, vector<vector<bool>> &visited, int i, int j, int *ans, int count_0)
    {
        if (!(i < 0 || j < 0 || i >= grid.size() || j >= grid[0].size() || grid[i][j] == -1 || visited[i][j] == true))
        {
            // 超出边界或者障碍(-1)或者已经访问过的节点，nothing to be done
            if (grid[i][j] == 0 || grid[i][j] == 1)
            {
                visited[i][j] = true;
                dfs_helper(grid, visited, i - 1, j, ans, count_0 - 1);
                dfs_helper(grid, visited, i + 1, j, ans, count_0 - 1);
                dfs_helper(grid, visited, i, j - 1, ans, count_0 - 1);
                dfs_helper(grid, visited, i, j + 1, ans, count_0 - 1);
                visited[i][j] = false;
            }
            else if (grid[i][j] == 2)
            {
                if (count_0 == 0)
                {
                    (*ans)++;
                }
            }
        }
    }
    int uniquePathsIII(vector<vector<int>> &grid)
    {
        const int m = grid.size(), n = grid[0].size();
        int ans = 0, count_0 = 0;
        int row, col;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (grid[i][j] == 1)
                {
                    // start point
                    row = i, col = j;
                    count_0++;
                }
                else if (grid[i][j] == 0)
                {
                    count_0++;
                }
            }
        }
        vector<vector<bool>> visited(m, vector<bool>(n, false));
        dfs_helper(grid, visited, row, col, &ans, count_0);
        return ans;
    }
    ```

- [981. Time Based Key-Value Store](https://leetcode.com/problems/time-based-key-value-store/)

	- 假定给定的set序列中timestamp是self-increasing的，则用hashmap来存储一个key2value-timestamp对，然后get的时候寻找给定timestamp在hashmap[key]中的上界即可，时间复杂度$O(mlog(n))$，其中m是get操作的数量，n是set操作的数量，即每次set是$O(1)$的，每次get是$O(log(n))$

	```cpp
	class TimeMap 
	{
	private:
		unordered_map<string, vector<pair<int, string>>> key2value;

	public:
		/** Initialize your data structure here. */
		TimeMap()
		{
			key2value.clear();
		}

		void set(string key, string value, int timestamp)
		{
			key2value[key].push_back(make_pair(timestamp, value));
		}

		string get(string key, int timestamp)
		{
			/**
			* lower_bound 从begin到end-1寻找第一个大于等于val的地址，没有则返回end
			* upper_bound 从begin到end-1寻找第一个大于val的地址，没有则返回end
			*/
			auto it = upper_bound(key2value[key].begin(), key2value[key].end(), pair<int, string>(timestamp, ""), [](const auto &a, const auto &b) -> bool { return a.first < b.first; });
			return key2value[key].begin() == it ? "" : prev(it)->second;
		}
	};
	```

    - 假定给定的set序列中timestamp不能保证self-increasing，则使用ordered_map来保存可以的hash值

    ```cpp
	class TimeMap
	{
	private:
		unordered_map<string, map<int, string>> key2value;

	public:
		/** Initialize your data structure here. */
		TimeMap()
		{
			key2value.clear();
		}

		void set(string key, string value, int timestamp)
		{
			key2value[key].insert({timestamp, value});
		}

		string get(string key, int timestamp)
		{
			auto it = key2value[key].upper_bound(timestamp);
			return (it == key2value[key].begin()) ? "" : prev(it)->second;
		}
	};
    ```

- [983. Minimum Cost For Tickets](https://leetcode.com/problems/minimum-cost-for-tickets/)

    dynamic plan，时序性的动态规划，注意状态表示与状态转移方程，时间复杂度$O(n)$

	```cpp
	int mincostTickets(vector<int> &days, vector<int> &costs)
	{
		int n = days.size(), ret = 0;
		if (n > 0)
		{
			unordered_set<int> days_set(days.begin(), days.end());
			vector<vector<int>> dp(366, vector<int>(3, 0));
			for (auto day = 1; day <= 365; day++)
			{
				if (days_set.find(day) != days_set.end())
				{
					dp[day][0] = *min_element(dp[day - 1].begin(), dp[day - 1].end()) + costs[0];
					int a = day >= 7 ? day - 7 : 0, b = day >= 30 ? day - 30 : 0;
					dp[day][1] = *min_element(dp[a].begin(), dp[a].end()) + costs[1];
					dp[day][2] = *min_element(dp[b].begin(), dp[b].end()) + costs[2];
				}
				else
				{
					dp[day] = dp[day - 1];
				}
			}
			ret = *min_element(dp.back().begin(), dp.back().end());
		}
		return ret;
	}
	```

- [984. String Without AAA or BBB](https://leetcode.com/problems/string-without-aaa-or-bbb/)

    贪心原则，时间复杂度$O(A+B)$

    ```cpp
    string strWithout3a3b(int A, int B)
    {
        char a = 'a', b = 'b';
        if (A < B)
        {
            swap(A, B), swap(a, b);
        }
        string ret(A + B, ' ');
        int i = 0;
        while (A > 0 && B > 0 && A > B)
        {
            ret[i++] = a, ret[i++] = a, A -= 2;
            ret[i++] = b, B--;
        }
        while (A > 0 && B > 0)
        {
            A--, B--;
            ret[i++] = a, ret[i++] = b;
        }
        while (A > 0)
        {
            ret[i++] = a, A--;
        }
        while (B > 0)
        {
            ret[i++] = b, B--;
        }
        return ret;
    }
    ```

- [985](https://leetcode.com/problems/sum-of-even-numbers-after-queries/)
    注意每次query对应下标的数字在query前后的奇偶性，分别有不同的操作。time complexity O(n+q)，其中n(size of array) and q(the number of queries)。
