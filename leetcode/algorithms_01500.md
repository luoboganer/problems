# 1401-1500

- [1401. Circle and Rectangle Overlapping](https://leetcode.com/problems/circle-and-rectangle-overlapping/)

    - 这是一道几何计算题目，时间复杂度$O(1)$，参考[reference](https://www.zhihu.com/question/24251545/answer/27184960)

    ```cpp
    bool checkOverlap(int radius, int x_center, int y_center, int x1, int y1, int x2, int y2)
    {
        // 转换坐标原点为矩形正中心O，此时矩形在坐标系正中心,矩形右上角坐标点必然在第一象限记为A，可以将圆中心坐标映射到对称的第一象限B
        double x_center_rectangle = x1 + (x2 - x1) * 0.5, y_center_rectangle = y1 + (y2 - y1) * 0.5;
        double x_center_new = abs(x_center - x_center_rectangle), y_center_new = abs(y_center - y_center_rectangle);
        double x2_new = x2 - x_center_rectangle, y2_new = y2 - y_center_rectangle;
        // 计算向量 AB
        double x = x_center_new - x2_new, y = y_center_new - y2_new;
        /**
         *  计算矩形右上角A点到圆心B点的距离，分四种情况讨论
         * 1. x>0 y>0 该距离小于等于圆半径，即为有交点（相切或者相交)
         * 2. x<0 y>0 若x <= radius则相交或者相切
         * 3. x>0 y<0 若y <= radius则相交或者相切
         * 4. x<=0 y<=0 必然相交
        */
        double square_of_distance_AB = x * x + y * y, square_of_radius = radius * radius;
        return (x <= 0 && y <= 0) || (x > 0 && y < 0 && x <= radius) || (x < 0 && y > 0 && y <= radius) || square_of_distance_AB <= square_of_radius;
    }
    ```

- [1406. Stone Game III](https://leetcode.com/problems/stone-game-iii/)

	DP，时间复杂度$O(n)$

	```cpp
	string stoneGameIII(vector<int> &stoneValue)
	{
		/**
		 * dynamic plan, dp[i]表示从在给定stoneValue[i,i+1,...,n-1]先手可以获得的最大值
		 * dp[i] = max(suffix_sum[i] - dp[i+x]), 1 <= x <= 3
		*/
		const int n = stoneValue.size();
		vector<int> suffix_sum(n + 1, 0), dp(n + 1, numeric_limits<int>::min());
		dp[n] = 0; // initialization
		for (int i = n - 1; i >= 0; i--)
		{
			suffix_sum[i] = stoneValue[i] + suffix_sum[i + 1];
		}
		for (int i = n; i >= 0; i--)
		{
			for (auto step = 1; step <= 3 && i + step <= n; step++)
			{
				dp[i] = max(dp[i], suffix_sum[i] - dp[i + step]);
			}
		}
		int alice = dp[0], bob = suffix_sum[0] - dp[0];
		string ret;
		if (alice > bob)
		{
			ret = "Alice";
		}
		else if (alice < bob)
		{
			ret = "Bob";
		}
		else
		{
			ret = "Tie";
		}
		return ret;
	}
	```

- [1410. HTML Entity Parser](https://leetcode.com/problems/html-entity-parser/)

    - 暴力比较是否字符串相同，LeetCode评测机$\color{red}{TLE}$

    ```cpp
    class Solution
    {
    private:
        int get_index(string text, int start_pointer)
        {
            // start_pointer is the index of the next char of '&'
            int index = 6; // not found
            bool not_found = true;
            int count = text.length(), partial_length = count - start_pointer;
            if (not_found && partial_length >= 6 && text.substr(start_pointer, 6).compare("frasl;") == 0)
            {
                index = 5, not_found = false;
            }
            if (not_found && partial_length >= 5 && text.substr(start_pointer, 5).compare("quot;") == 0)
            {
                index = 0, not_found = false;
            }
            if (not_found && partial_length >= 5 && text.substr(start_pointer, 5).compare("apos;") == 0)
            {
                index = 1, not_found = false;
            }
            if (not_found && partial_length >= 4 && text.substr(start_pointer, 4).compare("amp;") == 0)
            {
                index = 2, not_found = false;
            }
            if (not_found && partial_length >= 3 && text.substr(start_pointer, 3).compare("gt;") == 0)
            {
                index = 3, not_found = false;
            }
            if (not_found && partial_length >= 3 && text.substr(start_pointer, 3).compare("lt;") == 0)
            {
                index = 4, not_found = false;
            }
            return index;
        }

    public:
        string entityParser(string text)
        {
            vector<char> entities{'"', '\'', '&', '>', '<', '/'};
            vector<int> lengths{5, 5, 4, 3, 3, 6};
            string ret;
            const int count = text.length();
            for (int i = 0; i < count; i++)
            {
                char ch = text[i];
                if (ch == '&')
                {
                    int index = get_index(text, i + 1);
                    if (index < 6)
                    {
                        ch = entities[index];
                        i += lengths[index];
                    }
                }
                ret.push_back(ch);
            }
            return ret;
        }
    };
    ```

    - two pointer method，时间效率$\color{red}{160 ms, 99.67\%}$

    ```cpp
    string entityParser(string text)
    {
        // 用字符串的长度记录特殊字符串对应的转义字符
        /**
         * vector<pair<string, char>> sp2char_map[6] = ((), (), (("gt", \'>\'), ("lt", \'<\')), (("amp", \'&\')), (("quot", \'"\'), ("apos", \'\'\')), (("frasl", \'/\')));
         * 为通过gitbook编译，使用 () 代替了 {}，实际运行代码请替换回
        */
        // 用start和and记录每一对&和;的位置
        int start = 0, current = 0, count = text.length();
        for (auto i = 0; i < count; i++, current++)
        {
            text[current] = text[i]; // update current character
            if (text[current] == '&')
            {
                start = current;
            }
            if (text[current] == ';')
            {
                auto size = current - start - 1;
                if (size >= 2 && size <= 5)
                {
                    //所有可能的特殊字符序列长度为2,3,4,5
                    for (auto &[enc, dec] : sp2char_map[size])
                    {
                        if (text.compare(start + 1, size, enc) == 0)
                        {
                            current = start;
                            text[current] = dec;
                            break;
                        }
                    }
                }
                start = current + 1;
            }
        }
        text.resize(current);
        return text;
    }
    ```

- [1414. Find the Minimum Number of Fibonacci Numbers Whose Sum Is K](https://leetcode.com/problems/find-the-minimum-number-of-fibonacci-numbers-whose-sum-is-k/)

    - 动态规划，DP，$O(k^2)$，结果正确但是$\color{red}{TLE}$

    ```cpp
    int findMinFibonacciNumbers(int k)
    {
        vector<int> fibonacci{1, 2};
        int last = 1;
        while (fibonacci[last] < k)
        {
            fibonacci.push_back(fibonacci[last - 1] + fibonacci[last]);
            last++;
        }
        vector<int> dp(k + 1, 0);
        for (auto i = 0; i <= k; i++)
        {
            dp[i] = i; // 全部用1
        }
        for (auto &&v : fibonacci)
        {
            for (auto i = v; i <= k; i++)
            {
                dp[i] = min(dp[i - v] + 1, dp[i]);
            }
        }
        return dp.back();
    }
    ```

    - 贪心法则，greedy，$f(k)=f(k-x)+1$，递归式写法，时间复杂度$O((log(k))^2)$

    ```cpp
    int findMinFibonacciNumbers(int k)
    {
        int ret;
        if (k < 2)
        {
            ret = k;
        }
        else
        {
            int a = 1, b = 1;
            while (b <= k)
            {
                a += b;
                swap(a, b);
            }
            ret = 1 + findMinFibonacciNumbers(k - a);
        }
        return ret;
    }
    ```

    - 贪心法则，greedy，$f(k)=f(k-x)+1$，迭代式写法，时间复杂度$O(log(k))$

    ```cpp
    int findMinFibonacciNumbers(int k)
    {
        int ret = 0, a = 1, b = 1, temp;
        while (b <= k)
        {
            temp = a + b;
            a = b;
            b = temp;
        }
        while (a > 0)
        {
            if (a <= k)
            {
                k -= a;
                ret++;
            }
            temp = b - a;
            b = a;
            a = temp;
        }
        return ret;
    }
    ```

- [1415. The k-th Lexicographical String of All Happy Strings of Length n](https://leetcode.com/problems/the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/)

    - 用组合数学的方法直接计算第k个happy string，时间复杂度$O(n)$

    ```cpp
    string getHappyString(int n, int k)
    {
        vector<int> powersOf2{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
        string letters_all = "abc", letters_next = "bcacab", ret;
        if (k <= 3 * powersOf2[n - 1])
        {
            char fisrt_letter = letters_all[(--k) / powersOf2[n - 1]];
            ret.push_back(fisrt_letter), k %= powersOf2[n - 1];
            for (auto i = 1; i < n; i++)
            {
                int next_index = k < powersOf2[n - 1 - i] ? 0 : 1;
                ret.push_back(letters_next[(int)(ret.back() - 'a') * 2 + next_index]);
                k %= powersOf2[n - 1 - i];
            }
        }
        return ret;
    }
    ```

- [1424. Diagonal Traverse II](https://leetcode.com/problems/diagonal-traverse-ii/)

    - 从左下角到右上角遍历所有可能的坐标，时间复杂度$O(rows*cols)$，提交结果答案正确但是$\color{red}{TLE}$

    ```cpp
    vector<int> findDiagonalOrder(vector<vector<int>> &nums)
    {
        int rows = nums.size(), total = 0, k = 0, max_col = 0;
        vector<int> cols(rows, 0), prefix_max_cols(rows, 0);
        for (auto i = 0; i < rows; i++)
        {
            cols[i] = nums[i].size();
            max_col = max(max_col, cols[i]), total += cols[i];
            prefix_max_cols[i] = max_col;
        }
        vector<int> ret(total, 0); // 保存结果
        int max_i_j_sum = rows + max_col - 2;
        for (auto i_j_sum = 0; i_j_sum <= max_i_j_sum && k < total; i_j_sum++)
        {
            /*
                1. k<total 元素总数目约数剪枝
            */
            int i = min(i_j_sum, rows - 1), j = i_j_sum - i;
            while (i >= 0 && j < prefix_max_cols[i] && k < total)
            {
                if (j < cols[i])
                {
                    ret[k++] = nums[i][j];
                }
                i--, j++; // 从左下角向右上角走
            }
        }
        return ret;
    }
    ```

    - 采用桶排序的方式将下标$i+j$相同的对角线元素全部放入同一个桶里面，时间复杂度$O(n)$，其中n为总的元素个数

    ```cpp
    vector<int> findDiagonalOrder(vector<vector<int>> &nums)
    {
        int rows = nums.size(), max_key = 0;
        unordered_map<int, vector<int>> buckets;
        vector<int> ret;
        for (auto i = 0; i < rows; i++)
        {
            auto cols = nums[i].size();
            for (auto j = 0; j < cols; j++)
            {
                max_key = max(max_key, i + j);
                buckets[i + j].push_back(nums[i][j]);
            }
        }
        for (auto i = 0; i <= max_key; i++)
        {
            for (auto v = buckets[i].rbegin(); v != buckets[i].rend(); v++)
            {
                ret.push_back(*v);
            }
        }
        return ret;
    }
    ```

- [1449. Form Largest Integer With Digits That Add up to Target](https://leetcode.com/problems/form-largest-integer-with-digits-that-add-up-to-target/)

    1-9的每个数字需要消耗的weight是cost数组，在相同的cost下优先选择值更大的数字v，然后使用DP方法动态规划背包容量为target时的最大结果，这里注意不超过背包容量的最优解和恰好装满背包的最优解（需要从target=0开始使用，所有值初始化为inf，最优解为max时inf为max，最优解为min时inf为min，然后dp[0]=0），时间复杂度$O(target)$

	```cpp
    class Solution
    {
    private:
        string max_string(string a, string b, char inf_ch)
        {
            // a是新构造的 dp[j-weight]+ch  b是原本的dp[j]
            string ret;
            int length_a = a.length(), length_b = b.length();
            if (a.front() == inf_ch)
            {
                ret = b;
            }
            else if (length_a > length_b)
            {
                ret = a;
            }
            else if (length_a < length_b)
            {
                ret = b;
            }
            else
            {
                ret = a.compare(b) >= 0 ? a : b;
            }
            return ret;
        }

    public:
        string largestNumber(vector<int> &cost, int target)
        {
            unordered_map<int, int> cost2digital;
            for (auto i = 0; i < cost.size(); i++)
            {
                auto it = cost2digital.find(cost[i]);
                if (it == cost2digital.end())
                {
                    cost2digital[cost[i]] = i + 1;
                }
                else
                {
                    cost2digital[cost[i]] = max(i + 1, cost2digital[cost[i]]);
                }
            }
            vector<int> digital2cost(10, -1);
            for (auto &&item : cost2digital)
            {
                digital2cost[item.second] = item.first;
            }
            // 设置inf_string 哨兵是为了背包恰好完全装满
            char inf_ch = '0';
            string inf_string{inf_ch};
            vector<string> dp(target + 1, inf_string);
            dp[0] = "";
            for (int i = 9; i >= 0; --i)
            {
                if (digital2cost[i] != -1)
                {
                    char ch = (char)(i + '0');
                    int weight = digital2cost[i];
                    for (int j = weight; j <= target; j++)
                    {
                        string temp = dp[j - weight];
                        temp.push_back(ch);
                        dp[j] = max_string(temp, dp[j], inf_ch);
                    }
                }
            }
            string ret = dp.back();
            if (ret.size() == 0)
            {
                ret.push_back('0');
            }
            return ret;
        }
    };
	```

- [1452. People Whose List of Favorite Companies Is Not a Subset of Another List](https://leetcode.com/problems/people-whose-list-of-favorite-companies-is-not-a-subset-of-another-list/)
    
    早cpp中判断set a是否为set b的子集，将a和b排序后处理，时间复杂度$O(kn^2)$，其中$n=favoriteCompanies.length,k=max_i(favoriteCompanies[i])$

	```cpp
    class Solution
    {
    private:
        int cmp(vector<string> &a, vector<string> &b)
        {
            int m = a.size(), n = b.size();
            int i = 0, j = 0;
            if (m > n)
            {
                return 1;
            }
            else if (m == n && m == 0)
            {
                return 0;
            }
            else
            {
                // m == n 且不为0 或者m<n
                int i = 0, j = 0;
                while (i < m && j < n)
                {
                    if (a[i].compare(b[j]) == 0)
                    {
                        i++, j++;
                    }
                    else
                    {
                        j++;
                    }
                }
                int ret;
                if (i == m)
                {
                    ret = (j == n) ? 0 : -1;
                }
                else
                {
                    ret = 1;
                }
                return ret;
            }
            return -2; // unknown
        }

    public:
        vector<int> peopleIndexes(vector<vector<string>> &favoriteCompanies)
        {
            for (auto &&item : favoriteCompanies)
            {
                sort(item.begin(), item.end());
            }
            const int count = favoriteCompanies.size();
            vector<vector<int>> records(count, vector<int>(count, 0));
            // -2 未知 -1 i<j 0 i=j 1 i>j
            for (auto i = 0; i < count; i++)
            {
                records[i][i] = 0;
                for (auto j = 0; j < count; j++)
                {
                    if (i != j)
                    {
                        records[i][j] = cmp(favoriteCompanies[i], favoriteCompanies[j]);
                    }
                }
            }
            vector<int> ret;
            for (auto i = 0; i < count; i++)
            {
                auto j = 0;
                while (j < count)
                {
                    (i == j || records[i][j] == 1) ? j++ : j = count + 1; // j=count+1 means break
                }
                if (j == count)
                {
                    ret.push_back(i);
                }
            }
            return ret;
        }
    };
	```

- [1453. Maximum Number of Darts Inside of a Circular Dartboard](https://leetcode.com/problems/maximum-number-of-darts-inside-of-a-circular-dartboard/)
    
    任意两个点和半径r可以确定一个圆，然后遍历其它点在该圆内的数量，时间复杂度$O(n^3)$

	```cpp
    class Solution
    {
    private:
        double eps = 1e-8;
        double distance_square(vector<int> &a, vector<int> &b)
        {
            return (double)((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]));
        }
        double distance_square(vector<int> &a, vector<double> &b)
        {
            return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
        }

    public:
        int numPoints(vector<vector<int>> &points, int r)
        {
            // n^3做法，每两个点确定一个圆，统计在该圆中点的数量
            int count = points.size();
            int ret = 1, d_square = 4 * r * r, r_square = r * r;
            if (count > 1)
            {
                for (auto i = 0; i < count; i++)
                {
                    for (auto j = i + 1; j < count; j++)
                    {
                        // 遍历每两个点构成的所有的圆
                        double center_r_square = distance_square(points[i], points[j]);
                        if (center_r_square <= d_square)
                        {
                            // 给定圆的半径r可以覆盖这两个点，判断其它点在这两个点构成的圆内有多少个
                            int number_points = 0;
                            double mid_ij_x = (points[i][0] + points[j][0]) / 2.0;
                            double mid_ij_y = (points[i][1] + points[j][1]) / 2.0;
                            vector<double> mid{mid_ij_x, mid_ij_y};
                            double angle = atan2((double)(points[i][0] - points[j][0]), (double)(points[j][1] - points[i][1]));
                            double d = sqrt(r * r - distance_square(points[i], mid));
                            double center_x = mid_ij_x + d * cos(angle);
                            double center_y = mid_ij_y + d * sin(angle);
                            vector<double> center{center_x, center_y};
                            for (auto &&item : points)
                            {
                                if (distance_square(item, center) <= r_square + eps)
                                {
                                    number_points++;
                                }
                            }
                            ret = max(ret, number_points);
                        }
                    }
                }
            }
            else
            {
                ret = max(count, 1);
            }
            return ret;
        }
    };
	```

- [1458. Max Dot Product of Two Subsequences](https://leetcode.com/problems/max-dot-product-of-two-subsequences/)

    最长公共子序列问题（LCS），DP，时间复杂度$O(n^2)$

	```cpp
	int maxDotProduct(vector<int> &nums1, vector<int> &nums2)
	{
		int m = nums1.size(), n = nums2.size();
		vector<vector<int>> dp(m + 1, vector<int>(n + 1, numeric_limits<int>::min()));
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				vector<int> auxiliary = {max(0, dp[i][j]) + nums1[i] * nums2[j], dp[i + 1][j], dp[i][j + 1]};
				dp[i + 1][j + 1] = *max_element(auxiliary.begin(), auxiliary.end());
			}
		}
		return dp.back().back();
	}
	```

- [1462. Course Schedule IV](https://leetcode.com/problems/course-schedule-iv/)

    **相关题目**

    - [207. Course Schedule](https://leetcode.com/problems/course-schedule/)
    - [210. Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)
    - [630. Course Schedule III](https://leetcode.com/problems/course-schedule-iii/)

    所有课程的前后依赖关系构成了directed graph/tree，DFS检查每一个查询对a/b在这个有向图中b是否是a的子节点，在check(a,b)的过程中memorization中间结果达到缩短后续查询对距离的剪枝效果，时间复杂度$O(V*E)$

	```cpp
    class Solution
    {
    private:
        int dfs_helper(vector<vector<int>> &graph, int a, int b, int n)
        {
            if (graph[a][b] == -1)
            {
                // unknown, need to check
                if (graph[b][a] == 1)
                {
                    // b是a的前继，a一定不是b的前继
                    graph[a][b] = 0;
                }
                else
                {
                    // graph[b][a] == -1 ，b是否为a的前继未知，a是否为b的前继也未知
                    // graph[b][a] == 0 ，b不是a的前继，a是否为b的前继不一定
                    for (auto i = 0; i < n; i++)
                    {
                        if (graph[a][i] == 1 && dfs_helper(graph, i, b, n) == 1)
                        {
                            graph[a][b] = 1;
                            break;
                        }
                    }
                    if (graph[a][b] != 1)
                    {
                        graph[a][b] = 0;
                    }
                }
            }
            return graph[a][b]; // 此时返回值只有0/1两种状态，不再是未知-1
        }

    public:
        vector<bool> checkIfPrerequisite(int n, vector<vector<int>> &prerequisites, vector<vector<int>> &queries)
        {
            // 建立邻接矩阵
            vector<vector<int>> graph(n, vector<int>(n, -1));
            for (auto &&edge : prerequisites)
            {
                graph[edge[0]][edge[1]] = 1;
                /*
                    1. graph[a][b] = 1 for one directed edge from a to b
                    2. graph[a][b] = 0 for no one directed edge from a to b
                    3. graph[a][b] = -1 unknown
                */
            }
            vector<bool> ret;
            for (auto &&q : queries)
            {
                ret.push_back(dfs_helper(graph, q[0], q[1], n) == 1);
            }
            return ret;
        }
    };
	```