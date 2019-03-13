# record about problems in [leetcode](https://leetcode.com/)

## [algorithms](https://leetcode.com/problemset/algorithms/)


- [1](https://leetcode.com/problems/two-sum/)
  
  题目要求在一个数组中寻找两个和为给定值的数，暴力枚举时两层遍历的时间复杂度为$O(n^2)$，因此使用**unorder_map**的接近$O(1)$的查询效率来实现$O(n)$一遍扫描的做法，这里注意cpp中STL template **unorder_map** 的用法

- [53](https://leetcode.com/problems/maximum-subarray/)
  
  一维dp(dynamic plan)

- [62](https://leetcode.com/problems/unique-paths/)

    一维dp(dynamic plan)
    $$dp[m,n]=dp[m-1,n]+dp[m,n-1],\left\{\begin{matrix} dp[0,0]=0\\  dp[0,1]=1\\ dp[1,0]=1 \end{matrix}\right.$$

- [63](https://leetcode.com/problems/unique-paths-ii/submissions/)

    在[62-unique-path](https://leetcode.com/problems/unique-paths/)的基础上增加了障碍点，因此需要考虑初始化调价，即第一行、第一列有障碍的问题，同时咱有障碍的点路径数为0。

    另外需要注意由于障碍点的0路径导致最终结果在int表示范围内，但是计算过程中可能会出现超出int表示范围的数字，需要用long long来表示并取模(mod INT_MAX)。

- [69](https://leetcode.com/problems/sqrtx/)
  
  牛顿迭代法

- [167](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)
  
  在给定的有序数组中寻找两个数使其和为给定的target，O(n)时间复杂度
  ```cpp
    vector<int> twoSum(vector<int> &numbers, int target)
	{
		int left = 0, right = numbers.size() - 1;
		while (left < right)
		{
			if (numbers.at(left) + numbers.at(right) == target)
			{
				break;
			}
			else if (numbers.at(left) + numbers.at(right) > target)
			{
				right--;
			}
			else if (numbers.at(left) + numbers.at(right) < target)
			{
				left++;
			}
		}
		return vector<int>{left + 1, right + 1};
	}
  ```
- [217](https://leetcode.com/problems/contains-duplicate/submissions/)

    在给定数组中查找是否有重复值，典型的集合的应用

    - 先排序然后一遍扫描，O(nlog(n))
    - hash set，近似O(n)
    ```cpp
        bool containsDuplicate(vector<int> &nums)
        {
            // method 1, sort and check, time complexity O(nlog(n))
            // sort(nums.begin(), nums.end());
            // int count = nums.size() - 1;
            // bool res = false;
            // for (int i = 0; i < count; i++)
            // {
            // 	if (nums.at(i) == nums.at(i + 1))
            // 	{
            // 		res = true;
            // 		break;
            // 	}
            // }
            // return res;

            // method 2, hashset, approximate O(n)
            unordered_set<int> nums_set;
            int count = nums.size();
            bool res = false;
            for (int i = 0; i < count; i++)
            {
                if (nums_set.find(nums.at(i)) != nums_set.end())
                {
                    res = true;
                    break;
                }
                nums_set.insert(nums.at(i));
            }
            return res;
        }
    ```

- [258](https://leetcode.com/problems/add-digits/)
  
  对一个数字求各个数位的和，递归直到这个数是个位数
  - 循环或者递归的方法
    ```cpp
        int addDigitals(int n)
        {
            while (n > 9)
            {
                int base = n;
                n = 0;
                while (base)
                {
                    n += base % 10;
                    base /= 10;
                }
            }
            return n;
        }
    ```
  - O(1)方法，可以[数学证明](https://leetcode.com/problems/add-digits/discuss/241551/Explanation-of-O(1)-Solution-(modular-arithmetic))
    ```cpp
        return (n - 1) % 9 + 1
    ```
- [268](https://leetcode.com/problems/missing-number/)
  - Gauss formual
    
    从0到n的和是$S_0=\frac{n(n+1)}{2}$，missing一个数x以后的和是$S=\frac{n(n+1)}{2}-x$，则丢失的数是$x=S_0-S$。
  - bit XOR

    下标是0到n-1，补充一个n作为初始值，然后这些数字是0-n且missing一个数，相当于从0-n除了missing的这个数只出现一次之外其他数字都出现了两次，因此可以用XOR操作找到这个只出现了一次的数即可。

- [283](https://leetcode.com/problems/move-zeroes/)
  
  将一个数列中的0元素全部移动到数列尾部，O(1)空间复杂度和O(n)时间复杂度，保持原数列的稳定性
    ```cpp
        void moveZeroes(vector<int> &nums)
        {
            // int count = nums.size(), i = 0, k = 0;
            // for (int i = 0; i < count; i++)
            // {
            // 	if (nums[i])
            // 	{
            // 		nums[k++] = nums[i];
            // 	}
            // }
            // for (; k < count; k++)
            // {
            // 	nums[k++] = 0;
            // }

            // optimization
            int count = nums.size(), k = 0, i = 0;
            for (; i < count; i++)
            {
                if (nums[i])
                {
                    swap(nums[i], nums[k++]);
                }
            }
        }
    ```

- [338](https://leetcode.com/problems/counting-bits/)

    注意分组统计的办法，类似和二进制表示中1的数量有关的问题和2的倍数有关系

- [442](https://leetcode.com/problems/find-all-duplicates-in-an-array/)
    
    在一个数组$a_n,1 \le a_i \le n$中找出出现了两次的数字（其他数字只出现了一次），要求不占用额外的内存空间、$O(n)$时间复杂度
    - 逐位负数标记法
    - 交换法

- [509](https://leetcode.com/problems/fibonacci-number/)
    
    斐波那契数列
    - 递归法
    - 非递归循环 faster

- [561](https://leetcode.com/problems/array-partition-i/)
    
    2n个给定范围的数据划分成n组使得每组最小值求和最大，基本思路是对所有数排序后对基数位置上的数求和即可，这里类似于NMS非极大值抑制抑制的思路，主要的时间复杂度在排序上。
    - 基本想法是quick sort，时间复杂度$O(nlog(n))$
    - 本题给出了数据范围，可以bucket sort，时间复杂度$O(n)$，但是需要$O(N)$的额外空间
  
- [733](https://leetcode.com/problems/flood-fill/)
    
    类似于图像处理中区域增长的方式，采用DFS递归写法或者用栈stack实现

- [852](https://leetcode.com/problems/peak-index-in-a-mountain-array/)

    寻找给定数组中的山峰数下标，所谓山峰数根据题目定义即为**全局最大值**，因此binary search是理论上的最佳算法，time complexity O(log(n))

- [876](https://leetcode.com/problems/middle-of-the-linked-list/solution/)
    寻找单链表的中间节点，两种方法
    - O(n)时间遍历一遍将所有节点的地址存储到一个数组，则可以在O(1)时间内索引到，O(n)空间复杂度
    - 用不同步长的两个指针同时遍历，slow每次前进一步，fast每次前进两步，fast到终点时slow即可到中间点，O(n)时间，O(1)空间

- [877](https://leetcode.com/problems/stone-game/submissions/)
    - 动态规划:假定dp数组表示第一个人的得分，则
      - 第i次轮到第一个人拿的时候，他一定拿使自己得分最高的那个数
        $$dp[i+1][j+1]=max(piles[i]+dp[i+2][j+1],piles[j]+dp[i+1][j]$$
      - 第i次轮到第二个人拿的时候，他一定拿使对方得分最低的那个数
        $$dp[i+1][j+1]=min(-piles[i]+dp[i+2][j+1],-piles[j]+dp[i+1][j]$$
    - 数学方法可以证明第一个开始游戏的人一定取胜

- [884](https://leetcode.com/problems/uncommon-words-from-two-sentences/)
    问题描述：Uncommon Words from Two Sentences  
    统计两句话中只出现过一次的单词，主要的点有：
    - 字符串的分割，使用istringstream进行copy，back_inserter的使用
    ```cpp
        vector<string> getWords(const string &s)
        {
            istringstream buffer(s);
            vector<string> ret;
            copy(istream_iterator<string>(buffer), istream_iterator<string>(), back_inserter(ret));
            return ret;
        }
    ```
    - 使用unorder_map<string,int>进行word的词频统计
    ```cpp
        vector<string> uncommonFromSentences(string A, string B)
        {
            vector<string> res;
            unordered_map<string, int> count;
            auto a = getWords(A);
            auto b = getWords(B);
            for (const auto &x : a)
            {
                count[x]++;
            }
            for (const auto &x : b)
            {
                count[x]++;
            }
            for (const auto &[word, freq] : count)
            {
                if (freq == 1)
                {
                    res.push_back(word);
                }
            }
            return res;
        }
    ```
- [896](https://leetcode.com/problems/monotonic-array/)
    判断一个数列是否单调，单调包含单调递增和单调递减，非严格单调还包含相等的情况
    - two pass，第一遍扫描判断是否全部 <= ，第二遍扫描判断是否全部 >=，两次结果取或关系
    - one pass，一遍扫描过程中用${-1,0,1}$分别表示<,=,>三种状态，然后在第二次出现非零元素的情况下，如果和第一次非零元素不同，即可返回false
- [905](https://leetcode.com/problems/sort-array-by-parity/)
    cpp的两个标准库函数，用于将vector按照一定的条件划分，例如将一个int类型数组按照奇数偶数划分
    - [partition()](https://en.cppreference.com/w/cpp/algorithm/partition)
    - [stable_partition()](https://en.cppreference.com/w/cpp/algorithm/stable_partition)
    
    eg:
    ```cpp
      std::vector<int> v = {0,1,2,3,4,5,6,7,8,9};
      auto it = std::partition(v.begin(), v.end(), [](int i){return i % 2 == 0;});
    ```

 - [929](https://leetcode.com/problems/unique-email-addresses/)
    两个考察点
    - cpp STL中的string操作，子串、查找、替换等
    - cpp STL中集合(set)的使用，或者自己实现一个set

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
 - [985](https://leetcode.com/problems/sum-of-even-numbers-after-queries/)
    注意每次query对应下标的数字在query前后的奇偶性，分别有不同的操作。time complexity O(n+q)，其中n(size of array) and q(the number of queries)。