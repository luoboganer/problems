# record about problems in [leetcode](https://leetcode.com/)

## [algorithms](https://leetcode.com/problemset/algorithms/)
  
- [53](https://leetcode.com/problems/maximum-subarray/)
  
  一维dp(dynamic plan)

- [69](https://leetcode.com/problems/sqrtx/)
  
  牛顿迭代法

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
- [877](https://leetcode.com/problems/stone-game/submissions/)
    - 动态规划:假定dp数组表示第一个人的得分，则
      - 第i次轮到第一个人拿的时候，他一定拿使自己得分最高的那个数
        $$dp[i+1][j+1]=max(piles[i]+dp[i+2][j+1],piles[j]+dp[i+1][j]$$
      - 第i次轮到第二个人拿的时候，他一定拿使对方得分最低的那个数
        $$dp[i+1][j+1]=min(-piles[i]+dp[i+2][j+1],-piles[j]+dp[i+1][j]$$
    - 数学方法可以证明第一个开始游戏的人一定取胜

- [905](https://leetcode.com/problems/sort-array-by-parity/)
    cpp的两个标准库函数，用于将vector按照一定的条件划分，例如将一个int类型数组按照奇数偶数划分
    - [partition()](https://en.cppreference.com/w/cpp/algorithm/partition)
    - [stable_partition()](https://en.cppreference.com/w/cpp/algorithm/stable_partition)
    
    eg:
    ```cpp
      std::vector<int> v = {0,1,2,3,4,5,6,7,8,9};
      auto it = std::partition(v.begin(), v.end(), [](int i){return i % 2 == 0;});
    ```