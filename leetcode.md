# record about problems in [leetcode](https://leetcode.com/)

## [algorithms](https://leetcode.com/problemset/algorithms/)
  
- [53](https://leetcode.com/problems/maximum-subarray/)
  
  一维dp(dynamic plan)

- [69](https://leetcode.com/problems/sqrtx/)
  
  牛顿迭代法

- [442](https://leetcode.com/problems/find-all-duplicates-in-an-array/)
    
    在一个数组$a_n,1 \le a_i \le n$中找出出现了两次的数字（其他数字只出现了一次），要求不适用额外的内存空间、$O(n)$时间复杂度
    - 逐位负数标记法
    - 交换法

- [877](https://leetcode.com/problems/stone-game/submissions/)
    - 动态规划:假定dp数组表示第一个人的得分，则
      - 第i次轮到第一个人拿的时候，他一定拿使自己得分最高的那个数
        $$dp[i+1][j+1]=max(piles[i]+dp[i+2][j+1],piles[j]+dp[i+1][j]$$
      - 第i次轮到第二个人拿的时候，他一定拿使对方得分最低的那个数
        $$dp[i+1][j+1]=min(-piles[i]+dp[i+2][j+1],-piles[j]+dp[i+1][j]$$
    - 数学方法可以证明第一个开始游戏的人一定取胜