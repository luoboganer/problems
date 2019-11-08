<!--
 * @Filename:
 * @Author: shifaqiang
 * @Email: 14061115@buaa.edu.cn
 * @Github: https://github.com/luoboganer
 * @Date: 2019-09-13 13:35:19
 * @LastEditors: shifaqiang
 * @LastEditTime: 2019-11-08 10:31:53
 * @Software: Visual Studio Code
 * @Description:
 -->

# record about problems in [leetcode](https://leetcode.com/)

## [algorithms](https://leetcode.com/problemset/algorithms/)

- [1](https://leetcode.com/problems/two-sum/)

    题目要求在一个数组中寻找两个和为给定值的数，暴力枚举时两层遍历的时间复杂度为$O(n^2)$，因此使用**unorder_map**的接近$O(1)$的查询效率来实现$O(n)$一遍扫描的做法，这里注意cpp中STL template **unorder_map** 的用法

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

- [11](https://leetcode.com/problems/container-with-most-water/)

    在一组通过数组给定高度的立柱中选择两根立柱，时期中间的空间可以装水最多，即在具有高度$\{a_0,a_1,...,a_n\}$的这些柱子中选择两根$a_i,a_j$使得$min(a_i,a_j)*(j-i)$最大。

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

    罗马数字转阿拉伯数字，主要是思想是对罗马数字字符序列从右到作扫描，注意IXC的位置和表示的数字有关即可。

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

- [46](https://leetcode.com/problems/permutations/)

    - 注意全排列的实现，递归的和非递归的，字典序的和非字典序的
    - cpp的STL中有*next_permutation*和*prev_permutation*两个函数，注意他们的实现方式

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

- [48](https://leetcode.com/problems/rotate-image/)

    Rotate Image，旋转图片90度，即将一个二维数组原地旋转90度。
    注意每次的坐标变换即可，按照每次旋转一个环（一圈），圈内从左到右一层一层旋转，每次基本的旋转单元只有四个数。

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

- [63](https://leetcode.com/problems/unique-paths-ii/submissions/)

    在[62-unique-path](https://leetcode.com/problems/unique-paths/)的基础上增加了障碍点，因此需要考虑初始化调价，即第一行、第一列有障碍的问题，同时咱有障碍的点路径数为0。

    另外需要注意由于障碍点的0路径导致最终结果在int表示范围内，但是计算过程中可能会出现超出int表示范围的数字，需要用long long来表示并取模(mod INT_MAX)。

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

- [75](https://leetcode.com/problems/sort-colors/)

    即荷兰国旗问题，三色排序，快速排序的基本思想练习

    ```cpp
    void sortColors(vector<int> &nums)
    {
        // method 1, two pass

        // int count[3] = {0, 0, 0};
        // for (auto x : nums)
        // {
        // 	count[x]++;
        // }
        // int k = 0;
        // for (int i = 0; i < 3; i++)
        // {
        // 	int j = 0;
        // 	while (j < count[i])
        // 	{
        // 		j++;
        // 		nums[k++] = count[i];
        // 	}
        // }

        // method 2, one pass, quich sort

        // int left = 0, right = nums.size() - 1, cur = 0;
        // while (cur <= right)
        // {
        // 	if (nums[cur] == 0)
        // 	{
        // 		swap(nums[left++], nums[cur++]);
        // 	}
        // 	else if (nums[cur] == 2)
        // 	{
        // 		swap(nums[right--], nums[cur]);
        // 	}
        // 	else
        // 	{
        // 		cur++;
        // 	}
        // }

        // method 3, one pass , [0,i) - 0   [i,j) - 1    [j,k) - 2

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

- [101](https://leetcode.com/problems/symmetric-tree/)

    分析一棵树对称问题的本质，是：

    - 空树是对称的
    - 没有左右子树的叶节点是对称的
    - 非空的非叶子节点对称的条件有：
        - 左右子树同时存在且左右子树的值相同（对称）
        - 左子树的左子树和右子树的右子树递归对称，左子树的右子树和右子树的左子树递归对称

- [104](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

    计算一颗二叉树的最大深度，这是典型的使用递归解决tree类问题的模板，按照top-down和bottom-up两种思路，参见[article](https://leetcode.com/explore/learn/card/data-structure-tree/17/solve-problems-recursively/534/)。
    - top-down

    ```cpp
    int ans = 0;
    void updateDepth(TreeNode *root, int depth){
        if (root){
            ans = max(ans, depth);
            updateDepth(root->left, depth + 1);
            updateDepth(root->right, depth + 1);
        }
    }
    int maxDepth(TreeNode *root){
        if (root){
            updateDepth(root, 1);
        }
        return ans;
    }
    ```

    - bottom-up

    ```cpp
    int maxDepth(TreeNode* root) {
        if(!root){
            return 0;
        }else{
            int left=maxDepth(root->left);
            int right=maxDepth(root->right);
            return 1+max(left,right);
        }
    }
    ```

- [110](https://leetcode.com/problems/balanced-binary-tree/)

    判断二叉树是否是平衡二叉树（任何节点左右子树的高度差小于等于1），在递归求二叉树最大深度的过程中维护一个全局变量balanced，随时比较任意节点的左右子树高度差即可。

- [116](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)

    更一般化的问题是如题[117](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/)所示的条件，给定一个二叉树，将每一个节点的next指针指向他的同深度的右侧兄弟节点，简单BFS(Breadth-First-Search)，即层序遍历然后将同层的节点扫描一遍将每个节点的next指针指向同层下一个节点即可。

    ```cpp
    Node *connect(Node *root)
    {
        if(root){
            vector<Node *> level{root}, next_level;
            while (!level.empty()){
                for (auto &&node : level){
                    if(node->left){
                        next_level.push_back(node->left);
                    }
                    if(node->right){
                        next_level.push_back(node->right);
                    }
                }
                int i = 0, count = next_level.size() - 1;
                while(i < count){
                    next_level[i]->next = next_level[i + 1];
                    i++;
                }
                level = next_level;
                next_level.clear();
            }
        }
        return root;
    }
    ```

    当将给定二叉树限定为本题所示的Perfect Binary Tree的时候，可以用递归的方式来完成而无需BFS层序遍历的庞大空间开销，需要注意递归到子节点时需要利用父节点的next指针信息。

    ```cpp
    Node *connect(Node *root)
    {
        if(root && root->left){
            root->left->next = root->right;
            if(root->next){
                root->right->next = root->next->left;
            }
            connect(root->left);
            connect(root->right);
        }
        return root;
    }
    ```

- [136](https://leetcode.com/problems/single-number/)

    一个数组中只有一个数落单、其他数均成对出现，采用异或的按位操作一遍扫描找到那个落单的数。
    
    进一步[260](https://leetcode.com/problems/single-number-iii/)中寻找两个落单的数，一遍异或操作可以得到这两个落单数的异或结果，然后其他数按照这个结果二进制表示中第一个非零位是否为1分为两组异或一遍即可。
    
    还有[137](https://leetcode.com/problems/single-number-ii/)中除了一个落单的数字、其他数字均出现了三次，可以统计int的32位表示中每一位为1的个数，然后这个统计结果为不能被3整除的那些bit位为1的二进制表示结果即为所求的落单的那个数。

- [141](https://leetcode.com/problems/linked-list-cycle/)

    检查单向链表中是否存在cycle，两种方法，O(n)时间复杂度
    - hastset 存储节点，不停的检查set中是否存在cur->next
    - 快慢双指针遍历，当快慢指针相同的时候出现cycle

    ```cpp
    bool hasCycle(ListNode *head)
    {
        // method 1, hashset
        // unordered_set<ListNode *> record;
        // bool ret = false;
        // while (head)
        // {
        // 	if (record.find(head) != record.end())
        // 	{
        // 		ret = true;
        // 		break;
        // 	}
        // 	else
        // 	{
        // 		record.insert(head);
        // 	}
        // 	head = head->next;
        // }
        // return ret;

        // method 2, two pointer (slower and faster)
        bool ret = false;
        if (head != NULL && head->next != NULL)
        {
            ret = true;
            ListNode *slow = head, *fast = head->next;
            while (slow != fast)
            {
                if (slow == NULL || fast == NULL)
                {
                    ret = false;
                    break;
                }
                else
                {
                    slow = slow->next;
                    fast = fast->next->next;
                }
            }
        }
        return ret;
    }
    ```

- [142](https://leetcode.com/problems/linked-list-cycle-ii/)

    在 141判断链表中是否有cycle的基础上找出cycle的入口entry pointer，[detail algorithm](https://leetcode.com/problems/linked-list-cycle-ii/discuss/44781/Concise-O(n)-solution-by-using-C%2B%2B-with-Detailed-Alogrithm-Description)。

- [143](https://leetcode.com/problems/reorder-list/)

    链表、树、图等指针操作千千万万要注意空指针甚至是输入根节点为空的情况。

- [162](https://leetcode.com/problems/find-peak-element/)

    寻找peak element即是寻找局部最大值，因此可以二分搜索在$O(log(n))$时间内实现，如果是全局最大值则至少要遍历一次，需要$O(n)$时间来实现

    ```cpp
    int findPeakElement(vector<int>& nums) {
        int left=0,right=nums.size()-1;
        while(left<right){
            int mid=(left+right)>>1;
            if(nums[mid]>nums[mid+1]){
                right=mid;
            }else{
                left=mid+1;
            }
        }
        return left;
    }
    ```

    注意peak element只需要比自己左右两侧的值大就可以了，没必要比最左端和最右端的值大

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

- [169](https://leetcode.com/problems/majority-element/)

    找出给定数组中出现次数超过一半的数，后续还有升级版[229](https://leetcode.com/problems/majority-element-ii/)寻找给定数组中出现次数超过$\frac{1}{3}$的数。

    - 首先遍历数组用hashmap统计每个数出现的次数，然后遍历hashmap来比较是否有出现次数超过一半，假定hashmap的$O(1)$存取效率，可以实现$O(n)$的限行时间复杂度
    - 首先对数组排序，从而使得相同数字全部相邻，然后从左到右遍历是否有出现次数超过一半的数字，时间复杂度来自排序算法，一般为$O(nlog(n))$
    - MOOER voting 算法，可以实现真$O(n)$的线性时间和$O(1)$的常量空间，[here](https://www.hijerry.cn/p/45987.html)and[here](https://blog.csdn.net/tinyjian/article/details/79110473)有相关技术博客。
    
        > 摩尔投票法的基本原理是数学上可以证明数组中出现次数超过数组长度一半的元素最多仅有一个。在从左向右遍历数组的过程中维护两个变量，当前大多数cur_majority和当前大多数的投票得分cur_votes，在遍历数组的过程中遇到投票vote：
        - 如果vote和当前大多数相同，则得分cur_votes++
        - 如果vote和当前大多数不同
            - 此时如果当前大多数的得分已经为0，则用vote更新当前大多数，且得分更新为1
            - 此时如果当前大多数的得分尚大于0，则得分cur_vote--
        > 最终的当前大多数即是得票过半的 majority element。
    
    ```cpp
    int majorityElement(vector<int> &nums)
    {
        // method 1, sorting and search

        // sort(nums.begin(), nums.end());
        // int cur_count = 1, count = nums.size(), half = count / 2, pivot = nums[0];
        // for (int i = 1; i < count; i++)
        // {
        // 	if (cur_count > half)
        // 	{
        // 		break;
        // 	}
        // 	if (nums[i] == pivot)
        // 	{
        // 		cur_count++;
        // 	}
        // 	else
        // 	{
        // 		cur_count = 1;
        // 		pivot = nums[i];
        // 	}
        // }
        // return pivot;

        // method 2, mooer voting

        int cur_majority = nums[0], cur_votes = 1;
        for (int i = 1; i < nums.size(); i++)
        {
            if (nums[i] == cur_majority)
            {
                cur_votes++;
            }
            else if (cur_votes > 0)
            {
                cur_votes--;
            }
            else
            {
                cur_majority = nums[i];
                cur_votes = 1;
            }
        }
        return cur_majority;
    }
    ```

- [172](https://leetcode.com/problems/factorial-trailing-zeroes/)

    计算给定数n的阶乘($n!$)的结果中0的个数，理论上结果中0只能由5的偶数倍得来，因此计算从1到n的所有数中因子5的个数即可，时间复杂度$O(log(n))$。

    ```cpp
    int trailingZeroes(int n)
    {
        int ans = 0, power = 1;
        long base = 5;
        while (base <= n)
        {
            ans += n / base;
            base *= 5;
            power++;
        }
        return ans;
    }
    ```

- [209](https://leetcode.com/problems/minimum-size-subarray-sum/)

    用左右双指针left、right设置滑动窗口capacity来满足sum和的要求，求滑动窗口可能的最小值即可
    
    ```cpp
    int minSubArrayLen(int s, vector<int> &nums)
    {
        int left = 0, right = 0, capacity = 0, count = nums.size(), ret = numeric_limits<int>::max();
        while (right < count)
        {
            // while (right < count && capacity < s)
            // {
            // 	capacity += nums.at(right++);
            // }
            // while (left < count && capacity >= s)
            // {
            // 	ret = min(right - left, ret);
            // 	capacity -= nums.at(left++);
            // }

            capacity += nums[right++];
            while (capacity >= s)
            {
                ret = min(ret, right - left);
                capacity -= nums[left--];
            }
        }
        ret = (ret == numeric_limits<int>::max()) ? 0 : ret;
        return ret;
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

- [227](https://leetcode.com/problems/basic-calculator-ii/)

    计算只包含加减乘除和正整数的表达式的值

    - 方法一，给加减和乘除定义不同的优先级，然后使用栈将其转化为逆波兰表达式，最终求值的

    ```cpp
    int calculate(string s) {
        vector<string> items;
        unordered_map<string,int> priority;
        priority["+"]=0;
        priority["-"]=0;
        priority["*"]=1;
        priority["/"]=1;
        string item;
        stack<string> ops;
        for (int i = 0; i < s.length(); i++)
        {
            if(s[i]==' '){
                continue;
            }else if(isdigit(s[i])){
                item+=s[i];
            }else{
                if(item.length()>0){
                    items.push_back(item);
                    item.clear();
                }
                item=s[i];
                while(!ops.empty() && priority[item]<=priority[ops.top()]){
                    items.push_back(ops.top());
                    ops.pop();
                }
                ops.push(item);
                item.clear();
            }
        }
        if(item.length()>0){
            items.push_back(item);
        }
        while (!ops.empty())
        {
            items.push_back(ops.top());
            ops.pop();
        }
        // for calculate
        stack<int> nums;
        for (const string &x:items)
        {
            if(isdigit(x[0])){
                nums.push(stoi(x));
            }else
            {
                char c=x[0];
                int b=nums.top();
                nums.pop();
                int a=nums.top();
                nums.pop();
                switch (c)
                {
                case '+':
                    /* code */
                    nums.push(a+b);
                    break;
                case '-':
                    /* code */
                    nums.push(a-b);
                    break;
                case '*':
                    /* code */
                    nums.push(a*b);
                    break;
                case '/':
                    /* code */
                    nums.push(a/b);
                    break;
                default:
                    break;
                }
            }
        }
        return nums.top();
    }
    ```

    - 方法二，因为没有使用括号，因此可以在 one pass 扫描的时候直接计算高优先级的乘除的值，并将减法转化为相反数的加法，最后将所有的数直接求和即可

    ```cpp
    int calculate(string s)
    {
        int ans = 0;
        if (s.length() > 0)
        {
            stack<int> nums;
            string ops = "+-*/";
            nums.push(0);
            char op = '+';
            int cur_value = 0;
            string::iterator it = s.begin();
            while (true)
            {
                if (isdigit(*it))
                {
                    cur_value = cur_value * 10 + (int)(*it - '0');
                }
                if (it + 1 == s.end() || (ops.find(*it) != ops.npos))
                {
                    if (op == '+')
                    {
                        nums.push(cur_value);
                    }
                    else if (op == '-')
                    {
                        nums.push(-cur_value);
                    }
                    else if (op == '*')
                    {
                        int temp = nums.top();
                        nums.pop();
                        nums.push(temp * cur_value);
                    }
                    else if (op == '/')
                    {
                        int temp = nums.top();
                        nums.pop();
                        nums.push(temp / cur_value);
                    }
                    cur_value = 0;
                    op = *it;
                }
                if (it + 1 == s.end())
                {
                    break;
                }
                else
                {
                    it++;
                }
            }
            while (!nums.empty())
            {
                ans += nums.top();
                nums.pop();
            }
        }
        return ans;
    }
    ```

- [229](https://leetcode.com/problems/majority-element-ii/)

    在给定数组中寻找出现次数超过$\frac{1}{3}$的数字，是[169](https://leetcode.com/problems/majority-element/)题的升级版，同样用hashmap进行统计，排序，摩尔投票等方法解决。

    ```cpp
    vector<int> majorityElement(vector<int> &nums)
    {
        // method 1, hash map counting

        // unordered_map<int, int> count;
        // for (auto x : nums)
        // {
        // 	if (count.find(x) != count.end())
        // 	{
        // 		count[x]++;
        // 	}
        // 	else
        // 	{
        // 		count[x] = 1;
        // 	}
        // }
        // int length_one_of_third = nums.size() / 3;
        // vector<int> ret;
        // for (const auto &pair : count)
        // {
        // 	if (pair.second > length_one_of_third)
        // 	{
        // 		ret.push_back(pair.first);
        // 	}
        // }
        // return ret;

        // method 2, mooer voting
        vector<int> ret;
        if (nums.size() > 0)
        {
            int cur_majority_A = nums[0], cur_majority_B = nums[0];
            int cur_votes_A = 0, cur_votes_B = 0;
            for (auto x : nums)
            {
                // 投票过程
                if (x == cur_majority_A)
                {
                    cur_votes_A++;
                    continue;
                }
                if (x == cur_majority_B)
                {
                    cur_votes_B++;
                    continue;
                }
                if (cur_votes_A == 0)
                {
                    cur_majority_A = x;
                    cur_votes_A = 1;
                    continue;
                }
                if (cur_votes_B == 0)
                {
                    cur_majority_B = x;
                    cur_votes_B = 1;
                    continue;
                }

                // 此时AB均为被投票且他们的得票数都大于0，因此都减一分
                cur_votes_A--;
                cur_votes_B--;
            }
            /*
            投票结束，题目并未保证给定数组中一定有两个出现次数超过1/3的数字，
            所以还要检查这两个数出现的次数是否真的超过1/3,
            这是因为摩尔投票法仅能找出出现次数超过一半的数而不能找出出现次数最多的众数
        	*/
            cur_votes_A = 0, cur_votes_B = 0;
            for (auto x : nums)
            {
                if (x == cur_majority_A)
                {
                    cur_votes_A++;
                }
                else if (x == cur_majority_B)
                {
                    cur_votes_B++;
                }
            }
            if (cur_votes_A > nums.size() / 3)
            {
                ret.push_back(cur_majority_A);
            }
            if (cur_votes_B > nums.size() / 3)
            {
                ret.push_back(cur_majority_B);
            }
        }
        return ret;
    }
    ```

- [237](https://leetcode.com/problems/delete-node-in-a-linked-list/)

    给定链表中某个节点，删除该节点，难点在于该节点的前继节点未知。如果该节点有后继，则用后继节点来代替该节点，如果没有后继，则该节点指向空即可。

    ```cpp
    void deleteNode(ListNode* node) {
        if(node->next){
            node->val=node->next->val;
            node->next=node->next->next;
        }else{
            node=nullptr;
        }
    }
    ```

- [234](https://leetcode.com/problems/palindrome-linked-list/)

    在$O(n)$时间和$O(1)$空间下判断一个链表是否回文，第一次遍历找到中点，然后翻转后半部分和前半部分进行比较即可

- [239](https://leetcode.com/problems/sliding-window-maximum/)

    给定数组nums和窗口大小k，求数组在窗口滑动过程中的最大值，这里主要是双端队列的使用。

    ```cpp
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
		vector<int> ans;
		deque<int> dq;
		for (int i = 0; i < nums.size(); i++)
		{
			while(!dq.empty()&&nums[dq.back()]<nums[i]){
				dq.pop_back();
			}
			dq.push_back(i); // 记录当前最大值的下标
			if(dq.back()-dq.front()>k-1){
				dq.pop_front();
			}
			if(i>=k-1){
				ans.push_back(nums[dq.front()]);
			}
		}
		return ans;
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

- [264](https://leetcode.com/problems/ugly-number-ii/)

    定义：素因子只有2,3,5的数正整数成为ugly数，1是特殊的ugly数。

    问题：寻找第n个ugly数

    方法：
    - 暴力：从1开始逐个检查所有正整数序列是否为ugly数直到统计量count达到n
    - dynamic program：除1外，下一个ugly数必然是2,3,5的倍数(倍率分别从1开始，每使用一次倍率增长一次)中较小的一个

- [268](https://leetcode.com/problems/missing-number/)
    - Gauss formual
    
    从0到n的和是$S_0=\frac{n(n+1)}{2}$，missing一个数x以后的和是$S=\frac{n(n+1)}{2}-x$，则丢失的数是$x=S_0-S$。

    - bit XOR

    下标是0到n-1，补充一个n作为初始值，然后这些数字是0-n且missing一个数，相当于从0-n除了missing的这个数只出现一次之外其他数字都出现了两次，因此可以用XOR操作找到这个只出现了一次的数即可。

- [278](https://leetcode.com/problems/first-bad-version/)

    在形如$(1,2,3,4,5,...,n,n+1,n+1,n+1,n+1,n+1,n+1)$这样的数组中中寻找第一个$n+1$的下标位置，这是二叉搜索的另一种形式，即每次命中$n+1$的右侧都是$n+1$，而没有命中的左侧都不是。相似的题目还有寻找有重复的排序数组中某个元素出现的下标区间[leetcode 34](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)等。

    ```cpp
    int firstBadVersion(int n) {
        unsigned int lo=1,hi=n;
        while(lo<hi){
            unsigned int mid=(lo+hi)>>1;
            if(isBadVersion(mid)){
                hi=mid;
            }else{
                lo=mid+1;
            }
        }
        return lo;
    }
    ```

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

- [297](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)

    序列化或者反序列化一颗树

    ```cpp
    class Codec {
    public:
        // Encodes a tree to a single string.
        string serialize(TreeNode *root)
        {
            vector<TreeNode *> serialized, current_level, next_level;
            if (root)
            {
                // serialize a tree
                current_level.push_back(root);
                while (!current_level.empty())
                {
                    // add current level to serialized
                    for (int i = 0; i < current_level.size(); i++)
                    {
                        serialized.push_back(current_level[i]);
                        if (current_level[i])
                        {
                            next_level.push_back(current_level[i]->left);
                            next_level.push_back(current_level[i]->right);
                        }
                    }
                    current_level = next_level;
                    next_level.clear();
                }
                // remove nullptr in the end
                while (!current_level.empty())
                {
                    if (current_level.back() == nullptr)
                    {
                        current_level.pop_back();
                    }
                    else
                    {
                        break;
                    }
                }
            }
            // convert serialized to string
            string ans;
            for (auto &&node : serialized)
            {
                if (node)
                {
                    ans += to_string(node->val);
                }
                else
                {
                    ans += "NULL";
                }
                ans.push_back(',');
            }
            if (ans.length() > 0)
            {
                ans.pop_back();
            }
            ans = '[' + ans + ']';
            return ans;
        }
        // Decodes your encoded data to tree.
        TreeNode *deserialize(string data)
        {
            TreeNode *root = nullptr;
            data = data.substr(1, data.length() - 2);
            if (data.length() > 0)
            {
                stringstream ss;
                string item;
                ss.str(data);
                getline(ss, item, ',');
                root = new TreeNode(stoi(item));
                queue<TreeNode *> qe;
                qe.push(root);
                while (true)
                {
                    TreeNode *cur_node = qe.front();
                    qe.pop();
                    if (!getline(ss, item, ','))
                    {
                        break; // end of the string
                    }
                    if (item.compare("NULL") != 0)
                    {
                        TreeNode *left = new TreeNode(stoi(item));
                        cur_node->left = left;
                        qe.push(left);
                    }
                    if (!getline(ss, item, ','))
                    {
                        break; // end of the string
                    }
                    if (item.compare("NULL") != 0)
                    {
                        TreeNode *right = new TreeNode(stoi(item));
                        cur_node->right = right;
                        qe.push(right);
                    }
                }
            }
            return root;
        }

    };

    // test case
    // [1,2,3,null,null,4,5]
    // [3,5,1,6,2,0,8,null,null,7,4,null,9,null,null,null,8,5]
    // [3,5,1,6,2,0,8,null,null,7,4,null,9,null,8,5]

    // Your Codec object will be instantiated and called as such:
    // Codec codec;
    // codec.deserialize(codec.serialize(root));
    ```

- [300](https://leetcode.com/problems/longest-increasing-subsequence/)

    求给定无序顺序的最长升序子序列

    - 思路一，dynamic plan，时间复杂度$O(n^2)$

    ```cpp
    int lengthOfLIS(vector<int> &nums)
    {
        const int count = nums.size();
        vector<int> length(count, 1);
        int ans = 0;
        for (int i = 0; i < count; i++)
        {
            int j = 0;
            while (j < i)
            {
                if (nums[j] < nums[i])
                {
                    length[i] = max(length[j] + 1, length[i]);
                }
                j++;
            }
            ans = max(ans, length[i]);
        }
        return ans;
    }
    ```

    - 思路二，dynamic + binary search，时间复杂度$O(nlog(n))$

    ```cpp
    int lengthOfLIS(vector<int> &nums)
    {
        const int count = nums.size();
        int ans = 0;
        if (count < 2)
        {
            ans = count;
        }
        else
        {
            vector<int> longestIncreasingSeries;
            longestIncreasingSeries.push_back(nums[0]);
            for (int i = 1; i < count; i++)
            {
                if (nums[i] > longestIncreasingSeries.back())
                {
                    longestIncreasingSeries.push_back(nums[i]);
                }
                int left = 0, right = longestIncreasingSeries.size() - 1;
                // 在当前 longIncreasingSeries 中用 nums[i] 替换比其大的最小值
                while (left < right)
                {
                    int mid = left + ((right - left) >> 1);
                    if (nums[i] > longestIncreasingSeries[mid])
                    {
                        left = mid + 1;
                    }
                    else
                    {
                        right = mid;
                    }
                }
                longestIncreasingSeries[left] = nums[i];
            }
            ans = longestIncreasingSeries.size();
        }
        return ans;
    }
    ```

- [315](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)

	求逆序数对的问题，线段树或者归并排序(稳定排序)的过程中统计交换的次数或者BST。

	```cpp
	class Solution
	{
	private:
		struct myTreeNode
		{
			/* data */
			int val;
			int count;      // number of the value in nums
			int less_count; // number of x in nums where x < value and x is in the right of value
			myTreeNode *left;
			myTreeNode *right;
			myTreeNode(int x) : val(x), count(1), less_count(0), left(NULL), right(NULL) {}
		};

		int update_BST(myTreeNode *root, int val)
		{
			int cur_count = 0;
			if (root)
			{
				// three case of comparision between val and root->val
				if (val < root->val)
				{
					// go to left subtree
					root->less_count++;
					if (root->left)
					{
						cur_count += update_BST(root->left, val);
					}
					else
					{
						root->left = new myTreeNode(val);
					}
				}
				else if (val == root->val)
				{
					// return the result without update on the BST
					cur_count = root->less_count;
					root->count++;
				}
				else
				{
					// go to right subtree
					cur_count += root->count + root->less_count;
					if (root->right)
					{
						cur_count += update_BST(root->right, val);
					}
					else
					{
						root->right = new myTreeNode(val);
					}
				}
			}
			return cur_count;
		}

	public:
		vector<int> countSmaller(vector<int> &nums)
		{
			const int count = nums.size();
			vector<int> ans(count, 0);
			if (count > 0)
			{
				// iteration from right to left
				myTreeNode *root = new myTreeNode(nums.back());
				for (int i = count - 2; i >= 0; i--)
				{
					ans[i] = update_BST(root, nums.at(i));
				}
			}
			return ans;
		}
	};
	```

- [337](https://leetcode.com/problems/house-robber-iii/)

    - 最简单版本[198](https://leetcode.com/problems/house-robber/)
        所有房子的价值以数组nums形式给出，顺序遍历dp即可
        $$dp[i]=\left\{\begin{matrix} x[0], i=0\\ max(x[0],x[1]), i=1\\ max(dp[i-2]+x[i],dp[i-1]), i\geqslant 2 \end{matrix}\right.$$

    - 升级版[213](https://leetcode.com/problems/house-robber-ii/)
        此时所有房子形成了一个环，即第一个和最后一个互相接壤，会互相影响，对数组nums[0,n-2],nums[1,n-1]执行两次dp取较大值即可

    - 本题中所有房子价值按照二叉树形式给出，只有一个入口，即根节点root，优质discussion在[这里](https://leetcode.com/problems/house-robber-iii/discuss/79330/Step-by-step-tackling-of-the-problem)
        - naive dp with recursion **[TLE](Time Limit Exceeded)**

        ```cpp
      	int rob(TreeNode* root) {
            int ret=0;
            if(root){
                int cur_value=root->val;
                if(root->left){
                    cur_value+=rob(root->left->left)+rob(root->left->right);
                }
                if(root->right){
                    cur_value+=rob(root->right->left)+rob(root->right->right);
                }
                ret=max(rob(root->left)+rob(root->right),cur_value);
            return ret;
        }
        ```

        - 朴素的递归过程中有很多的重复计算问题，可以通过hashmap来记录拜访过的子节点，在递归的过程中相当于实现了自底向上的dp过程，可以在$O(n)$时间和$O(n)$空间限制内解决问题

        ```cpp
        unordered_map<TreeNode*,int> visited;
        int rob(TreeNode* root) {
            int ret=0;
            if(root){
                if(visited.find(root)!=visited.end()){
                    ret=visited[root];
                }else{
                    int cur_value=root->val;
                    if(root->left){
                        cur_value+=rob(root->left->left)+rob(root->left->right);
                    }
                    if(root->right){
                        cur_value+=rob(root->right->left)+rob(root->right->right);
                    }
                    ret=max(rob(root->left)+rob(root->right),cur_value);
                    visited[root]=ret;
                }
            }
            return ret;
        }
        ```

        - 在每个节点计算两个值pair<int,int>，其中pair.first表示当前节点不选择时左右子树获得的最大价值，pair.second代表选择当前节点时当前节点及其子树可以获得的最大价值，递归之后在根节点的两个值中选择最大值即可

        ```cpp
        int rob(TreeNode* root) {
            pair<int,int> res=robsub(root);
            return max(res.first,res.second);
        }

        pair<int,int> robsub(TreeNode* root){
            if(root){
                /*
                    pair.first	当前节点不选择的最大价值
                    pair.second	当前节点选择的最大价值
                */
                pair<int,int> left=robsub(root->left);
                pair<int,int> right=robsub(root->right);
                return make_pair(max(left.first,left.second)+max(right.first,right.second),root->val+left.first+right.first);
            }else{
                return make_pair(0,0);
            }
        }
        ```

- [338](https://leetcode.com/problems/counting-bits/)

    注意分组统计的办法，类似和二进制表示中1的数量有关的问题和2的倍数有关系，主要有两条规则：
    - 一个偶数乘以2之后其二进制表示中1的个数不变
    - 一个奇数的二进制表示中1的个数等于它的前一个数（必然是偶数）的二进制表示中1的个数加一

    ```cpp
	vector<int> countBits(int num)
	{
		// method, count for every number
		// vector<int> ret(num + 1, 0);
		// for (int i = 1; i <= num; i++)
		// {
		// 	int count = 0, n = i;
		// 	while (n)
		// 	{
		// 		count += n & 1;
		// 		n = (n >> 1);
		// 	}
		// 	ret[i] = count;
		// }
		// return ret;

		// method 2, divide and computing
		vector<int> ret(num + 1, 0);
		for (int i = 1, distance = 1; i <= num; i++)
		{
			distance = (i == (distance << 1)) ? (distance << 1) : distance;
			ret[i] = ret[i - distance] + 1;
		}
		return ret;
	}
    ```

- [367](https://leetcode.com/problems/valid-perfect-square/)

    线性时间内判断一个数是否是完全平方数而不用开方函数，代码如下：

    ```cpp
    bool isPerfectSquare(int num) {
        long long i = 1, sum = 0;
        while (sum < num)
        {
            sum += i;
            i += 2;
        }
        return sum == num;
    }
    ```

    在数学上可以证明对于任何一个完全平方数有：
    $$
    \begin{array}{l}{n^{2}=1+3+5+ \ldots +(2 \cdot n-1)=\sum_{i=1}^{n}(2 \cdot i-1)} \\
    {\text { provement:}}
    \\ {\quad 1+3+5+\ldots+(2 \cdot n-1)} \\
    {=(2 \cdot 1-1)+(2 \cdot 2 - 1) + (2 \cdot 3-1)+\ldots+(2 \cdot n-1)} \\
    {=2 \cdot(1+2+3+\ldots+n)-(\underbrace{1+1+\ldots+1}_{n \text { times }})} \\
    {=2 \cdot \frac{n(n+1)}{2}-n} \\
    {=n^{2}+n-n} \\
    {=n^{2}}\end{array}
    $$

- [371](https://leetcode.com/problems/sum-of-two-integers/)

    不用加减操作求两个有符号整数的和，使用位操作实现，这里与一些列位操作相关问题[here](https://leetcode.com/problems/sum-of-two-integers/discuss/84278/A-summary%3A-how-to-use-bit-manipulation-to-solve-problems-easily-and-efficiently)

    ```cpp
    int getSum(int a, int b) {
        long long c = 0;
        while (b)
        {
            c = a & b;
            a = a ^ b;
            b = (c & 0xffffffff) << 1;
            /*
            这里c定义为长整型而又和0xffffffff按位与的目的是
            1. 避免signed int表示的最小负数左移时溢出的问题
            2. 将c限制在unsigned int的32 bits内
            */
        }
        return a;
    }
    ```
    
- [394](https://leetcode.com/problems/decode-string/)

    关于括号中嵌套的问题，可以用递归的方法解决

    ```cpp
    string decodeString(string s)
    {
        string ans;
        if (s.length() > 0)
        {
            int i = 0, count = s.length();
            while (i < count)
            {
                if (isdigit(s[i]))
                {
                    int length = 0;
                    while (i < count && isdigit(s[i]))
                    {
                        length = length * 10 + (int)(s[i] - '0');
                        i++;
                    }
                    int open_start = i + 1, open_count = 1;
                    i++; // skip the first [
                    while (i < count && open_count > 0)
                    {
                        if (s[i] == '[')
                        {
                            open_count++;
                        }
                        else if (s[i] == ']')
                        {
                            open_count--;
                        }
                        i++;
                    }
                    string temp = s.substr(open_start, i - 1 - open_start); // substr(begin_index,length)
                    temp = decodeString(temp);
                    for (int j = 0; j < length; j++)
                    {
                        ans += temp;
                    }
                }
                else
                {
                    ans.push_back(s[i++]);
                }
            }
        }
        return ans;
    }
    ```

- [400](https://leetcode.com/problems/nth-digit/)
    
    在$1,2,3,4,5,6,7,8,9,10,11,...$的数列中找到第n个数字，思想简单，分别计算不同位宽的数字个数即可(k位数一共占据$k*(9*10^k)$个位置)，但是注意实现时的具体细节处理。

    ```cpp
    int findNthDigit(int n)
    {
        long base = 9, digits = 1;
        while (n - base * digits > 0)
        {
            n -= base * digits;
            base *= 10;
            digits++;
        }
        int index = (n % digits) ? (n % digits) : digits;
        long num = 1;
        for (int i = 1; i < digits; i++)
        {
            num *= 10;
        }
        num += (index == digits) ? n / digits - 1 : n / digits;
        for (int i = index; i < digits; i++)
        {
            num /= 10;
        }
        return num % 10;
    }
    ```

- [401](https://leetcode.com/problems/binary-watch/)

    有些问题要用逆向思维来解决，本题本来是要用给定二进制位中1的个数来拼凑出可能的时间表示，组合数过程不好写，可以反过来写所有$00:00 - 23:59$中所有可能的时间中二进制位中1的个数符合题目要求的时间点。

- [442](https://leetcode.com/problems/find-all-duplicates-in-an-array/)
    
    在一个数组$a_n,1 \le a_i \le n$中找出出现了两次的数字（其他数字只出现了一次），要求不占用额外的内存空间、$O(n)$时间复杂度
    - 逐位负数标记法
    - 交换法

- [462](https://leetcode.com/problems/minimum-moves-to-equal-array-elements-ii/)

    使用nth_element()函数找到中位数即可。注意理解这个函数的实现原理，[here](http://c.biancheng.net/view/566.html)and[here](http://www.cplusplus.com/reference/algorithm/nth_element/)。

- [447](https://leetcode.com/problems/number-of-boomerangs/)

    给定平面上的一堆点，求符合欧氏距离$d_{ij}=d_{ik}$的点的三元组$(i,j,k)$的数量。

    - 暴力搜索，时间复杂度$O(n^3)$

    ```cpp
    int numberOfBoomerangs(vector<vector<int>>& points) {
        int length=points.size();
        vector<vector<int>> distances(length,vector<int>(length,0));
        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < length; j++)
            {
                int x=points[i][0]-points[j][0];
                int y=points[i][1]-points[j][1];
                distances[i][j]=x*x+y*y;
            }
        }
        int ans=0;
        for (int i = 0; i < length; i++)
        {
            for (int j = i+1; j < length; j++)
            {
                for (int k = j+1; k < length; k++)
                {
                    if(distances[i][j]==distances[i][k]){
                        ans++;
                    }
                    if(distances[j][i]==distances[j][k]){
                        ans++;
                    }
                    if(distances[k][i]==distances[k][j]){
                        ans++;
                    }
                }
            }
        }
        return ans*2; // (i,j,k) and (i,k,j)
    }    
    ```

    - 使用哈希map和排列算法，即到任意一点i的距离为固定值d的点数为p时，符合条件的三元组数量为$p*(p-1)$，时间复杂度$O(n^2)$

    ```cpp
    int numberOfBoomerangs(vector<vector<int>>& points) {
        int length=points.size(),ans=0;
        unordered_map<long,int> count;
        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < length; j++)
            {
                if(i==j){
                    continue;
                }else{
                    auto x=points[i][0]-points[j][0];
                    auto y=points[i][1]-points[j][1];
                    count[x*x+y*y]++;
                }
            }
            for (auto &&x : count)
            {
                ans+=x.second*(x.second-1); // permutation
            }
            count.clear();
        }
        return ans;
    }
    ```

- [448](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)

    本题和[442](https://leetcode.com/problems/find-all-duplicates-in-an-array/)很像，在遍历数组的过程中可以简单的用一个bool数组来标记每个下标是否出现即可，在不使用额外空间的情况下，可以用正负标记来代替true和false的bool标记在原数组中标记，只不过每次读取原数组的时候取绝对值即可。

- [468](https://leetcode.com/problems/validate-ip-address/)

    验证据given string是否是符合given rules的IPv4或IPv6地址，本题的key points有两处，一是正确理解given rules并在代码中体现出来，而是代码的有效结构设计，精巧的设计模式可以有效降低代码量并提高代码的可读性。

    ```cpp
    class Solution {
        public:
            bool check_IPv4_block(string item)
            {
                int v = 0;
                bool ans = false;
                if (item.length() >= 1 && item.length() <= 3)
                {
                    for (auto &&ch : item)
                    {
                        if (isdigit(ch))
                        {
                            v = v * 10 + (int)(ch - '0');
                        }
                        else
                        {
                            break;
                        }
                    }
                    if (v >= 0 && v <= 255 && item.compare(to_string(v)) == 0)
                    {
                        ans = true;
                    }
                }
                return ans;
            }
            bool check_IPv6_block(string item)
            {
                string chars = "0123456789abcdefABCDEF";
                bool ans = true;
                if (!(item.length() >= 1 && item.length() <= 4))
                {
                    // ensure item.length in [1,2,3,4]
                    ans = false;
                }
                else
                {
                    // ensure all characters in item si valid (in given chars)
                    for (auto &&ch : item)
                    {
                        if (chars.find(ch) == chars.npos)
                        {
                            ans = false;
                            break;
                        }
                    }
                }
                return ans;
            }
            string validIPAddress(string IP)
            {
                stringstream ss(IP);
                string item;
                if (IP.find('.') != IP.npos)
                {
                    // 可能是IPv4
                    for (int i = 0; i < 4; i++)
                    {
                        // check four block for IPv4
                        if (!getline(ss, item, '.') || !check_IPv4_block(item))
                        {
                            return "Neither";
                        }
                    }
                    if (!ss.eof())
                    {
                        // if their are other extra charachers, returh Neither
                        return "Neither";
                    }
                    return "IPv4";
                }
                else
                {
                    // 可能是IPv6
                    for (int i = 0; i < 8; i++)
                    {
                        // check four block for IPv6

                        if (!getline(ss, item, ':') || !check_IPv6_block(item))
                        {
                            return "Neither";
                        }
                    }
                    if (!ss.eof())
                    {
                        // if their are other extra charachers, returh Neither
                        return "Neither";
                    }
                    return "IPv6";
                }
                return "Neither";
            }
    };
    ```

- [475](https://leetcode.com/problems/heaters/submissions/)

    在数轴上固定位置有一些house，同样在固定位置有一些heater，求heater的最小作用半径以覆盖所有的house，算法基本思想如下：
    - 首先对house坐标和heater坐标排序，时间复杂度为$max(O(nlog(n)),O(mlog(m)))$，其中m和n是house数组和heater数组的长度
    - 然后从右到左扫描确定每个house到其左侧最近的heater的距离，时间复杂度$O(m+n)$
    - 然后从左到右扫描确定每个house到其右侧最近的heater的距离，时间复杂度$O(m+n)$
    - 然后每个house在其左右两侧最近的两个heater中选择一个，记录其距离
    - 最后在所有house的距离中选择最大值即可实现全覆盖
    这样在给定house和heater数组有序的情况下可以实现线性时间复杂度

    ```cpp
    int findRadius(vector<int> &houses, vector<int> &heaters)
    {
        sort(houses.begin(), houses.end());
        sort(heaters.begin(), heaters.end());
        // 记录每个house到左侧最近的heater的距离，相同位置距离为0，左侧没有heater距离为无穷大
        vector<int> left(houses.size(), numeric_limits<int>::max());
        for (int i = houses.size() - 1, h = heaters.size() - 1; i >= 0 && h >= 0;)
        {
            if (houses[i] >= heaters[h])
            {
                left[i] = houses[i] - heaters[h];
                i--;
            }
            else
            {
                h--;
            }
        }
        // 记录每个house到右侧最近的heater的距离，相同位置距离为0，右侧没有heater距离为无穷大
        vector<int> right(houses.size(), numeric_limits<int>::max());
        for (int i = 0, h = 0; i < houses.size() && h < heaters.size();)
        {
            if (houses[i] <= heaters[h])
            {
                right[i] = -(houses[i] - heaters[h]);
                i++;
            }
            else
            {
                h++;
            }
        }
        int ans = 0;
        for (int i = 0; i < left.size(); i++)
        {
            ans = max(ans, min(left[i], right[i]));
            // 然后每个house选择左右两侧距离自己较近的那个heater
            // ans在所有house选择的heater距离中选择最大值即可
        }
        return ans;
    }
    ```

- [509](https://leetcode.com/problems/fibonacci-number/)
    
    斐波那契数列
    - 递归法
    - 非递归循环 faster

- [516](https://leetcode.com/problems/longest-palindromic-subsequence/)

    求给定字符串s中的最长回文子串长度，将s翻转后形成字符串t，用动态规划求s和t的最长公共子序列LCS长度即可‘时间复杂度$O(log(n))$

    ```cpp
    int longestPalindromeSubseq(string s)
    {
        string t = s;
        reverse(s.begin(), s.end());
        const int length = s.length();
        vector<vector<int>> dp(length + 1, vector<int>(length + 1, 0));
        for (int i = 1; i <= length; i++)
        {
            for (int j = 1; j <= length; j++)
            {
                if (s[i - 1] == t[j - 1])
                {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }
                else
                {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp.back().back();
    }
    ```

- [521](https://leetcode.com/problems/longest-uncommon-subsequence-i/)

    注意理解最长非公共子串儿的正确含义，即只有两个字符串完全相同时才构成公共子串，否则最长非公共子串就是两个字符串中较长的一个。

- [535](https://leetcode.com/problems/encode-and-decode-tinyurl/)

    tinyURL的encode与decode算法

    整体思路：采用hashmap或者字符串数组存储<key,value>对，key是个全局唯一的ID，value是其longURL。而shortURL则是key的64进制表示，这64个字符一般是[0-9a-zA-Z+-]。这里之所以是64进制是因为64为2的6次幂，进制转换效率高。

- [539](https://leetcode.com/problems/minimum-time-difference/)

    在24时制的"hh:mm"格式字符串表示的时间序列中寻找最近的两个时间段差值，将每个时间用转换为分钟数后排序，在任意相邻的两个时间的差值中取最小值即可，时间复杂度$O(nlog(n))$，特别注意排序之后的时间序列第一个值和最后一个值之间的差值也要考虑在内。

    ```cpp
    int findMinDifference(vector<string> &timePoints)
    {
        vector<int> minutes;
        for (auto &&point : timePoints)
        {
            minutes.push_back(((point[0] - '0') * 10 + (point[1] - '0')) * 60 + ((point[3] - '0') * 10 + (point[4] - '0')));
        }
        sort(minutes.begin(), minutes.end());
        minutes.push_back(minutes[0] + 1440);
        int diff = numeric_limits<int>::max();
        for (int i = 1; i < minutes.size(); i++)
        {
            diff = min(diff, min(minutes[i] - minutes[i - 1], minutes[i - 1] + 1440 - minutes[i]));
        }
        return diff;
    }
    ```

- [561](https://leetcode.com/problems/array-partition-i/)
    
    2n个给定范围的数据划分成n组使得每组最小值求和最大，基本思路是对所有数排序后对基数位置上的数求和即可，这里类似于NMS非极大值抑制的思路，主要的时间复杂度在排序上。
    - 基本想法是quick sort，时间复杂度$O(nlog(n))$
    - 本题给出了数据范围，可以bucket sort，时间复杂度$O(n)$，但是需要$O(N)$的额外空间

- [583](https://leetcode.com/problems/delete-operation-for-two-strings/)

    删除最少数量的字符使得两个字符串相等即可，因为删除完成之后要保证相等，因此保留下来的是最长公共子串(LCS)，递归求解LCS会TLE，需要二维DP或者一维DP。

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

- [733](https://leetcode.com/problems/flood-fill/)
    
    类似于图像处理中区域增长的方式，采用DFS递归写法或者用栈stack实现

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

- [849](https://leetcode.com/problems/maximize-distance-to-closest-person/)

    给定一排座位，0表示空，1表示有人，新来一个人要安排到某个空的座位，使得新人到原来座位上的人(1)的距离尽可能的远，输出这最远距离。整体思路在于某一点到1的最远距离为该点到左边1的距离和到右边1的距离的较小值。
    - left[i]表示点i到到左边最近1的距离，right[i]表示点i到右边最近1的距离，则点i到1的最远距离为$min(left[i],right[i])$。特别注意处理边界点的距离，即左边界的0到其左边的1的距离为无穷大，右边界的0到右边的1的距离为无穷。

	```cpp
    int maxDistToClosest(vector<int>& seats) {
        const int count=seats.size();
        int ans=0;
        if(count>=2){
            vector<int> left(count,0),right(count,0);
            left[0]=seats[0]?0:count;
            for (int i = 1; i < count; i++)
            {
                left[i]=seats[i]?0:left[i-1]+1;
            }
            right[count-1]=seats[count-1]?0:count;
            for (int i = count-1; i > 0; i--)
            {
                right[i-1]=seats[i-1]?0:right[i]+1;
            }
            for (int i = 0; i < count; i++)
            {
                ans=max(ans,min(left[i],right[i]));
            }
        }
        return ans;
    }
	```

    - 对于左右被1围困的连续的k个0，其距离最近的1的距离为$\frac{k+1}{2}$，边界情况特殊讨论

    ```cpp
	int maxDistToClosest(vector<int>& seats) {
		const int count=seats.size();
		int ans=0;
		if(count>=2){
			int k=0,i=0;
			for (i = 0; i < count; i++)
			{
				if(seats[i]){
					ans=max(ans,(k+1)/2);
					k=0;
				}else{
					k++;
				}
			}
			// left border
			i=0;
			while(seats[i]==0){
				i++;
			}
			ans=max(ans,i);
			i=0;
			while (seats[count-1-i]==0)
			{
				i++;
			}
			ans=max(ans,i);
		}
		return ans;
	}
	```

    - two pointer法，用两个指针prev和future，pre指向点i的左边一个1，future指向点i的右边一个1，在该点i到最近1的距离即为$min(i-prev,future-i)$，同样要注意边界点上是0的情况的处理0。

	```cpp
	int maxDistToClosest(vector<int>& seats) {
		const int count=seats.size();
		int ans=0;
		if(count>=2){
		int prev=-1,future=0; // two pointer
		for (int i = 0; i < count; i++)
		{
			if(seats[i]){
				prev=i;
			}else{
				while(future<count && seats[future]==0 || future<i){
						// 保证future指针不回溯
					future++;
				}
				int left=(prev==-1)?count:i-prev;
				int right=(future==count)?count:future-i;
				ans=max(ans,min(left,right));
				}
			}
		}
		return ans;
	}
	```

- [852](https://leetcode.com/problems/peak-index-in-a-mountain-array/)

    寻找给定数组中的山峰数下标，所谓山峰数根据题目定义即为**全局最大值**，因此binary search是理论上的最佳算法，time complexity O(log(n))

- [855](https://leetcode.com/problems/exam-room/)

    本题给定包含N个座位的一排空椅子椅子，维护这个椅子序列，新来的学生要坐到距离任何一个现有学生最远的地方，同时有学生离开空出某个椅子的操作，按照[849](https://leetcode.com/problems/maximize-distance-to-closest-person/)做法实现正确结果，但是**TLE**。新方法采用内部有序的**set**数据结构来存储已经有人的位置下标，插入时O(n)顺序扫描set表，其中新学生插入任何两个元素位置$i,j$间和$i,j$的最大距离为$d=\frac{j-i}{2}$，位置下标为$index=i+d$，全局寻找d最大的的index即可，边界（第一个椅子没人和最后一个椅子没人的情况）情况特殊处理。

    ```cpp
    class ExamRoom {
    public:
        set<int> students;
        int number_of_seats;
        ExamRoom(int N) {
            number_of_seats=N;
        }

        int seat() {
            int ans=0;
            if(!students.empty()){
                // students is emtpy, insert new student to position 0, else to find max distance to existing student;
                int pre=-1,cur_max_dist=0;
                for(const int cur:students){
                    if(pre!=-1){
                        int dist=(cur-pre)/2;
                        if(dist>cur_max_dist){
                            cur_max_dist=dist;
                            ans=pre+dist;
                        }
                    }else if(cur!=0){
                        // pre==-1 && cur!=0, the case of leftmost seat is not occcupied
                        int dist=cur;
                        if(cur>cur_max_dist){
                            cur_max_dist=cur;
                            ans=0;
                        }
                    }
                    pre=cur;
                }
                // for the case of rightmost is 0
                if(*students.rbegin()!=number_of_seats-1){
                    if(number_of_seats-1-pre>cur_max_dist){
                        ans=number_of_seats-1;
                    }
                }
            }
            students.insert(ans);
            return ans;
        }
        void leave(int p) {
            students.erase(p);
        }
    };
    ```

- [874](https://leetcode.com/problems/walking-robot-simulation/)

    用坐标$(0,1)-north,(1,0)-east,(0,-1)-north,(-1,0)-west$来表示四个方向模拟机器人行走过程即可，注意题目要求返回的是机器人在行走过程中距离原点的最远距离，而不是机器人结束行走后距离原点的最终距离。

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

- [880](https://leetcode.com/problems/decoded-string-at-index/)

    通过字符和重复次数给定一个字符串，求解码吼字符串中给定index的字符，类似这样只求指定位置字符的课通过下标运算获得，无需求出整个字符串，否则很容易memory limitation，这里注意$\color{red}{逆向思维}$

    ```cpp
    string decodeAtIndex(string S, int K)
    {
        string ans;
        // calculate the length of decoded string
        long long tape_length = 0;
        for (int i = 0; i < S.length(); i++)
        {
            if (S[i] <= '9' && S[i] >= '2')
            {
                tape_length *= (int)(S[i] - '0'); // digital
            }
            else
            {
                tape_length++; // lower case letter
            }
        }
        // Reverse inferring the Kth character in a string
        for (int i = S.length() - 1; i >= 0; i--)
        {
            K %= tape_length;
            if (K == 0 && islower(S[i]))
            {
                ans += S[i];
                break;
            }
            else if (S[i] <= '9' && S[i] >= '2')
            {
                tape_length /= (int)(S[i] - '0'); // digital
            }
            else
            {
                tape_length--;
            }
        }
        return ans;
    }
    ```

    Some test case

    ```cpp
    // "a23"
    // 6
    // "ajx37nyx97niysdrzice4petvcvmcgqn282zicpbx6okybw93vhk782unctdbgmcjmbqn25rorktmu5ig2qn2y4xagtru2nehmsp"
    // 976159153
    // "vzpp636m8y"
    // 2920
    // "a2b3c4d5e6f7g8h9"
    // 10
    // "leet2code3"
    // 10
    // "ha22"
    // 5
    // "a2345678999999999999999"
    // 1
    ```

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

- [929](https://leetcode.com/problems/unique-email-addresses/)
    两个考察点
    - cpp STL中的string操作，子串、查找、替换等
    - cpp STL中集合(set)的使用，或者自己实现一个set

- [930](https://leetcode.com/problems/binary-subarrays-with-sum/)

    待补充    

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

    本题充分体现了hash存储的优势。

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

- [985](https://leetcode.com/problems/sum-of-even-numbers-after-queries/)
    注意每次query对应下标的数字在query前后的奇偶性，分别有不同的操作。time complexity O(n+q)，其中n(size of array) and q(the number of queries)。

- [1002](https://leetcode.com/problems/find-common-characters/)

    用两个长度为26的数组来统计两个单词中每个字母出现的次数，然后共同字母的次数为两个单词分别统计结果中的较小值，如果有一方为0则表示没有共同出现过，时间复杂度$O(n)$，其中n是所有单词长度的和，即所有字母的个数。

    ```cpp
    vector<string> commonChars(vector<string> &A)
    {
        // string s = A[0];
        // sort(s.begin(), s.end());
        // for (int k = 1; k < A.size(); k++)
        // {
        // 	string x, t = A[k];
        // 	sort(t.begin(), t.end());
        // 	int i = 0, j = 0;
        // 	while (i < s.length() && j < t.length())
        // 	{
        // 		if (s[i] == t[j])
        // 		{
        // 			x += s[i];
        // 			i++;
        // 			j++;
        // 		}
        // 		else if (s[i] < t[j])
        // 		{
        // 			i++;
        // 		}
        // 		else
        // 		{
        // 			j++;
        // 		}
        // 	}
        // 	s = x;
        // }
        // vector<string> ret;
        // for (string::iterator it = s.begin(); it != s.end(); it++)
        // {
        // 	string t;
        // 	t += *it;
        // 	ret.push_back(t);
        // }
        // return ret;

        // method 2, O(n)
        const int length_of_lowercases = 26;
        vector<int> count(length_of_lowercases, numeric_limits<int>::max());
        for (auto word : A)
        {
            vector<int> cnt(length_of_lowercases, 0);
            for (auto c : word)
            {
                cnt[c - 'a']++;
            }
            for (int i = 0; i < length_of_lowercases; i++)
            {
                count[i] = min(count[i], cnt[i]);
            }
        }
        vector<string> ret;
        for (int i = 0; i < length_of_lowercases; i++)
        {
            for (int j = 0; j < count[i]; j++)
            {
                ret.push_back(string(1, i + 'a'));
            }
        }
        return ret;
    }
    ```

- [1008](https://leetcode.com/problems/construct-binary-search-tree-from-preorder-traversal/)

    从二叉搜索树BST的先序遍历preorder开始重建BST

    - 以preorder[0]为分界线将后面preorder后面的部分划分成两部份，左半部分递归重建左子树，右半部分递归重建右子树，时间复杂度$O(nlog(n))$

    ```cpp
    TreeNode *bstFromPreorder(vector<int> &preorder)
    {
        if (preorder.size() > 0)
        {
            TreeNode *root = new TreeNode(preorder[0]);
            int index = 1;
            while (index < preorder.size() && preorder[0] > preorder[index])
            {
                index++;
            }
            root->left = bstFromPreorder(vector<int>&(preorder.begin() + 1, preorder.begin() + index));
            root->right = bstFromPreorder(vector<int>&(preorder.begin() + index, preorder.end()));
            return root;
        }
        else
        {
            return NULL;
        }
    }
    ```

    - 重写函数为bstFromPreorder(vector<int>& preorder,int boundary)，设定一个界boundary，从preorder[i]开始用下标i遍历preorder，小于界boundary的递归建立左子树，大于界的递归建立右子树，可以实现线性时间复杂度$O(n)$

    ```cpp
    int i = 0;
    TreeNode *bstFromPreorder(vector<int> &preorder)
    {
        return bstFromPreorder(preorder, numeric_limits<int>::max());
    }
    TreeNode *bstFromPreorder(vector<int> &preorder, int boundary)
    {
        if (!(i < preorder.size()) || preorder[i] > boundary)
        {
            return NULL;
        }
        else
        {
            TreeNode *root = new TreeNode(preorder[i++]);
            root->left = bstFromPreorder(preorder, root->val);
            root->right = bstFromPreorder(preorder, boundary);
            return root;
        }
    }
    ```

    - 栈实现，时间复杂度$O(n)$，遍历数组中的每个元素item，找到第一个比item小的数p，然后把item挂到p的右孩子，因此需要维护一个从栈底到栈顶递减的栈序列，从而为数组迭代过程中的每个数item找到第一个比它小的数p，即栈顶元素；如果栈中不存在p比item小，则item当前最小，成为栈顶元素的左孩子。

    ```cpp
    TreeNode *bstFromPreorder(vector<int> &preorder)
    {
        TreeNode *cur_root = new TreeNode(numeric_limits<int>::max());
        stack<TreeNode *> st;
        st.push(cur_root);
        for (auto item : preorder)
        {
            TreeNode *cur = new TreeNode(item), *p = nullptr;
            while (st.top()->val < item)
            {
                p = st.top();
                st.pop();
            }
            if (p)
            {
                p->right = cur;
            }
            else
            {
                st.top()->left = cur;
            }
            st.push(cur);
        }
        return cur_root->left;
    }
    ```

- [1019](https://leetcode.com/problems/next-greater-node-in-linked-list/)

    用栈可以实现$O(n)$时间复杂度，即对数组从右往左遍历的过程中保持栈顶$st[i]<st[i-1]$，从栈底到栈顶是严格递增的顺序

    ```cpp
    vector<int> nextLargerNodes(ListNode *head)
    {
        // method 1, O(n^2)

        // vector<int> ret;
        // while (head)
        // {
        // 	ListNode *cur = head->next;
        // 	int base = 0;
        // 	while (cur->next)
        // 	{
        // 		if (head->val < cur->val)
        // 		{
        // 			base = cur->val;
        // 			break;
        // 		}
        // 		else
        // 		{
        // 			cur = cur->next;
        // 		}
        // 	}
        // 	ret.push_back(base);
        // 	head = head->next;
        // }
        // return ret;

        // method 2, O(n), using stack

        vector<int> ret, st;
        while (head)
        {
            // convert linked list to array
            ret.push_back(head->val);
            head = head->next;
        }
        for (int i = ret.size() - 1; i >= 0; i--)
        {
            // maintain the stack is decreasing from st.top() to st.bottom()
            while (!st.empty() && st.back() <= ret[i])
            {
                st.pop_back();
            }
            st.push_back(ret[i]);
            // st[i-1] is the next greater number of st[i]
            ret[i] = st.size() > 1 ? st[st.size() - 2] : 0;
        }
        return ret;
    }
    ```

- [1024](https://leetcode.com/problems/video-stitching/)

    一开始设定当前右端点$cur_right$，然后按照贪心策略寻找左端点小于等于当前右端点(保证可以连接而没有断点)且右端点最远(贪心原则，以便使用最少的视频片段)的没有用过的视频片段，直到所有视频片段被用完或者当前右端点$cur_right$超过了总时间长度要求$T$。

- [1032](https://leetcode.com/problems/stream-of-characters/)

    本题主要练习字典树的的构建与查询，tire tree是一种高效的单词存储与查询数据结构，比如可以完成IDE的代码自动提示语补全功能，[here](https://blog.csdn.net/v_july_v/article/details/6897097)有相关博客。

- [1035](https://leetcode.com/problems/uncrossed-lines/)

    类似于LCS最长公共子序列的问题，二维dp或者一维dp均可，注意问题中隐藏的dp思维。

- [1042](https://leetcode.com/problems/flower-planting-with-no-adjacent/)

    贪心思想，时间复杂度$O(n)$：经典的染色问题，在给定四种颜色且保证有满足条件的答案时可以确保每个节点的连通节点小于等于三个，因此外层循环遍历每个节点i，内层循环遍历该节点链接的所有节点j，使用j尚未使用的颜色染给i即可。

    ```cpp
    vector<int> gardenNoAdj(int N, vector<vector<int>>& paths) {
        vector<vector<int>> connections(N);
        vector<int> ans(N,0);
        for (int i = 0; i < paths.size(); i++)
        {
            connections[paths[i][0]-1].push_back(paths[i][1]-1);
            connections[paths[i][1]-1].push_back(paths[i][0]-1);
        }
        for (int i = 0; i < N; i++)
        {
            int colors[5]={0};
            for (int j : connections[i])
            {
                // 与i相连的节点j最多有3个
                colors[ans[j]]=1;
                // 表示颜色colors[ans[j]]已经被用了
            }
            for (int color = 1; color <= 4; color++)
            {
                if(!colors[color]){
                    // 颜色color尚未被与i相连的节点使用
                    ans[i]=color;
                    break;
                }
            }
        }
        return ans;
    }
    ```

- [1049](https://leetcode.com/problems/last-stone-weight-ii/)

    本题需要把数组stones分为两部分$A,B$使得$min(abs(sum(A)-sum(B)))$，是经典的$0,1$背包问题。

    ```cpp
    int lastStoneWeightII(vector<int>& stones) {
        bitset<1501> dp{1};
        int sum_stones=0,ans=numeric_limits<int>::max();
        // dp[x]表示可以在stones中选择一部分数使其和为x
        for(auto const &it:stones){
            for (int i = 1500; i >= it; i--)
            {
                dp[i]=dp[i]|dp[i-it];
            }
            sum_stones+=it;
        }
        for (int i = 0; i <= 1500; i++)
        {
            ans=min(ans,abs(sum_stones-dp[i]*2*i));
        }
        return ans;
    }
    ```
    
- [1114](https://leetcode.com/problems/print-in-order/)

    注意cpp中的多线程与线程锁机制

- [1115](https://leetcode.com/problems/print-foobar-alternately/)

    cpp中的多线程机制，线程锁的使用，mutex包和mutex，基本格式如下，两个线程A和B交替使用，则：

    ```cpp
    mutex a,b;
    b.lock();
    A{
        a.lock();
        something;
        b.unlock();
    }
    B{
        b.lock();
        something;
        a.unlock();
    }
    ```

    本题解法如下：

    ```cpp
    class FooBar {
        private:
            int n;
            mutex m1,m2;

        public:
            FooBar(int n) {
                this->n = n;
                m2.lock();
            }

            void foo(function<void()> printFoo) {
                for (int i = 0; i < n; i++) {
                    // printFoo() outputs "foo". Do not change or remove this line.
                    m1.lock();
                    printFoo();
                    m2.unlock();
                }
            }

            void bar(function<void()> printBar) {
                for (int i = 0; i < n; i++)
                    // printBar() outputs "bar". Do not change or remove this line.
                    m2.lock();
                    printBar();
                    m1.unlock();
                }
            }
    };
    ```

- [1128](https://leetcode.com/problems/number-of-equivalent-domino-pairs/)

    - brute force[$O(n^2)$]

    ```cpp
    int numEquivDominoPairs(vector<vector<int>>& dominoes) {
        int const count=dominoes.size();
        for (int i = 0; i < count; i++)
        {
            if(dominoes[i][0]>dominoes[i][1]){
                swap(dominoes[i][0],dominoes[i][1]);
            }
        }
        sort(dominoes.begin(),dominoes.end(),[](vector<int>& x,vector<int>&y)->bool{return x[0]<y[0];});
        int ans=0;
        for (int i = 0; i < count; i++)
        {
            int j=i+1;
            while(j<count && dominoes[i][0]==dominoes[j][0]){
                if(dominoes[i][1]==dominoes[j][1]){
                    ans++;
                }
                j++;
            }
        }
        return ans;
    }
    ```

    - encoding method[$O(n)$]

    ```cpp
    int numEquivDominoPairs(vector<vector<int>>& dominoes) {
        int ans=0;
        // 1 <= dominoes[i][j] <= 9
        vector<int> count(100,0);
        for (int i = 0; i < dominoes.size(); i++)
        {
            count[10*max(dominoes[i][0],dominoes[i][1])+min(dominoes[i][0],dominoes[i][1])]++;
        }
        for (int i = 0; i < count.size(); i++)
        {
            if(count[i]!=0){
                ans+=count[i]*(count[i]-1)/2;
            }
        }
        return ans;
    }
    ```

- [1230](https://leetcode.com/problems/toss-strange-coins/)

    有n个硬币，第i个随机扔起正面朝上的概率为$prob_i$，求将n个硬币全部随机扔起之后有target个正面朝上的概率。本题属于典型的动态规划DP类型，递推公式为$$dp_{ij}=dp_{i-1,j-1}*prob_i+dp_{i-1,j}*(1-prob_i)$$，其中$dp_{ij}$表示随机扔起i个硬币后有j个朝上的概率，即随机扔起i个后有j个朝上有两种情况，一是随机扔起$i-1$个有$j-1$个朝上并且第j个也朝上，二是随机扔起$i-1$个有$j$个朝上并且第j个朝下。

    ```cpp
    double probabilityOfHeads(vector<double>& prob, int target) {
        int const count=prob.size();
        vector<double> dp(target+1,0);
        dp[0]=1.0;
        for (int i = 0; i < count; i++)
        {
            for (int j = min(i+1,target); j > 0; j--)
            {
                dp[j]=dp[j-1]*prob[i]+dp[j]*(1-prob[i]);
            }
            dp[0]*=1-prob[i];
        }
        return dp.back();
    }
    ```

- [1245](https://leetcode.com/problems/tree-diameter/)

    给定一颗树，即N叉树，求树的直径，即树中任意两各节点之间的最远距离，通过两遍BFS来求解，首先BFS从任意节点start出发求得最远节点end，然后第二遍BFS从节点end出发求得最远节点last，end到last之间的距离即为树的直径。

    ```cpp
    int depth_BFS(int start, vector<vector<int>> &from_i_to_node, int &end)
    {
        int depth = 0;
        vector<int> cur_level{start};
        vector<bool> visited(from_i_to_node.size(), false);
        end = start;
        while (true)
        {
            vector<int> next_level;
            for (auto &&node : cur_level)
            {
                if (!visited[node])
                {
                    next_level.insert(next_level.end(), from_i_to_node[node].begin(), from_i_to_node[node].end());
                    if (!from_i_to_node[node].empty())
                    {
                        end = from_i_to_node[node].back();
                    }
                    visited[node] = true;
                }
            }
            if (next_level.empty())
            {
                break;
            }
            else
            {
                depth++;
                cur_level = next_level;
            }
        }
        return depth;
    }
    int treeDiameter(vector<vector<int>> &edges)
    {
        int ans = 0;
        const int count = edges.size();
        if (count > 0)
        {
            // 第一次bfs，从任意节点start出发找到最远节点end
            int start = 0, end = 0;
            // sort(edges.begin(), edges.end(), [](vector<int> &a, vector<int> &b) -> bool { return a[0] < b[0]; });
            vector<vector<int>> from_i_to_node(count + 1, vector<int>{});
            for (int i = 0; i < count; i++)
            {
                from_i_to_node[edges[i][0]].push_back(edges[i][1]);
                from_i_to_node[edges[i][1]].push_back(edges[i][0]);
            }
            int depth = depth_BFS(start, from_i_to_node, end);
            // 第二次bfs，从end出发找到最远节点last，end到last之间的距离即为树的直径 diameter
            ans = depth_BFS(end, from_i_to_node, end);
        }
        return ans;
    }
    ```

- [1249](https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/)

    在一个包含英文字母和左右小括号的字符串中，移除最少数量的左右括号，使得字符串中的左右括号相匹配，使用栈来实现即可。

    ```cpp
    string minRemoveToMakeValid(string s)
    {
        stack<int> st; // record index of left parenthese
        vector<int> removed_index;
        for (int i = 0; i < s.length(); i++)
        {
            if(s[i]=='('){
                st.push(i);
            }else if(s[i]==')'){
                if(st.empty()){
                    removed_index.push_back(i);
                }else{
                    st.pop();
                }
            }
        }
        while(!st.empty()){
            removed_index.push_back(st.top());
            st.pop();
        }
        sort(removed_index.begin(), removed_index.end());
        string ans;
        for (int i = 0, j = 0; i < s.length(); i++)
        {
            if(j<removed_index.size() && i==removed_index[j]){
                j++;
            }else{
                ans.push_back(s[i]);
            }
        }
        return ans;
    }
    ```

- [1250](https://leetcode.com/problems/check-if-it-is-a-good-array/)

    数论中存在一个裴蜀定理[wikipad](https://www.wikidata.org/wiki/Q513028)，简单讲就是两个整数$x,y$互质的充要条件为存在整数$a,b$使得$a*x+b*y=1$，因此本题给定一维数组$x$，如果它们的最大公约数（GCD）为1，则它们可以采用合适的权重$a$实现$\sum_{i=1}^{n}a_i*x_i=1$，即只需验证给定数组中是否存在一组互质的数即可。

    ```cpp
    int gcd(int a,int b){
        if(a<b){
            swap(a, b);
        }
        while(b){
            int r = a % b;
            a = b;
            b = r;
        }
        return a;
    }
    bool isGoodArray(vector<int> &nums)
    {
        bool ans = false;
        if(nums.size()>0){
            int v = nums.front();
            for (int i = 1; i < nums.size(); i++)
            {
                v = gcd(v, nums[i]);
            }
            if(v==1){
                ans = true;
            }
        }
        return ans;
    }
    ```

- [...](123)
