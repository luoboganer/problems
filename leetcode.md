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

        - 暴力
            从1开始逐个检查所有正整数序列是否为ugly数直到统计量count达到n
        - dynamic program
            除1外，下一个ugly数必然是2,3,5的倍数(倍率分别从1开始，每使用一次倍率增长一次)中较小的一个

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

    注意分组统计的办法，类似和二进制表示中1的数量有关的问题和2的倍数有关系

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
    $$\begin{array}{l}{n^{2}=1+3+5+\ldots+(2 \cdot n-1)=\sum_{i=1}^{n}(2 \cdot i-1)} \\ {\text { 证明如下： }} \\ {\quad 1+3+5+\ldots+(2 \cdot n-1)} \\ {=(2 \cdot 1-1)+(2 \cdot 2-1)+(2 \cdot 3-1)+\ldots+(2 \cdot n-1)} \\ {=2 \cdot(1+2+3+\ldots+n)-(\underbrace{1+1+\ldots+1}_{n \text { times }})} \\ {=2 \cdot \frac{n(n+1)}{2}-n} \\ {=n^{2}+n-n} \\ {=n^{2}}\end{array}$$

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

- [442](https://leetcode.com/problems/find-all-duplicates-in-an-array/)
    
    在一个数组$a_n,1 \le a_i \le n$中找出出现了两次的数字（其他数字只出现了一次），要求不占用额外的内存空间、$O(n)$时间复杂度
    - 逐位负数标记法
    - 交换法
- [448](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)

    本题和[442](https://leetcode.com/problems/find-all-duplicates-in-an-array/)很像，在遍历数组的过程中可以简单的用一个bool数组来标记每个下标是否出现即可，在不使用额外空间的情况下，可以用正负标记来代替true和false的bool标记在原数组中标记，只不过每次读取原数组的时候取绝对值即可。

- [509](https://leetcode.com/problems/fibonacci-number/)
    
    斐波那契数列
    - 递归法
    - 非递归循环 faster

- [561](https://leetcode.com/problems/array-partition-i/)
    
    2n个给定范围的数据划分成n组使得每组最小值求和最大，基本思路是对所有数排序后对基数位置上的数求和即可，这里类似于NMS非极大值抑制抑制的思路，主要的时间复杂度在排序上。
    - 基本想法是quick sort，时间复杂度$O(nlog(n))$
    - 本题给出了数据范围，可以bucket sort，时间复杂度$O(n)$，但是需要$O(N)$的额外空间

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

- [733](https://leetcode.com/problems/flood-fill/)
    
    类似于图像处理中区域增长的方式，采用DFS递归写法或者用栈stack实现

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

- [852](https://leetcode.com/problems/peak-index-in-a-mountain-array/)

    寻找给定数组中的山峰数下标，所谓山峰数根据题目定义即为**全局最大值**，因此binary search是理论上的最佳算法，time complexity O(log(n))

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
    - 栈实现，时间复杂度$O(n)$，遍历数组中的每个元素item，找到第一个比item小的数p，然后把item挂到p的右孩子，因此需要维护一个从栈底熬栈顶递减的栈序列，从而为数组迭代过程中的每个数item找到第一个比它小的数p，即栈顶元素；如果栈中不存在p比item小，则item当前最小，成为栈顶元素的左孩子。
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

 - [1030](https://leetcode.com/problems/next-greater-node-in-linked-list/)

    用栈可以实现$O(n)$时间复杂度，即对数组从右往左遍历的过程中保持栈顶st[i]<st[i-1]，从栈底到栈顶是严格递增的顺序

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