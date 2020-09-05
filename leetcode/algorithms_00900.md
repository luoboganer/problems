# 801-900

- [817. Linked List Components](https://leetcode.com/problems/linked-list-components/)

    用数组标记G中的所有数为true，然后遍历链表head，对head中相邻的两个节点cur和cur->next，如果值的标记都是flag，则将cur值的标记改为false，最后统计标记中有多少个true即可

    ```cpp
    int numComponents(ListNode *head, vector<int> &G)
    {
        vector<bool> flags(10001, false);
        for (auto &&v : G)
        {
            flags[v] = true;
        }
        if (head && head->next)
        {
            ListNode *next = head->next;
            while (next)
            {
                if (flags[head->val] && flags[next->val])
                {
                    flags[head->val] = false;
                }
                head = head->next, next = next->next;
            }
        }
        int ret = 0;
        for(auto &&v:flags){
            if(v){
                ret++;
            }
        }
        return ret;
    }
    ```

- [825. Friends Of Appropriate Ages](https://leetcode.com/problems/friends-of-appropriate-ages/)

    - 二重循环暴力遍历，时间复杂度$O(n^2)$，leetcode评测机$\color{red}{TLE}$

    ```cpp
	int numFriendRequests(vector<int> &ages)
	{
		int ret = 0, count = ages.size();
		for (auto i = 0; i < count; i++)
		{
			for (auto j = 0; j < count; j++)
			{
				if (i != j && !(ages[j] <= 0.5 * ages[i] + 7 || ages[j] > ages[i] || (ages[j] > 100 && ages[i] < 100)))
				{
					ret++;
				}
			}
		}
		return ret;
	}
	```

	- 不考虑遍历每一个人ages，而是遍历每一个元组(age,count)，对于每一个ageA和ageB,在符合条件的情况可以发送的请求数为$countA*countB$，在每一个age内部可以发送的请求数为$countA*(countA-1)$，时间复杂度为$O(A^2+N)$，其中A为可能的年龄数，N为全部的人数

	```cpp
	int numFriendRequests(vector<int> &ages)
	{
		int ret = 0, max_age = 120;
		vector<int> age2count(max_age + 1, 0);
		for (auto &&age : ages)
		{
			age2count[age]++;
		}
		for (auto i = 0; i <= max_age; i++)
		{
			for (auto j = 0; j <= i; j++)
			{
				if (!(j <= 0.5 * i + 7 || (j > 100 && i < 100)))
				{
					ret += age2count[i] * age2count[j];
					if (i == j)
					{
						// 去除相同年龄内部重复计算的部分
						ret -= age2count[i];
					}
				}
			}
		}
		return ret;
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

- [859. Buddy Strings](https://leetcode.com/problems/buddy-strings/)

    给定两个字符串A和B，判断交换A中任意两个字符是否可以使得$A.compare(B)==0$，分成A和B相同、A和B不相同两种情况讨论，时间复杂度$O(n)$

    ```cpp
    bool buddyStrings(string A, string B)
    {
        bool ret = false;
        if (A.length() == B.length())
        {
            vector<int> mis_match_index;
            for (int i = 0; i < A.length() && mis_match_index.size() <= 2; i++)
            {
                if (A[i] != B[i])
                {
                    mis_match_index.push_back(i);
                }
            }
            if (mis_match_index.size() == 2)
            {
                // 存在两个不一样的，交换之后相同
                if (A[mis_match_index[0]] == B[mis_match_index[1]] && A[mis_match_index[1]] == B[mis_match_index[0]])
                {
                    ret = true;
                }
            }
            else if (mis_match_index.empty())
            {
                // A和B在未交换前已经完全一样
                vector<int> count(26, 0);
                for (auto &&ch : A)
                {
                    count[(int)(ch - 'a')]++;
                    if (count[(int)(ch - 'a')] >= 2)
                    {
                        ret = true;
                        break;
                    }
                }
                for (int i = 0; i < 26; i++)
                {
                    if (count[i] >= 2)
                    {
                        ret = true;
                        break;
                    }
                }
            }
        }
        return ret;
    }
    ```

- [861. Score After Flipping Matrix](https://leetcode.com/problems/score-after-flipping-matrix/)

    贪心原则，尽量保证高位是1，则首先按行翻转抱着最高位全部为1，然后按列翻转保证每列0不多于1，时间复杂度$O(N)$，tow pass for A，其中N为给定矩阵A中全部元素的数量

    ```cpp
    int matrixScore(vector<vector<int>> &A)
    {
        int ret = 0;
        if (A.size() > 0 && A[0].size() > 0)
        {
            int rows = A.size(), cols = A[0].size();
            // 保证最高位全部是1
            for (int i = 0; i < rows; i++)
            {
                if (A[i][0] == 0)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        A[i][j] = 1 - A[i][j];
                    }
                }
            }
            ret = rows * (1 << (cols - 1));
            // 保证每一列1比0多
            for (int j = 1; j < cols; j++)
            {
                int number_of_one = 0;
                for (int i = 0; i < rows; i++)
                {
                    number_of_one += A[i][j];
                }
                ret += max(number_of_one, rows - number_of_one) * (1 << (cols - 1 - j));
            }
        }
        return ret;
    }
    ```

- [874](https://leetcode.com/problems/walking-robot-simulation/)

    用坐标$(0,1)-north,(1,0)-east,(0,-1)-north,(-1,0)-west$来表示四个方向模拟机器人行走过程即可，注意题目要求返回的是机器人在行走过程中距离原点的最远距离，而不是机器人结束行走后距离原点的最终距离。

- [875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)

    本题的题意为在$\sum_{i=0}^{n-1}\left \lceil \frac{piles[i]}{h}\right \rceil \le H$限制下求$min(h)$，符合题意的h值范围应该在区间$[1,max(piles)]$中，二分搜索即可，时间复杂度$O(nlog(m))$，其中$n=piles.length,m=max(piles)$

    ```cpp
    int minEatingSpeed(vector<int> &piles, int H)
    {
        int left = 1, right = *max_element(piles.begin(), piles.end());
        while (left < right)
        {
            int mid = left + ((right - left) >> 1), hours = 0;
            for (auto &&v : piles)
            {
                hours += (int)(ceil(v * 1.0 / mid));
            }
            if (hours <= H)
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

- [885. Spiral Matrix III](https://leetcode.com/problems/spiral-matrix-iii/)

    回字形填充，拒绝所有超出$R*C$矩阵范围的坐标即可，最坏情况下的时间复杂度$O(4*max(R,C)^2)$

    ```cpp
    vector<vector<int>> spiralMatrixIII(int R, int C, int r0, int c0)
    {
        vector<vector<int>> cords;
        int v = 0, total = R * C;
        int round = 0;
        cords.push_back({r0, c0}), v++;
        vector<int> directions{1, 0, -1, 0, 1};
        while (v < total)
        {
            r0--, c0++, round++;
            int i = r0, j = c0, distance = round * 2;
            for (int d = 0; d < 4; d++)
            {
                int r = directions[d], c = directions[d + 1];
                for (int k = 0; v < total && k < distance; k++)
                {
                    i += r, j += c;
                    if (i >= 0 && j >= 0 && i < R && j < C)
                    {
                        cords.push_back({i, j}), v++;
                    }
                }
            }
        }
        return cords;
    }
    ```

- [886. Possible Bipartition](https://leetcode.com/problems/possible-bipartition/)
    
    二部图染色问题，时间复杂度$O(V+E)$，与[785](https://leetcode.com/problems/is-graph-bipartite/)解法相同

	```cpp
	bool possibleBipartition(int N, vector<vector<int>> &dislikes)
	{
		// 建立邻接矩阵
		vector<vector<int>> graph(N + 1, vector<int>(N + 1, 0));
		for (auto &&dislike : dislikes)
		{
			graph[dislike[0]][dislike[1]] = 1;
			graph[dislike[1]][dislike[0]] = 1;
			// dislike[0]和dislike[1]互相不喜欢，他们之间构成一条边将两个节点分开
		}
		// 二部图染色
		vector<int> colors(N + 1, 0); // 0 unknow, 1 to A, -1 to B
		for (auto i = 1; i <= N; i++)
		{
			if (colors[i] == 0)
			{
				colors[i] = 1; // 给node i染色
				queue<int> qe{{i}};
				while (!qe.empty())
				{
					int cur_node = qe.front();
					qe.pop();
					// 将当点节点cur_node不喜欢/有边相连的节点均染为相反颜色
					for (auto j = 1; j <= N; j++)
					{
						if (graph[cur_node][j] == 1)
						{
							if (colors[j] == colors[cur_node])
							{
								return false;
							}
							if (colors[j] == 0)
							{
								colors[j] = -colors[cur_node];
								qe.push(j);
							}
						}
					}
				}
			}
		}
		return true;
	}
	```

- [890. Find and Replace Pattern](https://leetcode.com/problems/find-and-replace-pattern/)

    在pattern的字母和word的字母之间建立一一映射关系即可

    ```cpp
    vector<string> findAndReplacePattern(vector<string> &words, string pattern)
    {
        vector<string> ret;
        int length = pattern.length();
        for (auto &&s : words)
        {
            if (length == s.length())
            {
                // 建立s与pattern之间的一一映射
                unordered_map<char, char> map;
                vector<int> used(26, 0);
                bool flag = true;
                for (int i = 0; flag && i < length; i++)
                {
                    if (map.find(pattern[i]) != map.end())
                    {
                        // 已经有从pattern[i]出发的映射，则保证s[i]与映射值相同
                        if (map[pattern[i]] != s[i])
                        {
                            flag = false;
                        }
                    }
                    else
                    {
                        if (used[(int)(s[i] - 'a')] == 1)
                        {
                            flag = false; // 作为被映射值s[i]已经被使用了
                        }
                        else
                        {
                            used[(int)(s[i] - 'a')] = 1;
                            map[pattern[i]] = s[i]; // 设置从pattern[i]映射到s[i]
                        }
                    }
                }
                if (flag)
                {
                    ret.push_back(s);
                }
            }
        }
        return ret;
    }
    ```

- [894. All Possible Full Binary Trees](https://leetcode.com/problems/all-possible-full-binary-trees/)

    输出具有N个节点的所有完全二叉树，首先N必须是奇数，然后动态地递归生成，leetcode时间效率$\color{red}{112ms,94\%}$，其中动态递归的状态转移方程为：
    $$FBT(N) =[root],\left\{\begin{matrix}
    root.val=0 \\
    root.left=FBT(left) \\
    root.right=FBT(right)
    \end{matrix}\right. \forall left,\left\{\begin{matrix}1 \le left < N-1\\ left+right=N-1\end{matrix}\right.$$

    ```cpp
    TreeNode *clone(TreeNode *root)
    {
        TreeNode *ans = nullptr;
        if (root)
        {
            ans = new TreeNode(0);
            ans->left = clone(root->left), ans->right = clone(root->right);
        }
        return ans;
    }
    vector<TreeNode *> allPossibleFBT(int N)
    {
        vector<vector<TreeNode *>> dp;
        vector<TreeNode *> ret;
        if (N > 0 && (N & 0x1))
        {
            TreeNode *node = new TreeNode(0);
            dp.push_back(vector<TreeNode *>{node}); // first node
            for (int v = 3; v <= N; v += 2)
            {
                vector<TreeNode *> cur;
                for (int left = 1, right = v - 2; left < v && right >= 1; left += 2, right -= 2)
                {
                    for (auto &&left_node : dp[left / 2])
                    {
                        for (auto &&right_node : dp[right / 2])
                        {
                            TreeNode *cur_node = new TreeNode(0);
                            cur_node->left = clone(left_node), cur_node->right = clone(right_node);
                            cur.push_back(cur_node);
                        }
                    }
                }
                dp.push_back(cur);
            }
            ret = dp.back();
        }
        return ret;
    }
    ```

    由于树的链接结构，实际上子树的复制clone过程可以省略，直接用链接指针指向正确的子树即可，leetcode时间效率$\color{red}{88ms,98\%}$

    ```cpp
    vector<TreeNode *> allPossibleFBT(int N)
    {
        vector<vector<TreeNode *>> dp;
        vector<TreeNode *> ret;
        if (N > 0 && (N & 0x1))
        {
            TreeNode *node = new TreeNode(0);
            dp.push_back(vector<TreeNode *>{node}); // first node
            for (int v = 3; v <= N; v += 2)
            {
                vector<TreeNode *> cur;
                for (int left = 1, right = v - 2; left < v && right >= 1; left += 2, right -= 2)
                {
                    for (auto &&left_node : dp[left / 2])
                    {
                        for (auto &&right_node : dp[right / 2])
                        {
                            TreeNode *cur_node = new TreeNode(0);
                            cur_node->left = left_node, cur_node->right = right_node;
                            cur.push_back(cur_node);
                        }
                    }
                }
                dp.push_back(cur);
            }
            ret = dp.back();
        }
        return ret;
    }
    ```

- [896](https://leetcode.com/problems/monotonic-array/)
    判断一个数列是否单调，单调包含单调递增和单调递减，非严格单调还包含相等的情况
    - two pass，第一遍扫描判断是否全部 <= ，第二遍扫描判断是否全部 >=，两次结果取或关系
    - one pass，一遍扫描过程中用${-1,0,1}$分别表示<,=,>三种状态，然后在第二次出现非零元素的情况下，如果和第一次非零元素不同，即可返回false

- [900. RLE Iterator](https://leetcode.com/problems/rle-iterator/)

    稀疏编码数组的迭代器iterator

    ```cpp
    class RLEIterator
    {
    private:
        vector<int> records;
        int cur_index, length;

    public:
        RLEIterator(vector<int> &A)
        {
            records = A;
            cur_index = 0, length = A.size();
        }

        int next(int n)
        {
            while (cur_index < length && n > records[cur_index])
            {
                n -= records[cur_index];
                cur_index += 2;
            }
            int ret = -1;
            if (cur_index < length)
            {
                ret = records[cur_index + 1];
                records[cur_index] -= n;
            }
            return ret;
        }
    };
    ```