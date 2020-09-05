# 401-500

- [401](https://leetcode.com/problems/binary-watch/)

    有些问题要用逆向思维来解决，本题本来是要用给定二进制位中1的个数来拼凑出可能的时间表示，组合数过程不好写，可以反过来写所有$00:00 - 23:59$中所有可能的时间中二进制位中1的个数符合题目要求的时间点。

- [402. Remove K Digits](https://leetcode.com/problems/remove-k-digits/)

    用栈的思想在目标字符串中维护一个字符的字典序非严格升序（严格非降序）序列，即贪心地尽可能使高位的数字小，如果当前数字小于更高位，则在k允许的范围内移除更高位，在此过程中每个字符最多入栈、弹栈被处理两次，时间复杂度$O(n),n=num.length$，num为给定的数字字符串

    需要注意的点有：
    - 完成非严格升序栈后，可移除指标k仍然未用完，则从最低位开始移除，因为此时最低位数字值最大，移除之后先前的高位会顺次降为低位，且数字值更低
    - k个移除指标用完后，num的后续部分无需处理直接拼接到结果串中即可
    - 完成移除后要删除数字中可能出现的前导0
    - 如果结果串为空要输出字符'0'

    ```cpp
    string removeKdigits(string num, int k)
    {
        string ret;
        int i = 0, length = num.length();
        while (k > 0 && i < length)
        {
            while (!ret.empty() && k > 0 && num[i] < ret.back())
            {
                ret.pop_back(), k--;
            }
            ret.push_back(num[i]), i++;
        }
        if (i < length)
        {
            // k个可移除指标已经用完，num中剩余的部分直接拼接到ret尾部
            ret.insert(ret.end(), num.begin() + i, num.end());
        }
        while (k > 0 && !ret.empty())
        {
            k--, ret.pop_back(); // num已经遍历完，k个指标尚未用完，在ret中从后往前删除
        }
        i = 0, length = ret.length(); // 删除前导零
        while (i < length && ret[i] == '0')
        {
            i++;
        }
        ret = string(ret.begin() + i, ret.end());
        if (ret.empty())
        {
            ret.push_back('0');
        }
        return ret;
    }
    ```

- [419. Battleships in a Board](https://leetcode.com/problems/battleships-in-a-board/)

    - 每次发现battleship则统计值count自增，并DFS标记其周边所有相连的X均为同一个battleship，LeetCode时间效率$\color{red}{12ms,12.8\%}$

    ```CPP
    void mark_dfs(vector<vector<char>> &board, int i, int j)
    {
        if (i >= 0 && j >= 0 && i < board.size() && j < board[0].size() && board[i][j] == 'X')
        {
            board[i][j] = '.';
            vector<int> directions{1, 0, -1, 0, 1};
            for (int k = 0; k < 4; k++)
            {
                mark_dfs(board, i + directions[k], j + directions[k + 1]);
            }
        }
    }
    int countBattleships(vector<vector<char>> &board)
    {
        int ret = 0;
        if (board.size() > 0 && board[0].size() > 0)
        {
            int rows = board.size(), cols = board[0].size();
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (board[i][j] == 'X')
                    {
                        ret++;
                        mark_dfs(board, i, j);
                    }
                }
            }
        }
        return ret;
    }
    ```

    - 因为battleship只能成横线或者竖线$(1*n 或 n*1)$，因此从left-top到right-bottom扫描的过程中重复统计只可能来自上侧或者左侧，规避这种重复计数接口，时间复杂度$O(N)$，其中N为board中总的节点数，即one pass+ $O(1)$ extra memory + without modifying the value of the board，LeetCode时间效率$\color{red}{8ms,71.21\%}$

    ```cpp
    int countBattleships(vector<vector<char>> &board)
    {
        int ret = 0;
        if (board.size() > 0 && board[0].size() > 0)
        {
            int rows = board.size(), cols = board[0].size();
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (board[i][j] == 'X')
                    {
                        if (i > 0 && board[i - 1][j] == 'X')
                        {
                            continue; // 上方已经统计过
                        }
                        if (j > 0 && board[i][j - 1] == 'X')
                        {
                            continue; // 左侧已经统计过
                        }
                        ret++;
                    }
                }
            }
        }
        return ret;
    }
    ```
    
- [427. Construct Quad Tree](https://leetcode.com/problems/construct-quad-tree/)

    递归construct

    ```cpp
    Node *construct_rec(vector<vector<int>> &grid, int i, int j, int n)
    {
        Node *ret = nullptr;
        if (n == 1)
        {
            ret = new Node(grid[i][j], true, nullptr, nullptr, nullptr, nullptr);
        }
        else
        {
            int width = n / 2;
            Node *tl = construct_rec(grid, i, j, width);
            Node *tr = construct_rec(grid, i, j + width, width);
            Node *bl = construct_rec(grid, i + width, j, width);
            Node *br = construct_rec(grid, i + width, j + width, width);
            if (tl->isLeaf && tr->isLeaf && bl->isLeaf && br->isLeaf && tl->val == tr->val && bl->val == br->val && tl->val == bl->val)
            {
                ret = new Node(tl->val, true, nullptr, nullptr, nullptr, nullptr);
            }
            else
            {
                ret = new Node(grid[i][j], false, tl, tr, bl, br);
            }
        }
        return ret;
    }
    Node *construct(vector<vector<int>> &grid)
    {
        Node *root = nullptr;
        if (grid.size() > 0)
        {
            int n = grid.size();
            root = construct_rec(grid, 0, 0, n);
        }
        return root;
    }
    ```

- [430. Flatten a Multilevel Doubly Linked List](https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/)

    链表操作，拉平一个多级列表

    ```cpp
    Node *flatten(Node *head)
    {
        if (head)
        {
            stack<Node *> st;
            Node *pre = nullptr, *cur = head;
            while (cur || !st.empty())
            {
                if (cur)
                {
                    if (cur->child)
                    {
                        if(cur->next){
                            st.push(cur->next);
                        }
                        cur->next = cur->child;
                        cur->next->prev = cur;
                        cur->child = nullptr;
                    }
                    pre = cur, cur = cur->next;
                }
                else
                {
                    cur = pre, cur->next = st.top(), st.top()->prev = cur, cur = cur->next, st.pop();
                }
            }
        }
        return head;
    }
    ```

- [437. Path Sum III](https://leetcode.com/problems/path-sum-iii/)

    在一颗二叉树上任意节点为七点，统计向下的和为指定sum的路径数，首先统计以root为起点的和为sum的路径数，然后递归地统计以root的左右子树为起点的和为sum的路径数量

    ```cpp
    int helper(TreeNode *root, int pre, int sum)
    {
        int ret = 0;
        if (root)
        {
            pre += root->val;
            if (pre == sum)
            {
                ret = 1;
            }
            ret += helper(root->left, pre, sum) + helper(root->right, pre, sum);
        }
        return ret;
    }
    int pathSum(TreeNode *root, int sum)
    {
        int ret = 0;
        if (root)
        {
            ret = pathSum(root->left, sum) + pathSum(root->right, sum) + helper(root, 0, sum);
        }
        return ret;
    }
    ```

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

- [454. 4Sum II](https://leetcode.com/problems/4sum-ii/)

    - 四个数组，两两组合求和形成两个数组，然后再遍历两个数组，时间复杂度$O(n^4)$，LeetCode提交$\color{red}{TLE}$

    ```cpp
    int fourSumCount(vector<int> &A, vector<int> &B, vector<int> &C, vector<int> &D)
    {
        int count = 0, length = A.size(), length_2 = length * length;
        vector<int> a(length_2, 0), b(length_2, 0);
        for (int i = 0, cur = 0; i < length; i++)
        {
            for (int j = 0; j < length; j++)
            {
                a[cur] = A[i] + B[j], b[cur] = C[i] + D[j];
                cur++;
            }
        }
        for (int i = 0; i < length_2; i++)
        {
            for (int j = 0; j < length_2; j++)
            {
                if (a[i] + b[j] == 0)
                {
                    count++;
                }
            }
        }
        return count;
    }
    ```

    - 使用hashmap存储两两组合求和形成的结果，时间复杂度$O(n^2)$，实际上hashmap在查询的过程中$O(1)$操作也很耗时，LeetCode提交$172ms,72.52\%$

    ```cpp
    int fourSumCount(vector<int> &A, vector<int> &B, vector<int> &C, vector<int> &D)
    {
        int count = 0, length = A.size();
        unordered_map<int, int> record;
        for (int i = 0, v = 0; i < length; i++)
        {
            for (int j = 0; j < length; j++)
            {
                v = A[i] + B[j];
                auto it = record.find(v);
                if (it != record.end())
                {
                    it->second++;
                }
                else
                {
                    record[v] = 1;
                }
            }
        }
        for (int i = 0, v = 0; i < length; i++)
        {
            for (int j = 0; j < length; j++)
            {
                v = C[i] + D[j];
                auto it = record.find(-v);
                if (it != record.end())
                {
                    count += it->second;
                }
            }
        }
        return count;
    }
    ```

- [459. Repeated Substring Pattern](https://leetcode.com/problems/repeated-substring-pattern/)

    判断一个字符串s是否是由字符串t经过$[t_1,t_2,...,t_n],n\ge2$构成

    - 使用KMP，时间复杂度$O(n)$
    - 将字符串s分割为k段($k \in [2,s.length]$)，检查每段是否相同，时间复杂度$O(n^3)$

    ```cpp
    bool repeatedSubstringPattern(string s)
    {
        int const length = s.length();
        if (length > 1)
        {
            int count = (length >> 1);
            for (int length_of_substring = count; length_of_substring >= 1; length_of_substring--)
            {
                if (length % length_of_substring == 0)
                {
                    int count_of_substring = length / length_of_substring;
                    string base = s.substr(0, length_of_substring);
                    bool flag = true;
                    for (int j = 1; flag && j < count_of_substring; j++)
                    {
                        if (base.compare(s.substr(length_of_substring * j, length_of_substring)) != 0)
                        {
                            flag = false;
                            break;
                        }
                    }
                    if (flag)
                    {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    ```

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

- [470. Implement Rand10() Using Rand7()](https://leetcode.com/problems/implement-rand10-using-rand7/)

    拒绝采样即可

    ```cpp
    int rand10()
    {
        int ret;
        do
        {
            ret = rand7() + (rand7() - 1) * 7;
        } while (ret>40);
        return 1 + (ret - 1) % 10;
    }
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

- [477. Total Hamming Distance](https://leetcode.com/problems/total-hamming-distance/)

    - 二重遍历求异或值后统计bit位中1的个数，暴力计算，时间复杂度$O(n^2)$，显然遇到了$\color{red}{TLE}$

    ```cpp
    int count_bits(int v)
    {
        int ret = 0;
        while (v)
        {
            v = v & (v - 1);
            ret++;
        }
        return ret;
    }
    int totalHammingDistance(vector<int> &nums)
    {
        int ret = 0, count = nums.size();
        for (int i = 0; i < count; i++)
        {
            for (int j = i + 1; j < count; j++)
            {
                ret += count_bits(nums[i] ^ nums[j]);
            }
        }
        return ret;
    }
    ```

    - 统计所有数中在每一位置上1的个数，然后计算每个bit位0和1的组合数之和，时间复杂度$O(n)$

    ```cpp
    int totalHammingDistance(vector<int> &nums)
    {
        const int length = 32;
        vector<int> count(length, 0);
        for (auto &&v : nums)
        {
            for (int i = 0; i < length; i++)
            {
                if (v & 0x1)
                {
                    count[i]++;
                }
                v >>= 1;
            }
        }
        int ret = 0, total_numbers = nums.size();
        for (int i = 0; i < length; i++)
        {
            ret += count[i] * (total_numbers - count[i]);
        }
        return ret;
    }
    ```

- [478. Generate Random Point in a Circle](https://leetcode.com/problems/generate-random-point-in-a-circle/)

    在给定圆心坐标和半径的圆内，生成随机点的坐标，这里特别注意cpp中随机函数的应用，srand()函数设置随机数种子，rand()函数生成[0,RAND_MAX]之间的均匀分布随机数

    - 圆内随机点的坐标可以通过随机生成r和theta两个极坐标参数来确定

    ```cpp
    #define random() ((double)(rand())/RAND_MAX)
    class Solution {
    private:
        double r,x,y;
        const double PI=3.14159265358979732384626433832795;
    public:
        Solution(double radius, double x_center, double y_center) {
            r=radius,x=x_center,y=y_center;
        }
        vector<double> randPoint() {
            double r0=sqrt(random()); // 在二维空间内[0,r^2]才是均匀采样
            double theta=2*PI*random();
            return vector<double>{x+r0*r*cos(theta),y+r0*r*sin(theta)};
        }
    };
    ```
    
    - 拒绝采样法

    ```cpp
    #define random() ((double)(rand())/RAND_MAX)
    class Solution {
    private:
        double r,x,y;
    public:
        Solution(double radius, double x_center, double y_center) {
            r=radius,x=x_center,y=y_center;
        }
        vector<double> randPoint() {
            double x0,y0;
            do{
                x0=random()*2-1,y0=random()*2-1;
            }while(x0*x0+y0*y0>1);
            return {x+x0*r,y+y0*r};
        }
    };
    ```

- [497. Random Point in Non-overlapping Rectangles](https://leetcode.com/problems/random-point-in-non-overlapping-rectangles/)

	- 借用概率密度函数的特性，solution时间复杂度$O(n)$，pick时间复杂度$O(log(n))$

	**注意uppter_bound()函数的使用**

	```cpp
	class Solution
	{
	private:
		int total_area;
		vector<int> weights;
		vector<vector<int>> rects;

	public:
		Solution(vector<vector<int>> &rects)
		{
			this->rects = rects;
			for (auto &&rect : rects)
			{
				int width = rect[2] - rect[0] + 1, height = rect[3] - rect[1] + 1;
				weights.push_back(width * height);
			}
			int n = weights.size();
			for (int i = 1; i < n; i++)
			{
				weights[i] += weights[i - 1];
			}
			total_area = weights.back();
		}

		vector<int> pick()
		{
			int v = rand() % total_area;
			int index = upper_bound(weights.begin(), weights.end(), v) - weights.begin();
			vector<int> rect = rects[index];
			int width = rect[2] - rect[0] + 1, height = rect[3] - rect[1] + 1;
			int x = rand() % width + rect[0], y = rand() % height + rect[1];
			return vector<int>{x, y};
		}
	};
	/**
	* Your Solution object will be instantiated and called as such:
	* Solution* obj = new Solution(rects);
	* vector<int> param_1 = obj->pick();
	*/
	```