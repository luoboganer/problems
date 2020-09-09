<!--
 * @Filename: 
 * @Author: shifaqiang
 * @Email: 14061115@buaa.edu.cn
 * @Github: https://github.com/luoboganer
 * @Date: 2020-09-05 11:29:59
 * @LastEditors: shifaqiang
 * @LastEditTime: 2020-09-09 11:03:51
 * @Software: Visual Studio Code
 * @Description: 程序员面试金典
-->

# 程序员面试金典

- [面试题 01.06. 字符串压缩](https://leetcode-cn.com/problems/compress-string-lcci/)

    ```cpp
	string compressString(string S)
	{
		string ret;
		int count = 0;
		for (auto &&ch : S)
		{
			if (ret.empty() || ch != ret.back())
			{
				if (count > 0)
				{
					ret += to_string(count);
				}
				ret.push_back(ch);
				count = 1;
			}
			else
			{
				count++;
			}
		}
		if (count > 0)
		{
			ret += to_string(count);
		}
		return ret.length() < S.length() ? ret : S;
	}
    ```

- [面试题 02.06. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list-lcci/)

    - 转化为数组或者使用栈，然后检查是否回味，时间复杂度$O(n)$，空间复杂度$O(n)$

    ```cpp
	bool isPalindrome(ListNode *head)
	{
		vector<int> nums;
		while (head)
		{
			nums.push_back(head->val);
			head = head->next;
		}
		int left = 0, right = nums.size() - 1;
		while (left < right)
		{
			if (nums[left++] != nums[right--])
			{
				return false;
			}
		}
		return true;
	}
    ```

    - 使用快慢双指针找到链表的中间节点，然后翻转后半部分链表，然后对比前半部分和翻转后的后半部分是否完全一致，时间复杂度$O(n)$，空间复杂度优化到$O(1)$

    ```cpp
    class Solution
    {
    private:
        ListNode *find_middle_node(ListNode *head)
        {
            if (head)
            {
                ListNode *slow = head, *fast = head;
                while (fast && fast->next)
                {
                    slow = slow->next, fast = fast->next->next;
                }
                return fast ? slow->next : slow;
            }
            return head;
        }
        ListNode *reverse_list(ListNode *head)
        {
            if (head)
            {
                ListNode *pre = nullptr, *cur = head;
                while (cur)
                {
                    ListNode *temp = cur->next;
                    cur->next = pre;
                    pre = cur;
                    cur = temp;
                }
                return pre;
            }
            return head;
        }

    public:
        bool isPalindrome(ListNode *head)
        {
            bool ret = true;
            if (head)
            {
                ListNode *first = head, *second = reverse_list(find_middle_node(head));
                while (first && second)
                {
                    if (first->val != second->val)
                    {
                        ret = false;
                        break;
                    }
                    first = first->next, second = second->next;
                }
            }
            return ret;
        }
    };
    ```

    - some test cases

    ```cpp
    [1,2]
    [1,2,3,2,1]
    [1,2,3,3,2,1]
    [1,2,3,4,2,1]
    []
    [1]
    ```

- [面试题 04.05. 合法二叉搜索树](https://leetcode-cn.com/problems/legal-binary-search-tree-lcci/)

    判断一棵树是否为合法的二叉搜索树（binary search tree，BST）

    ```cpp
    class Solution
    {
    private:
        bool dfs(TreeNode *root, long long min_value, long long max_value)
        {
            bool ret = true;
            if (root)
            {
                long long v = root->val;
                ret = v < max_value && v > min_value && dfs(root->left, min_value, v) && dfs(root->right, v, max_value);
            }
            return ret;
        }

    public:
        bool isValidBST(TreeNode *root)
        {
            return dfs(root, numeric_limits<long long>::min(), numeric_limits<long long>::max());
        }
    };
    ```

- [...](123)
