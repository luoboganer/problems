<!--
 * @Filename: 
 * @Author: shifaqiang
 * @Email: 14061115@buaa.edu.cn
 * @Github: https://github.com/luoboganer
 * @Date: 2020-09-05 11:29:59
 * @LastEditors: shifaqiang
 * @LastEditTime: 2020-09-09 10:27:31
 * @Software: Visual Studio Code
 * @Description: 剑指Offer:名企面试官精讲典型编程题
-->

# 剑指Offer:名企面试官精讲典型编程题

- [剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

    - 使用**lower_bound()/upper_bound()**库函数

    ```cpp
	int search(vector<int> &nums, int target)
	{
		auto left = lower_bound(nums.begin(), nums.end(), target);
		auto right = upper_bound(nums.begin(), nums.end(), target);
		return right - left;
	}
    ```

    - 手动实现有限制条件（最左/最右）的二分查找

    ```cpp
    class Solution
    {
    private:
        int binary_search(vector<int> &nums, int target, int mode)
        {
            /**
            * mode = 0 'most_left', lower
            * mode = 1 'most_right', upper
            */
            int left = 0, right = nums.size();
            while (left < right)
            {
                int mid = left + (right - left) / 2;
                if (nums[mid] > target || (mode == 0 && nums[mid] == target))
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

    public:
        int search(vector<int> &nums, int target)
        {
            int left = binary_search(nums, target, 0);
            int right = binary_search(nums, target, 1);
            return right - left;
        }
    };
    ```

- [...](123)
