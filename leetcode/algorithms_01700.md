<!--
 * @Filename: 
 * @Author: shifaqiang
 * @Email: 14061115@buaa.edu.cn
 * @Github: https://github.com/luoboganer
 * @Date: 2020-09-05 11:29:59
 * @LastEditors: shifaqiang
 * @LastEditTime: 2020-10-04 13:34:25
 * @Software: Visual Studio Code
 * @Description: 1601-1700
-->

# 1601-1700

- [1610. Maximum Number of Visible Points](https://leetcode.com/problems/maximum-number-of-visible-points/)

    以人的位置为新的坐标原点，首先统计和人重合的点，然后将其余的点按照极角的大小排序，再以双指针的形式用滑动窗口扫描给定的角度范围内的最大点数即可，需要注意的点有：
    
    - 因为视角是可以循环的，因此算出所有极角并排序后，再将所有角度加360度后加入序列后面
    - 在计算角度的过程中用arctan函数不需要考虑值域范围最方便

    具体的cpp实现版本如下：

    - first version

    ```cpp
    class Solution
    {
    private:
        double pi = 3.141592653, err = 1e-5;
        double distance(const vector<int> &a, const vector<int> &b)
        {
            return sqrt(((1.0 * a[0] - b[0]) * (a[0] - b[0]) + (1.0 * a[1] - b[1]) * (a[1] - b[1])));
        }
        double compute_angle(const vector<int> &C, const vector<int> &B)
        {
            // B和C在同一条水平线上
            if (B[1] == C[1])
            {
                return C[0] > B[0] ? 180.0 : 0;
            }
            // c 原始的固定点，b是给定点
            double cos_theta = (B[0] - C[0]) / distance(B, C);
            double theta = acos(cos_theta) * 180 / pi;	// [0,2*pi) 并转化为角度值
            return (B[1] < C[1]) ? 360 - theta : theta; // 值域从[0,pi] 转化为[0,2*pi]
        }

    public:
        int visiblePoints(vector<vector<int>> &points, int angle, vector<int> &location)
        {
            int ret = 0, original = 0;
            vector<double> angles;
            for (auto &p : points)
            {
                if (p == location)
                {
                    original++;
                }
                else
                {
                    angles.push_back(compute_angle(location, p));
                }
            }
            sort(angles.begin(), angles.end());
            int left = 0, right = 0, n = angles.size();
            for (int i = 0; i < n; i++)
            {
                angles.emplace_back(angles[i] + 360); // 视角在旋转的过程中可以循环
            }
            n *= 2;
            while (left < n && right < n)
            {
                while (right < n && angles[right] - angles[left] <= angle + err)
                {
                    right++;
                }
                ret = max(right - left, ret);
                if (right < n && angles[right] - angles[left] > angle - err)
                {
                    left++;
                }
            }
            return ret + original;
        }
    };
    ```

    - second version

    ```cpp
	int visiblePoints(vector<vector<int>> &points, int angle, vector<int> &location)
	{
		int ret = 0, original = 0;
		double pi = 3.141592653, eps = 1e-8;
		vector<double> angles;
		double x = location[0], y = location[1];
		for (auto &p : points)
		{
			if (p == location)
			{
				original++;
			}
			else
			{
				angles.emplace_back(atan2(p[1] - y, p[0] - x) * 180 / pi);
			}
		}
		sort(angles.begin(), angles.end());
		int left = 0, right = 0, n = angles.size();
		for (int i = 0; i < n; i++)
		{
			angles.emplace_back(angles[i] + 360); // 视角在旋转的过程中可以循环
		}
		n *= 2;
		while (left < n && right < n)
		{
			while (right < n && angles[right] - angles[left] <= angle + eps)
			{
				right++;
			}
			ret = max(right - left, ret);
			if (right < n && angles[right] - angles[left] > angle - eps)
			{
				left++;
			}
		}
		return ret + original;
	}
    ```

- [...](123)
