<!--
 * @Filename: 
 * @Author: shifaqiang
 * @Email: 14061115@buaa.edu.cn
 * @Github: https://github.com/luoboganer
 * @Date: 2020-09-05 11:29:59
 * @LastEditors: shifaqiang
 * @LastEditTime: 2021-03-28 21:52:04
 * @Software: Visual Studio Code
 * @Description: 1801-1900
-->

# 1801-1900

- [1805. 字符串中不同整数的数目](https://leetcode-cn.com/problems/number-of-different-integers-in-a-string/)

    字符串解析，特别注意前导0的处理，时间复杂度$O(n)$

    ```cpp
	int numDifferentIntegers(string word)
	{
		unordered_set<string> nums;
		const int n = word.length();
		for (int i = 0; i < n;)
		{
			if (isdigit(word[i]))
			{
				string cur;
				while (i < n && isdigit(word[i]))
				{
					if (cur.length() > 0 || word[i] != '0')
					{
						cur += word[i];
					}
					i++;
				}
				if (cur.empty())
				{
					cur += '0';
				}
				nums.insert(cur);
			}
			else
			{
				i++;
			}
		}
        return nums.size();
	}
    ```

- [1807. 替换字符串中的括号内容](https://leetcode-cn.com/problems/evaluate-the-bracket-pairs-of-a-string/)

    滑动窗口解析括号，hashmap映射key/value关系，时间复杂度$O(m+n)$，其中$n=s.length(),m=knowledge.size()$

    ```cpp
	string evaluate(string s, vector<vector<string>> &knowledge)
	{
		unordered_map<string, string> knowledge_map;
		for (auto &item : knowledge)
		{
			knowledge_map[item[0]] = item[1];
		}
		string ret;
		const int n = s.length();
		int i = 0;
		while (i < n)
		{
			if (s[i] != '(')
			{
				ret += s[i++];
			}
			else
			{
				string keyword;
				i++; // 跳过左侧括号
				while (i < n && s[i] != ')')
				{
					keyword += s[i++];
				}
				auto it = knowledge_map.find(keyword);
				ret += it == knowledge_map.end() ? "?" : it->second;
				i++; // 跳过右侧括号
			}
		}
		return ret;
	}
    ```

- [...](123)
