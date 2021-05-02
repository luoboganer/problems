<!--
 * @Filename: 
 * @Author: shifaqiang
 * @Email: 14061115@buaa.edu.cn
 * @Github: https://github.com/luoboganer
 * @Date: 2020-09-05 11:29:59
 * @LastEditors: shifaqiang
 * @LastEditTime: 2021-04-26 15:19:18
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

- [1813. 句子相似性 III](https://leetcode-cn.com/problems/sentence-similarity-iii/)

	将句子转化为单词序列后，从左右两侧逼近，时间复杂度$O(n)$

	```cpp
	class Solution
	{
	private:
		vector<string> stringToTokens(string sentence, char delim)
		{
			vector<string> ret;
			if (sentence.size() > 0)
			{
				sentence.push_back(delim); // for the last token
				string token;
				for (auto &&ch : sentence)
				{
					if (ch == delim)
					{
						if (!token.empty())
						{
							ret.push_back(token);
							token.clear();
						}
					}
					else
					{
						token.push_back(ch);
					}
				}
			}
			return ret;
		}
		bool match(vector<string> &a, vector<string> &b)
		{
			if (a.size() < b.size())
			{
				return match(b, a);
			}
			int left = 0, right1 = a.size() - 1, right2 = b.size() - 1;
			int right = min(right1, right2);
			while (left <= right && a[left].compare(b[left]) == 0)
			{
				left++;
			}
			while (left <= right && a[right1].compare(b[right2]) == 0)
			{
				right1--, right2--;
				right--;
			}
			return left > right2;
		}

	public:
		bool areSentencesSimilar(string sentence1, string sentence2)
		{
			vector<string> words1 = stringToTokens(sentence1, ' ');
			vector<string> words2 = stringToTokens(sentence2, ' ');
			return match(words1, words2);
		}
	};
	```

- [1817. 查找用户活跃分钟数](https://leetcode-cn.com/problems/finding-the-users-active-minutes/)

	排序+数组扫描统计，时间复杂度$O(nlog(n))$

	```cpp
	vector<int> findingUsersActiveMinutes(vector<vector<int>> &logs, int k)
	{
		vector<int> answer(k, 0);
		const int n = logs.size();
		sort(logs.begin(), logs.end(), [](const auto &a, const auto &b) -> bool { return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]); });
		logs.emplace_back(vector<int>{logs.back()[0] + 1, -1}); // 哨兵
		for (int i = 1, count = 1; i <= n; i++)
		{
			if (logs[i][0] != logs[i - 1][0])
			{
				// 非同一个用户
				answer[count - 1]++;
				count = 1;
			}
			else
			{
				if (logs[i][1] != logs[i - 1][1])
				{
					count++;
				}
			}
		}
		return answer;
	}
	```

- [1837. K 进制表示下的各位数字总和](https://leetcode-cn.com/problems/sum-of-digits-in-base-k/)

	进制转换，短除法

	```cpp
	int sumBase(int n, int k)
	{
		int ret = 0;
		while (n)
		{
			ret += n % k;
			n /= k;
		}
		return ret;
	}
	```

- [...](123)
