# record about problems in [newcoder](https://www.nowcoder.com)

## algorithms

- [1]  [内心里的一把火](https://ac.nowcoder.com/acm/contest/289/D)
  
  题目要求：给定三角形三个顶点的坐标(x1,y1)，(x2,y2)，(x3,y3)，判断给定的一个点(x,y)是否在三角形内部

  主要算法原理：[向量叉乘的几何意义](https://blog.csdn.net/sixdaycoder/article/details/79791389)
  
  [叉乘](https://en.wikipedia.org/wiki/Cross_product)是判断一个点在一条直线那一侧的几何工具，通过该方法可以判断一个点是否在给定直线的左侧，亦即确定了三个点sep是否按照逆时针序排列

  ```
  typedef struct
    {
        float x;
        float y;
    }Point;

    //input  : 直线s->e,点p
    //return : 
    //> 0 p is left of the line
    //= 0 p is on the line
    //< 0 p is right of the line

    bool IsLeftPoint(Point s, Point e, Point p)
    {
        return ((e.x - s.x) * (p.y - s.y)) > ((p.x - s.x) * (e.y - s.y));
        // 其中xy均指代向量坐标
    }
    ```

    进一步延伸，如果逆时针走过三角形的三个顶点ABC构成的三个向量AB,BC,CA,在此过程中点P都在三个向量的左边，则点P在三角形ABC内部，否则在外部。
    
    ### 第二种不适合计算机的方法，等面积法
    如果一个点在三角形内部，那么它与三角形三个顶点的连线构成的三个小三角形的面积和等于大三角形的面积，知道相关点的坐标后所有的面积都可以通过海伦公式
    $$S=\sqrt{p*(p-a)*(p-a)*(p-c)},\ where \  p = \frac{a+b+c}{2}$$
    计算，其中a、b、c分别是三角形三条边的面积。

    此方法在逻辑上是严格正确的，但是不适合计算机使用。主要是因为计算过程中很多浮点运算在计算机中会因为double类型的精度问题而出现计算偏差导致判断失误。

- [2] [快速幂的计算](https://ac.nowcoder.com/acm/contest/322/A)
  
  快速幂$base^n$的计算应该通过不断取平方的办法在$O(log(n))$时间复杂度内完成。
  ```cpp
  int quick_pow_recursive(long base, int power, int m)
    {
        // recursive
        if (power == 0)
            return 1;
        long long  tmp = quick_pow_recursive(base, power / 2, m);
        if (power & 1)
            tmp = (((tmp * tmp) % m) * (base % m)) % m;
        else
            tmp = (tmp * tmp) % m;
        return tmp;
    }

    int quick_pow_nonrecursive(long base,int power,int m){
        long long ans=1;
        while(power){
            if(power&1)
                ans=(ans*base)%m;
            base=(base*base)%m;
            power=power>>1;
        }
        return ans;
    }
  ```
## mathematics

