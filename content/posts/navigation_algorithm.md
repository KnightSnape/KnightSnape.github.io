---
author: "KnightSnape"
title: "Navigation Algorithm"
date: "2024-1-22"
description: "The total result of navigation algorithm"
tags: ["navigation"]
aliases: ["migrate-from-jekyl"]
ShowToc: true
TocOpen: true
weight: 2
---

# 导航使用的算法

* [1.基本算法](#1-基本算法)

    * [1.1.DFS](#11-DFS)

    * [1.2.图的遍历](#12-图的遍历)

    * [1.3.最短路问题](#13-最短路问题)

* [2.代价地图](#2-代价地图)

* [3.拓展导航算法](#3-拓展导航算法)

    * [3.1.动态规划](#31-动态规划)

    * [3.2.遗传算法，蚁群算法](#32-遗传算法，蚁群算法)

    * [3.3.模拟退火](#33-模拟退火)

* [4.路径平滑算法](#4-路径平滑算法)

    * [4.1.多项式插值](#41-多项式插值)

    * [4.2.贝塞尔曲线](#42-贝塞尔曲线)

    * [4.3.三次样条曲线,B样条曲线](#43-三次样条曲线,B样条曲线)

    * [4.4.特殊曲线](#44-特殊曲线)

* [5. 空间采样算法](#5-空间采样算法)

    * [5.1.K近邻算法](#51K近邻算法)

    * [5.2.PRM算法](#52PRM算法)

    * [5.3.RRT算法](#53RRT算法)

    * [5.4.RRT-Connect](#54RRT-Connect)

    * [5.5.RRT*算法](#55RRT*算法)

# 1.基本算法

## 1.1.DFS

$\quad$ 搜索，也就是对状态空间进行枚举，通过穷尽所有的可能来找到最优解,或者统计合法解的个数。

$\quad$ 搜索有很多优化方式，如减小状态空间，更改搜索顺序，剪枝等等。

$\quad$ DFS为图论的概念。在搜索算法中，该词常常利用递归函数方便实现暴力枚举的算法。

$\quad$ 常见使用的可能方案，对应的框架如下:

```cpp
void dfs(int state)
{
    if(find)
    {
        return;
    }
    //可以有不同的遍历策略
    dfs(next(state));
}

```

$\quad$ 常见的实例如下：把正整数$n$分解成3个不同的正整数，如6=1+2+3。排在后面的数必须大于前面的数，输出所有的方案数。

可以考虑循环：

```cpp
for (int i = 1; i <= n; ++i)
  for (int j = i; j <= n; ++j)
    for (int k = j; k <= n; ++k)
      if (i + j + k == n) printf("%d = %d + %d + %d\n", n, i, j, k);
```

$\quad$ 可以考虑如下方案：设一组方案将正整数$n$分解成$k$个正整数$a_1,a_2,...,a_k$的和。

$\quad$ 我们将问题分层，第$i$层决定$a_i$。则为了进行第$i$层决策，我们需要记录三个状态变量$n - \sum_{j=1}^{i}{a_j}$，表示后面所有正整数的和；以及$a_{i-1}$，表示前一层的正整数，以确保正整数底层，以及$i$，确保我们最多输出$m$个正整数。

$\quad$ 代码如下：

```cpp
int m, arr[103];  // arr 用于记录方案

void dfs(int n, int i, int a) {
  if (n == 0) {
    for (int j = 1; j <= i - 1; ++j) printf("%d ", arr[j]);
    printf("\n");
  }
  if (i <= m) {
    for (int j = a; j <= n; ++j) {
      arr[i] = j;
      dfs(n - j, i + 1, j);  // 请仔细思考该行含义。
    }
  }
}

// 主函数
scanf("%d%d", &n, &m);
dfs(n, 1, 1);
```

## 1.2.图的遍历

**图 (graph)** 是一个二元组 $G=(V(G), E(G))$。其中 $V(G)$ 是非空集，称为 **点集 (vertex set)**，对于 $V$ 中的每个元素，我们称其为 **顶点 (vertex)** 或 **节点 (node)**，简称 **点**；$E(G)$ 为 $V(G)$ 各结点之间边的集合，称为 **边集 (edge set)**。

常用 $G=(V,E)$ 表示图。

当 $V,E$ 都是有限集合时，称 $G$ 为 **有限图**。

当 $V$ 或 $E$ 是无限集合时，称 $G$ 为 **无限图**。

图有多种，包括 **无向图 (undirected graph)**，**有向图 (directed graph)**，**混合图 (mixed graph)** 等。

若 $G$ 为无向图，则 $E$ 中的每个元素为一个无序二元组 $(u, v)$，称作 **无向边 (undirected edge)**，简称 **边 (edge)**，其中 $u, v \in V$。设 $e = (u, v)$，则 $u$ 和 $v$ 称为 $e$ 的 **端点 (endpoint)**。

若 $G$ 为有向图，则 $E$ 中的每一个元素为一个有序二元组 $(u, v)$，有时也写作 $u \to v$，称作 **有向边 (directed edge)** 或 **弧 (arc)**，在不引起混淆的情况下也可以称作 **边 (edge)**。设 $e = u \to v$，则此时 $u$ 称为 $e$ 的 **起点 (tail)**，$v$ 称为 $e$ 的 **终点 (head)**，起点和终点也称为 $e$ 的 **端点 (endpoint)**。并称 $u$ 是 $v$ 的直接前驱，$v$ 是 $u$ 的直接后继。

???+ note "为什么起点是 tail，终点是 head？"
    边通常用箭头表示，而箭头是从「尾」指向「头」的。

若 $G$ 为混合图，则 $E$ 中既有 **有向边**，又有 **无向边**。

若 $G$ 的每条边 $e_k=(u_k,v_k)$ 都被赋予一个数作为该边的 **权**，则称 $G$ 为 **赋权图**。如果这些权都是正实数，就称 $G$ 为 **正权图**。

图 $G$ 的点数 $\left| V(G) \right|$ 也被称作图 $G$ 的 **阶 (order)**。

形象地说，图是由若干点以及连接点与点的边构成的。

**途径 (walk)**：途径是连接一连串顶点的边的序列，可以为有限或无限长度。形式化地说，一条有限途径 $w$ 是一个边的序列 $e_1, e_2, \ldots, e_k$，使得存在一个顶点序列 $v_0, v_1, \ldots, v_k$ 满足 $e_i = (v_{i-1}, v_i)$，其中 $i \in [1, k]$。这样的途径可以简写为 $v_0 \to v_1 \to v_2 \to \cdots \to v_k$。通常来说，边的数量 $k$ 被称作这条途径的 **长度**（如果边是带权的，长度通常指途径上的边权之和，题目中也可能另有定义）。

**迹 (trail)**：对于一条途径 $w$，若 $e_1, e_2, \ldots, e_k$ 两两互不相同，则称 $w$ 是一条迹。

**路径 (path)**（又称 **简单路径 (simple path)**）：对于一条迹 $w$，若其连接的点的序列中点两两不同，则称 $w$ 是一条路径。

**回路 (circuit)**：对于一条迹 $w$，若 $v_0 = v_k$，则称 $w$ 是一条回路。

**环/圈 (cycle)**（又称 **简单回路/简单环 (simple circuit)**）：对于一条回路 $w$，若 $v_0 = v_k$ 是点序列中唯一重复出现的点对，则称 $w$ 是一个环。

关于路径的定义在不同地方可能有所不同，如，「路径」可能指本文中的「途径」，「环」可能指本文中的「回路」。如果在题目中看到类似的词汇，且没有「简单路径」/「非简单路径」（即本文中的「途径」）等特殊说明，最好询问一下具体指什么。

对于一张无向图 $G = (V, E)$，对于 $u, v \in V$，若存在一条途径使得 $v_0 = u, v_k = v$，则称 $u$ 和 $v$ 是 **连通的 (connected)**。由定义，任意一个顶点和自身连通，任意一条边的两个端点连通。

若无向图 $G = (V, E)$，满足其中任意两个顶点均连通，则称 $G$ 是 **连通图 (connected graph)**，$G$ 的这一性质称作 **连通性 (connectivity)**。

若 $H$ 是 $G$ 的一个连通子图，且不存在 $F$ 满足 $H\subsetneq F \subseteq G$ 且 $F$ 为连通图，则 $H$ 是 $G$ 的一个 **连通块/连通分量 (connected component)**（极大连通子图）。

对于一张有向图 $G = (V, E)$，对于 $u, v \in V$，若存在一条途径使得 $v_0 = u, v_k = v$，则称 $u$  **可达**  $v$。由定义，任意一个顶点可达自身，任意一条边的起点可达终点。（无向图中的连通也可以视作双向可达。）

若一张有向图的节点两两互相可达，则称这张图是 **强连通的 (strongly connected)**。

若一张有向图的边替换为无向边后可以得到一张连通图，则称原来这张有向图是 **弱连通的 (weakly connected)**。

与连通分量类似，也有 **弱连通分量 (weakly connected component)**（极大弱连通子图）和 **强连通分量 (strongly connected component)**（极大强连通子图）。

- 图的存储和遍历

### 直接存边

使用一个数组来存边，数组中的每个元素都包含一条边的起点与终点(带边权的图还包含边权)。(或者使用多个数组分别存起点，终点和边权。)

```cpp
#include <iostream>
#include <vector>
    
using namespace std;

struct Edge {
    int u, v;
};

int n, m;
vector<Edge> e;
vector<bool> vis;

bool find_edge(int u, int v) 
{
    for (int i = 1; i <= m; ++i) 
    {
        if (e[i].u == u && e[i].v == v) 
        {
            return true;
        }
    }
    return false;
}

void dfs(int u) 
{
    if (vis[u]) 
        return;
    vis[u] = true;
    for (int i = 1; i <= m; ++i) 
    {
        if (e[i].u == u) 
        {
            dfs(e[i].v);
        }
    }
}

int main() 
{
    cin >> n >> m;

    vis.resize(n + 1, false);
    e.resize(m + 1);

    for (int i = 1; i <= m; ++i) cin >> e[i].u >> e[i].v;

    return 0;
}
```

### 复杂度

查询是否存在某条边：$O(m)$。

遍历一个点的所有出边：$O(m)$。

遍历整张图：$O(nm)$。

空间复杂度：$O(m)$。

### 应用

由于直接存边的遍历效率低下，一般不用于遍历图。

在Kruskal算法 中，由于需要将边按边权排序，需要直接存边。

在有的题目中，需要多次建图（如建一遍原图，建一遍反图），此时既可以使用多个其它数据结构来同时存储多张图，也可以将边直接存下来，需要重新建图时利用直接存下的边来建图。

### 邻接矩阵

使用一个二维数组 `adj` 来存边，其中 `adj[u][v]` 为 1 表示存在 $u$ 到 $v$ 的边，为 0 表示不存在。如果是带边权的图，可以在 `adj[u][v]` 中存储 $u$ 到 $v$ 的边的边权。

```cpp
#include <iostream>
#include <vector>

using namespace std;

int n, m;
vector<bool> vis;
vector<vector<bool> > adj;

bool find_edge(int u, int v) { return adj[u][v]; }

void dfs(int u) 
{
    if (vis[u]) return;
    vis[u] = true;
    for (int v = 1; v <= n; ++v) 
    {
        if (adj[u][v]) 
        {
            dfs(v);
        }
    }
}

int main() 
{
    cin >> n >> m;

    vis.resize(n + 1, false);
    adj.resize(n + 1, vector<bool>(n + 1, false));

    for (int i = 1; i <= m; ++i) 
    {
        int u, v;
        cin >> u >> v;
        adj[u][v] = true;
    }

    return 0;
}
```
### 复杂度

查询是否存在某条边：$O(1)$。

遍历一个点的所有出边：$O(n)$。

遍历整张图：$O(n^2)$。

空间复杂度：$O(n^2)$。

### 应用

邻接矩阵只适用于没有重边（或重边可以忽略）的情况。

其最显著的优点是可以 $O(1)$ 查询一条边是否存在。

由于邻接矩阵在稀疏图上效率很低（尤其是在点数较多的图上，空间无法承受），所以一般只会在稠密图上使用邻接矩阵。

### 邻接表

使用一个支持动态增加元素的数据结构构成的数组，如 `vector<int> adj[n + 1]` 来存边，其中 `adj[u]` 存储的是点 $u$ 的所有出边的相关信息（终点、边权等）。

```cpp
#include <iostream>
#include <vector>

using namespace std;

int n, m;
vector<bool> vis;
vector<vector<int> > adj;

bool find_edge(int u, int v) 
{
    for (int i = 0; i < adj[u].size(); ++i) 
    {
        if (adj[u][i] == v) 
        {
            return true;
        }
    }
    return false;
}

void dfs(int u) 
{
    if (vis[u]) 
    return;
    vis[u] = true;
    for (int i = 0; i < adj[u].size(); ++i) 
    dfs(adj[u][i]);
}

int main() {
    cin >> n >> m;

    vis.resize(n + 1, false);
    adj.resize(n + 1);

    for (int i = 1; i <= m; ++i) 
    {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
    }

    return 0;
}
```

### 复杂度

查询是否存在 $u$ 到 $v$ 的边：$O(d^+(u))$（如果事先进行了排序就可以使用 [二分查找](../basic/binary.md) 做到 $O(\log(d^+(u)))$）。

遍历点 $u$ 的所有出边：$O(d^+(u))$。

遍历整张图：$O(n+m)$。

空间复杂度：$O(m)$。

### 链式前向星

```cpp
// head[u] 和 cnt 的初始值都为 -1
void add(int u, int v) {
    nxt[++cnt] = head[u];  // 当前边的后继
    head[u] = cnt;         // 起点 u 的第一条边
    to[cnt] = v;           // 当前边的终点
}

// 遍历 u 的出边
for (int i = head[u]; ~i; i = nxt[i]) {  // ~i 表示 i != -1
    int v = to[i];
}
```
参考示例

```cpp
#include <iostream>
#include <vector>

using namespace std;

int n, m;
vector<bool> vis;
vector<int> head, nxt, to;

void add(int u, int v) {
    nxt.push_back(head[u]);
    head[u] = to.size();
    to.push_back(v);
}

bool find_edge(int u, int v) {
    for (int i = head[u]; ~i; i = nxt[i]) {  // ~i 表示 i != -1
    if (to[i] == v) {
        return true;
    }
    }
    return false;
}

void dfs(int u) {
    if (vis[u]) return;
    vis[u] = true;
    for (int i = head[u]; ~i; i = nxt[i]) dfs(to[i]);
}

int main() {
    cin >> n >> m;

    vis.resize(n + 1, false);
    head.resize(n + 1, -1);

    for (int i = 1; i <= m; ++i) {
    int u, v;
    cin >> u >> v;
    add(u, v);
    }

    return 0;
}
```

### 复杂度

查询是否存在 $u$ 到 $v$ 的边：$O(d^+(u))$。

遍历点 $u$ 的所有出边：$O(d^+(u))$。

遍历整张图：$O(n+m)$。

空间复杂度：$O(m)$。

## 1.3.最短路问题

为了方便叙述，这里先给出下文将会用到的一些记号的含义。

-   $n$ 为图上点的数目，$m$ 为图上边的数目；
-   $s$ 为最短路的源点；
-   $D(u)$ 为 $s$ 点到 $u$ 点的 **实际** 最短路长度；
-   $dis(u)$ 为 $s$ 点到 $u$ 点的 **估计** 最短路长度。任何时候都有 $dis(u) \geq D(u)$。特别地，当最短路算法终止时，应有 $dis(u)=D(u)$。
-   $w(u,v)$ 为 $(u,v)$ 这一条边的边权。

### Floyd算法

是用来求任意两个结点之间的最短路的。

复杂度比较高，但是常数小，容易实现。

适用于任何图，不管有向无向，边权正负，但是最短路必须存在。（不能有个负环）

我们定义一个数组 `f[k][x][y]`，表示只允许经过结点 $1$ 到 $k$（也就是说，在子图 $V'={1, 2, \ldots, k}$ 中的路径，注意，$x$ 与 $y$ 不一定在这个子图中），结点 $x$ 到结点 $y$ 的最短路长度。

很显然，`f[n][x][y]` 就是结点 $x$ 到结点 $y$ 的最短路长度（因为 $V'={1, 2, \ldots, n}$ 即为 $V$ 本身，其表示的最短路径就是所求路径）。

接下来考虑如何求出 `f` 数组的值。

`f[0][x][y]`：$x$ 与 $y$ 的边权，或者 $0$，或者 $+\infty$（`f[0][x][y]` 什么时候应该是 $+\infty$？当 $x$ 与 $y$ 间有直接相连的边的时候，为它们的边权；当 $x = y$ 的时候为零，因为到本身的距离为零；当 $x$ 与 $y$ 没有直接相连的边的时候，为 $+\infty$）。

`f[k][x][y] = min(f[k-1][x][y], f[k-1][x][k]+f[k-1][k][y])`（`f[k-1][x][y]`，为不经过 $k$ 点的最短路径，而 `f[k-1][x][k]+f[k-1][k][y]`，为经过了 $k$ 点的最短路）。

上面两行都显然是对的，所以说这个做法空间是 $O(N^3)$，我们需要依次增加问题规模（$k$ 从 $1$ 到 $n$），判断任意两点在当前问题规模下的最短路。

```cpp
for (k = 1; k <= n; k++) 
{
    for (x = 1; x <= n; x++) 
    {
        for (y = 1; y <= n; y++) 
        {
            f[k][x][y] = min(f[k - 1][x][y], f[k - 1][x][k] + f[k - 1][k][y]);
        }
    }
}
```

因为第一维对结果无影响，我们可以发现数组的第一维是可以省略的，于是可以直接改成 `f[x][y] = min(f[x][y], f[x][k]+f[k][y])`。

### Dijkstra 算法

Dijkstra算法由荷兰计算机科学家 E. W. Dijkstra 于 1956 年发现，1959 年公开发表。是一种求解 **非负权图** 上单源最短路径的算法。

将结点分成两个集合：已确定最短路长度的点集（记为 $S$ 集合）的和未确定最短路长度的点集（记为 $T$ 集合）。一开始所有的点都属于 $T$ 集合。

初始化 $dis(s)=0$，其他点的 $dis$ 均为 $+\infty$。

然后重复这些操作：

1.  从 $T$ 集合中，选取一个最短路长度最小的结点，移到 $S$ 集合中。
2.  对那些刚刚被加入 $S$ 集合的结点的所有出边执行松弛操作。

直到 $T$ 集合为空，算法结束。

有多种方法来维护 1 操作中最短路长度最小的结点，不同的实现导致了 Dijkstra 算法时间复杂度上的差异。

-   暴力：不使用任何数据结构进行维护，每次 2 操作执行完毕后，直接在 $T$ 集合中暴力寻找最短路长度最小的结点。2 操作总时间复杂度为 $O(m)$，1 操作总时间复杂度为 $O(n^2)$，全过程的时间复杂度为 $O(n^2 + m) = O(n^2)$。
-   二叉堆：每成功松弛一条边 $(u,v)$，就将 $v$ 插入二叉堆中（如果 $v$ 已经在二叉堆中，直接修改相应元素的权值即可），1 操作直接取堆顶结点即可。共计 $O(m)$ 次二叉堆上的插入（修改）操作，$O(n)$ 次删除堆顶操作，而插入（修改）和删除的时间复杂度均为 $O(\log n)$，时间复杂度为 $O((n+m) \log n) = O(m \log n)$。
-   优先队列：和二叉堆类似，但使用优先队列时，如果同一个点的最短路被更新多次，因为先前更新时插入的元素不能被删除，也不能被修改，只能留在优先队列中，故优先队列内的元素个数是 $O(m)$ 的，时间复杂度为 $O(m \log m)$。
-   Fibonacci 堆：和前面二者类似，但 Fibonacci 堆插入的时间复杂度为 $O(1)$，故时间复杂度为 $O(n \log n + m)$，时间复杂度最优。
-   线段树：和二叉堆原理类似，不过将每次成功松弛后插入二叉堆的操作改为在线段树上执行单点修改，而 1 操作则是线段树上的全局查询最小值。时间复杂度为 $O(m \log n)$。

在稀疏图中，$m = O(n)$，使用二叉堆实现的 Dijkstra 算法较 Bellman–Ford 算法具有较大的效率优势；而在稠密图中，$m = O(n^2)$，这时候使用暴力做法较二叉堆实现更优。

算法实现：

```cpp
struct edge {
    int v, w;
};

vector<edge> e[maxn];
int dis[maxn], vis[maxn];

void dijkstra(int n, int s) 
{
    memset(dis, 63, sizeof(dis));
    dis[s] = 0;
    for (int i = 1; i <= n; i++) 
    {
        int u = 0, mind = 0x3f3f3f3f;
        for (int j = 1; j <= n; j++)
            if (!vis[j] && dis[j] < mind) 
            u = j, mind = dis[j];
        vis[u] = true;
        for (auto ed : e[u])
        {
            int v = ed.v, w = ed.w;
            if (dis[v] > dis[u] + w) 
            dis[v] = dis[u] + w;
        }
    }
}
```

队列优化的dijkstra:

```cpp
struct edge {
    int v, w;
};

struct node {
    int dis, u;
    //重载
    bool operator>(const node& a) const { return dis > a.dis; }
};

vector<edge> e[maxn];
int dis[maxn], vis[maxn];
priority_queue<node, vector<node>, greater<node> > q;

void dijkstra(int n, int s) 
{
    memset(dis, 63, sizeof(dis));
    dis[s] = 0;
    q.push({0, s});
    while (!q.empty()) 
    {
        int u = q.top().u;
        q.pop();
        if (vis[u]) continue;
        vis[u] = 1;
        for (auto ed : e[u]) 
        {
            int v = ed.v, w = ed.w;
            if (dis[v] > dis[u] + w) 
            {
                dis[v] = dis[u] + w;
                q.push({dis[v], v});
            }
        }
    }
}
```

## 不同方法的比较

| 最短路算法   | Floyd      | Bellman–Ford | Dijkstra     | Johnson       |
| ------- | ---------- | ------------ | ------------ | ------------- |
| 最短路类型   | 每对结点之间的最短路 | 单源最短路        | 单源最短路        | 每对结点之间的最短路    |
| 作用于     | 任意图        | 任意图          | 非负权图         | 任意图           |
| 能否检测负环？ | 能          | 能            | 不能           | 能             |
| 时间复杂度   | $O(N^3)$   | $O(NM)$      | $O(M\log M)$ | $O(NM\log M)$ |



# 2.代价地图


# 3.拓展导航算法

## 3.1.动态规划

## 3.2.遗传算法，蚁群算法

$\quad$ 遗传算法(Genetic Algorithm，简称GA) 起源于对生物系统所进行的计算机模拟研究，是一种随机全局搜索优化方法，它模拟了自然选择和遗传中发生的复制、交叉(crossover)和变异(mutation)等现象，从任一初始种群（Population）出发，通过随机选择、交叉和变异操作，产生一群更适合环境的个体，使群体进化到搜索空间中越来越好的区域，这样一代一代不断繁衍进化，最后收敛到一群最适应环境的个体（Individual），从而求得问题的优质解。

### 遗传算法术语

1. 染色体：染色体又可称为基因型个体，一定数量的个体组成了群体,群体中个体的数量叫做群体大小。

2. 位串：个体的表示形式。对应于遗传学中的染色体。

3. 基因：基因是染色体中的元素，用于表示个体的特征。

4. 特征值：在用串表示整数时，基因的特征值与二进制数的权一致；例如在串S=1011中，基因位置3中的1，它的基因特征值为2，基因位置1中的1，它的基因特征值为8。

5. 适应度：各个个体对环境中的适应程度叫做适应度。为了体现染色体的适应能力，引入了对问题中的每一个染色体都能进行度量的函数，叫适应度函数。这个函数通常会被用来计算个体在群体中被使用的概率。

6. 基因型：或称遗传型，是指基因组定义遗传特征和表现。对于于GA中的位串。

7. 表现型：生物体的基因型在特性环境下的表现特征。对应于GA中的位串解码的参数

### 遗传算法基本过程

$\quad$ 基本遗传算法(也称标准遗传算法或简单遗传算法，Simple Genetic Algorithm，简称SGA)是一种群体型操作，该操作以群体中的所有个体为对象，只使用基本遗传算子(Genetic Operator)：选择算子(Selection Operator)、交叉算子(Crossover Operator)和变异算子(Mutation Operator)，其遗传进化操作过程简单，容易理解，是其它一些遗传算法的基础，它不仅给各种遗传算法提供了一个基本框架，同时也具有一定的应用价值。选择、交叉和变异是遗传算法的3个主要操作算子，它们构成了遗传操作，使遗传算法具有了其它方法没有的特点。

其表示方法如下：

$$SGA = (C,E,P_0,M,\phi,\Gamma,\psi,T)$$

其中:

$C$表示个体的编码方案

$E$表示个体适应度评价函数

$P_0$表示初始种群

$M$表示种群大小

$\phi$表示选择算子

$\Gamma$表示交叉算子

$\psi$表示变异算子

$T$表示遗传算法终止条件

### 编码

$\quad$ 解空间中的解在遗传算法中的表示形式。从问题的解(solution)到基因型的映射称为编码，即把一个问题的可行解从其解空间转换到遗传算法的搜索空间的转换方法。遗传算法在进行搜索之前先将解空间的解表示成遗传算法的基因型串(也就是染色体)结构数据，这些串结构数据的不同组合就构成了不同的点。

$\quad$ 常见的编码方法有二进制编码、格雷码编码、 浮点数编码、各参数级联编码、多参数交叉编码等。

$\quad$ 二进制编码：即组成染色体的基因序列是由二进制数表示，具有编码解码简单易用，交叉变异易于程序实现等特点。

$\quad$ 格雷编码：两个相邻的数用格雷码表示，其对应的码位只有一个不相同，从而可以提高算法的局部搜索能力。这是格雷码相比二进制码而言所具备的优势。

$\quad$ 浮点数编码：是指将个体范围映射到对应浮点数区间范围，精度可以随浮点数区间大小而改变。

设某一参数的取值范围为$[U_1,U_2]$，我们可以用长度为$k$的二进制编码符号来表示该参数，则它总共产生$2^k$种不同的编码，可以使用参数编码时的对应关系

$$0000...0000 = 0 \to U_1$$

$$0000...0001 = 1 \to U_1 + \delta$$

$$...$$

$$1111...1111 = 2^k - 1 \to U_2$$

其中，$\delta = \frac{U_2 - U_1}{2^k - 1}$

### 解码

$\quad$ 遗传算法染色体向问题解的转换，假设某一个体的编码，对应的解码公式为:

$$X = U_1 + (\sum_{i=1}^{k}b_i·2^{i-1}) · 
\frac{U_2 - U_1}{2^k - 1}$$

### 初始群体的生成

$\quad$ 设置最大进化代数$T$，群体大小$M$，交叉概率$P_c$，变异概率$P_m$，随机生成$M$个个体作为初始化群体$P_0$。

### 适应度值评估检测

$\quad$ 适应度函数表明个体或解的优劣性。对于不同的问题，适应度函数的定义方式不同。根据具体的问题，计算P(t)各个个体的适应度。

$\quad$ 适应度尺度变换：一般来讲，是指算法迭代的不同阶段，能够通过适当改变个体的适应度大小，进而避免群体间适应度相当而造成的竞争减弱，导致种群收敛于局部最优解。

尺度变换选用的经典方法：线性尺度变换、乘幂尺度变换以及指数尺度变换。

- 线性尺度变换

$$F` = aF + b$$

其中$a$为比例系数，$b$为平移系数，$F$为变换前适应尺度。

- 乘幂尺度变换

$$F` = F^k$$

其中$k$为幂，$F$为转变前适应度尺度。

- 指数尺度变换

$$F` = e^{-\beta F}$$

### 遗传算子

#### 选择

$\quad$ 选择操作从旧群体中以一定概率选择优良个体组成新的种群，以繁殖得到下一代个体。个体被选中的概率跟适应度有关，个体适应度值越高，被选中的概率越大。以轮盘法为例，若设种群数为$M$，个体$i$的适应度为$f_i$，则个体被选取的概率为:

$$P_i = \frac{f_i}{\sum_{k=1}^{M}{f_k}}$$

$\quad$ 当个体选择的概率给定后，产生[0,1]之间均匀随机数来决定哪个个体参加交配。若个体的选择概率大，则有机会被多次选中，那么它的遗传基因就会在种群中扩大；若个体的选择概率小，则被淘汰的可能性会大。

#### 交叉

$\quad$ 交叉操作是指从种群中随机选择两个个体，通过两个染色体的交换组合，把父串的优秀特征遗传给子串，从而产生新的优秀个体。

$\quad$ 在实际应用中，使用率最高的是单点交叉算子，该算子在配对的染色体中随机的选择一个交叉位置，然后在该交叉位置对配对的染色体进行基因位变换。该算子的执行过程如下图所示。

a)双点交叉或多点交叉，即对配对的染色体随机设置两个或者多个交叉点，然后进行交叉运算，改变染色体基因序列。

b)均匀交叉，即配对的染色体基因序列上的每个位置都以等概率进行交叉，以此组成新的基因序列。

c)算术交叉，是指配对染色体之间采用线性组合方式进行交叉，改变染色体基因序列。

#### 变异

$\quad$ 为了防止遗传算法在优化过程中陷入局部最优解，在搜索过程中，需要对个体进行变异，在实际应用中，主要采用单点变异，也叫位变异，即只需要对基因序列中某一个位进行变异，以二进制编码为例，即0变为1，而1变为0。




## 3.3.模拟退火

$\quad$ 模拟退火算法从某一较高初温出发，伴随温度参数的不断下降,结合一定的概率突跳特性在解空间中随机寻找目标函数的全局最优解，即在局部最优解能概率性地跳出并最终趋于全局最优。

$\quad$ 模拟退火算法会结合概率突跳特性在解空间中随机寻找目标函数的全局最优解，那么具体的更新解的机制是什么呢？如果新解比当前解更优，则接受新解，否则基于Metropolis准则判断是否接受新解。接受概率为：

$$P = \begin{cases}
1, \quad E_{t+1} < E_t \\
e^{\frac{-(E_{t+1} - E_t)}{kT}} E_{t+1} \geq E_t
\end{cases}$$

如上公式，假设当前时刻搜索的解为$x_t$，对应的系统能量为$E_t$，对搜索点产生随机扰动，产生新解$x_{t+1}$,相应地，系统能量为$E_{t+1}$，那么系统对搜索点从$x_t$到$x_{t+1}$转变的接受概率为上述公式。

算法实质分两层循环，在任一温度水平下，随机扰动产生新解，并计算目标函数值的变化，决定是否被接受。由于算法初始温度比较高，这样，使$E$增大的新解在初始时也可能被接受，因而能跳出局部极小值，然后通过缓慢地降低温度，算法就最终可能收敛到全局最优解，具体流程为：

1. 令$T = T_0$，表示开始退火的初始温度，随机产生一个初始解$x_0$,并计算对应的目标函数值$E(x_0)$;

2. 令$T = kT,k \in (0,1)$，$k$为温度下降速率

3. 对当前解$x_t$施加随机扰动，在其邻域产生一个新解$x_{t+1}$,并计算目标函数值$E(x_{t+1})$

计算：

$$\Delta E = E(x_{t+1}) - E(x_t)$$

4. 若$\Delta < 0$，接受新解作为当前解，否则按照概率判断是否接受新解

5. 在温度$T$下，重复$L$次扰动和接受过程，即重复3,4

6. 判断温度是否达到终止温度水平，若是则终止算法，否则返回步骤2

# 4.路径平滑算法

## 4.1.多项式插值

$\quad$ 多项式插值的原理可以在数值分析中找到，这里可以展示相应的样例

```cpp
#include<iostream>
#define maxsize 100

using namespace std;
double L_interpolitation(int n,double x[maxsize],double y[maxsize],double vx,double result)
{
    double temp;
    for(int k=0;k<=n;k++)
    {
        temp=1;
        for(int i=0;i<=n;i++)
        {
            if(i!=k)
            {
                temp*=(vx-x[i])/(x[k]-x[i]);
            }
        }
        result += y[k]*temp;
    }
    return result;
}

int main()
{
    int n;
    double x[maxsize],y[maxsize],result=0,vx,temp;
    cin>>n>>vx;
    for(int i=0;i<=n;i++)
    {
        cin >> x[i] >> y[i];
    }
    result=L_interpolitation(n,x,y,vx,result);
    cout<<result<<endl;
    return 0;
}
```

## 4.2.贝塞尔曲线

$\quad$ 对于一阶曲线，我们可以看到是一条直线，通过几何知识，很容易根据t的值得出线段上那个点的坐标：

$$B_1(t) = P_0 + (P_1 - P_0) t,t \in [0,1]$$

$\quad$ 对于二阶，假设$A,B,C$三点不共线，在第一条线段上任选一个点$D$。计算该点到线段起点的距离$AD$，与该线段总长$AB$的比例。

$\quad$ 根据上一步得到的比例，从第二条线段上找出对应的点$E$，使得

$$\frac{AD}{AB} = \frac{BE}{BC}$$

$\quad$ 这时候$DE$又是一条直线了，就可以按照一阶的贝塞尔方程来进行线性插值了，即：$t = AD : AE$

$$B_2(t) = (1 - t)^2P_0 + 2t(1 - t)P_1 + t^2P_2,t \in [0,1]$$

$\quad$ 对于实际的贝塞尔曲线，有两种表达式：

$$P(t) = \sum_{i=0}^{n}{P_i}\tbinom{n}{i}t^i(1-t)^{n-i}[i=0,1,2,...,n]$$

$\quad$ 它本身也可以理解为二项式展开

$\quad$ 而另一种形式为递归形式：

$$B_{i,n}(t) = (1 - t)B_{i,n-1}(t) + tB_{i-1,n-1}(t)[i=0,1,...,n]$$

## 4.3.三次样条曲线,B样条曲线

$\quad$ 样条是一种数据插值的方式，在多项式插值中，多项式是给出的单一公式来尽可能满足所有的数据点，而样条则使用多个公式，每个公式都是低阶多项式，其能够保证通过所有的数据点。

$\quad$ 工程师制图时，把富有弹性的细长木条（所谓样条）用压铁固定在采样点上，在其他地方让它自由弯曲，然后沿木条画下曲线，称为样条曲线。

$\quad$ 在样条两个采样点之间自由弯曲的线段则为曲线段。求解三次样条曲线的本质就是求解两两采样点之间的曲线段表达公式。

$\quad$ 最简单的样条曲线为线性样条曲线，即把两个相邻点连接起来形成直线段，两个相邻点之间画的是线性函数$y = a_ix + b_i$ 

$\quad$ 线性样条可以对任意的$n$个点集进行插值，但是线性样条缺乏平滑性，而稍后介绍的三次样条可以很好的解决这个缺点。

- 三次样条曲线性质

1. 三次样条曲线在纽结处连续

$$S_i(x_i) = y_i,S_i(x_{i+1} = y_{y+1}),其中i=1,2,...,n-1$$

2. 三次样条曲线在纽结处斜率相同

$$S^{(1)}_{i-1}(x_i) = S^{(1)}_i(x_i),其中i=1,2,...,n-1$$

3. 三次样条曲线在纽结处曲率相同

$$S^{(2)}_{i-1}(x_i) = S^{(2)}_i(x_i),其中i=1,2,...,n-1$$

4. 三次自然样条端点条件

$$S^{(3)}_1(x_1) = 0,S^{(3)}_{n-1}(x_n) = 0$$

5. 斜率调整三次样条曲线/钳制三次样条(给定初始速度，终点速度)端点条件。

$$S^{(1)}_1(x_1) = v_1，S^{(1)}_{n-1}(x_n) = v_2$$

三次样条插值法的主要步骤如下：

1. 在每个小区间内拟合出一个三次多项式；
2. 设置边界条件，得到方程组；
3. 解出方程组，求出每个小区间的系数；
4. 给定插值点，利用系数求出对应的函数值。

使用Python代码演示如下：

```python
import numpy as np
import matplotlib.pyplot as plt

def splinecoeff(x,y,v1=0,vn=0):
    n = len(x)
    A = np.zeros((n,n))
    r = np.zeros((n,1))
    dx = np.diff(x)
    dy = np.diff(y)
    for i in range(1,n-1):
        A[i,i-1:i+2] = [dx[i-1],2*(dx[i-1] + dx[i]),dx[i]]
        r[i] = 3*(dy[i]/dx[i]-dy[i-1]/dx[i-1]) 
     A[0, 0] = 1 # 矩阵一行一列，值为1
    A[-1, -1] = 1 # 矩阵末行末列，值为1
    coeff = np.zeros((n, 3))
    coeff[:, 1] = np.linalg.solve(A, r).flatten() # 解矩阵方程，得c_i
    for i in range(n-1): # 代回，解b_i,d_i
        coeff[i, 2] = (coeff[i+1, 1]-coeff[i, 1])/(3*dx[i]) # d_i = (c_i - c_i)/(3*dx)
        coeff[i, 0] = dy[i]/dx[i]-dx[i]*(2*coeff[i, 1]+coeff[i+1, 1])/3 # b_i = dy/dx - (dx/3) * (2*c_i + c_{i+1})
    return coeff[:-1] #coeff[:-1]指代了多项式拟合函数返回的系数coeff中除了最后一个（行）元素以外的所有元素，也就是前n-1个系数

def splineplot(x, y, k):
    n = len(x)
    coeff = splinecoeff(x, y)
    x1 = []
    y1 = []
    for i in range(n-1): # 再代回 s_i(x) = a_i + b_i(x-x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3, x_i < x <x_{i+1} 
        xs = np.linspace(x[i], x[i+1], k+1)
        dx = xs - x[i]
        ys = coeff[i, 2] * dx # 秦九韶算法 录入求值
        ys = (ys + coeff[i, 1]) * dx
        ys = (ys + coeff[i, 0]) * dx + y[i]
        x1.extend(xs[:-1]) # xs 的第一个元素到倒数第二个元素（不包括最后一个元素）添加到列表 x1 的末尾
        y1.extend(ys[:-1])
    x1.append(x[-1]) # x 的最后一个元素添加到列表 x1 的末尾
    y1.append(y[-1])
    plt.plot(x, y, 'o', x1, y1)
    plt.show()


x = np.array([1, 2, 3, 4, 5])
y = np.array([7, 2, 6, 1, 9])
k = 1000
splineplot(x, y, k)


```


使用C++示例如下(调用Eigen)

```cpp
#pragma once
#include <iostream>
#include <Eigen/Dense>
 
//using Eigen::MatrixXd;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;
 
using namespace std;
 
class CubicSpline {
 
private:
 
	vector<vector<double>> point2D;
	vector<vector<double>> CubicSplineParameter;//a, b, c, d.
	vector<double> h;
	vector<double> m;
 
public:
 
	void CubicSpline_init(vector<vector<double>> point2D_input) {
 
		point2D = point2D_input;
 
		//init h
		h.clear();
		h.resize(point2D.size() - 1);
		for (int i = 0; i < point2D.size() - 1; i++) {
			double x1 = point2D[i][0];
			double x2 = point2D[i + 1][0];
			double h_i = abs(x2 - x1);
			h[i] = h_i;
		}
 
		//init m. m.size = point2D.size()
		//1, compute yh coefficient
		vector<double> yh(point2D_input.size());
		for (int i = 0; i < yh.size(); i++) {
			if (i == 0 || i == yh.size() - 1) {
				yh[i] = 0;
			}
			else {
				yh[i] = 6 * ((point2D[i + 1][1] - point2D[i][1]) / h[i] - (point2D[i][1] - point2D[i - 1][1]) / h[i - 1]);
			}
		}
 
		MatrixXf A(point2D.size(), point2D.size());
		MatrixXf B(point2D.size(), 1);
		MatrixXf m;
 
		//2, init A, B
		B(0, 0) = yh[0];
		B(point2D.size() - 1, 0) = yh[point2D.size() - 1];
 
		for (int i = 0; i < point2D.size() - 1; i++) {
 
			A(0, i) = 0;
			A(point2D.size() - 1, i) = 0;
 
		}
		A(0, 0) = 1;
		A(point2D.size() - 1, point2D.size() - 2) = 1;
 
		for (int i = 1; i < point2D.size() - 1; i++) {
 
			B(i, 0) = yh[i];
 
			for (int j = 0; j < point2D.size(); j++) {
 
				if (j == i) {
					A(i, j) = 2 * (h[i - 1] + h[i]);
				}
				else if (j == i - 1) {
					A(i, j) = h[i - 1];
				}
				else if (j == i + 1) {
					A(i, j) = h[i];
				}
				else {
					A(i, j) = 0;
				}
 
			}
 
		}
 
		m = A.llt().solve(B);
		vector<double> mV(point2D.size());
		for (int i = 0; i < point2D.size(); i++) {
			mV[i] = m(i, 0);
		}
 
		for (int i = 0; i < m.size() - 1; i++) {
 
			vector<double> CubicSplineParameter_i;
			double a = point2D[i][1];
			double b = (point2D[i + 1][1] - point2D[i][1]) / h[i] - h[i] / 2 * mV[i] - h[i] / 6 * (mV[i + 1] - mV[i]);
			double c = mV[i] / 2;
			double d = (mV[i + 1] - mV[i]) / (6 * h[i]);
			CubicSplineParameter_i.push_back(a);
			CubicSplineParameter_i.push_back(b);
			CubicSplineParameter_i.push_back(c);
			CubicSplineParameter_i.push_back(d);
			CubicSplineParameter.push_back(CubicSplineParameter_i);
 
		}
 
	}
 
	vector<vector<double>> CubicSpline_Insert(int step) {
 
		vector<vector<double>> insertList;
 
		for (int i = 0; i < CubicSplineParameter.size(); i++) {
			double h_i = h[i] / (double)step;
			insertList.push_back(point2D[i]);
			double a = CubicSplineParameter[i][0];
			double b = CubicSplineParameter[i][1];
			double c = CubicSplineParameter[i][2];
			double d = CubicSplineParameter[i][3];
			for (int j = 1; j < step; j++) {
				double x_new = point2D[i][0] + h_i * j;
				double y_new = a + b * (x_new - point2D[i][0])
					+ c * (x_new - point2D[i][0]) * (x_new - point2D[i][0])
					+ d * (x_new - point2D[i][0]) * (x_new - point2D[i][0]) * (x_new - point2D[i][0]);
				vector<double> p_new_ij;
				p_new_ij.push_back(x_new);
				p_new_ij.push_back(y_new);
				insertList.push_back(p_new_ij);
			}
		}
 
		insertList.push_back(point2D[point2D.size() - 1]);
		return insertList;
 
	}
 
	vector<vector<double>> CubicSpline_Insert(double step) {
 
		vector<vector<double>> insertList;
 
		for (int i = 0; i < CubicSplineParameter.size(); i++) {
			int h_i = h[i] / (double)step;
			insertList.push_back(point2D[i]);
			double a = CubicSplineParameter[i][0];
			double b = CubicSplineParameter[i][1];
			double c = CubicSplineParameter[i][2];
			double d = CubicSplineParameter[i][3];
			for (int j = 1; j < h_i; j++) {
				double x_new = point2D[i][0] + step * j;
				double y_new = a + b * (x_new - point2D[i][0])
					+ c * (x_new - point2D[i][0]) * (x_new - point2D[i][0])
					+ d * (x_new - point2D[i][0]) * (x_new - point2D[i][0]) * (x_new - point2D[i][0]);
				vector<double> p_new_ij;
				p_new_ij.push_back(x_new);
				p_new_ij.push_back(y_new);
				insertList.push_back(p_new_ij);
			}
		}
 
		insertList.push_back(point2D[point2D.size() - 1]);
		return insertList;
 
	}
 
};

```

$\quad$ B样条方法具有表示与设计自由型曲线曲面的强大功能，是形状数学描述的主流方法之一，另外B样条方法是目前工业产品几何定义国际标准——有理B样条方法 (NURBS)的基础。B样条方法兼备了Bezier方法的一切优点，包括几何不变性，仿射不变性等等，同时克服了Bezier方法中由于整体表示带来不具有局部性质的缺点（移动一个控制顶点将会影响整个曲线）。B样条曲线方程可表示为:

$$p(u) = \sum_{i=0}^n{d_iN_{i,k}(u)}$$

$\quad$ 其中,$d_i(i=0,1,...,n)$为控制顶点(坐标)，$N_{i,k}(i=0,1,...,n)$为$k$次规范$B$样条基函数，最高次数是$k$。基函数是有一个成为节点矢量的非递减参数$u$序列$U$:$u_0 \leq u_1 \leq ... \leq u_{n+k+1}$ 所决定的$k$ 次分段多项式

$\quad$ B样条的基函数通常采用Cox-deBoor递推公式：

$$\begin{cases} N_{i,0} = \begin{cases} 1 \quad if \leq u_i \leq u \leq u_{i+1} \\ 0 \quad others \end{cases} \\ N_{i,k}(u) = \frac{u - u_i}{u_{i+k} - u_i}N_{i,k-1}(u) + \frac{u_{i+k+1} - u}{u_{i+k+1} - u_{i+1}}N_{i+1,k-1}(u) \\ define \quad \frac{0}{0} = 0 \end{cases}$$

$\quad$ 式中$i$为节点序号，$k$是基函数的次数，共有$n+1$个控制顶点。注意区分节点和控制顶点，节点是在节点矢量$U$中取得，控制顶点则是坐标点，决定$B$样条的控制多边形。Cox-deBoor地推公式是B样条曲线的定义的核心，该公式在程序中的实现可采用递归的方式：

```cpp

double N[max1][max2][max3];
double u_data[max3];

init(u_data);//init the data of u_data

for(int i=0;i<max1;i++)
{
    for(int u=0;u<max3;u++)
    {
        if(k == 0)
        {
            if(u >= u_data[i+1] && u < u_data[i+2])
                N[i][0][u] = 0;
            else
                N[i][0][u] = 1;
        }
        for(int k=1;k<max2;k++)
        {   
            double length1 = u_data[i+k] - u_data[i];
            double length2 = u_data[i+k+1] - u_data[i+1];
            if(length1 == 0)
                length1 = 1.0;
            if(length2 == 0)
                length2 = 1.0;
            double temp1 = (u - u_data[i]) / length1 * N[i][k-1][u];
            double temp2 = (u_data[i+k+1] - u) / length2 * N[i+1][k-1][u];
            N[i][k][u] = temp1 + temp2;
        }
    }
}

```

根据节点矢量中节点的分布情况不同，可以划分4种不同类型的B样条曲线。不同类型的B样条曲线主要在于节点矢量，对于具有$n+1$个控制顶点$(P_0,P_1,...,P_n)$的$k$次$B$样条曲线，无论是哪种类型都具有$n+k+2$个节点$[u_0,u_1,...,u_{k+1}]$

- 均匀B样条曲线

节点矢量中节点为沿参数轴均匀或等距分布

对应的节点矢量为：$[0,\frac{1}{7},\frac{2}{7},\frac{3}{7},\frac{4}{7},\frac{5}{7},\frac{6}{7},1]$

- 准均匀B样条曲线

其节点矢量中两端节点具有重复度$k+1$，即$u_0=u_1=...=u_k,u_{n+1},u_{n+2},...,u_{n+k+1}$，所有的内节点均匀分布，具有重复度1

对应的节点矢量为：$[0,0,0,\frac{1}{3},\frac{2}{3},1,1,1]$

- 分段Bezier曲线

其节点矢量中两端节点的重复度与类型2相同，为k+1。不同的是内节点重复度为$k$。该类型有限制条件，控制顶点数减1必须等于次数的正整数倍，即必须满足$\frac{n}{k} \in N$。

对应的节点矢量: $[0,0,0,\frac{1}{2},\frac{1}{2},1,1,1]$

- 一般非均匀B样条曲线

对任意分布的节点矢量$U=[u_0,u_1,...,u_{n+k+1}]$，只要在数学上成立都可选取。

这里给出准均匀B样条和分段Bezier曲线的生成节点矢量的代码，均匀B样条的很简单就不列出了。假设共$n+1$个控制顶点，$k$次B样条，输入参数n,k，输出节点到矢量NodeVector中。

```cpp
#include <vector>

std::vector<double> U_quasi_uniform(int n, int k) 
{
    std::vector<double> NodeVector(n + k + 2, 0.0);
    int piecewise = n - k + 1;

    if (piecewise == 1) 
    {
        for (int i = n + 1; i < n + k + 2; ++i) 
        {
            NodeVector[i] = 1.0;
        }
    } 
    else 
    {
        int flag = 1;
        while (flag != piecewise) {
            NodeVector[k + flag] = NodeVector[k + flag - 1] + 1.0 / piecewise;
            ++flag;
        }
        for (int i = n + 1; i < n + k + 2; ++i) {
            NodeVector[i] = 1.0;
        }
    }

    return NodeVector;
}
```

- B样条曲线的计算

根据B样条曲线的定义公式(1)，曲线上任一点坐标值是参数变量u的函数，用矩阵形式表示

$$p(u) = (d_0,d_1,..,d_n)
\left(
\begin{matrix}
N_{0,k}(u) \\
N_{1,k}(u) \\
... \\
N_{n,k}(u) 
\end{matrix}
\right)$$

可以看出只要已知控制顶点坐标$d_i$，曲线的次数$k$以及基函数$N_{i,k}(u)$，就完全确定了B样条曲线，其中基函数

## 4.4.特殊曲线

# 5.空间采样算法

## 5.1 K近邻算法

k近邻算法是一种基本分类和回归方法。本篇文章只讨论分类问题的k近邻法。

K近邻算法，即是给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的K个实例，这K个实例的多数属于某个类，就把该输入实例分类到这个类中。

k近邻算法是在训练数据集中找到与该实例最邻近的K个实例，这K个实例的多数属于某个类，我们就说预测点属于哪个类。

步骤如下：

假设有一组集：

$$T = \{(x_1,y_1),(x_2,y_2),...,(x_N,y_N)\}$$

其中

$$x_i \in X \subseteq (x_N,y_N)$$

为$n$维的实例特征向量

$$y_i \in Y = \{c_1,c_2,...,c_K\}$$

为实例的类别，其中，$i=1,2,...,N$，预测实例$x$。

输出：预测实例$x$所属类别$y$。

算法执行步骤：

1. 根据给定的距离量度方法(一般情况下使用欧式距离)在训练集T中找出与$x$最相近的$k$个样本点，并将这$k$个样本点所表示的集合记为$N_k(x)$

2. 根据如下所示的多数投票的原则确定实例$x$所属类别$y$:

$$y = argmax\sum_{x_i \in N_{k(x)}}I(y_i,c_j),i=1,2,...,N;j=1,2,...,K$$

上式中$I$为指示函数

$$I(x,y) = \begin{cases}
1, if \quad x = y \\
0, if \quad x \neq y
\end{cases}$$

通过上述KNN算法原理的讲解，我们发现要使KNN算法能够运行必须首先确定两个因素：

1. 算法超参数k；

2. 模型向量空间的距离量度。

- K值的确定

KNN算法中只有一个超参数k，k值的确定对KNN算法的预测结果有着至关重要的影响。接下来，我们讨论一下k值大小对算法结果的影响以及一般情况下如何选择k值。

如果k值比较小，相当于我们在较小的领域内训练样本对实例进行预测。这时，算法的近似误差（Approximate Error）会比较小，因为只有与输入实例相近的训练样本才会对预测结果起作用。

但是，它也有明显的缺点：算法的估计误差比较大，预测结果会对近邻点十分敏感，也就是说，如果近邻点是噪声点的话，预测就会出错。因此，k值过小容易导致KNN算法的过拟合。

同理，如果k值选择较大的话，距离较远的训练样本也能够对实例预测结果产生影响。这时候，模型相对比较鲁棒，不会因为个别噪声点对最终预测结果产生影响。但是缺点也十分明显：算法的近邻误差会偏大，距离较远的点（与预测实例不相似）也会同样对预测结果产生影响，使得预测结果产生较大偏差，此时模型容易发生欠拟合。

因此，在实际工程实践中，我们一般采用交叉验证的方式选取k值。通过以上分析可知，一般k值选得比较小，我们会在较小范围内选取k值，同时把测试集上准确率最高的那个确定为最终的算法超参数k。

样本空间内的两个点之间的距离量度表示两个样本点之间的相似程度：距离越短，表示相似程度越高；反之，相似程度越低。

- 闵可夫斯基距离

闵可夫斯基距离本身不是一种距离，而是一类距离的定义。对于n维空间中的两个点x(x1,x2,…,xn)和y(y1,y2,…,yn)，x和y之间的闵可夫斯基距离可以表示为：

$$d_{xy} = (\sum_{k=1}^{n}{(x_k - y_k)^p})^{\frac{1}{n}}$$

其中，$p$是一个可变参数:

当$p=1$时，被称为曼哈顿距离;

当$p=2$时，被称为欧式距离;

当$p=\infty$，被称为切比雪夫距离

- 欧式距离

欧式距离是最易于理解的一种距离计算方法，源自欧氏空间中两点间的距离公式，也是最常用的距离量度。

- 曼哈顿距离

曼哈顿距离的计算公式可以写为：

$$d_{xy} = \sum_{k = 1}^{n}{|x_k - y_k|}$$


这里展示KNN的简单cpp示例:

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

struct Point {
    double x, y;
    int label;
};

double euclidean_distance(const Point& a, const Point& b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

int knn_classify(const Point& p, const vector<Point>& neighbors, int k) {
    vector<pair<double, int>> distances;
    for (const auto& n : neighbors) {
        distances.push_back(make_pair(euclidean_distance(p, n), n.label));
    }

    sort(distances.begin(), distances.end());

    vector<int> votes(3, 0); // Assuming there are 3 classes labeled as 0, 1, and 2
    for (int i = 0; i < k; i++) {
        votes[distances[i].second]++;
    }

    return max_element(votes.begin(), votes.end()) - votes.begin();
}

int main() 
{
    vector<Point> neighbors = { {1, 2, 0}, {3, 4, 1}, {5, 6, 2} };
    Point p = {2, 3, -1}; // Point to be classified

    int k = 2; // Number of neighbors to consider
    p.label = knn_classify(p, neighbors, k);

    cout << "The point is classified as: " << p.label << endl;

    return 0;
}
```

## 5.2 PRM算法

PRM算法的步骤如下：

- 初始化两个集合，其中$N$: 随机点集，$E$:路径集

- 随机撒点，将撒入的点放入$N$中，随机撒点的过程中：

1. 必须是自由空间的随机点

2. 每个点都要确保与障碍物无碰撞(最基本约束)

- 对每一个新的节点C，我们从当前N中选择一系列的相邻点$n$，并且使用local_planner进行路径规划

- 将可行的路径的边界$(c,n)$加入到$E$的集合中，不可行的路径去掉

它本身实际上是基于图搜索的方法，一共分为两个步骤：学习阶段和查询阶段。

它将连续空间转换成离散空间，再利用最短路等在路线图上寻找路径，提高搜索效率。可以用相对少的随机采样点来找到一个解，对多数问题而言，相对少的样本足以覆盖大部分可行的空间，并且找到路径的概率为1。当采样点太少或者分布不合理时，PRM算法是不完备的。

```cpp
#include<opencv2/opencv.hpp>

#include<unistd.h>
#include<typeinfo>

#include<iostream>
#include<vector>
#include<string>
#include<queue>

#include<random>
#define INF 13421772
#define sampleNum 200
#define START 0
#define GOAL 1

#define PI 3.141592653589793238
double toDegree(double radian)
{
    return radian * 180 / PI;
}

double toRadian(double degree)
{
    return degree * PI / 180;
}

struct GraphNode{
    int label;
    std::vector<GraphNode*> neighbors;
    GraphNode(int x):label(x){};
};

bool checkCollision(const std::vector<int> point,const cv::Mat img_src)
{
    bool reach = true;
    if(img_src.at<cv::Vec3b>(point[1],point[0])[0] == 0 &&
       img_src.at<cv::Vec3b>(point[1],point[0])[1] == 0 &&
       img_src.at<cv::Vec3b>(point[1],point[0])[2] == 0)
       {
            reach = false;
       }
    return reach;
}

bool checkPath(const std::vector<int> point_a,
               const std::vector<int> point_b,
               const cv::Mat map,int split_num)
{
    std::vector<double> path_x;
    std::vector<double> path_y;

    double interval_x = (point_b[0] - point_a[0]) / split_num;
    double interval_y = (point_b[1] - point_a[1]) / split_num;

    for(int i = 0; i <= split_num; i++)
    {
        path_x.push_back(point_a[0] + i * interval_x);
        path_y.push_back(point_a[1] + i * interval_y);
    }

    for(int i = 0; i < split_num; i++)
    {
        if(!checkCollision({int(path_x[i]),int(path_y[i])},map))
        {
            return false;
        }
    }

    return true;
}

double calcDistance(const std::vector<int> point_a,
                    const std::vector<int> point_b)
{
    return sqrt(std::pow(point_a[0] - point_b[0],2) + std::pow(point_a[1] - point_b[1],2));
}

struct Dijkstra {
    struct node{
        int point;
        double value;
        node(int _point, double _value):point(_point), value(_value){}
        bool operator < (const node& rhs) const{
                return value > rhs.value;
        }
    };

    std::vector<node> edges[sampleNum];
    double dist[sampleNum];
    int path[sampleNum];

    void init()
	{
		for(int i = 0; i < sampleNum; i++)
        {
            edges[i].clear();
            dist[i] = 0;
            path[i] = 0;
        }
	}

    void addEdge(int from, int to, double dist)
	{
		edges[from].push_back(node(to,dist));
        // edges[to].push_back(node(from,dist));
	}

    void showEdge()
    {
        std::cout << "------------------------" << std::endl;

        for(int i = 0; i< sampleNum; i++)
        {
            // cout << "-----" << i << "-----" << endl;

            for(int j=0; j<edges[i].size(); j++)
            {
                std::cout << i << "," << edges[i][j].point << "," << edges[i][j].value << std::endl;
            }

            // cout << "**********************" << endl;

        }

        std::cout << "------------------------" << std::endl;
    }

    std::vector<int> dijkstra(int s, int t)
	{
		std::priority_queue <node> q;

		for(int i = 0; i < sampleNum; i++)
            dist[i] = INF;

		dist[s] = 0;
		q.push(node(s, dist[s]));

		while(!q.empty())
		{
            node x = q.top(); q.pop();
            for (int i = 0; i < edges[x.point].size(); i++)
            {
                // std::cout << edges[x.point].size() << std::endl;
                node y = edges[x.point][i];
                if (dist[y.point] > dist[x.point] + y.value)
                {
                    dist[y.point] = dist[x.point] + y.value;
                    path[y.point] = x.point;
                    q.push(node(y.point, dist[y.point]));
                }
            }
		}

        std::vector<int> result;

        // 存距离
        // result.push_back(dist[t]);
        std::cout << "dist[t]:" << dist[t] << std::endl;

        while(t)
        {
            result.push_back(t);
            t = path[t];
        }
        
        result.push_back(path[0]);


        reverse(result.begin(),result.end()); //起点->终点 ，+ 距离

        return result;
    }
};

struct Dijkstra DijkstraPlanning;

int main(int argc,char **argv)
{
    cv::Mat dismap = cv::imread("../map/map_2.bmp");

    int mapLength = dismap.cols;
    int mapWidth = dismap.rows; 

    std::vector<int> pStrat = {10, 10};
    std::vector<int> pGoal = {490, 490};

    std::vector<std::vector<int> > sampleMap;
    sampleMap.reserve(sampleNum + 2);
    sampleMap.push_back(pStrat);
    sampleMap.push_back(pGoal);

    //初始化采样点
    while (sampleMap.size() < sampleNum + 2)
    // while (sampleMap.size() < 10)
    {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, dismap.size[0]-1);
        
        
        std::vector<int> res;
        // sampleMap.push_back(std::vector<int>());
        for(int i=0; i<2; ++i){
            res.push_back(dis(gen));
            // sampleMap.back().push_back(dis(gen));
        }

        if(checkCollision(res, dismap)){
            sampleMap.push_back({res[0], res[1]});
        }
        
        // std::cout << res[0] << "," << res[1] << std::endl;
    }

    const int MAX_N = sampleNum + 2;
    GraphNode *Graph[MAX_N];

    const double DISTANCE = 100; 
    const int SPLIT_N = 10;

    DijkstraPlanning.init();

    for (int i = 0; i < MAX_N; i++)
    {
        for(int j = 0; j < MAX_N; j++){
            if (calcDistance(sampleMap[i], sampleMap[j]) <= DISTANCE && 
                checkPath(sampleMap[i], sampleMap[j], dismap, SPLIT_N) && i!= j)
            {
                cv::line(dismap, cv::Point(sampleMap[i][0], sampleMap[i][1]), 
                                 cv::Point(sampleMap[j][0], sampleMap[j][1]), 
                                 cv::Scalar(0, 0, 255), 1);
                // Graph[i]->neighbors.push_back(Graph(i));

                DijkstraPlanning.addEdge(i, j, calcDistance(sampleMap[i], sampleMap[j]));
            }
            
        }
    }
    DijkstraPlanning.showEdge();

    std::vector<int> result = DijkstraPlanning.dijkstra(START, GOAL);

    if(result.size() > 2){
        for(int i=0;i < result.size() - 1; i++)
        {
            // std::cout << result[i] << " ";
            cv::line(dismap, cv::Point(sampleMap[result[i]][0], sampleMap[result[i]][1]), 
                    cv::Point(sampleMap[result[i+1]][0], sampleMap[result[i+1]][1]), 
                    cv::Scalar(0, 255, 0), 5);
        }
    }

    for(int i=0; i<sampleMap.size(); i++){
        // std::cout << i << ":";
        for(int j=0; j<sampleMap[i].size(); j++){
            // std::cout << sampleMap[i][j] << ",";
            dismap.at<cv::Vec3b>(sampleMap[i][1],sampleMap[i][0])[0] = 0;
            dismap.at<cv::Vec3b>(sampleMap[i][1],sampleMap[i][0])[1] = 255; 
            dismap.at<cv::Vec3b>(sampleMap[i][1],sampleMap[i][0])[2] = 0;

        }
        // std::cout << std::endl;
    }
    
    // std::cout << sampleMap.size() << std::endl;

    cv::imshow("dismap", dismap);
    cv::waitKey(10000);
    
    return 0;
}
```

## 5.3 RRT算法

RRT是Steven M. LaValle和James J.Kuffner Jr.提出的一种通过随机构建Space Filling Tree实现对非凸高维空间快速搜索的算法。该算法可以很容易的处理包含障碍物和差分运动约束的场景，因而广泛的被应用在各种机器人的运动规划场景中。

原始的RRT算法中将搜索的起点位置作为跟节点，然后通过随机采样增加叶子节点的方式，生成一个随机拓展树。当随机树的叶子节点进入目标区域，就得到了从起点位置到目标位置的路径。

```cpp
Path RRT(const Map& M,const Node& x_start,const Node& x_end)
{
    Path path = new Path(x_start);
    Node x_new = new Node();
    while(x_new != x_end)
    {
        x_rand = Sample(M);
        x_near = Near(x_rand,path);
        x_new = Steer(x_rand,x_near,StepSize);
        E = Edge(x_new,x_near);
        if(CollisionFree(M,E))
        {
            path.addNode(x_new);
            path.addEdge(E);
        }
    }
    return path;
}
```

上面的代码中，M是地图环境，x_start是起始位置，x_goal是目标位置。路径空间搜索的过程从起点开始，先随机撒点x_rand，然后查找距离x_rand最近的节点x_near，然后沿着x_near到x_rand方向前进Stepsize距离得到x_new,使用CollisionFree(M,E)方法检测Edge(x_new,x_near)是否与地图环境中的障碍物有碰撞，如果没有碰撞，则完成一次空间搜索拓展。重复上述流程，直到达到目标位置。

为了加快随机数收敛到目标位置的速度，基于概率的RRT算法在随机树拓展的步骤中引入了一个概率$p$，根据概率的值来选择树的生长方向是随即生长还是朝向目标位置生成，引入向目标生长的机制可以加快路径搜索的收敛速度。

```cpp
x_rand = ChooseTarget(x_rand,x_end,p);
```

## 5.4 RRT-Connect

RRT-Connect在RRT的基础上引入了双树拓展环节，即分别以起点和目标点为根节点生成两个树进行双向拓展，当两棵树建立连接时可认为路径规划成功。通过一次采样得到一个采样点x_rand，然后两棵树同时向x_rand方向进行拓展，加快两棵树建立连接的速度。相较于单树拓展的RRT算法，RRT-Connect加入了启发式步骤，加快了搜索速度，对于狭窄通道也有较好的效果。

## 5.5 RRT*算法

RRT*算法是一种渐进最优算法。

算法流程与RRT算法流程基本相同，不同之处就在于最后加入将x_new加入搜索树时父节点的选择策略。

RRT*算法在选择父节点时会有一个重连的过程，也就是在以x_new为圆心，半径为r的邻域内，找到与x_new连接后路径代价(从起点移动到x_new的路径长度)最小的节点，并重新选择x_min作为x_new的父节点，而不是x_near。





