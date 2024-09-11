// Don't use cin.tie when problem is interactive
// Use '\n' instead of endl; endl - flushes output buffer
// Strings are immutable, concatenation leads to reallocation. Use ostringstream instead.
// Reserve Memory for vector / string. v.reserve(n)
// 
// |   n   |  Worst Runtime     |         Comment           |
// | <=10  | O(n!), O(n^6)      | Enumerating permutations  |
// | <=20  | O(2^n * n)         |                           |
// | <=100 | O(n^4)             |                           |
// | <=400 | O(n^3)             |                           |
// | <=2000| O(n^2 log n)       |                           |
// | <=10^4| O(n^2)             | Insertion Sort            |
// | <=10^6| O(n log n)         | Merge Sort                |
// | <=10^8| O(n)               | Slow I/O => often n<=10^6 |

typedef long long ll;
#define debug(x) \
    (std::cerr << #x << ": " << (x) << '\n')

#include <iostream>
int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    return 0;
}

// COPYPASTE FROM GIT
// strings
// ahocorasick 29-144
#include <bits/stdc++.h>
#define int long long

using namespace std;

// Implementation of a trie node for Aho-Corasick

// The "cout << ..." command can be replaced by something different
// (e.g. by something that puts the number in a result vector).

// Note that the number of matches will also be printed for position -1,
// which equals the number of empty strings fed into the algorithm
// (empty strings already match at the beginning of the string, even when
// no character has been read yet).
struct ACTrie {

    map<char, ACTrie*> edges; // Outgoing edges of the trie node
    ACTrie* lsp = nullptr; // Longest Suffix-Prefix

    // Number of input strings being a suffix
    // of the string associated with this trie node.
    int cnt = 0;

    void insert(string &t, int i = 0) {
        if (i == t.length()) {
            cnt++;
            return;
        }
        if (edges.count(t[i]) == 0)
            edges[t[i]] = new ACTrie;
        edges[t[i]]->insert(t, i+1);
    }

    // Searches at the current node for matches in the string s[i..].
    // 'print' denotes whether the number of matches should be printed
    // for the current position. The only case when we don't want this is
    // when an LSP-jump has been done, and the correct result for i has already
    // been printed.
    void search(string &s, int i = 0, bool print = true) {
        if (print) cout << cnt << " matches ending at " << i-1 << "\n";

        if (i == s.length()) return; // processing of the string is done

        if (edges.count(s[i]) == 0) {
            // The trie node doesn't have the needed character edge...
            if (lsp == nullptr) search(s, i+1, true); // we are at the root node
            else lsp->search(s, i, false); // try to continue search at the LSP
        } else {
            // Edge was found, continue search there and advance the string
            // pointer by one...
            edges[s[i]]->search(s, i+1, true);
        }
    }
};

// Should be called after inserting strings into the trie and before searching
// for matches in another string.
void preprocess(ACTrie &root) {
    queue<ACTrie*> q;
    root.lsp = nullptr; q.push(&root);
    while (!q.empty()) {
        ACTrie *u = q.front(); q.pop();
        for (auto it : u->edges) { // edge u--c-->v
            char c = it.first; ACTrie *v = it.second;

            // the 'lsp' and 'cnt' values of v will be calculated now...
            ACTrie *l = u->lsp;
            while (l != nullptr && l->edges.count(c) == 0)
                l = l->lsp;
            if (l == nullptr) {
                v->lsp = &root; // there is no strict suffix-prefix
            } else {
                v->lsp = l->edges[c];
                v->cnt += v->lsp->cnt;
            }
            q.push(v);
        }
    }
}


// The following code shows how to use the implementation above.
// As an example, it first reads the small strings that will be matched.
// It follows a list of large strings, in which all matches of the small
// strings are to be found.

// Example Input:
/*
6
baba
b
abab
ababac
abaca
ac
1
abaababcababaca
*/
int32_t main() {
    ACTrie root;

    int k; cin >> k;
    for (int i = 0; i < k; ++i) {
        string t; cin >> t;
        root.insert(t);
    }
    preprocess(root);

    int l; cin >> l;
    for (int i = 0; i < l; ++i) {
        string s; cin >> s;
        root.search(s);
    }
}

// KMP 145-181
#include <bits/stdc++.h>
#define int long long

using namespace std;

string s, t;
int n, m;

int32_t main() {
    cin >> s >> t;
    n = s.size(); m = t.size();

    vector<int> lsp(m, 0);
    for (int i = 1, prev = 0; i < m; ) {
      if (t[i] == t[prev]) {
        prev++;
        lsp[i] = prev;
        i++; 
      } else if (prev == 0) {
        lsp[i] = 0;
        i++;
      } else { prev = lsp[prev-1]; }
    }

    int start = 0, len = 0;
    while (start + len < n) {
        while (len >= m || s[start+len] != t[len]) {
            if (len == 0) { start++; len = -1; break; }
            int skip = len - lsp[len-1];
            start += skip; len -= skip;
        }
        len++;
        if (len == m)
            cout << "t matches s at " << start << "\n";
    }
}

// Trie 183-222
#include <bits/stdc++.h>
#define int long long

using namespace std;

struct trie {
    bool isEndOfString = false;
    map<char, trie*> edges;
    void insert(string &s, int i = 0) {
        if (i == s.length()) {
            isEndOfString = true;
            return;
        }
        if (edges.count(s[i]) == 0)
            edges[s[i]] = new trie;
        edges[s[i]]->insert(s, i+1);
    }
    bool contains(string &s, int i = 0) {
        if (i == s.length())
            return isEndOfString;
        return edges.count(s[i]) > 0 &&
            edges[s[i]]->contains(s, i+1);
    }
};

trie t;

int32_t main() {
    int n; cin >> n;
    for (int i = 0; i < n; ++i) {
        string s; cin >> s;
        t.insert(s);
    }
    int m; cin >> m;
    for (int i = 0; i < m; ++i) {
        string s; cin >> s;
        cout << t.contains(s) << "\n";
    }
}

// math
// fast exp 225-235
// Computes m^n modulo p
ll fexp(ll m, ll n, ll p) {
    if (n == 0) return 1;
    else if (n % 2 == 1)
        return (m * fexp(m, n-1, p)) % p;
    else { // n is even
        ll r = fexp(m, n/2, p);
        return (r * r) % p;
    }
}

// order statistics 236-260
// a set with the features of an array
// implemented in STL, but only supported by GNU C++

// #include <bits/stdc++.h>
// #include <ext/pb_ds/assoc_container.hpp>
// #include <ext/pb_ds/tree_policy.hpp>

// using namespace __gnu_pbds;
// using namespace std;

// typedef tree<int, null_type, less<int>, rb_tree_tag, tree_order_statistics_node_update> ost;
// // other types and comparators are also possible (eg. pair<int,int> and greater<pair<int,int>>)

// int main(){
// 	// usage
// 	ost t;
// 	t.insert(-1); t.insert(10); t.insert(21); t.insert(42);
// 	cout << *t.find_by_order(3) << "\n"; // key at index 3 (42) in O(log n)
// 	t.insert(37);
// 	cout << *t.find_by_order(3) << "\n"; // now this outputs 37
// 	cout << t.order_of_key(42) << "\n"; // position of key 42 (4) in O(log n)
// 	// actually calculates number of smaller keys
// }

// segment tree 261-344
#include <bits/stdc++.h>
using namespace std;

class segtree {
private:
    vector<long long> values;
    size_t size;

    int parent(int i) {
        return i / 2;
    }
    int left(int i) {
        return 2 * i;
    }
    int right(int i) {
        return 2 * i + 1;
    }

    void update(int i) {
        values[i] = values[left(i)] + values[right(i)];
        if (i > 1) update(parent(i));
    }

    // Query sum of interval [i, j)
    // current_node represents the interval [l, r)
    long long query(int i, int j, int l, int r, int current_node) {
        if (r <= i || j <= l) return 0; // current interval and query interval don't intersect
        if (r <= j && i <= l) return values[current_node]; // current interval contained in query interval

        int m = (l + r) / 2;
        return query(i, j, l, m, left(current_node)) + query(i, j, m, r, right(current_node));
    }

public:
    segtree(size_t n) {
	size = 1<<(int)ceil(log2(n));
        values.assign(2 * size, 0);
    }

    segtree(vector<long long> v): segtree(v.size()) {
        for (size_t i = 0; i < v.size(); ++i) values[i + size] = v[i];
        for (size_t i = size - 1; i > 0; --i) values[i] = values[left(i)] + values[right(i)];
    }

    // Query sum of interval [i, j)
    long long query(int i, int j) {
        return query(i, j, 0, size, 1);
    }

    // Set value at position i to val
    void update(int i, long long val) {
        values[i + size] = val;
        update(parent(i + size));
    }
};

typedef long long ll;
int main()
{
    int n, q;
    cin >> n >> q;
    vector<ll> v(n);
    for (ll& l: v) cin >> l;

    segtree s(v);
    while (q--) {
        int type;
        cin >> type;

        if (type == 1) {
            // update
            ll k, u;
            cin >> k >> u;
            s.update(k - 1, u);
        } else {
            // query
            int a, b;
            cin >> a >> b;
            cout << s.query(a - 1, b) << endl;
        }
    }
}

// dp
// fibonacci 347-357
#include <bits/stdc++.h>
using namespace std;

long long fib(int n) {
    vector<long long> dp(max(n+1, 2), 0);
    dp[0] = dp[1] = 1;
    for (int i = 2; i <= n; ++i)
        dp[i] = dp[i-1] + dp[i-2];
    return dp[n];
}

// knapsack 359-375
#include <bits/stdc++.h>
using namespace std;

int knapsack(int n, int maxW, vector<int> &w, vector<int> &p) {
    vector<vector<int>> dp(n, vector<int>(maxW+1));
    for (int v = 0; v <= maxW; ++v)
        dp[0][v] = (v >= w[0] ? p[0] : 0);
    for (int i = 1; i < n; ++i) {
        for (int v = 0; v <= maxW; ++v) {
            dp[i][v] = dp[i-1][v];
            if (v >= w[i])
                dp[i][v] = max(dp[i][v], p[i] + dp[i-1][v - w[i]]);
        }
    }
    return dp[n-1][maxW];
}

// knapsack 2 377-395
#include <bits/stdc++.h>
using namespace std;

int knapsack_(vector<vector<int>> &dp, int i, int v, vector<int> &w, vector<int> &p) {
    if (dp[i][v] != -1) return dp[i][v];
    int &res = dp[i][v];
    if (i == 0) res = (v >= w[0]) ? p[0] : 0;
    else {
        res = knapsack_(dp, i-1, v, w, p);
        if (v >= w[i]) res = max(res, p[i] + knapsack_(dp, i-1, v-w[i], w, p));
    }
    return res;
}

int knapsack(int n, int maxW, vector<int> &w, vector<int> &p) {
    vector<vector<int>> dp(n, vector<int>(maxW+1, -1));
    return knapsack_(dp, n-1, maxW, w, p);
}

// lis 398-413
#include <bits/stdc++.h>
using namespace std;

int lis(const vector<int> &a) {
    int n = a.size();
    vector<int> dp(n, 0);
    int result = 0;
    for (int i = 0; i < n; ++i) {
        dp[i] = 1;
        for (int j = 0; j < i; ++j)
            if (a[j] < a[i])
                dp[i] = max(dp[i], dp[j] + 1);
        result = max(result, dp[i]);
    }
    return result;
}

// lis2 416-440
#include <bits/stdc++.h>
using namespace std;

int lis(const vector<int> &a, deque<int> &seq) {
    int n = a.size();
    vector<int> dp(n, 0), predecessor(n, -1);
    int result = 0, lastElement = -1;
    for (int i = 0; i < n; ++i) {
        dp[i] = 1;
        for (int j = 0; j < i; ++j)
            if (a[j] < a[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                predecessor[i] = j;
            }
        if (dp[i] > result) {
            result = dp[i];
            lastElement = i;
        }
    }
    while (lastElement != -1) {
        seq.push_front(lastElement);
        lastElement = predecessor[lastElement];
    }
    return result;
}

// basic
// binary search 444-457
#include <bits/stdc++.h>
using namespace std;

// returns position in S or -1
int binary_search(vector<int>& S, int k) {
    int mini = 0, maxi = S.size();
    while (mini < maxi - 1) {
        int middle = (mini + maxi) / 2;
        if (S[middle] <= k) mini = middle;
        else maxi = middle;
    }
    if (S[mini] == k) return mini;
    else return -1;
}

// binary search 2 459-474
#include <bits/stdc++.h>
using namespace std;

bool check(long long i);

// returns minimal t so that check(t) is true
long long binary_search() {
    long long mini = -1, maxi = numeric_limits<long long>::max() / 2;
    while (mini < maxi - 1) {
        long long middle = (mini + maxi) / 2;
        if (check(middle)) maxi = middle;
        else mini = middle;
    }
    return maxi;
}

// exponential search 476-493
#include <bits/stdc++.h>
using namespace std;

bool check(long long i);

// returns minimal t so that check(t) is true
long long exponential_search() {
    long long mini = -1, distance = 1;
    while (!check(mini + distance)) distance *= 2;
    long long maxi = mini + distance;
    while (mini < maxi - 1) {
        long long middle = (mini + maxi) / 2;
        if (check(middle)) maxi = middle;
        else mini = middle;
    }
    return maxi;
}

// graphs
// articulation points and bridges 496-538
#include <bits/stdc++.h>
using namespace std;

int V; // number of nodes
vector<vector<int>> adj; // the graph

int dfs_counter = 0;
const int UNVISITED = -1;
int dfsRoot, rootChildren;

vector<int> dfs_num(V, UNVISITED);
vector<int> dfs_min(V, UNVISITED);
vector<int> dfs_parent(V, -1);

void dfs(int u) {
    dfs_min[u] = dfs_num[u] = dfs_counter++;
    for (auto v: adj[u]) {
        if (dfs_num[v] == UNVISITED) { // Tree Edge
            dfs_parent[v] = u;
            if (u == dfsRoot) rootChildren++;

            dfs(v);

            if (dfs_num[u] <= dfs_min[v] && u != dfsRoot)
                cout << u << " is AP" << endl;
            if (dfs_num[u] < dfs_min[v])
                cout << u << "-" << v << " is Bridge" << endl;
            dfs_min[u] = min(dfs_min[u], dfs_min[v]);
        } else if (v != dfs_parent[u]) // Back Edge
            dfs_min[u] = min(dfs_min[u], dfs_num[v]);
    }
}

void articulation_points_and_bridges() {
for (int i = 0; i < V; i++)
    if (dfs_num[i] == UNVISITED) {
        dfsRoot = i; rootChildren = 0;
        dfs(i); // code on next slide
        if (rootChildren > 1)
            cout << i << " is AP" << endl;
    }
}

// bellman-ford 540-561
#include <bits/stdc++.h>
using namespace std;

int V; // number of nodes
int INF; // a sufficiently large number
vector<vector<pair<int, int>>> adj; // the graph

void bellman_ford(int start)
{
    vector<int> dist(V, INF);
    dist[start] = 0;
    for (int i = 0; i < V - 1; i++) {
        for (int v = 0; v < V; v++) {
            for (auto p: adj[v]) {
                int u = p.first;
                int w = p.second;
                dist[u] = min(dist[u], dist[v] + w);
            }
        }
    }
}

// bfs 563-584
#include <bits/stdc++.h>
using namespace std;

int V; // number of nodes
int INF; // a sufficiently large number
vector<vector<int>> adj; // the graph

void bfs(int start, vector<int> distance) {
    distance.assign(V, INF);
    queue<int> Q;
    distance[start] = 0; Q.push(start);
    while (!Q.empty()) {
        int v = Q.front(); Q.pop();
        for (int u: adj[v]) {
            if (distance[u] == INF) { // not visited
                distance[u] = distance[v] + 1;
                Q.push(u);
            }
        }
    }
}

// bipartite 586-610
#include <bits/stdc++.h>
using namespace std;

int V; // number of nodes
vector<vector<int>> adj; // the graph
vector<int> colors(V, -1); // -1 means unvisited

void dfs(int v) {
    for (auto u: adj[v])
        if (colors[u] == -1) {
            colors[u] = 1 - colors[v];
            dfs(u);
        } else if (colors[u] == colors[v]) {
            cout << "Impossible" << endl;
            exit(0);
        }
}

void is_bipartite(int start) {
    colors[start] = 0; // colors are 0, 1
    // assume adj to be connected
    dfs(start);
    cout << "Possible" << endl;
}

// dfs 612-633
#include <bits/stdc++.h>
using namespace std;

int V; // number of nodes
vector<vector<int>> adj; // the graph

void dfs(int start) {
    vector<bool> visited(V, false);
    stack<int> S;          // LIFO

    S.push(start);         //
    visited[start] = true; // start vertex
    while (!S.empty()) {
        int v = S.top(); S.pop();
        for (int u: adj[v])
            if (!visited[u]) {
                S.push(u);
                visited[u] = true;
            }
    }
}

// dfs 2 635-648
#include <bits/stdc++.h>
using namespace std;

int V; // number of nodes
vector<vector<int>> adj; // the graph
vector<bool> visited(V, false);

void dfs(int v) {
    visited[v] = true;
    for (int u: adj[v])
        if (!visited[u])
            dfs(u);
}

// dijkstra 650-676
#include <bits/stdc++.h>
using namespace std;

int V; // number of nodes
int INF; // a sufficiently large number
vector<vector<pair<int, int>>> adj; // the graph

void dijkstra(int start) {
    vector<int> dist(V, INF);
    dist[start] = 0;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int,int>>> pq;
    pq.push({0, start}); // <distance, vertex>
    while (!pq.empty()) {
        auto front = pq.top(); pq.pop();
        int d = front.first, v = front.second;
        if (d > dist[v]) continue; // lazy deletion
        for (auto p: adj[v]) { // <target, weight>
            int u = p.first;
            int w = p.second;
            if (dist[v] + w < dist[u]) {
                dist[u] = dist[v] + w;
                pq.push({dist[u], u}); // can push duplicate vertices
            }
        }
    }
}

// edmonds-karp 678-747
#include<bits/stdc++.h>

using namespace std;
#define INF numeric_limits<int>::max()

vector<vector<int>> capacity;
vector<vector<int>> adj;
vector<int> parent;

void bfs(int s) {
    parent.assign(adj.size(), -1);
    parent[s] = -2; // s is visited

    queue<int> Q;
    Q.push(s);
    while (!Q.empty()) {
        int u = Q.front(); Q.pop();
        for (int v : adj[u]) // go u -> v
            if (parent[v] == -1 and capacity[u][v] > 0) {
                Q.push(v);
                parent[v] = u;
            }
    }
}

int maxflow(int s, int t) {
    int totalflow = 0;
    int u;
    while (true) {
        // build bfs tree
        bfs(s);
        if (parent[t] == -1) // t unreachable
            break;
        // find bottleneck capacity
        int bottleneck = INF;
        u = t;
        while (u != s) {
            int v = parent[u];
            bottleneck = min(bottleneck, capacity[v][u]);
            u = v;
        }
        // update capacities along path
        u = t;
        while (u != s) {
            int v = parent[u];
            capacity[v][u] -= bottleneck;
            capacity[u][v] += bottleneck;
            u = v;
        }
        totalflow += bottleneck;
    }
    return totalflow;
}

int main() {
    int s, t, V, E;
    cin >> V >> E >> s >> t;
    adj.resize(V);
    capacity.assign(V, vector<int>(V, 0));
    int u, v, c;
    for (int i = 0; i < E; i++) {
        cin >> u >> v >> c;
        adj[u].push_back(v);
        adj[v].push_back(u);
        capacity[u][v] += c;
    }
    int out = maxflow(s, t);
    cout << out << endl;
}

// euler 749-786
#include <bits/stdc++.h>
using namespace std;

int V; // number of nodes
int E; // number of edges
vector<vector<int>> adj; // the graph
vector<int> indegree; // store indegree of each vertex
deque<int> cycle;

void find_cycle(int u) {
    while (adj[u].size()) {
        int v = adj[u].back();
        adj[u].pop_back();
        find_cycle(v);
    }
    cycle.push_front(u);
}

void euler_cycle() {
    // test if solution can exist
    for (int i = 0; i < V; i++)
        if (indegree[i] != adj[i].size()) {
            cout << "IMPOSSIBLE" << endl;
            exit(0);
        }

    // start anywhere
    find_cycle(0); // populate cycle
    // test against disconnected graphs
    if (cycle.size() != E + 1) {
        cout << "IMPOSSIBLE" << endl;
        exit(0);
    }
    for (auto v: cycle)
        cout << v << " ";
    cout << endl;
}

// find cycles 788-815
#include <bits/stdc++.h>
using namespace std;

int V; // number of nodes
vector<vector<int>> adj; // the graph

int UNVISITED = 0, EXPLORED = 1, VISITED = 2;
vector<int> visited(V, UNVISITED);

void dfs(int v) {
    visited[v] = EXPLORED;
    for (auto u: adj[v])
        if (visited[u] == UNVISITED) {
            dfs(u);
        } else { // not part of dfs tree
            if (visited[u] == EXPLORED) {
                cout << "Cycle found" << endl;
                exit(0);
            }
        }
    visited[v] = VISITED;
}

void find_cycles(int start) {
    dfs(start);
    cout << "Graph is acyclic" << endl;
}

// floyd-warshall 817-829
#include <bits/stdc++.h>
using namespace std;

int V; // number of nodes
vector<vector<int>> adj; // adjacency matrix (!) of the graph

void floyd_warshall() {
    for (int k = 0; k < V; k++)
        for (int i = 0; i < V; i++)
            for (int j = 0; j < V; j++)
                adj[i][j] = min(adj[i][j], adj[i][k] + adj[k][j]);
}

// hamiltonian path 831-883
#include <bits/stdc++.h>

/*
 * sample input:
4 4
1 4
2 3
3 4
2 4
 * sample output:
0 : 1
1 : 1
2 : 1
3 : 0
 */

using namespace std;

int main(){
	int n, m;
	cin >> n >> m;
	set<pair<int,int>> edges;
	for(int i = 0; i < m; i++){
		int u, v;
		cin >> u >> v;
		u--, v--;
		edges.insert({u,v}), edges.insert({v,u});
	}

	vector<vector<bool>> dp(1<<n, vector<bool>(n, false)); // subset, endnode

	for(int i = 0; i < n; i++){
		dp[1 << i][i] = true;
	}

	for(int X = 1; X < (1<<n); X++){ // all sets
	if(__builtin_popcount(X) < 2) continue;

	for(int v = 0; v < n; v++) {   // all ending vertices
		if(!(X & (1<<v))) continue;  // v not in X

		for(int u = 0; u < n; u++) {
		if(dp[X ^ (1<<v)][u] && edges.count({u, v})){
			dp[X][v] = true;
		}
		}
	}
	}
	for(int i = 0; i < n; i++){
		cout << i << " : " << dp[(1<<n)-1][i] << "\n";
	}
}

// kruskal 885-935
#include<bits/stdc++.h>

using namespace std;


class UnionFind {
    private:
        vector<int> parent, rank;
    public: 
        UnionFind(int N) {
            rank.assign(N, 0);
            parent.assign(N, 0);
            for (int i = 0; i < N; i++) parent[i] = i;
        }
        int findSet(int i) {
            if (parent[i] == i)
                return i;
            else     // path compression
                return parent[i] = findSet(parent[i]);
        }
        bool isSameSet(int i, int j) {
            return findSet(i) == findSet(j);
        }
        void unionSet(int i, int j) {
            if (!isSameSet(i, j)) {
                int x = findSet(i), y = findSet(j);
                if (rank[x] > rank[y])
                    parent[y] = x;
                else {
                    parent[x] = y;
                    if (rank[x] == rank[y]) rank[y]++;
                }
            }
        }
};


int main() {
    auto UF = UnionFind(5);
    vector<tuple<int, int, int>> edgeList;
    sort(edgeList.begin(), edgeList.end());
    int w, u, v;
    for (auto edge: edgeList) {
        tie(w, u, v) = edge;
        if (!UF.isSameSet(u, v)) {
            UF.unionSet(u, v);
        }
    }
}

// lca 937-987
#include <bits/stdc++.h>

using namespace std;

void dfs(int u, vector<vector<int>> &g, vector<int> &h, vector<int> &p){
	for(int v : g[u]) if(v != p[u]){
		p[v] = u;
		h[v] = h[u]+1;
		dfs(v, g, h, p);
	}
}

int kth(int u, int k, vector<vector<int>> &p){
	for(int i = p.size()-1; i >= 0; i--){
		if(k & (1<<i)){
			u = p[i][u];
		}
	}
	return u;
}

int lca(int u, int v, vector<vector<int>> &p, vector<int> &h){
	if(h[v] > h[u]) swap(u, v);
	u = kth(u, h[u]-h[v], p); // lift lower node to the same height
	if(u == v) return u;

	for(int i = p.size()-1; i >= 0; i--){
		if(p[i][u] != p[i][v]){
			u = p[i][u];
			v = p[i][v];
		}
	}
	return p[0][u];
}

int main(){
	int n, root;
	vector<vector<int>> g; // graph
	int l = ceil(log2(n))+1;

	vector<int> h(n); // heights of the nodes
	vector<vector<int>> p(l, vector<int>(n));

	p[0][root] = root;
	dfs(root, g, h, p[0]); // precompute first ancestors and heights

	for(int i = 1; i < l; i++) // order of loops matters!
		for(int j = 0; j < n; j++)
			p[i][j] = p[i-1][p[i-1][j]];
}

// prims 988-1030
#include<bits/stdc++.h>

using namespace std;

typedef tuple<int, int, int> queue_entry;

void visit(int v, vector<bool>& visited, vector<vector<pair<int,int>>>& adj, priority_queue<queue_entry, vector<queue_entry>, greater<queue_entry>>& PQ) {
    visited[v] = true;
    for (auto p: adj[v]) {
        int u = p.first;
        int w = p.second;
        if (!visited[u]) PQ.push({w, v, u});
    }
}

int main() {
    int V, E;
    cin >> V >> E;
    vector<vector<pair<int,int>>> adj(V);

    // read adj
    int from, to, weight;
    for (int i = 0; i < E; i++) {
        cin >> from >> to >> weight;
        adj[from].push_back({to, weight});
        adj[to].push_back({from, weight});
    }

    vector<bool> visited(V, false);
    // <weight, from, to>
    priority_queue<queue_entry, vector<queue_entry>, greater<queue_entry>> PQ;
    visit(0, visited, adj, PQ);
    while (!PQ.empty()) {
        auto front = PQ.top(); PQ.pop();
        int w, from, to;
        tie(w, from, to) = front;
        if (!visited[to]){
            cout << "Add " << from << "-" << to << " to MST\n";
            visit(to, visited, adj, PQ);
        }
    }
}

// scc 1032-1075
#include <bits/stdc++.h>
using namespace std;

int V; // number of nodes
vector<vector<int>> adj; // the graph

stack<int> S; // stack
int dfs_counter = 0;
const int UNVISITED = -1;

vector<int> dfs_num(V, UNVISITED);
vector<int> dfs_min(V, UNVISITED);
vector<bool> on_stack(V, false);

void dfs(int u) {
    dfs_min[u] = dfs_num[u] = dfs_counter++;
    S.push(u);
    on_stack[u] = true;
    for (auto v: adj[u]) {
        if (dfs_num[v] == UNVISITED) {
            dfs(v);
            dfs_min[u] = min(dfs_min[u], dfs_min[v]);
        }
        else if (on_stack[v]) // only on_stack can use back edge
            dfs_min[u] = min(dfs_min[u], dfs_num[v]);
    }
    if (dfs_min[u] == dfs_num[u]) { // output result
        cout << "SCC: ";
        int v = -1;
        while (v != u) { // output SCC starting in u
            v = S.top(); S.pop(); on_stack[v] = false;
            cout << v << " ";
        }
        cout << endl;
    }
}

void scc() {
    for (int i = 0; i < V; i++) {
        if (dfs_num[i] == UNVISITED)
            dfs(i); // on next slide
    }
}

// spfa 1077-1104
#include <bits/stdc++.h>
using namespace std;

int V; // number of nodes
int INF; // a sufficiently large number
vector<vector<pair<int, int>>> adj; // the graph

void spfa(int start) {
    vector<int> dist (V, INF);
    queue<int> Q;
    vector<bool> inQ (V, false);
    dist[start] = 0; Q.push(start); inQ[start] = true;
    while (!Q.empty()) {
        int v = Q.front(); Q.pop(); inQ[v] = false;
        for (auto p: adj[v]) {
            int u = p.first;
            int w = p.second;
            if (dist[u] > dist[v] + w) {
                dist[u] = dist[v] + w;
                if (!inQ[u]) {
                    Q.push(u);
                    inQ[u] = true;
                }
            }
        }
    }
}

// toposort 1106-1121
#include <bits/stdc++.h>
using namespace std;

int V; // number of nodes
vector<vector<int>> adj; // the graph
vector<bool> visited(V, false);
deque<int> ts; // the final topological sort

void dfs(int v) { // modified dfs for toposort
    visited[v] = true;
    for (int u: adj[v])
        if (!visited[u])
            dfs(u);
    ts.push_front(v);
}

// unionfind 1123-1173
#include<bits/stdc++.h>

using namespace std;


class UnionFind {
    private:
        vector<int> parent, rank;
    public: 
        UnionFind(int N) {
            rank.assign(N, 0);
            parent.assign(N, 0);
            for (int i = 0; i < N; i++) parent[i] = i;
        }
        int findSet(int i) {
            if (parent[i] == i)
                return i;
            else     // path compression
                return parent[i] = findSet(parent[i]);
        }
        bool isSameSet(int i, int j) {
            return findSet(i) == findSet(j);
        }
        void unionSet(int i, int j) {
            if (!isSameSet(i, j)) {
                int x = findSet(i), y = findSet(j);
                if (rank[x] > rank[y])
                    parent[y] = x;
                else {
                    parent[x] = y;
                    if (rank[x] == rank[y]) rank[y]++;
                }
            }
        }
};


int main() {
    auto u = UnionFind(5);
    u.unionSet(1,4);
    cout << u.findSet(1) << endl;
    cout << u.findSet(4) << endl;
    u.unionSet(1,3);
    cout << u.findSet(4) << endl;
    cout << u.findSet(3) << endl;
    cout << u.findSet(2) << endl;
    cout << u.findSet(1) << endl;
    cout << u.findSet(0) << endl;
}

// freestylo

Sweeping Line Algorithm
Problem: The Sweeping Line algorithm is a technique used for solving various geometric problems such as:

Finding intersections of line segments.
Closest pair of points in 2D space. It works by sweeping a vertical line across the plane and maintaining relevant data (e.g., active line segments or points) in an efficient data structure.
Example: Line Segment Intersection Detection

Explanation:
This simple implementation checks for intersections between line segments using a brute force approach.
The sweeping line approach can be optimized with balanced data structures (like sets) and event-based sweeping where you process endpoints of segments in sorted order.


#include <iostream>
#include <set>
#include <vector>
#include <algorithm>

using namespace std;

// Structure to represent a point
struct Point {
    int x, y;
};

// Structure to represent a line segment
struct Segment {
    Point p1, p2;
};

// Helper function to find the orientation of the ordered triplet (A, B, C)
int orientation(Point A, Point B, Point C) {
    int val = (B.y - A.y) * (C.x - B.x) - (B.x - A.x) * (C.y - B.y);
    if (val == 0) return 0;   // Collinear
    return (val > 0) ? 1 : -1;  // Clockwise or counterclockwise
}

// Function to check if two line segments intersect
bool doIntersect(Segment s1, Segment s2) {
    int o1 = orientation(s1.p1, s1.p2, s2.p1);
    int o2 = orientation(s1.p1, s1.p2, s2.p2);
    int o3 = orientation(s2.p1, s2.p2, s1.p1);
    int o4 = orientation(s2.p1, s2.p2, s1.p2);

    if (o1 != o2 && o3 != o4) return true;  // General case

    return false;  // No intersection
}

// Sweeping line algorithm to detect intersections
bool sweepingLine(vector<Segment>& segments) {
    int n = segments.size();
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (doIntersect(segments[i], segments[j])) {
                return true;  // Found an intersection
            }
        }
    }
    return false;  // No intersections
}

int main() {
    vector<Segment> segments = {
        {{1, 1}, {10, 1}},
        {{1, 2}, {10, 2}},
        {{5, 0}, {5, 3}}
    };

    if (sweepingLine(segments)) {
        cout << "Segments intersect!" << endl;
    } else {
        cout << "No intersections found." << endl;
    }

    return 0;
}

====================

Graham Scan
Problem: The Graham Scan algorithm is used to find the convex hull of a set of points. The convex hull is the smallest convex polygon that contains all the given points. Graham Scan is efficient for this problem and sorts the points by polar angle and then constructs the hull by traversing the sorted points.

#include <iostream>
#include <vector>
#include <algorithm>
#include <stack>
using namespace std;

// Structure to represent a point
struct Point {
    double x, y;
};

// Function to find the orientation of the ordered triplet (A, B, C)
// Returns 0 if collinear, 1 if counterclockwise, -1 if clockwise
int orientation(Point A, Point B, Point C) {
    double val = (B.y - A.y) * (C.x - B.x) - (B.x - A.x) * (C.y - B.y);
    if (val == 0) return 0;
    return (val > 0) ? -1 : 1;  // Clockwise or Counterclockwise
}

// Function to compute the squared distance between two points
double dist(Point A, Point B) {
    return (A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y);
}

// Comparator function for sorting by polar angle and distance from reference point
bool compare(Point p1, Point p2, Point p0) {
    int o = orientation(p0, p1, p2);
    if (o == 0) return dist(p0, p1) < dist(p0, p2);
    return (o == 1);
}

// Function to implement the Graham Scan algorithm
vector<Point> grahamScan(vector<Point>& points) {
    int n = points.size();
    if (n < 3) return {};

    // Find the point with the lowest y-coordinate (and the leftmost one if there are ties)
    Point p0 = points[0];
    for (int i = 1; i < n; i++) {
        if (points[i].y < p0.y || (points[i].y == p0.y && points[i].x < p0.x)) {
            p0 = points[i];
        }
    }

    // Sort points based on polar angle with p0 as the reference
    sort(points.begin(), points.end(), [p0](Point p1, Point p2) {
        return compare(p1, p2, p0);
    });

    // Initialize the convex hull stack
    stack<Point> hull;
    hull.push(points[0]);
    hull.push(points[1]);

    // Process remaining points
    for (int i = 2; i < n; i++) {
        while (hull.size() > 1) {
            Point second = hull.top(); hull.pop();
            Point first = hull.top();
            if (orientation(first, second, points[i]) != -1) {
                hull.push(second);
                break;
            }
        }
        hull.push(points[i]);
    }

    // Convert stack to vector for output
    vector<Point> convexHull;
    while (!hull.empty()) {
        convexHull.push_back(hull.top());
        hull.pop();
    }

    return convexHull;
}

int main() {
    vector<Point> points = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {0, 3}, {3, 0}};
    vector<Point> convexHull = grahamScan(points);

    cout << "Points in the convex hull:" << endl;
    for (const auto& point : convexHull) {
        cout << "(" << point.x << ", " << point.y << ")" << endl;
    }

    return 0;
}

====================

A CCW (counterclockwise) function is used to determine the orientation of three points. It helps in checking whether the points form a counterclockwise turn, a clockwise turn, or are collinear. This is particularly useful in computational geometry for problems like convex hull construction or polygon orientation.

struct Point {
    double x, y;
};

int ccw(Point A, Point B, Point C) {
    // Calculate the cross product of vector AB and vector AC
    double val = (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);

    if (val > 0) {
        return 1; // Counterclockwise
    } else if (val < 0) {
        return -1; // Clockwise
    } else {
        return 0; // Collinear
    }
}

int main() {
    Point A, B, C;

    cout << "Enter the coordinates of point A (x y): ";
    cin >> A.x >> A.y;
    cout << "Enter the coordinates of point B (x y): ";
    cin >> B.x >> B.y;
    cout << "Enter the coordinates of point C (x y): ";
    cin >> C.x >> C.y;

    int orientation = ccw(A, B, C);

    if (orientation == 1) {
        cout << "The points are in counterclockwise order." << endl;
    } else if (orientation == -1) {
        cout << "The points are in clockwise order." << endl;
    } else {
        cout << "The points are collinear." << endl;
    }

    return 0;
}

====================

**Inclusion-Exclusion Principle**

**Problem:**  
The inclusion-exclusion principle is a combinatorial method used to count the number of elements in the union of overlapping sets by accounting for overlaps. It is used to avoid over-counting elements that belong to multiple sets.

**Use Cases:**  
- Solving counting problems where multiple conditions overlap.
- Problems involving set operations like unions and intersections.
- Calculating probabilities and counting distinct outcomes in problems with multiple constraints.

**Formula:**  
For two sets A and B:
\[ |A \cup B| = |A| + |B| - |A \cap B| \]

For three sets A, B, and C:
\[ |A \cup B \cup C| = |A| + |B| + |C| - |A \cap B| - |A \cap C| - |B \cap C| + |A \cap B \cap C| \]

**Time Complexity:**  
- **O(2^n)** for a general case with n sets, since all subsets need to be considered in larger cases.

**Space Complexity:**  
- **O(1)** for small sets, but can grow depending on how many sets or intersections are involved.

====================

**Segment Tree**

**Problem:**  
A segment tree is a data structure used to efficiently answer range queries (such as sum, minimum, maximum, etc.) and handle point updates on an array. It divides the array into segments and builds a tree to store information about these segments, allowing for fast query and update operations.

**Use Cases:**  
- Solving range query problems such as sum, minimum, maximum, or greatest common divisor over a subarray.
- Handling dynamic array updates while still answering queries efficiently.

**Time Complexity:**  
- **Build:** O(n), where n is the size of the array.
- **Query:** O(log n), for answering range queries.
- **Update:** O(log n), for point updates on the array.

**Space Complexity:**  
- **O(2n),** since a segment tree generally requires space proportional to twice the size of the input array.

====================

Minimum Cut

Problem:
The minimum cut problem involves finding the smallest set of edges that, when removed, disconnects the source vertex from the sink vertex in a flow network. It is closely related to the max flow problem because of the Max-Flow Min-Cut Theorem, which states that the maximum flow in a network is equal to the capacity of the minimum cut.

Use Cases:

Solving network reliability problems, where you need to find the weakest points of a network.
Determining the minimum number of edges to remove to separate a graph into disjoint parts.
Time Complexity:

Using Edmonds-Karp or Dinic's Algorithm for max flow, the minimum cut can be found in the same time:
O(V * E²) for Edmonds-Karp.
O(V² * E) for Dinic's Algorithm.
Space Complexity:

O(V + E), for storing the residual graph and related information.

====================

**Bipartite Matching**

**Problem:**  
Bipartite matching involves finding the maximum matching in a bipartite graph, which is a set of edges such that no two edges share a vertex. The goal is to match the maximum number of vertices from one set to the other.

**Use Cases:**  
- Solving matching problems in bipartite graphs, such as job assignments, task allocation, and stable marriages.
- Problems where items in two disjoint sets must be paired optimally.

**Time Complexity:**  
- **Hungarian Algorithm:** O(V³), where V is the number of vertices.
- **Hopcroft-Karp Algorithm (for unweighted bipartite matching):** O(√V * E), where E is the number of edges and V is the number of vertices.

**Space Complexity:**  
- **O(V + E),** for storing the graph and matching results.

====================

Edmonds-Karp Algorithm

Problem:
Edmonds-Karp is an implementation of the Ford-Fulkerson algorithm using BFS to find the shortest augmenting path in terms of the number of edges. This makes it a more structured and reliable version of Ford-Fulkerson.

Use Cases:

Efficient for solving max flow problems with smaller graphs or where shortest augmenting paths are useful.
Useful in situations where Ford-Fulkersons runtime is poor due to large capacities.
Time Complexity:

O(V * E²), where V is the number of vertices and E is the number of edges. It improves Ford-Fulkersons performance by bounding the number of augmenting paths.
Space Complexity:

O(V + E), for storing the graph and residual capacities.

====================

Ford-Fulkerson Algorithm

Problem:
Ford-Fulkerson is an algorithm used to compute the maximum flow in a flow network. It repeatedly finds augmenting paths from the source to the sink, along which flow can be pushed, and increases the total flow until no more augmenting paths can be found.

Use Cases:

Solving maximum flow problems in networks.
Problems like bipartite matching, project selection, and circulation with demands.
Time Complexity:

O(E * F), where E is the number of edges and F is the maximum flow in the network. This time complexity depends on the capacities and is not polynomial in the worst case because F can be large.
Space Complexity:

O(V + E), for storing the graph and residual capacities.


====================

Max Flow

Problem:
The Max Flow problem is about finding the maximum amount of flow that can be sent from a source vertex to a sink vertex in a flow network, where each edge has a capacity and the flow must respect these capacities. The flow entering a node must equal the flow leaving the node, except for the source and sink.

Use Cases:

Solving network flow problems such as data routing, transportation, or bipartite matching.
Problems where you need to find the maximum possible throughput between two points in a network.
Time Complexity:

Ford-Fulkerson: O(E * F), where E is the number of edges and F is the maximum flow in the network.
Edmonds-Karp (BFS-based Ford-Fulkerson): O(V * E²), where V is the number of vertices and E is the number of edges.
Dinic's Algorithm: O(V² * E) for general graphs, O(E * √V) for bipartite graphs.
Space Complexity:

O(V + E), for storing the graph and residual capacities.

====================

Lowest Common Ancestor (LCA)

Problem:
The LCA of two nodes in a binary tree is the deepest node that is an ancestor of both nodes. It's used in tree-based problems where we need to find a common parent of two nodes in the least amount of time.

Use Cases:

Finding relationships between nodes in a tree.
Solving range queries in trees.
Optimizing search for hierarchical structures (e.g., family trees, organizational charts).
Time Complexity:

Naive approach (traversing both nodes' paths to the root): O(N), where N is the number of nodes.
Optimized approach (Binary Lifting/Preprocessing with Euler Tour and RMQ): O(log N) for queries after O(N log N) preprocessing.
Space Complexity:

Naive approach: O(1) (additional space if recursive stack depth is considered: O(H), where H is the height of the tree).
Optimized approach (Binary Lifting): O(N log N) for storing ancestors at each level for fast retrieval.

====================

Kruskals Algorithm

Problem:
Kruskals algorithm is used to find the Minimum Spanning Tree (MST) of a connected, undirected graph. It works by selecting the smallest edge that doesnt form a cycle until all vertices are included in the tree.

Use Cases:

Network design (e.g., connecting cities with the least amount of cable or road).
Time Complexity:

O(E log E) where E is the number of edges (because of sorting the edges).
Space Complexity:

O(V), where V is the number of vertices (for storing disjoint sets/union-find data structure).

===================

Prims Algorithm

Problem:
Prims algorithm is another approach to finding the Minimum Spanning Tree (MST) of a connected, undirected graph. It starts from an arbitrary vertex and grows the MST by repeatedly adding the smallest edge that connects a vertex in the tree to a vertex outside the tree.

Use Cases:

Network design problems (similar to Kruskals).
Efficient in dense graphs because it grows the MST progressively from one vertex.
Time Complexity:

With adjacency list and binary heap: O(E log V), where E is the number of edges and V is the number of vertices.
With adjacency matrix: O(V²), suitable for dense graphs.
Space Complexity:

O(V) for storing the MST and the priority queue.

===================

Tarjans Algorithm for Strongly Connected Components (SCC)

Problem:
Tarjans algorithm is used to find all Strongly Connected Components (SCCs) in a directed graph. A strongly connected component is a maximal subgraph where every pair of vertices is reachable from each other.

Use Cases:

Identifying cycles in directed graphs.
Optimizing dependency analysis in compilers (e.g., determining which functions or modules are interdependent).
Problems where you need to collapse SCCs into a single vertex for simplification (e.g., 2-SAT problems).
Time Complexity:

O(V + E), where V is the number of vertices and E is the number of edges. This is because the algorithm performs a depth-first search (DFS).
Space Complexity:

O(V), for storing the stack, low-link values, and other auxiliary arrays.

===================

Articulation Points & Bridges

Problem:

Articulation Points (Cut Vertices): These are vertices in a graph, removing which increases the number of connected components. In other words, they are critical for maintaining the graphs connectivity.
Bridges (Cut Edges): These are edges, removing which increases the number of connected components in the graph. They are critical edges whose removal disconnects parts of the graph.
Use Cases:

Network design and analysis (finding vulnerable points in networks, such as routers or links that are critical for connectivity).
Fault tolerance in communication networks.
Graph-based problems that require understanding the connectivity and robustness of the structure (e.g., road networks, social networks).
Time Complexity:

O(V + E) where V is the number of vertices and E is the number of edges. This is achieved using Depth-First Search (DFS) and computing discovery and low-link values for each node.
Space Complexity:

O(V) for storing discovery times, low-link values, and other auxiliary arrays.

===================

Bipartite Graph Checking

Problem:
A bipartite graph is a graph that can be divided into two disjoint sets such that every edge connects a vertex in one set to a vertex in the other set. The problem is to determine if a given graph is bipartite.

Use Cases:

Checking if a graph can be colored with two colors without any two adjacent nodes having the same color (2-coloring problem).
Solving matching problems in bipartite graphs.
Detecting whether a graph contains odd-length cycles (a graph is not bipartite if it contains an odd-length cycle).
Time Complexity:

O(V + E), where V is the number of vertices and E is the number of edges. This is achieved using BFS or DFS to color the graph.
Space Complexity:

O(V), for storing colors and other auxiliary arrays.

===================

Floyd-Warshall Algorithm

Problem:
The Floyd-Warshall algorithm is used to find the shortest paths between all pairs of vertices in a weighted graph. It works for both directed and undirected graphs and can handle negative weights, but not negative weight cycles.

Use Cases:

Solving all-pairs shortest path problems.
Detecting negative weight cycles in a graph.
Calculating the transitive closure of a graph.
Time Complexity:

O(V³), where V is the number of vertices, since it iterates through each pair of vertices and updates the shortest path between them.
Space Complexity:

O(V²), for storing the distance matrix.

===================

Bellman-Ford Algorithm

Problem:
The Bellman-Ford algorithm is used to find the shortest paths from a single source to all other vertices in a weighted graph. It can handle graphs with negative weight edges and can detect negative weight cycles.

Use Cases:

Single-source shortest path problems in graphs with negative weights.
Detecting negative weight cycles in graphs.
Solving shortest path problems where Dijkstra's algorithm cannot be used due to negative weights.
Time Complexity:

O(V * E), where V is the number of vertices and E is the number of edges. The algorithm relaxes all edges V - 1 times.
Space Complexity:

O(V), for storing distances from the source vertex.

===================

Dijkstra's Algorithm

Problem:
Dijkstra's algorithm is used to find the shortest paths from a single source vertex to all other vertices in a graph with non-negative edge weights.

Use Cases:

Single-source shortest path problems where all edge weights are non-negative.
Optimizing shortest path queries in weighted graphs, such as routing problems.
Time Complexity:

O((V + E) log V) using a priority queue (where V is the number of vertices and E is the number of edges).
O(V²) when using an adjacency matrix.
Space Complexity:

O(V), for storing distances and the priority queue.

===================

**Topological Sort (Toposort)**

**Problem:**  
Topological sorting is used to order the vertices of a Directed Acyclic Graph (DAG) in a linear order such that for every directed edge u -> v, vertex u comes before vertex v in the ordering.

**Use Cases:**  
- Solving problems where tasks or events must be completed in a specific sequence (without real-life context, focusing on dependency ordering in algorithmic problems).
- Scheduling tasks in a DAG.
- Finding order of compilation in a system of interdependent modules.

**Time Complexity:**  
- **O(V + E),** where V is the number of vertices and E is the number of edges. This is achieved using DFS or Kahns algorithm (BFS-based approach).

**Space Complexity:**  
- **O(V),** for storing the result and auxiliary data structures like the in-degree array or stack.

===================

Edit Distance (Levenshtein Distance)

Problem:
The edit distance between two strings is the minimum number of operations (insertions, deletions, or substitutions) required to convert one string into the other.

Use Cases:

String comparison problems where you need to measure similarity between two sequences.
Solving problems like correcting spelling mistakes, comparing DNA sequences, or text alignment.
Time Complexity:

O(m * n), where m and n are the lengths of the two strings. This is achieved using dynamic programming to compute the minimum operations.
Space Complexity:

O(m * n), for storing the dynamic programming table.
Can be optimized to O(min(m, n)) space by storing only the current and previous rows.

===================

**Aho-Corasick Algorithm**

**Problem:**  
Aho-Corasick is used for searching multiple patterns simultaneously in a given text. It builds a trie from a set of patterns and processes the text in a single pass, outputting all occurrences of the patterns in the text.

**Use Cases:**  
- Solving multiple pattern matching problems efficiently.
- Handling dictionary matching problems, where a large number of patterns need to be found in a large text.

**Time Complexity:**  
- **O(n + m + z),** where n is the length of the text, m is the total length of all patterns, and z is the number of matches found. The time complexity is linear in the input size.

**Space Complexity:**  
- **O(m),** for storing the trie and the failure links, where m is the total length of all patterns.

===================

**Trie (Prefix Tree)**

**Problem:**  
A trie is a tree-like data structure used to store a dynamic set of strings where each node represents a common prefix. It allows efficient searching, insertion, and deletion of strings, particularly when dealing with prefixes.

**Use Cases:**  
- Solving prefix-matching problems, such as autocomplete or dictionary lookup.
- Efficiently storing and retrieving a large set of strings or words.
- Finding all words in a text that share a common prefix.

**Time Complexity:**  
- **Insertion/Search:** O(L), where L is the length of the string (word) being inserted or searched.
- **Deletion:** O(L), similar to insertion and search.

**Space Complexity:**  
- **O(N * L),** where N is the number of strings and L is the average length of the strings. This is because each character of every string is stored in the trie.

===================

**KMP (Knuth-Morris-Pratt) Algorithm**

**Problem:**  
KMP is used for finding occurrences of a pattern string within a text string. It preprocesses the pattern to create a partial match table (also called the "lps" array) that allows the algorithm to skip unnecessary comparisons during the search.

**Use Cases:**  
- Solving exact pattern matching problems efficiently in strings.
- Problems where multiple pattern occurrences need to be found without rechecking the same parts of the text.

**Time Complexity:**  
- **O(n + m),** where n is the length of the text and m is the length of the pattern. The preprocessing takes O(m), and the search takes O(n).

**Space Complexity:**  
- **O(m),** for storing the lps array used for skipping characters during the search.

===================

Knapsack Problem

Problem:
The knapsack problem involves selecting items with given weights and values to include in a knapsack of limited capacity, such that the total value is maximized without exceeding the capacity. There are different variations, but the most common one is the 0/1 knapsack, where each item can either be included or excluded.

Use Cases:

Solving optimization problems where resources are limited and a selection must be made to maximize value.
Commonly used in dynamic programming challenges.
Time Complexity:

O(n * W), where n is the number of items and W is the knapsacks capacity. This applies to the dynamic programming approach.
Space Complexity:

O(n * W), for the dynamic programming table.
Can be optimized to O(W) by using a single-dimensional DP array.

===================

**Longest Increasing Subsequence (LIS)**

**Problem:**  
The Longest Increasing Subsequence problem involves finding the length of the longest subsequence in a sequence of numbers where each number is greater than the previous one in the subsequence.

**Use Cases:**  
- Solving problems related to sequence analysis and optimizing subsequence-related queries.
- Dynamic programming and combinatorial optimization challenges.

**Time Complexity:**  
- **O(n²),** using dynamic programming where n is the length of the sequence.
- **O(n log n),** using a more optimized approach with binary search.

**Space Complexity:**  
- **O(n),** for storing the lengths of the subsequences or the dynamic programming table.

===================

**Prefix Sum**

**Problem:**  
The prefix sum is a technique where you preprocess an array to quickly compute the sum of elements in a given range. It involves creating an auxiliary array where each element at index i stores the sum of elements from the start of the array to i.

**Use Cases:**  
- Solving range query problems efficiently (sum of elements between two indices).
- Problems where repeated range queries need to be answered quickly.

**Time Complexity:**  
- **O(n)** for preprocessing the prefix sum array.
- **O(1)** for answering each range sum query.

**Space Complexity:**  
- **O(n),** for storing the prefix sum array.

===================

**Dynamic Programming (DP)**

**Problem:**  
Dynamic programming is an optimization technique used to solve problems by breaking them down into simpler overlapping subproblems and storing the solutions to these subproblems to avoid redundant calculations.

**Use Cases:**  
- Solving problems involving optimization, such as shortest paths, knapsack, longest common subsequence, and more.
- Problems with overlapping subproblems and optimal substructure, where the problem can be broken down into smaller, similar problems.

**Time Complexity:**  
- Depends on the specific problem, typically **O(n * m)** for problems with two dimensions or **O(n)** for one-dimensional problems, where n and m represent the dimensions of the problem.

**Space Complexity:**  
- Depends on the problem, but often **O(n)** or **O(n * m)** for storing solutions to subproblems.

===================

Meet in the Middle

Problem:
Meet in the middle is a technique used to solve problems with large input sizes by dividing the problem into two smaller subproblems, solving each, and then combining the results to find the solution. It is often used when the input size is too large for brute force but can be split into manageable parts.

Use Cases:

Solving problems like subset-sum, knapsack, or combinatorial problems where brute force would be too slow.
Problems that can be split into two independent parts and then combined to form the solution.
Time Complexity:

O(2^(n/2)), where n is the size of the input, as the problem is divided into two parts.
Space Complexity:

O(2^(n/2)), for storing the solutions to the subproblems.

===================

**Greedy Scheduling**

**Problem:**  
Greedy scheduling algorithms are used to solve optimization problems where a series of tasks need to be scheduled to maximize or minimize a certain objective (such as minimizing completion time or maximizing the number of tasks completed). The greedy approach makes local, optimal choices at each step in the hope of finding the global optimum.

**Use Cases:**  
- Interval scheduling maximization: selecting the maximum number of non-overlapping intervals.
- Problems like activity selection, job scheduling with deadlines, or minimizing the sum of completion times.

**Time Complexity:**  
- **O(n log n),** where n is the number of tasks. Sorting the tasks based on some criteria (like start time or finish time) is the most expensive step.

**Space Complexity:**  
- **O(1)** or **O(n),** depending on whether you need to store additional information about selected tasks.

===================

**Square Root Decomposition**

**Problem:**  
Square root decomposition is a technique used to solve range query problems efficiently by dividing the input data into blocks of size approximately equal to the square root of the total input size. Preprocessing is done to store partial results for each block, allowing faster query operations while minimizing the need for recalculations.

**Use Cases:**  
- Solving range queries like sum, minimum, or maximum over subarrays in a faster way than brute force.
- Problems where updates and queries on a static array are frequent.

**Time Complexity:**  
- **Preprocessing:** O(n), where n is the size of the input.
- **Query:** O(√n), as you need to process a constant number of blocks.
- **Update:** O(1) to O(√n), depending on whether you only update a block or recompute the blocks value.

**Space Complexity:**  
- **O(√n),** for storing precomputed information for each block.

===================

// lazy segment tree
// Works for: range add, range sum query, range update, point query
#include <iostream>
#include <vector>
using namespace std;

class SegmentTree {
    vector<long long> tree, lazy;
    int n;

public:
    SegmentTree(int size) {
        n = size;
        tree.assign(4 * n, 0);
        lazy.assign(4 * n, 0);
    }

    void propagate(int node, int start, int end) {
        if (lazy[node] != 0) {
            tree[node] += (end - start + 1) * lazy[node];
            if (start != end) {
                lazy[2 * node + 1] += lazy[node];
                lazy[2 * node + 2] += lazy[node];
            }
            lazy[node] = 0;
        }
    }

    void rangeUpdate(int node, int start, int end, int l, int r, int val) {
        propagate(node, start, end);
        if (start > r || end < l) return;

        if (start >= l && end <= r) {
            lazy[node] += val;
            propagate(node, start, end);
            return;
        }

        int mid = (start + end) / 2;
        rangeUpdate(2 * node + 1, start, mid, l, r, val);
        rangeUpdate(2 * node + 2, mid + 1, end, l, r, val);
        tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
    }

    long long rangeQuery(int node, int start, int end, int l, int r) {
        propagate(node, start, end);
        if (start > r || end < l) return 0;

        if (start >= l && end <= r) return tree[node];

        int mid = (start + end) / 2;
        long long leftSum = rangeQuery(2 * node + 1, start, mid, l, r);
        long long rightSum = rangeQuery(2 * node + 2, mid + 1, end, l, r);
        return leftSum + rightSum;
    }

    void update(int l, int r, int val) {
        rangeUpdate(0, 0, n - 1, l, r, val);
    }

    long long query(int l, int r) {
        return rangeQuery(0, 0, n - 1, l, r);
    }
};

int main() {
    int n, q;
    cout << "Enter the number of elements: ";
    cin >> n;
    SegmentTree segTree(n);

    cout << "Enter the number of queries: ";
    cin >> q;

    while (q--) {
        int type;
        cout << "Enter query type (1 for add, 2 for sum): ";
        cin >> type;

        if (type == 1) {
            int l, r, val;
            cout << "Enter the range (l, r) and value to add: ";
            cin >> l >> r >> val;
            segTree.update(l, r, val);
        } else if (type == 2) {
            int l, r;
            cout << "Enter the range (l, r) to query the sum: ";
            cin >> l >> r;
            cout << "Sum on segment: " << segTree.query(l, r) << endl;
        }
    }

    return 0;
}

// probably better lazy segment tree:
#include <iostream>
#include <vector>
using namespace std;

class SegmentTree {
    vector<long long> tree, lazy;
    int n;

public:
    SegmentTree(int size) {
        n = size;
        tree.assign(4 * n, 0);
        lazy.assign(4 * n, 0);
    }

    // Function to propagate the lazy values
    void propagate(int node, int start, int end) {
        if (lazy[node] != 0) {
            tree[node] += (end - start + 1) * lazy[node];  // Apply the lazy value to the current segment
            if (start != end) {
                lazy[2 * node + 1] += lazy[node];  // Mark lazy for the left child
                lazy[2 * node + 2] += lazy[node];  // Mark lazy for the right child
            }
            lazy[node] = 0;  // Clear the lazy value for the current node
        }
    }

    // Range update to add value to a segment
    void rangeUpdate(int node, int start, int end, int l, int r, int val) {
        propagate(node, start, end);  // Ensure any pending updates are applied
        if (start > r || end < l) return;  // Out of range

        if (start >= l && end <= r) {  // Segment is fully within range
            lazy[node] += val;
            propagate(node, start, end);
            return;
        }

        int mid = (start + end) / 2;
        rangeUpdate(2 * node + 1, start, mid, l, r, val);
        rangeUpdate(2 * node + 2, mid + 1, end, l, r, val);
        tree[node] = tree[2 * node + 1] + tree[2 * node + 2];  // Recompute the value for the current node
    }

    // Point update to add value to a specific index
    void pointUpdate(int node, int start, int end, int idx, int val) {
        propagate(node, start, end);  // Apply any pending updates
        if (start == end) {  // Leaf node
            tree[node] += val;
            return;
        }

        int mid = (start + end) / 2;
        if (idx <= mid) {
            pointUpdate(2 * node + 1, start, mid, idx, val);
        } else {
            pointUpdate(2 * node + 2, mid + 1, end, idx, val);
        }
        tree[node] = tree[2 * node + 1] + tree[2 * node + 2];  // Recompute the value for the current node
    }

    // Range query to find the sum of elements in a segment
    long long rangeQuery(int node, int start, int end, int l, int r) {
        propagate(node, start, end);  // Apply any pending updates
        if (start > r || end < l) return 0;  // Out of range

        if (start >= l && end <= r) return tree[node];  // Segment is fully within range

        int mid = (start + end) / 2;
        long long leftSum = rangeQuery(2 * node + 1, start, mid, l, r);
        long long rightSum = rangeQuery(2 * node + 2, mid + 1, end, l, r);
        return leftSum + rightSum;
    }

    // Point query to find the value at a specific index
    long long pointQuery(int node, int start, int end, int idx) {
        propagate(node, start, end);  // Apply any pending updates
        if (start == end) return tree[node];  // Leaf node

        int mid = (start + end) / 2;
        if (idx <= mid) {
            return pointQuery(2 * node + 1, start, mid, idx);
        } else {
            return pointQuery(2 * node + 2, mid + 1, end, idx);
        }
    }

    // Helper functions to call the segment tree functions
    void updateRange(int l, int r, int val) {
        rangeUpdate(0, 0, n - 1, l, r, val);
    }

    void updatePoint(int idx, int val) {
        pointUpdate(0, 0, n - 1, idx, val);
    }

    long long queryRange(int l, int r) {
        return rangeQuery(0, 0, n - 1, l, r);
    }

    long long queryPoint(int idx) {
        return pointQuery(0, 0, n - 1, idx);
    }
};

int main() {
    int n, q;
    cout << "Enter the number of elements: ";
    cin >> n;
    SegmentTree segTree(n);

    cout << "Enter the number of queries: ";
    cin >> q;

    while (q--) {
        int type;
        cout << "Enter query type (1 for range add, 2 for point add, 3 for range sum, 4 for point query): ";
        cin >> type;

        if (type == 1) {
            int l, r, val;
            cout << "Enter the range (l, r) and value to add: ";
            cin >> l >> r >> val;
            segTree.updateRange(l, r, val);
        } else if (type == 2) {
            int idx, val;
            cout << "Enter the index and value to add: ";
            cin >> idx >> val;
            segTree.updatePoint(idx, val);
        } else if (type == 3) {
            int l, r;
            cout << "Enter the range (l, r) to query the sum: ";
            cin >> l >> r;
            cout << "Sum on segment: " << segTree.queryRange(l, r) << endl;
        } else if (type == 4) {
            int idx;
            cout << "Enter the index to query: ";
            cin >> idx;
            cout << "Value at index: " << segTree.queryPoint(idx) << endl;
        }
    }

    return 0;
}

// greedy scheduling

Idea is to always select the next possible event that ends as early as possible. This algorithm selects the following events:

It turns out that this algorithm always produces an optimal solution.
The reason for this is that it is always an optimal choice to first select an event that ends as early as possible.
After this, it is an optimal choice to select the next event using the same strategy, etc., until any other event cant be selected.
One way the algorithm works is to consider what happens if first select an event that ends later than the event that ends as early as possible.
Now, with having at most an equal number of choices how the next event can be selected.
Hence, selecting an event that ends later can never yield a better solution, and the greedy algorithm is correct.

// square root decomposition

#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

// Square Root Decomposition

int main() {
    // input data
    int n;
    cout << "Enter the size of the array: ";
    cin >> n;
    
    vector<int> a(n); // Initialize vector `a` with size `n`
    
    cout << "Enter the elements of the array: ";
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
    }

    // preprocessing
    int len = (int)sqrt(n + .0) + 1; // size of the block and the number of blocks
    vector<int> b(len, 0); // Initialize block sum array `b`
    
    for (int i = 0; i < n; ++i)
        b[i / len] += a[i];

    // answering the queries
    while (true) {
        int l, r;
        cout << "Enter the range [l, r] (or -1 -1 to exit): ";
        cin >> l >> r;

        if (l == -1 && r == -1) break; // Exit condition

        int sum = 0;
        int c_l = l / len, c_r = r / len; // Block numbers for l and r
        
        if (c_l == c_r) {
            // If l and r are in the same block
            for (int i = l; i <= r; ++i)
                sum += a[i];
        } else {
            // Sum from l to the end of its block
            for (int i = l, end = (c_l + 1) * len - 1; i <= end; ++i)
                sum += a[i];
            // Sum whole blocks between c_l and c_r
            for (int i = c_l + 1; i <= c_r - 1; ++i)
                sum += b[i];
            // Sum from the start of c_r block to r
            for (int i = c_r * len; i <= r; ++i)
                sum += a[i];
        }

        cout << "Sum in range [" << l << ", " << r << "] is " << sum << endl;
    }

    return 0;
}


// Minimum cut

#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <cstring>
using namespace std;

#define V 6 // Number of vertices in the graph

// A BFS function to check if there's a path from source 's' to sink 't'
// in the residual graph. It also fills parent[] to store the path.
bool bfs(int rGraph[V][V], int s, int t, int parent[]) {
    // A visited array to keep track of visited vertices
    bool visited[V];
    memset(visited, 0, sizeof(visited));

    // Create a queue for BFS
    queue<int> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;

    // Standard BFS loop
    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v = 0; v < V; v++) {
            if (!visited[v] && rGraph[u][v] > 0) {  // Check for unvisited vertex with positive capacity
                // If we found a connection to the sink, return true
                if (v == t) {
                    parent[v] = u;
                    return true;
                }
                q.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
    }
    // No path found
    return false;
}

// Ford-Fulkerson algorithm using Edmonds-Karp to find the max flow and identify the min-cut
int fordFulkerson(int graph[V][V], int s, int t) {
    int u, v;

    // Create a residual graph and initialize residual capacities with the original capacities
    int rGraph[V][V];
    for (u = 0; u < V; u++)
        for (v = 0; v < V; v++)
            rGraph[u][v] = graph[u][v];

    // This array stores the augmenting path found by BFS
    int parent[V];

    int max_flow = 0; // Initialize max flow

    // Augment the flow while there is a path from source to sink
    while (bfs(rGraph, s, t, parent)) {
        // Find the minimum residual capacity of the edges along the path filled by BFS
        int path_flow = INT_MAX;
        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            path_flow = min(path_flow, rGraph[u][v]);
        }

        // Update the residual capacities of the edges and reverse edges along the path
        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            rGraph[u][v] -= path_flow;
            rGraph[v][u] += path_flow;
        }

        // Add the path flow to the overall flow
        max_flow += path_flow;
    }

    // Now that we have the max flow, use BFS to find the vertices that are reachable from the source
    bool visited[V];
    memset(visited, false, sizeof(visited));
    queue<int> q;
    q.push(s);
    visited[s] = true;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v = 0; v < V; ++v) {
            if (!visited[v] && rGraph[u][v] > 0) {  // Check for unvisited vertex with positive residual capacity
                q.push(v);
                visited[v] = true;
            }
        }
    }

    // Print the edges which form the minimum cut
    cout << "The edges in the minimum cut are: \n";
    for (int u = 0; u < V; u++) {
        for (int v = 0; v < V; v++) {
            if (visited[u] && !visited[v] && graph[u][v] > 0) {
                cout << u << " - " << v << "\n";
            }
        }
    }

    return max_flow;  // Return the overall flow
}

int main() {
    // Create a graph with V vertices
    int graph[V][V] = {
        {0, 16, 13, 0, 0, 0},
        {0, 0, 10, 12, 0, 0},
        {0, 4, 0, 0, 14, 0},
        {0, 0, 9, 0, 0, 20},
        {0, 0, 0, 7, 0, 4},
        {0, 0, 0, 0, 0, 0}
    };

    int source = 0, sink = 5;

    cout << "The maximum possible flow is " << fordFulkerson(graph, source, sink) << endl;

    return 0;
}

// Polygon Area

#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

// Function to calculate the area of a polygon using the Shoelace Theorem
double polygonArea(const vector<pair<double, double>>& points) {
    int n = points.size(); // Number of vertices in the polygon
    double area = 0.0;

    // Apply the Shoelace formula
    for (int i = 0; i < n; i++) {
        int next = (i + 1) % n; // To ensure that the last vertex connects back to the first
        area += points[i].first * points[next].second;
        area -= points[i].second * points[next].first;
    }

    // The area is half of the absolute value of the result
    return fabs(area) / 2.0;
}

int main() {
    // Define the polygon as a vector of pairs (x, y)
    vector<pair<double, double>> polygon = {
        {0, 0}, {4, 0}, {4, 3}, {0, 3}
    };

    // Calculate and display the area
    double area = polygonArea(polygon);
    cout << "The area of the polygon is: " << area << endl;

    return 0;
}

// Point in polygon

#include <iostream>
#include <vector>
using namespace std;

// Structure to represent a point
struct Point {
    double x, y;
};

// Function to check if a point q lies on a line segment p1-p2
bool onSegment(Point p1, Point p2, Point q) {
    if (q.x <= max(p1.x, p2.x) && q.x >= min(p1.x, p2.x) &&
        q.y <= max(p1.y, p2.y) && q.y >= min(p1.y, p2.y)) {
        return true;
    }
    return false;
}

// To find the orientation of the ordered triplet (p1, p2, q)
// The function returns:
// 0 -> p1, p2 and q are collinear
// 1 -> Clockwise
// 2 -> Counterclockwise
int orientation(Point p1, Point p2, Point q) {
    double val = (p2.y - p1.y) * (q.x - p2.x) - (p2.x - p1.x) * (q.y - p2.y);
    if (val == 0) return 0;  // collinear
    return (val > 0) ? 1 : 2; // clockwise or counterclockwise
}

// Function to check if the line segment p1-p2 intersects with q1-q2
bool doIntersect(Point p1, Point p2, Point q1, Point q2) {
    // Find the four orientations needed for general and special cases
    int o1 = orientation(p1, p2, q1);
    int o2 = orientation(p1, p2, q2);
    int o3 = orientation(q1, q2, p1);
    int o4 = orientation(q1, q2, p2);

    // General case
    if (o1 != o2 && o3 != o4) {
        return true;
    }

    // Special cases:
    // p1, p2 and q1 are collinear and q1 lies on segment p1-p2
    if (o1 == 0 && onSegment(p1, p2, q1)) return true;

    // p1, p2 and q2 are collinear and q2 lies on segment p1-p2
    if (o2 == 0 && onSegment(p1, p2, q2)) return true;

    // q1, q2 and p1 are collinear and p1 lies on segment q1-q2
    if (o3 == 0 && onSegment(q1, q2, p1)) return true;

    // q1, q2 and p2 are collinear and p2 lies on segment q1-q2
    if (o4 == 0 && onSegment(q1, q2, p2)) return true;

    // Doesn't fall in any of the above cases
    return false;
}

// Function to check if the point p lies inside the polygon
bool isPointInPolygon(const vector<Point>& polygon, Point p) {
    int n = polygon.size();
    if (n < 3) return false; // A polygon must have at least 3 vertices

    // Create a point for the ray (a point far outside the polygon)
    Point extreme = {10000, p.y};

    // Count intersections of the ray with polygon edges
    int count = 0, i = 0;
    do {
        int next = (i + 1) % n;

        // Check if the point is on the segment
        if (doIntersect(polygon[i], polygon[next], p, extreme)) {
            if (orientation(polygon[i], polygon[next], p) == 0) {
                return onSegment(polygon[i], polygon[next], p);
            }
            count++;
        }
        i = next;
    } while (i != 0);

    // Return true if count is odd, false if even
    return (count % 2 == 1);
}

int main() {
    // Define the polygon as a vector of points
    vector<Point> polygon = {
        {0, 0}, {10, 0}, {10, 10}, {0, 10}
    };

    Point p = {5, 5}; // Point to check

    if (isPointInPolygon(polygon, p)) {
        cout << "Point (" << p.x << ", " << p.y << ") is inside the polygon.\n";
    } else {
        cout << "Point (" << p.x << ", " << p.y << ") is outside the polygon.\n";
    }

    return 0;
}

// Algorithm descriptions

1. **Lowest Common Ancestor (LCA)**  
   *Identify:* Tree structure, find deepest common ancestor of two nodes.  
   *Description:* Finds the lowest common ancestor of two nodes.

2. **Kruskals Algorithm**  
   *Identify:* Undirected graph, find minimum spanning tree (MST).  
   *Description:* Builds MST by adding smallest edges without cycles.

3. **Prims Algorithm**  
   *Identify:* Undirected graph, find MST, better for dense graphs.  
   *Description:* Grows MST from an arbitrary starting vertex.

4. **Eulers Totient Function**  
   *Identify:* Count numbers coprime with n in range [1, n].  
   *Description:* Counts integers coprime with a given number n.

5. **Tarjans Algorithm for SCC**  
   *Identify:* Directed graph, find strongly connected components (SCC).  
   *Description:* Finds SCCs using DFS and low-link values.

6. **Articulation Points & Bridges**  
   *Identify:* Graph, find critical vertices/edges that disconnect components.  
   *Description:* Identifies critical nodes/edges affecting graph connectivity.

7. **Bipartite Graph Checking**  
   *Identify:* Graph, check if it can be colored with two colors.  
   *Description:* Determines if a graph is bipartite (2-colorable).

8. **Floyd-Warshall Algorithm**  
   *Identify:* Weighted graph, find shortest paths between all pairs.  
   *Description:* Finds all-pairs shortest paths in a graph.

9. **Bellman-Ford Algorithm**  
   *Identify:* Graph with negative weights, single-source shortest paths.  
   *Description:* Finds shortest paths, detects negative weight cycles.

10. **Dijkstras Algorithm**  
   *Identify:* Weighted graph, no negative weights, single-source shortest paths.  
   *Description:* Finds shortest paths from a source in non-negative graphs.

11. **Topological Sort (Toposort)**  
   *Identify:* Directed Acyclic Graph (DAG), linear order of vertices.  
   *Description:* Produces linear order of vertices in a DAG.

12. **Counting Connected Components**  
   *Identify:* Graph, count the number of connected components.  
   *Description:* Counts disconnected subgraphs in an undirected graph.

13. **Edit Distance**  
   *Identify:* Compare strings, find minimum edits (insert, delete, replace).  
   *Description:* Finds the minimum number of edits between two strings.

14. **Aho-Corasick Algorithm**  
   *Identify:* Multiple pattern matching in a text.  
   *Description:* Matches multiple patterns in a text simultaneously.

15. **Trie**  
   *Identify:* Large set of strings, need prefix search or storage.  
   *Description:* Efficiently stores and searches strings using prefixes.

16. **KMP Algorithm**  
   *Identify:* Pattern matching, optimize search by skipping characters.  
   *Description:* Finds patterns in text using partial match table.

17. **Knapsack Problem**  
   *Identify:* Select items with weights and values, maximize total value.  
   *Description:* Maximizes value under a weight constraint.

18. **Longest Increasing Subsequence (LIS)**  
   *Identify:* Sequence, find longest subsequence with increasing elements.  
   *Description:* Finds the longest subsequence of increasing numbers.

19. **Prefix Sum**  
   *Identify:* Repeated sum queries on subarrays.  
   *Description:* Preprocesses array for quick range sum queries.

20. **Dynamic Programming (DP)**  
   *Identify:* Problem with overlapping subproblems and optimal substructure.  
   *Description:* Solves problems by breaking into simpler subproblems.

21. **Meet in the Middle**  
   *Identify:* Large input size, break into smaller subproblems.  
   *Description:* Divides problem into two halves and combines results.

22. **Greedy Scheduling**  
   *Identify:* Scheduling tasks, optimize resource allocation.  
   *Description:* Makes locally optimal choices for scheduling.

23. **Square Root Decomposition**  
   *Identify:* Array, range queries with updates.  
   *Description:* Divides array into blocks for efficient range queries.

24. **2-SAT**  
   *Identify:* Boolean formula, clauses with two literals each.  
   *Description:* Determines satisfiability of 2-literal clauses.

25. **3-SAT**  
   *Identify:* Boolean formula, clauses with three literals each.  
   *Description:* Determines satisfiability of 3-literal clauses (NP-complete).

26. **Max Flow**  
   *Identify:* Network, maximize flow from source to sink.  
   *Description:* Finds maximum flow in a flow network.

27. **Ford-Fulkerson Algorithm**  
   *Identify:* Network, find max flow using augmenting paths.  
   *Description:* Finds max flow by iteratively adding flow.

28. **Edmonds-Karp Algorithm**  
   *Identify:* Max flow problem, use BFS to find augmenting paths.  
   *Description:* BFS-based Ford-Fulkerson to find max flow.

29. **Bipartite Matching**  
   *Identify:* Bipartite graph, find maximum matching between two sets.  
   *Description:* Matches vertices from two disjoint sets.

30. **Minimum Cut**  
   *Identify:* Network, find minimum edge set to disconnect flow.  
   *Description:* Finds the smallest edge set disconnecting source from sink.

31. **Segment Tree**  
   *Identify:* Range queries and updates on an array.  
   *Description:* Efficiently handles range queries and updates.

32. **Inclusion-Exclusion Principle**  
   *Identify:* Counting problems with overlapping sets.  
   *Description:* Counts elements in overlapping sets using inclusion-exclusion.

33. **Polygon Area**  
   *Identify:* Given vertices of a polygon, find its area.  
   *Description:* Calculates area of a polygon using its vertices.

34. **CCW (Counterclockwise)**  
   *Identify:* Geometric problems, check if points form a counterclockwise turn.  
   *Description:* Determines orientation of three points (CCW or CW).

35. **Graham Scan**  
   *Identify:* Given points, find the convex hull.  
   *Description:* Finds the convex hull of a set of points.

36. **Sweeping Line Algorithm**  
   *Identify:* Geometric problems, detect intersections between line segments.  
   *Description:* Processes geometric events by sweeping a line across points.

37. **Union-Find (Disjoint Set Union - DSU)**  
   *Identify:* Dynamic connectivity problems, check or unite sets of elements.  
   *Description:* Tracks and merges disjoint sets efficiently.

38. **Inclusion-Exclusion Principle**  
   *Identify:* Counting problems with multiple overlapping conditions or sets.  
   *Description:* Corrects overcounting by adding and subtracting intersections.