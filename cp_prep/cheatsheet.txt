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

// freestylo - my own algos!
