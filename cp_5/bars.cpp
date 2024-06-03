#include <iostream>
#include "bits/stdc++.h"

using namespace std;

class Graph {
    int V;
    vector<vector<int>> adjList;
    int solution;
public:
    Graph(int V);
    void addEdge(int v, int w);
    void DFS(int v, vector<bool>& visited, int d);
};

Graph::Graph(int V) {
    this->V = V;
    this->solution = 0;
    adjList.resize(V);
}

void Graph::addEdge(int v, int w) {
    adjList[v].push_back(w);
    adjList[w].push_back(v);
}

void Graph::DFS(int v, vector<bool>& visited, int d) {
    if (d >= 0) {
        visited[v] = true;
        // cout << v << " ";
        for (int i : adjList[v]) {
            if (!visited[i]) {
                DFS(i, visited, d - 1);
            }
        }
    }
}

int main() {
    int n, m, s, d; cin >> n >> m >> s >> d;

    Graph *g = new Graph(n + 1);

    for (int i = 0; i < m; i++) {
        int a, b; cin >> a >> b;
        g->addEdge(a, b);
    }

    vector<bool> visited(n + 1, false);

    g->DFS(s, visited, d);

    int countTrue = count(visited.begin(), visited.end(), true);
    cout << countTrue;

    return 0;
}