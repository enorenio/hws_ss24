#include <iostream>
#include <vector>
#include <set>
#include <climits>
#include "bits/stdc++.h"

using namespace std;

// Function to perform DFS and find all nodes in the connected component
void dfs(int node, const vector<vector<int>>& adj, set<int>& visited, vector<int>& component) {
    visited.insert(node);
    component.push_back(node);
    for (int neighbor : adj[node]) {
        if (visited.find(neighbor) == visited.end()) {
            dfs(neighbor, adj, visited, component);
        }
    }
}

int main() {
    int n, m;
    cin >> n >> m;

    vector<long long> c(n);
    for (long long &i : c) cin >> i;

    vector<vector<int>> adj(n);
    for (int i = 0; i < m; i++) {
        int a, b;
        cin >> a >> b;
        a--; // Adjusting for 0-based indexing
        b--; // Adjusting for 0-based indexing
        adj[a].push_back(b);
        adj[b].push_back(a);
    }

    // Finding connected components
    set<int> visited;
    vector<vector<int>> components;

    for (int node = 0; node < n; ++node) {
        if (visited.find(node) == visited.end()) {
            vector<int> component;
            dfs(node, adj, visited, component);
            components.push_back(component);
        }
    }

    // Output the connected components and calculate the sum of minimum costs
    long long sum = 0;
    for (const auto& component : components) {
        long long min_cost = LLONG_MAX;
        for (int node : component) {
            min_cost = min(min_cost, c[node]);
        }
        sum += min_cost;
    }

    cout << sum << endl;

    return 0;
}
