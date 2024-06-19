#include <iostream>
#include <vector>
#include <queue>
#include <functional>

using namespace std;

struct Edge {
    int to;
    int quality;
};

vector<vector<Edge>> adjList;

int maximumSpanningTree(int n) {
    vector<bool> inMST(n, false);
    priority_queue<pair<int, int>> pq; // (quality, vertex)
    
    // Start from vertex 0 and add a self-loop
    pq.push({0, 0});
    int totalQuality = 0;

    while (!pq.empty()) {
        auto [quality, u] = pq.top();
        pq.pop();

        if (inMST[u]) continue;
        inMST[u] = true;

        totalQuality += quality;

        for (const auto& edge : adjList[u]) {
            if (!inMST[edge.to]) {
                pq.push({edge.quality, edge.to});
            }
        }
    }

    return totalQuality;
}

int main() {
    int n, m;
    cin >> n >> m;
    adjList.resize(n);

    for (int i = 0; i < m; ++i) {
        int s, d, v;
        cin >> s >> d >> v;
        --s; // Adjust for 0-based indexing
        --d; // Adjust for 0-based indexing
        adjList[s].push_back({d, v});
        adjList[d].push_back({s, v});
    }

    int result = maximumSpanningTree(n);

    // Output a single integer: The maximum total quality
    cout << result << endl;

    return 0;
}
