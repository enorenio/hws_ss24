#include <iostream>
#include <vector>

using namespace std;

// square root decomposition technique
// divide the array into blocks of size sqrt(n)
// and store the sum of each block
// query: iterate over the blocks that are completely inside the range
// and add the sum of those blocks
// update: update the block that contains the index and update the sum of that block
// O(sqrt(n)) query and update

// segment tree
// divide the array into a binary tree
// each node stores the sum of the elements in the range represented by that node
// query: iterate over the nodes that are completely inside the range
// and add the sum of those nodes
// update: update the node that contains the index and update the sum of that node
// O(log(n)) query and update

class SegmentTree {
private:
    vector<int> tree;
    vector<int> data;
    int n;

    void build(int node, int start, int end) {
        if (start == end) {
            // Leaf node will have a single element
            tree[node] = data[start] % 2;
        } else {
            int mid = (start + end) / 2;
            build(2 * node + 1, start, mid);
            build(2 * node + 2, mid + 1, end);
            tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
        }
    }

    void update(int idx, int value, int node, int start, int end) {
        if (start == end) {
            // Leaf node
            data[idx] = value;
            tree[node] = value % 2;
        } else {
            int mid = (start + end) / 2;
            if (start <= idx && idx <= mid) {
                update(idx, value, 2 * node + 1, start, mid);
            } else {
                update(idx, value, 2 * node + 2, mid + 1, end);
            }
            tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
        }
    }

    int query(int l, int r, int node, int start, int end) {
        if (r < start || end < l) {
            // range represented by a node is completely outside the given range
            return 0;
        }
        if (l <= start && end <= r) {
            // range represented by a node is completely inside the given range
            return tree[node];
        }
        int mid = (start + end) / 2;
        int left_child = query(l, r, 2 * node + 1, start, mid);
        int right_child = query(l, r, 2 * node + 2, mid + 1, end);
        return left_child + right_child;
    }

public:
    SegmentTree(vector<int>& input) {
        data = input;
        n = input.size();
        tree.resize(4 * n);
        build(0, 0, n - 1);
    }

    void update(int idx, int value) {
        update(idx, value, 0, 0, n - 1);
    }

    int query(int l, int r) {
        return query(l, r, 0, 0, n - 1);
    }
};

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);

    int n; cin >> n;
    int q; cin >> q;

    vector<int> b(n);

    for(int i = 0; i < n; i++) {
        cin >> b[i];
    }

    SegmentTree st(b);

    for(int i = 0; i < q; i++) {
        char c; cin >> c;
        if (c == 'U') {
            int x, y; cin >> x >> y;
            st.update(x - 1, y);
        } else {
            int x, y; cin >> x >> y;
            cout << st.query(x - 1, y - 1) << endl;
        }
    }

    return 0;
}