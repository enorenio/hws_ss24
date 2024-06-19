#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>

using namespace std;

struct Node {
    string name;
    vector<Node*> children;
    Node* parent;

    Node(const string& name) : name(name), parent(nullptr) {}
};

class FamilyTree {
private:
    Node* root;
    unordered_map<string, Node*> nodes;
    unordered_map<string, int> depth;
    unordered_map<string, vector<Node*>> up;  // Binary lifting table
    int LOG;  // Maximum power of 2 needed

    void printTree(Node* node, int level) {
        if (node == nullptr) {
            return;
        }
        for (int i = 0; i < level; ++i) {
            cout << "  ";
        }
        cout << node->name << endl;
        for (auto* child : node->children) {
            printTree(child, level + 1);
        }
    }

    void dfs(Node* node, Node* parent) {
        depth[node->name] = depth[parent->name] + 1;
        up[node->name][0] = parent;
        for (int i = 1; i <= LOG; ++i) {
            if (up[node->name][i - 1] != nullptr) {
                up[node->name][i] = up[up[node->name][i - 1]->name][i - 1];
            } else {
                up[node->name][i] = nullptr;
            }
        }
        for (auto* child : node->children) {
            dfs(child, node);
        }
    }

public:
    FamilyTree() : root(nullptr) {}

    void addPerson(const std::string& name) {
        if (nodes.find(name) == nodes.end()) {
            nodes[name] = new Node(name);
        }
    }

    void setMotherChild(const std::string& mother, const std::string& child) {
        if (nodes.find(mother) == nodes.end() || nodes.find(child) == nodes.end()) {
            return;
        }
        nodes[mother]->children.push_back(nodes[child]);
        nodes[child]->parent = nodes[mother];
    }

    void setRoot(const std::string& name) {
        if (nodes.find(name) != nodes.end()) {
            root = nodes[name];
            LOG = log2(nodes.size()) + 1;
            for (auto& node : nodes) {
                up[node.first] = vector<Node*>(LOG + 1, nullptr);
            }
            depth[root->name] = 0;
            dfs(root, root);
        }
    }

    void printTree() {
        printTree(root, 0);
    }

    Node* lca(Node* a, Node* b) {
        if (depth[a->name] < depth[b->name]) {
            swap(a, b);
        }
        int k = depth[a->name] - depth[b->name];
        for (int i = LOG; i >= 0; --i) {
            if ((k >> i) & 1) {
                a = up[a->name][i];
            }
        }
        if (a == b) {
            return a;
        }
        for (int i = LOG; i >= 0; --i) {
            if (up[a->name][i] != up[b->name][i]) {
                a = up[a->name][i];
                b = up[b->name][i];
            }
        }
        return up[a->name][0];
    }

    int distance(Node* a, Node* b) {
        Node* lcaNode = lca(a, b);
        return depth[a->name] + depth[b->name] - 2 * depth[lcaNode->name];
    }

    Node* getNode(const std::string& name) {
        if (nodes.find(name) != nodes.end()) {
            return nodes[name];
        }
        return nullptr;
    }
};

int main() {
    int n, q;
    cin >> n >> q;

    vector<string> l(n);
    for (string &i : l) cin >> i;

    vector<string> s(n-1);
    for (string &i : s) cin >> i;

    FamilyTree familyTree;

    for(const string& name : l) {
        familyTree.addPerson(name);
    }

    for (int i = 0; i < n-1; i++) {
        familyTree.setMotherChild(s[i], l[i+1]);
    }

    familyTree.setRoot(l[0]);

    for (int i = 0; i < q; i++) {
        string a, b;
        cin >> a >> b;
        Node* nodeA = familyTree.getNode(a);
        Node* nodeB = familyTree.getNode(b);
        Node* lcaNode = familyTree.lca(nodeA, nodeB);
        int dist = familyTree.distance(nodeA, nodeB);
        cout << lcaNode->name << " " << dist << endl;
    }

    return 0;
}
