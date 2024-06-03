#include <iostream>
#include "bits/stdc++.h"

using namespace std;

class TrieNode {
public:
    unordered_map<char, TrieNode*> children;
    int count; // To store the number of words passing through this node

    TrieNode() : count(0) {}
};

class Trie {
private:
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    void insert(const string& word) {
        TrieNode* node = root;
        for (char ch : word) {
            if (node->children.find(ch) == node->children.end()) {
                node->children[ch] = new TrieNode();
            }
            node = node->children[ch];
            node->count += 1;
        }
    }

    int countPrefix(const string& prefix) {
        TrieNode* node = root;
        for (char ch : prefix) {
            if (node->children.find(ch) == node->children.end()) {
                return 0;
            }
            node = node->children[ch];
        }
        return node->count;
    }
};

int main() {
    int n; cin >> n;
    Trie trie;

    for (int i = 0; i < n; i++) {
        string op, name;
        cin >> op >> name;
        
        if (op[0] == 'a') {
            trie.insert(name);
        } else {
            cout << trie.countPrefix(name) << endl;
        }
    }

    return 0;
}