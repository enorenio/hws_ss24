#include <iostream>
#include <string>
#include <vector>

using namespace std;

vector<int> computePrefixFunction(const string &pattern) {
    int n = pattern.length();
    vector<int> pi(n, 0);
    for (int i = 1; i < n; ++i) {
        int j = pi[i - 1];
        while (j > 0 && pattern[i] != pattern[j]) {
            j = pi[j - 1];
        }
        if (pattern[i] == pattern[j]) {
            j++;
        }
        pi[i] = j;
    }
    return pi;
}

int main() {
    string s, t;
    cin >> s >> t;
    string combined = t + '#' + s;
    vector<int> pi = computePrefixFunction(combined);
    int maxOverlap = pi.back();
    int minLettersNeeded = s.length() + t.length() - maxOverlap;
    cout << minLettersNeeded << endl;
    return 0;
}
