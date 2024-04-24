#include "bits/stdc++.h"

using namespace std;

int main(){
    ios_base::sync_with_stdio(false);

    int joy[11][11];

    for (int i = 0; i < 11; ++i) {
        for (int j = 0; j < 11; ++j) {
            cin >> joy[i][j];
        }
    }

    vector<int> permutation(11);
    for (int i = 0; i < 11; ++i) {
        permutation[i] = i;
    }

    int maxJoy = 0;
    vector<int> bestAssignment;

    do {
        int currentJoy = 0;
        for (int i = 0; i < 11; ++i) {
            currentJoy += joy[i][permutation[i]];
        }
        if (currentJoy > maxJoy) {
            maxJoy = currentJoy;
            bestAssignment = permutation;
        }
    } while (next_permutation(permutation.begin(), permutation.end()));

    cout << maxJoy << "\n";
    // for (int i = 0; i < 11; ++i) {
    //     cout << "Friend " << (i + 1) << " gets wine glass " << (bestAssignment[i] + 1) << "\n";
    // }

    // cerr << endl << "Time elapsed: " << 1.0 * clock() / CLOCKS_PER_SEC << " s." << endl;
}