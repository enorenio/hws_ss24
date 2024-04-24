#include "bits/stdc++.h"

using namespace std;

// void solve() {
//     // input n and then n integers
//     int n;
//     cin >> n;
//     vector<int> v(n);
//     for (int i = 0; i < n; ++i) {
//         cin >> v[i];
//     }

//     int q = 2;
//     vector<int> p(2);
//     int p_i = 0;
//     int flat = -1;
//     for (int i = 0; i < n - 1; ++i) {
//         if (q < 0) {
//             cout << "impossible" << "\n";
//             break;
//         }
//         // if there is a flat, store the first index of flat region
//         if ((flat == -1) && (v[i + 1] == v[i])) {
//             flat = i;
//         }
//         if (v[i + 1] < v[i]) {
//             // if more than 2 out-of-order elements, break
//             q = q - 1;

//             // store the index of 2 out-of-order elements
//             if (flat != -1) {
//                 p[p_i] = flat + 1;
//             } else {
//                 p[p_i] = i + 1;
//             }

//             p_i = p_i + 1;
//         }
//         if (v[i + 1] != v[i]) {
//             flat = -1;
//         }
//     }

//     if (q >= 0) {
//         if (p[1] == 0) {
//             p[1] = n;
//         }
//         cout << p[0] << " " << p[1] << "\n";
//     }
// }

string solve(int n, vector<int> v) {
    bool failed = false;
    int q = 2;

    int p_1 = -1, p_2 = -1, p_3 = -1, p_4 = -1;

    for (int i = 0; i < n - 1; ++i) {
        if (q < 0) {
            failed = true;
            break;
        }

        if (v[i + 1] < v[i]) {
            q = q - 1;
            if (q == 1) {
                p_1 = i;
                p_2 = i + 1;
            }
            if (q == 0) {
                p_3 = i;
                p_4 = i + 1;
            }
        }
    }

    if ((p_1 != -1) && (p_2 != -1) && (p_3 != -1) && (p_4 != -1)) {
        
    }

    if ((p_1 != -1) && (p_2 != -1)) {
        // copy vector, cut p_1 from its position and put it where it suits
        vector<int> newVector = v;
        newVector.erase(newVector.begin() + p_1);
        auto it = lower_bound(newVector.begin(), newVector.end(), v[p_1]);
        newVector.insert(it, v[p_1]);
        bool isSorted = is_sorted(newVector.begin(), newVector.end());
        if (isSorted) {
            return to_string(p_1 + 1) + " " + to_string(it - newVector.begin() + 1);
        }

        // copy vector, cut p_2 from its position and put it where it suits
        vector<int> newVector2 = v;
        newVector2.erase(newVector2.begin() + p_2);
        auto it2 = lower_bound(newVector2.begin(), newVector2.end(), v[p_2]);
        newVector2.insert(it2, v[p_2]);
        bool isSorted2 = is_sorted(newVector2.begin(), newVector2.end());
        if (isSorted2) {
            return to_string(p_2 + 1) + " " + to_string(it2 - newVector2.begin() + 1);
        }
    }

    if (failed) {
        return "impossible";
    } else {
        return "???";
    }
}

int main(){
    ios_base::sync_with_stdio(false);

    vector<pair<int, vector<int>>> test_cases = {
        {2, {2, 1}},
        {5, {1, 2, 4, 3, 5}},
        {6, {1, 5, 3, 4, 2, 6}},
        {5, {2, 2, 2, 1, 2}},
        {6, {1, 3, 2, 6, 5, 4}},
        {5, {5, 4, 3, 2, 1}},
        {4, {4, 2, 3, 1}},
        {5, {3, 1, 2, 5, 4}}
    };

    vector<string> expected = {
        "1 2",
        "3 4",
        "2 5",
        "4 1",
        "impossible",
        "1 5",
        "1 4",
        "2 5"
    };

    bool all_tests_pass = true;
    for (int i = 0; i < test_cases.size(); ++i) {
        string result = solve(test_cases[i].first, test_cases[i].second);
        if (result != expected[i]) {
            all_tests_pass = false;
            cout << "Test " << i + 1 << " FAILED." << endl;
            cout << "Expected: \"" << expected[i] << "\", got: \"" << result << "\"." << endl;
        } else {
            cout << "Test " << i + 1 << " passed." << endl;
        }
    }

    if (all_tests_pass) {
        cout << "All tests passed!" << endl;
    } else {
        cout << "Some tests failed." << endl;
    }

    return 0;

    // cerr << endl << "Time elapsed: " << 1.0 * clock() / CLOCKS_PER_SEC << " s." << endl;
}