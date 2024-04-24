#include "bits/stdc++.h"

using namespace std;

std::string checkSortedWithLessThanOneSwap(int n, const std::vector<int>& nums) {
    std::vector<int> tmp(nums);
    std::sort(tmp.begin(), tmp.end());

    std::vector<int> changedIndices;
    for (size_t i = 0; i < nums.size(); ++i) {
        if (nums[i] != tmp[i]) {
            changedIndices.push_back(i);
        }
    }

    if (changedIndices.size() > 2) {
        return "impossible";
    } else if (changedIndices.size() == 2) {
        std::ostringstream oss;
        oss << changedIndices[0] + 1 << " " << changedIndices[1] + 1;
        return oss.str();
    } else {
        return "impossible";
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
        "1 4",
        "impossible",
        "impossible",
        "1 4",
        "impossible"
    };

    bool all_tests_pass = true;
    for (int i = 0; i < test_cases.size(); ++i) {
        string result = checkSortedWithLessThanOneSwap(test_cases[i].first, test_cases[i].second);
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
}