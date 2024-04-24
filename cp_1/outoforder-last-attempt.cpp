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

    int n;
    cin >> n;
    vector<int> v(n);
    for (int i = 0; i < n; ++i) {
        cin >> v[i];
    }

    cout << checkSortedWithLessThanOneSwap(n, v);

    return 0;
}