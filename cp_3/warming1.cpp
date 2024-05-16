#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>

using namespace std;

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);

    int n, x; cin >> n >> x;

    vector<int> nums(n);

    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }

    vector<int> prefixLIS(n);
    vector<int> suffixLIS(n);
    
    {
        vector<int> dp;
        for (int i = 0; i < n; ++i) {
            auto it = lower_bound(dp.begin(), dp.end(), nums[i]);
            if (it == dp.end()) {
                dp.push_back(nums[i]);
            } else {
                *it = nums[i];
            }
            prefixLIS[i] = dp.size();
        }
    }
    
    {
        vector<int> dp;
        for (int i = n - 1; i >= 0; --i) {
            auto it = lower_bound(dp.begin(), dp.end(), nums[i]);
            if (it == dp.end()) {
                dp.push_back(nums[i]);
            } else {
                *it = nums[i];
            }
            suffixLIS[i] = dp.size();
        }
    }
    
    int maxLIS = 0;
    
    for (int i = 0; i < n; ++i) {
        int prefix = (i > 0) ? prefixLIS[i - 1] : 0;
        int suffix = suffixLIS[i];
        maxLIS = max(maxLIS, prefix + suffix);
    }

    cout << maxLIS << endl;

    return 0;
}