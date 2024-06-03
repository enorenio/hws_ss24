#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int lengthOfLIS(const vector<int>& nums) {
    vector<int> dp;
    for (int num : nums) {
        auto it = lower_bound(dp.begin(), dp.end(), num);
        if (it == dp.end()) {
            dp.push_back(num);
        } else {
            *it = num;
        }
    }
    return dp.size();
}

int largestLISAfterChange(vector<int>& nums, int x) {
    int n = nums.size();
    vector<int> prefixLIS(n);
    vector<int> suffixLIS(n);

    // Compute prefix LIS
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

    // Compute suffix LIS
    dp.clear();
    for (int i = n - 1; i >= 0; --i) {
        auto it = lower_bound(dp.begin(), dp.end(), nums[i]);
        if (it == dp.end()) {
            dp.push_back(nums[i]);
        } else {
            *it = nums[i];
        }
        suffixLIS[i] = dp.size();
    }

    int maxLIS = 0;

    // Calculate max LIS if adding x to all elements from i to n-1
    for (int i = 0; i < n; ++i) {
        int prefix = (i > 0) ? prefixLIS[i - 1] : 0;
        
        vector<int> temp;
        for (int j = i; j < n; ++j) {
            temp.push_back(nums[j] + x);
        }
        int suffix = lengthOfLIS(temp);
        
        maxLIS = max(maxLIS, prefix + suffix);
    }

    return maxLIS;
}

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);

    int n, x; 
    cin >> n >> x;

    vector<int> nums(n);

    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }

    cout << largestLISAfterChange(nums, x) << endl;

    return 0;
}
