#include "bits/stdc++.h"

using namespace std;

vector<long long> threeSum(vector<long long>& nums, long long target) {
    sort(nums.begin(), nums.end());  // Sorted array for easier management of duplicates
    unordered_map<long long, long long> hashMap;

    // Build hash map for all elements
    for (long long i = 0; i < nums.size(); ++i) {
        hashMap[nums[i]] = i;
    }

    for (long long i = 0; i < nums.size() - 2; ++i) {
        if (i > 0 && nums[i] == nums[i-1]) continue;  // Skip duplicates for the first number

        for (long long j = i + 1; j < nums.size() - 1; ++j) {
            if (j > i + 1 && nums[j] == nums[j-1]) continue;  // Skip duplicates for the second number

            long long required = target - (nums[i] + nums[j]);
            if (hashMap.count(required) && hashMap[required] > j) {  // Check if the required number exists beyond j
                return {nums[i], nums[j], required};  // Return immediately when the first valid triplet is found
            }
        }
    }
    return {};  // Return empty if no valid triplet is found
}

int main() {
    ios_base::sync_with_stdio(false);
    long long n;
    cin >> n;
    vector<long long> v(n);
    for (long long i = 0; i < n; ++i) {
        cin >> v[i];
    }

    vector<long long> result = threeSum(v, 42);  // Change the target sum as needed

    if (result.empty()) {
        cout << "impossible" << endl;
    } else {
        for (long long num : result) {
            cout << num << " ";
        }
        cout << endl;
    }

    return 0;
}

// source:
// site: https://leetcode.com/problems/3sum/solutions/
// god, who blessed me with idea to look at constraints and change int to long long at 6:35am.