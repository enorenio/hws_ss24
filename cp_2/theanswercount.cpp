#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

// what is bit_is_set function?
// rewrite the function to use bit_is_set function?

vector<long long> computeSubsetSums(const vector<int>& nums) {
    vector<long long> subsetSums = {0};
    for (int num : nums) {
        vector<long long> newSums;
        for (long long sum : subsetSums) {
            newSums.push_back(sum + num);
        }
        subsetSums.insert(subsetSums.end(), newSums.begin(), newSums.end());
    }
    return subsetSums;
}

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);

    int n; cin >> n;
    long long x; cin >> x;

    vector<int> nums(n);

    for(int i = 0; i < n; i++) {
        cin >> nums[i];
    }

    int mid = nums.size() / 2;
    
    // Split the array into two halves
    vector<int> left(nums.begin(), nums.begin() + mid);
    vector<int> right(nums.begin() + mid, nums.end());
    
    // Get all subset sums for both halves
    vector<long long> leftSums = computeSubsetSums(left);
    vector<long long> rightSums = computeSubsetSums(right);
    
    // Use a hash map to count sums in the left half
    unordered_map<long long, int> leftSumCount;
    for (long long sum : leftSums) {
        ++leftSumCount[sum];
    }
    
    // Count the number of valid pairs
    long long count = 0;
    for (long long sum : rightSums) {
        long long complement = x - sum;
        if (leftSumCount.find(complement) != leftSumCount.end()) {
            count += leftSumCount[complement];
        }
    }

    cout << count << endl;

    return 0;
}

// SUBSET-SUM PROBLEM (SSP), meet-in-the-middle solution. O(2^(n/2) * n) time complexity.