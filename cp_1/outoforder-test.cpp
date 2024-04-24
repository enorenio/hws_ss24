#include "bits/stdc++.h"

using namespace std;

std::string checkSortedWithLessThanOneSwap(const std::vector<int>& nums) {
    if (nums.size() <= 2) {
        return "impossible";  // Automatically return "no" for very small vectors
    }

    std::vector<int> tmp(nums);  // Copy the original vector
    std::sort(tmp.begin(), tmp.end());  // Sort the copied vector

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
        oss << changedIndices[0]+1 << " " << changedIndices[1]+1;
        return oss.str();
    } else {
        return "impossible";
    }
}

// A utility function to print an array of size n
void printArray(int arr[], int n)
{
    int i;
    for (i=0; i < n; i++)
        cout << arr[i] << " ";
    cout << endl;
}
 
/* Driver program to test insertion sort */
int main()
{
    vector<int> arr = {10, 20, 50, 30, 30};
 
    cout << checkSortedWithLessThanOneSwap(arr);
 
    return 0;
}