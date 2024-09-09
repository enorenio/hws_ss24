#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <queue>
#include <tuple>
#include <algorithm>
#include <numeric>

void show(std::vector<int> v) {
    for (auto i : v) {
        std::cout << i << " ";
    }
    std::cout << '\n';
}

void show(std::map<int, int> m) {
    for (auto p : m) {
        std::cout << p.first << " " << p.second << '\n';
    }
}

void show(std::unordered_map<int, int> m) {
    for (auto p : m) {
        std::cout << p.first << " " << p.second << '\n';
    }
}

void show(std::set<int> s) {
    for (auto i : s) {
        std::cout << i << " ";
    }
    std::cout << '\n';
}

std::pair<int, int> minMax(int a, int b, int c) {
    return {std::min({a, b, c}), std::max({a, b, c})};
}

int main() {
    std::vector<int> a = {1, 4, 2, 5, 8, 7};

    show(a);

    // Sort
    std::sort(a.begin(), a.end());

    show(a);

    // Reverse
    std::reverse(a.begin(), a.end());

    show(a);

    // Count frequency
    std::vector<int> b = {1, 1, 1, 2, 3, 2, 4, 5, 6, 4, 5, 7, 6, 7, 8, 10, 9};
    std::unordered_map<int, int> freq;

    for (auto i : b) {
        freq[i]++;
    }

    show(freq);
    std::cout << '\n';

    std::map<int, int> sorted_freq(freq.begin(), freq.end());
    
    show(sorted_freq);

    // Binary search
    std::vector<int> v = {1, 3, 5, 7, 9};

    if (binary_search(v.begin(), v.end(), 5)) {
        std::cout << "5 exists in the array" << '\n';
    }

    std::vector<int> v2 = {1, 2, 2, 3, 3, 3, 4, 5};
    auto lb = std::lower_bound(v2.begin(), v2.end(), 3);
    auto ub = std::upper_bound(v2.begin(), v2.end(), 3);

    std::cout << "First occurence of 3 at index: " << (lb - v2.begin()) << '\n';
    std::cout << "Last occurence of 3 is before index: " << (ub - v2.begin()) << '\n';

    // set is used to maintain unique sorted elements
    std::set<int> s = {5, 1, 6, 3, 9};

    s.insert(2);
    s.erase(5);

    show(s);

    // priority queue (max-heap by default)
    /*
    Priority queues (heaps) are useful in greedy algorithms, like finding the k-th largest element, or in Dijkstra's shortest path algorithm.
    */

    std::priority_queue<int> pq;
    
    pq.push(10);
    pq.push(1);
    pq.push(5);

    std::cout << "Max element in priority queue: " << pq.top() << '\n';
    pq.pop(); // removes the max element

    // Min-heap using priority queue
    std::priority_queue<int, std::vector<int>, std::greater<int>> minHeap;
    minHeap.push(10);
    minHeap.push(1);
    minHeap.push(5);

    std::cout << "Min element in priority queue: " << minHeap.top() << '\n';
    minHeap.pop(); // removes the min element

    // Pair and Tie are useful when you need to return multiple values from a function without creating complex structures
    auto [minVal, maxVal] = minMax(3, 7, 2);
    std::cout << minVal << '\n';
    std::cout << maxVal << '\n';

    // Summing a range of numbers
    // accumulate and iota
    // accumulate is handy when solving problems related to summation, total cost or prefix sums.

    std::vector<int> c(10);

    std::iota(c.begin(), c.end(), 1); // Fills vector with {1, 2, 3, ..., 10}

    // Sum all elements
    int sum = std::accumulate(c.begin(), c.end(), 0);
    std::cout << "Sum of elements: " << sum << '\n';

    // gcd - calculating the greatest common divisor

    int t = 36, y = 60;

    int g = std::gcd(t, y);
    std::cout << "GCD: " << g << '\n';

    return 0;
}