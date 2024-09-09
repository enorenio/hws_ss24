typedef long long ll;
#define debug(x) \
    (std::cerr << #x << ": " << (x) << '\n')

#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    std::vector<std::vector<int>> joy(11, std::vector<int>(11, 0));

    for (int i = 0; i < 11; ++i) {
        for (int j = 0; j < 11; ++j) {
            std::cin >> joy[i][j];
        }
    }

    std::vector<int> c(11);
    std::iota(c.begin(), c.end(), 0);

    int maxJoy = 0;
    std::vector<int> bestPermutation;

    do {
        int cur = 0;
        for (int i = 0; i < 11; ++i) {
            cur += joy[i][c[i]];
        }
        maxJoy = std::max(maxJoy, cur);
    } while (std::next_permutation(c.begin(), c.end()));

    std::cout << maxJoy << '\n';

    return 0;
}