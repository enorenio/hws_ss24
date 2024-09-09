#include <iostream>
#include <vector>
#include <algorithm>

#define debug(x) \
    (std::cerr << #x << ": " << (x) << '\n')

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    std::vector<int> v= {13, 9, 4, 8, 2, 3, 5, 1};

    int n = 0;
    do {
        n++;
        for (auto i : v) {
            std::cout << i << " ";
        }
        std::cout << '\n';
    } while (std::next_permutation(v.begin(), v.end()));

    std::cout << n;
    return 0;
}