#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);

    int t; cin >> t;

    while (t--) {
        string n; cin >> n;

        bool zero_present = false;
        bool even_present = false;
        int sum = 0;

        for (int i=0; i<n.size(); i++) {
            // convert char to int
            int x = n[i] - '0';

            // 60 = 2 * 2 * 3 * 5
            // zero_present = 5 * 2
            // even_present = 2,4,6,8 or one more 0
            // divisable_by_3 = sum % 3

            if ((x % 2 == 0 && x != 0) || (x == 0 && zero_present == true)) {
                even_present = true;
            }

            if (x == 0) {
                zero_present = true;
            }

            sum += x;
        }

        bool divisable_by_3 = false;
        if (sum % 3 == 0) {
            divisable_by_3 = true;
        }

        if (zero_present && divisable_by_3 && even_present) {
            cout << "red\n";
        } else {
            cout << "cyan\n";
        }
    }
    
    return 0;
}