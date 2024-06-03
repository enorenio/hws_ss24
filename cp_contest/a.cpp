#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
    int a, b, c, d;
    cin >> a >> b >> c >> d;

    int claims[4] = {a, b, c, d};
    int tq = 4;
    int sum = a + b + c + d;

    if (sum <= tq) {
        cout << 0 << endl;
        return 0;
    }
    else {
        int excess = sum - tq;
        int count = 0;
        sort(claims, claims + 4, greater<int>());

        for (int i = 0; i < 4; i++) {
            if (excess <= 0) break;
            int possible = min(excess, claims[i] - (tq /4));
            if (possible > 0) {
                excess -= possible;
                count++;
            }
        }
        
        cout << count << endl;
    }


    return 0;
}
