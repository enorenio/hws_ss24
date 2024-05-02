#include <iostream>
#include <vector>
#include <climits>

using namespace std;

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);

    int n; cin >> n;

    vector<long long> d(n), v(n), diff(n);
    long long positives_sum = 0;
    long long max_vi = -LLONG_MAX;
    long long min_di = LLONG_MAX;

    while (n--) {
        long long di, vi; cin >> di >> vi;
        diff[n] = vi - di;
        if (diff[n] > 0) {
            positives_sum += diff[n];
            if (vi > max_vi) {
                max_vi = vi;
            }
        }
        if (di < min_di) {
            min_di = di;
        }
        // cout << diff[n] << endl;
    }
    // cout << "---" << endl;
    if (positives_sum == 0) {
        cout << 0 << " " << 0 << endl;
    } else {
        cout << abs(positives_sum - max_vi) << " " << positives_sum << endl;
    }
    return 0;
}