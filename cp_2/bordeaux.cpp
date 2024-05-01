#include <iostream>
#include <vector>

using namespace std;

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);

    int n; cin >> n;

    vector<long long> d(n), v(n), diff(n);
    long long positives = 0;
    long long min_di = 1e9;

    while (n--) {
        long long di, vi; cin >> di >> vi;
        diff[n] = vi - di;
        if (diff[n] > 0) {
            positives += diff[n];
        }
        if (di < min_di) {
            min_di = di;
        }
        cout << diff[n] << endl;
    }
    cout << "---" << endl;
    if (positives == 0) {
        cout << 0 << " " << 0 << endl;
    } else {
        cout << min_di << " " << positives << endl;
    }
    return 0;
}