#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>

using namespace std;

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);

    int n; cin >> n;

    vector<pair<long long, long long> > dv(n);
    long long had_to_borrow_total = 0;
    long long current_have = 0;

    while (n--) {
      long long di, vi; cin >> di >> vi;
      dv[n] = make_pair(di, vi);
    }

    sort(dv.begin(), dv.end());

    for (int i = 0; i < dv.size(); i++) {
        if (dv[i].second > dv[i].first) {
            if (current_have < dv[i].first) {
                long long had_to_borrow_now = dv[i].first - current_have;

                had_to_borrow_total += had_to_borrow_now;
                current_have += had_to_borrow_now;
            }
            current_have -= dv[i].first;
            current_have += dv[i].second;
        }
    }

    cout << had_to_borrow_total << " " << current_have - had_to_borrow_total << endl;
    return 0;
}