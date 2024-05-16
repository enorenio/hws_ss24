#include "bits/stdc++.h"
#include <iostream>

using namespace std;

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);

    int n, m;
    cin >> n >> m;
    vector<int> bottles(n);
    for (int i = 0; i < n; ++i) {
        cin >> bottles[i];
    }

    vector<int> dp(n + 1, 0);
    vector<int> days(n + 1, -1);

    for (int i = 0; i < n; ++i) {
        // If we decide not to ask on day i
        if (i > 0) {
            dp[i + 1] = dp[i];
            days[i + 1] = days[i];
        }

        // If we decide to ask on day i
        int previous_day = (i >= m ? dp[i - m + 1] : 0) + bottles[i];
        if (previous_day > dp[i + 1]) {
            dp[i + 1] = previous_day;
            days[i + 1] = i;
        }
    }

    // Retrieve the best days to ask
    vector<int> result_days;
    for (int i = n; i > 0; ) {
        if (days[i] != days[i-1]) {
            result_days.push_back(days[i] + 1); // store 1-based index
            i = days[i] - m + 1;
        } else {
            --i;
        }
    }

    reverse(result_days.begin(), result_days.end());

    // Output results
    cout << dp[n] << " " << result_days.size() << endl;
    for (int day : result_days) {
        cout << day << " ";
    }
    cout << endl;

    return 0;
}