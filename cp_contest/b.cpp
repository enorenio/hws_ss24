#include <iostream>
#include <vector>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);

    int n; cin >> n;

    vector<int> p(n);
    for (auto &i : p) cin >> i;

    vector<int> a(n + 1);
    for (auto &i : a) cin >> i;

    long long max = 0;
    int daysSkipped = 0;
    for (int j = 0; j < n + 1; j++) {  
        long long sum = 0;
        int currentEnergy = a[j];
        for (int i = j; i < n; i++) {
            if (currentEnergy <= 0) break;
            currentEnergy -= 1;
            sum += p[i];
            if (max < sum) {
                max = sum;
                daysSkipped = j + 1;
            }
        }
    }

    cout << max << " " << daysSkipped << endl;

    return 0;
}