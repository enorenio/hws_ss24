#include <iostream>
#include <unordered_map>

using namespace std;

int main() {
    int t;
    cin >> t;
    
    while (t--) {
        int n;
        cin >> n;
        
        unordered_map<int, int> freq;
        int result = -1;
        
        int j = 0;
        for (; j < n; j++) {
            int a;
            cin >> a;
            if (++freq[a] == 3) {
                result = a;
                break;
            }
        }

        cout << result << "\n";

        int a;
        while (++j < n) cin >> a;
    }

    return 0;
}
