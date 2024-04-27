#include <iostream>
#include <vector>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);

    int t; cin >> t;

    while (t--) {
        string v[9];
        for (int i=0; i<9; i++) {
            cin >> v[i];

            for (char j=0; j<9; j++) {
                
                if (v[i][j] == '1') {
                    v[i][j] = '2';
                }
            }
        }
        for (int i=0; i<9; i++) {
            cout << v[i] << "\n";
        }
    }
    
    return 0;
}