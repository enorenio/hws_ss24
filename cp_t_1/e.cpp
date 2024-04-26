#include <iostream>
#include <vector>
using namespace std;

int interact(int i , int j){
    cout << "? "<< i+1 << ' ' << j+1 << endl;
    int x;
    cin >> x;
    return x;
}

int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    
    int n;
    cin >> n;

    int a = interact(0, 2);
    int b = interact(0, 1);
    int c = interact(1, 2);

    vector<int>ans(n);
    ans[0] = a - c;
    ans[1] = c + b - a;
    ans[2] = a - b;

    for(int i=3; i<n; i++){
        int x = interact(i-1,i);
        ans[i] = x - ans[i-1];
    }

    cout << "! ";
    for(auto i : ans) cout << i << " ";
    
    return 0;
}