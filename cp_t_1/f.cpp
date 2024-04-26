#include <iostream>
#include <vector>

using namespace std;

int interact(int i , int j){
  cout << "? "<< i + 1 << ' ' << j + 1 << endl;
  int x;
  cin >> x;
  return x - 1;
}

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);

  int n;
  cin >> n;
  
  int answer = 0;
  int smax = interact(0, n - 1);

  // right case
  if (smax == 0 || interact(0, smax) != smax) {
    int l = smax;
    int r = n - 1;

    while (r - l > 1) {
      int m = (l + r) / 2;
      if (interact(smax, m) == smax) {
        r = m;
      } else {
        l = m;
      }
    }
    answer = r + 1;
  } else {
    // left case

    int l = 0;
    int r = smax;

    while (r - l > 1) {
      int m = (l + r) / 2;
      if (interact(m, smax) != smax) {
        r = m;
      } else {
        l = m;
      }
    }
    answer = l + 1;
  }

  cout << "! " << answer << endl;

  return 0;
}
