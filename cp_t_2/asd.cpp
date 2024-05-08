#include <iostream>
#include "bits/stdc++.h"
using namespace std;

int main() {
  vector<int> a, b;
  int n, q;
  cin >> n >> q;
  a.resize(n);

  // what does it do?
  // reference ../cp_2/microbes.cpp input loop
  for(int& i:a) {
    cin >> i;
    i %= 2;
  }

}