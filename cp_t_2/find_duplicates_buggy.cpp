#include <iostream>
#include "bits/stdc++.h"
using namespace std;

bool found_duplicate = false;
bool character_found[26];

int main()
{
    string input;
    cin >> input;

    for (char c: input) {
        // transform c to a number between 0 (a) and 25 (z)
        if (c >= 'a' && c <= 'z') c -= 96;
        else if (c >= 'A' && c <= 'Z') c -= 64;
        else assert(false); // invalid input

        if (character_found[c]) found_duplicate = true;
        else character_found[c] = true;
    }

    if (found_duplicate) cout << "duplicate\n";
    else cout << "no duplicate\n";
}

// g++ -std=c++17 -O2 -g ./cp_t_2/find_duplicates_buggy.cpp -o ./cp_t_2/find_duplicates_buggy
// g++ -std=c++17 -g ./cp_t_2/find_duplicates_buggy.cpp -o ./cp_t_2/find_duplicates_buggy
// abcdefghijklmnopqrstuvwxyz
// notice bug?

// then use lecture_1 slide 52 to debug
// 
// this creates a runtime error index 26 out of bounds.
// g++ -std=gnu++17 -Wall -Wextra -fsanitize=undefined,address -D GLIBCXX_DEBUG -g ./cp_t_2/find_duplicates_buggy.cpp -o ./cp_t_2/find_duplicates_buggy

// turns out the bug is in line 15 where we substract 96 from c instead of 97
// better way instead of using magic numbers is to substract 'a' or 'A' from c
