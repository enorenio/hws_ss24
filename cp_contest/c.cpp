#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n; cin >> n;

    vector<long long> cards(n);
    for (int i = 0; i < n; ++i) {
        cin >> cards[i];
    }
    
    unordered_map<long long, int> count_map;
    for (const auto& card : cards) {
        count_map[card]++;
    }


    vector<long long> simon_cards;
    for (const auto& card : cards) {
        if (count_map[card] > 0) {
            long long possible_emma_card = card * 4 / 3;
            if ((card * 4) % 3 == 0 && count_map[possible_emma_card] > 0) {
                simon_cards.push_back(card);
                count_map[card]--;
                count_map[possible_emma_card]--;
            }
        }
    }
    
    sort(simon_cards.begin(), simon_cards.end());
        
    for (const auto& simon_card : simon_cards) {
        cout << simon_card << "\n";
    }

    return 0;
}