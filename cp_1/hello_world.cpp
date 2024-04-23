#include <iostream>
#include <string>

int main() {
    int t;
    std::cin >> t;
    std::cin.ignore();

    for (int i = 0; i < t; ++i) {
        std::string str;
        std::getline(std::cin, str);
        std::cout << "Hello " << str << "!" << std::endl;
    }
    return 0;
}