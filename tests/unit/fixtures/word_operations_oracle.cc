#include "kickmix/util/word_ops.h"
#include <iostream>
int main() {
    uint64_t a,b,c,d;
    while (std::cin >> a >> b >> c >> d) {
        unsigned long long sum, difference;
        uint64_t low, high;
        bool carry = kickmix::add_carry_u64(c != 0, a, b, &sum);
        bool borrow = kickmix::sub_borrow_u64(c != 0, a, b, &difference);
        kickmix::full_fma64(a,b,c,d,&low,&high);
        std::cout << sum << ' ' << carry << ' ' << difference << ' ' << borrow
                  << ' ' << kickmix::extract_bits_u64(a,b) << ' ' << low << ' ' << high << '\n';
    }
}
