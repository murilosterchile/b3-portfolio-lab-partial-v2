#include "qkp_solver.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    qkp::Instance instance{
        .names = {"A", "B", "C"},
        .linear = {5.0, 4.0, 3.0},
        .pair = {{0.0, 4.0, -2.0}, {4.0, 0.0, 1.0}, {-2.0, 1.0, 0.0}},
        .costs = {3.0, 3.0, 2.0},
        .capacity = 6.0,
        .min_cardinality = 1,
        .max_cardinality = 2,
    };
    auto result = qkp::ExactBranchAndBound{}.solve(instance);
    assert(result.optimal);
    assert(result.selected.size() == 2);
    assert(result.selected[0] == 0 && result.selected[1] == 1);
    assert(std::abs(result.objective - 13.0) < 1e-9);
    std::cout << "qkp_tests: ok\n";
}
