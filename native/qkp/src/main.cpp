#include "qkp_solver.hpp"

#include <iostream>

int main() {
    qkp::Instance instance{
        .names = {"A", "B", "C", "D"},
        .linear = {8.0, 7.0, 6.0, 5.0},
        .pair = {{0, -1, 2, 0}, {-1, 0, 1, 0}, {2, 1, 0, -2}, {0, 0, -2, 0}},
        .costs = {4, 4, 3, 2},
        .capacity = 8,
        .min_cardinality = 1,
        .max_cardinality = 3,
    };
    const auto result = qkp::ExactBranchAndBound{}.solve(instance);
    std::cout << "objective=" << result.objective << " cost=" << result.total_cost << " selected=";
    for (auto i : result.selected) std::cout << instance.names[i] << ' ';
    std::cout << '\n';
    return 0;
}
