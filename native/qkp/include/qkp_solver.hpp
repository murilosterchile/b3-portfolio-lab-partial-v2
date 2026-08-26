#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace qkp {

struct Instance {
    std::vector<std::string> names;
    std::vector<double> linear;
    std::vector<std::vector<double>> pair;
    std::vector<double> costs;
    double capacity{};
    std::size_t min_cardinality{1};
    std::size_t max_cardinality{};
};

struct Result {
    std::vector<std::size_t> selected;
    double objective{};
    double total_cost{};
    bool optimal{false};
};

class ExactBranchAndBound {
public:
    Result solve(const Instance& instance) const;
};

double evaluate(const Instance& instance, const std::vector<std::size_t>& selected);

}  // namespace qkp
