#include "qkp_solver.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace qkp {

namespace {

void validate(const Instance& in) {
    const auto n = in.names.size();
    if (n == 0 || in.linear.size() != n || in.costs.size() != n || in.pair.size() != n) {
        throw std::invalid_argument("invalid QKP dimensions");
    }
    for (const auto& row : in.pair) {
        if (row.size() != n) throw std::invalid_argument("pair matrix must be square");
    }
    if (in.capacity < 0.0) throw std::invalid_argument("capacity must be non-negative");
    for (double c : in.costs) if (c < 0.0) throw std::invalid_argument("costs must be non-negative");
}

}  // namespace

double evaluate(const Instance& instance, const std::vector<std::size_t>& selected) {
    double value = 0.0;
    for (std::size_t a = 0; a < selected.size(); ++a) {
        const auto i = selected[a];
        value += instance.linear[i];
        for (std::size_t b = a + 1; b < selected.size(); ++b) {
            value += instance.pair[i][selected[b]];
        }
    }
    return value;
}

Result ExactBranchAndBound::solve(const Instance& in) const {
    validate(in);
    const std::size_t n = in.names.size();
    const std::size_t max_card = in.max_cardinality == 0 ? n : in.max_cardinality;
    if (in.min_cardinality > max_card || max_card > n) {
        throw std::invalid_argument("invalid cardinality bounds");
    }

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::vector<double> density(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double positive_pair = 0.0;
        for (std::size_t j = 0; j < n; ++j) positive_pair += std::max(0.0, in.pair[i][j]);
        density[i] = (std::max(0.0, in.linear[i]) + positive_pair) / std::max(in.costs[i], 1e-9);
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return density[a] > density[b];
    });

    double best = -std::numeric_limits<double>::infinity();
    std::vector<std::size_t> best_selected;
    std::vector<std::size_t> selected;

    auto upper_bound = [&](std::size_t index, double current) {
        double ub = current;
        for (std::size_t p = index; p < n; ++p) {
            const auto i = order[p];
            ub += std::max(0.0, in.linear[i]);
            for (const auto chosen : selected) ub += std::max(0.0, in.pair[i][chosen]);
        }
        for (std::size_t p = index; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                ub += std::max(0.0, in.pair[order[p]][order[q]]);
            }
        }
        return ub;
    };

    std::function<void(std::size_t, double, double)> dfs;
    dfs = [&](std::size_t index, double current, double cost) {
        const auto remaining = n - index;
        if (selected.size() > max_card || selected.size() + remaining < in.min_cardinality) return;
        if (cost > in.capacity + 1e-9) return;
        if (upper_bound(index, current) <= best + 1e-12) return;
        if (index == n) {
            if (selected.size() >= in.min_cardinality && selected.size() <= max_card && current > best) {
                best = current;
                best_selected = selected;
            }
            return;
        }

        const auto i = order[index];
        if (selected.size() < max_card && cost + in.costs[i] <= in.capacity + 1e-9) {
            double incremental = in.linear[i];
            for (const auto chosen : selected) incremental += in.pair[i][chosen];
            selected.push_back(i);
            dfs(index + 1, current + incremental, cost + in.costs[i]);
            selected.pop_back();
        }
        dfs(index + 1, current, cost);
    };

    dfs(0, 0.0, 0.0);
    if (!std::isfinite(best)) return Result{{}, 0.0, 0.0, true};
    std::sort(best_selected.begin(), best_selected.end());
    double total_cost = 0.0;
    for (auto i : best_selected) total_cost += in.costs[i];
    return Result{best_selected, best, total_cost, true};
}

}  // namespace qkp
