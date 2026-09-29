#include "layer_range.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using intelhive::LayerRange;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void same(const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size(), "shape mismatch");
    for (size_t i = 0; i < a.size(); ++i)
        require(std::isfinite(a[i]) && std::abs(a[i] - b[i]) < 1e-5f, "split graph differs from full graph");
}
template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid input accepted");
}
int main() {
    try {
        rejects([] { LayerRange invalid(-1, 2); });
        rejects([] { LayerRange invalid(4, 3); });
        rejects([] { LayerRange invalid(0, 6); });
        // Test every two-way split, plus three independent workers below.
        std::vector<float> prompt(3 * LayerRange::width);
        for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = std::cos(float(i));
        for (int boundary = 0; boundary < 5; ++boundary) {
            LayerRange full(0, 5), first(0, boundary), last(boundary + 1, 5);
            same(full.execute(prompt, 0), last.execute(first.execute(prompt, 0), 0));
            for (int position = 3; position < 7; ++position) {
                std::vector<float> token(LayerRange::width, position * 0.1f);
                same(full.execute(token, position), last.execute(first.execute(token, position), position));
            }
        }
        LayerRange full(0, 5), a(0, 1), b(2, 3), c(4, 5);
        auto expected = full.execute(prompt, 0);
        same(expected, c.execute(b.execute(a.execute(prompt, 0), 0), 0));
        require(a.owned_layers() == 2 && b.cached_tokens() == 3, "shard ownership/KV mismatch");
        rejects([&] { a.execute(prompt, 0); });
        rejects([&] { a.execute({1}, 3); });
        rejects([&] { a.execute(std::vector<float>(33 * LayerRange::width), 3); });
        require(a.cached_tokens() == 3, "rejection mutated KV state");
        full.reset(); a.reset(); b.reset(); c.reset();
        same(expected, c.execute(b.execute(a.execute(prompt, 0), 0), 0));
        // A second sequence has independent state; releasing/restarting is deterministic.
        LayerRange independent(0, 5);
        same(expected, independent.execute(prompt, 0));
        std::cout << "PASS: all split boundaries, three shards, prefill, four decode steps, KV isolation/reset and invalid inputs\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
