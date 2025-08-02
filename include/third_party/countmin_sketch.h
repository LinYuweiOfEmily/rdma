#pragma once
#include <vector>
#include <atomic>
#include <random>
#include <limits>
#include <cstdint>

class CountMinSketch {
public:
    CountMinSketch(size_t depth = 4, size_t width = 1 << 16)
    : depth_(depth), width_(width), table_() {
        table_.reserve(depth_);  // 预分配空间
        for (size_t i = 0; i < depth_; ++i) {
            table_.emplace_back(width_);  // 构造一个 width_ 大小的 vector，默认构造 std::atomic<uint32_t>
            for (auto& counter : table_[i]) {
                counter.store(0, std::memory_order_relaxed);  // 手动初始化
            }
        }

        std::random_device rd;
        for (size_t i = 0; i < depth_; ++i)
            hash_seeds_.push_back(rd());
    }

    void record(uint64_t key) {
        for (size_t i = 0; i < depth_; ++i) {
            size_t h = hash(key, i);
            table_[i][h].fetch_add(1, std::memory_order_relaxed);
        }
    }

    uint32_t estimate(uint64_t key) const {
        uint32_t min_count = std::numeric_limits<uint32_t>::max();
        for (size_t i = 0; i < depth_; ++i) {
            size_t h = hash(key, i);
            min_count = std::min(min_count, table_[i][h].load(std::memory_order_relaxed));
        }
        return min_count;
    }

    // 禁用拷贝构造和赋值运算符，防止编译错误
    CountMinSketch(const CountMinSketch&) = delete;
    CountMinSketch& operator=(const CountMinSketch&) = delete;

private:
    size_t hash(uint64_t key, size_t i) const {
        return (key ^ hash_seeds_[i]) % width_;
    }

    const size_t depth_;
    const size_t width_;
    std::vector<std::vector<std::atomic<uint32_t>>> table_;
    std::vector<uint64_t> hash_seeds_;
};
