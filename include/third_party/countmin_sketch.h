#pragma once
#include <vector>
#include <atomic>
#include <random>
#include <limits>
#include <cstdint>
#include <algorithm>

class CountMinSketch {
public:
    CountMinSketch(size_t depth = 4, size_t width = 1 << 16)
        : depth_(depth), width_(width), table_(depth * width), hash_seeds_(depth) 
    {
        // 初始化计数器
        for (auto& counter : table_) {
            counter.store(0, std::memory_order_relaxed);
        }

        // 初始化哈希种子
        std::random_device rd;
        for (size_t i = 0; i < depth_; ++i) {
            hash_seeds_[i] = rd();
        }
    }

    inline void record(uint64_t key) {
        for (size_t i = 0; i < depth_; ++i) {
            size_t h = hash(key, i);
            table_[i * width_ + h].fetch_add(1, std::memory_order_relaxed);
        }
    }

    inline uint32_t estimate(uint64_t key) const {
        uint32_t min_count = std::numeric_limits<uint32_t>::max();
        for (size_t i = 0; i < depth_; ++i) {
            size_t h = hash(key, i);
            uint32_t val = table_[i * width_ + h].load(std::memory_order_relaxed);
            if (val < min_count) {
                min_count = val;
            }
        }
        return min_count;
    }

    CountMinSketch(const CountMinSketch&) = delete;
    CountMinSketch& operator=(const CountMinSketch&) = delete;

private:
    inline size_t hash(uint64_t key, size_t i) const {
        if ((width_ & (width_ - 1)) == 0) { 
            // 如果是 2 的幂
            return (key ^ hash_seeds_[i]) & (width_ - 1);
        }
        return (key ^ hash_seeds_[i]) % width_;
    }

    size_t depth_;
    size_t width_;
    std::vector<std::atomic<uint64_t>> table_;  // 单一连续内存
    std::vector<uint64_t> hash_seeds_;
};
