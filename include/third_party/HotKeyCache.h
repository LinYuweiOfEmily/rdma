#pragma once
#include <libcuckoo/cuckoohash_map.hh>
#include <vector>
#include <atomic>
#include <algorithm>
#include <memory>
#include <random>  // [MODIFIED]
#include <mutex>   // [MODIFIED]
#include "inlineskiplist.h"
#include "countmin_sketch.h"
#include "../CacheEntry.h"
using Key = uint64_t;

template <class Comparator>
class HotKeyCacheShard {
public:
    explicit HotKeyCacheShard(size_t max_sz)
        : max_size(max_sz),
          eviction_trigger_threshold(max_sz * EVICTION_TRIGGER_RATIO / 100),
          eviction_batch_size(max_sz * EVICTION_BATCH_RATIO / 100) {}

    typename InlineSkipList<Comparator>::Node* Get(const Key& key) {
        thread_local uint32_t rng = 1;
        if ((++rng & 0xF) == 0) cms.record(key);
        typename InlineSkipList<Comparator>::Node* result = nullptr;
        map.find_fn(key, [&](const HotKeyCacheEntry& entry) {
            result = entry.node;
        });
        return result;

    }

    void Add(const Key& key, typename InlineSkipList<Comparator>::Node* node) {
        if (cms.estimate(key) < CMS_PROMOTION_THRESHOLD) {
            return;  // 频率不够，不加入 cache
        }
        bool updated = map.update_fn(key, [&](HotKeyCacheEntry& entry) {
            if (entry.node == nullptr) return;
            auto val = (const CacheEntry *)entry.node->Key();
            if (val->ptr == nullptr) {
                entry.node = node;  
            }
        });
        if (updated) {
            return;
        }
        auto entry = HotKeyCacheEntry{node};
        bool success = map.insert(key, std::move(entry));

        if (!success) return;
        auto& tls_keys = GetThreadLocalKeys();
        if (tls_keys.size() < 20) {
            tls_keys.push_back(key);
        }
        auto sz = current_size.fetch_add(1, std::memory_order_relaxed) + 1;
        current_valid_size.fetch_add(1, std::memory_order_relaxed);
        if (!evicting_flag.test_and_set(std::memory_order_acq_rel) && sz > eviction_trigger_threshold) {
            Evict();
            evicting_flag.clear(std::memory_order_release);
        }
    }

private:
    struct HotKeyCacheEntry {
        typename InlineSkipList<Comparator>::Node* node;

    };

    // 获取线程本地key列表
    std::vector<Key>& GetThreadLocalKeys() {
        thread_local std::vector<Key> tls_keys;
        return tls_keys;
    }
    void Evict() {
        auto& tls_keys = GetThreadLocalKeys();
        // uint32_t evicted_count = 0;
        for (auto it = tls_keys.rbegin(); it != tls_keys.rend(); ++it) {
            auto key = *it;
            map.erase(key);

        }
        tls_keys.clear();
    }

    static constexpr size_t EVICTION_TRIGGER_RATIO = 90; // 触发阈值百分比
    static constexpr size_t EVICTION_BATCH_RATIO = 5;   // 批量淘汰百分比
    libcuckoo::cuckoohash_map<Key, HotKeyCacheEntry> map;
    const size_t max_size;
    size_t eviction_trigger_threshold;  // 触发淘汰的大小阈值
    size_t eviction_batch_size;          // 每次淘汰多少个key
    std::atomic_flag evicting_flag = ATOMIC_FLAG_INIT;
    
    CountMinSketch cms;
    static constexpr uint32_t CMS_PROMOTION_THRESHOLD = 100;
    std::atomic<size_t> current_size = 0;  // 类成员变量
    std::atomic<size_t> current_valid_size = 0;

};

template <class Comparator>
class HotKeyCache {
public:
    HotKeyCache(size_t shard_cnt = 16, size_t per_shard_sz = 512)
        : shard_count(shard_cnt) {
        shards.reserve(shard_count);
        for (size_t i = 0; i < shard_count; ++i) {
            shards.emplace_back(std::make_unique<HotKeyCacheShard<Comparator>>(per_shard_sz));
        }
    }

    // 显式定义移动赋值
    HotKeyCache& operator=(HotKeyCache&& other) noexcept {
        if (this != &other) {
            shards = std::move(other.shards);
            // shard_count 必须是相同的
        }
        return *this;
    }

    typename InlineSkipList<Comparator>::Node* Get(const Key& key) {
        return shards[HashKeyToShard(key)]->Get(key);
    }

    void Add(const Key& key, typename InlineSkipList<Comparator>::Node* node) {
        shards[HashKeyToShard(key)]->Add(key, node);
    }

private:
    size_t HashKeyToShard(const Key& key) const {
        return std::hash<Key>{}(key) % shard_count;
    }

    std::vector<std::unique_ptr<HotKeyCacheShard<Comparator>>> shards;
    size_t shard_count;  // 移除 const
};