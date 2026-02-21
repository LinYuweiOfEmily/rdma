#pragma once
#include <unordered_map>
#include <list>
#include "inlineskiplist.h"
#include "../CacheEntry.h"
#include "countmin_sketch.h"

using Key = uint64_t;

template <class Comparator>
class ThreadLocalHotCache {
public:
    explicit ThreadLocalHotCache(size_t max_sz = 512)
        : max_size(max_sz),
          window_size(max_sz / 8),
          main_size(max_sz - window_size) {}

    typename InlineSkipList<Comparator>::Node* Get(const Key& key) {
        // Window 区
        auto wit = window_map.find(key);
        if (wit != window_map.end()) {
            window_list.splice(window_list.begin(), window_list, wit->second);
            cms.record(key);
            TryPromote(key);
            return wit->second->second;
        }

        // Main 区
        auto mit = main_map.find(key);
        if (mit != main_map.end()) {
            cms.record(key);
            return mit->second->second;
        }

        return nullptr;
    }

    void Add(const Key& key, typename InlineSkipList<Comparator>::Node* node) {
        if (!node) return;

        auto wit = window_map.find(key);
        if (wit != window_map.end()) {
            wit->second->second = node;
            window_list.splice(window_list.begin(), window_list, wit->second);
            cms.record(key);
            TryPromote(key);
            return;
        }

        auto mit = main_map.find(key);
        if (mit != main_map.end()) {
            mit->second->second = node;
            cms.record(key);
            return;
        }

        // 新 key，插入 Window
        if (window_list.size() >= window_size) EvictWindow();
        window_list.emplace_front(key, node);
        window_map[key] = window_list.begin();
        cms.record(key);
        TryPromote(key);
    }

    static ThreadLocalHotCache& Instance() {
        thread_local ThreadLocalHotCache instance;
        return instance;
    }

private:
    void EvictWindow() {
        auto& victim = window_list.back();
        window_map.erase(victim.first);
        window_list.pop_back();
    }

    void EvictMain() {
        auto& victim = main_list.back();
        main_map.erase(victim.first);
        main_list.pop_back();
    }

    void TryPromote(const Key& key) {
        if (cms.estimate(key) >= promotion_threshold) {
            auto wit = window_map.find(key);
            if (wit == window_map.end()) return;

            if (main_list.size() >= main_size) EvictMain();
            main_list.emplace_front(wit->second->first, wit->second->second);
            main_map[key] = main_list.begin();

            window_list.erase(wit->second);
            window_map.erase(wit);
        }
    }

private:
    size_t max_size;
    size_t window_size;
    size_t main_size;

    // Window 区 (LRU)
    std::list<std::pair<Key, typename InlineSkipList<Comparator>::Node*>> window_list;
    std::unordered_map<Key, typename std::list<std::pair<Key, typename InlineSkipList<Comparator>::Node*>>::iterator> window_map;

    // Main 区 (近似 LFU)
    std::list<std::pair<Key, typename InlineSkipList<Comparator>::Node*>> main_list;
    std::unordered_map<Key, typename std::list<std::pair<Key, typename InlineSkipList<Comparator>::Node*>>::iterator> main_map;

    CountMinSketch cms;
    static constexpr uint32_t promotion_threshold = 5;
};
