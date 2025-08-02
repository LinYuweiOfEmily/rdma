  #pragma once

  #include <atomic>
  // #include <queue>
  #include <list>
  #include <vector>

  #include "CacheEntry.h"
  #include "HugePageAlloc.h"
  #include "Timer.h"
  #include "WRLock.h"
  #include "third_party/inlineskiplist.h"
  #include "third_party/HotKeyCache.h"
  #include "thread_epoch.h"

  using CacheSkipList = InlineSkipList<CacheEntryComparator>;
  // using CacheBPTree = dfly::BPTree<CacheEntry>;
  using HotCache = HotKeyCache<CacheEntryComparator>;
  struct alignas(64) DelayFreeList {
    std::list<std::pair<void *, uint64_t>> list;
    WRLock lock;
  };

  class IndexCache {
  public:
    // 构造函数
    IndexCache(int cache_size);
    ~IndexCache();

    // 添加到缓存
    bool add_to_cache(InternalPage *page, int thread_id);
    // 添加一个子节点（通常是 InternalEntry 对应的子索引项）到缓存中。
    bool add_sub_node(GlobalAddress addr, int group_id, int granularity,
                      int guard_offset, InternalEntry *guard, int size, Key min,
                      Key max, int thread_id);
    
    // 从缓存中查找包含 k 的 CacheEntry。
    const CacheEntry *search_from_cache(const Key &k, GlobalAddress *addr,
                                        GlobalAddress *parent_addr);
    // 范围查找，从 from 到 to 的所有命中的 InternalPage 都填入结果。
    void search_range_from_cache(const Key &from, const Key &to,
                                std::vector<InternalPage *> &result);
    
    // 添加一个 key 范围和其对应的 page。
    bool add_entry(const Key &from, const Key &to, InternalPage *ptr);
    // 精确匹配或者范围查找，返回符合条件的 CacheEntry 指针。
    const CacheEntry *find_entry(const Key &k);
    const CacheEntry *find_entry_hot_cache(const Key &from, const Key &to);
    const CacheEntry *find_entry(const Key &from, const Key &to);

    // 根据 CacheEntry 或 Key 范围使缓存失效。
    bool invalidate(const CacheEntry *entry, int thread_id);
    bool invalidate(const Key &from, const Key &to, int thread_id);

    // 获取一个随机缓存条目，freq 表示被访问频率。
    const CacheEntry *get_a_random_entry(uint64_t &freq);

    void statistics();

    void bench();

    void free_delay();
    
    ThreadStatus *thread_status;

  private:
    uint64_t cache_size;  // MB;
    std::atomic<int64_t> free_page_cnt;  // 空闲页数量
    std::atomic<int64_t> skiplist_node_cnt; // 跳表节点数量
    std::atomic<uint64_t> max_key{0}; // 最大键值
    int64_t all_page_cnt; // 所有的页数量

    // std::queue<std::pair<void *, uint64_t>> delay_free_list;
    // WRLock free_lock;

    DelayFreeList delay_free_lists[MAX_APP_THREAD];
    std::atomic_bool delay_free_stop_flag{false};
    std::thread free_delay_thread_;

    // SkipList
    CacheSkipList *skiplist;
    // HopKeyCache
    HotCache hot_cache;

    // 比较两个CacheEntry的比较器
    CacheEntryComparator cmp; 

    // 自定义分配器对象，给跳表分配内存
    Allocator alloc;

    // 被动驱逐策略
    void evict_one(int thread_id);
  };

  /**
   * @brief IndexCache 类的构造函数，用于初始化索引缓存对象。
   * 
   * @param cache_size 缓存大小，单位为 MB。
   */
  inline IndexCache::IndexCache(int cache_size) : cache_size(cache_size) {
    // 创建一个跳表实例，使用 CacheEntryComparator 作为比较器，Allocator 对象进行内存分配，最大层数为 21
    skiplist = new CacheSkipList(cmp, &alloc, 21, 2);
    // HopKeyCache
    hot_cache = HotCache();
    
    // tree = new CacheBPTree();
    // 计算缓存的总内存大小，单位为字节
    uint64_t memory_size = define::MB * cache_size;

    // 创建 ThreadStatus 对象，用于管理线程状态，传入最大应用线程数
    thread_status = new ThreadStatus(MAX_APP_THREAD);

    // 计算所有页的总数量，考虑每个内部页的最大分组数
    all_page_cnt = memory_size / kInternalPageSize * kMaxInternalGroup;
    // 初始化空闲页数量为所有页的总数量
    free_page_cnt.store(all_page_cnt);
    // 初始化跳表节点数量为 0
    skiplist_node_cnt.store(0);
    // 创建一个新线程，执行 free_delay 方法，用于延迟释放内存
    free_delay_thread_ = std::thread(&IndexCache::free_delay, this);
  }

  /**
   * @brief IndexCache 类的析构函数，用于释放 IndexCache 对象占用的资源。
   * 
   * 此析构函数会停止延迟释放线程，等待其结束，然后释放跳表和线程状态对象的内存。
   */
  IndexCache::~IndexCache() {
    // 设置延迟释放线程的停止标志，使用 release 内存序确保之前的操作对其他线程可见
    delay_free_stop_flag.store(true, std::memory_order_release);
    // 检查延迟释放线程是否可加入（即线程正在运行且未被加入过）
    if (free_delay_thread_.joinable()) {
      // 等待延迟释放线程执行完毕
      free_delay_thread_.join();
    }
    // 释放跳表对象占用的内存
    delete skiplist;
    // 释放线程状态对象占用的内存
    delete thread_status;
  }

  /**
   * @brief 向索引缓存中添加一个条目。
   * 
   * 该方法尝试在跳表中插入一个新的缓存条目。如果插入成功，会更新跳表节点计数和最大键值。
   * 
   * @param from 缓存条目的起始键值。
   * @param to 缓存条目的结束键值（实际存储时会减 1）。
   * @param ptr 指向内部页的指针，该内部页关联到这个缓存条目。
   * @return 如果插入成功返回 true，否则返回 false。
   */
  inline bool IndexCache::add_entry(const Key &from, const Key &to,
                                    InternalPage *ptr) {
    // TODO memory leak
    // 从跳表分配器中分配足够的内存来存储一个 CacheEntry 对象
    auto buf = skiplist->AllocateKey(sizeof(CacheEntry));
    // 将分配的内存 reinterpret_cast 为 CacheEntry 引用，方便后续操作
    auto &e = *(CacheEntry *)buf;
    // 设置缓存条目的起始键值
    e.from = from;
    // 设置缓存条目的结束键值，注意这里减 1 是一个重要操作
    e.to = to - 1; // !IMPORTANT;
    // 设置缓存条目关联的内部页指针
    e.ptr = ptr;

    // 尝试以并发安全的方式将新的缓存条目插入到跳表中
    // bool res = skiplist->InsertConcurrently(buf);
    static thread_local void* hint = nullptr;
    bool res = skiplist->InsertWithHintConcurrently(buf, &hint);
    if (res) {
      // 如果插入成功，原子地增加跳表节点的计数
      skiplist_node_cnt.fetch_add(1);
      // 检查新条目的起始键值是否大于当前最大键值
      if (from > max_key.load(std::memory_order_acquire)) {
        // 如果是，则更新最大键值，使用 release 内存序确保之前的操作对其他线程可见
        max_key.store(from, std::memory_order_release);
      }else {
        // 插入失败时可以选择重置 hint，以避免下次复用可能是无效 splice
        delete reinterpret_cast<CacheSkipList::Splice*>(hint);
        hint = nullptr;
      }
    }
    return res;
  }

  /**
   * @brief 在索引缓存中查找指定键范围的缓存条目。
   * 
   * 该方法使用跳表迭代器从跳表中查找第一个键范围大于等于指定范围的缓存条目。
   * 如果找到有效的条目，则返回该条目的指针；否则返回 nullptr。
   * 
   * @param from 查找范围的起始键值。
   * @param to 查找范围的结束键值（实际比较时会减 1）。
   * @return 若找到符合条件的缓存条目，返回该条目的指针；未找到则返回 nullptr。
   */
  inline const CacheEntry *IndexCache::find_entry(const Key &from,
                                                  const Key &to) {
    // 创建一个跳表迭代器，用于遍历跳表
    CacheSkipList::Iterator iter(skiplist);
    // static thread_local void* hint = nullptr;

    // 创建一个临时的 CacheEntry 对象，设置其起始键和结束键
    CacheEntry e;
    e.from = from;
    // 结束键减 1，与存储时的处理方式保持一致
    e.to = to - 1;
    // 将迭代器定位到第一个键范围大于等于临时 CacheEntry 的位置
    iter.Seek((char *)&e);
    // iter.SeekWithHint((char *)&e, &hint);

    // 检查迭代器是否指向有效的条目
    if (iter.Valid()) {
      // 获取迭代器当前指向的缓存条目
      auto val = (const CacheEntry *)iter.key();
      // hot_cache.Add(from, iter.GetNode());
      return val;
    } else {
      // 未找到符合条件的条目，返回 nullptr
      return nullptr;
    }
  }

  inline const CacheEntry *IndexCache::find_entry(const Key &k) {
    CacheSkipList::Node* node = hot_cache.Get(k);
    if (node != nullptr) {
      auto val = (const CacheEntry *)node->Key();
      // printf("find_entry_hot_cache %lu %lu\n", val->from, val->to);
      if (val->ptr != nullptr) {
        return val;
      }
    }
    return find_entry_hot_cache(k, k + 1);
  }
  inline const CacheEntry *IndexCache::find_entry_hot_cache(const Key &from,
                                                  const Key &to) {
    // 创建一个跳表迭代器，用于遍历跳表
    CacheSkipList::Iterator iter(skiplist);
    // static thread_local void* hint = nullptr;

    // 创建一个临时的 CacheEntry 对象，设置其起始键和结束键
    CacheEntry e;
    e.from = from;
    // 结束键减 1，与存储时的处理方式保持一致
    e.to = to - 1;
    // 将迭代器定位到第一个键范围大于等于临时 CacheEntry 的位置
    iter.Seek((char *)&e);
    // iter.SeekWithHint((char *)&e, &hint);

    // 检查迭代器是否指向有效的条目
    if (iter.Valid()) {
      // 获取迭代器当前指向的缓存条目
      auto val = (const CacheEntry *)iter.key();
      hot_cache.Add(from, iter.GetNode());
      return val;
    } else {
      // 未找到符合条件的条目，返回 nullptr
      return nullptr;
    }
  }

  /**
   * @brief 将一个内部页添加到索引缓存中。
   * 
   * 该方法尝试将传入的内部页添加到索引缓存中。如果添加过程中出现冲突，会尝试更新已存在的缓存条目。
   * 若空闲页数量不足，会触发缓存淘汰操作。
   * 
   * @param page 指向要添加到缓存的内部页的指针。
   * @param thread_id 执行此操作的线程 ID。
   * @return 如果成功添加或更新缓存条目，返回 true；否则返回 false。
   */
  inline bool IndexCache::add_to_cache(InternalPage *page, int thread_id) {
    // 分配一块新的内存，大小为内部页的大小，用于存储复制后的内部页
    InternalPage *new_page = (InternalPage *)malloc(kInternalPageSize);
    // 将传入的内部页数据复制到新分配的内存中
    memcpy(reinterpret_cast<void *>(new_page), page, kInternalPageSize);
    // 初始化新内部页的缓存访问频率为 0
    new_page->hdr.index_cache_freq = 0;
    // 遍历所有内部页分组
    for (int i = 0 ; i < kMaxInternalGroup; ++i) {
      // 将新内部页的每个分组标记为已缓存
      new_page->hdr.grp_in_cache[i] = true;
      // 设置新内部页每个分组的读取粒度为传入页的读取粒度
      new_page->hdr.cache_read_gran[i] = page->hdr.read_gran;
    }
    // 断言新内部页的地址不为空
    assert(new_page->hdr.my_addr != GlobalAddress::Null());

    // 尝试将新内部页添加到缓存中，键范围为内部页的最低键到最高键
    if (this->add_entry(page->hdr.lowest, page->hdr.highest, new_page)) {
      // 原子地减少空闲页数量，减少的数量为最大内部页分组数
      auto v = free_page_cnt.fetch_sub(kMaxInternalGroup);
      // 若空闲页数量不足
      if (v <= 0) {
        // 触发缓存淘汰操作
        evict_one(thread_id);
      }

      return true;
    } else {  // 出现冲突，即缓存中已存在相同键范围的条目
      // 查找缓存中键范围相同的条目
      auto e = this->find_entry(page->hdr.lowest, page->hdr.highest);
      // 检查找到的条目是否与当前页的键范围完全匹配
      if (e && e->from == page->hdr.lowest && e->to == page->hdr.highest - 1) {
        // 获取该条目中原来指向的内部页指针
        auto ptr = e->ptr;

        // 使用原子操作尝试将条目中的内部页指针更新为新的内部页指针
        if (__sync_bool_compare_and_swap(&(e->ptr), ptr, new_page)) {
          // 若原来的内部页指针为空
          if (ptr == nullptr) {
            // 原子地减少空闲页数量，减少的数量为最大内部页分组数
            auto v = free_page_cnt.fetch_sub(kMaxInternalGroup);
            // 若空闲页数量不足
            if (v <= 0) {
              // 触发缓存淘汰操作
              evict_one(thread_id);
            }
          } else {
            // 计算原来内部页中已缓存的分组数量
            int old_cnt = 0;
            for (int i = 0; i < kMaxInternalGroup; ++i) {
              old_cnt += ptr->hdr.grp_in_cache[i];
            }
            // 若原来已缓存的分组数量小于最大分组数
            if (old_cnt < kMaxInternalGroup) {
              // 原子地减少空闲页数量，减少的数量为两者差值
              auto v = free_page_cnt.fetch_sub(kMaxInternalGroup - old_cnt);
              // 若空闲页数量不足
              if (v <= 0) {
                // 触发缓存淘汰操作
                evict_one(thread_id);
              }
            }
            // 加写锁，将原来的内部页指针添加到延迟释放列表中
            delay_free_lists[thread_id].lock.wLock();
            delay_free_lists[thread_id].list.push_back(
                std::make_pair(ptr, asm_rdtsc()));
            // 释放写锁
            delay_free_lists[thread_id].lock.wUnlock();
          }
          return true;
        }
      }

      // 若更新失败，释放新分配的内存
      free(new_page);
      return false;
    }
  }

  /**
   * @brief 向索引缓存中添加或更新一个子节点。
   * 
   * 该方法尝试将一个子节点添加到索引缓存中。如果缓存中已存在相同键范围的条目，则更新该条目；
   * 否则，创建一个新的缓存条目。若空闲页数量不足，会触发缓存淘汰操作。
   * 
   * @param addr 子节点的全局地址。
   * @param group_id 子节点所在的分组 ID。
   * @param granularity 子节点的读取粒度，取值为 gran_quarter 或 gran_half。
   * @param guard_offset 子节点数据在内部页中的偏移量。
   * @param guard 指向子节点数据的指针。
   * @param size 子节点数据的大小。
   * @param min 子节点的最小键值。
   * @param max 子节点的最大键值。
   * @param thread_id 执行此操作的线程 ID。
   * @return 如果成功添加或更新子节点，返回 true；否则返回 false。
   */
  inline bool IndexCache::add_sub_node(GlobalAddress addr, int group_id,
                                      int granularity, int guard_offset,
                                      InternalEntry *guard, int size, Key min,
                                      Key max, int thread_id) {
    // 分配一块新的内存，大小为内部页的大小，用于存储新的或更新后的内部页
    InternalPage *new_page = (InternalPage *)malloc(kInternalPageSize);
    // memset(new_page, 0, kInternalPageSize);

    // 在缓存中查找键范围为 [min, max) 的条目
    auto e = this->find_entry(min, max);
    if (e && e->from == min && e->to == max - 1) {  // 找到已有条目，进行更新操作
      // 获取已有条目中指向的内部页指针
      InternalPage *ptr = e->ptr;
      if (ptr) {
        // 更新操作：将已有内部页的数据复制到新页
        memcpy(reinterpret_cast<char *>(new_page), ptr, kInternalPageSize);
        // 将新的子节点数据复制到新页的指定偏移位置
        memcpy(reinterpret_cast<char *>(new_page) + guard_offset, guard, size);
      } else {
        // 添加操作：将新页的内存初始化为 0
        memset(reinterpret_cast<char *>(new_page), 0, kInternalPageSize);
        // 将新的子节点数据复制到新页的指定偏移位置
        memcpy(reinterpret_cast<char *>(new_page) + guard_offset, guard, size);
        // 设置新页的最低键值
        new_page->hdr.lowest = min;
        // 设置新页的最高键值
        new_page->hdr.highest = max;
        // 设置新页的兄弟指针为空
        new_page->hdr.sibling_ptr = GlobalAddress::Null();
        // 设置新页的层级为 1
        new_page->hdr.level = 1;
        // 初始化新页的缓存访问频率为 0
        new_page->hdr.index_cache_freq = 0;
        // 设置新页的全局地址
        new_page->hdr.my_addr = addr;
      }
      // 计算需要占用的页数量
      int cnt = 0;
      if (granularity == gran_quarter) {
        // 若粒度为四分之一，标记对应分组为已缓存，设置读取粒度为四分之一
        new_page->hdr.grp_in_cache[group_id] = true;
        new_page->hdr.cache_read_gran[group_id] = gran_quarter;
        cnt = 1;
      } else {
        cnt = 2;
        if (group_id < 2) {
          // 若分组 ID 小于 2，标记前两个分组为已缓存，设置读取粒度为二分之一
          new_page->hdr.grp_in_cache[0] = true;
          new_page->hdr.grp_in_cache[1] = true;
          new_page->hdr.cache_read_gran[0] = gran_half;
          new_page->hdr.cache_read_gran[1] = gran_half;
        } else {
          // 若分组 ID 大于等于 2，标记后两个分组为已缓存，设置读取粒度为二分之一
          new_page->hdr.grp_in_cache[2] = true;
          new_page->hdr.grp_in_cache[3] = true;
          new_page->hdr.cache_read_gran[2] = gran_half;
          new_page->hdr.cache_read_gran[3] = gran_half;
        }
      }
      // 使用原子操作尝试将条目中的内部页指针更新为新的内部页指针
      if (__sync_bool_compare_and_swap(&(e->ptr), ptr, new_page)) {
        if (ptr == nullptr) {
          // 若原来的内部页指针为空，原子地减少空闲页数量
          auto v = free_page_cnt.fetch_sub(cnt);
          // 若空闲页数量不足，触发缓存淘汰操作
          if (v <= 0) {
            evict_one(thread_id);
          }
        } else {
          // 计算原来内部页中已缓存的分组数量
          int old_cnt = 0;
          // 计算新内部页中已缓存的分组数量
          int new_cnt = 0;
          for (int i = 0; i < kMaxInternalGroup; ++i) {
            old_cnt += ptr->hdr.grp_in_cache[i];
            new_cnt += new_page->hdr.grp_in_cache[i];
          }
          // 若新旧缓存分组数量不同，原子地调整空闲页数量
          if (old_cnt != new_cnt) {
            auto v = free_page_cnt.fetch_sub(new_cnt - old_cnt);
            // 若空闲页数量不足，触发缓存淘汰操作
            if (v <= 0) {
              evict_one(thread_id);
            }
          }
          // 加写锁，将原来的内部页指针添加到延迟释放列表中
          delay_free_lists[thread_id].lock.wLock();
          delay_free_lists[thread_id].list.push_back(
              std::make_pair(ptr, asm_rdtsc()));
          // 释放写锁
          delay_free_lists[thread_id].lock.wUnlock();
        }
        return true;
      }
    } else {
      // 未找到已有条目，进行添加操作
      // 将新页的内存初始化为 0
      memset(reinterpret_cast<char *>(new_page), 0, kInternalPageSize);
      // 将新的子节点数据复制到新页的指定偏移位置
      memcpy(reinterpret_cast<char *>(new_page) + guard_offset, guard, size);
      // 设置新页的最低键值
      new_page->hdr.lowest = min;
      // 设置新页的最高键值
      new_page->hdr.highest = max;
      // 设置新页的兄弟指针为空
      new_page->hdr.sibling_ptr = GlobalAddress::Null();
      // 设置新页的层级为 1
      new_page->hdr.level = 1;
      // 初始化新页的缓存访问频率为 0
      new_page->hdr.index_cache_freq = 0;
      // 设置新页的全局地址
      new_page->hdr.my_addr = addr;
      // 计算需要占用的页数量
      int cnt = 0;
      if (granularity == gran_quarter) {
        cnt = 1;
        // 若粒度为四分之一，标记对应分组为已缓存，设置读取粒度为四分之一
        new_page->hdr.grp_in_cache[group_id] = true;
        new_page->hdr.cache_read_gran[group_id] = gran_quarter;

      } else {
        cnt = 2;
        if (group_id < 2) {
          // 若分组 ID 小于 2，标记前两个分组为已缓存，设置读取粒度为二分之一
          new_page->hdr.grp_in_cache[0] = true;
          new_page->hdr.grp_in_cache[1] = true;
          new_page->hdr.cache_read_gran[0] = gran_half;
          new_page->hdr.cache_read_gran[1] = gran_half;
        } else {
          // 若分组 ID 大于等于 2，标记后两个分组为已缓存，设置读取粒度为二分之一
          new_page->hdr.grp_in_cache[2] = true;
          new_page->hdr.grp_in_cache[3] = true;
          new_page->hdr.cache_read_gran[2] = gran_half;
          new_page->hdr.cache_read_gran[3] = gran_half;
        }
      }
      // 尝试将新的内部页添加到缓存中
      if (this->add_entry(min, max, new_page)) {
        // 原子地减少空闲页数量
        auto v = free_page_cnt.fetch_sub(cnt);
        // 若空闲页数量不足，触发缓存淘汰操作
        if (v <= 0) {
          evict_one(thread_id);
        }
        return true;
      }
    }
    // 若添加或更新失败，释放新分配的内存
    free(new_page);
    return false;
  }

  /**
   * @brief 从索引缓存中查找包含指定键的缓存条目，并获取相关地址信息。
   * 
   * 该方法会在索引缓存中查找包含指定键 `k` 的缓存条目，若找到有效条目且键在其范围内，
   * 会更新该条目的访问频率，检查键所在分组是否已缓存，根据缓存粒度确定查找范围，
   * 最终找到合适的子节点地址并更新 `addr`，同时将父节点地址更新到 `parent_addr`。
   * 
   * @param k 要查找的键。
   * @param addr 用于存储找到的子节点的全局地址。若未找到，会被设置为 `GlobalAddress::Null()`。
   * @param parent_addr 用于存储找到的子节点的父节点的全局地址。若不满足条件，不会被更新。
   * @return 若找到符合条件的缓存条目，返回该条目的指针；否则返回 nullptr。
   */
  inline const CacheEntry *IndexCache::search_from_cache(
      const Key &k, GlobalAddress *addr, GlobalAddress *parent_addr) {
    // 调用 find_entry 方法查找包含键 k 的缓存条目
    auto entry = find_entry(k);

    // 获取缓存条目对应的内部页指针，若条目不存在则为 nullptr
    InternalPage *page = entry ? entry->ptr : nullptr;

    // 检查内部页是否存在，且键 k 是否在内部页的键范围 [lowest, highest) 内
    if (page && k >= page->hdr.lowest && k < page->hdr.highest) {
      // 若内部页的缓存访问频率小于最大值，则增加访问频率
      if (page->hdr.index_cache_freq < UINT64_MAX) {
        page->hdr.index_cache_freq++;
      }

      // 计算键 k 所在的分组 ID
      int group_id = get_key_group(k, page->hdr.lowest, page->hdr.highest);
      // 检查该分组是否已缓存，若未缓存则返回 nullptr
      if (page->hdr.grp_in_cache[group_id] == false) {
        return nullptr;
      }
      // 获取当前分组的读取粒度
      uint8_t cur_group_gran = page->hdr.cache_read_gran[group_id];

      // 定义查找的起始索引和条目数量
      int start_idx, cnt;
      if (cur_group_gran == gran_quarter) {
        // 若粒度为四分之一，计算起始索引和条目数量
        start_idx = kGroupCardinality * group_id;
        cnt = kGroupCardinality;
      } else if (cur_group_gran == gran_half) {
        // 若粒度为二分之一，根据分组 ID 计算起始索引和条目数量
        start_idx = group_id < 2 ? 0 : kGroupCardinality * 2;
        cnt = kGroupCardinality * 2;
      } else {
        // 断言粒度为全部，设置起始索引为 0，条目数量为内部页的总条目数
        assert(cur_group_gran == gran_full);
        start_idx = 0;
        cnt = kInternalCardinality;
      }
      // 向前扩展一个条目，在之前的分组中多查找一个
      --start_idx;
      // 增加查找的条目数量
      ++cnt;
      // 获取内部页中从起始索引开始的条目指针
      InternalEntry *entries = page->records + start_idx;
      // 初始化找到的条目的索引为 -1，表示未找到
      int idx = -1;
      // 遍历指定数量的条目
      for (int i = 0; i < cnt; ++i) {
        // 检查条目指针是否有效
        if (entries[i].ptr != GlobalAddress::Null()) {
          // 检查键 k 是否大于等于当前条目的键
          if (k >= entries[i].key) {
            // 若未找到合适条目或当前条目的键更大，则更新找到的条目的索引
            if (idx == -1 || entries[i].key > entries[idx].key) {
              idx = i;
            }
          }
        }
      }
      if (idx != -1) {
        // 若找到合适条目，将其指针赋值给 addr
        *addr = entries[idx].ptr;
      } else {
        // 若未找到合适条目，将 addr 设置为 Null
        *addr = GlobalAddress::Null();
      }
      // 检查缓存条目对应的内部页指针是否有效，且找到的子节点地址不为 Null
      if (entry->ptr && *addr != GlobalAddress::Null()) {
        // 将父节点地址赋值给 parent_addr
        *parent_addr = page->hdr.my_addr;
        // 返回找到的缓存条目指针
        return entry;
      }
    }

    // 若未找到符合条件的条目，返回 nullptr
    return nullptr;
  }

  /**
   * @brief 从索引缓存中进行范围查找，将命中的内部页指针添加到结果向量中。
   * 
   * 该方法会在索引缓存中查找键范围在 [from, to] 内的所有缓存条目，
   * 并将符合条件的内部页指针添加到结果向量中。查找过程会跳过被完全包含的内部页。
   * 
   * @param from 查找范围的起始键值。
   * @param to 查找范围的结束键值。
   * @param result 用于存储命中的内部页指针的向量，函数开始时会清空该向量。
   */
  inline void IndexCache::search_range_from_cache(
      const Key &from, const Key &to, std::vector<InternalPage *> &result) {
    // 创建一个跳表迭代器，用于遍历跳表
    CacheSkipList::Iterator iter(skiplist);

    // 清空结果向量，确保结果向量为空
    result.clear();

    // 创建一个临时的 CacheEntry 对象，设置起始键和结束键为 from，用于定位迭代器
    CacheEntry e;
    e.from = from;
    e.to = from;
    // 将迭代器定位到第一个键范围大于等于临时 CacheEntry 的位置
    iter.Seek((char *)&e);

    // 当迭代器指向有效条目时，持续遍历
    while (iter.Valid()) {
      // 获取迭代器当前指向的缓存条目
      auto val = (const CacheEntry *)iter.key();
      // 检查缓存条目对应的内部页指针是否有效
      if (val->ptr) {
        // 若当前缓存条目的起始键大于结束键 to，说明后续条目也不在查找范围内，直接返回
        if (val->from > to) {
          return;
        }
        // 若结果向量为空，或者当前内部页不被结果向量中最后一个内部页完全包含
        if (result.size() == 0 ||
            (result.back()->hdr.lowest < val->ptr->hdr.lowest &&
            result.back()->hdr.highest > val->ptr->hdr.highest)) {
          // 将当前内部页指针添加到结果向量中
          result.push_back(val->ptr);
        }
      }
      // 迭代器移动到下一个条目
      iter.Next();
    }
  }

  /**
   * @brief 使指定的缓存条目失效，并将对应的内部页标记为待释放。
   * 
   * 该方法会尝试将指定缓存条目的内部页指针置为 0，若操作成功，会统计该内部页中已缓存的分组数量，
   * 并将该内部页添加到延迟释放列表中，同时增加空闲页的数量。
   * 
   * @param entry 指向要失效的缓存条目的指针。
   * @param thread_id 执行此操作的线程 ID。
   * @return 若成功使缓存条目失效，返回 true；否则返回 false。
   */
  inline bool IndexCache::invalidate(const CacheEntry *entry, int thread_id) {
    // 获取缓存条目对应的内部页指针
    auto ptr = entry->ptr;

    // 若内部页指针为空，说明该条目已无有效数据，直接返回 false
    if (ptr == nullptr) {
      return false;
    }

    // 使用原子操作尝试将缓存条目的内部页指针置为 0
    // 若操作成功，说明成功抢占该条目，可进行后续失效操作
    if (__sync_bool_compare_and_swap(&(entry->ptr), ptr, 0)) {
      // 初始化计数器，用于统计内部页中已缓存的分组数量
      int cnt = 0;
      // 遍历内部页的所有分组
      for (int i = 0; i < kMaxInternalGroup; ++i) {
        // 累加已缓存的分组数量
        cnt += ptr->hdr.grp_in_cache[i];
      }
      // 加写锁，确保多线程环境下对延迟释放列表操作的线程安全
      delay_free_lists[thread_id].lock.wLock();
      // 将内部页指针和当前时间戳作为一对元素添加到延迟释放列表中
      delay_free_lists[thread_id].list.push_back(
          std::make_pair(ptr, asm_rdtsc()));
      // 释放写锁
      delay_free_lists[thread_id].lock.wUnlock();
      // 原子地增加空闲页的数量，增加的数量为已缓存的分组数量
      free_page_cnt.fetch_add(cnt);
      return true;
    }

    // 若原子操作失败，说明该条目已被其他线程修改，返回 false
    return false;
  }

  /**
   * @brief 根据键范围使对应的缓存条目失效。
   * 
   * 该方法会在索引缓存中查找指定键范围的缓存条目，若找到且键范围完全匹配，
   * 则调用 `invalidate` 方法使该缓存条目失效。
   * 
   * @param from 要查找的缓存条目的起始键值。
   * @param to 要查找的缓存条目的结束键值（实际比较时会减 1）。
   * @param thread_id 执行此操作的线程 ID。
   * @return 若成功找到匹配的缓存条目并使其失效，返回 true；否则返回 false。
   */
  inline bool IndexCache::invalidate(const Key &from, const Key &to,
                                    int thread_id) {
    // 调用 find_entry 方法查找指定键范围的缓存条目
    auto e = find_entry(from, to);
    // 检查是否找到有效的缓存条目，且其起始键和结束键与传入的键范围完全匹配
    if (e && e->from == from && e->to == to - 1) {
      // 若匹配成功，调用 invalidate 方法使该缓存条目失效
      return invalidate(e, thread_id);
    }
    // 若未找到匹配的缓存条目，返回 false
    return false;
  }

  /**
   * @brief 获取一个随机的缓存条目，并返回其访问频率。
   * 
   * 该方法会生成一个随机键，在跳表中查找包含该键的缓存条目。
   * 若找到有效的缓存条目且其内部页指针未改变，则返回该条目指针并更新传入的访问频率参数；
   * 若未找到符合条件的条目，则重新生成随机键进行查找。
   * 
   * @param freq 引用参数，用于存储找到的缓存条目的访问频率。
   * @return 若找到符合条件的缓存条目，返回该条目的指针；否则持续重试。
   */
  inline const CacheEntry *IndexCache::get_a_random_entry(uint64_t &freq) {
    // 使用当前时间戳作为随机数种子，确保每次生成的随机数不同
    uint32_t seed = asm_rdtsc();
  retry:
    // 生成一个随机键，范围是 0 到当前最大键值（采用 relaxed 内存序读取）
    auto k = rand_r(&seed) % max_key.load(std::memory_order_relaxed);
    // 创建一个跳表迭代器，用于遍历跳表
    CacheSkipList::Iterator iter(skiplist);
    // 创建一个临时的 CacheEntry 对象，设置其起始键和结束键为随机生成的键
    CacheEntry tmp;
    tmp.from = k;
    tmp.to = k;
    // 将迭代器定位到第一个键范围大于等于临时 CacheEntry 的位置
    iter.Seek((char *)&tmp);

    // 当迭代器指向有效条目时，持续遍历
    while (iter.Valid()) {
      // 获取迭代器当前指向的缓存条目
      CacheEntry *e = (CacheEntry *)iter.key();
      // 获取缓存条目对应的内部页指针，若条目不存在则为 nullptr
      InternalPage *ptr = e ? e->ptr : nullptr;
      // 检查内部页指针是否有效
      if (ptr) {
        // 获取该内部页的缓存访问频率
        freq = ptr->hdr.index_cache_freq;
        // 再次检查缓存条目的内部页指针是否仍然指向之前获取的内部页
        if (e->ptr == ptr) {
          // 若指针未改变，返回该缓存条目指针
          return e;
        }
      }
      // 迭代器移动到下一个条目
      iter.Next();
    }
    // 若未找到符合条件的条目，跳转到 retry 标签处重新生成随机键查找
    goto retry;
  }

  /**
   * @brief 从索引缓存中驱逐一个缓存条目。
   * 
   * 该方法通过随机选择两个缓存条目，比较它们的访问频率，
   * 并将访问频率较低的条目从缓存中驱逐，使其失效。
   * 
   * @param thread_id 执行此操作的线程 ID，用于延迟释放列表的操作。
   */
  inline void IndexCache::evict_one(int thread_id) {
    // 定义两个变量，用于存储随机选择的两个缓存条目的访问频率
    uint64_t freq1, freq2;
    // 调用 get_a_random_entry 方法，随机获取一个缓存条目，并将其访问频率存储在 freq1 中
    auto e1 = get_a_random_entry(freq1);
    // 再次调用 get_a_random_entry 方法，随机获取另一个缓存条目，并将其访问频率存储在 freq2 中
    auto e2 = get_a_random_entry(freq2);

    // 比较两个缓存条目的访问频率
    if (freq1 < freq2) {
      // 若 freq1 小于 freq2，说明 e1 的访问频率较低，调用 invalidate 方法使 e1 失效
      invalidate(e1, thread_id);
    } else {
      // 否则，说明 e2 的访问频率较低，调用 invalidate 方法使 e2 失效
      invalidate(e2, thread_id);
    }
  }

  inline void IndexCache::statistics() {
    printf("[skiplist node: %ld]  [page cache: %ld]\n", skiplist_node_cnt.load(),
          all_page_cnt - free_page_cnt.load());
  }

  inline void IndexCache::bench() {
    Timer t;
    t.begin();
    const int loop = 100000;

    for (int i = 0; i < loop; ++i) {
      uint64_t r = rand() % (5 * define::MB);
      this->find_entry(r);
    }

    t.end_print(loop);
  }


  /**
   * @brief 延迟释放延迟释放列表中的内存。
   * 
   * 该方法会持续运行，直到 `delay_free_stop_flag` 被设置为 true。
   * 它会定期将所有线程的延迟释放列表中的元素转移到本地列表，
   * 等待 RCU 屏障确保没有线程正在访问这些内存，
   * 然后释放那些已经延迟足够长时间的内存。
   */
  void IndexCache::free_delay() {
    // 定义一个本地列表，用于临时存储从各个线程的延迟释放列表中转移过来的元素
    std::list<std::pair<void *, uint64_t>> local_list;

    // 持续循环，直到延迟释放停止标志被设置为 true
    while (!delay_free_stop_flag.load(std::memory_order_acquire)) {
      // 遍历所有应用线程
      for (int i = 0; i < MAX_APP_THREAD; ++i) {
        // 加写锁，确保在转移元素时不会有其他线程修改该线程的延迟释放列表
        delay_free_lists[i].lock.wLock();
        // 将当前线程的延迟释放列表中的所有元素转移到本地列表的末尾
        local_list.splice(local_list.end(), delay_free_lists[i].list);
        // 释放写锁
        delay_free_lists[i].lock.wUnlock();
      }
      // 执行 RCU 屏障，确保所有正在进行的读操作完成，避免释放正在被访问的内存
      thread_status->rcu_barrier();
      // 获取本地列表的起始迭代器
      auto it = local_list.begin();
      // 遍历本地列表中的元素
      for (; it != local_list.end(); ++it) {
        // 计算当前时间与元素添加到延迟释放列表时的时间差
        if (asm_rdtsc() - it->second > 5000ul * 10) {
          // 若时间差大于设定的阈值（5000 * 10），则释放该元素对应的内存
          free(it->first);
        } else {
          // 若时间差小于阈值，说明延迟时间不够，停止遍历
          break;
        }
      }
      // 从本地列表的起始位置到迭代器 it 之前的元素（不包括 it 指向的元素）进行擦除
      local_list.erase(local_list.begin(), it);  
      // 线程休眠 5 微秒，减少 CPU 占用
      usleep(5);
    }
  }
