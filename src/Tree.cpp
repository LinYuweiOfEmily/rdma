  #include "Tree.h"

  #include <city.h>
  #include <algorithm>
  #include <atomic>
  #include <iostream>
  #include <queue>
  #include <utility>
  #include <vector>
  #include <emmintrin.h>

  #include "IndexCache.h"
  #include "RdmaBuffer.h"
  #include "Timer.h"
  #include "Common.h"

  uint64_t leaf_update_hit_cnt[MAX_APP_THREAD] = {0};
  uint64_t leaf_insert_empty_cnt[MAX_APP_THREAD] = {0};
  uint64_t leaf_insert_path_group_fast_cnt[MAX_APP_THREAD] = {0};
  uint64_t leaf_insert_path_page_lock_cnt[MAX_APP_THREAD] = {0};
  uint64_t leaf_insert_retry_event_cnt[MAX_APP_THREAD] = {0};
  uint64_t leaf_insert_retry_step_cnt[MAX_APP_THREAD] = {0};
  uint64_t leaf_upgrade_to_x_cnt[MAX_APP_THREAD] = {0};
  uint64_t leaf_split_cnt[MAX_APP_THREAD] = {0};
  uint64_t leaf_split_occupancy_sum[MAX_APP_THREAD] = {0};
  uint64_t leaf_split_occupancy_max[MAX_APP_THREAD] = {0};
  uint64_t prefill_leaf_split_cnt[MAX_APP_THREAD] = {0};
  uint64_t prefill_leaf_split_page_occupancy_sum[MAX_APP_THREAD] = {0};
  uint64_t prefill_leaf_split_page_occupancy_max[MAX_APP_THREAD] = {0};
  uint64_t prefill_leaf_split_bucket_occupancy_sum[MAX_APP_THREAD] = {0};
  uint64_t prefill_leaf_split_bucket_occupancy_max[MAX_APP_THREAD] = {0};
  uint64_t prefill_leaf_split_group_occupancy_sum[MAX_APP_THREAD] = {0};
  uint64_t prefill_leaf_split_group_occupancy_max[MAX_APP_THREAD] = {0};
  uint64_t leaf_sibling_chase_cnt[MAX_APP_THREAD] = {0};
  uint64_t leaf_insert_parent_update_cnt[MAX_APP_THREAD] = {0};
  uint64_t leaf_insert_root_split_cnt[MAX_APP_THREAD] = {0};
  uint64_t leaf_stash_insert_attempt_cnt[MAX_APP_THREAD] = {0};
  uint64_t leaf_stash_insert_success_cnt[MAX_APP_THREAD] = {0};
  uint64_t leaf_stash_insert_full_cnt[MAX_APP_THREAD] = {0};
  uint64_t leaf_stash_insert_threshold_cnt[MAX_APP_THREAD] = {0};
  uint64_t leaf_split_stash_occupancy_sum[MAX_APP_THREAD] = {0};
  uint64_t leaf_split_stash_occupancy_max[MAX_APP_THREAD] = {0};
  uint64_t optimistic_update_attempt_cnt[MAX_APP_THREAD] = {0};
  uint64_t optimistic_update_success_cnt[MAX_APP_THREAD] = {0};
  uint64_t optimistic_update_cas_fail_cnt[MAX_APP_THREAD] = {0};
  uint64_t optimistic_update_split_abort_cnt[MAX_APP_THREAD] = {0};
  uint64_t optimistic_leaf_fast_path_hot_bypass_cnt[MAX_APP_THREAD] = {0};
  uint64_t optimistic_insert_attempt_cnt[MAX_APP_THREAD] = {0};
  uint64_t optimistic_insert_success_cnt[MAX_APP_THREAD] = {0};
  uint64_t optimistic_insert_cas_fail_cnt[MAX_APP_THREAD] = {0};
  uint64_t optimistic_insert_split_abort_cnt[MAX_APP_THREAD] = {0};
  uint64_t optimistic_insert_consistency_fail_cnt[MAX_APP_THREAD] = {0};
  uint64_t optimistic_insert_fallback_cnt[MAX_APP_THREAD] = {0};

// ================= Experiment Switches =================
// Baseline recommendation:
//   keep:   USE_SX_LOCK, BATCH_LOCK_READ, FINE_GRAINED_LEAF_NODE,
//           FINE_GRAINED_INTERNAL_NODE
//   disable: USE_OPTIMISTIC_UPDATE_HIT, USE_ADAPTIVE_HOT_BYPASS
//
// Current optimized recommendation:
//   enable: USE_OPTIMISTIC_UPDATE_HIT, USE_ADAPTIVE_HOT_BYPASS
//
// User-added scheduler / debug switches:
//   USE_AP, USE_BATCH_POLL, ENABLE_STATS
#define USE_AP
#define USE_SX_LOCK
#define BATCH_LOCK_READ
#define FINE_GRAINED_LEAF_NODE
#define FINE_GRAINED_INTERNAL_NODE
#define USE_OPTIMISTIC_UPDATE_HIT
#define USE_ADAPTIVE_HOT_BYPASS
// #define USE_BATCH_POLL
// #define ENABLE_STATS
// #define USE_CRC
// #define USE_LOCAL_LOCK
// ======================================================

#if defined(USE_SX_LOCK) && defined(USE_LOCAL_LOCK)
#error "local lock only for normal lock"
#endif

#if defined(USE_ADAPTIVE_HOT_BYPASS) && !defined(USE_OPTIMISTIC_UPDATE_HIT)
#error "adaptive hot bypass requires optimistic update-hit"
#endif

  uint64_t cache_miss[MAX_APP_THREAD][8];
  uint64_t cache_hit[MAX_APP_THREAD][8];
  uint64_t latency[MAX_APP_THREAD][LATENCY_WINDOWS];
  #ifdef ENABLE_STATS
  uint64_t probe_counts = 0;
  uint64_t call_find_counts = 0;
  #endif
  thread_local uint64_t total_cmp = 0;
  thread_local uint64_t total_seek = 0;

  // 统计变量
uint64_t tries_per_lock[MAX_APP_THREAD][5001];  
uint64_t lock_rdma_faa_cnt[MAX_APP_THREAD] = {0};  // 用于拿票/退票的 Faa 次数
uint64_t lock_rdma_read_cnt[MAX_APP_THREAD] = {0}; // 用于自旋等锁的 Read 次数     // 单次拿锁最大尝试次数
uint64_t lock_retry_data_reread_cnt[MAX_APP_THREAD] = {0};
uint64_t lock_retry_data_reread_bytes[MAX_APP_THREAD] = {0};

  // ================= 微基准测试开关 =================
  // #define ENABLE_MICROBENCH 1
  constexpr uint64_t MICROBENCH_MEM_LIMIT = 62ull * define::GB; // 在服务端的 1GB 范围内随机读
  // ==================================================


  // ---------------- 操作级 RDMA 统计 ----------------
  // 0: 未追踪 (如后台任务), 1: 追踪 Insert, 2: 追踪 Search
  thread_local int tracking_mode = 0; 

  // Insert 统计
  uint64_t insert_rtt_cnt[MAX_APP_THREAD][8] = {0};  
  uint64_t insert_byte_cnt[MAX_APP_THREAD][8] = {0}; 
  uint64_t insert_op_cnt[MAX_APP_THREAD][8] = {0};   

  // Search 统计
  uint64_t search_rtt_cnt[MAX_APP_THREAD][8] = {0};  
  uint64_t search_byte_cnt[MAX_APP_THREAD][8] = {0}; 
  uint64_t search_op_cnt[MAX_APP_THREAD][8] = {0};   

inline void track_rdma(uint16_t tid, uint64_t rtt, uint64_t bytes) {
  if (tracking_mode == 1) {
    insert_rtt_cnt[tid][0] += rtt;
    insert_byte_cnt[tid][0] += bytes;
  } else if (tracking_mode == 2) {
      search_rtt_cnt[tid][0] += rtt;
      search_byte_cnt[tid][0] += bytes;
  }
}

inline void record_lock_retry(uint16_t tid, uint64_t retry_cnt) {
  uint64_t bucket = retry_cnt + 1;
  if (bucket > 5000) {
    bucket = 5000;
  }
  tries_per_lock[tid][bucket]++;
}
// ----------------------------------------------------

  StatHelper stat_helper;

  thread_local CoroCall Tree::worker[define::kMaxCoro];
  thread_local CoroCall Tree::master;
  thread_local uint64_t Tree::coro_ops_total;
  thread_local uint64_t Tree::coro_ops_cnt_start;
  thread_local uint64_t Tree::coro_ops_cnt_finish;
  thread_local GlobalAddress path_stack[define::kMaxCoro]
                                      [define::kMaxLevelOfTree];

  constexpr uint64_t XS_LOCK_FAA_MASK = 0x8000800080008000ul;
  // high->low
  // X_CUR X_TIC S_CUR S_TIC
  // lock: increase TIC, unlock: increase CUR

  constexpr uint64_t ADD_S_LOCK = 1;
  constexpr uint64_t ADD_S_UNLOCK = 1ul << 16;
  constexpr uint64_t ADD_X_LOCK = 1ul << 32;
  constexpr uint64_t ADD_X_UNLOCK = 1ul << 48;
constexpr size_t kHotLeafBypassTableSize = 4096;
constexpr uint8_t kHotLeafBypassThreshold = 2;
constexpr uint8_t kHotLeafBypassBaseCooldown = 4;
constexpr uint8_t kHotLeafBypassMaxCooldown = 16;

struct HotLeafBypassEntry {
  uint64_t page_sig = 0;
  uint8_t cooldown = 0;
  uint8_t fail_count = 0;
};

  #ifdef USE_LOCAL_LOCK
  thread_local std::queue<uint16_t> hot_wait_queue;
  #endif
#ifdef USE_ADAPTIVE_HOT_BYPASS
thread_local HotLeafBypassEntry hot_leaf_bypass_table[kHotLeafBypassTableSize];
#endif
  std::atomic<bool> g_prefill_split_stats_enabled{false};

  inline uint64_t get_leaf_page_sig(GlobalAddress page_addr) {
    return (static_cast<uint64_t>(page_addr.nodeID) << 48) | page_addr.offset;
  }

inline uint64_t count_entries(const LeafEntry *entries, int cnt) {
  uint64_t occupied = 0;
  for (int i = 0; i < cnt; ++i) {
    if (entries[i].lv.val != kValueNull) {
      ++occupied;
    }
  }
  return occupied;
}

inline uint64_t count_bucket_occupancy(const LeafEntryGroup *group,
                                       bool is_front) {
  return count_entries(is_front ? group->front : group->back, kAssociativity);
}

inline uint64_t count_group_occupancy(const LeafEntryGroup *group) {
  return count_entries(group->front, kAssociativity) +
         count_entries(group->back, kAssociativity) +
         count_entries(group->overflow, kGroupOverflowSlots);
}

#ifdef USE_LEAF_STASH
constexpr uint64_t kLeafStashSplitLoadPercent = 85;

inline uint8_t leaf_stash_group_bit(int group_id) {
  return static_cast<uint8_t>(1u << group_id);
}

inline bool leaf_stash_may_have(const LeafPage *page, int group_id) {
  return (page->stash_group_mask() & leaf_stash_group_bit(group_id)) != 0;
}

inline uint64_t count_leaf_stash_occupancy(const LeafPage *page) {
  uint64_t occupied = 0;
  for (int i = 0; i < kLeafStashSlots; ++i) {
    if (page->stash_entry(i)->lv.val != kValueNull) {
      ++occupied;
    }
  }
  return occupied;
}

inline uint64_t count_leaf_page_occupancy(const LeafPage *page) {
  uint64_t occupied = count_leaf_stash_occupancy(page);
  for (int i = 0; i < kNumGroup; ++i) {
    occupied += count_group_occupancy(page->group_at(i));
  }
  return occupied;
}

inline LeafEntry *find_leaf_stash_entry(LeafPage *page, const Key &k,
                                        int group_id) {
  if (!leaf_stash_may_have(page, group_id)) {
    return nullptr;
  }
  for (int i = 0; i < kLeafStashSlots; ++i) {
    LeafEntry *entry = page->stash_entry(i);
    if (entry->lv.val != kValueNull && entry->key == k) {
      return entry;
    }
  }
  return nullptr;
}

inline const LeafEntry *find_leaf_stash_entry(const LeafPage *page,
                                              const Key &k, int group_id) {
  return find_leaf_stash_entry(const_cast<LeafPage *>(page), k, group_id);
}

inline LeafEntry *find_empty_leaf_stash_entry(LeafPage *page) {
  for (int i = 0; i < kLeafStashSlots; ++i) {
    if (page->stash_entry(i)->lv.val == kValueNull) {
      return page->stash_entry(i);
    }
  }
  return nullptr;
}

inline void refresh_leaf_stash_metadata(LeafPage *page) {
  uint8_t mask = 0;
  uint8_t count = 0;
  for (int i = 0; i < kLeafStashSlots; ++i) {
    LeafEntry *entry = page->stash_entry(i);
    if (entry->lv.val != kValueNull) {
      mask |= leaf_stash_group_bit(key_hash_bucket(entry->key) / 2);
      ++count;
    }
  }
  page->set_stash_group_mask(mask);
  page->set_stash_count(count);
}

inline bool insert_leaf_stash_entry(LeafPage *page, const Key &k,
                                    const Value &v, int group_id) {
  LeafEntry *entry = find_empty_leaf_stash_entry(page);
  if (entry == nullptr) {
    return false;
  }
  entry->key = k;
  entry->lv.cl_ver = page->version();
  entry->lv.val = v;
  page->set_stash_group_mask(page->stash_group_mask() |
                             leaf_stash_group_bit(group_id));
  page->set_stash_count(page->stash_count() + 1);
  return true;
}
#endif

inline bool insert_leaf_entry_for_rebuild(LeafPage *page, const Key &k,
                                          const Value &v) {
  int bucket_id = key_hash_bucket(k);
  if (page->group_at(bucket_id / 2)
          ->insert_for_split(k, v, !(bucket_id % 2))) {
    return true;
  }
#ifdef USE_LEAF_STASH
  return insert_leaf_stash_entry(page, k, v, bucket_id / 2);
#else
  return false;
#endif
}

#ifdef USE_LEAF_STASH
static_assert(LeafPage::stash_group_mask_page_offset() % sizeof(uint64_t) == 0,
              "leaf stash mask word must be 8-byte aligned");

inline bool set_leaf_stash_mask_bit_remote(DSMClient *dsm_client,
                                           GlobalAddress page_addr,
                                           int group_id,
                                           uint64_t *cas_buffer,
                                           CoroContext *ctx, uint16_t tid) {
  GlobalAddress mask_addr =
      GADD(page_addr, LeafPage::stash_group_mask_page_offset());

  dsm_client->ReadSync(reinterpret_cast<char *>(cas_buffer), mask_addr,
                       sizeof(uint64_t), ctx);
  track_rdma(tid, 1, sizeof(uint64_t));

  for (int retry = 0; retry < 8; ++retry) {
    uint64_t expected = *cas_buffer;
    uint64_t desired = expected | leaf_stash_group_bit(group_id);
    if (desired == expected) {
      return true;
    }

    bool cas_ok = dsm_client->CasSync(mask_addr, expected, desired, cas_buffer,
                                      ctx);
    track_rdma(tid, 1, sizeof(uint64_t));
    if (cas_ok) {
      return true;
    }
  }
  return false;
}

inline bool try_optimistic_leaf_stash_insert(DSMClient *dsm_client,
                                             GlobalAddress page_addr,
                                             LeafPage *page, char *page_buffer,
                                             const Key &k, const Value &v,
                                             int group_id, RdmaBuffer &rbuf,
                                             CoroContext *ctx, uint16_t tid) {
  LeafEntry *stash_entry = find_empty_leaf_stash_entry(page);
  if (stash_entry == nullptr) {
    return false;
  }

  leaf_stash_insert_attempt_cnt[tid]++;

  uint64_t *mask_cas_buffer = rbuf.get_cas_buffer();
  if (!set_leaf_stash_mask_bit_remote(dsm_client, page_addr, group_id,
                                      mask_cas_buffer, ctx, tid)) {
    return false;
  }

  for (int i = 0; i < kLeafStashSlots; ++i) {
    stash_entry = page->stash_entry(i);
    if (stash_entry->lv.val != kValueNull) {
      continue;
    }

    uint64_t *swap_buffer = rbuf.get_cas_buffer();
    LeafEntry *swap_entry = reinterpret_cast<LeafEntry *>(swap_buffer);
    swap_entry->key = k;
    swap_entry->lv.cl_ver = page->version();
    swap_entry->lv.val = v;

    uint64_t *cas_ret_buffer = rbuf.get_cas_buffer();
    uint64_t *mask_buffer = rbuf.get_cas_buffer();
    mask_buffer[0] = mask_buffer[1] = ~0ull;

    bool cas_ok = dsm_client->CasMaskSync(
        GADD(page_addr, reinterpret_cast<char *>(stash_entry) - page_buffer), 4,
        reinterpret_cast<uint64_t>(stash_entry),
        reinterpret_cast<uint64_t>(swap_buffer), cas_ret_buffer,
        reinterpret_cast<uint64_t>(mask_buffer), ctx);
    track_rdma(tid, 1, sizeof(LeafEntry));
    if (cas_ok || (__bswap_64(cas_ret_buffer[0]) == k)) {
      leaf_stash_insert_success_cnt[tid]++;
      leaf_insert_empty_cnt[tid]++;
      return true;
    }

    stash_entry->key = __bswap_64(cas_ret_buffer[0]);
    stash_entry->lv.raw = __bswap_64(cas_ret_buffer[1]);
  }

  return false;
}
#endif

inline size_t get_hot_leaf_bypass_slot(uint64_t page_sig) {
  uint64_t mixed = page_sig;
  mixed ^= mixed >> 33;
  mixed *= 0xff51afd7ed558ccdULL;
  mixed ^= mixed >> 33;
  return mixed & (kHotLeafBypassTableSize - 1);
}

inline bool should_bypass_optimistic_leaf_fast_path(GlobalAddress page_addr) {
#ifdef USE_ADAPTIVE_HOT_BYPASS
  uint64_t page_sig = get_leaf_page_sig(page_addr);
  auto &entry = hot_leaf_bypass_table[get_hot_leaf_bypass_slot(page_sig)];
  if (entry.page_sig != page_sig || entry.cooldown == 0) {
    return false;
  }
    --entry.cooldown;
    if (entry.cooldown == 0) {
      entry.page_sig = 0;
    entry.fail_count = 0;
  }
  return true;
#else
  return false;
#endif
}

inline void mark_hot_leaf_fast_path_conflict(GlobalAddress page_addr) {
#ifdef USE_ADAPTIVE_HOT_BYPASS
  uint64_t page_sig = get_leaf_page_sig(page_addr);
  auto &entry = hot_leaf_bypass_table[get_hot_leaf_bypass_slot(page_sig)];
  if (entry.page_sig != page_sig) {
    entry.page_sig = page_sig;
    entry.cooldown = 0;
      entry.fail_count = 0;
    }
    if (entry.fail_count < kHotLeafBypassMaxCooldown) {
      ++entry.fail_count;
    }
    if (entry.fail_count >= kHotLeafBypassThreshold) {
      uint8_t level = entry.fail_count - kHotLeafBypassThreshold;
      uint8_t cooldown = static_cast<uint8_t>(kHotLeafBypassBaseCooldown << level);
      if (cooldown > kHotLeafBypassMaxCooldown) {
        cooldown = kHotLeafBypassMaxCooldown;
      }
      entry.cooldown = cooldown;
  } else {
    entry.cooldown = 0;
  }
#endif
}

inline void clear_hot_leaf_fast_path_conflict(GlobalAddress page_addr) {
#ifdef USE_ADAPTIVE_HOT_BYPASS
  uint64_t page_sig = get_leaf_page_sig(page_addr);
  auto &entry = hot_leaf_bypass_table[get_hot_leaf_bypass_slot(page_sig)];
  entry.page_sig = page_sig;
  entry.cooldown = 0;
  entry.fail_count = 0;
#endif
}

  Tree::Tree(DSMClient *dsm_client, uint16_t tree_id)
      : dsm_client_(dsm_client), tree_id(tree_id) {
    // 分配锁
    for (int i = 0; i < dsm_client_->get_server_size(); ++i) {
      local_locks[i] = new LocalLockNode[define::kNumOfLock];
      for (size_t k = 0; k < define::kNumOfLock; ++k) {
        auto &n = local_locks[i][k];
        n.ticket_lock.store(0);
        n.hand_over = false;
        n.hand_time = 0;
      }
    }

    assert(dsm_client_->IsRegistered());
    print_verbose();

    index_cache = new IndexCache(define::kIndexCacheSize);

    root_ptr_ptr = get_root_ptr_ptr();

    // try to init tree and install root pointer
    char *page_buffer = (dsm_client_->get_rbuf(0)).get_page_buffer();
    GlobalAddress root_addr = dsm_client_->Alloc(kLeafPageSize);
    [[maybe_unused]] LeafPage *root_page = new (page_buffer) LeafPage;

    dsm_client_->WriteSync(page_buffer, root_addr, kLeafPageSize);

    uint64_t *cas_buffer = (dsm_client_->get_rbuf(0)).get_cas_buffer();
    bool res = dsm_client_->CasSync(root_ptr_ptr, 0, root_addr.raw, cas_buffer);
    if (res) {
      std::cout << "Tree root pointer value " << root_addr << std::endl;
    } else {
      // std::cout << "fail\n";
    }
  }

  Tree::~Tree() { delete index_cache; }

  void Tree::print_verbose() {
    constexpr int kLeafHdrOffset = offsetof(LeafPage, hdr);
    constexpr int kInternalHdrOffset = offsetof(InternalPage, hdr);
    static_assert(kLeafHdrOffset == kInternalHdrOffset, "format error");
    // if (kLeafHdrOffset != kInternalHdrOffset) {
    //   std::cerr << "format error" << std::endl;
    // }

    if (dsm_client_->get_my_client_id() == 0) {
      std::cout << "Header size: " << sizeof(Header) << std::endl;
      std::cout << "Internal Page size: " << sizeof(InternalPage) << " ["
                << kInternalPageSize << "]" << std::endl;
      std::cout << "Internal per Page: " << kInternalCardinality << std::endl;
      std::cout << "Leaf Page size: " << sizeof(LeafPage) << " [" << kLeafPageSize
                << "]" << std::endl;
      std::cout << "Leaf per Page: " << kLeafCardinality << std::endl;
      std::cout << "LeafEntry size: " << sizeof(LeafEntry) << std::endl;
      std::cout << "InternalEntry size: " << sizeof(InternalEntry) << std::endl;
      static_assert(sizeof(InternalPage) <= kInternalPageSize);
      static_assert(sizeof(LeafPage) <= kLeafPageSize);
    }
  }

  inline void Tree::before_operation(CoroContext *ctx, int coro_id) {
    for (size_t i = 0; i < define::kMaxLevelOfTree; ++i) {
      path_stack[coro_id][i] = GlobalAddress::Null();
    }
  }

  GlobalAddress Tree::get_root_ptr_ptr() {
    GlobalAddress addr;
    addr.node_version = 0;
    addr.nodeID = 0;
    addr.offset =
        define::kRootPointerStoreOffest + sizeof(GlobalAddress) * tree_id;

    return addr;
  }

  extern GlobalAddress g_root_ptr;
  extern int g_root_level;
  extern bool enable_cache;
  GlobalAddress Tree::get_root_ptr(CoroContext *ctx, bool force_read) {
    if (force_read || g_root_ptr == GlobalAddress::Null()) {
      char *page_buffer =
          (dsm_client_->get_rbuf(ctx ? ctx->coro_id : 0)).get_page_buffer();
      dsm_client_->ReadSync(page_buffer, root_ptr_ptr, sizeof(GlobalAddress),
                            ctx);
      GlobalAddress root_ptr = *(GlobalAddress *)page_buffer;
      g_root_ptr = root_ptr;
      return root_ptr;
    } else {
      return g_root_ptr;
    }

    // std::cout << "root ptr " << root_ptr << std::endl;
  }

  // void Tree::broadcast_new_root(GlobalAddress new_root_addr, int root_level) {
  //   RawMessage m;
  //   m.type = RpcType::NEW_ROOT;
  //   m.addr = new_root_addr;
  //   m.level = root_level;
  //   for (int i = 0; i < dsm_client_->get_server_size(); ++i) {
  //     dsm_client_->RpcCallDir(m, i);
  //   }
  //   g_root_ptr = new_root_addr;
  //   g_root_level = root_level;
  //   if (root_level >= 3) {
  //     enable_cache = true;
  //   }
  // }

  bool Tree::update_new_root(GlobalAddress left, const Key &k,
                            GlobalAddress right, int level,
                            GlobalAddress old_root, CoroContext *ctx) {
    auto &rbuf = dsm_client_->get_rbuf(ctx ? ctx->coro_id : 0);
    char *page_buffer = rbuf.get_page_buffer();
    uint64_t *cas_buffer = rbuf.get_cas_buffer();
    InternalPage *new_root =
        new (page_buffer) InternalPage(left, k, right, level);
    new_root->hdr.is_root = true;

    GlobalAddress new_root_addr = dsm_client_->Alloc(kInternalPageSize);

    new_root->hdr.my_addr = new_root_addr;
    dsm_client_->WriteSync(page_buffer, new_root_addr, kInternalPageSize, ctx);
    if (dsm_client_->CasSync(root_ptr_ptr, old_root, new_root_addr, cas_buffer,
                            ctx)) {
      // broadcast_new_root(new_root_addr, level);
      printf("new root level %d [%d, %ld]\n", level, new_root_addr.nodeID,
            new_root_addr.offset);
      g_root_ptr = new_root_addr;
      return true;
    } else {
      printf(
          "cas root fail: left [%d,%lu] right [%d,%lu] old root [%d,%lu] try new "
          "root [%d,%lu]\n",
          left.nodeID, left.offset, right.nodeID, right.offset, old_root.nodeID,
          old_root.offset, new_root_addr.nodeID, new_root_addr.offset);
    }

    return false;
  }
  inline bool Tree::try_lock_addr(GlobalAddress lock_addr, uint64_t *buf,
                                  CoroContext *ctx) {

    u_int16_t my_thread_id = dsm_client_->get_my_thread_id();     
  #ifdef USE_LOCAL_LOCK
    bool hand_over = acquire_local_lock(lock_addr, ctx, ctx ? ctx->coro_id : 0);
    if (hand_over) {
      return true;
    }
  #endif

    {
      uint64_t retry_cnt = 0;
      uint64_t pre_tag = 0;
      uint64_t conflict_tag = 0;
    retry:
      retry_cnt++;
      if (retry_cnt > 5000000) {
        // 记录这次失败的尝试
        tries_per_lock[my_thread_id][5000]++;

        std::cout << "Deadlock " << lock_addr << std::endl;

        std::cout << dsm_client_->get_my_client_id() << ", "
                  << dsm_client_->get_my_thread_id() << " locked by "
                  << (conflict_tag >> 32) << ", " << (conflict_tag << 32 >> 32)
                  << std::endl;
        assert(false);
        exit(-1);
      }
      auto tag = dsm_client_->get_thread_tag();
      bool res = dsm_client_->CasDmSync(lock_addr, 0, tag, buf, ctx);

      if (!res) {
        // conflict_tag = *buf - 1;
        conflict_tag = *buf;
        if (conflict_tag != pre_tag) {
          retry_cnt = 0;
          pre_tag = conflict_tag;
        }
        goto retry;
      }
    record_lock_retry(my_thread_id, retry_cnt);

    }
    // 成功获取共享锁
    return true;
  }

  void Tree::print_lock_stats() {
      uint64_t global_hist[5001] = {0}; // 初始化为0

      for (int t = 0; t < MAX_APP_THREAD; ++t) {
          for (int k = 1; k <= 5000; ++k) { // 跳过 k=0（除非你明确使用）
              global_hist[k] += tries_per_lock[t][k];
          }
      }

      uint64_t total_events = 0;
      for (int k = 1; k <= 5000; ++k) {
          total_events += global_hist[k];
      }
      if (total_events == 0) {
          // 无数据，无法计算
          return;
      }


      double targets[] = {0.25, 0.50, 0.75, 0.99};
      int result[4] = {0};
      int target_idx = 0;

      uint64_t cumsum = 0;
      for (int k = 1; k <= 5000; ++k) {
          cumsum += global_hist[k];
          double ratio = static_cast<double>(cumsum) / total_events;

          // 满足当前及之前所有未满足的分位点
          while (target_idx < 4 && ratio >= targets[target_idx]) {
              result[target_idx] = k;
              target_idx++;
          }
          if (target_idx >= 4) break; // 所有分位已找到
      }

      // 处理极端情况：如果 99% 仍未达到（长尾超出 5000）
      if (target_idx < 4) {
          // 说明即使 k=5000，累积比例仍 < 0.99
          // 可设为 >5000，或报 warning
          for (int i = target_idx; i < 4; ++i) {
              result[i] = 5001; // 表示 "超过 5000 次"
          }
      }
      printf("Lock retry count percentiles:\n");
      printf("  P25: %d\n", result[0]);
      printf("  P50: %d\n", result[1]);
      printf("  P75: %d\n", result[2]);
      printf("  P99: %d\n", result[3]);
      // 导出直方图为 CSV
      FILE* csv = fopen("lock_retry_histogram.csv", "w");
      if (csv) {
          fprintf(csv, "retry_count,frequency\n"); // 表头
          for (int k = 1; k <= 5000; ++k) {
              if (global_hist[k] > 0) { // 只输出非零项（可选）
                  fprintf(csv, "%d,%llu\n", k, (unsigned long long)global_hist[k]);
              }
          }
          fclose(csv);
          printf("Histogram exported to lock_retry_histogram.csv\n");
      }
      #ifdef ENABLE_STATS
      double avg_probes = (double)probe_counts / (double)call_find_counts;
      printf("Total probes: %llu\n", (unsigned long long)probe_counts);
      printf("Total call find: %llu\n", (unsigned long long)call_find_counts);
      printf("Average probes per search: %.3f\n", avg_probes);
      #endif
      double avg = double(total_cmp) / total_seek;
      printf("Average compares per seek: %.3f\n", avg);
      // 【新增】聚合全网锁 RDMA 统计
      uint64_t total_lock_faa = 0;
      uint64_t total_lock_read = 0;
      uint64_t total_lock_retry_data_reread = 0;
      uint64_t total_lock_retry_data_reread_bytes = 0;
      for (int t = 0; t < MAX_APP_THREAD; ++t) {
          total_lock_faa += lock_rdma_faa_cnt[t];
          total_lock_read += lock_rdma_read_cnt[t];
          total_lock_retry_data_reread += lock_retry_data_reread_cnt[t];
          total_lock_retry_data_reread_bytes += lock_retry_data_reread_bytes[t];
      }
      uint64_t total_lock_rdma = total_lock_faa + total_lock_read;

      printf("\n=== Lock RDMA Overhead Stats ===\n");
      printf("  Total Lock FAA ops (Acquire/Release): %llu\n", (unsigned long long)total_lock_faa);
      printf("  Total Lock Spin READ ops (Retries):   %llu\n", (unsigned long long)total_lock_read);
      printf("  Lock retry data rereads:              %llu\n", (unsigned long long)total_lock_retry_data_reread);
      printf("  Lock retry reread bytes:              %llu\n", (unsigned long long)total_lock_retry_data_reread_bytes);
      printf("  Total RDMA ops wasted on Locks:       %llu\n", (unsigned long long)total_lock_rdma);
      
      // 如果你有统计总的 operations 或者有用的 RDMA 次数，可以在这里算个占比
      // 比如：double overhead_ratio = (double)total_lock_rdma / (total_lock_rdma + valid_rdma_ops);
      printf("================================\n");
    uint64_t total_insert_rtt = 0, total_insert_bytes = 0, total_insert_ops = 0;
    uint64_t total_search_rtt = 0, total_search_bytes = 0, total_search_ops = 0;
    uint64_t total_leaf_update_hit = 0, total_leaf_insert_empty = 0;
    uint64_t total_leaf_insert_group_fast = 0;
    uint64_t total_leaf_insert_page_lock = 0;
    uint64_t total_leaf_insert_retry_event = 0;
    uint64_t total_leaf_insert_retry_step = 0;
    uint64_t total_leaf_upgrade_to_x = 0, total_leaf_split = 0;
    uint64_t total_leaf_split_occupancy = 0;
    uint64_t max_leaf_split_occupancy = 0;
    uint64_t total_leaf_sibling_chase = 0;
    uint64_t total_leaf_insert_parent_update = 0;
    uint64_t total_leaf_insert_root_split = 0;
    uint64_t total_leaf_stash_insert_attempt = 0;
    uint64_t total_leaf_stash_insert_success = 0;
    uint64_t total_leaf_stash_insert_full = 0;
    uint64_t total_leaf_stash_insert_threshold = 0;
    uint64_t total_leaf_split_stash_occupancy = 0;
    uint64_t max_leaf_split_stash_occupancy = 0;
    uint64_t total_optimistic_update_attempt = 0;
    uint64_t total_optimistic_update_success = 0;
    uint64_t total_optimistic_update_cas_fail = 0;
    uint64_t total_optimistic_update_split_abort = 0;
    uint64_t total_optimistic_leaf_fast_path_hot_bypass = 0;
    uint64_t total_optimistic_insert_attempt = 0;
    uint64_t total_optimistic_insert_success = 0;
    uint64_t total_optimistic_insert_cas_fail = 0;
    uint64_t total_optimistic_insert_split_abort = 0;
    uint64_t total_optimistic_insert_consistency_fail = 0;
    uint64_t total_optimistic_insert_fallback = 0;

    for (int t = 0; t < MAX_APP_THREAD; ++t) {
      total_insert_rtt += insert_rtt_cnt[t][0];
      total_insert_bytes += insert_byte_cnt[t][0];
      total_insert_ops += insert_op_cnt[t][0];

      total_search_rtt += search_rtt_cnt[t][0];
      total_search_bytes += search_byte_cnt[t][0];
      total_search_ops += search_op_cnt[t][0];
      total_leaf_update_hit += leaf_update_hit_cnt[t];
      total_leaf_insert_empty += leaf_insert_empty_cnt[t];
      total_leaf_insert_group_fast += leaf_insert_path_group_fast_cnt[t];
      total_leaf_insert_page_lock += leaf_insert_path_page_lock_cnt[t];
      total_leaf_insert_retry_event += leaf_insert_retry_event_cnt[t];
      total_leaf_insert_retry_step += leaf_insert_retry_step_cnt[t];
      total_leaf_upgrade_to_x += leaf_upgrade_to_x_cnt[t];
      total_leaf_split += leaf_split_cnt[t];
      total_leaf_split_occupancy += leaf_split_occupancy_sum[t];
      if (leaf_split_occupancy_max[t] > max_leaf_split_occupancy) {
        max_leaf_split_occupancy = leaf_split_occupancy_max[t];
      }
      total_leaf_sibling_chase += leaf_sibling_chase_cnt[t];
      total_leaf_insert_parent_update += leaf_insert_parent_update_cnt[t];
      total_leaf_insert_root_split += leaf_insert_root_split_cnt[t];
      total_leaf_stash_insert_attempt += leaf_stash_insert_attempt_cnt[t];
      total_leaf_stash_insert_success += leaf_stash_insert_success_cnt[t];
      total_leaf_stash_insert_full += leaf_stash_insert_full_cnt[t];
      total_leaf_stash_insert_threshold += leaf_stash_insert_threshold_cnt[t];
      total_leaf_split_stash_occupancy += leaf_split_stash_occupancy_sum[t];
      if (leaf_split_stash_occupancy_max[t] >
          max_leaf_split_stash_occupancy) {
        max_leaf_split_stash_occupancy = leaf_split_stash_occupancy_max[t];
      }
      total_optimistic_update_attempt += optimistic_update_attempt_cnt[t];
      total_optimistic_update_success += optimistic_update_success_cnt[t];
      total_optimistic_update_cas_fail += optimistic_update_cas_fail_cnt[t];
      total_optimistic_update_split_abort += optimistic_update_split_abort_cnt[t];
      total_optimistic_leaf_fast_path_hot_bypass +=
          optimistic_leaf_fast_path_hot_bypass_cnt[t];
      total_optimistic_insert_attempt += optimistic_insert_attempt_cnt[t];
      total_optimistic_insert_success += optimistic_insert_success_cnt[t];
      total_optimistic_insert_cas_fail += optimistic_insert_cas_fail_cnt[t];
      total_optimistic_insert_split_abort += optimistic_insert_split_abort_cnt[t];
      total_optimistic_insert_consistency_fail +=
          optimistic_insert_consistency_fail_cnt[t];
      total_optimistic_insert_fallback += optimistic_insert_fallback_cnt[t];
    }

    printf("\n=== Operation RDMA Profiling ===\n");
    if (total_insert_ops > 0) {
      double avg_ins_rtt = (double)total_insert_rtt / total_insert_ops;
      double avg_ins_bytes = (double)total_insert_bytes / total_insert_ops;
      printf("[INSERT] Total Ops: %llu | Avg RTT: %.2f | Avg Data: %.2f Bytes\n", 
            (unsigned long long)total_insert_ops, avg_ins_rtt, avg_ins_bytes);
      printf("[INSERT] Lock retry rereads/op: %.4f | Reread bytes/op: %.2f\n",
            (double)total_lock_retry_data_reread / total_insert_ops,
            (double)total_lock_retry_data_reread_bytes / total_insert_ops);
    }

    if (total_search_ops > 0) {
      double avg_sch_rtt = (double)total_search_rtt / total_search_ops;
      double avg_sch_bytes = (double)total_search_bytes / total_search_ops;
      printf("[SEARCH] Total Ops: %llu | Avg RTT: %.2f | Avg Data: %.2f Bytes\n", 
            (unsigned long long)total_search_ops, avg_sch_rtt, avg_sch_bytes);
      printf("[SEARCH] Lock retry rereads/op: %.4f | Reread bytes/op: %.2f\n",
            (double)total_lock_retry_data_reread / total_search_ops,
            (double)total_lock_retry_data_reread_bytes / total_search_ops);
    }
    if (total_insert_ops > 0) {
      printf("\n=== Leaf Insert Path Breakdown ===\n");
      printf("  Update hits:          %llu (%.4f / insert)\n",
            (unsigned long long)total_leaf_update_hit,
            (double)total_leaf_update_hit / total_insert_ops);
      printf("  Empty-slot inserts:   %llu (%.4f / insert)\n",
            (unsigned long long)total_leaf_insert_empty,
            (double)total_leaf_insert_empty / total_insert_ops);
      printf("  Group-fast inserts:   %llu (%.4f / insert)\n",
            (unsigned long long)total_leaf_insert_group_fast,
            (double)total_leaf_insert_group_fast / total_insert_ops);
      printf("  Page-lock inserts:    %llu (%.4f / insert)\n",
            (unsigned long long)total_leaf_insert_page_lock,
            (double)total_leaf_insert_page_lock / total_insert_ops);
      printf("  Retry events:         %llu (%.4f / insert)\n",
            (unsigned long long)total_leaf_insert_retry_event,
            (double)total_leaf_insert_retry_event / total_insert_ops);
      printf("  Retry steps:          %llu (%.4f / insert)\n",
            (unsigned long long)total_leaf_insert_retry_step,
            (double)total_leaf_insert_retry_step / total_insert_ops);
      printf("  S->X upgrades:        %llu (%.4f / insert)\n",
            (unsigned long long)total_leaf_upgrade_to_x,
            (double)total_leaf_upgrade_to_x / total_insert_ops);
      printf("  Leaf splits:          %llu (%.4f / insert)\n",
            (unsigned long long)total_leaf_split,
            (double)total_leaf_split / total_insert_ops);
      if (total_leaf_split > 0) {
        double avg_split_occupancy =
            (double)total_leaf_split_occupancy / total_leaf_split;
        double avg_split_load_factor =
            avg_split_occupancy / (double)kLeafCardinality;
        double max_split_load_factor =
            (double)max_leaf_split_occupancy / (double)kLeafCardinality;
        printf("  Split avg occupancy:  %.2f / %d (load %.2f%%)\n",
              avg_split_occupancy, kLeafCardinality,
              avg_split_load_factor * 100.0);
        printf("  Split max occupancy:  %llu / %d (load %.2f%%)\n",
              (unsigned long long)max_leaf_split_occupancy, kLeafCardinality,
              max_split_load_factor * 100.0);
      }
      printf("  Sibling chases:       %llu (%.4f / insert)\n",
            (unsigned long long)total_leaf_sibling_chase,
            (double)total_leaf_sibling_chase / total_insert_ops);
      printf("  Parent updates:       %llu (%.4f / insert)\n",
            (unsigned long long)total_leaf_insert_parent_update,
            (double)total_leaf_insert_parent_update / total_insert_ops);
      printf("  Root splits:          %llu (%.4f / insert)\n",
            (unsigned long long)total_leaf_insert_root_split,
            (double)total_leaf_insert_root_split / total_insert_ops);
#ifdef USE_LEAF_STASH
      printf("\n=== Leaf Conflict Stash Breakdown ===\n");
      printf("  Stash attempts:       %llu (%.4f / insert)\n",
            (unsigned long long)total_leaf_stash_insert_attempt,
            (double)total_leaf_stash_insert_attempt / total_insert_ops);
      printf("  Stash success:        %llu (%.4f / insert)\n",
            (unsigned long long)total_leaf_stash_insert_success,
            (double)total_leaf_stash_insert_success / total_insert_ops);
      printf("  Stash full fallbacks: %llu (%.4f / insert)\n",
            (unsigned long long)total_leaf_stash_insert_full,
            (double)total_leaf_stash_insert_full / total_insert_ops);
      printf("  Stash threshold skip: %llu (%.4f / insert)\n",
            (unsigned long long)total_leaf_stash_insert_threshold,
            (double)total_leaf_stash_insert_threshold / total_insert_ops);
      if (total_leaf_split > 0) {
        printf("  Split stash avg occ:  %.2f / %d\n",
              (double)total_leaf_split_stash_occupancy / total_leaf_split,
              kLeafStashSlots);
        printf("  Split stash max occ:  %llu / %d\n",
              (unsigned long long)max_leaf_split_stash_occupancy,
              kLeafStashSlots);
      }
#endif
      printf("\n=== Optimistic Update Breakdown ===\n");
      printf("  Attempts:             %llu (%.4f / insert)\n",
            (unsigned long long)total_optimistic_update_attempt,
            (double)total_optimistic_update_attempt / total_insert_ops);
      printf("  Success:              %llu (%.4f / insert)\n",
            (unsigned long long)total_optimistic_update_success,
            (double)total_optimistic_update_success / total_insert_ops);
      printf("  CAS failures:         %llu (%.4f / insert)\n",
            (unsigned long long)total_optimistic_update_cas_fail,
            (double)total_optimistic_update_cas_fail / total_insert_ops);
      printf("  Split aborts:         %llu (%.4f / insert)\n",
            (unsigned long long)total_optimistic_update_split_abort,
            (double)total_optimistic_update_split_abort / total_insert_ops);
      printf("  Fast-path hot bypasses: %llu (%.4f / insert)\n",
            (unsigned long long)total_optimistic_leaf_fast_path_hot_bypass,
            (double)total_optimistic_leaf_fast_path_hot_bypass /
                total_insert_ops);
      printf("\n=== Optimistic Insert Breakdown ===\n");
      printf("  Attempts:             %llu (%.4f / insert)\n",
            (unsigned long long)total_optimistic_insert_attempt,
            (double)total_optimistic_insert_attempt / total_insert_ops);
      printf("  Success:              %llu (%.4f / insert)\n",
            (unsigned long long)total_optimistic_insert_success,
            (double)total_optimistic_insert_success / total_insert_ops);
      printf("  CAS failures:         %llu (%.4f / insert)\n",
            (unsigned long long)total_optimistic_insert_cas_fail,
            (double)total_optimistic_insert_cas_fail / total_insert_ops);
      printf("  Split aborts:         %llu (%.4f / insert)\n",
            (unsigned long long)total_optimistic_insert_split_abort,
            (double)total_optimistic_insert_split_abort / total_insert_ops);
      printf("  Consistency failures: %llu (%.4f / insert)\n",
            (unsigned long long)total_optimistic_insert_consistency_fail,
            (double)total_optimistic_insert_consistency_fail / total_insert_ops);
      printf("  Fallbacks:            %llu (%.4f / insert)\n",
            (unsigned long long)total_optimistic_insert_fallback,
            (double)total_optimistic_insert_fallback / total_insert_ops);
      printf("================================\n");
    }
    printf("================================\n");
  }

  void Tree::set_prefill_split_stats(bool enabled) {
    g_prefill_split_stats_enabled.store(enabled, std::memory_order_relaxed);
  }

  void Tree::print_prefill_split_stats() {
    uint64_t total_split = 0;
    uint64_t total_page_occupancy = 0;
    uint64_t max_page_occupancy = 0;
    uint64_t total_bucket_occupancy = 0;
    uint64_t max_bucket_occupancy = 0;
    uint64_t total_group_occupancy = 0;
    uint64_t max_group_occupancy = 0;
    uint64_t total_leaf_insert_empty = 0;
    uint64_t total_leaf_insert_group_fast = 0;
    uint64_t total_leaf_insert_page_lock = 0;
    uint64_t total_leaf_insert_retry_event = 0;
    uint64_t total_leaf_insert_retry_step = 0;
    uint64_t total_leaf_upgrade_to_x = 0;
    uint64_t total_leaf_split = 0;
    uint64_t total_leaf_stash_insert_attempt = 0;
    uint64_t total_leaf_stash_insert_success = 0;
    uint64_t total_leaf_stash_insert_full = 0;
    uint64_t total_leaf_stash_insert_threshold = 0;
    uint64_t total_optimistic_insert_attempt = 0;
    uint64_t total_optimistic_insert_success = 0;
    uint64_t total_optimistic_insert_cas_fail = 0;
    uint64_t total_optimistic_insert_split_abort = 0;
    uint64_t total_optimistic_insert_consistency_fail = 0;
    uint64_t total_optimistic_insert_fallback = 0;

    for (int t = 0; t < MAX_APP_THREAD; ++t) {
      total_split += prefill_leaf_split_cnt[t];
      total_page_occupancy += prefill_leaf_split_page_occupancy_sum[t];
      total_bucket_occupancy += prefill_leaf_split_bucket_occupancy_sum[t];
      total_group_occupancy += prefill_leaf_split_group_occupancy_sum[t];
      if (prefill_leaf_split_page_occupancy_max[t] > max_page_occupancy) {
        max_page_occupancy = prefill_leaf_split_page_occupancy_max[t];
      }
      if (prefill_leaf_split_bucket_occupancy_max[t] > max_bucket_occupancy) {
        max_bucket_occupancy = prefill_leaf_split_bucket_occupancy_max[t];
      }
      if (prefill_leaf_split_group_occupancy_max[t] > max_group_occupancy) {
        max_group_occupancy = prefill_leaf_split_group_occupancy_max[t];
      }
      total_leaf_insert_empty += leaf_insert_empty_cnt[t];
      total_leaf_insert_group_fast += leaf_insert_path_group_fast_cnt[t];
      total_leaf_insert_page_lock += leaf_insert_path_page_lock_cnt[t];
      total_leaf_insert_retry_event += leaf_insert_retry_event_cnt[t];
      total_leaf_insert_retry_step += leaf_insert_retry_step_cnt[t];
      total_leaf_upgrade_to_x += leaf_upgrade_to_x_cnt[t];
      total_leaf_split += leaf_split_cnt[t];
      total_leaf_stash_insert_attempt += leaf_stash_insert_attempt_cnt[t];
      total_leaf_stash_insert_success += leaf_stash_insert_success_cnt[t];
      total_leaf_stash_insert_full += leaf_stash_insert_full_cnt[t];
      total_leaf_stash_insert_threshold += leaf_stash_insert_threshold_cnt[t];
      total_optimistic_insert_attempt += optimistic_insert_attempt_cnt[t];
      total_optimistic_insert_success += optimistic_insert_success_cnt[t];
      total_optimistic_insert_cas_fail += optimistic_insert_cas_fail_cnt[t];
      total_optimistic_insert_split_abort += optimistic_insert_split_abort_cnt[t];
      total_optimistic_insert_consistency_fail +=
          optimistic_insert_consistency_fail_cnt[t];
      total_optimistic_insert_fallback += optimistic_insert_fallback_cnt[t];
    }

    printf("\n=== Prefill Split Occupancy ===\n");
    if (total_split == 0) {
      printf("  No prefill leaf splits recorded.\n");
    } else {
      printf("  Leaf splits:              %llu\n", (unsigned long long)total_split);
      printf("  Page avg occupancy:       %.2f / %d (load %.2f%%)\n",
             (double)total_page_occupancy / total_split, kLeafCardinality,
             (double)total_page_occupancy * 100.0 /
                 (double)(total_split * kLeafCardinality));
      printf("  Page max occupancy:       %llu / %d (load %.2f%%)\n",
             (unsigned long long)max_page_occupancy, kLeafCardinality,
             (double)max_page_occupancy * 100.0 / (double)kLeafCardinality);
      printf("  Bucket avg occupancy:     %.2f / %d (load %.2f%%)\n",
             (double)total_bucket_occupancy / total_split, kAssociativity,
             (double)total_bucket_occupancy * 100.0 /
                 (double)(total_split * kAssociativity));
      printf("  Bucket max occupancy:     %llu / %d (load %.2f%%)\n",
             (unsigned long long)max_bucket_occupancy, kAssociativity,
             (double)max_bucket_occupancy * 100.0 / (double)kAssociativity);
      printf("  Group avg occupancy:      %.2f / %d (load %.2f%%)\n",
             (double)total_group_occupancy / total_split, kGroupSize,
             (double)total_group_occupancy * 100.0 /
                 (double)(total_split * kGroupSize));
      printf("  Group max occupancy:      %llu / %d (load %.2f%%)\n",
             (unsigned long long)max_group_occupancy, kGroupSize,
             (double)max_group_occupancy * 100.0 / (double)kGroupSize);
    }
    printf("\n=== Prefill Insert Path Breakdown ===\n");
    printf("  Empty-slot inserts:       %llu\n",
           (unsigned long long)total_leaf_insert_empty);
    printf("  Group-fast inserts:       %llu\n",
           (unsigned long long)total_leaf_insert_group_fast);
    printf("  Page-lock inserts:        %llu\n",
           (unsigned long long)total_leaf_insert_page_lock);
    printf("  Retry events:             %llu\n",
           (unsigned long long)total_leaf_insert_retry_event);
    printf("  Retry steps:              %llu\n",
           (unsigned long long)total_leaf_insert_retry_step);
    printf("  S->X upgrades:            %llu\n",
           (unsigned long long)total_leaf_upgrade_to_x);
    printf("  Leaf splits:              %llu\n",
           (unsigned long long)total_leaf_split);
#ifdef USE_LEAF_STASH
    printf("  Stash attempts:           %llu\n",
           (unsigned long long)total_leaf_stash_insert_attempt);
    printf("  Stash success:            %llu\n",
           (unsigned long long)total_leaf_stash_insert_success);
    printf("  Stash full fallbacks:     %llu\n",
           (unsigned long long)total_leaf_stash_insert_full);
    printf("  Stash threshold skip:     %llu\n",
           (unsigned long long)total_leaf_stash_insert_threshold);
#endif
    printf("\n=== Prefill Optimistic Insert Breakdown ===\n");
    printf("  Attempts:                 %llu\n",
           (unsigned long long)total_optimistic_insert_attempt);
    printf("  Success:                  %llu\n",
           (unsigned long long)total_optimistic_insert_success);
    printf("  CAS failures:             %llu\n",
           (unsigned long long)total_optimistic_insert_cas_fail);
    printf("  Split aborts:             %llu\n",
           (unsigned long long)total_optimistic_insert_split_abort);
    printf("  Consistency failures:     %llu\n",
           (unsigned long long)total_optimistic_insert_consistency_fail);
    printf("  Fallbacks:                %llu\n",
           (unsigned long long)total_optimistic_insert_fallback);
    printf("================================\n");
  }





  inline void Tree::unlock_addr(GlobalAddress lock_addr, uint64_t *buf,
                                CoroContext *ctx, bool async) {
  #ifdef USE_LOCAL_LOCK
    bool hand_over_other = can_hand_over(lock_addr);
    if (hand_over_other) {
      releases_local_lock(lock_addr);
      return;
    }
  #endif

    *buf = 0;
    if (async) {
      dsm_client_->WriteDm((char *)buf, lock_addr, sizeof(uint64_t), false);
    } else {
      dsm_client_->WriteDmSync((char *)buf, lock_addr, sizeof(uint64_t), ctx);
    }

  #ifdef USE_LOCAL_LOCK
    releases_local_lock(lock_addr);
  #endif
  }

  inline void Tree::acquire_sx_lock(GlobalAddress lock_addr,
                                    uint64_t *lock_buffer, CoroContext *ctx,
                                    bool share_lock, bool upgrade_from_s) {
    assert(!upgrade_from_s || !share_lock);
    uint64_t add_val;
    if (share_lock) {
      add_val = ADD_S_LOCK;
    } else {
      add_val = upgrade_from_s ? (ADD_X_LOCK | ADD_S_UNLOCK) : ADD_X_LOCK;
    }
    // Timer timer;
    // timer.begin();

    dsm_client_->FaaDmBoundSync(lock_addr, 3, add_val, lock_buffer,
                                XS_LOCK_FAA_MASK, ctx);
    u_int16_t my_thread_id = dsm_client_->get_my_thread_id();  
    track_rdma(my_thread_id, 1, 8);
    lock_rdma_faa_cnt[my_thread_id]++;  // 【新增】记录 1 次加锁 FAA 报文                      
    uint16_t s_tic = *lock_buffer & 0xffff;
    uint16_t s_cnt = (*lock_buffer >> 16) & 0xffff;
    uint16_t x_tic = (*lock_buffer >> 32) & 0xffff;
    uint16_t x_cnt = (*lock_buffer >> 48) & 0xffff;

    if (upgrade_from_s) {
      ++s_cnt;
    }
    uint64_t retry_cnt = 0;
  retry:
    if (share_lock && x_cnt == x_tic) {
      // ok
      // 成功获取共享锁
    record_lock_retry(my_thread_id, retry_cnt);
    } else if (!share_lock && x_cnt == x_tic && s_cnt == s_tic) {
      // ok
    record_lock_retry(my_thread_id, retry_cnt);
    } else {
      ++retry_cnt;
      if (retry_cnt > 5000000) {
        tries_per_lock[my_thread_id][5000]++;
        printf(
            "Deadlock [%u, %lu] my thread %d coro_id %d try %d lock upgrade "
            "%d\n",
            lock_addr.nodeID, lock_addr.offset, dsm_client_->get_my_thread_id(),
            ctx ? ctx->coro_id : 0, share_lock, upgrade_from_s);
        printf("s [%u, %u] x [%u, %u]\n", s_tic, s_cnt, x_tic, x_cnt);
        fflush(stdout);
        assert(false);
        exit(-1);
      }
      dsm_client_->ReadDmSync((char *)lock_buffer, lock_addr, 8, ctx);
      track_rdma(my_thread_id, 1, 8);
      lock_rdma_read_cnt[my_thread_id]++; // 【新增】记录 1 次等锁的 Read 报文
      s_cnt = (*lock_buffer >> 16) & 0xffff;
      x_cnt = (*lock_buffer >> 48) & 0xffff;
      goto retry;
    }
    // uint64_t t = timer.end();
    // stat_helper.add(dsm_client_->get_my_thread_id(), lat_lock, t);
  }

  inline void Tree::release_sx_lock(GlobalAddress lock_addr,
                                    uint64_t *lock_buffer, CoroContext *ctx,
                                    bool async, bool share_lock) {
    uint64_t add_val = share_lock ? ADD_S_UNLOCK : ADD_X_UNLOCK;
    if (async) {
      dsm_client_->FaaDmBound(lock_addr, 3, add_val, lock_buffer,
                              XS_LOCK_FAA_MASK, false);
    } else {
      dsm_client_->FaaDmBoundSync(lock_addr, 3, add_val, lock_buffer,
                                  XS_LOCK_FAA_MASK, ctx);
    }
    track_rdma(dsm_client_->get_my_thread_id(), 1, 8);

    // 【新增】记录 1 次解锁 FAA 报文
    lock_rdma_faa_cnt[dsm_client_->get_my_thread_id()]++;
  }

  inline void Tree::acquire_lock(GlobalAddress lock_addr, uint64_t *lock_buffer,
                                CoroContext *ctx, bool share_lock,
                                bool upgrade_from_s) {
  #ifdef USE_SX_LOCK
    acquire_sx_lock(lock_addr, lock_buffer, ctx, share_lock, upgrade_from_s);
  #else
    try_lock_addr(lock_addr, lock_buffer, ctx);
  #endif
  }

  inline void Tree::release_lock(GlobalAddress lock_addr, uint64_t *lock_buffer,
                                CoroContext *ctx, bool async, bool share_lock) {
  #ifdef USE_SX_LOCK
    uint64_t add_val = share_lock ? ADD_S_UNLOCK : ADD_X_UNLOCK;
    if (async) {
      dsm_client_->FaaDmBound(lock_addr, 3, add_val, lock_buffer,
                              XS_LOCK_FAA_MASK, false);
    } else {
      dsm_client_->FaaDmBoundSync(lock_addr, 3, add_val, lock_buffer,
                                  XS_LOCK_FAA_MASK, ctx);
    }
    track_rdma(dsm_client_->get_my_thread_id(), 1, 8);
  #else

  #ifdef USE_LOCAL_LOCK
    bool hand_over_other = can_hand_over(lock_addr);
    if (hand_over_other) {
      releases_local_lock(lock_addr);
      return;
    }
  #endif

    *lock_buffer = 0;
    if (async) {
      dsm_client_->WriteDm((char *)lock_buffer, lock_addr, sizeof(uint64_t),
                          false);
    } else {
      dsm_client_->WriteDmSync((char *)lock_buffer, lock_addr, sizeof(uint64_t),
                              ctx);
    }

  #ifdef USE_LOCAL_LOCK
    releases_local_lock(lock_addr);
  #endif

  #endif
  }

  void Tree::write_and_unlock(char *write_buffer, GlobalAddress write_addr,
                              int write_size, uint64_t *cas_buffer,
                              GlobalAddress lock_addr, CoroContext *ctx,
                              bool async, bool sx_lock) {
    Timer timer;
    timer.begin();


  #ifdef USE_SX_LOCK
    dsm_client_->Write(write_buffer, write_addr, write_size, false);
    track_rdma(dsm_client_->get_my_thread_id(), 1, write_size);
    release_sx_lock(lock_addr, cas_buffer, ctx, async, sx_lock);
  #else

  #ifdef USE_LOCAL_LOCK
    bool hand_over_other = can_hand_over(lock_addr);
    if (hand_over_other) {
      dsm_client_->WriteSync(write_buffer, write_addr, write_size, ctx);
      releases_local_lock(lock_addr);
      return;
    }
  #endif

    RdmaOpRegion rs[2];
    rs[0].source = (uint64_t)write_buffer;
    rs[0].dest = write_addr;
    rs[0].size = write_size;
    rs[0].is_on_chip = false;

    rs[1].source = (uint64_t)cas_buffer;
    rs[1].dest = lock_addr;
    rs[1].size = sizeof(uint64_t);
    rs[1].is_on_chip = true;

    *(uint64_t *)rs[1].source = 0;
    if (async) {
      dsm_client_->WriteBatch(rs, 2, false);
    } else {
      dsm_client_->WriteBatchSync(rs, 2, ctx);
    }

  #ifdef USE_LOCAL_LOCK
    releases_local_lock(lock_addr);
  #endif

  #endif
    auto t = timer.end();
    stat_helper.add(dsm_client_->get_my_thread_id(), lat_write_page, t);
  }

  void Tree::cas_and_unlock(GlobalAddress cas_addr, int log_cas_size,
                            uint64_t *cas_buffer, uint64_t equal, uint64_t swap,
                            uint64_t mask, GlobalAddress lock_addr,
                            uint64_t *lock_buffer, bool share_lock,
                            CoroContext *ctx, bool async) {
    Timer timer;
    timer.begin();

  #ifdef USE_SX_LOCK
    dsm_client_->CasMask(cas_addr, log_cas_size, equal, swap, cas_buffer, mask,
                        false);
    track_rdma(dsm_client_->get_my_thread_id(), 1, (1 << log_cas_size)); // log_cas_size 3为8B, 4为16B
    release_sx_lock(lock_addr, lock_buffer, ctx, async, share_lock);
  #else  // not USE_SX_LOCK

  #ifdef USE_LOCAL_LOCK
    bool hand_over_other = can_hand_over(lock_addr);
    if (hand_over_other) {
      dsm_client_->CasMaskSync(cas_addr, log_cas_size, equal, swap, cas_buffer,
                              mask, ctx);
      releases_local_lock(lock_addr);
      return;
    }
  #endif

    RdmaOpRegion rs[2];
    rs[0].source = (uint64_t)cas_buffer;
    rs[0].dest = cas_addr;
    rs[0].log_sz = log_cas_size;
    rs[0].is_on_chip = false;

    rs[1].source = (uint64_t)lock_buffer;
    rs[1].dest = lock_addr;
    rs[1].size = sizeof(uint64_t);
    rs[1].is_on_chip = true;
    *(uint64_t *)rs[1].source = 0;
    if (async) {
      dsm_client_->CasMaskWrite(rs[0], equal, swap, mask, rs[1], false);
    } else {
      dsm_client_->CasMaskWriteSync(rs[0], equal, swap, mask, rs[1], ctx);
    }
  #endif

  #ifdef USE_LOCAL_LOCK
    releases_local_lock(lock_addr);
  #endif

    uint64_t t = timer.end();
    stat_helper.add(dsm_client_->get_my_thread_id(), lat_write_page, t);
  }

void Tree::lock_and_read(GlobalAddress lock_addr, bool share_lock,
                        bool upgrade_from_s, uint64_t *lock_buffer,
                        GlobalAddress read_addr, int read_size,
                        char *read_buffer, CoroContext *ctx) {
  u_int16_t my_thread_id = dsm_client_->get_my_thread_id();                        
#ifdef BATCH_LOCK_READ
  // 统计总拿锁操作次数
#ifdef USE_SX_LOCK
    Timer timer;
    timer.begin();
    assert(!upgrade_from_s || !share_lock);
    uint64_t add_val;
    if (share_lock) {
      add_val = ADD_S_LOCK;
    } else {
      add_val = upgrade_from_s ? (ADD_X_LOCK | ADD_S_UNLOCK) : ADD_X_LOCK;
    }
    dsm_client_->FaaDmBound(lock_addr, 3, add_val, lock_buffer, XS_LOCK_FAA_MASK,
                            false);
    lock_rdma_faa_cnt[my_thread_id]++; // 【新增】记录 1 次批量操作中的加锁 FAA
    track_rdma(my_thread_id, 1, 8); // FAA 请求 8 字节

    dsm_client_->ReadSync(read_buffer, read_addr, read_size, ctx);
    track_rdma(my_thread_id, 1, read_size); // 读数据
    uint16_t s_tic = *lock_buffer & 0xffff;
    uint16_t s_cnt = (*lock_buffer >> 16) & 0xffff;
    uint16_t x_tic = (*lock_buffer >> 32) & 0xffff;
    uint16_t x_cnt = (*lock_buffer >> 48) & 0xffff;

    if (upgrade_from_s) {
      ++s_cnt;
    }

    uint64_t retry_cnt = 0;
  retry:
    if (share_lock && x_cnt == x_tic) {
      // ok
      // 成功获取共享锁
    record_lock_retry(my_thread_id, retry_cnt);
    } else if (!share_lock && x_cnt == x_tic && s_cnt == s_tic) {
      // ok
      // 成功获取排他锁
    record_lock_retry(my_thread_id, retry_cnt);
    } else {
      ++retry_cnt;
      if (retry_cnt > 5000000) {
        // 记录这次失败的尝试
        tries_per_lock[my_thread_id][5000]++;
        printf(
            "Deadlock [%u, %lu] my thread %d coro_id %d try %d lock upgrade "
            "%d\n",
            lock_addr.nodeID, lock_addr.offset, dsm_client_->get_my_thread_id(),
            ctx ? ctx->coro_id : 0, share_lock, upgrade_from_s);
        printf("s [%u, %u] x [%u, %u]\n", s_tic, s_cnt, x_tic, x_cnt);
        fflush(stdout);
        assert(false);
        exit(-1);
      }

      // Wait on the lock word only; re-read payload once after the lock is
      // actually granted to avoid repeated wasted data fetches under
      // contention.
      dsm_client_->ReadDmSync((char *)lock_buffer, lock_addr, 8, ctx);
      lock_rdma_read_cnt[my_thread_id]++;
      track_rdma(my_thread_id, 1, 8);
      s_cnt = (*lock_buffer >> 16) & 0xffff;
      x_cnt = (*lock_buffer >> 48) & 0xffff;
      goto retry;
    }

    if (retry_cnt > 0) {
      dsm_client_->ReadSync(read_buffer, read_addr, read_size, ctx);
      lock_retry_data_reread_cnt[my_thread_id]++;
      lock_retry_data_reread_bytes[my_thread_id] += read_size;
      track_rdma(my_thread_id, 1, read_size);
    }

    uint64_t t = timer.end();
    stat_helper.add(dsm_client_->get_my_thread_id(), lat_read_page, t);

  #else  // not sx lock
    RdmaOpRegion rs[2];
    {
      uint64_t retry_cnt = 0;
      uint64_t pre_tag = 0;
      uint64_t conflict_tag = 0;
      auto tag = dsm_client_->get_thread_tag();
    retry:
      rs[0].source = (uint64_t)lock_buffer;
      rs[0].dest = lock_addr;
      rs[0].size = 8;
      rs[0].is_on_chip = true;

      rs[1].source = (uint64_t)(read_buffer);
      rs[1].dest = read_addr;
      rs[1].size = read_size;
      rs[1].is_on_chip = false;
      if (retry_cnt > 5000000) {
        printf("Deadlock [%u, %lu] my thread %d coro_id %d\n", lock_addr.nodeID,
              lock_addr.offset, dsm_client_->get_my_thread_id(),
              ctx ? ctx->coro_id : 0);
        fflush(stdout);
        assert(false);
        exit(-1);
      }

      Timer timer;
      timer.begin();
      bool res = dsm_client_->CasReadSync(rs[0], rs[1], 0, tag, ctx);
      uint64_t t = timer.end();
      stat_helper.add(dsm_client_->get_my_thread_id(), lat_read_page, t);
      if (!res) {
        // conflict_tag = *buf - 1;
        conflict_tag = *lock_buffer;
        if (conflict_tag != pre_tag) {
          retry_cnt = 0;
          pre_tag = conflict_tag;
        }
        goto retry;
      }
    }
  #endif

  #else  // not batch

    Timer timer;
    timer.begin();
    acquire_lock(lock_addr, lock_buffer, ctx, share_lock, upgrade_from_s);
    dsm_client_->ReadSync(read_buffer, read_addr, read_size, ctx);
    uint64_t t = timer.end();
    stat_helper.add(dsm_client_->get_my_thread_id(), lat_read_page, t);
  #endif
  }

  void Tree::lock_bench(const Key &k, CoroContext *ctx, int coro_id) {
    // uint64_t lock_index = CityHash64((char *)&k, sizeof(k)) %
    // define::kNumOfLock;

    // GlobalAddress lock_addr;
    // lock_addr.nodeID = 0;
    // lock_addr.offset = lock_index * sizeof(uint64_t);
    // auto cas_buffer = dsm->get_rbuf(coro_id).get_cas_buffer();

    // // bool res = dsm->cas_sync(lock_addr, 0, 1, cas_buffer, ctx);
    // // try_lock_addr(lock_addr, 1, cas_buffer, ctx, coro_id);
    // // unlock_addr(lock_addr, 1, cas_buffer, ctx, coro_id, true);
    // bool sx_lock = false;
    // acquire_sx_lock(lock_addr, 1, cas_buffer, ctx, coro_id, sx_lock);
    // release_sx_lock(lock_addr, 1, cas_buffer, ctx, coro_id, true, sx_lock);

    // read page test
    auto &rbuf = dsm_client_->get_rbuf(coro_id);
    uint64_t *cas_buffer = rbuf.get_cas_buffer();
    auto page_buffer = rbuf.get_page_buffer();

    GlobalAddress lock_addr;
    lock_addr.nodeID = 0;
    uint64_t lock_index = k % define::kNumOfLock;
    lock_addr.offset = lock_index * sizeof(uint64_t);

    GlobalAddress page_addr;
    page_addr.nodeID = 0;
    constexpr uint64_t page_num = 4ul << 20;
    page_addr.offset = (k % page_num) * kLeafPageSize;
    // GlobalAddress entry_addr = page_addr;
    // entry_addr.offset += 512;

    // dsm->read(page_buffer, page_addr, 32, false, ctx);
    // dsm->read_sync(page_buffer, page_addr, kLeafPageSize, ctx);

    write_and_unlock(page_buffer, page_addr, 128, cas_buffer, lock_addr, ctx,
                    false, false);

    // lock_and_read_page(page_buffer, page_addr, kLeafPageSize, cas_buffer,
    //                    lock_addr, ctx, coro_id, true);
    // release_sx_lock(lock_addr, cas_buffer, ctx, coro_id, false, true);
  }

  void Tree::insert_internal_update_left_child(const Key &k, GlobalAddress v,
                                              const Key &left_child,
                                              GlobalAddress left_child_val,
                                              CoroContext *ctx, int level) {
    assert(left_child_val != GlobalAddress::Null());
    auto root = get_root_ptr(ctx);
    SearchResult result;

    GlobalAddress p = root;
    int level_hint = -1;
    Key min = kKeyMin;
    Key max = kKeyMax;

  next:
    if (!page_search(p, level_hint, p.read_gran, min, max, left_child, result,
                    ctx)) {
      std::cout << "SEARCH WARNING insert" << std::endl;
      p = get_root_ptr(ctx);
      level_hint = -1;
      sleep(1);
      goto next;
    }

    assert(result.level != 0);
    if (result.sibling != GlobalAddress::Null()) {
      p = result.sibling;
      level_hint = result.level;
      min = result.next_min; // remain the max
      goto next;
    }

    if (result.level >= level + 1) {
      p = result.next_level;
      if (result.level > level + 1) {
        level_hint = result.level - 1;
        min = result.next_min;
        max = result.next_max;
        goto next;
      }
    }

    // internal_page_store(p, k, v, root, level, ctx, coro_id);
    assert(result.level == level + 1 || result.level == level);
    internal_page_store_update_left_child(p, k, v, left_child, left_child_val,
                                          root, level, ctx);
  }

  void Tree::insert(const Key &k, const Value &v, CoroContext *ctx) {
    assert(dsm_client_->IsRegistered());
    int coro_id = ctx ? ctx->coro_id : 0;

    before_operation(ctx, coro_id);

    Key min = kKeyMin;
    Key max = kKeyMax;

    if (enable_cache) {
      GlobalAddress cache_addr, parent_addr;
      auto entry = index_cache->search_from_cache(k, &cache_addr, &parent_addr);
      if (entry) {  // cache hit
        path_stack[coro_id][1] = parent_addr;
        auto root = get_root_ptr(ctx);
  #ifdef USE_SX_LOCK
        bool status = leaf_page_store(cache_addr, k, v, root, 0, ctx, true, true);
  #else
        bool status = leaf_page_store(cache_addr, k, v, root, 0, ctx, true);
  #endif
        if (status) {
          cache_hit[dsm_client_->get_my_thread_id()][0]++;
          return;
        }
        // cache stale, from root,
        index_cache->invalidate(entry, dsm_client_->get_my_thread_id());
      }
      cache_miss[dsm_client_->get_my_thread_id()][0]++;
    }

    auto root = get_root_ptr(ctx);
    SearchResult result;

    GlobalAddress p = root;
    int level_hint = -1;
    int cnt = 0;

  next:
    if (!page_search(p, level_hint, p.read_gran, min, max, k, result, ctx)) {
      std::cout << "SEARCH WARNING insert" << std::endl;
      p = get_root_ptr(ctx);
      level_hint = -1;
      min = kKeyMin;
      max = kKeyMax;
      sleep(1);
      goto next;
    }

    if (!result.is_leaf) {
      assert(result.level != 0);
      if (result.sibling != GlobalAddress::Null()) {
        p = result.sibling;
        min = result.next_min; // remain the max
        level_hint = result.level;
        goto next;
      }

      p = result.next_level;
      level_hint = result.level - 1;
      if (result.level != 1) {
        min = result.next_min;
        max = result.next_max;
        goto next;
      }
    }

    bool res = false;
    
    // while (!res) {

  #ifdef USE_SX_LOCK
    res = leaf_page_store(p, k, v, root, 0, ctx, false, true);
  #else
    res = leaf_page_store(p, k, v, root, 0, ctx, false);
  #endif
    if (!res) {
      // retry
      p = get_root_ptr(ctx);
      level_hint = -1;
      min = kKeyMin;
      max = kKeyMax;
      goto next;
      ++cnt;
      if (cnt > 1) {
        printf("retry insert <k:%lu v:%lu> %d\n", k, v, cnt);
      }
    }
    // }
  }

  bool Tree::search(const Key &k, Value &v, CoroContext *ctx, int coro_id) {
    assert(dsm_client_->IsRegistered());

    auto p = get_root_ptr(ctx);
    SearchResult result;
    Key min = kKeyMin;
    Key max = kKeyMax;

    int level_hint = -1;

    bool from_cache = false;
    const CacheEntry *entry = nullptr;
    if (enable_cache) {
      // Timer timer;
      // timer.begin();
      GlobalAddress cache_addr, parent_addr;
      entry = index_cache->search_from_cache(k, &cache_addr, &parent_addr);
      if (entry) {  // cache hit
        cache_hit[dsm_client_->get_my_thread_id()][0]++;
        from_cache = true;
        p = cache_addr;
        level_hint = 0;
      } else {
        cache_miss[dsm_client_->get_my_thread_id()][0]++;
      }
      // auto t = timer.end();
      // stat_helper.add(dsm_client_->get_my_thread_id(), lat_cache_search, t);
    }

  next:
    if (!page_search(p, level_hint, p.read_gran, min, max, k, result, ctx,
                    from_cache)) {
      if (from_cache) {  // cache stale
        index_cache->invalidate(entry, dsm_client_->get_my_thread_id());
        cache_hit[dsm_client_->get_my_thread_id()][0]--;
        cache_miss[dsm_client_->get_my_thread_id()][0]++;
        from_cache = false;

        p = get_root_ptr(ctx);
        level_hint = -1;
      } else {
        std::cout << "SEARCH WARNING search" << std::endl;
        sleep(1);
      }
      min = kKeyMin;
      max = kKeyMax;
      goto next;
    }
    if (result.is_leaf) {
      if (result.val != kValueNull) {  // find
        v = result.val;
        return true;
      }
      if (result.sibling != GlobalAddress::Null()) {  // turn right
        p = result.sibling;
        min = result.next_min; // remain the max
        level_hint = 0;
        goto next;
      }
      return false;  // not found
    } else {         // internal
      if (result.sibling != GlobalAddress::Null()) {
        p = result.sibling;
        min = result.next_min; // remain the max
        level_hint = result.level;
      } else {
        p = result.next_level;
        min = result.next_min;
        max = result.next_max;
        level_hint = result.level - 1;
      }
      goto next;
    }
  }

  int Tree::range_query(const Key &from, const Key &to, Value *value_buffer,
                        int max_cnt, CoroContext *ctx, int coro_id) {
    const int kParaFetch = 32;
    thread_local std::vector<InternalPage *> result;
    thread_local std::vector<GlobalAddress> leaves;

    result.clear();
    leaves.clear();

    index_cache->thread_status->rcu_progress(dsm_client_->get_my_thread_id());

    index_cache->search_range_from_cache(from, to, result);

    // FIXME: here, we assume all innernal nodes are cached in compute node
    if (result.empty()) {
      index_cache->thread_status->rcu_exit(dsm_client_->get_my_thread_id());
      return 0;
    }

    int counter = 0;
    for (auto page : result) {
      std::vector<InternalEntry> tmp_records;
      tmp_records.reserve(kInternalCardinality);
      for (int i = 0; i < kInternalCardinality; ++i) {
        if (page->records[i].ptr != GlobalAddress::Null()) {
          tmp_records.push_back(page->records[i]);
        }
      }
      std::sort(tmp_records.begin(), tmp_records.end());
      int cnt = tmp_records.size();
      if (cnt > 0) {
        if (page->hdr.leftmost_ptr != GlobalAddress::Null()) {
          if (tmp_records[0].key > from && page->hdr.lowest < to) {
            leaves.push_back(page->hdr.leftmost_ptr);
          }
        }
        for (int i = 1; i < cnt; i++) {
          if (tmp_records[i].key > from && tmp_records[i - 1].key < to) {
            leaves.push_back(tmp_records[i - 1].ptr);
          }
        }
        if (page->hdr.highest > from && tmp_records[cnt - 1].key < to) {
          leaves.push_back(tmp_records[cnt - 1].ptr);
        }
      }
    }

    int cq_cnt = 0;
    char *range_buffer = (dsm_client_->get_rbuf(coro_id)).get_range_buffer();
    for (size_t i = 0; i < leaves.size(); ++i) {
      if (i > 0 && i % kParaFetch == 0) {
        dsm_client_->PollRdmaCq(kParaFetch);
        cq_cnt -= kParaFetch;
        for (int k = 0; counter < max_cnt && k < kParaFetch; ++k) {
          auto page = (LeafPage *)(range_buffer + k * kLeafPageSize);
          for (int idx = 0; counter < max_cnt && idx < kNumGroup; ++idx) {
            LeafEntryGroup *g = &page->groups[idx];
            for (int j = 0; counter < max_cnt && j < kAssociativity; ++j) {
              auto &r = g->front[j];
              if (r.lv.val != kValueNull && r.key >= from && r.key < to) {
                value_buffer[counter++] = r.lv.val;
              }
            }
            for (int j = 0; counter < max_cnt && j < kAssociativity; ++j) {
              auto &r = g->back[j];
              if (r.lv.val != kValueNull && r.key >= from && r.key < to) {
                value_buffer[counter++] = r.lv.val;
              }
            }
            for (int j = 0; counter < max_cnt && j < kGroupOverflowSlots; ++j) {
              auto &r = g->overflow[j];
              if (r.lv.val != kValueNull && r.key >= from && r.key < to) {
                value_buffer[counter++] = r.lv.val;
              }
            }
          }
          for (int j = 0; counter < max_cnt && j < kLeafStashSlots; ++j) {
            auto &r = *page->stash_entry(j);
            if (r.lv.val != kValueNull && r.key >= from && r.key < to) {
              value_buffer[counter++] = r.lv.val;
            }
          }
        }
      }
      dsm_client_->Read(range_buffer + kLeafPageSize * (i % kParaFetch),
                        leaves[i], kLeafPageSize, true);
      cq_cnt++;
    }

    if (cq_cnt != 0) {
      dsm_client_->PollRdmaCq(cq_cnt);
      for (int k = 0; counter < max_cnt && k < cq_cnt; ++k) {
        auto page = (LeafPage *)(range_buffer + k * kLeafPageSize);
        for (int idx = 0; counter < max_cnt && idx < kNumGroup; ++idx) {
          LeafEntryGroup *g = &page->groups[idx];
          for (int j = 0; counter < max_cnt && j < kAssociativity; ++j) {
            LeafEntry &r = g->front[j];
            if (r.lv.val != kValueNull && r.key >= from && r.key < to) {
              value_buffer[counter++] = r.lv.val;
            }
          }
          for (int j = 0; counter < max_cnt && j < kAssociativity; ++j) {
            auto &r = g->back[j];
            if (r.lv.val != kValueNull && r.key >= from && r.key < to) {
              value_buffer[counter++] = r.lv.val;
            }
          }
          for (int j = 0; counter < max_cnt && j < kGroupOverflowSlots; ++j) {
            auto &r = g->overflow[j];
            if (r.lv.val != kValueNull && r.key >= from && r.key < to) {
              value_buffer[counter++] = r.lv.val;
            }
          }
        }
        for (int j = 0; counter < max_cnt && j < kLeafStashSlots; ++j) {
          auto &r = *page->stash_entry(j);
          if (r.lv.val != kValueNull && r.key >= from && r.key < to) {
            value_buffer[counter++] = r.lv.val;
          }
        }
      }
    }

    index_cache->thread_status->rcu_exit(dsm_client_->get_my_thread_id());
    return counter;
  }

  void Tree::del(const Key &k, CoroContext *ctx, int coro_id) {
    assert(dsm_client_->IsRegistered());

    before_operation(ctx, coro_id);
    Key min = kKeyMin;
    Key max = kKeyMax;

    if (enable_cache) {
      GlobalAddress cache_addr, parent_addr;
      auto entry = index_cache->search_from_cache(k, &cache_addr, &parent_addr);
      if (entry) {  // cache hit
        if (leaf_page_del(cache_addr, k, 0, ctx, coro_id, true)) {
          cache_hit[dsm_client_->get_my_thread_id()][0]++;
          return;
        }
        // cache stale, from root,
        index_cache->invalidate(entry, dsm_client_->get_my_thread_id());
      }
      cache_miss[dsm_client_->get_my_thread_id()][0]++;
    }

    auto root = get_root_ptr(ctx);
    SearchResult result;

    GlobalAddress p = root;
    int level_hint = -1;

  next:

    if (!page_search(p, level_hint, p.read_gran, min, max, k, result, ctx)) {
      std::cout << "SEARCH WARNING del" << std::endl;
      p = get_root_ptr(ctx);
      min = kKeyMin;
      max = kKeyMax;
      level_hint = -1;
      sleep(1);
      goto next;
    }

    if (!result.is_leaf) {
      assert(result.level != 0);
      if (result.sibling != GlobalAddress::Null()) {
        p = result.sibling;
        min = result.next_min;
        level_hint = result.level;
        goto next;
      }

      p = result.next_level;
      level_hint = result.level - 1;
      if (result.level != 1) {
        min = result.next_min;
        max = result.next_max;
        goto next;
      }
    }

    leaf_page_del(p, k, 0, ctx, coro_id);
  }

  bool Tree::leaf_page_group_search(GlobalAddress page_addr, const Key &k,
                                    SearchResult &result, CoroContext *ctx,
                                    bool from_cache) {
    char *page_buffer =
        (dsm_client_->get_rbuf(ctx ? ctx->coro_id : 0)).get_page_buffer();

    int bucket_id = key_hash_bucket(k);
    int group_id = bucket_id / 2;

    Header *hdr = (Header *)(page_buffer + offsetof(LeafPage, hdr));
    int group_offset =
        offsetof(LeafPage, groups) + sizeof(LeafEntryGroup) * group_id;
    LeafEntryGroup *group = (LeafEntryGroup *)(page_buffer + group_offset);
    int bucket_offset =
        group_offset + (bucket_id % 2 ? kBackOffset : kFrontOffset);
    bool has_header = false;
    int header_offset = offsetof(LeafPage, hdr);

    int read_counter = 0;
  re_read:
    if (++read_counter > 10) {
      printf("re-read (leaf_page_group) too many times\n");
      sleep(1);
    }

    if (has_header) {
      dsm_client_->Read(page_buffer + bucket_offset,
                        GADD(page_addr, bucket_offset), kReadBucketSize, false);
      track_rdma(dsm_client_->get_my_thread_id(), 1, kReadBucketSize); // 📝【埋点：异步读 Bucket】
      // read header
      dsm_client_->ReadSync(page_buffer + header_offset,
                            GADD(page_addr, header_offset), sizeof(Header), ctx);
      track_rdma(dsm_client_->get_my_thread_id(), 1, sizeof(Header));  // 📝【埋点：同步读 Header】          
    } else {
      dsm_client_->ReadSync(page_buffer + bucket_offset,
                            GADD(page_addr, bucket_offset), kReadBucketSize, ctx);
      track_rdma(dsm_client_->get_my_thread_id(), 1, kReadBucketSize); // 📝【埋点：同步读 Bucket】
    }

    result.clear();
    result.is_leaf = true;
    result.level = 0;
    uint8_t actual_version;
    if (!group->check_consistency(!(bucket_id % 2), page_addr.node_version,
                                  actual_version)) {
      if (from_cache) {
        return false;
      } else {
        has_header = true;
        page_addr.node_version = actual_version;
        goto re_read;
      }
    }
    if (has_header && k >= hdr->highest) {
      result.sibling = hdr->sibling_ptr;
      result.next_min = hdr->highest;
      return true;
    }

    bool res = group->find(k, result, !(bucket_id % 2));
    if (!res) {
#ifndef USE_LEAF_STASH
      if (from_cache) {
        if (!has_header) {
          dsm_client_->ReadSync(page_buffer + header_offset,
                                GADD(page_addr, header_offset), sizeof(Header),
                                ctx);
          track_rdma(dsm_client_->get_my_thread_id(), 1, sizeof(Header));
          has_header = true;
        }
        if (k >= hdr->highest || k < hdr->lowest) {
          return false;
        }
      }
      return true;
#else
      if (!has_header) {
        dsm_client_->ReadSync(page_buffer + header_offset,
                              GADD(page_addr, header_offset), sizeof(Header),
                              ctx);
        track_rdma(dsm_client_->get_my_thread_id(), 1, sizeof(Header));
        has_header = true;
      }
      if (hdr->version != page_addr.node_version) {
        page_addr.node_version = hdr->version;
        goto re_read;
      }
      if (k >= hdr->highest || k < hdr->lowest) {
        if (from_cache) {
          return false;
        }
        if (k >= hdr->highest) {
          result.sibling = hdr->sibling_ptr;
          result.next_min = hdr->highest;
          return true;
        }
      }
      LeafPage *page = reinterpret_cast<LeafPage *>(page_buffer);
      if (leaf_stash_may_have(page, group_id)) {
        size_t stash_offset = offsetof(LeafPage, stash);
        dsm_client_->ReadSync(page_buffer + stash_offset,
                              GADD(page_addr, stash_offset),
                              sizeof(LeafEntry) * kLeafStashSlots, ctx);
        track_rdma(dsm_client_->get_my_thread_id(), 1,
                   sizeof(LeafEntry) * kLeafStashSlots);
        LeafEntry *stash_entry = find_leaf_stash_entry(page, k, group_id);
        if (stash_entry) {
          result.val = stash_entry->lv.val;
          return true;
        }
      }
#endif
    }
    return true;
  }

  bool Tree::page_search(GlobalAddress page_addr, int level_hint, int read_gran,
                        Key min, Key max, const Key &k, SearchResult &result,
                        CoroContext *ctx, bool from_cache) {
  #ifdef FINE_GRAINED_LEAF_NODE
    if (page_addr != g_root_ptr && level_hint == 0) {
      return leaf_page_group_search(page_addr, k, result, ctx, from_cache);
    }
  #endif

  #ifndef FINE_GRAINED_INTERNAL_NODE
    read_gran = gran_full;
  #endif

    char *page_buffer =
        (dsm_client_->get_rbuf(ctx ? ctx->coro_id : 0)).get_page_buffer();
    Header *hdr = (Header *)(page_buffer + offsetof(LeafPage, hdr));
    bool has_header = false;
    InternalEntry *guard = nullptr;
    size_t start_offset = 0;
    int internal_read_cnt;
    int group_id = -1;

    int read_counter = 0;
  re_read:
    if (++read_counter > 10) {
      printf("re-read (page_search) too many times\n");
      sleep(1);
      assert(false);
    }

    result.clear();
    if (read_gran == gran_full) {
      has_header = true;
      dsm_client_->ReadSync(page_buffer, page_addr,
                            std::max(kInternalPageSize, kLeafPageSize), ctx);
      track_rdma(dsm_client_->get_my_thread_id(), 1, std::max(kInternalPageSize, kLeafPageSize));                      
      size_t start_offset =
          offsetof(InternalPage, records) - sizeof(InternalEntry);
      internal_read_cnt = kInternalCardinality + 1;
      guard = reinterpret_cast<InternalEntry *>(page_buffer + start_offset);
      // has header
      result.is_leaf = hdr->level == 0;
      result.level = hdr->level;
      if (page_addr == g_root_ptr) {
        if (hdr->is_root == false) {
          // update root ptr
          get_root_ptr(ctx, true);
        }
      }
  #ifdef USE_CRC
    Timer t_crc;
    t_crc.begin();
    uint32_t c;
    if (result.is_leaf) {
      LeafPage *p = (LeafPage *)page_buffer;
      c = p->check_crc();
    } else {
      InternalPage *p = (InternalPage *)page_buffer;
      c = p->check_crc();
    }
    uint64_t t = t_crc.end();
    stat_helper.add(dsm_client_->get_my_thread_id(), lat_crc, t);
    stat_helper.add(dsm_client_->get_my_thread_id(), lat_cache_search, c);
  #endif
    } else {
      group_id = get_key_group(k, min, max);
      // read one more entry
      if (read_gran == gran_quarter) {
        start_offset = offsetof(InternalPage, records) +
                      (group_id * kGroupCardinality - 1) * sizeof(InternalEntry);
        internal_read_cnt = kGroupCardinality + 1;
      } else {
        // half
        int begin = (group_id < 2 ? 0 : kGroupCardinality * 2) - 1;
        start_offset =
            offsetof(InternalPage, records) + begin * sizeof(InternalEntry);
        internal_read_cnt = kGroupCardinality * 2 + 1;
      }
      // DEBUG:
      // dsm_client_->ReadSync(page_buffer, page_addr,
      //                       std::max(kInternalPageSize, kLeafPageSize), ctx);
      dsm_client_->ReadSync(page_buffer + start_offset,
                            GADD(page_addr, start_offset),
                            internal_read_cnt * sizeof(InternalEntry), ctx);
      guard = reinterpret_cast<InternalEntry *>(page_buffer + start_offset);
      track_rdma(dsm_client_->get_my_thread_id(), 1, internal_read_cnt * sizeof(InternalEntry));
      assert(level_hint != 0);
      result.is_leaf = false;
      result.level = level_hint;
    }

    path_stack[ctx ? ctx->coro_id : 0][result.level] = page_addr;

    if (result.is_leaf) {
      LeafPage *page = (LeafPage *)page_buffer;
      int bucket_id = key_hash_bucket(k);
      LeafEntryGroup *g = &page->groups[bucket_id / 2];
      // check version
      uint8_t actual_version;
      if (!g->check_consistency(!(bucket_id % 2), page_addr.node_version,
                                actual_version)) {
        if (from_cache) {
          return false;
        } else {
          page_addr.node_version = actual_version;
          goto re_read;
        }
      }
      if (has_header) {
        if (k >= hdr->highest) {
          result.sibling = hdr->sibling_ptr;
          result.next_min = hdr->highest;
          return true;
        } else if (k < hdr->lowest) {
          assert(false);
          return false;
        }
      }
      bool res = g->find(k, result, !(bucket_id % 2));
#ifdef USE_LEAF_STASH
      if (!res) {
        LeafEntry *stash_entry =
            find_leaf_stash_entry(page, k, bucket_id / 2);
        if (stash_entry) {
          result.val = stash_entry->lv.val;
          res = true;
        }
      }
#endif
      if (!res && from_cache && !has_header) {
        // check header
        size_t header_offset = offsetof(LeafPage, hdr);
        dsm_client_->ReadSync(page_buffer + header_offset,
                              GADD(page_addr, header_offset), sizeof(Header),
                              ctx);
        if (k >= hdr->highest || k < hdr->lowest) {
          // cache is stale
          return false;
        }
      }
    } else {
      // Internal Page
      assert(!from_cache);
      InternalPage *page = (InternalPage *)page_buffer;
      // check version
      char *end = (char *)(guard + internal_read_cnt);
      uint8_t actual_version;
      if (!page->check_consistency((char *)guard, end, page_addr.node_version,
                                  actual_version)) {
        page_addr.node_version = actual_version;
        read_gran = gran_full;
        goto re_read;
      }
      if (has_header) {
        if (k >= hdr->highest) {
          result.sibling = hdr->sibling_ptr;
          result.next_min = hdr->highest;
          return true;
        } else if (k < hdr->lowest) {
          assert(false);
          return false;
        }
      }

      if (read_gran == gran_full) {
        // maybe is a retry: update group id, gran, guard
        uint8_t actual_gran = hdr->read_gran;
        if (actual_gran != gran_full) {
          group_id = get_key_group(k, hdr->lowest, hdr->highest);
          if (actual_gran == gran_quarter) {
            start_offset =
                offsetof(InternalPage, records) +
                (group_id * kGroupCardinality - 1) * sizeof(InternalEntry);
            internal_read_cnt = kGroupCardinality + 1;
          } else {
            // half
            int begin = (group_id < 2 ? 0 : kGroupCardinality * 2) - 1;
            start_offset =
                offsetof(InternalPage, records) + begin * sizeof(InternalEntry);
            internal_read_cnt = kGroupCardinality * 2 + 1;
          }
          guard = reinterpret_cast<InternalEntry *>(page_buffer + start_offset);
        }
      }

      // Timer timer;
      // timer.begin();
      internal_page_slice_search(guard, internal_read_cnt, k, result);
      assert(result.sibling != GlobalAddress::Null() ||
            result.next_level != GlobalAddress::Null());
      if (result.level == 1 && enable_cache) {
        if (read_gran == gran_full) {
          index_cache->add_to_cache(page, dsm_client_->get_my_thread_id());
        } else {
          index_cache->add_sub_node(page_addr, group_id, read_gran, start_offset,
                                    guard,
                                    internal_read_cnt * sizeof(InternalEntry),
                                    min, max, dsm_client_->get_my_thread_id());
        }
      }
      // auto t = timer.end();
      // stat_helper.add(dsm_client_->get_my_thread_id(), lat_internal_search, t);
    }

    return true;
  }

  void Tree::update_ptr_internal(const Key &k, GlobalAddress v, CoroContext *ctx,
                                int level) {
    GlobalAddress root = get_root_ptr(ctx);
    SearchResult result;

    GlobalAddress p = root;
    int level_hint = -1;
    Key min = kKeyMin;
    Key max = kKeyMax;

  next:
    if (!page_search(p, level_hint, p.read_gran, min, max, k, result, ctx)) {
      std::cout << "SEARCH WARNING insert" << std::endl;
      p = get_root_ptr(ctx);
      level_hint = -1;
      sleep(1);
      goto next;
    }

    assert(result.level != 0);
    if (result.sibling != GlobalAddress::Null()) {
      p = result.sibling;
      level_hint = result.level;
      min = result.next_min; // remain the max
      goto next;
    }

    if (result.level >= level + 1) {
      p = result.next_level;
      if (result.level > level + 1) {
        level_hint = result.level - 1;
        min = result.next_min;
        max = result.next_max;
        goto next;
      }
    }

    // internal_page_store(p, k, v, root, level, ctx, coro_id);
    assert(result.level == level + 1 || result.level == level);
  #ifdef USE_SX_LOCK
    internal_page_update(p, k, v, level, ctx, true);
  #else
    internal_page_update(p, k, v, level, ctx, false);
  #endif
  }

  inline void Tree::internal_page_slice_search(InternalEntry *entries, int cnt,
                                              const Key k,
                                              SearchResult &result) {
    // find next_min: maxium <= k; next_max: minium > k
    int min_idx = -1, max_idx = -1;
    for (int i = 0; i < cnt; ++i) {
      if (entries[i].ptr != GlobalAddress::Null()) {
        if (entries[i].key <= k) {
          if (min_idx == -1 || entries[i].key > entries[min_idx].key) {
            min_idx = i;
          }
        } else {
          if (max_idx == -1 || entries[i].key < entries[max_idx].key) {
            max_idx = i;
          }
        }
      }
    }
    assert(min_idx != -1);
    result.next_level = entries[min_idx].ptr;
    if (max_idx == -1) {
      // can't know max of next level, so read full page
      result.next_level.read_gran = gran_full;
    } else {
      result.next_min = entries[min_idx].key;
      result.next_max = entries[max_idx].key;
    }
  }

  void Tree::internal_page_update(GlobalAddress page_addr, const Key &k,
                                  GlobalAddress value, int level,
                                  CoroContext *ctx, bool sx_lock) {
    GlobalAddress lock_addr = get_lock_addr(page_addr);

    auto &rbuf = dsm_client_->get_rbuf(ctx ? ctx->coro_id : 0);
    uint64_t *lock_buffer = rbuf.get_cas_buffer();
    char *page_buffer = rbuf.get_page_buffer();

    lock_and_read(lock_addr, sx_lock, false, lock_buffer, page_addr,
                  kInternalPageSize, page_buffer, ctx);

    InternalPage *page = reinterpret_cast<InternalPage *>(page_buffer);
    Header *hdr = &page->hdr;
    assert(hdr->level == level);
    if (k >= hdr->highest) {
      assert(page->hdr.sibling_ptr != GlobalAddress::Null());
      release_lock(lock_addr, lock_buffer, ctx, true, sx_lock);
      internal_page_update(page->hdr.sibling_ptr, k, value, level, ctx, sx_lock);
      return;
    }
    assert(k >= page->hdr.lowest);
    if (k == page->hdr.lowest) {
      assert(page->hdr.leftmost_ptr == value);
      page->hdr.leftmost_ptr = value;
      char *modfiy = (char *)&page->hdr.leftmost_ptr;
      write_and_unlock(modfiy, GADD(page_addr, (modfiy - page_buffer)),
                      sizeof(GlobalAddress), lock_buffer, lock_addr, ctx, true,
                      sx_lock);
      return;
    } else {
      int group_id = get_key_group(k, page->hdr.lowest, page->hdr.highest);
      uint8_t cur_group_gran = page->hdr.read_gran;
      int begin_idx, end_idx;
      if (cur_group_gran == gran_quarter) {
        begin_idx = group_id * kGroupCardinality;
        end_idx = begin_idx + kGroupCardinality;
      } else if (cur_group_gran == gran_half) {
        begin_idx = group_id < 2 ? 0 : kGroupCardinality * 2;
        end_idx = begin_idx + kGroupCardinality * 2;
      } else {
        assert(cur_group_gran == gran_full);
        begin_idx = 0;
        end_idx = kInternalCardinality;
      }

      for (int i = begin_idx; i < end_idx; ++i) {
        if (page->records[i].ptr == value && page->records[i].key == k) {
          page->records[i].ptr = value;
          char *modify = (char *)&page->records[i].ptr;
          write_and_unlock(modify, GADD(page_addr, (modify - page_buffer)),
                          sizeof(GlobalAddress), lock_buffer, lock_addr, ctx,
                          true, sx_lock);
          return;
        }
      }
    }

    assert(false);
    release_lock(lock_addr, lock_buffer, ctx, true, sx_lock);
  }

  void Tree::internal_page_store_update_left_child(GlobalAddress page_addr,
                                                  const Key &k, GlobalAddress v,
                                                  const Key &left_child,
                                                  GlobalAddress left_child_val,
                                                  GlobalAddress root, int level,
                                                  CoroContext *ctx) {
    GlobalAddress lock_addr = get_lock_addr(page_addr);

    auto &rbuf = dsm_client_->get_rbuf(ctx ? ctx->coro_id : 0);
    uint64_t *lock_buffer = rbuf.get_cas_buffer();
    char *page_buffer = rbuf.get_page_buffer();

    lock_and_read(lock_addr, false, false, lock_buffer, page_addr,
                  kInternalPageSize, page_buffer, ctx);

    InternalPage *page = (InternalPage *)page_buffer;

    assert(page->hdr.level == level);
    // auto cnt = page->hdr.cnt;

    if (left_child_val != GlobalAddress::Null() &&
        left_child >= page->hdr.highest) {
      assert(page->hdr.sibling_ptr != GlobalAddress::Null());
      release_lock(lock_addr, lock_buffer, ctx, true, false);
      internal_page_store_update_left_child(page->hdr.sibling_ptr, k, v,
                                            left_child, left_child_val, root,
                                            level, ctx);
      return;
    }
    if (k >= page->hdr.highest) {
      // left child in current node, new sibling leaf in sibling node
      if (left_child_val != GlobalAddress::Null()) {
        GlobalAddress *modify = nullptr;
        if (left_child == page->hdr.lowest) {
          assert(page->hdr.leftmost_ptr == left_child_val);
          page->hdr.leftmost_ptr = left_child_val;
          modify = &page->hdr.leftmost_ptr;
        } else {
          int idx = page->find_records_not_null(left_child);
          if (idx != -1) {
            page->records[idx].ptr = left_child_val;
            modify = &page->records[idx].ptr;
          }
        }
        if (modify) {
          dsm_client_->Write((char *)modify,
                            GADD(page_addr, ((char *)modify - page_buffer)),
                            sizeof(GlobalAddress), false);
        }
      }
      release_lock(lock_addr, lock_buffer, ctx, true, false);
      internal_page_store_update_left_child(page->hdr.sibling_ptr, k, v, kKeyMin,
                                            GlobalAddress::Null(), root, level,
                                            ctx);
      return;
    }
    assert(k >= page->hdr.lowest);

    int group_id = get_key_group(k, page->hdr.lowest, page->hdr.highest);
    uint8_t cur_gran = page->hdr.read_gran;
    int begin_idx, max_cnt;

    InternalEntry *left_update_addr = nullptr;
    InternalEntry *insert_addr = nullptr;

    if (left_child_val != GlobalAddress::Null()) {
      // find left_child
      if (left_child == page->hdr.lowest) {
        assert(page->hdr.leftmost_ptr == left_child_val);
        left_update_addr = (InternalEntry *)&page->hdr.leftmost_ptr;
      } else {
        int left_idx = page->find_records_not_null(left_child);
        if (left_idx != -1) {
          assert(page->records[left_idx].ptr == left_child_val);
          left_update_addr = &page->records[left_idx];
        }
      }
    }
    int new_gran = cur_gran;
    for (; new_gran >= gran_full; --new_gran) {
      // update k, not found key, can't be the last one of previous group
      if (new_gran == gran_quarter) {
        begin_idx = kGroupCardinality * group_id;
        max_cnt = kGroupCardinality;
      } else if (new_gran == gran_half) {
        begin_idx = group_id < 2 ? 0 : kGroupCardinality * 2;
        max_cnt = kGroupCardinality * 2;
      } else {
        begin_idx = 0;
        max_cnt = kInternalCardinality;
      }

      int empty_idx = page->find_empty(begin_idx, max_cnt);
      if (empty_idx != -1) {
        insert_addr = page->records + empty_idx;
        break;
      }
    }

    assert(left_child_val == GlobalAddress::Null() ||
          left_update_addr != nullptr);

    if (insert_addr) {  // has empty slot
      uint64_t *mask_buffer = rbuf.get_cas_buffer();
      mask_buffer[0] = mask_buffer[1] = ~0ull;

      int last_idx = begin_idx + max_cnt - 1;  // the largest in group
      if (k > page->records[last_idx].key) {
        // swap k to current rightmost
        uint64_t *old_buffer = rbuf.get_cas_buffer();
        memcpy(old_buffer, insert_addr, sizeof(InternalEntry));
        insert_addr->key = page->records[last_idx].key;
        if (left_update_addr && left_update_addr == &page->records[last_idx]) {
          // if current rightmost is the left_child
          assert(page->records[last_idx].key == left_child);
          insert_addr->ptr = left_child_val;
          left_update_addr = nullptr; // already update left_child
        } else {
          insert_addr->ptr = page->records[last_idx].ptr;
        }
        uint64_t *cas_ret_buffer = rbuf.get_cas_buffer();
        // lock-based, must cas succeed
  #if KEY_SIZE == 8
        dsm_client_->CasMask(GADD(page_addr, ((char *)insert_addr - page_buffer)),
                            4, (uint64_t)old_buffer, (uint64_t)insert_addr,
                            cas_ret_buffer, (uint64_t)mask_buffer, false);
  #else
        dsm_client_->Write(
            (char *)&insert_addr->key,
            GADD(page_addr, (char *)&insert_addr->key - page_buffer),
            sizeof(InternalKey), false);
        InternalEntry *old_entry = (InternalEntry *)old_buffer;
        dsm_client_->Cas(GADD(page_addr, (char *)&insert_addr->ptr - page_buffer),
                        old_entry->ptr.raw, insert_addr->ptr.raw, cas_ret_buffer,
                        false);
  #endif
        insert_addr = page->records + last_idx;  // update insert_addr
      }

      uint64_t *cas_ret_buffer = rbuf.get_cas_buffer();
  #if KEY_SIZE == 8
      uint64_t *old_buffer = rbuf.get_cas_buffer();
      memcpy(old_buffer, insert_addr, sizeof(InternalEntry));
      insert_addr->key = k;
      insert_addr->ptr = v;
      dsm_client_->CasMask(GADD(page_addr, ((char *)insert_addr - page_buffer)),
                          4, (uint64_t)old_buffer, (uint64_t)insert_addr,
                          cas_ret_buffer, (uint64_t)mask_buffer, false);
  #else
      insert_addr->key = k;
      insert_addr->ptr = GlobalAddress::Null();
      dsm_client_->Write((char *)insert_addr,
                        GADD(page_addr, ((char *)insert_addr - page_buffer)),
                        sizeof(InternalEntry), false);
      GlobalAddress new_ptr;
      new_ptr.raw = insert_addr->ptr.raw;
      new_ptr = v;
      // bool cas_ok = dsm_client_->CasSync(
      //     GADD(page_addr, (char *)&insert_addr->ptr - page_buffer),
      //     insert_addr->ptr.raw, new_ptr.raw, cas_ret_buffer, ctx);
      // assert(cas_ok);
      dsm_client_->Cas(GADD(page_addr, (char *)&insert_addr->ptr - page_buffer),
                      insert_addr->ptr.raw, new_ptr.raw, cas_ret_buffer, false);
  #endif

      if (left_update_addr) {
        uint64_t *cas_ret_buffer = rbuf.get_cas_buffer();
        uint64_t old = left_update_addr->ptr.raw;
        left_update_addr->ptr = left_child_val;
        dsm_client_->Cas(
            GADD(page_addr, ((char *)&(left_update_addr->ptr) - page_buffer)),
            old, left_update_addr->ptr.raw, cas_ret_buffer, false);
      }

      if (new_gran != cur_gran) {
        // update header and parent ptr;
        page->hdr.read_gran = new_gran;
        page_addr.read_gran = new_gran;
        write_and_unlock(page_buffer + offsetof(InternalPage, hdr),
                        GADD(page_addr, offsetof(InternalPage, hdr)),
                        sizeof(Header), lock_buffer, lock_addr, ctx, true,
                        false);

        GlobalAddress up_level = path_stack[ctx ? ctx->coro_id : 0][level + 1];
        if (up_level != GlobalAddress::Null()) {
          internal_page_update(up_level, page->hdr.lowest, page_addr, level + 1,
                              ctx, true);
        } else {
          update_ptr_internal(page->hdr.lowest, page_addr, ctx, level + 1);
        }
      } else {
        release_lock(lock_addr, lock_buffer, ctx, true, false);
      }
      if (level == 1 && enable_cache) {
        index_cache->add_to_cache(page, dsm_client_->get_my_thread_id());
      }
    } else {
      // need split and insert
      GlobalAddress sibling_addr = dsm_client_->Alloc(kInternalPageSize);
      char *sibling_buf = rbuf.get_sibling_buffer();
      InternalPage *sibling = new (sibling_buf) InternalPage(page->hdr.level);
      sibling->hdr.my_addr = sibling_addr;

      if (left_update_addr) {
        left_update_addr->ptr = left_child_val;
      }
      std::vector<InternalEntry> tmp_records(
          page->records, page->records + kInternalCardinality);
      tmp_records.push_back({.ptr = v, .key = k});
      std::sort(tmp_records.begin(), tmp_records.end());
      int m = kInternalCardinality / 2;
      Key split_key = tmp_records[m].key;
      GlobalAddress split_val = tmp_records[m].ptr;
      int sib_gran = sibling->rearrange_records(tmp_records.data() + m + 1,
                                                kInternalCardinality - m,
                                                split_key, page->hdr.highest);
      sibling->hdr.sibling_ptr = page->hdr.sibling_ptr;
      sibling->hdr.leftmost_ptr = split_val;
      sibling->hdr.read_gran = sib_gran;
      sibling_addr.read_gran = sib_gran;
      sibling_addr.node_version = sibling->hdr.version;

      page_addr.read_gran = page->rearrange_records(tmp_records.data(), m,
                                                    page->hdr.lowest, split_key);
      page_addr.node_version = page->update_version();
      page->hdr.sibling_ptr = sibling_addr;

      if (root == page_addr) {
        page->hdr.is_root = false;
      }

      if (sibling_addr.nodeID == page_addr.nodeID) {
        dsm_client_->Write(sibling_buf, sibling_addr, kInternalPageSize, false);
      } else {
        dsm_client_->WriteSync(sibling_buf, sibling_addr, kInternalPageSize, ctx);
      }
      
      write_and_unlock(page_buffer, page_addr, kInternalPageSize, lock_buffer,
                      lock_addr, ctx, true, false);
      if (root == page_addr) {
        if (update_new_root(page_addr, split_key, sibling_addr, level + 1, root,
                            ctx)) {
          return;
        }
      }

      GlobalAddress up_level = path_stack[ctx ? ctx->coro_id : 0][level + 1];
      if (up_level != GlobalAddress::Null()) {
        internal_page_store_update_left_child(up_level, split_key, sibling_addr,
                                              page->hdr.lowest, page_addr, root,
                                              level + 1, ctx);
      } else {
        insert_internal_update_left_child(
            split_key, sibling_addr, page->hdr.lowest, page_addr, ctx, level + 1);
      }
      if (level == 1 && enable_cache) {
        int my_thread_id = dsm_client_->get_my_thread_id();
        index_cache->add_to_cache(page, my_thread_id);
        index_cache->add_to_cache(sibling, my_thread_id);
        index_cache->invalidate(page->hdr.lowest, sibling->hdr.highest,
                                my_thread_id);
      }
    }
  }

  bool Tree::leaf_page_store(GlobalAddress page_addr, const Key &k,
                            const Value &v, GlobalAddress root, int level,
                            CoroContext *ctx, bool from_cache, bool share_lock) {
    GlobalAddress lock_addr = get_lock_addr(page_addr);
    uint16_t tid = dsm_client_->get_my_thread_id();

    auto &rbuf = dsm_client_->get_rbuf(ctx ? ctx->coro_id : 0);
    uint64_t *lock_buffer = rbuf.get_cas_buffer();
    uint64_t *cas_ret_buffer = rbuf.get_cas_buffer();
    char *page_buffer = rbuf.get_page_buffer();

    [[maybe_unused]] bool upgrade_from_s = false;
    int bucket_id = key_hash_bucket(k);
    int group_id = bucket_id / 2;
    Header *header = (Header *)(page_buffer + offsetof(LeafPage, hdr));
    LeafEntry *update_addr = nullptr;
    LeafEntry *insert_addr = nullptr;
    int group_offset =
        offsetof(LeafPage, groups) + sizeof(LeafEntryGroup) * group_id;
    LeafEntryGroup *group = (LeafEntryGroup *)(page_buffer + group_offset);
    bool hold_x_lock = false;
    uint8_t actual_version = 0;

    auto write_and_unlock_leaf_x = [&](char *write_buffer, GlobalAddress write_addr,
                                      int write_size, bool async) {
      dsm_client_->Write(write_buffer, write_addr, write_size, false);
      track_rdma(dsm_client_->get_my_thread_id(), 1, write_size);
      release_sx_lock(lock_addr, lock_buffer, ctx, async, false);
    };

    auto cas_and_unlock_leaf_x = [&](GlobalAddress cas_addr, int log_cas_size,
                                    uint64_t equal, uint64_t swap, bool async) {
      dsm_client_->CasMask(cas_addr, log_cas_size, equal, swap, cas_ret_buffer,
                          ~0ull, false);
      track_rdma(dsm_client_->get_my_thread_id(), 1, (1 << log_cas_size));
      release_sx_lock(lock_addr, lock_buffer, ctx, async, false);
    };
    // try upsert hash group
#ifdef FINE_GRAINED_LEAF_NODE
    size_t bucket_offset =
        group_offset + (bucket_id % 2 ? kBackOffset : kFrontOffset);
#ifdef USE_OPTIMISTIC_UPDATE_HIT
    bool optimistic_bucket_consistent = false;
    bool optimistic_update_candidate = false;
    constexpr int kOptimisticUpdateRetry = 2;
    bool bypass_optimistic_leaf_fast_path =
        should_bypass_optimistic_leaf_fast_path(page_addr);
    if (bypass_optimistic_leaf_fast_path) {
      optimistic_leaf_fast_path_hot_bypass_cnt[tid]++;
    }

    for (int optimistic_retry = 0;
        !bypass_optimistic_leaf_fast_path &&
        optimistic_retry < kOptimisticUpdateRetry;
        ++optimistic_retry) {
      dsm_client_->ReadSync(page_buffer + bucket_offset,
                            GADD(page_addr, bucket_offset), kReadBucketSize, ctx);
      track_rdma(tid, 1, kReadBucketSize);

      if (!group->check_consistency(!(bucket_id % 2), page_addr.node_version,
                                    actual_version)) {
        optimistic_insert_consistency_fail_cnt[tid]++;
        if (from_cache) {
          return false;
        }
        break;
      }

      optimistic_bucket_consistent = true;
      update_addr = nullptr;
      insert_addr = nullptr;
      group->find(k, !(bucket_id % 2), &update_addr, &insert_addr);
      optimistic_update_candidate = (update_addr != nullptr);

      if (!optimistic_update_candidate) {
        // try optimistic empty-slot insert before taking a leaf lock
        if (insert_addr != nullptr) {
  #if KEY_SIZE == 8
          optimistic_insert_attempt_cnt[tid]++;
          uint64_t *swap_buffer = rbuf.get_cas_buffer();
          LeafEntry *swap_entry = reinterpret_cast<LeafEntry *>(swap_buffer);
          swap_entry->key = k;
          swap_entry->lv.cl_ver = insert_addr->lv.cl_ver;
          swap_entry->lv.val = v;

          uint64_t *mask_buffer = rbuf.get_cas_buffer();
          mask_buffer[0] = mask_buffer[1] = ~0ull;
          bool cas_ok = dsm_client_->CasMaskSync(
              GADD(page_addr, reinterpret_cast<char *>(insert_addr) -
                                  page_buffer),
              4, reinterpret_cast<uint64_t>(insert_addr),
              reinterpret_cast<uint64_t>(swap_buffer), cas_ret_buffer,
              reinterpret_cast<uint64_t>(mask_buffer), ctx);
          track_rdma(tid, 1, sizeof(LeafEntry));
          if (cas_ok || (__bswap_64(cas_ret_buffer[0]) == k)) {
            optimistic_insert_success_cnt[tid]++;
            leaf_insert_empty_cnt[tid]++;
            clear_hot_leaf_fast_path_conflict(page_addr);
            return true;
          }
          optimistic_insert_cas_fail_cnt[tid]++;
          mark_hot_leaf_fast_path_conflict(page_addr);
          insert_addr->key = __bswap_64(cas_ret_buffer[0]);
          insert_addr->lv.raw = __bswap_64(cas_ret_buffer[1]);
          continue;
  #endif
        }
        optimistic_insert_fallback_cnt[tid]++;
        break;
      }

      optimistic_update_attempt_cnt[tid]++;

      LeafValue cas_val(update_addr->lv.cl_ver, v);
      bool cas_ok = dsm_client_->CasSync(
          GADD(page_addr, ((char *)&update_addr->lv - page_buffer)),
          update_addr->lv.raw, cas_val.raw, cas_ret_buffer, ctx);
      track_rdma(tid, 1, sizeof(LeafValue));
      if (!cas_ok) {
        optimistic_update_cas_fail_cnt[tid]++;
        mark_hot_leaf_fast_path_conflict(page_addr);
        continue;
      }
      clear_hot_leaf_fast_path_conflict(page_addr);
      optimistic_update_success_cnt[tid]++;
      leaf_update_hit_cnt[tid]++;
      return true;
    }
#endif
    // 1. lock, read, and check consistency
    lock_and_read(lock_addr, share_lock, false, lock_buffer,
                  GADD(page_addr, bucket_offset), kReadBucketSize,
                  page_buffer + bucket_offset, ctx);
    hold_x_lock = !share_lock;
    leaf_insert_path_page_lock_cnt[tid]++;

    if (!group->check_consistency(!(bucket_id % 2), page_addr.node_version,
                                  actual_version)) {
      if (from_cache) {
        release_lock(lock_addr, lock_buffer, ctx, true, share_lock);
        return false;
      } else {
        // lock-based, no need to re-read, just read header to check sibling
        dsm_client_->ReadSync(page_buffer + offsetof(LeafPage, hdr),
                              GADD(page_addr, offsetof(LeafPage, hdr)),
                              sizeof(Header), ctx);
        if (k >= header->highest) {
          leaf_sibling_chase_cnt[tid]++;
          release_lock(lock_addr, lock_buffer, ctx, true, share_lock);
          return leaf_page_store(header->sibling_ptr, k, v, root, level, ctx,
                                false, share_lock);
        }
      }
    }

    // 2. try update
    // 2.1 check main bucket
    int retry_cnt = 0;
  retry_insert:
    update_addr = nullptr;
    insert_addr = nullptr;
    group->find(k, !(bucket_id % 2), &update_addr, &insert_addr);
#ifdef USE_LEAF_STASH
    if (!update_addr && insert_addr == nullptr) {
      dsm_client_->ReadSync(page_buffer + offsetof(LeafPage, hdr),
                            GADD(page_addr, offsetof(LeafPage, hdr)),
                            sizeof(Header), ctx);
      track_rdma(tid, 1, sizeof(Header));
      LeafPage *page = reinterpret_cast<LeafPage *>(page_buffer);
      size_t stash_offset = offsetof(LeafPage, stash);
      dsm_client_->ReadSync(page_buffer + stash_offset,
                            GADD(page_addr, stash_offset),
                            sizeof(LeafEntry) * kLeafStashSlots, ctx);
      track_rdma(tid, 1, sizeof(LeafEntry) * kLeafStashSlots);
      if (leaf_stash_may_have(page, group_id)) {
        update_addr = find_leaf_stash_entry(page, k, group_id);
      }
      if (update_addr == nullptr &&
          try_optimistic_leaf_stash_insert(dsm_client_, page_addr, page,
                                           page_buffer, k, v, group_id, rbuf,
                                           ctx, tid)) {
        release_lock(lock_addr, lock_buffer, ctx, true, share_lock);
        return true;
      }
    }
#endif
    if (update_addr) {
      leaf_update_hit_cnt[tid]++;
      LeafValue cas_val(update_addr->lv.cl_ver, v);
      cas_and_unlock(GADD(page_addr, ((char *)&update_addr->lv - page_buffer)),
                    3, cas_ret_buffer, update_addr->lv.raw, cas_val.raw, ~0ull,
                    lock_addr, lock_buffer, share_lock, ctx, false);
      return true;
    } else if (insert_addr) {
      // 3. try insert via CAS
  #if KEY_SIZE == 8
      uint64_t *swap_buffer = rbuf.get_cas_buffer();
      LeafEntry *swap_entry = (LeafEntry *)swap_buffer;
      swap_entry->key = k;
      swap_entry->lv.cl_ver = insert_addr->lv.cl_ver;
      swap_entry->lv.val = v;
      uint64_t *mask_buffer = rbuf.get_cas_buffer();
      mask_buffer[0] = mask_buffer[1] = ~0ull;
      bool cas_ok = dsm_client_->CasMaskSync(
          GADD(page_addr, ((char *)insert_addr - page_buffer)), 4,
          (uint64_t)insert_addr, (uint64_t)swap_buffer, cas_ret_buffer,
          (uint64_t)mask_buffer, ctx);
      // cas succeed or same key inserted by other thread
      if (cas_ok || (__bswap_64(cas_ret_buffer[0]) == k)) {
        leaf_insert_empty_cnt[tid]++;
        release_lock(lock_addr, lock_buffer, ctx, true, share_lock);
        return true;
      }
      // 3.1 retry insert
      // big-endian for 16-byte CAS
      insert_addr->key = __bswap_64(cas_ret_buffer[0]);
      insert_addr->lv.raw = __bswap_64(cas_ret_buffer[1]);
  #else
      LeafValue cas_val(insert_addr->lv.cl_ver, -1);
      bool cas_ok = dsm_client_->CasSync(
          GADD(page_addr, ((char *)&insert_addr->lv - page_buffer)),
          insert_addr->lv.raw, cas_val.raw, cas_ret_buffer, ctx);
      if (cas_ok) {
        leaf_insert_empty_cnt[tid]++;
        insert_addr->key = k;
        insert_addr->lv.val = -1;
        dsm_client_->Write(
            (char *)&insert_addr->key,
            GADD(page_addr, (char *)&insert_addr->key - page_buffer),
            sizeof(InternalKey), false);
        cas_val.val = v;
        cas_and_unlock(GADD(page_addr, ((char *)&insert_addr->lv - page_buffer)),
                      3, cas_ret_buffer, insert_addr->lv.raw, cas_val.raw, ~0ull,
                      lock_addr, lock_buffer, share_lock, ctx, false);
        return true;
      }
      insert_addr->lv.raw = cas_ret_buffer[0];
  #endif
      if (++retry_cnt > 10) {
        printf("retry insert %d times\n", retry_cnt);
        assert(false);
      }
      if (retry_cnt == 1) {
        leaf_insert_retry_event_cnt[tid]++;
      }
      leaf_insert_retry_step_cnt[tid]++;
      goto retry_insert;
    }

    // 4. prepare to split
  #ifdef USE_SX_LOCK
    // upgrade to x lock
    if (share_lock) {
      leaf_upgrade_to_x_cnt[tid]++;
    }
    upgrade_from_s = share_lock;
    share_lock = false;
  #endif

  #endif  // FINE_GRAINED_LEAF_NODE

  #ifdef USE_SX_LOCK
  retry_with_xlock:
  #endif
    if (hold_x_lock) {
      dsm_client_->ReadSync(page_buffer, page_addr, kLeafPageSize, ctx);
    } else {
      // not holding lock, or only share lock
      lock_and_read(lock_addr, share_lock, upgrade_from_s, lock_buffer, page_addr,
                    kLeafPageSize, page_buffer, ctx);
      hold_x_lock = !share_lock;
    }
    LeafPage *page = (LeafPage *)page_buffer;

    assert(header->level == level);

    if (k < header->lowest || k >= header->highest ||
        page_addr.node_version != header->version) {  // cache is stale
      // Note: when very slow, may need recurse very large times and cause stack
      // overflow
      release_lock(lock_addr, lock_buffer, ctx, true, share_lock);
      return false;
    }

    // if (k >= header->highest) {
    //   // note that retry may also get here
    //   assert(header->sibling_ptr != GlobalAddress::Null());
    //   release_lock(lock_addr, lock_buffer, ctx, true, share_lock);
    //   leaf_page_store(header->sibling_ptr, k, v, root, level, ctx, from_cache,
    //                   share_lock);
    //   return true;
    // }
    assert(k >= header->lowest);

    if (header->version != page_addr.node_version) {
      page_addr.node_version = header->version;
    }

    // maybe split by others? check again
    // hash-based
  retry_insert_2:
    update_addr = nullptr;
    insert_addr = nullptr;
    group->find(k, !(bucket_id % 2), &update_addr, &insert_addr);
#ifdef USE_LEAF_STASH
    if (!update_addr) {
      update_addr = find_leaf_stash_entry(page, k, group_id);
    }
#endif

    if (update_addr) {
      leaf_update_hit_cnt[tid]++;
      LeafValue cas_val(update_addr->lv.cl_ver, v);
  #ifdef USE_CRC
      Timer t_crc;
      t_crc.begin();
      uint32_t c = page->set_crc();
      uint64_t t = t_crc.end();
      stat_helper.add(dsm_client_->get_my_thread_id(), lat_crc, t);
      stat_helper.add(dsm_client_->get_my_thread_id(), lat_cache_search, c);
      if (share_lock) {
        write_and_unlock(page_buffer, page_addr, kLeafPageSize, lock_buffer,
                        lock_addr, ctx, true, share_lock);
      } else {
        write_and_unlock_leaf_x(page_buffer, page_addr, kLeafPageSize, true);
      }
  #else
      if (share_lock) {
        cas_and_unlock(GADD(page_addr, ((char *)&(update_addr->lv) - page_buffer)),
                      3, cas_ret_buffer, update_addr->lv.raw, cas_val.raw, ~0ull,
                      lock_addr, lock_buffer, share_lock, ctx, true);
      } else {
        cas_and_unlock_leaf_x(
            GADD(page_addr, ((char *)&(update_addr->lv) - page_buffer)), 3,
            update_addr->lv.raw, cas_val.raw, true);
      }
  #endif
      return true;
    } else if (insert_addr) {
  #if KEY_SIZE == 8
      uint64_t *swap_buffer = rbuf.get_cas_buffer();
      LeafEntry *swap_entry = (LeafEntry *)swap_buffer;
      swap_entry->key = k;
      swap_entry->lv.cl_ver = insert_addr->lv.cl_ver;
      swap_entry->lv.val = v;
      uint64_t *mask_buffer = rbuf.get_cas_buffer();
      mask_buffer[0] = mask_buffer[1] = ~0ull;
      bool cas_ok = dsm_client_->CasMaskSync(
          GADD(page_addr, ((char *)insert_addr - page_buffer)), 4,
          (uint64_t)insert_addr, (uint64_t)swap_buffer, cas_ret_buffer,
          (uint64_t)mask_buffer, ctx);
      // cas succeed or same key inserted by other thread
      if (cas_ok || (__bswap_64(cas_ret_buffer[0]) == k)) {
        leaf_insert_empty_cnt[tid]++;
        release_lock(lock_addr, lock_buffer, ctx, true, share_lock);
        return true;
      }
      insert_addr->key = __bswap_64(cas_ret_buffer[0]);
      insert_addr->lv.raw = __bswap_64(cas_ret_buffer[1]);
  #else
      LeafValue cas_val(insert_addr->lv.cl_ver, -1);
      bool cas_ok = dsm_client_->CasSync(
          GADD(page_addr, ((char *)&insert_addr->lv - page_buffer)),
          insert_addr->lv.raw, cas_val.raw, cas_ret_buffer, ctx);
      if (cas_ok) {
        leaf_insert_empty_cnt[tid]++;
        insert_addr->key = k;
        insert_addr->lv.val = -1;
        dsm_client_->Write(
            (char *)&insert_addr->key,
            GADD(page_addr, (char *)&insert_addr->key - page_buffer),
            sizeof(InternalKey), false);
        cas_val.val = v;
        if (share_lock) {
          cas_and_unlock(GADD(page_addr, ((char *)&insert_addr->lv - page_buffer)),
                        3, cas_ret_buffer, insert_addr->lv.raw, cas_val.raw,
                        ~0ull, lock_addr, lock_buffer, share_lock, ctx, true);
        } else {
          cas_and_unlock_leaf_x(
              GADD(page_addr, ((char *)&insert_addr->lv - page_buffer)), 3,
              insert_addr->lv.raw, cas_val.raw, true);
        }
        return true;
      }
      insert_addr->lv.raw = cas_ret_buffer[0];
  #endif
      leaf_insert_retry_event_cnt[tid]++;
      leaf_insert_retry_step_cnt[tid]++;
      assert(share_lock);
      goto retry_insert_2;
    }

    // should hold x lock
  #ifdef USE_SX_LOCK
    if (share_lock) {
      leaf_upgrade_to_x_cnt[tid]++;
      upgrade_from_s = true;
      share_lock = false;
      goto retry_with_xlock;
    }
  #endif

#ifdef USE_LEAF_STASH
    leaf_stash_insert_attempt_cnt[tid]++;
    uint64_t page_occupancy = count_leaf_page_occupancy(page);
    if (page_occupancy * 100 <
        kLeafCardinality * kLeafStashSplitLoadPercent) {
      if (insert_leaf_stash_entry(page, k, v, group_id)) {
        leaf_stash_insert_success_cnt[tid]++;
        leaf_insert_empty_cnt[tid]++;
        write_and_unlock_leaf_x(page_buffer, page_addr, kLeafPageSize, false);
        return true;
      }
      leaf_stash_insert_full_cnt[tid]++;
    } else {
      leaf_stash_insert_threshold_cnt[tid]++;
    }
#endif

    // split
    leaf_split_cnt[tid]++;
    GlobalAddress sibling_addr;
    sibling_addr = dsm_client_->Alloc(kLeafPageSize);
    char *sibling_buf = rbuf.get_sibling_buffer();
    LeafPage *sibling = new (sibling_buf) LeafPage(page->hdr.level);

    LeafKVEntry tmp_records[kLeafCardinality];
    int cnt = 0;
    for (int i = 0; i < kNumGroup; ++i) {
      LeafEntryGroup *g = &page->groups[i];
      for (int j = 0; j < kAssociativity; ++j) {
        if (g->front[j].lv.val != kValueNull) {
          tmp_records[cnt++] = g->front[j];
        }
        if (g->back[j].lv.val != kValueNull) {
          tmp_records[cnt++] = g->back[j];
        }
      }
      for (int j = 0; j < kGroupOverflowSlots; ++j) {
        if (g->overflow[j].lv.val != kValueNull) {
          tmp_records[cnt++] = g->overflow[j];
        }
      }
    }
#ifdef USE_LEAF_STASH
    uint64_t split_stash_occupancy = 0;
    split_stash_occupancy = count_leaf_stash_occupancy(page);
    for (int i = 0; i < kLeafStashSlots; ++i) {
      LeafEntry *stash_entry = page->stash_entry(i);
      if (stash_entry->lv.val != kValueNull) {
        tmp_records[cnt++] = *stash_entry;
      }
    }
    leaf_split_stash_occupancy_sum[tid] += split_stash_occupancy;
    if (split_stash_occupancy > leaf_split_stash_occupancy_max[tid]) {
      leaf_split_stash_occupancy_max[tid] = split_stash_occupancy;
    }
#endif
    leaf_split_occupancy_sum[tid] += cnt;
    if ((uint64_t)cnt > leaf_split_occupancy_max[tid]) {
      leaf_split_occupancy_max[tid] = cnt;
    }
    if (g_prefill_split_stats_enabled.load(std::memory_order_relaxed)) {
      uint64_t bucket_occupancy =
          count_bucket_occupancy(group, !(bucket_id % 2));
      uint64_t group_occupancy = count_group_occupancy(group);
      prefill_leaf_split_cnt[tid]++;
      prefill_leaf_split_page_occupancy_sum[tid] += cnt;
      prefill_leaf_split_bucket_occupancy_sum[tid] += bucket_occupancy;
      prefill_leaf_split_group_occupancy_sum[tid] += group_occupancy;
      if ((uint64_t)cnt > prefill_leaf_split_page_occupancy_max[tid]) {
        prefill_leaf_split_page_occupancy_max[tid] = cnt;
      }
      if (bucket_occupancy > prefill_leaf_split_bucket_occupancy_max[tid]) {
        prefill_leaf_split_bucket_occupancy_max[tid] = bucket_occupancy;
      }
      if (group_occupancy > prefill_leaf_split_group_occupancy_max[tid]) {
        prefill_leaf_split_group_occupancy_max[tid] = group_occupancy;
      }
    }
    std::sort(tmp_records, tmp_records + cnt);

    int m = cnt / 2;
    Key split_key = tmp_records[m].key;
    assert(split_key > page->hdr.lowest);
    assert(split_key < page->hdr.highest);

    memset(reinterpret_cast<void *>(page->groups), 0, sizeof(page->groups));
#ifdef USE_LEAF_STASH
    memset(reinterpret_cast<void *>(page->stash_entry(0)), 0,
           sizeof(LeafEntry) * kLeafStashSlots);
    page->set_stash_group_mask(0);
    page->set_stash_count(0);
    sibling->set_stash_group_mask(0);
    sibling->set_stash_count(0);
#endif
    for (int i = 0; i < m; ++i) {
      bool ok =
          insert_leaf_entry_for_rebuild(page, tmp_records[i].key,
                                        tmp_records[i].val);
      assert(ok);
    }
    for (int i = m; i < cnt; ++i) {
      bool ok =
          insert_leaf_entry_for_rebuild(sibling, tmp_records[i].key,
                                        tmp_records[i].val);
      assert(ok);
    }

    sibling->hdr.lowest = split_key;
    sibling->hdr.highest = page->hdr.highest;
    page->hdr.highest = split_key;

    page_addr.node_version = page->update_version();
    sibling_addr.node_version = sibling->hdr.version;

    // link
    sibling->hdr.sibling_ptr = page->hdr.sibling_ptr;
    page->hdr.sibling_ptr = sibling_addr;

    // insert k
    bool res;
    if (k < split_key) {
      res = insert_leaf_entry_for_rebuild(page, k, v);
    } else {
      res = insert_leaf_entry_for_rebuild(sibling, k, v);
    }

    if (sibling_addr.nodeID == page_addr.nodeID) {
      dsm_client_->Write(sibling_buf, sibling_addr, kLeafPageSize, false);
    } else {
      dsm_client_->WriteSync(sibling_buf, sibling_addr, kLeafPageSize, ctx);
    }
    track_rdma(dsm_client_->get_my_thread_id(), 1, kLeafPageSize); // 记录写新 Page 报文
    if (root == page_addr) {
      page->hdr.is_root = false;
    }

    write_and_unlock_leaf_x(page_buffer, page_addr, kLeafPageSize, false);

    if (root == page_addr) {  // update root
      leaf_insert_root_split_cnt[tid]++;
      if (update_new_root(page_addr, split_key, sibling_addr, level + 1, root,
                          ctx)) {
        return res;
      }
    }

    GlobalAddress up_level = path_stack[ctx ? ctx->coro_id : 0][level + 1];

    if (up_level != GlobalAddress::Null()) {
      leaf_insert_parent_update_cnt[tid]++;
      internal_page_store_update_left_child(up_level, split_key, sibling_addr,
                                            page->hdr.lowest, page_addr, root,
                                            level + 1, ctx);
    } else {
      assert(false);
      insert_internal_update_left_child(split_key, sibling_addr, page->hdr.lowest,
                                        page_addr, ctx, level + 1);
    }

    return res;
  }

  bool Tree::leaf_page_del(GlobalAddress page_addr, const Key &k, int level,
                          CoroContext *ctx, int coro_id, bool from_cache) {
    GlobalAddress lock_addr = get_lock_addr(page_addr);

    auto &rbuf = dsm_client_->get_rbuf(coro_id);
    uint64_t *cas_buffer = rbuf.get_cas_buffer();
    auto page_buffer = rbuf.get_page_buffer();

    lock_and_read(lock_addr, false, false, cas_buffer, page_addr, kLeafPageSize,
                  page_buffer, ctx);

    auto page = (LeafPage *)page_buffer;

    assert(page->hdr.level == level);

    if (from_cache &&
        (k < page->hdr.lowest || k >= page->hdr.highest)) {  // cache is stale
      release_lock(lock_addr, cas_buffer, ctx, true, false);
      return false;
    }

    if (k >= page->hdr.highest) {
      release_lock(lock_addr, cas_buffer, ctx, true, false);
      assert(page->hdr.sibling_ptr != GlobalAddress::Null());
      this->leaf_page_del(page->hdr.sibling_ptr, k, level, ctx, coro_id);
      return true;
    }

    assert(k >= page->hdr.lowest);

    LeafEntry *update_addr = nullptr;
    int bucket_id = key_hash_bucket(k);
    LeafEntryGroup *g = &page->groups[bucket_id / 2];
    if (bucket_id % 2) {
      // back
      for (int i = 0; i < kAssociativity; ++i) {
        LeafEntry *p = &g->back[i];
        if (p->key == k) {
          p->lv.val = kValueNull;
          update_addr = p;
          break;
        }
      }
    } else {
      // front
      for (int i = 0; i < kAssociativity; ++i) {
        LeafEntry *p = &g->front[i];
        if (p->key == k) {
          p->lv.val = kValueNull;
          update_addr = p;
          break;
        }
      }
    }

    // overflow
    if (update_addr == nullptr) {
      for (int i = 0; i < kGroupOverflowSlots; ++i) {
        LeafEntry *p = &g->overflow[i];
        if (p->key == k) {
          p->lv.val = kValueNull;
          update_addr = p;
          break;
        }
      }
    }
    if (update_addr == nullptr) {
#ifdef USE_LEAF_STASH
      update_addr = find_leaf_stash_entry(page, k, bucket_id / 2);
#endif
    }

    if (update_addr) {
#ifdef USE_LEAF_STASH
      bool was_stash_entry =
          ((char *)update_addr >= (char *)page->stash_entry(0) &&
           (char *)update_addr <
               (char *)page->stash_entry(0) +
                   sizeof(LeafEntry) * kLeafStashSlots);
      if (was_stash_entry) {
        update_addr->lv.val = kValueNull;
        refresh_leaf_stash_metadata(page);
        dsm_client_->Write(page_buffer, page_addr, kLeafPageSize, false);
        track_rdma(dsm_client_->get_my_thread_id(), 1, kLeafPageSize);
        release_lock(lock_addr, cas_buffer, ctx, false, false);
        return true;
      }
#endif
      dsm_client_->Write((char *)update_addr,
                        GADD(page_addr, ((char *)update_addr - (char *)page)),
                        sizeof(LeafEntry), false);
      track_rdma(dsm_client_->get_my_thread_id(), 1, sizeof(LeafEntry));
      release_lock(lock_addr, cas_buffer, ctx, false, false);
    } else {
      this->release_lock(lock_addr, cas_buffer, ctx, false, false);
    }
    return true;
  }

  void Tree::run_coroutine(CoroFunc func, int id, int coro_cnt, bool lock_bench,
                          uint64_t total_ops) {
    using namespace std::placeholders;
    coro_ops_total = total_ops;
    coro_ops_cnt_start = 0;
    coro_ops_cnt_finish = 0;

    assert(coro_cnt <= define::kMaxCoro);
    for (int i = 0; i < coro_cnt; ++i) {
      auto gen = func(i, dsm_client_, id);
      worker[i] =
          CoroCall(std::bind(&Tree::coro_worker, this, _1, gen, i, lock_bench));
    }

    master = CoroCall(std::bind(&Tree::coro_master, this, _1, coro_cnt));

    master();
  }

  void Tree::coro_worker(CoroYield &yield, RequstGen *gen, int coro_id,
                        bool lock_bench) {
    CoroContext ctx;
    ctx.coro_id = coro_id;
    ctx.master = &master;
    ctx.yield = &yield;

    Timer coro_timer;
    auto thread_id = dsm_client_->get_my_thread_id();

    // while (true) {
    while (coro_ops_cnt_start < coro_ops_total) {
      auto r = gen->next();

      coro_timer.begin();
      ++coro_ops_cnt_start;
  #ifdef ENABLE_MICROBENCH
      // ==========================================
      // 物理极限压测：绕过 B+ 树，直接敲底层网卡！
      // ==========================================
      this->microbench_op(&ctx, coro_id, r.is_search, r.k);
  #else

      if (lock_bench) {
        this->lock_bench(r.k, &ctx, coro_id);
      } else {
        if (r.is_search) {
          Value v;
          tracking_mode = 2; // 【新增】：进入 Search 追踪模式
          this->search(r.k, v, &ctx);
          tracking_mode = 0; // 退出追踪
          
          search_op_cnt[thread_id][0]++; // 记录一次完成的 Search
        } else {
          tracking_mode = 1; // 【新增】：进入 Insert 追踪模式
          this->insert(r.k, r.v, &ctx);
          tracking_mode = 0; // 退出追踪
          
          insert_op_cnt[thread_id][0]++; // 记录一次完成的 Insert
        }
      }
  #endif
      auto t = coro_timer.end();
      auto us_10 = t / 100;
      if (us_10 >= LATENCY_WINDOWS) {
        us_10 = LATENCY_WINDOWS - 1;
      }
      latency[thread_id][us_10]++;
      stat_helper.add(thread_id, lat_op, t);
      ++coro_ops_cnt_finish;
    }
    // printf("thread %d coro_id %d start %lu finish %lu\n",
    //        dsm_client_->get_my_thread_id(), coro_id, coro_ops_cnt_start,
    //        coro_ops_cnt_finish);
    // fflush(stdout);
    yield(master);
  }

  void Tree::coro_master(CoroYield &yield, int coro_cnt) {
  for (int i = 0; i < coro_cnt; ++i) {
    yield(worker[i]);
  }
  #ifdef USE_DOORBELL_BATCHING
    dsm_client_->FlushDoorbell();
  #endif
    uint64_t poll_total = 0;
    uint64_t poll_hit = 0;
    uint64_t total_poll_time_ns = 0;  // 累计轮询时间（纳秒）
  #ifdef USE_AP
    // 配置参数
    int spin_gap = 8;           // 初始比较小
    const int SPIN_MIN = 1;
    const int MISS_THRESHOLD = 16; // 超过这个 miss 次数就进入事件驱动
    
    // 核心融合 1：引入批量大小定义与就绪数组
    const int BATCH_CQ_SIZE = 16;
    uint64_t ready_coros[BATCH_CQ_SIZE];

    int idle_counter = 0;
    int miss_counter = 0;

    while (coro_ops_cnt_finish < coro_ops_total) {
      while (dsm_client_->get_pending_event_count() == 0) {}
      if (++idle_counter < spin_gap) {
        _mm_pause();
        continue;
      }
      idle_counter = 0;

      ++poll_total;
      
      // 核心融合 2：使用 PollRdmaCqBatch 替换 PollRdmaCqOnce
      int n = dsm_client_->PollRdmaCqBatch(BATCH_CQ_SIZE, ready_coros);
      
      if (n > 0) {
        poll_hit += n;
        miss_counter = 0;

        // 核心融合 3：集中唤醒所有就绪的工作协程
        for (int i = 0; i < n; ++i) {
          yield(worker[ready_coros[i]]);
          // 每次唤醒协程处理完后，对应扣减一个 pending event
          dsm_client_->decrease_pending_event(); 
        }
        
  #ifdef USE_DOORBELL_BATCHING
        // 核心融合 4：循环结束后，统一 Flush 门铃，将刚才积压的多个 RDMA 请求一次性打入网卡！
        dsm_client_->FlushDoorbell();
  #endif

        // 命中 → 乘法减小自旋间隔
        spin_gap >>= 1;
        if (spin_gap < SPIN_MIN) spin_gap = SPIN_MIN;

      } else {
        // 未命中 → 线性增加
        spin_gap += 8;

        ++miss_counter;
        if (miss_counter > MISS_THRESHOLD) {
          // === 进入事件驱动模式 ===
          struct ibv_cq *ev_cq;
          void *ev_ctx;
          if (ibv_get_cq_event(dsm_client_->get_comp_channel(),
                              &ev_cq, &ev_ctx) == 0) {
            ibv_ack_cq_events(ev_cq, 1);
            ibv_req_notify_cq(ev_cq, 0);

            // 醒来后同样尝试批量 poll 一次
            int ev_n = dsm_client_->PollRdmaCqBatch(BATCH_CQ_SIZE, ready_coros);
            if (ev_n > 0) {
              poll_hit += ev_n;
              for (int i = 0; i < ev_n; ++i) {
                yield(worker[ready_coros[i]]);
                dsm_client_->decrease_pending_event();
              }
              
  #ifdef USE_DOORBELL_BATCHING
              // 事件驱动模式下批量唤醒后，统一 Flush 门铃
              dsm_client_->FlushDoorbell();
  #endif
            }
          }
          // 回到短 spin 模式
          miss_counter = 0;
          spin_gap = SPIN_MIN;
        }
      }
    }
  #elif defined(USE_BATCH_POLL)
  // 核心优化 1：设置合理的批量大小（过小摊薄不够，过大增加尾延迟，16-32是黄金甜点）
    const int BATCH_CQ_SIZE = 16;
    uint64_t ready_coros[BATCH_CQ_SIZE];

    while (coro_ops_cnt_finish < coro_ops_total) {
      if (dsm_client_->get_pending_event_count() == 0) {
        _mm_pause();
        continue;
      }

      ++poll_total;
      
      // 核心优化 2：一次性收割多个网卡完成事件
      int n = dsm_client_->PollRdmaCqBatch(BATCH_CQ_SIZE, ready_coros);
      
      if (n > 0) {
        poll_hit += n; // 或者只 ++poll_hit，看你想要统计物理命中次数还是逻辑命中数
        
        // 核心优化 3：集中唤醒！
        // 减少 master 在无数据时频繁介入，一口气把就绪的协程全跑一遍
        for (int i = 0; i < n; ++i) {
          dsm_client_->decrease_pending_event();
          yield(worker[ready_coros[i]]);
        }
  #ifdef USE_DOORBELL_BATCHING
        dsm_client_->FlushDoorbell();
  #endif
      } else {
        // 核心优化 4：微架构级暂停 (Micro-architectural Pause)
        // 当网卡真的没数据时，不要疯狂 while(true) 空转抢占 CPU 发热
        // _mm_pause 会让 CPU 休息几十个周期，极大降低功耗并让出超线程资源
        _mm_pause(); 
      }
    }
  #else 
    while (coro_ops_cnt_finish < coro_ops_total) {
      uint64_t next_coro_id;

      ++poll_total;
      // auto start = std::chrono::high_resolution_clock::now();
      
      bool poll_result = dsm_client_->PollRdmaCqOnce(next_coro_id);
      
      // 结束计时
      // auto end = std::chrono::high_resolution_clock::now();
      // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
      // total_poll_time_ns += duration.count();
          
      if (poll_result) {
        ++poll_hit;
        yield(worker[next_coro_id]);
      }
    }

  #endif 
    // 计算统计信息
    double hit_rate = (poll_total > 0) 
                      ? (double)poll_hit / (double)poll_total 
                      : 0.0;
    
    // double avg_poll_time_ns = (poll_total > 0)
    //                          ? (double)total_poll_time_ns / (double)poll_total
    //                          : 0.0;
    
    // double avg_poll_time_us = avg_poll_time_ns / 1000.0;
    
    printf("thread %d poll hit rate: %.2f%% (%lu/%lu)\n",
          dsm_client_->get_my_thread_id(),
          hit_rate * 100, poll_hit, poll_total);
    // printf("thread %d average poll time: %.3f us (%.3f ns), total polls: %lu\n",
    //        dsm_client_->get_my_thread_id(),
    //        avg_poll_time_us, avg_poll_time_ns, poll_total);
    fflush(stdout);
  }

  #ifdef USE_LOCAL_LOCK
  // Local Locks
  inline bool Tree::acquire_local_lock(GlobalAddress lock_addr, CoroContext *ctx,
                                      int coro_id) {
    auto &node =
        local_locks[lock_addr.nodeID][lock_addr.offset / define::kLockSize];

    uint64_t lock_val = node.ticket_lock.fetch_add(1);

    uint32_t ticket = lock_val << 32 >> 32;
    uint32_t current = lock_val >> 32;

    while (ticket != current) {  // lock failed

      if (ctx != nullptr) {
        hot_wait_queue.push(coro_id);
        (*ctx->yield)(*ctx->master);
      }

      current = node.ticket_lock.load(std::memory_order_relaxed) >> 32;
    }

    node.hand_time++;

    return node.hand_over;
  }

  inline bool Tree::can_hand_over(GlobalAddress lock_addr) {
    auto &node =
        local_locks[lock_addr.nodeID][lock_addr.offset / define::kLockSize];
    uint64_t lock_val = node.ticket_lock.load(std::memory_order_relaxed);

    uint32_t ticket = lock_val << 32 >> 32;
    uint32_t current = lock_val >> 32;

    if (ticket <= current + 1) {  // no pending locks
      node.hand_over = false;
    } else {
      node.hand_over = node.hand_time < define::kMaxHandOverTime;
    }
    if (!node.hand_over) {
      node.hand_time = 0;
    }

    return node.hand_over;
  }

  inline void Tree::releases_local_lock(GlobalAddress lock_addr) {
    auto &node =
        local_locks[lock_addr.nodeID][lock_addr.offset / define::kLockSize];

    node.ticket_lock.fetch_add((1ull << 32));
  }
  #endif

  void Tree::index_cache_statistics() {
    index_cache->statistics();
    // index_cache->bench();
  }

  void Tree::clear_statistics() {
  for (int i = 0; i < MAX_APP_THREAD; ++i) {
    cache_hit[i][0] = 0;
    cache_miss[i][0] = 0;
    lock_rdma_faa_cnt[i] = 0;
    lock_rdma_read_cnt[i] = 0;
    lock_retry_data_reread_cnt[i] = 0;
    lock_retry_data_reread_bytes[i] = 0;
      leaf_update_hit_cnt[i] = 0;
      leaf_insert_empty_cnt[i] = 0;
      leaf_insert_path_group_fast_cnt[i] = 0;
      leaf_insert_path_page_lock_cnt[i] = 0;
      leaf_insert_retry_event_cnt[i] = 0;
      leaf_insert_retry_step_cnt[i] = 0;
      leaf_upgrade_to_x_cnt[i] = 0;
      leaf_split_cnt[i] = 0;
      leaf_split_occupancy_sum[i] = 0;
      leaf_split_occupancy_max[i] = 0;
      prefill_leaf_split_cnt[i] = 0;
      prefill_leaf_split_page_occupancy_sum[i] = 0;
      prefill_leaf_split_page_occupancy_max[i] = 0;
      prefill_leaf_split_bucket_occupancy_sum[i] = 0;
      prefill_leaf_split_bucket_occupancy_max[i] = 0;
      prefill_leaf_split_group_occupancy_sum[i] = 0;
      prefill_leaf_split_group_occupancy_max[i] = 0;
      leaf_sibling_chase_cnt[i] = 0;
      leaf_insert_parent_update_cnt[i] = 0;
      leaf_insert_root_split_cnt[i] = 0;
      leaf_stash_insert_attempt_cnt[i] = 0;
      leaf_stash_insert_success_cnt[i] = 0;
      leaf_stash_insert_full_cnt[i] = 0;
      leaf_stash_insert_threshold_cnt[i] = 0;
      leaf_split_stash_occupancy_sum[i] = 0;
      leaf_split_stash_occupancy_max[i] = 0;
      optimistic_update_attempt_cnt[i] = 0;
      optimistic_update_success_cnt[i] = 0;
      optimistic_update_cas_fail_cnt[i] = 0;
      optimistic_update_split_abort_cnt[i] = 0;
      optimistic_leaf_fast_path_hot_bypass_cnt[i] = 0;
      optimistic_insert_attempt_cnt[i] = 0;
      optimistic_insert_success_cnt[i] = 0;
      optimistic_insert_cas_fail_cnt[i] = 0;
      optimistic_insert_split_abort_cnt[i] = 0;
      optimistic_insert_consistency_fail_cnt[i] = 0;
      optimistic_insert_fallback_cnt[i] = 0;
      insert_rtt_cnt[i][0] = 0;
      insert_byte_cnt[i][0] = 0;
      insert_op_cnt[i][0] = 0;
      search_rtt_cnt[i][0] = 0;
      search_byte_cnt[i][0] = 0;
      search_op_cnt[i][0] = 0;
      for (int k = 0; k <= 5000; ++k) {
        tries_per_lock[i][k] = 0;
      }
    }
  }
  void Tree::microbench_op(CoroContext *ctx, int coro_id, bool is_search, Key k) {
    auto &rbuf = dsm_client_->get_rbuf(coro_id);
    char *page_buffer = rbuf.get_page_buffer();

    GlobalAddress target_addr;
    target_addr.nodeID = k % dsm_client_->get_server_size(); 
    
    if (is_search) {
      uint64_t max_buckets = MICROBENCH_MEM_LIMIT / kReadBucketSize;
      target_addr.offset = (k % max_buckets) * kReadBucketSize;
      
      // 【必须用 ReadSync】：发请求 -> 挂起当前协程 -> 等待网卡回执 -> 唤醒
      dsm_client_->ReadSync(page_buffer, target_addr, kReadBucketSize, ctx);
    } else {
      // 【改成细粒度 16B 写】：测试网卡极致小包写入性能！
      uint64_t max_entries = MICROBENCH_MEM_LIMIT / sizeof(LeafEntry);
      target_addr.offset = (k % max_entries) * sizeof(LeafEntry);
      
      // 【必须用 WriteSync】：发请求 -> 挂起当前协程 -> 等待网卡回执 -> 唤醒
      dsm_client_->WriteSync(page_buffer, target_addr, sizeof(LeafEntry), ctx);
    }
  }
