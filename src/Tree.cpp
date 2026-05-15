#include "Tree.h"

#include <city.h>
#include <algorithm>
#include <atomic>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>
#include <emmintrin.h>
#include <chrono>

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
uint64_t sxlock_update_cas_fail_cnt[MAX_APP_THREAD] = {0};
uint64_t sxlock_update_cas_retry_event_cnt[MAX_APP_THREAD] = {0};
uint64_t sxlock_update_cas_retry_step_cnt[MAX_APP_THREAD] = {0};
uint64_t sxlock_insert_cas_fail_cnt[MAX_APP_THREAD] = {0};
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
uint64_t optimistic_update_attempt_cnt[MAX_APP_THREAD] = {0};
uint64_t optimistic_update_success_cnt[MAX_APP_THREAD] = {0};
uint64_t optimistic_update_cas_fail_cnt[MAX_APP_THREAD] = {0};
uint64_t optimistic_update_cas_retry_event_cnt[MAX_APP_THREAD] = {0};
uint64_t optimistic_update_cas_retry_step_cnt[MAX_APP_THREAD] = {0};
uint64_t optimistic_update_cas_retry_exhaust_cnt[MAX_APP_THREAD] = {0};
uint64_t optimistic_update_split_abort_cnt[MAX_APP_THREAD] = {0};
uint64_t optimistic_leaf_fast_path_hot_bypass_cnt[MAX_APP_THREAD] = {0};
uint64_t optimistic_insert_attempt_cnt[MAX_APP_THREAD] = {0};
uint64_t optimistic_insert_success_cnt[MAX_APP_THREAD] = {0};
uint64_t optimistic_insert_cas_fail_cnt[MAX_APP_THREAD] = {0};
uint64_t optimistic_insert_split_abort_cnt[MAX_APP_THREAD] = {0};
uint64_t optimistic_insert_consistency_fail_cnt[MAX_APP_THREAD] = {0};
uint64_t optimistic_insert_fallback_cnt[MAX_APP_THREAD] = {0};
uint64_t leaf_group_reread_too_many_cnt[MAX_APP_THREAD] = {0};
uint64_t leaf_group_reread_fallback_cnt[MAX_APP_THREAD] = {0};

// ================= Experiment Switches =================
// Baseline SXLOCK CAS-failure experiment:
//   keep only the original SXLOCK path and record CAS failures/retries.
//   disabled: optimistic update/insert fast path, adaptive hot bypass,
//             hot read cache, hot elastic leaf, local lock, AP/batch poll.
// #define USE_AP
#define USE_OPTIMISTIC_UPDATE_HIT
#define USE_ADAPTIVE_HOT_BYPASS
#define USE_OPTIMISTIC_GUARD_RETRY
// #define USE_HOT_READ_CACHE
// #define USE_HOT_ELASTIC_LEAF
#define USE_LEAF_SPLIT_GUARD
#define USE_LOCAL_GROUP_WRITE_COMBINE
#define USE_LOCAL_GROUP_READ_COMBINE
#define USE_SPLIT_ONLY_X_LOCK
#define USE_SX_LOCK
#define BATCH_LOCK_READ
#define FINE_GRAINED_LEAF_NODE
#define FINE_GRAINED_INTERNAL_NODE
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
#if defined(USE_OPTIMISTIC_UPDATE_HIT) && !defined(USE_LEAF_SPLIT_GUARD)
#error "optimistic leaf fast path requires USE_LEAF_SPLIT_GUARD"
#endif

#if defined(USE_SPLIT_ONLY_X_LOCK) && !defined(USE_SX_LOCK)
#error "USE_SPLIT_ONLY_X_LOCK still needs USE_SX_LOCK as the X-lock backend"
#endif

#if defined(USE_SPLIT_ONLY_X_LOCK) && !defined(USE_LEAF_SPLIT_GUARD)
#error "USE_SPLIT_ONLY_X_LOCK requires USE_LEAF_SPLIT_GUARD"
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

// ---------------- Local group lock / split guard stats ----------------
// These counters diagnose tail latency caused by local group-lock queues
// blocking split-guard inflight drain and delayed split-guard ACKs.


uint64_t local_group_wc_owner_cnt[MAX_APP_THREAD] = {0};
uint64_t local_group_wc_waiter_cnt[MAX_APP_THREAD] = {0};
uint64_t local_group_wc_waiter_return_cnt[MAX_APP_THREAD] = {0};
uint64_t local_group_wc_apply_cnt[MAX_APP_THREAD] = {0};
uint64_t local_group_wc_nochange_cnt[MAX_APP_THREAD] = {0};
uint64_t local_group_wc_replace_cnt[MAX_APP_THREAD] = {0};
uint64_t local_group_wc_bypass_cnt[MAX_APP_THREAD] = {0};
uint64_t local_group_rc_owner_cnt[MAX_APP_THREAD] = {0};
uint64_t local_group_rc_waiter_cnt[MAX_APP_THREAD] = {0};
uint64_t local_group_rc_waiter_return_cnt[MAX_APP_THREAD] = {0};
uint64_t local_group_rc_bypass_cnt[MAX_APP_THREAD] = {0};
uint64_t local_group_rc_bytes_saved[MAX_APP_THREAD] = {0};
uint64_t local_group_rc_wait_us_sum[MAX_APP_THREAD] = {0};
uint64_t local_group_rc_wait_us_max[MAX_APP_THREAD] = {0};
uint64_t split_guard_begin_cnt[MAX_APP_THREAD] = {0};
uint64_t split_guard_wait_event_cnt[MAX_APP_THREAD] = {0};
uint64_t split_guard_wait_yield_cnt[MAX_APP_THREAD] = {0};
uint64_t split_guard_wait_us_sum[MAX_APP_THREAD] = {0};
uint64_t split_guard_wait_us_max[MAX_APP_THREAD] = {0};
uint64_t split_guard_wait_inflight_us_sum[MAX_APP_THREAD] = {0};
uint64_t split_guard_wait_ack_us_sum[MAX_APP_THREAD] = {0};
uint64_t split_guard_wait_both_us_sum[MAX_APP_THREAD] = {0};
uint64_t split_guard_inflight_max[MAX_APP_THREAD] = {0};

std::atomic<uint64_t> split_guard_ack_immediate_cnt{0};
std::atomic<uint64_t> split_guard_ack_queued_cnt{0};
std::atomic<uint64_t> split_guard_ack_sent_after_wait_cnt{0};
std::atomic<uint64_t> split_guard_ack_flush_blocked_cnt{0};
std::atomic<uint64_t> split_guard_ack_queue_wait_us_sum{0};
std::atomic<uint64_t> split_guard_ack_queue_wait_us_max{0};
std::atomic<uint64_t> split_guard_ack_pending_max{0};
std::atomic<uint64_t> split_guard_ack_blocking_inflight_max{0};

inline uint64_t now_us() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

inline uint32_t fast_rand_u32() {
  static thread_local uint32_t state =
      static_cast<uint32_t>(now_us()) ^ 0x9e3779b9u;
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

inline bool valid_app_tid(uint16_t tid) {
  return tid < MAX_APP_THREAD;
}

inline void update_max_u64(uint64_t &target, uint64_t value) {
  if (value > target) {
    target = value;
  }
}

inline void atomic_update_max_u64(std::atomic<uint64_t> &target,
                                  uint64_t value) {
  uint64_t cur = target.load(std::memory_order_relaxed);
  while (cur < value &&
         !target.compare_exchange_weak(cur, value,
                                       std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {
  }
}

#ifdef USE_HOT_READ_CACHE
uint64_t hot_read_cache_hit_cnt[MAX_APP_THREAD] = {0};
uint64_t hot_read_cache_miss_cnt[MAX_APP_THREAD] = {0};
uint64_t hot_read_cache_fill_cnt[MAX_APP_THREAD] = {0};
uint64_t hot_read_cache_update_cnt[MAX_APP_THREAD] = {0};
uint64_t hot_read_cache_invalidate_cnt[MAX_APP_THREAD] = {0};
uint64_t hot_read_cache_leaf_lazy_invalidate_cnt[MAX_APP_THREAD] = {0};
#endif
uint64_t hot_leaf_update_hot_transition_cnt[MAX_APP_THREAD] = {0};
uint64_t hot_leaf_conflict_hot_transition_cnt[MAX_APP_THREAD] = {0};
uint64_t hot_leaf_insert_hot_transition_cnt[MAX_APP_THREAD] = {0};
uint64_t hot_leaf_fold_pending_transition_cnt[MAX_APP_THREAD] = {0};
uint64_t hot_leaf_controller_bypass_cnt[MAX_APP_THREAD] = {0};
uint64_t hot_leaf_cache_gate_deny_cnt[MAX_APP_THREAD] = {0};
uint64_t hot_leaf_optimistic_probe_cnt[MAX_APP_THREAD] = {0};

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
thread_local CoroQueue busy_waiting_queue;
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

enum class HotLeafState : uint8_t {
  Normal = 0,
  UpdateHot,
  ConflictHot,
  InsertHot,
  FoldPending,
};

constexpr size_t kHotElasticLeafTableSize = 4096;
constexpr uint8_t kHotLeafUpdateHotThreshold = 4;
constexpr uint8_t kHotLeafConflictHotThreshold = 3;
constexpr uint8_t kHotLeafInsertHotThreshold = 2;
constexpr uint8_t kHotLeafStateCooldownMax = 64;
constexpr uint8_t kHotLeafConflictCooldownScale = 4;
constexpr uint8_t kHotLeafOptimisticProbeInterval = 8;

struct HotLeafBypassEntry {
  uint64_t page_sig = 0;
  uint8_t cooldown = 0;
  uint8_t fail_count = 0;
};

struct HotElasticLeafEntry {
  uint64_t page_sig = 0;
  HotLeafState state = HotLeafState::Normal;
  uint8_t update_hits = 0;
  uint8_t cas_fails = 0;
  uint8_t insert_conflicts = 0;
  uint8_t cooldown = 0;
  uint8_t optimistic_probe_credit = 0;
};

#ifdef USE_LOCAL_LOCK
thread_local std::queue<uint16_t> hot_wait_queue;
#endif
#ifdef USE_ADAPTIVE_HOT_BYPASS
thread_local HotLeafBypassEntry hot_leaf_bypass_table[kHotLeafBypassTableSize];
#endif
#ifdef USE_HOT_ELASTIC_LEAF
thread_local HotElasticLeafEntry hot_elastic_leaf_table[kHotElasticLeafTableSize];
thread_local bool hot_leaf_controller_has_write_signal = false;
#endif
std::atomic<bool> g_prefill_split_stats_enabled{false};
#ifdef USE_HOT_READ_CACHE
std::atomic<bool> g_hot_read_cache_enabled{false};
#endif

inline uint64_t get_leaf_page_sig(GlobalAddress page_addr) {
  return (static_cast<uint64_t>(page_addr.nodeID) << 48) | page_addr.offset;
}

constexpr size_t kLeafSplitGuardTableSize = define::kNumOfLock;

struct LeafSplitGuardToken {
  uint16_t origin_node_id = 0;
  uint16_t origin_app_id = 0;
  uint32_t generation = 0;
};

#ifdef USE_LOCAL_GROUP_WRITE_COMBINE
struct alignas(64) LocalGroupWriteCombineState {
  std::mutex combine_mutex;
  static constexpr int kLocalGroupWriteCombineSlots = 8;
  struct LocalGroupWriteCombineSlot {
    bool valid = false;
    bool active = false;
    Key key = 0;
    Value value = 0;
    uint64_t epoch = 0;
  };
  LocalGroupWriteCombineSlot combine_slots[kLocalGroupWriteCombineSlots];
};
#endif

#ifdef USE_LOCAL_GROUP_READ_COMBINE
struct alignas(64) LocalGroupReadCombineState {
  std::atomic<uint32_t> active{0};
  std::atomic<uint64_t> epoch{0};
  std::atomic<uint64_t> active_target_epoch{0};
  std::atomic<uint32_t> waiters{0};
  std::atomic<size_t> bytes[2] = {};
  std::atomic<uint64_t> buffer_epoch[2] = {};
  char bucket_buffer[2][kReadBucketSize];
};
#endif

struct LeafSplitGuardEntry {
  std::atomic<uint32_t> inflight{0};
  std::mutex active_mutex;
  std::vector<LeafSplitGuardToken> active_tokens;

#ifdef USE_LOCAL_GROUP_WRITE_COMBINE
  LocalGroupWriteCombineState combine_states[kNumGroup];
#endif
#ifdef USE_LOCAL_GROUP_READ_COMBINE
  LocalGroupReadCombineState read_combine_states[kNumGroup][2];
#endif
};

struct SplitGuardAckState {
  uint64_t page_sig = 0;
  uint32_t generation = 0;
  uint32_t ack_count = 0;
  uint64_t ack_node_mask = 0;
};

LeafSplitGuardEntry leaf_split_guard_table[kLeafSplitGuardTableSize];
std::atomic<uint32_t> leaf_split_guard_generation{1};
struct PendingLeafSplitGuardAck {
  RawMessage msg;
  uint64_t queued_us = 0;
  uint32_t max_inflight_seen = 0;
};

std::mutex pending_leaf_split_guard_acks_mutex;
std::vector<PendingLeafSplitGuardAck> pending_leaf_split_guard_acks;
std::mutex split_guard_ack_states_mutex;
std::vector<SplitGuardAckState> split_guard_ack_states;
std::atomic<bool> leaf_split_guard_progress_started{false};

inline size_t get_leaf_split_guard_slot(GlobalAddress page_addr) {
  GlobalAddress a;
  a.offset = page_addr.offset;
  a.nodeID = page_addr.nodeID;
  return CityHash64(reinterpret_cast<const char *>(&a), sizeof(a)) %
          kLeafSplitGuardTableSize;
}

inline bool leaf_split_guard_entry_has_active_tokens(
  LeafSplitGuardEntry *entry) {
  std::lock_guard<std::mutex> lock(entry->active_mutex);
  return !entry->active_tokens.empty();
}

inline LeafSplitGuardEntry *get_leaf_split_guard_entry(GlobalAddress page_addr) {
  return &leaf_split_guard_table[get_leaf_split_guard_slot(page_addr)];
}

inline bool leaf_split_guard_is_blocked(LeafSplitGuardEntry *entry) {
  return leaf_split_guard_entry_has_active_tokens(entry);
}

inline SplitGuardAckState *find_split_guard_ack_state(uint64_t page_sig,
                                                    uint32_t generation) {
  for (auto &state : split_guard_ack_states) {
    if (state.page_sig == page_sig && state.generation == generation) {
      return &state;
    }
  }
  return nullptr;
}

inline void register_split_guard_ack_state(uint64_t page_sig,
                                          uint32_t generation) {
  std::lock_guard<std::mutex> lock(split_guard_ack_states_mutex);
  if (find_split_guard_ack_state(page_sig, generation) == nullptr) {
    split_guard_ack_states.push_back({page_sig, generation, 0, 0});
  }
}

inline void unregister_split_guard_ack_state(uint64_t page_sig,
                                            uint32_t generation) {
  std::lock_guard<std::mutex> lock(split_guard_ack_states_mutex);
  split_guard_ack_states.erase(
      std::remove_if(split_guard_ack_states.begin(),
                      split_guard_ack_states.end(),
                      [=](const SplitGuardAckState &s) {
                        return s.page_sig == page_sig &&
                              s.generation == generation;
                      }),
      split_guard_ack_states.end());
}

inline bool split_guard_ack_state_ready(uint64_t page_sig, uint32_t generation,
                                      uint32_t expected_acks) {
  std::lock_guard<std::mutex> lock(split_guard_ack_states_mutex);
  SplitGuardAckState *state =
      find_split_guard_ack_state(page_sig, generation);
  return state != nullptr && state->ack_count >= expected_acks;
}

inline bool mark_leaf_split_guard_ack_from_node(uint64_t *ack_node_mask,
                                              uint16_t node_id) {
  if (node_id >= 64) {
    return true;
  }
  uint64_t bit = 1ull << node_id;
  if ((*ack_node_mask & bit) != 0) {
    return false;
  }
  *ack_node_mask |= bit;
  return true;
}

inline void record_split_guard_ack_state(const RawMessage &msg) {
  std::lock_guard<std::mutex> lock(split_guard_ack_states_mutex);
  SplitGuardAckState *state = find_split_guard_ack_state(
      msg.arg0, static_cast<uint32_t>(msg.arg1));
  if (state != nullptr &&
      mark_leaf_split_guard_ack_from_node(&state->ack_node_mask,
                                          msg.node_id)) {
    ++state->ack_count;
  }
}

inline uint16_t leaf_split_guard_control_app_id() {
  return define::kSplitGuardControlAppID;
}

inline bool leaf_split_guard_same_token(const LeafSplitGuardToken &token,
                                      uint16_t origin_node_id,
                                      uint16_t origin_app_id,
                                      uint32_t generation) {
  return token.origin_node_id == origin_node_id &&
          token.origin_app_id == origin_app_id &&
          token.generation == generation;
}

inline void apply_leaf_split_guard_block(GlobalAddress page_addr,
                                        uint16_t origin_node_id,
                                        uint16_t origin_app_id,
                                        uint32_t generation) {
  LeafSplitGuardEntry *entry = get_leaf_split_guard_entry(page_addr);
  {
    std::lock_guard<std::mutex> lock(entry->active_mutex);
    for (const auto &token : entry->active_tokens) {
      if (leaf_split_guard_same_token(token, origin_node_id, origin_app_id,
                                      generation)) {
        return;
      }
    }
    entry->active_tokens.push_back({origin_node_id, origin_app_id, generation});
  }
}

inline void apply_leaf_split_guard_unblock(GlobalAddress page_addr,
                                          uint16_t origin_node_id,
                                          uint16_t origin_app_id,
                                          uint32_t generation) {
  LeafSplitGuardEntry *entry = get_leaf_split_guard_entry(page_addr);
  {
    std::lock_guard<std::mutex> lock(entry->active_mutex);
    entry->active_tokens.erase(
        std::remove_if(entry->active_tokens.begin(), entry->active_tokens.end(),
                        [=](const LeafSplitGuardToken &token) {
                          return leaf_split_guard_same_token(
                              token, origin_node_id, origin_app_id, generation);
                        }),
        entry->active_tokens.end());
  }
}

inline void send_leaf_split_guard_message(DSMClient *dsm_client,
                                        RpcType type,
                                        GlobalAddress page_addr,
                                        uint64_t page_sig,
                                        uint32_t generation) {
  RawMessage m{};
  m.type = type;
  m.addr = page_addr;
  m.arg0 = page_sig;
  m.arg1 = generation;
  m.requester_node_id = dsm_client->get_my_client_id();
  m.requester_app_id = leaf_split_guard_control_app_id();
  dsm_client->RpcCallDir(m, page_addr.nodeID);
}

inline void send_leaf_split_guard_ack(DSMClient *dsm_client,
                                    const RawMessage &block_msg) {
  RawMessage ack{};
  ack.type = RpcType::SPLIT_GUARD_ACK;
  ack.addr = block_msg.addr;
  ack.arg0 = block_msg.arg0;
  ack.arg1 = block_msg.arg1;
  ack.requester_node_id = block_msg.requester_node_id;
  ack.requester_app_id = block_msg.requester_app_id;
  dsm_client->RpcCallDir(ack, block_msg.addr.nodeID);
}

inline void queue_leaf_split_guard_ack(const RawMessage &block_msg) {
  std::lock_guard<std::mutex> lock(pending_leaf_split_guard_acks_mutex);
  pending_leaf_split_guard_acks.push_back(
      PendingLeafSplitGuardAck{block_msg, now_us(), 0});
  split_guard_ack_queued_cnt.fetch_add(1, std::memory_order_relaxed);
  atomic_update_max_u64(split_guard_ack_pending_max,
                        pending_leaf_split_guard_acks.size());
}

inline void flush_pending_leaf_split_guard_acks(DSMClient *dsm_client) {
  std::lock_guard<std::mutex> lock(pending_leaf_split_guard_acks_mutex);

  atomic_update_max_u64(split_guard_ack_pending_max,
                        pending_leaf_split_guard_acks.size());

  for (size_t i = 0; i < pending_leaf_split_guard_acks.size();) {
    PendingLeafSplitGuardAck &pending = pending_leaf_split_guard_acks[i];
    const RawMessage &msg = pending.msg;

    LeafSplitGuardEntry *entry = get_leaf_split_guard_entry(msg.addr);
    uint32_t inflight = entry->inflight.load(std::memory_order_acquire);
    bool ready = inflight == 0;

    if (ready) {
      uint64_t wait_us = now_us() - pending.queued_us;
      split_guard_ack_queue_wait_us_sum.fetch_add(wait_us,
                                                  std::memory_order_relaxed);
      atomic_update_max_u64(split_guard_ack_queue_wait_us_max, wait_us);
      split_guard_ack_sent_after_wait_cnt.fetch_add(1,
                                                    std::memory_order_relaxed);

      send_leaf_split_guard_ack(dsm_client, msg);

      pending_leaf_split_guard_acks[i] =
          pending_leaf_split_guard_acks.back();
      pending_leaf_split_guard_acks.pop_back();
      continue;
    }

    split_guard_ack_flush_blocked_cnt.fetch_add(1, std::memory_order_relaxed);
    if (inflight > pending.max_inflight_seen) {
      pending.max_inflight_seen = inflight;
    }
    atomic_update_max_u64(split_guard_ack_blocking_inflight_max, inflight);

    ++i;
  }
}

inline bool handle_leaf_split_guard_message(DSMClient *dsm_client,
                                          const RawMessage &msg,
                                          uint64_t wait_page_sig,
                                          uint32_t wait_generation,
                                          uint32_t *ack_count,
                                          uint64_t *ack_node_mask = nullptr) {
  if (msg.type == RpcType::SPLIT_GUARD_BLOCK) {
    apply_leaf_split_guard_block(msg.addr, msg.requester_node_id,
                                  msg.requester_app_id,
                                  static_cast<uint32_t>(msg.arg1));
    LeafSplitGuardEntry *entry = get_leaf_split_guard_entry(msg.addr);
    uint32_t inflight = entry->inflight.load(std::memory_order_acquire);
    bool ready = inflight == 0;
    if (ready) {
      split_guard_ack_immediate_cnt.fetch_add(1, std::memory_order_relaxed);
      send_leaf_split_guard_ack(dsm_client, msg);
    } else {
      atomic_update_max_u64(split_guard_ack_blocking_inflight_max, inflight);
      queue_leaf_split_guard_ack(msg);
    }
    return true;
  }
  if (msg.type == RpcType::SPLIT_GUARD_UNBLOCK) {
    apply_leaf_split_guard_unblock(msg.addr, msg.requester_node_id,
                                    msg.requester_app_id,
                                    static_cast<uint32_t>(msg.arg1));
    return true;
  }
  if (msg.type == RpcType::SPLIT_GUARD_ACK) {
    if (ack_count != nullptr && msg.arg0 == wait_page_sig &&
        static_cast<uint32_t>(msg.arg1) == wait_generation &&
        (ack_node_mask == nullptr ||
          mark_leaf_split_guard_ack_from_node(ack_node_mask, msg.node_id))) {
      ++(*ack_count);
    }
    record_split_guard_ack_state(msg);
    return true;
  }
  return false;
}

inline void drain_leaf_split_guard_messages(DSMClient *dsm_client) {
  RawMessage msg;
  while (dsm_client->PollRpcCqOnce(msg)) {
    // printf("CN%d received Split Guard from MN (originated from CN%d)\n", dsm_client->get_my_client_id(), msg.requester_node_id);
    handle_leaf_split_guard_message(dsm_client, msg, 0, 0, nullptr);
  }
  flush_pending_leaf_split_guard_acks(dsm_client);
}

inline void leaf_split_guard_progress_loop(DSMClient *dsm_client) {
  dsm_client->RegisterThreadAt(leaf_split_guard_control_app_id());
  bindCoreToNuma(define::kSplitGuardProgressCore, 0);
  while (true) {
    RawMessage msg;
    if (dsm_client->PollRpcCqOnce(msg)) {
      handle_leaf_split_guard_message(dsm_client, msg, 0, 0, nullptr);
      flush_pending_leaf_split_guard_acks(dsm_client);
    } else {
      flush_pending_leaf_split_guard_acks(dsm_client);
      _mm_pause();
    }
  }
}

inline void start_leaf_split_guard_progress_thread(DSMClient *dsm_client) {
  bool expected = false;
  if (!leaf_split_guard_progress_started.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    return;
  }
  std::thread([dsm_client]() { leaf_split_guard_progress_loop(dsm_client); })
      .detach();
}

inline bool enter_leaf_split_guard(DSMClient *dsm_client,
                                  GlobalAddress page_addr) {
  // drain_leaf_split_guard_messages(dsm_client);
  LeafSplitGuardEntry *entry = get_leaf_split_guard_entry(page_addr);
  if (leaf_split_guard_is_blocked(entry)) {
    return false;
  }
  entry->inflight.fetch_add(1, std::memory_order_acq_rel);
  if (leaf_split_guard_is_blocked(entry)) {
    entry->inflight.fetch_sub(1, std::memory_order_acq_rel);
    return false;
  }
  return true;
}

inline void wait_until_enter_leaf_split_guard(DSMClient *dsm_client,
                                            GlobalAddress page_addr,
                                            CoroContext *ctx) {
  while (!enter_leaf_split_guard(dsm_client, page_addr)) {
    if (ctx != nullptr && ctx->busy_waiting_queue != nullptr) {
      ctx->busy_waiting_queue->push(
          std::make_pair(ctx->coro_id, []() { return true; }));
      (*ctx->yield)(*ctx->master);
    } else {
      _mm_pause();
      // drain_leaf_split_guard_messages(dsm_client);
    }
  }
}

inline void optimistic_update_retry_pause(int retry_cnt) {
  constexpr int kOptimisticUpdateBackoffMaxWindow = 3;
  int window = retry_cnt;
  if (window > kOptimisticUpdateBackoffMaxWindow) {
    window = kOptimisticUpdateBackoffMaxWindow;
  }
  if (window < 0) {
    window = 0;
  }
  int pause_cnt = static_cast<int>(fast_rand_u32() % (window + 1));
  for (int i = 0; i < pause_cnt; ++i) {
    _mm_pause();
  }
}

inline void leave_leaf_split_guard(DSMClient *dsm_client, GlobalAddress page_addr) {
  LeafSplitGuardEntry *entry = get_leaf_split_guard_entry(page_addr);
  entry->inflight.fetch_sub(1, std::memory_order_acq_rel);
// flush_pending_leaf_split_guard_acks(dsm_client);
}

#ifdef USE_LOCAL_GROUP_WRITE_COMBINE
struct LocalGroupWriteCombineToken {
  LeafSplitGuardEntry *entry = nullptr;
  int group_id = -1;
  int slot_idx = -1;
  Key key = 0;
  uint64_t target_epoch = 0;
  bool owner = false;
  bool valid = false;
};

inline LocalGroupWriteCombineToken begin_local_group_write_combine(
    LeafSplitGuardEntry *entry, int group_id, const Key &k, const Value &v,
    uint16_t tid) {
  assert(group_id >= 0 && group_id < kNumGroup);

  LocalGroupWriteCombineToken token;
  token.entry = entry;
  token.group_id = group_id;
  token.key = k;

  LocalGroupWriteCombineState *state = &entry->combine_states[group_id];
  std::lock_guard<std::mutex> guard(state->combine_mutex);

  int empty_slot = -1;
  int inactive_same_key = -1;
  int inactive_victim = -1;
  for (int i = 0; i < LocalGroupWriteCombineState::kLocalGroupWriteCombineSlots; ++i) {
    auto &slot = state->combine_slots[i];
    if (slot.valid && slot.active && slot.key == k) {
      slot.value = v;
      token.slot_idx = i;
      token.target_epoch = slot.epoch + 1;
      token.owner = false;
      token.valid = true;
      if (valid_app_tid(tid)) {
        local_group_wc_waiter_cnt[tid]++;
      }
      return token;
    }
    if (slot.valid && !slot.active && slot.key == k) {
      inactive_same_key = i;
    }
    if (slot.valid && !slot.active && inactive_victim < 0) {
      inactive_victim = i;
    }
    if (!slot.valid && empty_slot < 0) {
      empty_slot = i;
    }
  }

  int owner_slot = inactive_same_key >= 0 ? inactive_same_key : empty_slot;
  if (owner_slot < 0) {
    owner_slot = inactive_victim;
  }
  if (owner_slot < 0) {
    if (valid_app_tid(tid)) {
      local_group_wc_bypass_cnt[tid]++;
    }
    return token;
  }

  auto &slot = state->combine_slots[owner_slot];
  if (slot.valid && slot.key != k && valid_app_tid(tid)) {
    local_group_wc_replace_cnt[tid]++;
  }
  slot.valid = true;
  slot.active = true;
  slot.key = k;
  slot.value = v;

  token.slot_idx = owner_slot;
  token.target_epoch = slot.epoch + 1;
  token.owner = true;
  token.valid = true;
  if (valid_app_tid(tid)) {
    local_group_wc_owner_cnt[tid]++;
  }
  return token;
}

inline bool local_group_write_combine_finished(
    const LocalGroupWriteCombineToken &token) {
  if (!token.valid || token.owner || token.entry == nullptr ||
      token.group_id < 0 || token.slot_idx < 0) {
    return true;
  }

  LocalGroupWriteCombineState *state =
      &token.entry->combine_states[token.group_id];
  std::lock_guard<std::mutex> guard(state->combine_mutex);
  auto &slot = state->combine_slots[token.slot_idx];
  if (!slot.valid || slot.key != token.key) {
    return true;
  }
  return slot.epoch >= token.target_epoch;
}

inline void wait_for_local_group_write_combine(
    const LocalGroupWriteCombineToken &token, CoroContext *ctx) {
  while (!local_group_write_combine_finished(token)) {
    if (ctx != nullptr && ctx->busy_waiting_queue != nullptr) {
      ctx->busy_waiting_queue->push(std::make_pair(
          ctx->coro_id, [token]() { return local_group_write_combine_finished(token); }));
      (*ctx->yield)(*ctx->master);
    } else {
      _mm_pause();
    }
  }
}

inline void finish_local_group_write_combine(
    const LocalGroupWriteCombineToken &token) {
  if (!token.valid || !token.owner || token.entry == nullptr ||
      token.group_id < 0 || token.slot_idx < 0) {
    return;
  }

  LocalGroupWriteCombineState *state =
      &token.entry->combine_states[token.group_id];
  std::lock_guard<std::mutex> guard(state->combine_mutex);
  auto &slot = state->combine_slots[token.slot_idx];
  if (slot.valid && slot.active && slot.key == token.key) {
    slot.active = false;
    slot.epoch++;
  }
}

inline bool consume_local_group_write_combine(LeafSplitGuardEntry *entry,
                                              int group_id, const Key &k,
                                              Value &v, uint16_t tid) {
  assert(group_id >= 0 && group_id < kNumGroup);

  LocalGroupWriteCombineState *state = &entry->combine_states[group_id];
  std::lock_guard<std::mutex> guard(state->combine_mutex);
  for (int i = 0; i < LocalGroupWriteCombineState::kLocalGroupWriteCombineSlots; ++i) {
    auto &slot = state->combine_slots[i];
    if (slot.valid && slot.key == k) {
      if (slot.value != v) {
        v = slot.value;
        if (valid_app_tid(tid)) {
          local_group_wc_apply_cnt[tid]++;
        }
        return true;
      }
      if (valid_app_tid(tid)) {
        local_group_wc_nochange_cnt[tid]++;
      }
      return false;
    }
  }
  return false;
}
#endif

#ifdef USE_LOCAL_GROUP_READ_COMBINE
struct LocalGroupReadCombineToken {
  LeafSplitGuardEntry *entry = nullptr;
  int group_id = -1;
  bool is_front = false;
  uint64_t target_epoch = 0;
  bool owner = false;
  bool valid = false;
};

inline LocalGroupReadCombineToken begin_local_group_read_combine(
    LeafSplitGuardEntry *entry, int group_id, bool is_front, uint16_t tid) {
  assert(group_id >= 0 && group_id < kNumGroup);

  LocalGroupReadCombineToken token;
  token.entry = entry;
  token.group_id = group_id;
  token.is_front = is_front;

  LocalGroupReadCombineState *state =
      &entry->read_combine_states[group_id][is_front ? 0 : 1];

  while (true) {
    uint32_t inactive = 0;
    if (state->active.compare_exchange_strong(
            inactive, 2, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
      token.target_epoch =
          state->epoch.load(std::memory_order_acquire) + 1;
      state->active_target_epoch.store(token.target_epoch,
                                       std::memory_order_release);
      state->waiters.store(0, std::memory_order_release);
      state->active.store(1, std::memory_order_release);
      token.owner = true;
      token.valid = true;
      if (valid_app_tid(tid)) {
        local_group_rc_owner_cnt[tid]++;
      }
      return token;
    }

    if (inactive == 1) {
      token.target_epoch =
          state->active_target_epoch.load(std::memory_order_acquire);
      state->waiters.fetch_add(1, std::memory_order_acq_rel);
      if (token.target_epoch == 0) {
        break;
      }
      token.owner = false;
      token.valid = true;
      if (valid_app_tid(tid)) {
        local_group_rc_waiter_cnt[tid]++;
      }
      return token;
    }

    if (inactive != 2) {
      break;
    }
    _mm_pause();
  }

  if (valid_app_tid(tid)) {
    local_group_rc_bypass_cnt[tid]++;
  }
  return token;
}

inline bool local_group_read_combine_finished(
    const LocalGroupReadCombineToken &token) {
  if (!token.valid || token.owner || token.entry == nullptr ||
      token.group_id < 0) {
    return true;
  }

  LocalGroupReadCombineState *state =
      &token.entry
           ->read_combine_states[token.group_id][token.is_front ? 0 : 1];
  return state->epoch.load(std::memory_order_acquire) >= token.target_epoch;
}

inline void wait_for_local_group_read_combine(
    const LocalGroupReadCombineToken &token, CoroContext *ctx, uint16_t tid) {
  uint64_t wait_start_us = now_us();
  while (!local_group_read_combine_finished(token)) {
    if (ctx != nullptr && ctx->busy_waiting_queue != nullptr) {
      ctx->busy_waiting_queue->push(std::make_pair(
          ctx->coro_id,
          [token]() { return local_group_read_combine_finished(token); }));
      (*ctx->yield)(*ctx->master);
    } else {
      _mm_pause();
    }
  }

  uint64_t wait_us = now_us() - wait_start_us;
  if (valid_app_tid(tid) && wait_us > 0) {
    local_group_rc_wait_us_sum[tid] += wait_us;
    update_max_u64(local_group_rc_wait_us_max[tid], wait_us);
  }
}

inline void finish_local_group_read_combine(
    const LocalGroupReadCombineToken &token, const char *src_bucket,
    size_t bytes) {
  if (!token.valid || !token.owner || token.entry == nullptr ||
      token.group_id < 0) {
    return;
  }

  LocalGroupReadCombineState *state =
      &token.entry
           ->read_combine_states[token.group_id][token.is_front ? 0 : 1];
  int buffer_idx = static_cast<int>(token.target_epoch & 1);
  state->buffer_epoch[buffer_idx].store(0, std::memory_order_release);
  state->bytes[buffer_idx].store(0, std::memory_order_release);

  if (state->waiters.load(std::memory_order_acquire) == 0) {
    state->epoch.store(token.target_epoch, std::memory_order_release);
    state->active.store(0, std::memory_order_release);
    return;
  }

  size_t copy_bytes = std::min(bytes, static_cast<size_t>(kReadBucketSize));
  memcpy(state->bucket_buffer[buffer_idx], src_bucket, copy_bytes);
  state->bytes[buffer_idx].store(copy_bytes, std::memory_order_release);
  state->buffer_epoch[buffer_idx].store(token.target_epoch,
                                        std::memory_order_release);
  state->epoch.store(token.target_epoch, std::memory_order_release);
  state->active.store(0, std::memory_order_release);
}

inline bool copy_local_group_read_combine_result(
    const LocalGroupReadCombineToken &token, char *dst_bucket, size_t bytes,
    uint16_t tid) {
  if (!token.valid || token.owner || token.entry == nullptr ||
      token.group_id < 0) {
    if (valid_app_tid(tid)) {
      local_group_rc_bypass_cnt[tid]++;
    }
    return false;
  }

  LocalGroupReadCombineState *state =
      &token.entry
           ->read_combine_states[token.group_id][token.is_front ? 0 : 1];
  uint64_t published_epoch = state->epoch.load(std::memory_order_acquire);
  if (published_epoch != token.target_epoch) {
    if (valid_app_tid(tid)) {
      local_group_rc_bypass_cnt[tid]++;
    }
    return false;
  }

  int buffer_idx = static_cast<int>(token.target_epoch & 1);
  if (state->buffer_epoch[buffer_idx].load(std::memory_order_acquire) !=
          token.target_epoch ||
      state->bytes[buffer_idx].load(std::memory_order_acquire) < bytes) {
    if (valid_app_tid(tid)) {
      local_group_rc_bypass_cnt[tid]++;
    }
    return false;
  }

  memcpy(dst_bucket, state->bucket_buffer[buffer_idx], bytes);

  if (state->buffer_epoch[buffer_idx].load(std::memory_order_acquire) !=
          token.target_epoch ||
      state->epoch.load(std::memory_order_acquire) != token.target_epoch) {
    if (valid_app_tid(tid)) {
      local_group_rc_bypass_cnt[tid]++;
    }
    return false;
  }

  if (valid_app_tid(tid)) {
    local_group_rc_waiter_return_cnt[tid]++;
    local_group_rc_bytes_saved[tid] += bytes;
  }
  return true;
}

inline void cancel_local_group_read_combine(
    const LocalGroupReadCombineToken &token) {
  if (!token.valid || !token.owner || token.entry == nullptr ||
      token.group_id < 0) {
    return;
  }

  LocalGroupReadCombineState *state =
      &token.entry
           ->read_combine_states[token.group_id][token.is_front ? 0 : 1];
  if (state->epoch.load(std::memory_order_acquire) + 1 ==
      token.target_epoch) {
    int buffer_idx = static_cast<int>(token.target_epoch & 1);
    state->buffer_epoch[buffer_idx].store(0, std::memory_order_release);
    state->bytes[buffer_idx].store(0, std::memory_order_release);
    state->epoch.store(token.target_epoch, std::memory_order_release);
    state->active.store(0, std::memory_order_release);
  }
}
#endif

inline uint32_t begin_leaf_split_guard(DSMClient *dsm_client,
                                      GlobalAddress page_addr,
                                      CoroContext *ctx) {
  uint64_t page_sig = get_leaf_page_sig(page_addr);
  uint32_t generation =
      leaf_split_guard_generation.fetch_add(1, std::memory_order_acq_rel) + 1;

  uint16_t tid = dsm_client->get_my_thread_id();
  if (valid_app_tid(tid)) {
    split_guard_begin_cnt[tid]++;
  }

  apply_leaf_split_guard_block(page_addr, dsm_client->get_my_client_id(),
                                leaf_split_guard_control_app_id(), generation);

  uint32_t expected_acks =
      dsm_client->get_client_size() > 0 ? dsm_client->get_client_size() - 1 : 0;
  register_split_guard_ack_state(page_sig, generation);
  if (expected_acks > 0) {
    send_leaf_split_guard_message(dsm_client, RpcType::SPLIT_GUARD_BLOCK,
                                  page_addr, page_sig, generation);
  }

  if (ctx != nullptr) {
    LeafSplitGuardEntry *entry = get_leaf_split_guard_entry(page_addr);

    uint64_t wait_start_us = now_us();
    uint64_t wait_inflight_us = 0;
    uint64_t wait_ack_us = 0;
    uint64_t wait_both_us = 0;
    uint64_t wait_yields = 0;
    uint32_t max_inflight = 0;
    bool waited = false;

    auto guard_ready = [=]() {
      return entry->inflight.load(std::memory_order_acquire) == 0 &&
             split_guard_ack_state_ready(page_sig, generation, expected_acks);
    };

    while (!guard_ready()) {
      uint64_t iter_start_us = now_us();

      uint32_t inflight =
          entry->inflight.load(std::memory_order_acquire);
      bool inflight_blocked = inflight != 0;
      bool ack_blocked =
          !split_guard_ack_state_ready(page_sig, generation, expected_acks);

      if (inflight > max_inflight) {
        max_inflight = inflight;
      }

      waited = true;
      ++wait_yields;

      if (ctx->busy_waiting_queue != nullptr) {
        ctx->busy_waiting_queue->push(
            std::make_pair(ctx->coro_id, guard_ready));
        (*ctx->yield)(*ctx->master);
      } else {
        _mm_pause();
      }

      uint64_t iter_us = now_us() - iter_start_us;
      if (inflight_blocked && ack_blocked) {
        wait_both_us += iter_us;
      } else if (inflight_blocked) {
        wait_inflight_us += iter_us;
      } else if (ack_blocked) {
        wait_ack_us += iter_us;
      }
    }

    uint64_t total_wait_us = now_us() - wait_start_us;
    if (valid_app_tid(tid)) {
      if (waited) {
        split_guard_wait_event_cnt[tid]++;
      }
      split_guard_wait_yield_cnt[tid] += wait_yields;
      split_guard_wait_us_sum[tid] += total_wait_us;
      split_guard_wait_inflight_us_sum[tid] += wait_inflight_us;
      split_guard_wait_ack_us_sum[tid] += wait_ack_us;
      split_guard_wait_both_us_sum[tid] += wait_both_us;
      update_max_u64(split_guard_wait_us_max[tid], total_wait_us);
      update_max_u64(split_guard_inflight_max[tid], max_inflight);
    }

    unregister_split_guard_ack_state(page_sig, generation);
    return generation;
  }

  uint32_t ack_count = 0;
  uint64_t ack_node_mask = 0;
  LeafSplitGuardEntry *entry = get_leaf_split_guard_entry(page_addr);

  uint64_t wait_start_us = now_us();
  uint64_t wait_inflight_us = 0;
  uint64_t wait_ack_us = 0;
  uint64_t wait_both_us = 0;
  uint64_t wait_iters = 0;
  uint32_t max_inflight = 0;
  bool waited = false;

  while (true) {
    uint64_t iter_start_us = now_us();

    uint32_t inflight = entry->inflight.load(std::memory_order_acquire);
    bool inflight_blocked = inflight != 0;
    bool ack_blocked =
        !split_guard_ack_state_ready(page_sig, generation, expected_acks);

    if (!inflight_blocked && !ack_blocked) {
      break;
    }

    waited = true;
    ++wait_iters;
    if (inflight > max_inflight) {
      max_inflight = inflight;
    }

    RawMessage msg;
    if (dsm_client->PollRpcCqOnce(msg)) {
      handle_leaf_split_guard_message(dsm_client, msg, page_sig, generation,
                                      &ack_count, &ack_node_mask);
      flush_pending_leaf_split_guard_acks(dsm_client);
    } else {
      flush_pending_leaf_split_guard_acks(dsm_client);
      _mm_pause();
    }

    uint64_t iter_us = now_us() - iter_start_us;
    if (inflight_blocked && ack_blocked) {
      wait_both_us += iter_us;
    } else if (inflight_blocked) {
      wait_inflight_us += iter_us;
    } else if (ack_blocked) {
      wait_ack_us += iter_us;
    }
  }

  uint64_t total_wait_us = now_us() - wait_start_us;
  if (valid_app_tid(tid)) {
    if (waited) {
      split_guard_wait_event_cnt[tid]++;
    }
    split_guard_wait_yield_cnt[tid] += wait_iters;
    split_guard_wait_us_sum[tid] += total_wait_us;
    split_guard_wait_inflight_us_sum[tid] += wait_inflight_us;
    split_guard_wait_ack_us_sum[tid] += wait_ack_us;
    split_guard_wait_both_us_sum[tid] += wait_both_us;
    update_max_u64(split_guard_wait_us_max[tid], total_wait_us);
    update_max_u64(split_guard_inflight_max[tid], max_inflight);
  }

  unregister_split_guard_ack_state(page_sig, generation);
  return generation;
}

inline void end_leaf_split_guard(DSMClient *dsm_client, GlobalAddress page_addr,
                                uint32_t generation) {
uint64_t page_sig = get_leaf_page_sig(page_addr);
apply_leaf_split_guard_unblock(page_addr, dsm_client->get_my_client_id(),
                                leaf_split_guard_control_app_id(), generation);
if (dsm_client->get_client_size() > 1) {
  send_leaf_split_guard_message(dsm_client, RpcType::SPLIT_GUARD_UNBLOCK,
                                page_addr, page_sig, generation);
}
}

inline size_t hot_elastic_leaf_slot(uint64_t page_sig) {
return CityHash64(reinterpret_cast<const char *>(&page_sig),
                  sizeof(page_sig)) &
        (kHotElasticLeafTableSize - 1);
}
#ifdef USE_HOT_READ_CACHE
inline HotElasticLeafEntry &hot_elastic_leaf_entry(GlobalAddress page_addr) {
uint64_t page_sig = get_leaf_page_sig(page_addr);
HotElasticLeafEntry &entry =
    hot_elastic_leaf_table[hot_elastic_leaf_slot(page_sig)];
if (entry.page_sig != page_sig) {
  entry = HotElasticLeafEntry{};
  entry.page_sig = page_sig;
}
return entry;
}
#endif
inline void set_hot_leaf_state(HotElasticLeafEntry &entry, HotLeafState state,
                              uint16_t tid) {
if (entry.state == state) {
  return;
}
entry.state = state;
if (state == HotLeafState::UpdateHot) {
  hot_leaf_update_hot_transition_cnt[tid]++;
} else if (state == HotLeafState::ConflictHot) {
  hot_leaf_conflict_hot_transition_cnt[tid]++;
} else if (state == HotLeafState::InsertHot) {
  hot_leaf_insert_hot_transition_cnt[tid]++;
} else if (state == HotLeafState::FoldPending) {
  hot_leaf_fold_pending_transition_cnt[tid]++;
}
}

inline void record_hot_leaf_update_success(GlobalAddress page_addr,
                                          uint16_t tid) {
#ifdef USE_HOT_ELASTIC_LEAF
hot_leaf_controller_has_write_signal = true;
HotElasticLeafEntry &entry = hot_elastic_leaf_entry(page_addr);
if (entry.update_hits < UINT8_MAX) {
  entry.update_hits++;
}
if (entry.cas_fails > 0) {
  entry.cas_fails--;
}
entry.cooldown = 0;
if (entry.update_hits >= kHotLeafUpdateHotThreshold &&
    entry.state != HotLeafState::ConflictHot) {
  set_hot_leaf_state(entry, HotLeafState::UpdateHot, tid);
}
#else
(void)page_addr;
(void)tid;
#endif
}

inline void record_hot_leaf_cas_fail(GlobalAddress page_addr, uint16_t tid) {
#ifdef USE_HOT_ELASTIC_LEAF
hot_leaf_controller_has_write_signal = true;
HotElasticLeafEntry &entry = hot_elastic_leaf_entry(page_addr);
if (entry.cas_fails < UINT8_MAX) {
  entry.cas_fails++;
}
entry.cooldown = std::min<uint8_t>(
    kHotLeafStateCooldownMax,
    static_cast<uint8_t>(entry.cas_fails * kHotLeafConflictCooldownScale +
                          kHotLeafBypassBaseCooldown));
if (entry.cas_fails >= kHotLeafConflictHotThreshold) {
  set_hot_leaf_state(entry, HotLeafState::ConflictHot, tid);
}
#else
(void)page_addr;
(void)tid;
#endif
}

inline void record_hot_leaf_consistency_fail(GlobalAddress page_addr,
                                            uint16_t tid) {
record_hot_leaf_cas_fail(page_addr, tid);
}

inline void record_hot_leaf_split_pressure(GlobalAddress page_addr,
                                          uint16_t tid) {
#ifdef USE_HOT_ELASTIC_LEAF
hot_leaf_controller_has_write_signal = true;
HotElasticLeafEntry &entry = hot_elastic_leaf_entry(page_addr);
if (entry.insert_conflicts < UINT8_MAX) {
  entry.insert_conflicts++;
}
if (entry.insert_conflicts >= kHotLeafInsertHotThreshold) {
  set_hot_leaf_state(entry, HotLeafState::InsertHot, tid);
}
#else
(void)page_addr;
(void)tid;
#endif
}

inline bool should_use_hot_leaf_optimistic_path(GlobalAddress page_addr,
                                              uint16_t tid) {
#ifdef USE_HOT_ELASTIC_LEAF
HotElasticLeafEntry &entry = hot_elastic_leaf_entry(page_addr);
if (entry.state == HotLeafState::ConflictHot) {
  if (entry.cooldown > 0) {
    entry.cooldown--;
    hot_leaf_controller_bypass_cnt[tid]++;
    return false;
  }
  entry.cas_fails = 0;
  set_hot_leaf_state(entry, HotLeafState::Normal, tid);
}
if (entry.state == HotLeafState::FoldPending) {
  hot_leaf_controller_bypass_cnt[tid]++;
  return false;
}
#else
(void)page_addr;
(void)tid;
#endif
return true;
}

inline bool should_probe_hot_leaf_optimistic_path(GlobalAddress page_addr,
                                                uint16_t tid) {
#ifdef USE_HOT_ELASTIC_LEAF
HotElasticLeafEntry &entry = hot_elastic_leaf_entry(page_addr);
if (entry.state != HotLeafState::UpdateHot &&
    entry.state != HotLeafState::ConflictHot) {
  return false;
}
if (++entry.optimistic_probe_credit < kHotLeafOptimisticProbeInterval) {
  return false;
}
entry.optimistic_probe_credit = 0;
hot_leaf_optimistic_probe_cnt[tid]++;
return true;
#else
(void)page_addr;
(void)tid;
return false;
#endif
}

inline bool hot_read_cache_allowed_for_leaf(GlobalAddress page_addr,
                                          uint16_t tid) {
#ifdef USE_HOT_ELASTIC_LEAF
HotElasticLeafEntry &entry = hot_elastic_leaf_entry(page_addr);
if (entry.state == HotLeafState::ConflictHot ||
    entry.state == HotLeafState::InsertHot ||
    entry.state == HotLeafState::FoldPending) {
  hot_leaf_cache_gate_deny_cnt[tid]++;
  return false;
}
#else
(void)page_addr;
(void)tid;
#endif
return true;
}

#ifndef HOT_READ_CACHE_TABLE_BITS
#define HOT_READ_CACHE_TABLE_BITS 12
#endif

constexpr size_t kHotReadCacheSetCount = 1ull << HOT_READ_CACHE_TABLE_BITS;
constexpr size_t kHotReadCacheAssociativity = 4;
constexpr size_t kHotReadCacheTableSize =
  kHotReadCacheSetCount * kHotReadCacheAssociativity;
static_assert((kHotReadCacheSetCount & (kHotReadCacheSetCount - 1)) == 0,
            "hot read cache set count must be a power of two");

struct HotReadCacheEntry {
Key key = kKeyMin;
uint64_t page_token = 0;
uint32_t entry_offset = 0;
};
static_assert(sizeof(HotReadCacheEntry) == 3 * sizeof(uint64_t),
            "hot read cache entry should stay compact");

#ifdef USE_HOT_READ_CACHE
thread_local HotReadCacheEntry hot_read_cache_table[kHotReadCacheTableSize];
thread_local uint8_t hot_read_cache_victim_way[kHotReadCacheSetCount];

inline size_t hot_read_cache_set(const Key &k) {
return CityHash64(reinterpret_cast<const char *>(&k), sizeof(k)) &
        (kHotReadCacheSetCount - 1);
}

inline HotReadCacheEntry &hot_read_cache_entry(size_t set, size_t way) {
return hot_read_cache_table[set * kHotReadCacheAssociativity + way];
}
#endif

inline uint64_t get_leaf_page_token(GlobalAddress page_addr) {
return (1ull << 63) | get_leaf_page_sig(page_addr) |
        (static_cast<uint64_t>(page_addr.node_version) << 56);
}

inline bool hot_read_cache_is_enabled() {
#ifdef USE_HOT_READ_CACHE
return g_hot_read_cache_enabled.load(std::memory_order_relaxed);
#else
return false;
#endif
}

inline bool hot_read_cache_lookup(const Key &k, GlobalAddress page_addr,
                                uint32_t &entry_offset, uint16_t tid) {
#ifdef USE_HOT_READ_CACHE
if (!hot_read_cache_is_enabled()) {
  return false;
}
#ifdef USE_HOT_ELASTIC_LEAF
if (hot_leaf_controller_has_write_signal &&
    !hot_read_cache_allowed_for_leaf(page_addr, tid)) {
  return false;
}
#endif
size_t set = hot_read_cache_set(k);
uint64_t expected_token = get_leaf_page_token(page_addr);
for (size_t way = 0; way < kHotReadCacheAssociativity; ++way) {
  HotReadCacheEntry &entry = hot_read_cache_entry(set, way);
  if (entry.page_token != 0 && entry.key == k &&
      entry.page_token == expected_token) {
    entry_offset = entry.entry_offset;
    return true;
  }
}
hot_read_cache_miss_cnt[tid]++;
#else
(void)k;
(void)page_addr;
(void)entry_offset;
(void)tid;
#endif
return false;
}

inline void hot_read_cache_fill(const Key &k, uint32_t entry_offset,
                              GlobalAddress page_addr, uint16_t tid) {
#ifdef USE_HOT_READ_CACHE
if (!hot_read_cache_is_enabled()) {
  return;
}
size_t set = hot_read_cache_set(k);
uint64_t expected_token = get_leaf_page_token(page_addr);
for (size_t way = 0; way < kHotReadCacheAssociativity; ++way) {
  HotReadCacheEntry &entry = hot_read_cache_entry(set, way);
  if (entry.page_token != 0 && entry.key == k &&
      entry.page_token == expected_token) {
    entry.entry_offset = entry_offset;
    hot_read_cache_fill_cnt[tid]++;
    return;
  }
}
for (size_t way = 0; way < kHotReadCacheAssociativity; ++way) {
  HotReadCacheEntry &entry = hot_read_cache_entry(set, way);
  if (entry.page_token == 0) {
    entry.key = k;
    entry.page_token = expected_token;
    entry.entry_offset = entry_offset;
    hot_read_cache_fill_cnt[tid]++;
    return;
  }
}
size_t way = hot_read_cache_victim_way[set]++ % kHotReadCacheAssociativity;
HotReadCacheEntry &entry = hot_read_cache_entry(set, way);
entry.key = k;
entry.page_token = expected_token;
entry.entry_offset = entry_offset;
hot_read_cache_fill_cnt[tid]++;
#else
(void)k;
(void)entry_offset;
(void)page_addr;
(void)tid;
#endif
}

inline void hot_read_cache_refresh(const Key &k, uint32_t entry_offset,
                                  GlobalAddress page_addr, uint16_t tid) {
#ifdef USE_HOT_READ_CACHE
if (!hot_read_cache_is_enabled()) {
  return;
}
hot_read_cache_fill(k, entry_offset, page_addr, tid);
hot_read_cache_update_cnt[tid]++;
#else
(void)k;
(void)entry_offset;
(void)page_addr;
(void)tid;
#endif
}

inline void hot_read_cache_invalidate_key(const Key &k, uint16_t tid) {
#ifdef USE_HOT_READ_CACHE
if (!hot_read_cache_is_enabled()) {
  return;
}
size_t set = hot_read_cache_set(k);
for (size_t way = 0; way < kHotReadCacheAssociativity; ++way) {
  HotReadCacheEntry &entry = hot_read_cache_entry(set, way);
  if (entry.page_token != 0 && entry.key == k) {
    entry.page_token = 0;
    hot_read_cache_invalidate_cnt[tid]++;
  }
}
#else
(void)k;
(void)tid;
#endif
}

inline void hot_read_cache_invalidate_leaf(GlobalAddress page_addr,
                                          uint16_t tid) {
#ifdef USE_HOT_READ_CACHE
if (!hot_read_cache_is_enabled()) {
  return;
}
// lazy invalidation: structural changes update the leaf address/version in
// the index path, so stale value-cache entries fail lookup by page/version.
hot_read_cache_leaf_lazy_invalidate_cnt[tid]++;
#else
(void)page_addr;
(void)tid;
#endif
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

inline bool insert_leaf_entry_for_rebuild(LeafPage *page, const Key &k,
                                        const Value &v) {
int bucket_id = key_hash_bucket(k);
if (page->group_at(bucket_id / 2)
        ->insert_for_split(k, v, !(bucket_id % 2))) {
  return true;
}
return false;
}

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
#ifdef USE_LEAF_SPLIT_GUARD
  start_leaf_split_guard_progress_thread(dsm_client_);
#endif
  print_verbose();

  index_cache = new IndexCache(define::kIndexCacheSize);

  root_ptr_ptr = get_root_ptr_ptr();

  // try to init tree and install root pointer
  char *page_buffer = (dsm_client_->get_rbuf(0)).get_page_buffer();
  GlobalAddress root_addr = dsm_client_->Alloc(kLeafPageSize);
  memset(page_buffer, 0, kLeafPageSize);
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
        printf("Lock retry count percentiles: no lock retry events recorded\n");
    } else {
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
    }
    #ifdef ENABLE_STATS
    double avg_probes = (double)probe_counts / (double)call_find_counts;
    printf("Total probes: %llu\n", (unsigned long long)probe_counts);
    printf("Total call find: %llu\n", (unsigned long long)call_find_counts);
    printf("Average probes per search: %.3f\n", avg_probes);
    #endif
    if (total_seek > 0) {
      double avg = double(total_cmp) / total_seek;
      printf("Average compares per seek: %.3f\n", avg);
    }
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

  uint64_t total_local_group_wc_owner = 0;
  uint64_t total_local_group_wc_waiter = 0;
  uint64_t total_local_group_wc_waiter_return = 0;
  uint64_t total_local_group_wc_apply = 0;
  uint64_t total_local_group_wc_nochange = 0;
  uint64_t total_local_group_wc_replace = 0;
  uint64_t total_local_group_wc_bypass = 0;
  uint64_t total_local_group_rc_owner = 0;
  uint64_t total_local_group_rc_waiter = 0;
  uint64_t total_local_group_rc_waiter_return = 0;
  uint64_t total_local_group_rc_bypass = 0;
  uint64_t total_local_group_rc_bytes_saved = 0;
  uint64_t total_local_group_rc_wait_us = 0;
  uint64_t max_local_group_rc_wait_us = 0;
  uint64_t total_split_guard_begin = 0;
  uint64_t total_split_guard_wait_event = 0;
  uint64_t total_split_guard_wait_yield = 0;
  uint64_t total_split_guard_wait_us = 0;
  uint64_t max_split_guard_wait_us = 0;
  uint64_t total_split_guard_wait_inflight_us = 0;
  uint64_t total_split_guard_wait_ack_us = 0;
  uint64_t total_split_guard_wait_both_us = 0;
  uint64_t max_split_guard_inflight = 0;

  for (int t = 0; t < MAX_APP_THREAD; ++t) {
    total_local_group_wc_owner += local_group_wc_owner_cnt[t];
    total_local_group_wc_waiter += local_group_wc_waiter_cnt[t];
    total_local_group_wc_waiter_return += local_group_wc_waiter_return_cnt[t];
    total_local_group_wc_apply += local_group_wc_apply_cnt[t];
    total_local_group_wc_nochange += local_group_wc_nochange_cnt[t];
    total_local_group_wc_replace += local_group_wc_replace_cnt[t];
    total_local_group_wc_bypass += local_group_wc_bypass_cnt[t];
    total_local_group_rc_owner += local_group_rc_owner_cnt[t];
    total_local_group_rc_waiter += local_group_rc_waiter_cnt[t];
    total_local_group_rc_waiter_return += local_group_rc_waiter_return_cnt[t];
    total_local_group_rc_bypass += local_group_rc_bypass_cnt[t];
    total_local_group_rc_bytes_saved += local_group_rc_bytes_saved[t];
    total_local_group_rc_wait_us += local_group_rc_wait_us_sum[t];
    if (local_group_rc_wait_us_max[t] > max_local_group_rc_wait_us) {
      max_local_group_rc_wait_us = local_group_rc_wait_us_max[t];
    }
    total_split_guard_begin += split_guard_begin_cnt[t];
    total_split_guard_wait_event += split_guard_wait_event_cnt[t];
    total_split_guard_wait_yield += split_guard_wait_yield_cnt[t];
    total_split_guard_wait_us += split_guard_wait_us_sum[t];
    total_split_guard_wait_inflight_us += split_guard_wait_inflight_us_sum[t];
    total_split_guard_wait_ack_us += split_guard_wait_ack_us_sum[t];
    total_split_guard_wait_both_us += split_guard_wait_both_us_sum[t];
    if (split_guard_wait_us_max[t] > max_split_guard_wait_us) {
      max_split_guard_wait_us = split_guard_wait_us_max[t];
    }
    if (split_guard_inflight_max[t] > max_split_guard_inflight) {
      max_split_guard_inflight = split_guard_inflight_max[t];
    }
  }

  double avg_split_guard_wait_us =
      total_split_guard_begin == 0
          ? 0.0
          : static_cast<double>(total_split_guard_wait_us) /
                total_split_guard_begin;

  printf("\n=== Local Group Write Combine Stats ===\n");
  printf("  Owner entries:        %llu\n",
         (unsigned long long)total_local_group_wc_owner);
  printf("  Waiter entries:       %llu\n",
         (unsigned long long)total_local_group_wc_waiter);
  printf("  Waiter direct returns:%llu\n",
         (unsigned long long)total_local_group_wc_waiter_return);
  printf("  Applied values:       %llu\n",
         (unsigned long long)total_local_group_wc_apply);
  printf("  No-change applies:    %llu\n",
         (unsigned long long)total_local_group_wc_nochange);
  printf("  Slot replacements:    %llu\n",
         (unsigned long long)total_local_group_wc_replace);
  printf("  Bypass entries:       %llu\n",
         (unsigned long long)total_local_group_wc_bypass);

  double avg_local_group_rc_wait_us =
      total_local_group_rc_waiter_return == 0
          ? 0.0
          : static_cast<double>(total_local_group_rc_wait_us) /
                total_local_group_rc_waiter_return;
  printf("\n=== Local Group Read Combine Stats ===\n");
  printf("  Owner reads:          %llu\n",
         (unsigned long long)total_local_group_rc_owner);
  printf("  Waiter reads:         %llu\n",
         (unsigned long long)total_local_group_rc_waiter);
  printf("  Waiter returns:       %llu\n",
         (unsigned long long)total_local_group_rc_waiter_return);
  printf("  Bypass reads:         %llu\n",
         (unsigned long long)total_local_group_rc_bypass);
  printf("  Bytes saved:          %llu\n",
         (unsigned long long)total_local_group_rc_bytes_saved);
  printf("  Avg waiter wait:      %.3f us\n",
         avg_local_group_rc_wait_us);
  printf("  Max waiter wait:      %llu us\n",
         (unsigned long long)max_local_group_rc_wait_us);

  printf("\n=== Split Guard Wait Stats ===\n");
  printf("  Begin count:          %llu\n",
         (unsigned long long)total_split_guard_begin);
  printf("  Wait events:          %llu\n",
         (unsigned long long)total_split_guard_wait_event);
  printf("  Wait yields/iters:    %llu\n",
         (unsigned long long)total_split_guard_wait_yield);
  printf("  Avg total wait:       %.3f us / begin\n",
         avg_split_guard_wait_us);
  printf("  Max total wait:       %llu us\n",
         (unsigned long long)max_split_guard_wait_us);
  printf("  Wait inflight only:   %llu us\n",
         (unsigned long long)total_split_guard_wait_inflight_us);
  printf("  Wait ACK only:        %llu us\n",
         (unsigned long long)total_split_guard_wait_ack_us);
  printf("  Wait both:            %llu us\n",
         (unsigned long long)total_split_guard_wait_both_us);
  printf("  Max inflight seen:    %llu\n",
         (unsigned long long)max_split_guard_inflight);

  uint64_t ack_after_wait =
      split_guard_ack_sent_after_wait_cnt.load(std::memory_order_relaxed);
  double avg_ack_queue_wait_us =
      ack_after_wait == 0
          ? 0.0
          : static_cast<double>(
                split_guard_ack_queue_wait_us_sum.load(
                    std::memory_order_relaxed)) /
                ack_after_wait;

  printf("\n=== Split Guard ACK Delay Stats ===\n");
  printf("  Immediate ACKs:       %llu\n",
         (unsigned long long)split_guard_ack_immediate_cnt.load(
             std::memory_order_relaxed));
  printf("  Queued ACKs:          %llu\n",
         (unsigned long long)split_guard_ack_queued_cnt.load(
             std::memory_order_relaxed));
  printf("  ACKs after wait:      %llu\n",
         (unsigned long long)ack_after_wait);
  printf("  Flush blocked checks: %llu\n",
         (unsigned long long)split_guard_ack_flush_blocked_cnt.load(
             std::memory_order_relaxed));
  printf("  Avg ACK queue wait:   %.3f us\n",
         avg_ack_queue_wait_us);
  printf("  Max ACK queue wait:   %llu us\n",
         (unsigned long long)split_guard_ack_queue_wait_us_max.load(
             std::memory_order_relaxed));
  printf("  Max pending ACKs:     %llu\n",
         (unsigned long long)split_guard_ack_pending_max.load(
             std::memory_order_relaxed));
  printf("  Max blocking inflight:%llu\n",
         (unsigned long long)split_guard_ack_blocking_inflight_max.load(
             std::memory_order_relaxed));
  printf("================================\n");

  uint64_t total_insert_rtt = 0, total_insert_bytes = 0, total_insert_ops = 0;
  uint64_t total_search_rtt = 0, total_search_bytes = 0, total_search_ops = 0;
  uint64_t total_leaf_update_hit = 0, total_leaf_insert_empty = 0;
  uint64_t total_leaf_insert_group_fast = 0;
  uint64_t total_leaf_insert_page_lock = 0;
  uint64_t total_leaf_insert_retry_event = 0;
  uint64_t total_leaf_insert_retry_step = 0;
  uint64_t total_sxlock_update_cas_fail = 0;
  uint64_t total_sxlock_update_cas_retry_event = 0;
  uint64_t total_sxlock_update_cas_retry_step = 0;
  uint64_t total_sxlock_insert_cas_fail = 0;
  uint64_t total_leaf_upgrade_to_x = 0, total_leaf_split = 0;
  uint64_t total_leaf_split_occupancy = 0;
  uint64_t max_leaf_split_occupancy = 0;
  uint64_t total_leaf_sibling_chase = 0;
  uint64_t total_leaf_insert_parent_update = 0;
  uint64_t total_leaf_insert_root_split = 0;
  uint64_t total_optimistic_update_attempt = 0;
  uint64_t total_optimistic_update_success = 0;
  uint64_t total_optimistic_update_cas_fail = 0;
  uint64_t total_optimistic_update_cas_retry_event = 0;
  uint64_t total_optimistic_update_cas_retry_step = 0;
  uint64_t total_optimistic_update_cas_retry_exhaust = 0;
  uint64_t total_optimistic_update_split_abort = 0;
  uint64_t total_optimistic_leaf_fast_path_hot_bypass = 0;
  uint64_t total_optimistic_insert_attempt = 0;
  uint64_t total_optimistic_insert_success = 0;
  uint64_t total_optimistic_insert_cas_fail = 0;
  uint64_t total_optimistic_insert_split_abort = 0;
  uint64_t total_optimistic_insert_consistency_fail = 0;
  uint64_t total_optimistic_insert_fallback = 0;
  uint64_t total_leaf_group_reread_too_many = 0;
  uint64_t total_leaf_group_reread_fallback = 0;
#ifdef USE_HOT_READ_CACHE
  uint64_t total_hot_read_cache_hit = 0;
  uint64_t total_hot_read_cache_miss = 0;
  uint64_t total_hot_read_cache_fill = 0;
  uint64_t total_hot_read_cache_update = 0;
  uint64_t total_hot_read_cache_invalidate = 0;
  uint64_t total_hot_read_cache_leaf_lazy_invalidate = 0;
#endif
  uint64_t total_hot_leaf_update_hot_transition = 0;
  uint64_t total_hot_leaf_conflict_hot_transition = 0;
  uint64_t total_hot_leaf_insert_hot_transition = 0;
  uint64_t total_hot_leaf_fold_pending_transition = 0;
  uint64_t total_hot_leaf_controller_bypass = 0;
  uint64_t total_hot_leaf_cache_gate_deny = 0;
  uint64_t total_hot_leaf_optimistic_probe = 0;

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
    total_sxlock_update_cas_fail += sxlock_update_cas_fail_cnt[t];
    total_sxlock_update_cas_retry_event += sxlock_update_cas_retry_event_cnt[t];
    total_sxlock_update_cas_retry_step += sxlock_update_cas_retry_step_cnt[t];
    total_sxlock_insert_cas_fail += sxlock_insert_cas_fail_cnt[t];
    total_leaf_upgrade_to_x += leaf_upgrade_to_x_cnt[t];
    total_leaf_split += leaf_split_cnt[t];
    total_leaf_split_occupancy += leaf_split_occupancy_sum[t];
    if (leaf_split_occupancy_max[t] > max_leaf_split_occupancy) {
      max_leaf_split_occupancy = leaf_split_occupancy_max[t];
    }
    total_leaf_sibling_chase += leaf_sibling_chase_cnt[t];
    total_leaf_insert_parent_update += leaf_insert_parent_update_cnt[t];
    total_leaf_insert_root_split += leaf_insert_root_split_cnt[t];
    total_optimistic_update_attempt += optimistic_update_attempt_cnt[t];
    total_optimistic_update_success += optimistic_update_success_cnt[t];
    total_optimistic_update_cas_fail += optimistic_update_cas_fail_cnt[t];
    total_optimistic_update_cas_retry_event +=
        optimistic_update_cas_retry_event_cnt[t];
    total_optimistic_update_cas_retry_step +=
        optimistic_update_cas_retry_step_cnt[t];
    total_optimistic_update_cas_retry_exhaust +=
        optimistic_update_cas_retry_exhaust_cnt[t];
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
    total_leaf_group_reread_too_many += leaf_group_reread_too_many_cnt[t];
    total_leaf_group_reread_fallback += leaf_group_reread_fallback_cnt[t];
#ifdef USE_HOT_READ_CACHE
    total_hot_read_cache_hit += hot_read_cache_hit_cnt[t];
    total_hot_read_cache_miss += hot_read_cache_miss_cnt[t];
    total_hot_read_cache_fill += hot_read_cache_fill_cnt[t];
    total_hot_read_cache_update += hot_read_cache_update_cnt[t];
    total_hot_read_cache_invalidate += hot_read_cache_invalidate_cnt[t];
    total_hot_read_cache_leaf_lazy_invalidate +=
        hot_read_cache_leaf_lazy_invalidate_cnt[t];
#endif
    total_hot_leaf_update_hot_transition +=
        hot_leaf_update_hot_transition_cnt[t];
    total_hot_leaf_conflict_hot_transition +=
        hot_leaf_conflict_hot_transition_cnt[t];
    total_hot_leaf_insert_hot_transition +=
        hot_leaf_insert_hot_transition_cnt[t];
    total_hot_leaf_fold_pending_transition +=
        hot_leaf_fold_pending_transition_cnt[t];
    total_hot_leaf_controller_bypass += hot_leaf_controller_bypass_cnt[t];
    total_hot_leaf_cache_gate_deny += hot_leaf_cache_gate_deny_cnt[t];
    total_hot_leaf_optimistic_probe += hot_leaf_optimistic_probe_cnt[t];
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
    printf("\n=== SXLOCK Leaf CAS Failure Breakdown ===\n");
    printf("  Update CAS failures:  %llu (%.4f / insert)\n",
          (unsigned long long)total_sxlock_update_cas_fail,
          (double)total_sxlock_update_cas_fail / total_insert_ops);
    printf("  Update retry events:  %llu (%.4f / insert)\n",
          (unsigned long long)total_sxlock_update_cas_retry_event,
          (double)total_sxlock_update_cas_retry_event / total_insert_ops);
    printf("  Update retry steps:   %llu (%.4f / insert)\n",
          (unsigned long long)total_sxlock_update_cas_retry_step,
          (double)total_sxlock_update_cas_retry_step / total_insert_ops);
    printf("  Insert CAS failures:  %llu (%.4f / insert)\n",
          (unsigned long long)total_sxlock_insert_cas_fail,
          (double)total_sxlock_insert_cas_fail / total_insert_ops);
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
    printf("  CAS retry events:     %llu (%.4f / insert)\n",
          (unsigned long long)total_optimistic_update_cas_retry_event,
          (double)total_optimistic_update_cas_retry_event / total_insert_ops);
    printf("  CAS retry steps:      %llu (%.4f / insert)\n",
          (unsigned long long)total_optimistic_update_cas_retry_step,
          (double)total_optimistic_update_cas_retry_step / total_insert_ops);
    printf("  CAS retry exhausts:   %llu (%.4f / insert)\n",
          (unsigned long long)total_optimistic_update_cas_retry_exhaust,
          (double)total_optimistic_update_cas_retry_exhaust / total_insert_ops);
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
    printf("  Leaf group reread too-many: %llu (%.6f / insert)\n",
          (unsigned long long)total_leaf_group_reread_too_many,
          (double)total_leaf_group_reread_too_many / total_insert_ops);
    printf("  Leaf group reread fallbacks: %llu (%.6f / insert)\n",
          (unsigned long long)total_leaf_group_reread_fallback,
          (double)total_leaf_group_reread_fallback / total_insert_ops);
    printf("================================\n");
  }
#ifdef USE_HOT_READ_CACHE
  printf("\n=== Hot Read Cache Stats ===\n");
  printf("  Hot read cache hits:          %llu\n",
        (unsigned long long)total_hot_read_cache_hit);
  printf("  Hot read cache misses:        %llu\n",
        (unsigned long long)total_hot_read_cache_miss);
  printf("  Hot read cache fills:         %llu\n",
        (unsigned long long)total_hot_read_cache_fill);
  printf("  Hot read cache updates:       %llu\n",
        (unsigned long long)total_hot_read_cache_update);
  printf("  Hot read cache invalidations: %llu\n",
        (unsigned long long)total_hot_read_cache_invalidate);
  printf("  Hot read cache leaf lazy invalidations: %llu\n",
        (unsigned long long)total_hot_read_cache_leaf_lazy_invalidate);
  printf("================================\n");
#endif
  printf("\n=== Hot Elastic Leaf Stats ===\n");
  printf("  Hot leaf update-hot transitions:   %llu\n",
        (unsigned long long)total_hot_leaf_update_hot_transition);
  printf("  Hot leaf conflict-hot transitions: %llu\n",
        (unsigned long long)total_hot_leaf_conflict_hot_transition);
  printf("  Hot leaf insert-hot transitions:   %llu\n",
        (unsigned long long)total_hot_leaf_insert_hot_transition);
  printf("  Hot leaf fold-pending transitions: %llu\n",
        (unsigned long long)total_hot_leaf_fold_pending_transition);
  printf("  Hot leaf controller bypasses:      %llu\n",
        (unsigned long long)total_hot_leaf_controller_bypass);
  printf("  Hot leaf cache gate denies:        %llu\n",
        (unsigned long long)total_hot_leaf_cache_gate_deny);
  printf("  Hot leaf optimistic probes:        %llu\n",
        (unsigned long long)total_hot_leaf_optimistic_probe);
  printf("================================\n");
}

void Tree::set_prefill_split_stats(bool enabled) {
  g_prefill_split_stats_enabled.store(enabled, std::memory_order_relaxed);
}

void Tree::set_hot_read_cache_enabled(bool enabled) {
#ifdef USE_HOT_READ_CACHE
  g_hot_read_cache_enabled.store(enabled, std::memory_order_relaxed);
#else
  (void)enabled;
#endif
}

void Tree::clear_hot_read_cache_local() {
#ifdef USE_HOT_READ_CACHE
  memset(hot_read_cache_table, 0, sizeof(hot_read_cache_table));
  memset(hot_read_cache_victim_way, 0, sizeof(hot_read_cache_victim_way));
#endif
#ifdef USE_HOT_ELASTIC_LEAF
  hot_leaf_controller_has_write_signal = false;
#endif
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

void Tree::poll_split_guard_messages() {
  drain_leaf_split_guard_messages(dsm_client_);
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
#if defined(USE_SPLIT_ONLY_X_LOCK)
  assert(!share_lock && "S unlock is disabled by USE_SPLIT_ONLY_X_LOCK");
#endif
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
#if defined(USE_SPLIT_ONLY_X_LOCK)
  // share_lock path becomes lock-free. Split safety must be provided by
  // USE_LEAF_SPLIT_GUARD at the caller side.
  if (share_lock) {
    assert(!upgrade_from_s);
    return;
  }
  acquire_sx_lock(lock_addr, lock_buffer, ctx, false, false);
#elif defined(USE_SX_LOCK)
  acquire_sx_lock(lock_addr, lock_buffer, ctx, share_lock, upgrade_from_s);
#else
  try_lock_addr(lock_addr, lock_buffer, ctx);
#endif
}

inline void Tree::release_lock(GlobalAddress lock_addr, uint64_t *lock_buffer,
                              CoroContext *ctx, bool async, bool share_lock) {
#if defined(USE_SPLIT_ONLY_X_LOCK)
  if (share_lock) {
    return;
  }
  release_sx_lock(lock_addr, lock_buffer, ctx, async, false);
#elif defined(USE_SX_LOCK)
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

#if defined(USE_SPLIT_ONLY_X_LOCK)
  dsm_client_->Write(write_buffer, write_addr, write_size, false);
  track_rdma(dsm_client_->get_my_thread_id(), 1, write_size);

  // sx_lock == true means old S-lock path. There is no S unlock now.
  // sx_lock == false means X-lock path, still release X.
  if (!sx_lock) {
    release_sx_lock(lock_addr, cas_buffer, ctx, async, false);
  }

#elif defined(USE_SX_LOCK)
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

#if defined(USE_SPLIT_ONLY_X_LOCK)
  dsm_client_->CasMask(cas_addr, log_cas_size, equal, swap, cas_buffer, mask,
                       false);
  track_rdma(dsm_client_->get_my_thread_id(), 1, (1 << log_cas_size));

  // Old S-lock path: no unlock.
  // X-lock path: still release X.
  if (!share_lock) {
    release_sx_lock(lock_addr, lock_buffer, ctx, async, false);
  }

#elif defined(USE_SX_LOCK)
  dsm_client_->CasMask(cas_addr, log_cas_size, equal, swap, cas_buffer, mask,
                       false);
  track_rdma(dsm_client_->get_my_thread_id(), 1, (1 << log_cas_size));
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
#if defined(USE_SPLIT_ONLY_X_LOCK)
  if (share_lock) {
    assert(!upgrade_from_s);

    Timer timer;
    timer.begin();

    dsm_client_->ReadSync(read_buffer, read_addr, read_size, ctx);
    track_rdma(my_thread_id, 1, read_size);

    uint64_t t = timer.end();
    stat_helper.add(dsm_client_->get_my_thread_id(), lat_read_page, t);
    return;
  }
#endif

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
    std::cout << "SEARCH WARNING insert1" << std::endl;
    p = get_root_ptr(ctx);
    level_hint = -1;
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
#ifdef USE_HOT_READ_CACHE
      uint32_t hot_entry_offset = 0;
      if (hot_read_cache_lookup(k, p, hot_entry_offset,
                                dsm_client_->get_my_thread_id())) {
        char *hot_entry_buffer =
            dsm_client_->get_rbuf(ctx ? ctx->coro_id : 0).get_page_buffer();
        LeafEntry *hot_entry = reinterpret_cast<LeafEntry *>(hot_entry_buffer);
        dsm_client_->ReadSync(hot_entry_buffer, GADD(p, hot_entry_offset),
                              sizeof(LeafEntry), ctx);
        track_rdma(dsm_client_->get_my_thread_id(), 1, sizeof(LeafEntry));
        if (hot_entry->key == k && hot_entry->lv.val != kValueNull) {
          v = hot_entry->lv.val;
          hot_read_cache_hit_cnt[dsm_client_->get_my_thread_id()]++;
          return true;
        }
        hot_read_cache_invalidate_key(k, dsm_client_->get_my_thread_id());
      }
#endif
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
    }
    min = kKeyMin;
    max = kKeyMax;
    goto next;
  }
  if (result.is_leaf) {
    if (result.val != kValueNull) {  // find
      v = result.val;
#ifdef USE_HOT_READ_CACHE
      hot_read_cache_fill(k, result.leaf_entry_offset, p,
                          dsm_client_->get_my_thread_id());
#endif
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
    }
  }

  index_cache->thread_status->rcu_exit(dsm_client_->get_my_thread_id());
  return counter;
}

void Tree::del(const Key &k, CoroContext *ctx, int coro_id) {
  assert(dsm_client_->IsRegistered());

  before_operation(ctx, coro_id);
#ifdef USE_HOT_READ_CACHE
  hot_read_cache_invalidate_key(k, dsm_client_->get_my_thread_id());
#endif
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
  uint16_t tid = dsm_client_->get_my_thread_id();

  auto fallback_full_leaf_search = [&]() -> bool {
    leaf_group_reread_fallback_cnt[tid]++;
    dsm_client_->ReadSync(page_buffer, page_addr, kLeafPageSize, ctx);
    track_rdma(tid, 1, kLeafPageSize);

    LeafPage *page = reinterpret_cast<LeafPage *>(page_buffer);
    Header *full_hdr = &page->hdr;
    result.clear();
    result.is_leaf = true;
    result.level = 0;

    if (k >= full_hdr->highest) {
      if (from_cache) {
        return false;
      }
      result.sibling = full_hdr->sibling_ptr;
      result.next_min = full_hdr->highest;
      return true;
    }
    if (k < full_hdr->lowest) {
      return !from_cache;
    }

    int fallback_bucket_id = key_hash_bucket(k);
    int fallback_group_id = fallback_bucket_id / 2;
    LeafEntryGroup *fallback_group = page->group_at(fallback_group_id);
    uint32_t fallback_group_offset =
        offsetof(LeafPage, groups) +
        sizeof(LeafEntryGroup) * fallback_group_id;
    bool found = fallback_group->find(k, result, !(fallback_bucket_id % 2),
                                      fallback_group_offset);
    return true;
  };

  int read_counter = 0;
re_read:
  if (++read_counter > 10) {
    printf("re-read (leaf_page_group) too many times\n");
    leaf_group_reread_too_many_cnt[tid]++;
    mark_hot_leaf_fast_path_conflict(page_addr);
    return fallback_full_leaf_search();
  }

  if (has_header) {
    dsm_client_->Read(page_buffer + bucket_offset,
                      GADD(page_addr, bucket_offset), kReadBucketSize, false);
    track_rdma(tid, 1, kReadBucketSize); // 📝【埋点：异步读 Bucket】
    // read header
    dsm_client_->ReadSync(page_buffer + header_offset,
                          GADD(page_addr, header_offset), sizeof(Header), ctx);
    track_rdma(tid, 1, sizeof(Header));  // 📝【埋点：同步读 Header】          
  } else {
#ifdef USE_LOCAL_GROUP_READ_COMBINE
    LeafSplitGuardEntry *read_combine_entry =
        get_leaf_split_guard_entry(page_addr);
    LocalGroupReadCombineToken read_combine_token =
        begin_local_group_read_combine(read_combine_entry, group_id,
                                       !(bucket_id % 2), tid);
    if (read_combine_token.valid && !read_combine_token.owner) {
      wait_for_local_group_read_combine(read_combine_token, ctx, tid);
      if (!copy_local_group_read_combine_result(
              read_combine_token, page_buffer + bucket_offset, kReadBucketSize,
              tid)) {
        dsm_client_->ReadSync(page_buffer + bucket_offset,
                              GADD(page_addr, bucket_offset), kReadBucketSize,
                              ctx);
        track_rdma(tid, 1, kReadBucketSize);
      }
    } else {
      dsm_client_->ReadSync(page_buffer + bucket_offset,
                            GADD(page_addr, bucket_offset), kReadBucketSize,
                            ctx);
      track_rdma(tid, 1, kReadBucketSize); // 📝【埋点：同步读 Bucket】
      if (read_combine_token.valid && read_combine_token.owner) {
        finish_local_group_read_combine(read_combine_token,
                                        page_buffer + bucket_offset,
                                        kReadBucketSize);
      }
    }
#else
    dsm_client_->ReadSync(page_buffer + bucket_offset,
                          GADD(page_addr, bucket_offset), kReadBucketSize, ctx);
    track_rdma(tid, 1, kReadBucketSize); // 📝【埋点：同步读 Bucket】
#endif
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
  bool res = group->find(k, result, !(bucket_id % 2), group_offset);
  if (!res) {
    if (!has_header) {
      dsm_client_->ReadSync(page_buffer + header_offset,
                            GADD(page_addr, header_offset), sizeof(Header),
                            ctx);
      track_rdma(tid, 1, sizeof(Header));
      has_header = true;
    }
    if (from_cache) {
      if (k >= hdr->highest || k < hdr->lowest) {
        return false;
      }
    }
    return true;
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
    uint32_t leaf_group_offset =
        offsetof(LeafPage, groups) +
        sizeof(LeafEntryGroup) * (bucket_id / 2);
    bool res = g->find(k, result, !(bucket_id % 2), leaf_group_offset);
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
#if defined(USE_SPLIT_ONLY_X_LOCK)
  // internal_page_update() does direct remote Write to internal pointer slots.
  // Do not make it lock-free unless you rewrite it to CAS/version protocol.
  // Treat this as split propagation and use X lock.
  internal_page_update(p, k, v, level, ctx, false);
#elif defined(USE_SX_LOCK)
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
#if defined(USE_SPLIT_ONLY_X_LOCK)
        internal_page_update(up_level, page->hdr.lowest, page_addr, level + 1,
                            ctx, false);
#else
        internal_page_update(up_level, page->hdr.lowest, page_addr, level + 1,
                            ctx, true);
#endif
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

  auto sxlock_leaf_value_cas = [&](GlobalAddress cas_addr, uint64_t equal,
                                    uint64_t swap) -> bool {
    bool cas_ok = dsm_client_->CasSync(cas_addr, equal, swap, cas_ret_buffer, ctx);
    track_rdma(tid, 1, sizeof(uint64_t));
    return cas_ok;
  };
#if defined(USE_SPLIT_ONLY_X_LOCK)
  bool leaf_read_guard_active = false;

  auto enter_leaf_write_guard = [&]() {
    if (share_lock && !leaf_read_guard_active) {
      wait_until_enter_leaf_split_guard(dsm_client_, page_addr, ctx);
      leaf_read_guard_active = true;
    }
  };

  auto leave_leaf_write_guard = [&]() {
    if (leaf_read_guard_active) {
      leave_leaf_split_guard(dsm_client_, page_addr);
      leaf_read_guard_active = false;
    }
  };

  auto finish_leaf_access = [&](bool async) {
    if (share_lock) {
      leave_leaf_write_guard();
    } else {
      release_lock(lock_addr, lock_buffer, ctx, async, false);
    }
  };
#else
  auto enter_leaf_write_guard = [&]() {};
  auto leave_leaf_write_guard = [&]() {};
  auto finish_leaf_access = [&](bool async) {
    release_lock(lock_addr, lock_buffer, ctx, async, share_lock);
  };
#endif
  constexpr int kSxLockCasRetryLimit = 10;

  // try upsert hash group
#ifdef FINE_GRAINED_LEAF_NODE
  size_t bucket_offset =
      group_offset + (bucket_id % 2 ? kBackOffset : kFrontOffset);
#ifdef USE_OPTIMISTIC_UPDATE_HIT
  bool optimistic_bucket_consistent = false;
  bool optimistic_update_candidate = false;
#ifndef USE_OPTIMISTIC_GUARD_RETRY
  constexpr int kOptimisticUpdateRetry = 2;
  bool controller_allows_optimistic_leaf_fast_path =
      should_use_hot_leaf_optimistic_path(page_addr, tid);
  bool legacy_hot_bypass = should_bypass_optimistic_leaf_fast_path(page_addr);
  bool probe_optimistic_leaf_fast_path =
      legacy_hot_bypass && controller_allows_optimistic_leaf_fast_path &&
      should_probe_hot_leaf_optimistic_path(page_addr, tid);
  bool bypass_optimistic_leaf_fast_path =
      !controller_allows_optimistic_leaf_fast_path ||
      (legacy_hot_bypass && !probe_optimistic_leaf_fast_path);
  if (bypass_optimistic_leaf_fast_path) {
    optimistic_leaf_fast_path_hot_bypass_cnt[tid]++;
  }
#else
  bool bypass_optimistic_leaf_fast_path = false;
#endif

#ifdef USE_OPTIMISTIC_GUARD_RETRY
  constexpr uint32_t kOptimisticInsertRetry = 10;
  uint32_t optimistic_insert_retry_cnt = 0;
  for (uint32_t optimistic_retry = 0;
      !bypass_optimistic_leaf_fast_path;
      ++optimistic_retry) {
    LeafSplitGuardEntry *optimistic_group_guard_entry =
        get_leaf_split_guard_entry(page_addr);
#ifdef USE_LOCAL_GROUP_WRITE_COMBINE
    LocalGroupWriteCombineToken optimistic_wc_token =
        begin_local_group_write_combine(optimistic_group_guard_entry, group_id,
                                        k, v, tid);
    if (optimistic_wc_token.valid && !optimistic_wc_token.owner) {
      wait_for_local_group_write_combine(optimistic_wc_token, ctx);
      if (valid_app_tid(tid)) {
        local_group_wc_waiter_return_cnt[tid]++;
      }
      return true;
    }
#endif
    wait_until_enter_leaf_split_guard(dsm_client_, page_addr, ctx);
    
#else
  for (int optimistic_retry = 0;
      !bypass_optimistic_leaf_fast_path &&
      optimistic_retry < kOptimisticUpdateRetry;
      ++optimistic_retry) {
    LeafSplitGuardEntry *optimistic_group_guard_entry =
        get_leaf_split_guard_entry(page_addr);
#ifdef USE_LOCAL_GROUP_WRITE_COMBINE
    LocalGroupWriteCombineToken optimistic_wc_token =
        begin_local_group_write_combine(optimistic_group_guard_entry, group_id,
                                        k, v, tid);
    if (optimistic_wc_token.valid && !optimistic_wc_token.owner) {
      wait_for_local_group_write_combine(optimistic_wc_token, ctx);
      if (valid_app_tid(tid)) {
        local_group_wc_waiter_return_cnt[tid]++;
      }
      return true;
    }
#endif
    if (!enter_leaf_split_guard(dsm_client_, page_addr)) {
#ifdef USE_LOCAL_GROUP_WRITE_COMBINE
      finish_local_group_write_combine(optimistic_wc_token);
#endif
      optimistic_insert_consistency_fail_cnt[tid]++;
      mark_hot_leaf_fast_path_conflict(page_addr);
      break;
    }
#endif
    bool optimistic_guard_active = true;
    auto leave_optimistic_guard = [&]() {
      if (optimistic_guard_active) {
        leave_leaf_split_guard(dsm_client_, page_addr);
#ifdef USE_LOCAL_GROUP_WRITE_COMBINE
        finish_local_group_write_combine(optimistic_wc_token);
#endif
        optimistic_guard_active = false;
      }
    };
    dsm_client_->ReadSync(page_buffer + bucket_offset,
                          GADD(page_addr, bucket_offset), kReadBucketSize,
                          ctx);
    track_rdma(tid, 1, kReadBucketSize);

    if (!group->check_consistency(!(bucket_id % 2), page_addr.node_version,
                                  actual_version)) {
      optimistic_insert_consistency_fail_cnt[tid]++;
      record_hot_leaf_consistency_fail(page_addr, tid);
      mark_hot_leaf_fast_path_conflict(page_addr);
      if (from_cache) {
        leave_optimistic_guard();
        return false;
      }
      leave_optimistic_guard();
#ifdef USE_OPTIMISTIC_GUARD_RETRY
      return false;
#else
      break;
#endif
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

        Value combined_insert_value = v;
#ifdef USE_LOCAL_GROUP_WRITE_COMBINE
        consume_local_group_write_combine(optimistic_group_guard_entry,
                                          group_id, k,
                                          combined_insert_value, tid);
#endif
        uint64_t *swap_buffer = rbuf.get_cas_buffer();
        LeafEntry *swap_entry = reinterpret_cast<LeafEntry *>(swap_buffer);
        swap_entry->key = k;
        swap_entry->lv.cl_ver = insert_addr->lv.cl_ver;
        swap_entry->lv.val = combined_insert_value;

        uint64_t *mask_buffer = rbuf.get_cas_buffer();
        mask_buffer[0] = mask_buffer[1] = ~0ull;

        bool insert_cas_ok = dsm_client_->CasMaskSync(
            GADD(page_addr, reinterpret_cast<char *>(insert_addr) - page_buffer),
            4, reinterpret_cast<uint64_t>(insert_addr),
            reinterpret_cast<uint64_t>(swap_buffer), cas_ret_buffer,
            reinterpret_cast<uint64_t>(mask_buffer), ctx);

        track_rdma(tid, 1, sizeof(LeafEntry));

        if (insert_cas_ok || (__bswap_64(cas_ret_buffer[0]) == k)) {
          optimistic_insert_success_cnt[tid]++;
          leaf_insert_empty_cnt[tid]++;
          insert_addr->key = k;
          insert_addr->lv.cl_ver = swap_entry->lv.cl_ver;
          insert_addr->lv.val = v;
          clear_hot_leaf_fast_path_conflict(page_addr);
          leave_optimistic_guard();
          return true;
        }

        optimistic_insert_cas_fail_cnt[tid]++;
        record_hot_leaf_cas_fail(page_addr, tid);
        mark_hot_leaf_fast_path_conflict(page_addr);

        // big-endian for 16-byte CAS return
        insert_addr->key = __bswap_64(cas_ret_buffer[0]);
        insert_addr->lv.raw = __bswap_64(cas_ret_buffer[1]);

    #ifdef USE_OPTIMISTIC_GUARD_RETRY
        if (++optimistic_insert_retry_cnt <= kOptimisticInsertRetry) {
          leave_optimistic_guard();
          continue;
        }
    #endif

        optimistic_insert_fallback_cnt[tid]++;
        leave_optimistic_guard();
        break;
    #else
        optimistic_insert_fallback_cnt[tid]++;
        leave_optimistic_guard();
        break;
    #endif
      }

      optimistic_insert_fallback_cnt[tid]++;
      leave_optimistic_guard();
      break;
    }

    optimistic_update_attempt_cnt[tid]++;

    Value combined_update_value = v;
#ifdef USE_LOCAL_GROUP_WRITE_COMBINE
    consume_local_group_write_combine(optimistic_group_guard_entry, group_id,
                                      k, combined_update_value, tid);
#endif
    constexpr int kOptimisticUpdateCasRetryLimit = 10;
    int optimistic_update_retry_cnt = 0;
    while (true) {
      LeafValue cas_val(update_addr->lv.cl_ver, combined_update_value);
      bool update_cas_ok = dsm_client_->CasSync(
          GADD(page_addr, ((char *)&update_addr->lv - page_buffer)),
          update_addr->lv.raw, cas_val.raw, cas_ret_buffer, ctx);
      track_rdma(tid, 1, sizeof(LeafValue));
      if (update_cas_ok) {
        break;
      }

      optimistic_update_cas_fail_cnt[tid]++;
      record_hot_leaf_cas_fail(page_addr, tid);
      mark_hot_leaf_fast_path_conflict(page_addr);
      update_addr->lv.raw = cas_ret_buffer[0];
      if (optimistic_update_retry_cnt == 0) {
        optimistic_update_cas_retry_event_cnt[tid]++;
      }
      optimistic_update_cas_retry_step_cnt[tid]++;
      if (++optimistic_update_retry_cnt <= kOptimisticUpdateCasRetryLimit) {
        optimistic_update_retry_pause(optimistic_update_retry_cnt);
        continue;
      }

      optimistic_update_cas_retry_exhaust_cnt[tid]++;
      leave_optimistic_guard();
      return true;
    }
    clear_hot_leaf_fast_path_conflict(page_addr);
    optimistic_update_success_cnt[tid]++;
    leaf_update_hit_cnt[tid]++;
    record_hot_leaf_update_success(page_addr, tid);
    leave_optimistic_guard();
    return true;
  }
#endif
#ifdef USE_SPLIT_ONLY_X_LOCK
  enter_leaf_write_guard();
#endif
  // 1. lock, read, and check consistency
  lock_and_read(lock_addr, share_lock, false, lock_buffer,
                GADD(page_addr, bucket_offset), kReadBucketSize,
                page_buffer + bucket_offset, ctx);
#ifdef USE_HOT_READ_CACHE
  hot_read_cache_invalidate_key(k, tid);
#endif
  hold_x_lock = !share_lock;
  leaf_insert_path_page_lock_cnt[tid]++;

  if (!group->check_consistency(!(bucket_id % 2), page_addr.node_version,
                                actual_version)) {
    if (from_cache) {
#ifdef USE_SPLIT_ONLY_X_LOCK
      finish_leaf_access(true);
#else
      release_lock(lock_addr, lock_buffer, ctx, true, share_lock);
#endif
      return false;
    } else {
      // lock-based, no need to re-read, just read header to check sibling
      dsm_client_->ReadSync(page_buffer + offsetof(LeafPage, hdr),
                            GADD(page_addr, offsetof(LeafPage, hdr)),
                            sizeof(Header), ctx);
      if (k >= header->highest) {
        leaf_sibling_chase_cnt[tid]++;
#ifdef USE_SPLIT_ONLY_X_LOCK
        finish_leaf_access(true);
#else
        release_lock(lock_addr, lock_buffer, ctx, true, share_lock);
#endif
        return leaf_page_store(header->sibling_ptr, k, v, root, level, ctx,
                              false, share_lock);
      }
    }
  }

  // 2. try update
  // 2.1 check main bucket
  int retry_cnt = 0;
  int update_retry_cnt = 0;
retry_insert:
  update_addr = nullptr;
  insert_addr = nullptr;
  group->find(k, !(bucket_id % 2), &update_addr, &insert_addr);
  if (update_addr) {
    LeafValue cas_val(update_addr->lv.cl_ver, v);
    bool update_cas_ok = sxlock_leaf_value_cas(
        GADD(page_addr, ((char *)&update_addr->lv - page_buffer)),
        update_addr->lv.raw, cas_val.raw);
    if (update_cas_ok) {
      leaf_update_hit_cnt[tid]++;
      record_hot_leaf_update_success(page_addr, tid);
#ifdef USE_SPLIT_ONLY_X_LOCK
      finish_leaf_access(true);
#else
      release_lock(lock_addr, lock_buffer, ctx, true, share_lock);
#endif
      return true;
    }
    sxlock_update_cas_fail_cnt[tid]++;
    if (++update_retry_cnt == 1) {
      sxlock_update_cas_retry_event_cnt[tid]++;
    }
    sxlock_update_cas_retry_step_cnt[tid]++;
    update_addr->lv.raw = cas_ret_buffer[0];
    if (update_retry_cnt > kSxLockCasRetryLimit) {
      // printf("retry sxlock update CAS %d times\n", update_retry_cnt);
      update_retry_cnt = 0;
      // assert(false);
    }
    goto retry_insert;
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
    bool insert_cas_ok = dsm_client_->CasMaskSync(
        GADD(page_addr, ((char *)insert_addr - page_buffer)), 4,
        (uint64_t)insert_addr, (uint64_t)swap_buffer, cas_ret_buffer,
        (uint64_t)mask_buffer, ctx);
    // cas succeed or same key inserted by other thread
    if (insert_cas_ok || (__bswap_64(cas_ret_buffer[0]) == k)) {
      leaf_insert_empty_cnt[tid]++;
      insert_addr->key = k;
      insert_addr->lv.cl_ver = swap_entry->lv.cl_ver;
      insert_addr->lv.val = v;
#ifdef USE_SPLIT_ONLY_X_LOCK
      finish_leaf_access(true);
#else
      release_lock(lock_addr, lock_buffer, ctx, true, share_lock);
#endif
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
    sxlock_insert_cas_fail_cnt[tid]++;
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
#if defined(USE_SPLIT_ONLY_X_LOCK)
  // No S lock exists, so there is no S->X upgrade.
  // Leave the split-read guard first; begin_leaf_split_guard() waits for
  // inflight readers to drain, so holding this guard here would self-block.
  if (share_lock) {
    leave_leaf_write_guard();
  }
  upgrade_from_s = false;
  share_lock = false;
#elif defined(USE_SX_LOCK)
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
#if defined(USE_SPLIT_ONLY_X_LOCK)
    enter_leaf_write_guard();
#endif
    lock_and_read(lock_addr, share_lock, upgrade_from_s, lock_buffer, page_addr,
                  kLeafPageSize, page_buffer, ctx);
    hold_x_lock = !share_lock;
  }
#ifdef USE_HOT_READ_CACHE
  hot_read_cache_invalidate_leaf(page_addr, tid);
#endif
  LeafPage *page = (LeafPage *)page_buffer;

  assert(header->level == level);

  if (k < header->lowest || k >= header->highest ||
      page_addr.node_version != header->version) {  // cache is stale
    // Note: when very slow, may need recurse very large times and cause stack
    // overflow
#ifdef USE_SPLIT_ONLY_X_LOCK
      finish_leaf_access(true);
#else
      release_lock(lock_addr, lock_buffer, ctx, true, share_lock);
#endif
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
  int update_retry_cnt_2 = 0;
retry_insert_2:
  update_addr = nullptr;
  insert_addr = nullptr;
  group->find(k, !(bucket_id % 2), &update_addr, &insert_addr);

  if (update_addr) {
    LeafValue cas_val(update_addr->lv.cl_ver, v);
#ifdef USE_CRC
    leaf_update_hit_cnt[tid]++;
    record_hot_leaf_update_success(page_addr, tid);
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
    {
      bool update_cas_ok = sxlock_leaf_value_cas(
          GADD(page_addr, ((char *)&(update_addr->lv) - page_buffer)),
          update_addr->lv.raw, cas_val.raw);
      if (update_cas_ok) {
        leaf_update_hit_cnt[tid]++;
        record_hot_leaf_update_success(page_addr, tid);
#ifdef USE_SPLIT_ONLY_X_LOCK
        finish_leaf_access(true);
#else
        release_lock(lock_addr, lock_buffer, ctx, true, share_lock);
#endif
        return true;
      }
      sxlock_update_cas_fail_cnt[tid]++;
      if (++update_retry_cnt_2 == 1) {
        sxlock_update_cas_retry_event_cnt[tid]++;
      }
      sxlock_update_cas_retry_step_cnt[tid]++;
      update_addr->lv.raw = cas_ret_buffer[0];
      if (update_retry_cnt_2 > kSxLockCasRetryLimit) {
        printf("retry sxlock update CAS %d times after page lock\n", update_retry_cnt_2);
        assert(false);
      }
      goto retry_insert_2;
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
    bool insert_cas_ok = dsm_client_->CasMaskSync(
        GADD(page_addr, ((char *)insert_addr - page_buffer)), 4,
        (uint64_t)insert_addr, (uint64_t)swap_buffer, cas_ret_buffer,
        (uint64_t)mask_buffer, ctx);
    // cas succeed or same key inserted by other thread
    if (insert_cas_ok || (__bswap_64(cas_ret_buffer[0]) == k)) {
      leaf_insert_empty_cnt[tid]++;
      insert_addr->key = k;
      insert_addr->lv.cl_ver = swap_entry->lv.cl_ver;
      insert_addr->lv.val = v;
#ifdef USE_SPLIT_ONLY_X_LOCK
      finish_leaf_access(true);
#else
      release_lock(lock_addr, lock_buffer, ctx, true, share_lock);
#endif
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
    sxlock_insert_cas_fail_cnt[tid]++;
    leaf_insert_retry_event_cnt[tid]++;
    leaf_insert_retry_step_cnt[tid]++;
    assert(share_lock);
    goto retry_insert_2;
  }

  // should hold x lock
#if defined(USE_SPLIT_ONLY_X_LOCK)
  if (share_lock) {
    leave_leaf_write_guard();
    upgrade_from_s = false;
    share_lock = false;
    goto retry_with_xlock;
  }
#elif defined(USE_SX_LOCK)
  if (share_lock) {
    leaf_upgrade_to_x_cnt[tid]++;
    upgrade_from_s = true;
    share_lock = false;
    goto retry_with_xlock;
  }
#endif

  // split
  leaf_split_cnt[tid]++;
  record_hot_leaf_split_pressure(page_addr, tid);
  mark_hot_leaf_fast_path_conflict(page_addr);
#ifdef USE_HOT_READ_CACHE
  hot_read_cache_invalidate_leaf(page_addr, tid);
#endif
#ifdef USE_LEAF_SPLIT_GUARD
  release_sx_lock(lock_addr, lock_buffer, ctx, false, false);
  hold_x_lock = false;
  upgrade_from_s = false;

  bool split_guard_wait_active = false;
  uint32_t split_guard_generation =
      begin_leaf_split_guard(dsm_client_, page_addr, ctx);
  split_guard_wait_active = true;

  lock_and_read(lock_addr, false, false, lock_buffer, page_addr,
                kLeafPageSize, page_buffer, ctx);
  hold_x_lock = true;
  page = (LeafPage *)page_buffer;

  if (k < header->lowest || k >= header->highest ||
      page_addr.node_version != header->version) {
    if (split_guard_wait_active) {
      end_leaf_split_guard(dsm_client_, page_addr, split_guard_generation);
      split_guard_wait_active = false;
    }
    release_sx_lock(lock_addr, lock_buffer, ctx, true, false);
    return false;
  }

#endif
  GlobalAddress sibling_addr;
  sibling_addr = dsm_client_->Alloc(kLeafPageSize);
  char *sibling_buf = rbuf.get_sibling_buffer();
#ifdef USE_LEAF_SPLIT_GUARD
  memset(sibling_buf, 0, kLeafPageSize);
#endif
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
#ifdef USE_LEAF_SPLIT_GUARD
  write_and_unlock_leaf_x(page_buffer, page_addr, kLeafPageSize, false);
  if (split_guard_wait_active) {
    end_leaf_split_guard(dsm_client_, page_addr, split_guard_generation);
    split_guard_wait_active = false;
  }
#else
  write_and_unlock(page_buffer, page_addr, kLeafPageSize, lock_buffer, lock_addr, ctx, false, false);
#endif
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
  if (update_addr) {
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
  ctx.busy_waiting_queue = &busy_waiting_queue;

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
  auto poll_busy_waiting_queue = [&]() -> bool {
    // drain_leaf_split_guard_messages(dsm_client_);
    if (busy_waiting_queue.empty()) {
      return false;
    }
    auto next = busy_waiting_queue.front();
    busy_waiting_queue.pop();
    uint16_t next_coro_id = next.first;
    if (next.second()) {
      yield(worker[next_coro_id]);
      return true;
    }
    busy_waiting_queue.push(next);
    return false;
  };
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
    if (poll_busy_waiting_queue()) {
      continue;
    }
    bool resumed_busy_waiter = false;
    while (dsm_client_->get_pending_event_count() == 0) {
      if (poll_busy_waiting_queue()) {
        resumed_busy_waiter = true;
        break;
      }
    }
    if (resumed_busy_waiter) {
      continue;
    }
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
        poll_busy_waiting_queue();
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
              poll_busy_waiting_queue();
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
    if (poll_busy_waiting_queue()) {
      continue;
    }
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
        poll_busy_waiting_queue();
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
    if (poll_busy_waiting_queue()) {
      continue;
    }
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
      poll_busy_waiting_queue();
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
    optimistic_update_attempt_cnt[i] = 0;
    optimistic_update_success_cnt[i] = 0;
    optimistic_update_cas_fail_cnt[i] = 0;
    optimistic_update_cas_retry_event_cnt[i] = 0;
    optimistic_update_cas_retry_step_cnt[i] = 0;
    optimistic_update_cas_retry_exhaust_cnt[i] = 0;
    optimistic_update_split_abort_cnt[i] = 0;
    optimistic_leaf_fast_path_hot_bypass_cnt[i] = 0;
    optimistic_insert_attempt_cnt[i] = 0;
    optimistic_insert_success_cnt[i] = 0;
    optimistic_insert_cas_fail_cnt[i] = 0;
    optimistic_insert_split_abort_cnt[i] = 0;
    optimistic_insert_consistency_fail_cnt[i] = 0;
    optimistic_insert_fallback_cnt[i] = 0;
    leaf_group_reread_too_many_cnt[i] = 0;
    leaf_group_reread_fallback_cnt[i] = 0;
#ifdef USE_HOT_READ_CACHE
    hot_read_cache_hit_cnt[i] = 0;
    hot_read_cache_miss_cnt[i] = 0;
    hot_read_cache_fill_cnt[i] = 0;
    hot_read_cache_update_cnt[i] = 0;
    hot_read_cache_invalidate_cnt[i] = 0;
    hot_read_cache_leaf_lazy_invalidate_cnt[i] = 0;
#endif
    hot_leaf_update_hot_transition_cnt[i] = 0;
    hot_leaf_conflict_hot_transition_cnt[i] = 0;
    hot_leaf_insert_hot_transition_cnt[i] = 0;
    hot_leaf_fold_pending_transition_cnt[i] = 0;
    hot_leaf_controller_bypass_cnt[i] = 0;
    hot_leaf_cache_gate_deny_cnt[i] = 0;
    hot_leaf_optimistic_probe_cnt[i] = 0;
    insert_rtt_cnt[i][0] = 0;
    insert_byte_cnt[i][0] = 0;
    insert_op_cnt[i][0] = 0;
    search_rtt_cnt[i][0] = 0;
    search_byte_cnt[i][0] = 0;
    search_op_cnt[i][0] = 0;

    local_group_wc_owner_cnt[i] = 0;
    local_group_wc_waiter_cnt[i] = 0;
    local_group_wc_waiter_return_cnt[i] = 0;
    local_group_wc_apply_cnt[i] = 0;
    local_group_wc_nochange_cnt[i] = 0;
    local_group_wc_replace_cnt[i] = 0;
    local_group_wc_bypass_cnt[i] = 0;
    local_group_rc_owner_cnt[i] = 0;
    local_group_rc_waiter_cnt[i] = 0;
    local_group_rc_waiter_return_cnt[i] = 0;
    local_group_rc_bypass_cnt[i] = 0;
    local_group_rc_bytes_saved[i] = 0;
    local_group_rc_wait_us_sum[i] = 0;
    local_group_rc_wait_us_max[i] = 0;
    split_guard_begin_cnt[i] = 0;
    split_guard_wait_event_cnt[i] = 0;
    split_guard_wait_yield_cnt[i] = 0;
    split_guard_wait_us_sum[i] = 0;
    split_guard_wait_us_max[i] = 0;
    split_guard_wait_inflight_us_sum[i] = 0;
    split_guard_wait_ack_us_sum[i] = 0;
    split_guard_wait_both_us_sum[i] = 0;
    split_guard_inflight_max[i] = 0;
    for (int k = 0; k <= 5000; ++k) {
      tries_per_lock[i][k] = 0;
    }
  }

  split_guard_ack_immediate_cnt.store(0, std::memory_order_relaxed);
  split_guard_ack_queued_cnt.store(0, std::memory_order_relaxed);
  split_guard_ack_sent_after_wait_cnt.store(0, std::memory_order_relaxed);
  split_guard_ack_flush_blocked_cnt.store(0, std::memory_order_relaxed);
  split_guard_ack_queue_wait_us_sum.store(0, std::memory_order_relaxed);
  split_guard_ack_queue_wait_us_max.store(0, std::memory_order_relaxed);
  split_guard_ack_pending_max.store(0, std::memory_order_relaxed);
  split_guard_ack_blocking_inflight_max.store(0, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(pending_leaf_split_guard_acks_mutex);
    pending_leaf_split_guard_acks.clear();
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
