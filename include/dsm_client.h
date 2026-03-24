#pragma once

#include <atomic>

// #include "Config.h"
#include "Cache.h"
#include "connection.h"
#include "dsm_keeper.h"
#include "GlobalAddress.h"
#include "LocalAllocator.h"
#include "RdmaBuffer.h"
#include "RawMessageConnection.h"
#include "ThreadConnection.h"
#include "Common.h"
class Directory;

class DSMClient {
 public:
  static DSMClient *GetInstance(const DSMConfig &conf) {
    static DSMClient dsm(conf);
    return &dsm;
  }

  // clear the network resources for all threads
  void ResetThread() { app_id_.store(0); }
#ifdef USE_DOORBELL_BATCHING
  void FlushDoorbell();
#endif
  // obtain netowrk resources for a thread
  void RegisterThread();
  bool IsRegistered() { return thread_id_ != -1; }

  uint16_t get_my_client_id() { return my_client_id_; }
  uint16_t get_my_thread_id() { return thread_id_; }
  uint16_t get_server_size() { return conf_.num_server; }
  uint16_t get_client_size() { return conf_.num_client; }
  uint64_t get_thread_tag() { return thread_tag_; }
  inline void decrease_pending_event() { --pending_event_count_; }
  inline void increase_pending_event() { ++pending_event_count_; }
  inline uint64_t get_pending_event_count() { return pending_event_count_; }
  ibv_comp_channel* get_comp_channel() {return  i_con_->comp_channel;}


  void Barrier(const std::string &ss) {
    keeper_->Barrier(ss, conf_.num_client, my_client_id_ == 0);
  }

  char *get_rdma_buffer() { return rdma_buffer_; }
  RdmaBuffer &get_rbuf(int coro_id) { return rbuf_[coro_id]; }

  // RDMA operations
  // buffer is registered memory
  void Read(char *buffer, GlobalAddress gaddr, size_t size, bool signal,
            CoroContext *ctx = nullptr);
  void ReadSync(char *buffer, GlobalAddress gaddr, size_t size,
                CoroContext *ctx = nullptr);

  void Write(const char *buffer, GlobalAddress gaddr, size_t size,
             bool signal = true, CoroContext *ctx = nullptr);
  void WriteSync(const char *buffer, GlobalAddress gaddr, size_t size,
                 CoroContext *ctx = nullptr);

  void ReadBatch(RdmaOpRegion *rs, int k, bool signal = true,
                 CoroContext *ctx = nullptr);
  void ReadBatchSync(RdmaOpRegion *rs, int k, CoroContext *ctx = nullptr);

  void WriteBatch(RdmaOpRegion *rs, int k, bool signal = true,
                  CoroContext *ctx = nullptr);
  void WriteBatchSync(RdmaOpRegion *rs, int k, CoroContext *ctx = nullptr);

  void WriteFaa(RdmaOpRegion &write_ror, RdmaOpRegion &faa_ror,
                uint64_t add_val, bool signal = true,
                CoroContext *ctx = nullptr);
  void WriteFaaSync(RdmaOpRegion &write_ror, RdmaOpRegion &faa_ror,
                    uint64_t add_val, CoroContext *ctx = nullptr);

  void WriteCas(RdmaOpRegion &write_ror, RdmaOpRegion &cas_ror, uint64_t equal,
                uint64_t val, bool signal = true, CoroContext *ctx = nullptr);
  void WriteCasSync(RdmaOpRegion &write_ror, RdmaOpRegion &cas_ror,
                    uint64_t equal, uint64_t val, CoroContext *ctx = nullptr);

  void Cas(GlobalAddress gaddr, uint64_t equal, uint64_t val,
           uint64_t *rdma_buffer, bool signal = true,
           CoroContext *ctx = nullptr);
  bool CasSync(GlobalAddress gaddr, uint64_t equal, uint64_t val,
               uint64_t *rdma_buffer, CoroContext *ctx = nullptr);

  void CasRead(RdmaOpRegion &cas_ror, RdmaOpRegion &read_ror, uint64_t equal,
               uint64_t val, bool signal = true, CoroContext *ctx = nullptr);
  bool CasReadSync(RdmaOpRegion &cas_ror, RdmaOpRegion &read_ror,
                   uint64_t equal, uint64_t val, CoroContext *ctx = nullptr);

  void FaaRead(RdmaOpRegion &faab_ror, RdmaOpRegion &read_ror, uint64_t add,
               bool signal = true, CoroContext *ctx = nullptr);
  void FaaReadSync(RdmaOpRegion &faab_ror, RdmaOpRegion &read_ror, uint64_t add,
                   CoroContext *ctx = nullptr);

  void FaaBoundRead(RdmaOpRegion &faab_ror, RdmaOpRegion &read_ror,
                    uint64_t add, uint64_t boundary, bool signal = true,
                    CoroContext *ctx = nullptr);
  void FaaBoundReadSync(RdmaOpRegion &faab_ror, RdmaOpRegion &read_ror,
                        uint64_t add, uint64_t boundary,
                        CoroContext *ctx = nullptr);

  void CasMask(GlobalAddress gaddr, int log_sz, uint64_t equal, uint64_t val,
               uint64_t *rdma_buffer, uint64_t mask = ~(0ull),
               bool signal = true, CoroContext *ctx = nullptr);
  bool CasMaskSync(GlobalAddress gaddr, int log_sz, uint64_t equal,
                   uint64_t val, uint64_t *rdma_buffer, uint64_t mask = ~(0ull),
                   CoroContext *ctx = nullptr);

  void CasMaskWrite(RdmaOpRegion &cas_ror, uint64_t equal, uint64_t swap,
                    uint64_t mask, RdmaOpRegion &write_ror, bool signal = true,
                    CoroContext *ctx = nullptr);
  bool CasMaskWriteSync(RdmaOpRegion &cas_ror, uint64_t equal, uint64_t swap,
                        uint64_t mask, RdmaOpRegion &write_ror,
                        CoroContext *ctx = nullptr);

  void FaaBound(GlobalAddress gaddr, int log_sz, uint64_t add_val,
                uint64_t *rdma_buffer, uint64_t mask, bool signal = true,
                CoroContext *ctx = nullptr);
  void FaaBoundSync(GlobalAddress gaddr, int log_sz, uint64_t add_val,
                    uint64_t *rdma_buffer, uint64_t mask,
                    CoroContext *ctx = nullptr);

  // for on-chip device memory
  void ReadDm(char *buffer, GlobalAddress gaddr, size_t size,
              bool signal = true, CoroContext *ctx = nullptr);
  void ReadDmSync(char *buffer, GlobalAddress gaddr, size_t size,
                  CoroContext *ctx = nullptr);

  void WriteDm(const char *buffer, GlobalAddress gaddr, size_t size,
               bool signal = true, CoroContext *ctx = nullptr);
  void WriteDmSync(const char *buffer, GlobalAddress gaddr, size_t size,
                   CoroContext *ctx = nullptr);

  void CasDm(GlobalAddress gaddr, uint64_t equal, uint64_t val,
             uint64_t *rdma_buffer, bool signal = true,
             CoroContext *ctx = nullptr);
  bool CasDmSync(GlobalAddress gaddr, uint64_t equal, uint64_t val,
                 uint64_t *rdma_buffer, CoroContext *ctx = nullptr);

  void CasDmMask(GlobalAddress gaddr, int log_sz, uint64_t equal, uint64_t val,
                 uint64_t *rdma_buffer, uint64_t mask = ~(0ull),
                 bool signal = true, CoroContext *ctx = nullptr);
  bool CasDmMaskSync(GlobalAddress gaddr, int log_sz, uint64_t equal,
                     uint64_t val, uint64_t *rdma_buffer,
                     uint64_t mask = ~(0ull), CoroContext *ctx = nullptr);

  void FaaDmBound(GlobalAddress gaddr, int log_sz, uint64_t add_val,
                  uint64_t *rdma_buffer, uint64_t mask, bool signal = true,
                  CoroContext *ctx = nullptr);
  void FaaDmBoundSync(GlobalAddress gaddr, int log_sz, uint64_t add_val,
                      uint64_t *rdma_buffer, uint64_t mask,
                      CoroContext *ctx = nullptr);

  uint64_t PollRdmaCq(int count = 1);
  bool PollRdmaCqOnce(uint64_t &wr_id);
  int PollRdmaCqBatch(int max_entries, uint64_t *wr_ids);

  uint64_t Sum(uint64_t value) {
    static uint64_t count = 0;
    return keeper_->Sum(std::string("sum-") + std::to_string(count++), value,
                        my_client_id_, conf_.num_client);
  }

  GlobalAddress Alloc(size_t size) {
    thread_local int next_target_node =
        (get_my_thread_id() + get_my_client_id()) % conf_.num_server;
    thread_local int next_target_dir_id =
        (get_my_thread_id() + get_my_client_id()) % NR_DIRECTORY;

    bool need_chunk = false;
    auto addr = local_allocator_.malloc(size, need_chunk);
    if (need_chunk) {
      RawMessage m;
      m.type = RpcType::MALLOC;
      this->RpcCallDir(m, next_target_node, next_target_dir_id);
      local_allocator_.set_chunck(RpcWait()->addr);

      if (++next_target_dir_id == NR_DIRECTORY) {
        next_target_node = (next_target_node + 1) % conf_.num_server;
        next_target_dir_id = 0;
      }

      // retry
      addr = local_allocator_.malloc(size, need_chunk);
    }

    return addr;
  }

  void Free(GlobalAddress addr) { local_allocator_.free(addr); }

  void RpcCallDir(const RawMessage &m, uint16_t node_id, uint16_t dir_id = 0) {
    auto buffer = (RawMessage *)i_con_->message->getSendPool();
    memcpy(reinterpret_cast<void *>(buffer), &m, sizeof(RawMessage));
    buffer->node_id = my_client_id_;
    buffer->app_id = thread_id_;
    i_con_->sendMessage2Dir(buffer, node_id, dir_id);
  }
  RawMessage *RpcWait() {
    ibv_wc wc;
    pollWithCQ(i_con_->rpc_cq, 1, &wc);
    return (RawMessage *)i_con_->message->getMessage();
  }

 private:
  DSMConfig conf_;
  std::atomic_int app_id_;
  Cache cache_;
  uint32_t my_client_id_;
#ifdef USE_DOORBELL_BATCHING
  // 【新增】为每个协程预分配持久化的 WR 和 SGE 结构
  // 保证协程 yield 期间，网卡 DMA 读取这些结构时内存依然合法
  // 提示：为了安全起见，建议这里的数组大小使用 define::kMaxCoro，与底下的 rbuf_ 保持对齐
  static thread_local ibv_send_wr coro_wr_[define::kCoroCnt];
  static thread_local ibv_sge coro_sge_[define::kCoroCnt];
  
  // 【新增】按目标 NodeID 划分的挂起队列（头尾指针），支持 $O(1)$ 尾部插入
  static thread_local ibv_send_wr** pending_wr_head_; 
  static thread_local ibv_send_wr** pending_wr_tail_;

  // 内部辅助函数：将协程的 WR 挂载到对应目标节点的延迟队列中
  void queue_wr(int node_id, ibv_send_wr* wr);

  // --- 【新增】实验性 Verbs 队列 (用于 CasMask, FaaBound 等) ---
  static thread_local ibv_exp_send_wr coro_exp_wr_[define::kCoroCnt];
  static thread_local ibv_sge coro_exp_sge_[define::kCoroCnt];
  static thread_local ibv_exp_send_wr** pending_exp_wr_head_;
  static thread_local ibv_exp_send_wr** pending_exp_wr_tail_;

  inline void queue_exp_wr(int node_id, ibv_exp_send_wr* wr) {
    wr->next = nullptr;
    if (pending_exp_wr_head_[node_id] == nullptr) {
      pending_exp_wr_head_[node_id] = wr;
      pending_exp_wr_tail_[node_id] = wr;
    } else {
      pending_exp_wr_tail_[node_id]->next = wr;
      pending_exp_wr_tail_[node_id] = wr;
    }
  }
#endif
  static thread_local int thread_id_;
  static thread_local ThreadConnection *i_con_;
  static thread_local char *rdma_buffer_;
  static thread_local LocalAllocator local_allocator_;
  static thread_local RdmaBuffer rbuf_[define::kMaxCoro];
  static thread_local uint64_t thread_tag_;
  static thread_local uint64_t pending_event_count_;
  RemoteConnectionToServer *conn_to_server_;

  ThreadConnection *th_con_[MAX_APP_THREAD];
  DSMClientKeeper *keeper_;
  Directory *dir_agent_[NR_DIRECTORY];

  DSMClient(const DSMConfig &conf);
  
  void InitRdmaConnection();
  void FillKeysDest(RdmaOpRegion &ror, GlobalAddress addr, bool is_chip);

};


