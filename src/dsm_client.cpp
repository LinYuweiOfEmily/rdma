#include "dsm_client.h"

thread_local int DSMClient::thread_id_ = -1;
thread_local ThreadConnection *DSMClient::i_con_ = nullptr;
thread_local char *DSMClient::rdma_buffer_ = nullptr;
thread_local LocalAllocator DSMClient::local_allocator_;
thread_local RdmaBuffer DSMClient::rbuf_[define::kMaxCoro];
thread_local uint64_t DSMClient::thread_tag_ = 0;
thread_local uint64_t DSMClient::pending_event_count_ = 0;
#ifdef USE_DOORBELL_BATCHING
thread_local ibv_send_wr DSMClient::coro_wr_[define::kCoroCnt];
thread_local ibv_sge DSMClient::coro_sge_[define::kCoroCnt];
thread_local ibv_send_wr** DSMClient::pending_wr_head_ = nullptr;
thread_local ibv_send_wr** DSMClient::pending_wr_tail_ = nullptr;

thread_local ibv_exp_send_wr DSMClient::coro_exp_wr_[define::kCoroCnt];
thread_local ibv_sge DSMClient::coro_exp_sge_[define::kCoroCnt];
thread_local ibv_exp_send_wr** DSMClient::pending_exp_wr_head_ = nullptr;
thread_local ibv_exp_send_wr** DSMClient::pending_exp_wr_tail_ = nullptr;
#endif
DSMClient::DSMClient(const DSMConfig &conf)
    : conf_(conf), app_id_(0), cache_(conf.cache_size) {
  Debug::notifyInfo("cache size: %dGB", conf_.cache_size);
  InitRdmaConnection();
  keeper_->Barrier("DSMClient-init", conf_.num_client, my_client_id_ == 0);
}

void DSMClient::InitRdmaConnection() {
  conn_to_server_ = new RemoteConnectionToServer[conf_.num_server];

  for (int i = 0; i < MAX_APP_THREAD; ++i) {
    // client thread to servers
    th_con_[i] =
        new ThreadConnection(i, (void *)cache_.data, cache_.size * define::GB,
                             conf_.num_server, conf_.rnic_id, conn_to_server_);
  }

  keeper_ = new DSMClientKeeper(th_con_, conn_to_server_, conf_.num_server);
  my_client_id_ = keeper_->get_my_client_id();
}
#ifdef USE_DOORBELL_BATCHING
inline void DSMClient::queue_wr(int node_id, ibv_send_wr* wr) {
  wr->next = nullptr;
  if (pending_wr_head_[node_id] == nullptr) {
    pending_wr_head_[node_id] = wr;
    pending_wr_tail_[node_id] = wr;
  } else {
    pending_wr_tail_[node_id]->next = wr;
    pending_wr_tail_[node_id] = wr;
  }
}

// 集中敲门铃函数：遍历所有目标节点，一次性把长链表 post 出去
void DSMClient::FlushDoorbell() {
  for (int i = 0; i < conf_.num_server; ++i) {
    // 1. 发送标准 WR 链表
    if (pending_wr_head_[i] != nullptr) {
      struct ibv_send_wr *bad_wr;
      if (ibv_post_send(i_con_->data[0][i], pending_wr_head_[i], &bad_wr)) {
        Debug::notifyError("Batch doorbell failed for node %d (Standard WR)", i);
      }
      pending_wr_head_[i] = nullptr;
      pending_wr_tail_[i] = nullptr;
    }
    
    // 2. 【新增】发送实验性 WR 链表 (注意使用的是 ibv_exp_post_send)
    if (pending_exp_wr_head_[i] != nullptr) {
      struct ibv_exp_send_wr *bad_exp_wr;
      if (ibv_exp_post_send(i_con_->data[0][i], pending_exp_wr_head_[i], &bad_exp_wr)) {
        Debug::notifyError("Batch doorbell failed for node %d (Experimental WR)", i);
      }
      pending_exp_wr_head_[i] = nullptr;
      pending_exp_wr_tail_[i] = nullptr;
    }
  }
}
#endif

void DSMClient::RegisterThread() {
  static bool has_init[MAX_APP_THREAD];

  if (thread_id_ != -1) return;

  thread_id_ = app_id_.fetch_add(1);
  thread_tag_ = thread_id_ + (((uint64_t)get_my_client_id()) << 32) + 1;

  i_con_ = th_con_[thread_id_];

  if (!has_init[thread_id_]) {
    i_con_->message->initRecv();
    i_con_->message->initSend();

    has_init[thread_id_] = true;
  }

  rdma_buffer_ = (char *)cache_.data + thread_id_ * define::kPerThreadRdmaBuf;

  for (int i = 0; i < define::kMaxCoro; ++i) {
    rbuf_[i].set_buffer(rdma_buffer_ + i * define::kPerCoroRdmaBuf);
  }
#ifdef USE_DOORBELL_BATCHING
  if (pending_wr_head_ == nullptr) {
    pending_wr_head_ = new ibv_send_wr*[conf_.num_server](); 
    pending_wr_tail_ = new ibv_send_wr*[conf_.num_server]();

    // 【新增】初始化实验性 WR 队列的头尾指针数组
    pending_exp_wr_head_ = new ibv_exp_send_wr*[conf_.num_server]();
    pending_exp_wr_tail_ = new ibv_exp_send_wr*[conf_.num_server]();
  }
#endif
}

void DSMClient::Read(char *buffer, GlobalAddress gaddr, size_t size,
                     bool signal, CoroContext *ctx) {
  increase_pending_event();
  if (ctx == nullptr) {
    // 非协程环境（同步调用），保持原样，直接发出
    rdmaRead(i_con_->data[0][gaddr.nodeID], (uint64_t)buffer,
             conn_to_server_[gaddr.nodeID].dsm_base + gaddr.offset, size,
             i_con_->cacheLKey, conn_to_server_[gaddr.nodeID].dsm_rkey[0],
             signal);
  } else {
#ifdef USE_DOORBELL_BATCHING
    // 协程环境：进入 Doorbell Batching 延迟发送路径
    int coro_id = ctx->coro_id;
    ibv_send_wr& wr = coro_wr_[coro_id];
    ibv_sge& sg = coro_sge_[coro_id];

    // 1. 填充 SG List
    sg.addr = (uintptr_t)buffer;
    sg.length = size;
    sg.lkey = i_con_->cacheLKey;

    // 2. 填充 Work Request
    memset(&wr, 0, sizeof(wr));
    wr.sg_list = &sg;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    
    // 权衡点：目前每个协程依然需要自己的 CQE 来唤醒，所以保留 SIGNALED
    if (signal) wr.send_flags = IBV_SEND_SIGNALED; 
    
    wr.wr.rdma.remote_addr = conn_to_server_[gaddr.nodeID].dsm_base + gaddr.offset;
    wr.wr.rdma.rkey = conn_to_server_[gaddr.nodeID].dsm_rkey[0];
    wr.wr_id = coro_id; // 让 CQE 携带正确的 coro_id 用于精确唤醒

    // 3. 入队并让出 CPU
    queue_wr(gaddr.nodeID, &wr);
#else
    rdmaRead(i_con_->data[0][gaddr.nodeID], (uint64_t)buffer,
             conn_to_server_[gaddr.nodeID].dsm_base + gaddr.offset, size,
             i_con_->cacheLKey, conn_to_server_[gaddr.nodeID].dsm_rkey[0], true,
             ctx->coro_id);
#endif
    (*ctx->yield)(*ctx->master);
  }
}

void DSMClient::ReadSync(char *buffer, GlobalAddress gaddr, size_t size,
                         CoroContext *ctx) {
  Read(buffer, gaddr, size, true, ctx);

  if (ctx == nullptr) {
    ibv_wc wc;
    pollWithCQ(i_con_->cq, 1, &wc);
  }
}

void DSMClient::Write(const char *buffer, GlobalAddress gaddr, size_t size,
                      bool signal, CoroContext *ctx) {
  increase_pending_event();
  if (ctx == nullptr) {
    // 同步/非协程模式，保持原样，立即发出
    rdmaWrite(i_con_->data[0][gaddr.nodeID], (uint64_t)buffer,
              conn_to_server_[gaddr.nodeID].dsm_base + gaddr.offset, size,
              i_con_->cacheLKey, conn_to_server_[gaddr.nodeID].dsm_rkey[0], -1,
              signal);
  } else {
#ifdef USE_DOORBELL_BATCHING
    // 协程模式：进入 Doorbell Batching 延迟发送路径
    int coro_id = ctx->coro_id;
    ibv_send_wr& wr = coro_wr_[coro_id];
    ibv_sge& sg = coro_sge_[coro_id];

    // 1. 填充 SG List
    sg.addr = (uintptr_t)buffer;
    sg.length = size;
    sg.lkey = i_con_->cacheLKey;

    // 2. 填充 Work Request
    memset(&wr, 0, sizeof(wr));
    wr.sg_list = &sg;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    
    wr.send_flags = 0;
    // 协程模式为了精确唤醒，默认需要 CQE (原代码强传了 true)
    if (signal) {
      wr.send_flags |= IBV_SEND_SIGNALED;
    }
    
    // 【关键】：保留小包 Inline 优化，省去一次网卡 DMA 拉取数据的 PCIe 读事务！
    // MAX_INLINE_DATA 通常在你的 Rdma.h 里定义，取决于网卡配置（通常是 64 或 256 字节）
    if (size < MAX_INLINE_DATA) {
      wr.send_flags |= IBV_SEND_INLINE;
    }

    wr.wr.rdma.remote_addr = conn_to_server_[gaddr.nodeID].dsm_base + gaddr.offset;
    wr.wr.rdma.rkey = conn_to_server_[gaddr.nodeID].dsm_rkey[0];
    wr.wr_id = coro_id; // 携带 coro_id 用于 pollOnce 唤醒

    // 3. 挂入目标节点的门铃等待队列，并让出 CPU 控制权
    queue_wr(gaddr.nodeID, &wr);
#else
    rdmaWrite(i_con_->data[0][gaddr.nodeID], (uint64_t)buffer,
                  conn_to_server_[gaddr.nodeID].dsm_base + gaddr.offset, size,
                  i_con_->cacheLKey, conn_to_server_[gaddr.nodeID].dsm_rkey[0], -1,
                  true, ctx->coro_id);
#endif
    (*ctx->yield)(*ctx->master);
  }
}

void DSMClient::WriteSync(const char *buffer, GlobalAddress gaddr, size_t size,
                          CoroContext *ctx) {
  Write(buffer, gaddr, size, true, ctx);

  if (ctx == nullptr) {
    ibv_wc wc;
    pollWithCQ(i_con_->cq, 1, &wc);
  }
}

void DSMClient::FillKeysDest(RdmaOpRegion &ror, GlobalAddress gaddr,
                             bool is_chip) {
  ror.lkey = i_con_->cacheLKey;
  if (is_chip) {
    ror.dest = conn_to_server_[gaddr.nodeID].lock_base + gaddr.offset;
    ror.remoteRKey = conn_to_server_[gaddr.nodeID].lock_rkey[0];
  } else {
    ror.dest = conn_to_server_[gaddr.nodeID].dsm_base + gaddr.offset;
    ror.remoteRKey = conn_to_server_[gaddr.nodeID].dsm_rkey[0];
  }
}

void DSMClient::ReadBatch(RdmaOpRegion *rs, int k, bool signal,
                          CoroContext *ctx) {
  int node_id = -1;
  for (int i = 0; i < k; ++i) {
    GlobalAddress gaddr;
    gaddr.raw = rs[i].dest;
    node_id = gaddr.nodeID;
    FillKeysDest(rs[i], gaddr, rs[i].is_on_chip);
  } 

  if (ctx == nullptr) {
    rdmaReadBatch(i_con_->data[0][node_id], rs, k, signal);
  } else {
    rdmaReadBatch(i_con_->data[0][node_id], rs, k, true, ctx->coro_id);
    (*ctx->yield)(*ctx->master);
  }
}

void DSMClient::ReadBatchSync(RdmaOpRegion *rs, int k, CoroContext *ctx) {
  ReadBatch(rs, k, true, ctx);

  if (ctx == nullptr) {
    ibv_wc wc;
    pollWithCQ(i_con_->cq, 1, &wc);
  }
}

void DSMClient::WriteBatch(RdmaOpRegion *rs, int k, bool signal,
                           CoroContext *ctx) {
  int node_id = -1;
  for (int i = 0; i < k; ++i) {
    GlobalAddress gaddr;
    gaddr.raw = rs[i].dest;
    node_id = gaddr.nodeID;
    FillKeysDest(rs[i], gaddr, rs[i].is_on_chip);
  }

  if (ctx == nullptr) {
    rdmaWriteBatch(i_con_->data[0][node_id], rs, k, signal);
  } else {
    rdmaWriteBatch(i_con_->data[0][node_id], rs, k, true, ctx->coro_id);
    (*ctx->yield)(*ctx->master);
  }
}

void DSMClient::WriteBatchSync(RdmaOpRegion *rs, int k, CoroContext *ctx) {
  WriteBatch(rs, k, true, ctx);

  if (ctx == nullptr) {
    ibv_wc wc;
    pollWithCQ(i_con_->cq, 1, &wc);
  }
}
// 没用到
void DSMClient::WriteFaa(RdmaOpRegion &write_ror, RdmaOpRegion &faa_ror,
                         uint64_t add_val, bool signal, CoroContext *ctx) {
  int node_id;
  {
    GlobalAddress gaddr;
    gaddr.raw = write_ror.dest;
    node_id = gaddr.nodeID;

    FillKeysDest(write_ror, gaddr, write_ror.is_on_chip);
  }
  {
    GlobalAddress gaddr;
    gaddr.raw = faa_ror.dest;

    FillKeysDest(faa_ror, gaddr, faa_ror.is_on_chip);
  }
  if (ctx == nullptr) {
    rdmaWriteFaa(i_con_->data[0][node_id], write_ror, faa_ror, add_val, signal);
  } else {
    rdmaWriteFaa(i_con_->data[0][node_id], write_ror, faa_ror, add_val, true,
                 ctx->coro_id);
    (*ctx->yield)(*ctx->master);
  }
}

void DSMClient::WriteFaaSync(RdmaOpRegion &write_ror, RdmaOpRegion &faa_ror,
                             uint64_t add_val, CoroContext *ctx) {
  WriteFaa(write_ror, faa_ror, add_val, true, ctx);
  if (ctx == nullptr) {
    ibv_wc wc;
    pollWithCQ(i_con_->cq, 1, &wc);
  }
}

// 没用到
void DSMClient::WriteCas(RdmaOpRegion &write_ror, RdmaOpRegion &cas_ror,
                         uint64_t equal, uint64_t val, bool signal,
                         CoroContext *ctx) {
  int node_id;
  {
    GlobalAddress gaddr;
    gaddr.raw = write_ror.dest;
    node_id = gaddr.nodeID;

    FillKeysDest(write_ror, gaddr, write_ror.is_on_chip);
  }
  {
    GlobalAddress gaddr;
    gaddr.raw = cas_ror.dest;

    FillKeysDest(cas_ror, gaddr, cas_ror.is_on_chip);
  }
  if (ctx == nullptr) {
    rdmaWriteCas(i_con_->data[0][node_id], write_ror, cas_ror, equal, val,
                 signal);
  } else {
    rdmaWriteCas(i_con_->data[0][node_id], write_ror, cas_ror, equal, val, true,
                 ctx->coro_id);
    (*ctx->yield)(*ctx->master);
  }
}

void DSMClient::WriteCasSync(RdmaOpRegion &write_ror, RdmaOpRegion &cas_ror,
                             uint64_t equal, uint64_t val, CoroContext *ctx) {
  WriteCas(write_ror, cas_ror, equal, val, true, ctx);
  if (ctx == nullptr) {
    ibv_wc wc;
    pollWithCQ(i_con_->cq, 1, &wc);
  }
}

// 没用到
void DSMClient::CasRead(RdmaOpRegion &cas_ror, RdmaOpRegion &read_ror,
                        uint64_t equal, uint64_t val, bool signal,
                        CoroContext *ctx) {
  int node_id;
  {
    GlobalAddress gaddr;
    gaddr.raw = cas_ror.dest;
    node_id = gaddr.nodeID;
    FillKeysDest(cas_ror, gaddr, cas_ror.is_on_chip);
  }
  {
    GlobalAddress gaddr;
    gaddr.raw = read_ror.dest;
    FillKeysDest(read_ror, gaddr, read_ror.is_on_chip);
  }

  if (ctx == nullptr) {
    rdmaCasRead(i_con_->data[0][node_id], cas_ror, read_ror, equal, val,
                signal);
  } else {
    rdmaCasRead(i_con_->data[0][node_id], cas_ror, read_ror, equal, val, true,
                ctx->coro_id);
    (*ctx->yield)(*ctx->master);
  }
}

bool DSMClient::CasReadSync(RdmaOpRegion &cas_ror, RdmaOpRegion &read_ror,
                            uint64_t equal, uint64_t val, CoroContext *ctx) {
  CasRead(cas_ror, read_ror, equal, val, true, ctx);

  if (ctx == nullptr) {
    ibv_wc wc;
    pollWithCQ(i_con_->cq, 1, &wc);
  }

  return equal == *(uint64_t *)cas_ror.source;
}

// 没用到
void DSMClient::FaaRead(RdmaOpRegion &faa_ror, RdmaOpRegion &read_ror,
                        uint64_t add, bool signal, CoroContext *ctx) {
  int node_id;
  {
    GlobalAddress gaddr;
    gaddr.raw = faa_ror.dest;
    node_id = gaddr.nodeID;
    FillKeysDest(faa_ror, gaddr, faa_ror.is_on_chip);
  }
  {
    GlobalAddress gaddr;
    gaddr.raw = read_ror.dest;
    FillKeysDest(read_ror, gaddr, read_ror.is_on_chip);
  }

  if (ctx == nullptr) {
    rdmaFaaRead(i_con_->data[0][node_id], faa_ror, read_ror, add, signal);
  } else {
    rdmaFaaRead(i_con_->data[0][node_id], faa_ror, read_ror, add, true,
                ctx->coro_id);
    (*ctx->yield)(*ctx->master);
  }
}

void DSMClient::FaaReadSync(RdmaOpRegion &faa_ror, RdmaOpRegion &read_ror,
                            uint64_t add, CoroContext *ctx) {
  FaaRead(faa_ror, read_ror, add, true, ctx);

  if (ctx == nullptr) {
    ibv_wc wc;
    pollWithCQ(i_con_->cq, 1, &wc);
  }
}

// 没用到
void DSMClient::FaaBoundRead(RdmaOpRegion &faab_ror, RdmaOpRegion &read_ror,
                             uint64_t add, uint64_t boundary, bool signal,
                             CoroContext *ctx) {
  int node_id;
  {
    GlobalAddress gaddr;
    gaddr.raw = faab_ror.dest;
    node_id = gaddr.nodeID;
    FillKeysDest(faab_ror, gaddr, faab_ror.is_on_chip);
  }
  {
    GlobalAddress gaddr;
    gaddr.raw = read_ror.dest;
    FillKeysDest(read_ror, gaddr, read_ror.is_on_chip);
  }

  if (ctx == nullptr) {
    rdmaFaaBoundRead(i_con_->data[0][node_id], faab_ror, read_ror, add,
                     boundary, signal);
  } else {
    rdmaFaaBoundRead(i_con_->data[0][node_id], faab_ror, read_ror, add,
                     boundary, true, ctx->coro_id);
    (*ctx->yield)(*ctx->master);
  }
}

void DSMClient::FaaBoundReadSync(RdmaOpRegion &faab_ror, RdmaOpRegion &read_ror,
                                 uint64_t add, uint64_t boundary,
                                 CoroContext *ctx) {
  FaaBoundRead(faab_ror, read_ror, add, boundary, true, ctx);

  if (ctx == nullptr) {
    ibv_wc wc;
    pollWithCQ(i_con_->cq, 1, &wc);
  }
}


// 可改
void DSMClient::Cas(GlobalAddress gaddr, uint64_t equal, uint64_t val,
                    uint64_t *rdma_buffer, bool signal, CoroContext *ctx) {
  if (ctx == nullptr) {
    rdmaCompareAndSwap(i_con_->data[0][gaddr.nodeID], (uint64_t)rdma_buffer,
                       conn_to_server_[gaddr.nodeID].dsm_base + gaddr.offset,
                       equal, val, i_con_->cacheLKey,
                       conn_to_server_[gaddr.nodeID].dsm_rkey[0], signal);
  } else {
#ifdef USE_DOORBELL_BATCHING
    int coro_id = ctx->coro_id;
    ibv_send_wr& wr = coro_wr_[coro_id];
    ibv_sge& sg = coro_sge_[coro_id];

    sg.addr = (uintptr_t)rdma_buffer;
    sg.length = 8; // CAS 操作固定 8 字节
    sg.lkey = i_con_->cacheLKey;

    memset(&wr, 0, sizeof(wr));
    wr.sg_list = &sg;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;

    if (signal) wr.send_flags = IBV_SEND_SIGNALED;

    wr.wr.atomic.remote_addr = conn_to_server_[gaddr.nodeID].dsm_base + gaddr.offset;
    wr.wr.atomic.rkey = conn_to_server_[gaddr.nodeID].dsm_rkey[0];
    wr.wr.atomic.compare_add = equal;
    wr.wr.atomic.swap = val;
    wr.wr_id = coro_id;

    queue_wr(gaddr.nodeID, &wr);
#else
    rdmaCompareAndSwap(i_con_->data[0][gaddr.nodeID], (uint64_t)rdma_buffer,
                       conn_to_server_[gaddr.nodeID].dsm_base + gaddr.offset,
                       equal, val, i_con_->cacheLKey,
                       conn_to_server_[gaddr.nodeID].dsm_rkey[0], true,
                       ctx->coro_id);
#endif
    (*ctx->yield)(*ctx->master);
  }
}

bool DSMClient::CasSync(GlobalAddress gaddr, uint64_t equal, uint64_t val,
                        uint64_t *rdma_buffer, CoroContext *ctx) {
  Cas(gaddr, equal, val, rdma_buffer, true, ctx);

  if (ctx == nullptr) {
    ibv_wc wc;
    pollWithCQ(i_con_->cq, 1, &wc);
  }

  return equal == *rdma_buffer;
}


// 需要改
void DSMClient::CasMask(GlobalAddress gaddr, int log_sz, uint64_t equal,
                        uint64_t val, uint64_t *rdma_buffer, uint64_t mask,
                        bool signal, CoroContext *ctx) {
  if (ctx == nullptr) {
    rdmaCompareAndSwapMask(
        i_con_->data[0][gaddr.nodeID], (uint64_t)rdma_buffer,
        conn_to_server_[gaddr.nodeID].dsm_base + gaddr.offset, log_sz, equal,
        val, i_con_->cacheLKey, conn_to_server_[gaddr.nodeID].dsm_rkey[0], mask,
        signal);
  } else {
#ifdef USE_DOORBELL_BATCHING
    int coro_id = ctx->coro_id;
    ibv_exp_send_wr& wr = coro_exp_wr_[coro_id]; // 【关键】使用 exp WR
    ibv_sge& sg = coro_exp_sge_[coro_id];

    sg.addr = (uintptr_t)rdma_buffer;
    sg.length = 1 << log_sz; // 长度由 log_sz 决定
    sg.lkey = i_con_->cacheLKey;

    memset(&wr, 0, sizeof(wr));
    wr.sg_list = &sg;
    wr.num_sge = 1;
    
    wr.exp_opcode = IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP;
    wr.exp_send_flags = IBV_EXP_SEND_EXT_ATOMIC_INLINE;
    if (signal) wr.exp_send_flags |= IBV_EXP_SEND_SIGNALED;
    
    wr.wr_id = coro_id;

    wr.ext_op.masked_atomics.log_arg_sz = log_sz;
    wr.ext_op.masked_atomics.remote_addr = conn_to_server_[gaddr.nodeID].dsm_base + gaddr.offset;
    wr.ext_op.masked_atomics.rkey = conn_to_server_[gaddr.nodeID].dsm_rkey[0];

    auto &op = wr.ext_op.masked_atomics.wr_data.inline_data.op.cmp_swap;
    op.compare_val = equal;
    op.swap_val = val;
    op.compare_mask = mask;
    op.swap_mask = mask;

    queue_exp_wr(gaddr.nodeID, &wr); // 【关键】排入 exp 队列
#else
    rdmaCompareAndSwapMask(
        i_con_->data[0][gaddr.nodeID], (uint64_t)rdma_buffer,
        conn_to_server_[gaddr.nodeID].dsm_base + gaddr.offset, log_sz, equal,
        val, i_con_->cacheLKey, conn_to_server_[gaddr.nodeID].dsm_rkey[0], mask,
        true, ctx->coro_id);
#endif
    (*ctx->yield)(*ctx->master);
  }
}

bool DSMClient::CasMaskSync(GlobalAddress gaddr, int log_sz, uint64_t equal,
                            uint64_t val, uint64_t *rdma_buffer, uint64_t mask,
                            CoroContext *ctx) {
  CasMask(gaddr, log_sz, equal, val, rdma_buffer, mask, true, ctx);

  if (ctx == nullptr) {
    ibv_wc wc;
    pollWithCQ(i_con_->cq, 1, &wc);
  }

  if (log_sz <= 3) {
    return (equal & mask) == (*rdma_buffer & mask);
  } else {
    uint64_t *eq = (uint64_t *)equal;
    uint64_t *old = (uint64_t *)rdma_buffer;
    uint64_t *m = (uint64_t *)mask;
    for (int i = 0; i < (1 << (log_sz - 3)); i++) {
      if ((eq[i] & m[i]) != (__bswap_64(old[i]) & m[i])) {
        return false;
      }
    }
    return true;
  }
}

void DSMClient::CasMaskWrite(RdmaOpRegion &cas_ror, uint64_t equal,
                             uint64_t swap, uint64_t mask,
                             RdmaOpRegion &write_ror, bool signal,
                             CoroContext *ctx) {
  int node_id;
  {
    GlobalAddress gaddr;
    gaddr.raw = cas_ror.dest;
    node_id = gaddr.nodeID;
    FillKeysDest(cas_ror, gaddr, cas_ror.is_on_chip);
  }
  {
    GlobalAddress gaddr;
    gaddr.raw = write_ror.dest;
    FillKeysDest(write_ror, gaddr, write_ror.is_on_chip);
  }

  if (ctx == nullptr) {
    rdmaCasMaskWrite(i_con_->data[0][node_id], cas_ror, equal, swap, mask,
                     write_ror, signal);
  } else {
    rdmaCasMaskWrite(i_con_->data[0][node_id], cas_ror, equal, swap, mask,
                     write_ror, true, ctx->coro_id);
    (*ctx->yield)(*ctx->master);
  }
}

bool DSMClient::CasMaskWriteSync(RdmaOpRegion &cas_ror, uint64_t equal,
                                 uint64_t swap, uint64_t mask,
                                 RdmaOpRegion &write_ror, CoroContext *ctx) {
  CasMaskWrite(cas_ror, equal, swap, mask, write_ror, true, ctx);
  if (ctx == nullptr) {
    ibv_wc wc;
    pollWithCQ(i_con_->cq, 1, &wc);
  }

  if (cas_ror.log_sz <= 3) {
    return (equal & mask) == (*(uint64_t *)cas_ror.source & mask);
  } else {
    uint64_t *eq = (uint64_t *)equal;
    uint64_t *old = (uint64_t *)cas_ror.source;
    uint64_t *m = (uint64_t *)mask;
    for (int i = 0; i < (1 << (cas_ror.log_sz - 3)); ++i) {
      if ((eq[i] & m[i]) != (__bswap_64(old[i]) & m[i])) {
        return false;
      }
    }
    return true;
  }
}

void DSMClient::FaaBound(GlobalAddress gaddr, int log_sz, uint64_t add_val,
                         uint64_t *rdma_buffer, uint64_t mask, bool signal,
                         CoroContext *ctx) {
  if (ctx == nullptr) {
    rdmaFetchAndAddBoundary(
        i_con_->data[0][gaddr.nodeID], log_sz, (uint64_t)rdma_buffer,
        conn_to_server_[gaddr.nodeID].dsm_base + gaddr.offset, add_val,
        i_con_->cacheLKey, conn_to_server_[gaddr.nodeID].dsm_rkey[0], mask,
        signal);
  } else {
    rdmaFetchAndAddBoundary(
        i_con_->data[0][gaddr.nodeID], log_sz, (uint64_t)rdma_buffer,
        conn_to_server_[gaddr.nodeID].dsm_base + gaddr.offset, add_val,
        i_con_->cacheLKey, conn_to_server_[gaddr.nodeID].dsm_rkey[0], mask,
        true, ctx->coro_id);
    (*ctx->yield)(*ctx->master);
  }
}

void DSMClient::FaaBoundSync(GlobalAddress gaddr, int log_sz, uint64_t add_val,
                             uint64_t *rdma_buffer, uint64_t mask,
                             CoroContext *ctx) {
  FaaBound(gaddr, log_sz, add_val, rdma_buffer, mask, true, ctx);
  if (ctx == nullptr) {
    ibv_wc wc;
    pollWithCQ(i_con_->cq, 1, &wc);
  }
}

// 看着改看看
// ================= 看着改看看: ReadDm =================
void DSMClient::ReadDm(char *buffer, GlobalAddress gaddr, size_t size,
                       bool signal, CoroContext *ctx) {
  if (ctx == nullptr) {
    rdmaRead(i_con_->data[0][gaddr.nodeID], (uint64_t)buffer,
             conn_to_server_[gaddr.nodeID].lock_base + gaddr.offset, size,
             i_con_->cacheLKey, conn_to_server_[gaddr.nodeID].lock_rkey[0],
             signal);
  } else {
#ifdef USE_DOORBELL_BATCHING
    int coro_id = ctx->coro_id;
    ibv_send_wr& wr = coro_wr_[coro_id];
    ibv_sge& sg = coro_sge_[coro_id];

    sg.addr = (uintptr_t)buffer;
    sg.length = size;
    sg.lkey = i_con_->cacheLKey;

    memset(&wr, 0, sizeof(wr));
    wr.sg_list = &sg;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    
    if (signal) wr.send_flags = IBV_SEND_SIGNALED; 
    
    // 注意这里是 lock_base 和 lock_rkey
    wr.wr.rdma.remote_addr = conn_to_server_[gaddr.nodeID].lock_base + gaddr.offset;
    wr.wr.rdma.rkey = conn_to_server_[gaddr.nodeID].lock_rkey[0];
    wr.wr_id = coro_id;

    queue_wr(gaddr.nodeID, &wr);
#else
    rdmaRead(i_con_->data[0][gaddr.nodeID], (uint64_t)buffer,
             conn_to_server_[gaddr.nodeID].lock_base + gaddr.offset, size,
             i_con_->cacheLKey, conn_to_server_[gaddr.nodeID].lock_rkey[0],
             true, ctx->coro_id);
#endif
    (*ctx->yield)(*ctx->master);
  }
}

void DSMClient::ReadDmSync(char *buffer, GlobalAddress gaddr, size_t size,
                           CoroContext *ctx) {
  ReadDm(buffer, gaddr, size, true, ctx);

  if (ctx == nullptr) {
    ibv_wc wc;
    pollWithCQ(i_con_->cq, 1, &wc);
  }
}

// 看着改
// ================= 看着改: WriteDm =================
void DSMClient::WriteDm(const char *buffer, GlobalAddress gaddr, size_t size,
                        bool signal, CoroContext *ctx) {
  if (ctx == nullptr) {
    rdmaWrite(i_con_->data[0][gaddr.nodeID], (uint64_t)buffer,
              conn_to_server_[gaddr.nodeID].lock_base + gaddr.offset, size,
              i_con_->cacheLKey, conn_to_server_[gaddr.nodeID].lock_rkey[0], -1,
              signal);
  } else {
#ifdef USE_DOORBELL_BATCHING
    int coro_id = ctx->coro_id;
    ibv_send_wr& wr = coro_wr_[coro_id];
    ibv_sge& sg = coro_sge_[coro_id];

    sg.addr = (uintptr_t)buffer;
    sg.length = size;
    sg.lkey = i_con_->cacheLKey;

    memset(&wr, 0, sizeof(wr));
    wr.sg_list = &sg;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    
    wr.send_flags = 0;
    if (signal) wr.send_flags |= IBV_SEND_SIGNALED;
    // 保留 Inline 优化
    if (size < MAX_INLINE_DATA) wr.send_flags |= IBV_SEND_INLINE;

    // 注意这里是 lock_base 和 lock_rkey
    wr.wr.rdma.remote_addr = conn_to_server_[gaddr.nodeID].lock_base + gaddr.offset;
    wr.wr.rdma.rkey = conn_to_server_[gaddr.nodeID].lock_rkey[0];
    wr.wr_id = coro_id;

    queue_wr(gaddr.nodeID, &wr);
#else
    rdmaWrite(i_con_->data[0][gaddr.nodeID], (uint64_t)buffer,
              conn_to_server_[gaddr.nodeID].lock_base + gaddr.offset, size,
              i_con_->cacheLKey, conn_to_server_[gaddr.nodeID].lock_rkey[0], -1,
              true, ctx->coro_id);
#endif
    (*ctx->yield)(*ctx->master);
  }
}

void DSMClient::WriteDmSync(const char *buffer, GlobalAddress gaddr,
                            size_t size, CoroContext *ctx) {
  WriteDm(buffer, gaddr, size, true, ctx);

  if (ctx == nullptr) {
    ibv_wc wc;
    pollWithCQ(i_con_->cq, 1, &wc);
  }
}

// 看着改
void DSMClient::CasDm(GlobalAddress gaddr, uint64_t equal, uint64_t val,
                      uint64_t *rdma_buffer, bool signal, CoroContext *ctx) {
  if (ctx == nullptr) {
    rdmaCompareAndSwap(i_con_->data[0][gaddr.nodeID], (uint64_t)rdma_buffer,
                       conn_to_server_[gaddr.nodeID].lock_base + gaddr.offset,
                       equal, val, i_con_->cacheLKey,
                       conn_to_server_[gaddr.nodeID].lock_rkey[0], signal);
  } else {
#ifdef USE_DOORBELL_BATCHING
    int coro_id = ctx->coro_id;
    ibv_send_wr& wr = coro_wr_[coro_id];
    ibv_sge& sg = coro_sge_[coro_id];

    sg.addr = (uintptr_t)rdma_buffer;
    sg.length = 8;
    sg.lkey = i_con_->cacheLKey;

    memset(&wr, 0, sizeof(wr));
    wr.sg_list = &sg;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;

    if (signal) wr.send_flags = IBV_SEND_SIGNALED;

    // 注意这里是 lock_base 和 lock_rkey
    wr.wr.atomic.remote_addr = conn_to_server_[gaddr.nodeID].lock_base + gaddr.offset;
    wr.wr.atomic.rkey = conn_to_server_[gaddr.nodeID].lock_rkey[0];
    wr.wr.atomic.compare_add = equal;
    wr.wr.atomic.swap = val;
    wr.wr_id = coro_id;

    queue_wr(gaddr.nodeID, &wr);
#else
    rdmaCompareAndSwap(i_con_->data[0][gaddr.nodeID], (uint64_t)rdma_buffer,
                       conn_to_server_[gaddr.nodeID].lock_base + gaddr.offset,
                       equal, val, i_con_->cacheLKey,
                       conn_to_server_[gaddr.nodeID].lock_rkey[0], true,
                       ctx->coro_id);
#endif
    (*ctx->yield)(*ctx->master);
  }
}

bool DSMClient::CasDmSync(GlobalAddress gaddr, uint64_t equal, uint64_t val,
                          uint64_t *rdma_buffer, CoroContext *ctx) {
  CasDm(gaddr, equal, val, rdma_buffer, true, ctx);

  if (ctx == nullptr) {
    ibv_wc wc;
    pollWithCQ(i_con_->cq, 1, &wc);
  }

  return equal == *rdma_buffer;
}

void DSMClient::CasDmMask(GlobalAddress gaddr, int log_sz, uint64_t equal,
                          uint64_t val, uint64_t *rdma_buffer, uint64_t mask,
                          bool signal, CoroContext *ctx) {
  if (ctx == nullptr) {
    rdmaCompareAndSwapMask(
        i_con_->data[0][gaddr.nodeID], (uint64_t)rdma_buffer,
        conn_to_server_[gaddr.nodeID].lock_base + gaddr.offset, log_sz, equal,
        val, i_con_->cacheLKey, conn_to_server_[gaddr.nodeID].lock_rkey[0],
        mask, signal);
  } else {
    rdmaCompareAndSwapMask(
        i_con_->data[0][gaddr.nodeID], (uint64_t)rdma_buffer,
        conn_to_server_[gaddr.nodeID].lock_base + gaddr.offset, log_sz, equal,
        val, i_con_->cacheLKey, conn_to_server_[gaddr.nodeID].lock_rkey[0],
        mask, true, ctx->coro_id);
    (*ctx->yield)(*ctx->master);
  }
}

bool DSMClient::CasDmMaskSync(GlobalAddress gaddr, int log_sz, uint64_t equal,
                              uint64_t val, uint64_t *rdma_buffer,
                              uint64_t mask, CoroContext *ctx) {
  CasDmMask(gaddr, log_sz, equal, val, rdma_buffer, mask, true, ctx);
  if (ctx == nullptr) {
    ibv_wc wc;
    pollWithCQ(i_con_->cq, 1, &wc);
  }

  if (log_sz <= 3) {
    return (equal & mask) == (*rdma_buffer & mask);
  } else {
    uint64_t *eq = (uint64_t *)equal;
    uint64_t *old = (uint64_t *)rdma_buffer;
    uint64_t *m = (uint64_t *)mask;
    for (int i = 0; i < (1 << (log_sz - 3)); i++) {
      if ((eq[i] & m[i]) != (__bswap_64(old[i]) & m[i])) {
        return false;
      }
    }
    return true;
  }
}

void DSMClient::FaaDmBound(GlobalAddress gaddr, int log_sz, uint64_t add_val,
                           uint64_t *rdma_buffer, uint64_t mask, bool signal,
                           CoroContext *ctx) {
  if (ctx == nullptr) {
    rdmaFetchAndAddBoundary(
        i_con_->data[0][gaddr.nodeID], log_sz, (uint64_t)rdma_buffer,
        conn_to_server_[gaddr.nodeID].lock_base + gaddr.offset, add_val,
        i_con_->cacheLKey, conn_to_server_[gaddr.nodeID].lock_rkey[0], mask,
        signal);
  } else {
    rdmaFetchAndAddBoundary(
        i_con_->data[0][gaddr.nodeID], log_sz, (uint64_t)rdma_buffer,
        conn_to_server_[gaddr.nodeID].lock_base + gaddr.offset, add_val,
        i_con_->cacheLKey, conn_to_server_[gaddr.nodeID].lock_rkey[0], mask,
        true, ctx->coro_id);
    (*ctx->yield)(*ctx->master);
  }
}

void DSMClient::FaaDmBoundSync(GlobalAddress gaddr, int log_sz,
                               uint64_t add_val, uint64_t *rdma_buffer,
                               uint64_t mask, CoroContext *ctx) {
  FaaDmBound(gaddr, log_sz, add_val, rdma_buffer, mask, true, ctx);
  if (ctx == nullptr) {
    ibv_wc wc;
    pollWithCQ(i_con_->cq, 1, &wc);
  }
}

uint64_t DSMClient::PollRdmaCq(int count) {
  ibv_wc wc;
  pollWithCQ(i_con_->cq, count, &wc);

  return wc.wr_id;
}

bool DSMClient::PollRdmaCqOnce(uint64_t &wr_id) {
  ibv_wc wc;
  int res = pollOnce(i_con_->cq, 1, &wc);

  wr_id = wc.wr_id;

  return res == 1;
}

int DSMClient::PollRdmaCqBatch(int max_entries, uint64_t *wr_ids) {
  // 分配临时数组接收完成事件
  ibv_wc wc[max_entries];
  
  // 一次性向底层网卡驱动拉取最多 max_entries 个事件
  int n = pollOnce(i_con_->cq, max_entries, wc);

  // 提取出所有就绪的协程 ID
  for (int i = 0; i < n; ++i) {
    wr_ids[i] = wc[i].wr_id;
  }

  return n;
}
