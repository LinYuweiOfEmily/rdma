#define STRIP_FLAG_HELP 1    // this must go before the #include!
#include <gflags/gflags.h>
#include "Timer.h"
#include "Tree.h"
#include "Common.h"

#include <city.h>
#include <stdlib.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <random>

//////////////////// workload parameters /////////////////////

DEFINE_int32(numa_id, 0, "numa node id");
DEFINE_int32(server_count, 1, "server count");
DEFINE_int32(client_count, 1, "client count");
DEFINE_int32(num_bench_threads, 1, "bench thread");

//////////////////// workload parameters /////////////////////

std::thread th[MAX_APP_THREAD];
uint64_t tp[MAX_APP_THREAD][8];

Tree *tree;
DSMClient *dsm_client;

std::atomic<int64_t> warmup_cnt{0};
std::atomic_bool ready{false};

// ===================================================================
// 【核心修改】：纯硬件物理极限测试 (Microbenchmark)
// ===================================================================
void microbench_thread_run(int id) {
  // bindCore(id);
  dsm_client->RegisterThread();

  // 1. 等待所有测试线程就绪
  warmup_cnt.fetch_add(1);
  if (id == 0) {
    while (warmup_cnt.load() != FLAGS_num_bench_threads)
      ;
    printf("All threads registered. Starting Microbenchmark...\n");
    ready = true;
    warmup_cnt.store(0);
  }
  while (!ready.load())
    ;

  // 2. 准备本地接收 Buffer
  char *local_buf = dsm_client->get_rdma_buffer();
  
  // 假设远端有 1GB 可用内存空间，防止越界
  uint64_t remote_memory_size = 1024ull * 1024 * 1024; 
  unsigned int seed = dsm_client->get_my_client_id() * 100 + id;

  Timer timer;
  
  // 3. 极简的主循环：疯狂发起底层 RDMA 请求
  while (true) {
    // 随机生成 64 字节对齐的物理地址偏移 (模拟极度分散的读请求)
    uint64_t random_offset = (rand_r(&seed) % (remote_memory_size / 64)) * 64;
    
    GlobalAddress target_addr;
    target_addr.nodeID = 0; // 假设向 Node 0 发起请求，若有多台可按需取模
    target_addr.offset = random_offset;

    timer.begin();
    
    // 【扒掉底裤的裸测】：直接调 ReadSync，无 B+ 树，无并发控制，无协程切换
    dsm_client->ReadSync(local_buf, target_addr, 64, nullptr);

    auto t = timer.end();
    
    // 记录延迟和吞吐量
    stat_helper.add(id, lat_op, t);
    tp[id][0]++;
  }
}

void print_args() {
  printf(
      "Microbenchmark Mode -> ServerCount %d, ClientCount %d, BenchThreadCount %d\n",
      FLAGS_server_count, FLAGS_client_count, FLAGS_num_bench_threads);
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  print_args();

  DSMConfig config;
  config.rnic_id = FLAGS_numa_id;
  config.num_server = FLAGS_server_count;
  config.num_client = FLAGS_client_count;
  dsm_client = DSMClient::GetInstance(config);

  dsm_client->RegisterThread();
  
  // 仅为了初始化底层的 Chunk 分配，不需要对其做任何 Insert 操作
  tree = new Tree(dsm_client); 

  dsm_client->Barrier("benchmark");

  // 启动压测线程
  for (int i = 0; i < FLAGS_num_bench_threads; i++) {
    th[i] = std::thread(microbench_thread_run, i);
  }

  timespec s, e;
  uint64_t pre_tp = 0;

  clock_gettime(CLOCK_REALTIME, &s);
  while (true) {
    sleep(2);
    clock_gettime(CLOCK_REALTIME, &e);
    int microseconds = (e.tv_sec - s.tv_sec) * 1000000 +
                       (double)(e.tv_nsec - s.tv_nsec) / 1000;

    uint64_t all_tp = 0;
    for (int i = 0; i < FLAGS_num_bench_threads; ++i) {
      all_tp += tp[i][0];
    }
    uint64_t cap = all_tp - pre_tp;
    pre_tp = all_tp;

    uint64_t stat_lat[lat_end];
    uint64_t stat_cnt[lat_end];
    for (int k = 0; k < lat_end; k++) {
      stat_lat[k] = 0;
      stat_cnt[k] = 0;
      for (int i = 0; i < MAX_APP_THREAD; ++i) {
        stat_lat[k] += stat_helper.latency_[i][k];
        stat_helper.latency_[i][k] = 0;
        stat_cnt[k] += stat_helper.counter_[i][k];
        stat_helper.counter_[i][k] = 0;
      }
    }

    clock_gettime(CLOCK_REALTIME, &s);

    double per_node_tp = cap * 1.0 / microseconds;
    uint64_t cluster_tp = dsm_client->Sum((uint64_t)(per_node_tp * 1000));

    printf("CN: %d, throughput %.4f Mops/s\n", dsm_client->get_my_client_id(),
           per_node_tp);

    if (dsm_client->get_my_client_id() == 0) {
      printf("cluster throughput %.3f Mops/s\n", cluster_tp / 1000.0);
      printf("%d avg pure read latency: %.3lf us\n", dsm_client->get_my_client_id(),
             (double)stat_lat[lat_op] / stat_cnt[lat_op] / 1000.0);
    }
  }

  return 0;
}