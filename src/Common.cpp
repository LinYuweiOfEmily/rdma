#include "Common.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <net/if.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <arpa/inet.h>

namespace {

constexpr uint16_t kPhysicalCoresPerNuma = 28;
constexpr uint16_t kLogicalCoresPerNuma = 56;

uint16_t mapCoreToCpu(uint16_t core, uint16_t numa_id) {
  const uint16_t preferred_numa = numa_id % 2;
  const uint16_t local_core = core % kLogicalCoresPerNuma;
  const uint16_t target_numa =
      (core < kLogicalCoresPerNuma) ? preferred_numa : 1 - preferred_numa;

  if (target_numa == 0) {
    return local_core < kPhysicalCoresPerNuma
               ? local_core
               : 56 + (local_core - kPhysicalCoresPerNuma);
  }

  return local_core < kPhysicalCoresPerNuma
             ? 28 + local_core
             : 84 + (local_core - kPhysicalCoresPerNuma);
}

void bindCpu(uint16_t cpu) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu, &cpuset);
  int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    Debug::notifyError("can't bind core!");
  }
}

}  // namespace

void bindCore(uint16_t core) { bindCoreToNuma(core, 0); }

void bindCoreToNuma(uint16_t core, uint16_t numa_id) {
  bindCpu(mapCoreToCpu(core, numa_id));
}

char *getIP() {
  struct ifreq ifr;
  int fd = socket(AF_INET, SOCK_DGRAM, 0);

  ifr.ifr_addr.sa_family = AF_INET;
  strncpy(ifr.ifr_name, "ib0", IFNAMSIZ - 1);

  ioctl(fd, SIOCGIFADDR, &ifr);
  close(fd);

  return inet_ntoa(((struct sockaddr_in *)&ifr.ifr_addr)->sin_addr);
}

char *getMac() {
  static struct ifreq ifr;
  int fd = socket(AF_INET, SOCK_DGRAM, 0);

  ifr.ifr_addr.sa_family = AF_INET;
  strncpy(ifr.ifr_name, "ens2", IFNAMSIZ - 1);

  ioctl(fd, SIOCGIFHWADDR, &ifr);
  close(fd);

  return (char *)ifr.ifr_hwaddr.sa_data;
}
