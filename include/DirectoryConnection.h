#ifndef __DIRECTORYCONNECTION_H__
#define __DIRECTORYCONNECTION_H__

#include "Common.h"
#include "RawMessageConnection.h"

struct RemoteConnectionToClient;

// directory thread
struct DirectoryConnection {
  static constexpr uint16_t kSplitGuardControlAppID =
      define::kSplitGuardControlAppID;
  uint16_t dirID;

  RdmaContext ctx;
  ibv_cq *cq;

  RawMessageConnection *message;
  uint32_t num_client;

  ibv_qp **data2app[MAX_APP_THREAD];

  ibv_mr *dsmMR;
  void *dsmPool;
  uint64_t dsmSize;
  uint32_t dsmLKey;

  ibv_mr *lockMR;
  void *lockPool; // address on-chip
  uint64_t lockSize;
  uint32_t lockLKey;

  RemoteConnectionToClient *remote_con_;

  DirectoryConnection(uint16_t dirID, void *dsmPool, uint64_t dsmSize,
                      uint32_t machineNR, uint16_t rnic_id,
                      RemoteConnectionToClient *remote_con);

  void sendMessage2App(RawMessage *m, uint16_t node_id, uint16_t th_id);
  void broadcastMessage2App(RawMessage *m, uint16_t th_id,
                            uint16_t skip_node_id);
  void broadcastMessage2ControlApp(RawMessage *m, uint16_t skip_node_id);
};

#endif /* __DIRECTORYCONNECTION_H__ */
