#include "DirectoryConnection.h"

#include "connection.h"

DirectoryConnection::DirectoryConnection(uint16_t dirID, void *dsmPool,
                                         uint64_t dsmSize, uint32_t num_client,
                                         uint16_t rnic_id,
                                         RemoteConnectionToClient *remote_con)
    : dirID(dirID), num_client(num_client), remote_con_(remote_con) {
  createContext(&ctx, rnic_id);
  cq = ibv_create_cq(ctx.ctx, RAW_RECV_CQ_COUNT, NULL, NULL, 0);
  message = new RawMessageConnection(ctx, cq, DIR_MESSAGE_NR);

  message->initRecv();
  message->initSend();

  // dsm memory
  this->dsmPool = dsmPool;
  this->dsmSize = dsmSize;
  this->dsmMR = createMemoryRegion((uint64_t)dsmPool, dsmSize, &ctx);
  this->dsmLKey = dsmMR->lkey;

  // on-chip lock memory
  if (dirID == 0) {
    this->lockPool = (void *)define::kLockStartAddr;
    this->lockSize = define::kLockChipMemSize;
    this->lockMR = createMemoryRegionOnChip((uint64_t)this->lockPool,
                                            this->lockSize, &ctx);
    this->lockLKey = lockMR->lkey;
  }

  // app, RC
  for (int i = 0; i < MAX_APP_THREAD; ++i) {
    data2app[i] = new ibv_qp *[num_client];
    // client
    for (size_t k = 0; k < num_client; ++k) {
      createQueuePair(&data2app[i][k], IBV_QPT_RC, cq, &ctx);
    }
  }
}

void DirectoryConnection::sendMessage2App(RawMessage *m, uint16_t node_id,
                                          uint16_t th_id) {
  if (node_id >= num_client || th_id >= MAX_APP_THREAD) {
    Debug::notifyError(
        "Invalid sendMessage2App target: node_id=%u num_client=%u th_id=%u "
        "MAX_APP_THREAD=%u type=%u src_node=%u src_app=%u requester_node=%u "
        "requester_app=%u",
        static_cast<unsigned>(node_id), num_client, static_cast<unsigned>(th_id),
        static_cast<unsigned>(MAX_APP_THREAD), static_cast<unsigned>(m->type),
        static_cast<unsigned>(m->node_id), static_cast<unsigned>(m->app_id),
        static_cast<unsigned>(m->requester_node_id),
        static_cast<unsigned>(m->requester_app_id));
    assert(false);
    return;
  }
  message->sendRawMessage(m, remote_con_[node_id].app_message_qpn[th_id],
                          remote_con_[node_id].dir_to_app_ah[dirID][th_id]);
}

void DirectoryConnection::broadcastMessage2App(RawMessage *m, uint16_t th_id,
                                               uint16_t skip_node_id) {
  for (uint16_t client_id = 0; client_id < num_client; ++client_id) {
    if (client_id == skip_node_id) {
      continue;
    }
    sendMessage2App(m, client_id, th_id);
  }
}

void DirectoryConnection::broadcastMessage2ControlApp(RawMessage *m,
                                                      uint16_t skip_node_id) {
  broadcastMessage2App(m, kSplitGuardControlAppID, skip_node_id);
}
