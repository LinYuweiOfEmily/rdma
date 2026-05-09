#ifndef __RAWMESSAGECONNECTION_H__
#define __RAWMESSAGECONNECTION_H__

#include "AbstractMessageConnection.h"
#include "GlobalAddress.h"

#include <thread>

enum RpcType : uint8_t {
  MALLOC,
  FREE,
  NEW_ROOT,
  SPLIT_GUARD_BLOCK,
  SPLIT_GUARD_ACK,
  SPLIT_GUARD_UNBLOCK,
  TERMINATE,
  NOP,
};

struct RawMessage {
  RpcType type;
  
  uint16_t node_id;
  uint16_t app_id;

  GlobalAddress addr; // for malloc
  int level;
  uint64_t arg0;
  uint64_t arg1;
  uint16_t requester_node_id;
  uint16_t requester_app_id;
} __attribute__((packed));

class RawMessageConnection : public AbstractMessageConnection {

public:
  RawMessageConnection(RdmaContext &ctx, ibv_cq *cq, uint32_t messageNR);

  void initSend();
  void sendRawMessage(RawMessage *m, uint32_t remoteQPN, ibv_ah *ah);
};

#endif /* __RAWMESSAGECONNECTION_H__ */
