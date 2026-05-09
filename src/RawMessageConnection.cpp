#include "RawMessageConnection.h"

#include <cassert>
#include <cerrno>
#include <cstring>

RawMessageConnection::RawMessageConnection(RdmaContext &ctx, ibv_cq *cq,
                                           uint32_t messageNR)
    : AbstractMessageConnection(IBV_QPT_UD, 0, 40, ctx, cq, messageNR) {}

void RawMessageConnection::initSend() {}

void RawMessageConnection::sendRawMessage(RawMessage *m, uint32_t remoteQPN,
                                          ibv_ah *ah) {

  if ((sendCounter & SIGNAL_BATCH) == 0 && sendCounter > 0) {
    ibv_wc wc;
    int count = 0;
    int retry = 0;
    do {
      int n = ibv_poll_cq(send_cq, 1, &wc);
      if (n < 0) {
        count = n;
        break;
      }
      if (n == 0) {
        ++retry;
      } else {
        count = n;
        retry = 0;
      }
    } while (count < 1 && retry < 1000);

    if (count < 0) {
      Debug::notifyError(
          "RawMessage send CQ poll failed: sendCounter=%llu remoteQPN=%u "
          "errno=%d (%s)",
          (unsigned long long)sendCounter, remoteQPN, errno, strerror(errno));
    } else if (count == 0) {
      Debug::notifyError(
          "RawMessage send CQ poll timed out: sendCounter=%llu remoteQPN=%u",
          (unsigned long long)sendCounter, remoteQPN);
    } else if (count > 0 && wc.status != IBV_WC_SUCCESS) {
      Debug::notifyError(
          "RawMessage send CQ error: status=%s (%d) wr_id=%llu sendCounter=%llu "
          "remoteQPN=%u type=%u dst_node=%u dst_app=%u requester_node=%u "
          "requester_app=%u",
          ibv_wc_status_str(wc.status), wc.status, (unsigned long long)wc.wr_id,
          (unsigned long long)sendCounter, remoteQPN,
          static_cast<unsigned>(m->type), static_cast<unsigned>(m->node_id),
          static_cast<unsigned>(m->app_id),
          static_cast<unsigned>(m->requester_node_id),
          static_cast<unsigned>(m->requester_app_id));
    }
  }

  ibv_sge sg;
  memset(&sg, 0, sizeof(sg));
  sg.addr = (uintptr_t)m - sendPadding;
  sg.length = sizeof(RawMessage) + sendPadding;
  sg.lkey = messageLkey;

  ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.sg_list = &sg;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.wr.ud.ah = ah;
  wr.wr.ud.remote_qpn = remoteQPN;
  wr.wr.ud.remote_qkey = UD_PKEY;
  if ((sendCounter & SIGNAL_BATCH) == 0) {
    wr.send_flags = IBV_SEND_SIGNALED;
  }

  ibv_send_wr *bad_wr = nullptr;
  int post_ret = ibv_post_send(message, &wr, &bad_wr);
  if (post_ret != 0) {
    Debug::notifyError(
        "RawMessage post_send failed: ret=%d errno=%d (%s) sendCounter=%llu "
        "remoteQPN=%u type=%u dst_node=%u dst_app=%u requester_node=%u "
        "requester_app=%u bad_wr=%p",
        post_ret, errno, strerror(errno), (unsigned long long)sendCounter,
        remoteQPN, static_cast<unsigned>(m->type),
        static_cast<unsigned>(m->node_id), static_cast<unsigned>(m->app_id),
        static_cast<unsigned>(m->requester_node_id),
        static_cast<unsigned>(m->requester_app_id), (void *)bad_wr);
  }

  ++sendCounter;
}
