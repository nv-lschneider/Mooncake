// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NCCL_TRANSPORT_H_
#define NCCL_TRANSPORT_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include "tent/common/concurrent/thread_pool.h"
#include "tent/common/types.h"
#include "tent/platform/cuda.h"
#include "tent/runtime/control_plane.h"
#include "tent/runtime/transport.h"

namespace mooncake {
namespace tent {

struct NcclParams {
    size_t max_concurrent_tasks = 4;
    size_t gin_lanes = 4;
    bool wait_ack = false;
    bool force_gin = false;
};

struct NcclTask {
    Request request;
    std::atomic<TransferStatusEnum> status_word{TransferStatusEnum::PENDING};
    std::atomic<size_t> transferred_bytes{0};
    std::atomic<cudaEvent_t> completion_event{nullptr};

    NcclTask() = default;
    NcclTask(NcclTask&& other) noexcept
        : request(other.request),
          status_word(other.status_word.load(std::memory_order_relaxed)),
          transferred_bytes(
              other.transferred_bytes.load(std::memory_order_relaxed)),
          completion_event(other.completion_event.exchange(
              nullptr, std::memory_order_acq_rel)) {
    }
    NcclTask(const NcclTask&) = delete;
    NcclTask& operator=(const NcclTask&) = delete;
};

struct NcclSubBatch : public Transport::SubBatch {
    std::vector<NcclTask> task_list;
    size_t max_size = 0;
    CUDAStreamHandle stream;
    size_t size() const override { return task_list.size(); }
};

class NcclTransport : public Transport {
   public:
    NcclTransport();
    ~NcclTransport() override;

    Status install(std::string& local_segment_name,
                   std::shared_ptr<ControlService> metadata,
                   std::shared_ptr<Topology> local_topology,
                   std::shared_ptr<Config> conf = nullptr) override;

    Status uninstall() override;

    Status allocateSubBatch(SubBatchRef& batch, size_t max_size) override;

    Status freeSubBatch(SubBatchRef& batch) override;

    Status submitTransferTasks(
        SubBatchRef batch, const std::vector<Request>& request_list) override;

    Status getTransferStatus(SubBatchRef batch, int task_id,
                             TransferStatus& status) override;

    Status allocateLocalMemory(void** addr, size_t size,
                               MemoryOptions& options) override;

    Status freeLocalMemory(void* addr, size_t size) override;

    Status addMemoryBuffer(BufferDesc& desc,
                           const MemoryOptions& options) override;

    Status removeMemoryBuffer(BufferDesc& desc) override;

    Status preconnectSegment(SegmentID target_id);

    Status transferPagedSync(const PagedTransferRequest& request);

    const char* getName() const override { return "nccl"; }

   private:
    struct CommState;
    struct WindowState;
    struct TransferContext;

    bool isNcclAllocated(uint64_t addr) const;
    bool isCudaLocation(const std::string& location) const;
    Status markFailed(NcclTask& task, const std::string& reason);
    Status buildPreconnectContext(SegmentID target_id, TransferContext& ctx);
    Status buildTransferContext(const Request& request, TransferContext& ctx);
    Status buildTransferContext(const Request& request, size_t source_length,
                                size_t target_length, TransferContext& ctx);
    Status ensureComm(const TransferContext& ctx,
                      std::shared_ptr<CommState>& state);
    Status ensureWindow(const TransferContext& ctx,
                        const std::shared_ptr<CommState>& comm_state,
                        std::shared_ptr<WindowState>& state);
    Status ensureSourceWindow(const TransferContext& ctx,
                              const std::shared_ptr<CommState>& comm_state,
                              std::shared_ptr<WindowState>& state);
    Status postRemoteWaitSignal(const TransferContext& ctx,
                                const std::shared_ptr<CommState>& comm_state,
                                uint64_t signal_value);
    Status waitForComm(const std::string& session_key,
                       std::shared_ptr<CommState>& state);
    Status onBootstrapNccl(const NcclBootstrapDesc& request,
                           NcclBootstrapDesc& response);
    Status onRegisterNcclWindow(const NcclWindowDesc& request,
                                NcclWindowDesc& response);
    Status onWaitNcclSignal(const NcclSignalDesc& request,
                            NcclSignalDesc& response);
    void startTransfer(NcclTask* task, NcclSubBatch* batch);
    void startBackground(std::function<void()> fn);

   private:
    bool installed_ = false;
    bool allow_external_window_buffers_ = false;
    int nccl_version_ = 0;
    int default_cuda_device_ = -1;
    std::string local_segment_name_;
    std::shared_ptr<ControlService> metadata_;
    std::shared_ptr<Topology> local_topology_;
    std::shared_ptr<Config> conf_;
    CudaPlatform* platform_ = nullptr;
    NcclParams params_;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::atomic<bool> shutting_down_{false};

    mutable std::mutex allocation_mutex_;
    std::unordered_set<uint64_t> nccl_allocations_;

    std::mutex comm_mutex_;
    std::unordered_map<std::string, std::shared_ptr<CommState>> comms_;
    std::mutex window_mutex_;
    std::unordered_map<std::string, std::shared_ptr<WindowState>> windows_;
    std::mutex background_mutex_;
    std::vector<std::thread> background_threads_;
};

}  // namespace tent
}  // namespace mooncake

#endif  // NCCL_TRANSPORT_H_
