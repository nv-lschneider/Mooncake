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

#include "tent/transport/nccl/nccl_transport.h"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

#include "tent/common/status.h"
#include "tent/runtime/platform.h"
#include "tent/runtime/segment.h"
#include "tent/runtime/slab.h"
#include "tent/runtime/topology.h"
#include "tent/transport/nccl/paged_gin.h"

extern "C" cudaError_t tentNcclGinLaunchPut(
    ncclDevComm_t dev_comm, int peer, int lanes, ncclWindow_t dst_window,
    size_t dst_offset, ncclWindow_t src_window, size_t src_offset,
    size_t total_bytes, unsigned long long signal_value, cudaStream_t stream);

extern "C" cudaError_t tentNcclGinLaunchGet(
    ncclDevComm_t dev_comm, int peer, int lanes, ncclWindow_t remote_window,
    size_t remote_offset, ncclWindow_t local_window, size_t local_offset,
    size_t total_bytes, cudaStream_t stream);

extern "C" cudaError_t tentNcclGinLaunchWaitSignal(
    ncclDevComm_t dev_comm, int lanes, int signal_base,
    unsigned long long signal_value, cudaStream_t stream);

extern "C" cudaError_t tentNcclGinLaunchWaitAck(
    ncclDevComm_t dev_comm, int peer, int lanes,
    unsigned long long signal_value, cudaStream_t stream);

namespace mooncake {
namespace tent {
namespace {

Status ncclStatus(ncclResult_t result, const char* expr) {
    if (result == ncclSuccess) return Status::OK();
    return Status::InternalError(std::string(expr) + ": " +
                                 ncclGetErrorString(result) + LOC_MARK);
}

Status cudaStatus(cudaError_t result, const char* expr) {
    if (result == cudaSuccess) return Status::OK();
    return Status::InternalError(std::string(expr) + ": " +
                                 cudaGetErrorString(result) + LOC_MARK);
}

bool envFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value) return false;
    std::string flag(value);
    std::transform(flag.begin(), flag.end(), flag.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return flag == "1" || flag == "true" || flag == "yes" || flag == "on";
}

bool pagedGinDiagEnabled() {
    return envFlagEnabled("TENT_NCCL_PAGED_DIAG") ||
           envFlagEnabled("TRTLLM_MOONCAKE_PAGED_GIN_DIAG");
}

#define CHECK_NCCL(call)                       \
    do {                                       \
        Status _s = ncclStatus(call, #call);   \
        if (!_s.ok()) return _s;               \
    } while (0)

int hexValue(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

std::string serializeUniqueId(const ncclUniqueId& id) {
    static constexpr char kHex[] = "0123456789abcdef";
    const auto* bytes = reinterpret_cast<const unsigned char*>(&id);
    std::string out;
    out.resize(sizeof(id) * 2);
    for (size_t i = 0; i < sizeof(id); ++i) {
        out[2 * i] = kHex[bytes[i] >> 4];
        out[2 * i + 1] = kHex[bytes[i] & 0xf];
    }
    return out;
}

uint64_t uniqueIdFingerprint(const ncclUniqueId& id) {
    constexpr uint64_t kOffset = 1469598103934665603ULL;
    constexpr uint64_t kPrime = 1099511628211ULL;
    uint64_t hash = kOffset;
    const auto* bytes = reinterpret_cast<const unsigned char*>(&id);
    for (size_t i = 0; i < sizeof(id); ++i) {
        hash ^= bytes[i];
        hash *= kPrime;
    }
    return hash;
}

Status deserializeUniqueId(const std::string& raw, ncclUniqueId& id) {
    if (raw.size() != sizeof(id) * 2) {
        return Status::InvalidArgument(
            "Invalid NCCL unique id size" LOC_MARK);
    }
    auto* bytes = reinterpret_cast<unsigned char*>(&id);
    for (size_t i = 0; i < sizeof(id); ++i) {
        int hi = hexValue(raw[2 * i]);
        int lo = hexValue(raw[2 * i + 1]);
        if (hi < 0 || lo < 0) {
            return Status::InvalidArgument(
                "Invalid NCCL unique id encoding" LOC_MARK);
        }
        bytes[i] = static_cast<unsigned char>((hi << 4) | lo);
    }
    return Status::OK();
}

std::string makeSessionKey(const std::string& local_name,
                           const std::string& remote_name, int local_device,
                           int remote_device) {
    std::ostringstream local_endpoint;
    local_endpoint << local_name << ":cuda" << local_device;
    std::ostringstream remote_endpoint;
    remote_endpoint << remote_name << ":cuda" << remote_device;
    const std::string local_endpoint_key = local_endpoint.str();
    const std::string remote_endpoint_key = remote_endpoint.str();
    const auto& first = local_endpoint_key < remote_endpoint_key
                            ? local_endpoint_key
                            : remote_endpoint_key;
    const auto& second = local_endpoint_key < remote_endpoint_key
                             ? remote_endpoint_key
                             : local_endpoint_key;
    std::ostringstream ss;
    ss << "nccl-pair:" << first << "<->" << second;
    return ss.str();
}

Status findSingleNcclCudaDevice(const SegmentDesc& segment,
                                const char* endpoint, int& device) {
    if (segment.type != SegmentType::Memory) {
        return Status::InvalidArgument(std::string(endpoint) +
                                       " segment is not memory" LOC_MARK);
    }

    const auto& memory = segment.getMemory();
    const auto* transport_attrs = memory.getTransportAttrs(NCCL);
    if (transport_attrs && !transport_attrs->empty()) {
        try {
            const auto attrs = json::parse(*transport_attrs);
            if (attrs.contains("device_index")) {
                device = attrs.at("device_index").get<int>();
                if (device >= 0) return Status::OK();
            }
        } catch (const std::exception& e) {
            return Status::InvalidArgument(
                std::string(endpoint) +
                " segment has invalid NCCL endpoint metadata: " + e.what() +
                LOC_MARK);
        }
    }

    device = -1;
    for (const auto& buffer : memory.buffers) {
        if (std::find(buffer.transports.begin(), buffer.transports.end(), NCCL) ==
            buffer.transports.end()) {
            continue;
        }
        LocationParser location(buffer.location);
        if (location.type() != "cuda" || location.index() < 0) continue;
        if (device >= 0 && device != location.index()) {
            return Status::InvalidArgument(
                std::string(endpoint) +
                " segment has NCCL buffers on multiple CUDA devices" LOC_MARK);
        }
        device = location.index();
    }
    if (device < 0) {
        return Status::InvalidArgument(std::string(endpoint) +
                                       " segment has no NCCL CUDA buffer" LOC_MARK);
    }
    return Status::OK();
}

std::string makeWindowKey(const std::string& session_key,
                          const char* purpose, uint64_t base,
                          uint64_t length) {
    std::ostringstream ss;
    ss << session_key << ":window:" << purpose << ":" << std::hex << base
       << ":" << length;
    return ss.str();
}

std::string makePoolWindowKey(
    const std::string& session_key, const std::string& local_name,
    int local_device, uint64_t local_pool_base, uint64_t local_pool_length,
    const std::string& remote_name, int remote_device,
    uint64_t remote_pool_base, uint64_t remote_pool_length) {
    auto endpoint = [](const std::string& name, int device, uint64_t base,
                       uint64_t length) {
        std::ostringstream ss;
        ss << name << ":cuda" << device << ":pool:" << std::hex << base
           << ":" << length;
        return ss.str();
    };
    const std::string local = endpoint(local_name, local_device,
                                       local_pool_base, local_pool_length);
    const std::string remote = endpoint(remote_name, remote_device,
                                        remote_pool_base, remote_pool_length);
    const auto& first = local < remote ? local : remote;
    const auto& second = local < remote ? remote : local;
    std::ostringstream ss;
    ss << session_key << ":window:paged-pool:" << first << "<->" << second;
    return ss.str();
}

Status setCudaDevice(int device, int& previous_device) {
    CHECK_CUDA(cudaGetDevice(&previous_device));
    CHECK_CUDA(cudaSetDevice(device));
    return Status::OK();
}

Status initCommBlocking(ncclComm_t* comm, const ncclUniqueId& unique_id,
                        int rank, const char* expr) {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 1;
    return ncclStatus(ncclCommInitRankConfig(comm, 2, unique_id, rank, &config),
                      expr);
}

bool peerInLsaTeam(ncclComm_t comm, int peer_rank) {
    ncclTeam_t world = ncclTeamWorld(comm);
    ncclTeam_t lsa = ncclTeamLsa(comm);
    return ncclTeamRankIsMember(lsa, world, peer_rank);
}

Status getLsaPeerPointer(ncclWindow_t window, size_t offset, int peer,
                         void** ptr) {
    *ptr = nullptr;
    CHECK_NCCL(ncclGetPeerDevicePointer(window, offset, peer, ptr));
    if (!*ptr) {
        return Status::InvalidArgument(
            "NCCL peer window is not LSA reachable" LOC_MARK);
    }
    return Status::OK();
}

}  // namespace

struct NcclTransport::CommState {
    std::mutex mu;
    std::mutex collective_mu;
    std::mutex enqueue_mu;
    std::mutex remote_signal_mu;
    std::condition_variable remote_signal_cv;
    std::condition_variable cv;
    ncclComm_t comm = nullptr;
    ncclDevComm_t dev_comm{};
    bool dev_comm_created = false;
    cudaStream_t completion_stream = nullptr;
    std::atomic<uint64_t> signal_epoch{0};
    uint64_t next_remote_signal = 1;
    size_t lanes = 1;
    Status status;
    bool initializing = false;
    bool ready = false;
    bool comm_init_ready = false;
    bool device_init_requested = false;
    bool peer_in_lsa = false;
    int device_index = -1;
    int local_rank = -1;
    int peer_rank = -1;
};

struct NcclBootstrapPrepared {
    std::mutex mu;
    std::condition_variable cv;
    Status status;
    bool ready = false;
};

struct NcclTransport::WindowState {
    std::mutex mu;
    std::condition_variable cv;
    ncclWindow_t window = nullptr;
    void* local_buffer = nullptr;
    uint64_t length = 0;
    bool owns_local_buffer = false;
    bool initializing = false;
    bool ready = false;
    int device_index = -1;
    std::string session_key;
    Status status;
};

struct NcclTransport::TransferContext {
    SegmentID target_id = 0;
    std::string remote_segment_name;
    std::string remote_rpc_addr;
    uint64_t target_base = 0;
    uint64_t target_length = 0;
    uint64_t target_offset = 0;
    uint64_t source_base = 0;
    uint64_t source_length = 0;
    uint64_t source_offset = 0;
    uint64_t target_pool_base = 0;
    uint64_t target_pool_length = 0;
    uint64_t source_pool_base = 0;
    uint64_t source_pool_length = 0;
    uint64_t target_visible_length = 0;
    uint64_t source_visible_length = 0;
    int local_device = -1;
    int remote_device = -1;
    std::string session_key;
    std::string window_key;
    std::string source_window_key;
    bool use_paired_window_buffers = false;
};

NcclTransport::NcclTransport() = default;

NcclTransport::~NcclTransport() { uninstall(); }

Status NcclTransport::install(std::string& local_segment_name,
                              std::shared_ptr<ControlService> metadata,
                              std::shared_ptr<Topology> local_topology,
                              std::shared_ptr<Config> conf) {
    if (installed_) {
        return Status::InvalidArgument(
            "NCCL transport has been installed" LOC_MARK);
    }

    if (Platform::getLoader().type() != "cuda") {
        return Status::InvalidArgument(
            "NCCL transport requires CUDA platform" LOC_MARK);
    }

    platform_ = dynamic_cast<CudaPlatform*>(&Platform::getLoader());
    if (!platform_) {
        return Status::InvalidArgument(
            "NCCL transport could not load CUDA platform" LOC_MARK);
    }
    CHECK_CUDA(cudaGetDevice(&default_cuda_device_));

    CHECK_NCCL(ncclGetVersion(&nccl_version_));
    if (nccl_version_ < 23000) {
        return Status::InvalidArgument(
            "NCCL device GIN GET requires NCCL 2.30 or newer" LOC_MARK);
    }

    metadata_ = std::move(metadata);
    local_segment_name_ = local_segment_name;
    local_topology_ = std::move(local_topology);
    conf_ = std::move(conf);
    allow_external_window_buffers_ =
        conf_ ? conf_->get("transports/nccl/allow_external_window_buffers",
                           false)
              : false;
    if (conf_) {
        params_.max_concurrent_tasks =
            conf_->get("transports/nccl/max_concurrent_tasks",
                       params_.max_concurrent_tasks);
        params_.gin_lanes =
            conf_->get("transports/nccl/gin_lanes", params_.gin_lanes);
        params_.wait_ack =
            conf_->get("transports/nccl/wait_ack", params_.wait_ack);
        params_.force_gin =
            conf_->get("transports/nccl/force_gin", params_.force_gin);
    }
    if (std::getenv("MC_NCCL_FORCE_GIN")) {
        params_.force_gin = true;
    }
    if (params_.max_concurrent_tasks == 0) params_.max_concurrent_tasks = 1;
    if (params_.gin_lanes == 0) params_.gin_lanes = 1;
    if (params_.gin_lanes > 16) params_.gin_lanes = 16;
    if (params_.gin_lanes > 1 && !std::getenv("NCCL_GIN_NCONNECTIONS")) {
        const std::string connections = std::to_string(params_.gin_lanes);
        if (setenv("NCCL_GIN_NCONNECTIONS", connections.c_str(), 0) != 0) {
            LOG(WARNING) << "Failed to set NCCL_GIN_NCONNECTIONS="
                         << connections;
        } else {
            LOG(INFO) << "Defaulting NCCL_GIN_NCONNECTIONS=" << connections
                      << " for NCCL GIN context striping";
        }
    }
    shutting_down_.store(false, std::memory_order_release);
    thread_pool_ = std::make_unique<ThreadPool>(params_.max_concurrent_tasks);

    metadata_->setBootstrapNcclCallback(
        [this](const NcclBootstrapDesc& request, NcclBootstrapDesc& response) {
            auto status = onBootstrapNccl(request, response);
            if (!status.ok()) response.reply_msg = status.ToString();
            return status.ok() ? 0 : -1;
        });
    metadata_->setNcclWindowCallback(
        [this](const NcclWindowDesc& request, NcclWindowDesc& response) {
            auto status = onRegisterNcclWindow(request, response);
            if (!status.ok()) response.reply_msg = status.ToString();
            return status.ok() ? 0 : -1;
        });
    metadata_->setNcclSignalCallback(
        [this](const NcclSignalDesc& request, NcclSignalDesc& response) {
            auto status = onWaitNcclSignal(request, response);
            if (!status.ok()) response.reply_msg = status.ToString();
            return status.ok() ? 0 : -1;
        });

    // Cross-node transfers use device-side NCCL GIN. Same-node LSA peers use
    // NCCL peer device pointers and CUDA D2D copies over NVLink.
    caps.gpu_to_gpu = true;
    auto local_segment = metadata_->segmentManager().getLocal();
    if (!local_segment || local_segment->type != SegmentType::Memory) {
        return Status::InvalidArgument(
            "NCCL transport local segment is not memory" LOC_MARK);
    }
    auto& local_memory = std::get<MemorySegmentDesc>(local_segment->detail);
    local_memory.transport_attrs[TransportType::NCCL] =
        json{{"device_index", default_cuda_device_}}.dump();
    CHECK_STATUS(metadata_->segmentManager().synchronizeLocal());
    installed_ = true;

    LOG(INFO) << "NCCL transport installed: version=" << nccl_version_
              << " default_cuda_device=" << default_cuda_device_
              << " allow_external_window_buffers="
              << allow_external_window_buffers_
              << " max_concurrent_tasks=" << params_.max_concurrent_tasks
              << " gin_lanes=" << params_.gin_lanes
              << " wait_ack=" << params_.wait_ack
              << " force_gin=" << params_.force_gin;
    return Status::OK();
}

Status NcclTransport::uninstall() {
    if (!installed_) return Status::OK();

    if (metadata_) {
        metadata_->setBootstrapNcclCallback(nullptr);
        metadata_->setNcclWindowCallback(nullptr);
        metadata_->setNcclSignalCallback(nullptr);
    }

    shutting_down_.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(comm_mutex_);
        for (auto& [_, comm] : comms_) {
            if (comm) comm->remote_signal_cv.notify_all();
        }
    }
    thread_pool_.reset();

    std::vector<std::thread> background_threads;
    {
        std::lock_guard<std::mutex> lock(background_mutex_);
        background_threads.swap(background_threads_);
    }
    for (auto& thread : background_threads) {
        if (thread.joinable()) thread.join();
    }

    {
        std::lock_guard<std::mutex> lock(window_mutex_);
        for (auto& [_, window] : windows_) {
            if (!window) continue;
            std::unique_lock<std::mutex> state_lock(window->mu);
            if (window->ready && window->window) {
                std::shared_ptr<CommState> comm_state;
                ncclComm_t comm = nullptr;
                {
                    std::lock_guard<std::mutex> comm_lock(comm_mutex_);
                    auto comm_it = comms_.find(window->session_key);
                    if (comm_it != comms_.end()) comm_state = comm_it->second;
                }
                bool lsa_session = false;
                if (comm_state) {
                    std::lock_guard<std::mutex> comm_state_lock(
                        comm_state->mu);
                    if (comm_state->ready) {
                        comm = comm_state->comm;
                        lsa_session = comm_state->peer_in_lsa;
                    }
                }
                if (lsa_session) {
                    LOG(INFO) << "Deferring NCCL LSA window cleanup for session "
                              << window->session_key;
                    continue;
                }
                if (!comm) {
                    LOG(WARNING) << "Skipping NCCL window deregister without "
                                    "ready communicator for session "
                                 << window->session_key;
                    continue;
                }
                int previous_device = 0;
                auto status = setCudaDevice(window->device_index,
                                            previous_device);
                if (status.ok()) {
                    auto result =
                        ncclCommWindowDeregister(comm, window->window);
                    if (result != ncclSuccess)
                        LOG(WARNING) << "ncclCommWindowDeregister failed: "
                                     << ncclGetErrorString(result);
                    cudaSetDevice(previous_device);
                }
            }
            if (window->owns_local_buffer && window->local_buffer) {
                auto result = ncclMemFree(window->local_buffer);
                if (result != ncclSuccess)
                    LOG(WARNING) << "ncclMemFree window buffer failed: "
                                 << ncclGetErrorString(result);
            }
        }
        windows_.clear();
    }

    {
        std::lock_guard<std::mutex> lock(comm_mutex_);
        for (auto& [_, comm] : comms_) {
            if (comm && comm->ready) {
                int previous_device = 0;
                auto status = setCudaDevice(comm->device_index,
                                            previous_device);
                if (status.ok() && comm->comm) {
                    if (comm->peer_in_lsa) {
                        LOG(INFO) << "Deferring NCCL LSA communicator cleanup "
                                     "for device "
                                  << comm->device_index;
                    } else {
                        if (comm->dev_comm_created) {
                            auto result = ncclDevCommDestroy(comm->comm,
                                                             &comm->dev_comm);
                            if (result != ncclSuccess)
                                LOG(WARNING)
                                    << "ncclDevCommDestroy failed: "
                                    << ncclGetErrorString(result);
                        }
                        auto result = ncclCommDestroy(comm->comm);
                        if (result != ncclSuccess)
                            LOG(WARNING) << "ncclCommDestroy failed: "
                                         << ncclGetErrorString(result);
                    }
                }
                if (comm->completion_stream) {
                    cudaStreamDestroy(comm->completion_stream);
                    comm->completion_stream = nullptr;
                }
                if (status.ok()) cudaSetDevice(previous_device);
            }
        }
        comms_.clear();
    }

    std::lock_guard<std::mutex> lock(allocation_mutex_);
    if (!nccl_allocations_.empty()) {
        LOG(WARNING) << "NCCL transport uninstalling with "
                     << nccl_allocations_.size()
                     << " tracked ncclMemAlloc buffers still live";
    }
    metadata_.reset();
    local_topology_.reset();
    conf_.reset();
    platform_ = nullptr;
    installed_ = false;
    return Status::OK();
}

Status NcclTransport::allocateSubBatch(SubBatchRef& batch, size_t max_size) {
    auto nccl_batch = Slab<NcclSubBatch>::Get().allocate();
    if (!nccl_batch)
        return Status::InternalError("Unable to allocate NCCL sub-batch");
    batch = nccl_batch;
    nccl_batch->task_list.reserve(max_size);
    nccl_batch->max_size = max_size;
    CHECK_STATUS(platform_->getStreamFromPool(nccl_batch->stream));
    return Status::OK();
}

Status NcclTransport::freeSubBatch(SubBatchRef& batch) {
    auto nccl_batch = dynamic_cast<NcclSubBatch*>(batch);
    if (!nccl_batch)
        return Status::InvalidArgument("Invalid NCCL sub-batch" LOC_MARK);
    for (auto& task : nccl_batch->task_list) {
        auto event = task.completion_event.exchange(
            nullptr, std::memory_order_acq_rel);
        if (event) cudaEventDestroy(event);
    }
    Slab<NcclSubBatch>::Get().deallocate(nccl_batch);
    batch = nullptr;
    return Status::OK();
}

Status NcclTransport::markFailed(NcclTask& task, const std::string& reason) {
    LOG(WARNING) << "NCCL host RMA task failed: " << reason;
    task.transferred_bytes.store(0, std::memory_order_release);
    task.status_word.store(TransferStatusEnum::FAILED,
                           std::memory_order_release);
    return Status::OK();
}

Status NcclTransport::buildPreconnectContext(SegmentID target_id,
                                             TransferContext& ctx) {
    if (target_id == LOCAL_SEGMENT_ID) {
        return Status::InvalidArgument(
            "NCCL preconnect expects a remote target segment" LOC_MARK);
    }

    auto local_segment = metadata_->segmentManager().getLocal();
    if (!local_segment) {
        return Status::InvalidArgument(
            "NCCL preconnect local segment is unavailable" LOC_MARK);
    }
    if (default_cuda_device_ < 0) {
        return Status::InvalidArgument(
            "NCCL preconnect local CUDA device is unavailable" LOC_MARK);
    }
    ctx.local_device = default_cuda_device_;

    CHECK_STATUS(metadata_->segmentManager().withCachedSegment(
        target_id, [&](SegmentDesc* segment) {
            if (!segment) {
                return Status::NeedsRefreshCache(
                    "NCCL preconnect remote segment is unavailable" LOC_MARK);
            }
            CHECK_STATUS(findSingleNcclCudaDevice(*segment, "Remote",
                                                  ctx.remote_device));
            ctx.remote_segment_name = segment->name;
            ctx.remote_rpc_addr = segment->rpc_server_addr;
            return Status::OK();
        }));
    if (ctx.remote_rpc_addr.empty()) {
        return Status::InvalidArgument(
            "NCCL preconnect remote RPC address is empty" LOC_MARK);
    }

    ctx.target_id = target_id;
    ctx.session_key = makeSessionKey(local_segment_name_,
                                     ctx.remote_segment_name,
                                     ctx.local_device, ctx.remote_device);
    return Status::OK();
}

Status NcclTransport::preconnectSegment(SegmentID target_id) {
    TransferContext ctx;
    CHECK_STATUS(buildPreconnectContext(target_id, ctx));

    LOG(INFO) << "NCCL eager preconnect begin: local=" << local_segment_name_
              << ":cuda" << ctx.local_device << " remote="
              << ctx.remote_segment_name << ":cuda" << ctx.remote_device
              << " session=" << ctx.session_key;
    std::shared_ptr<CommState> comm_state;
    Status status = ensureComm(ctx, comm_state);
    int local_rank = -1;
    int peer_rank = -1;
    if (comm_state) {
        std::lock_guard<std::mutex> lock(comm_state->mu);
        local_rank = comm_state->local_rank;
        peer_rank = comm_state->peer_rank;
    }
    LOG(INFO) << "NCCL eager preconnect end: local=" << local_segment_name_
              << " remote=" << ctx.remote_segment_name
              << " session=" << ctx.session_key
              << " local_rank=" << local_rank << " peer_rank=" << peer_rank
              << " status=" << status.ToString();
    return status;
}

Status NcclTransport::configurePoolWindow(TransferContext& ctx) {
    if (ctx.source_pool_base == 0 || ctx.target_pool_base == 0 ||
        ctx.source_pool_length == 0 || ctx.target_pool_length == 0) {
        return Status::InvalidArgument(
            "NCCL paged pool window requires nonempty source and target pools"
            LOC_MARK);
    }

    constexpr uint64_t kWindowAlignment = NCCL_WIN_REQUIRED_ALIGNMENT;
    static_assert(kWindowAlignment > 0);
    const uint64_t source_delta = ctx.source_pool_base % kWindowAlignment;
    const uint64_t target_delta = ctx.target_pool_base % kWindowAlignment;
    if (ctx.source_pool_length >
            std::numeric_limits<uint64_t>::max() - source_delta ||
        ctx.target_pool_length >
            std::numeric_limits<uint64_t>::max() - target_delta) {
        return Status::InvalidArgument(
            "NCCL paged pool window extent overflows" LOC_MARK);
    }

    const uint64_t source_extent = source_delta + ctx.source_pool_length;
    const uint64_t target_extent = target_delta + ctx.target_pool_length;
    const uint64_t paired_span = std::min(source_extent, target_extent);
    if (paired_span <= source_delta || paired_span <= target_delta) {
        return Status::InvalidArgument(
            "NCCL paged pools have no common transfer-visible prefix"
            LOC_MARK);
    }

    ctx.source_base = ctx.source_pool_base - source_delta;
    ctx.target_base = ctx.target_pool_base - target_delta;
    ctx.source_length = paired_span;
    ctx.target_length = paired_span;
    ctx.source_offset = source_delta;
    ctx.target_offset = target_delta;
    ctx.source_visible_length = paired_span - source_delta;
    ctx.target_visible_length = paired_span - target_delta;
    ctx.use_paired_window_buffers = true;
    ctx.window_key = makePoolWindowKey(
        ctx.session_key, local_segment_name_, ctx.local_device,
        ctx.source_pool_base, ctx.source_pool_length, ctx.remote_segment_name,
        ctx.remote_device, ctx.target_pool_base, ctx.target_pool_length);
    ctx.source_window_key = ctx.window_key;

    if (pagedGinDiagEnabled()) {
        LOG(INFO) << "NCCL paged persistent pool window: session="
                  << ctx.session_key << " source_pool=0x" << std::hex
                  << ctx.source_pool_base << " source_base=0x"
                  << ctx.source_base << std::dec
                  << " source_delta=" << source_delta
                  << " source_pool_length=" << ctx.source_pool_length
                  << " source_visible_length=" << ctx.source_visible_length
                  << " target_pool=0x" << std::hex << ctx.target_pool_base
                  << " target_base=0x" << ctx.target_base << std::dec
                  << " target_delta=" << target_delta
                  << " target_pool_length=" << ctx.target_pool_length
                  << " target_visible_length=" << ctx.target_visible_length
                  << " paired_span=" << paired_span
                  << " key=" << ctx.window_key;
    }
    return Status::OK();
}

Status NcclTransport::preconnectPagedSegment(
    SegmentID target_id, void* local_pool_addr, size_t local_pool_length,
    uint64_t remote_pool_addr, size_t remote_pool_length) {
    TransferContext ctx;
    CHECK_STATUS(buildPreconnectContext(target_id, ctx));
    ctx.source_pool_base = reinterpret_cast<uint64_t>(local_pool_addr);
    ctx.source_pool_length = local_pool_length;
    ctx.target_pool_base = remote_pool_addr;
    ctx.target_pool_length = remote_pool_length;

    auto local_segment = metadata_->segmentManager().getLocal();
    if (!local_segment ||
        !local_segment->findBuffer(ctx.source_pool_base,
                                   ctx.source_pool_length)) {
        return Status::InvalidArgument(
            "NCCL local paged pool is not registered" LOC_MARK);
    }
    CHECK_STATUS(metadata_->segmentManager().withCachedSegment(
        target_id, [&](SegmentDesc* segment) {
            if (!segment ||
                !segment->findBuffer(ctx.target_pool_base,
                                     ctx.target_pool_length)) {
                return Status::NeedsRefreshCache(
                    "NCCL remote paged pool is not registered" LOC_MARK);
            }
            return Status::OK();
        }));
    CHECK_STATUS(configurePoolWindow(ctx));

    LOG(INFO) << "NCCL eager paged-pool preconnect begin: local="
              << local_segment_name_ << ":cuda" << ctx.local_device
              << " remote=" << ctx.remote_segment_name << ":cuda"
              << ctx.remote_device << " session=" << ctx.session_key
              << " window=" << ctx.window_key;
    std::shared_ptr<CommState> comm_state;
    CHECK_STATUS(ensureComm(ctx, comm_state));
    std::shared_ptr<WindowState> window_state;
    Status status = ensureWindow(ctx, comm_state, window_state);
    LOG(INFO) << "NCCL eager paged-pool preconnect end: local="
              << local_segment_name_ << " remote=" << ctx.remote_segment_name
              << " session=" << ctx.session_key
              << " window=" << ctx.window_key
              << " status=" << status.ToString();
    return status;
}

Status NcclTransport::buildTransferContext(const Request& request,
                                           TransferContext& ctx) {
    return buildTransferContext(request, request.length, request.length, ctx);
}

Status NcclTransport::buildTransferContext(const Request& request,
                                           size_t source_length,
                                           size_t target_length,
                                           TransferContext& ctx) {
    if (request.target_id == LOCAL_SEGMENT_ID) {
        return Status::InvalidArgument(
            "NCCL host RMA expects a remote target segment" LOC_MARK);
    }
    if (Platform::getLoader().getMemoryType(request.source) != MTYPE_CUDA) {
        return Status::InvalidArgument(
            "NCCL host RMA source must be CUDA memory" LOC_MARK);
    }

    auto local_locations = Platform::getLoader().getLocation(request.source, 1);
    if (local_locations.empty()) {
        return Status::InvalidArgument(
            "Unable to resolve local CUDA source location" LOC_MARK);
    }
    LocationParser local_location(local_locations[0].location);
    if (local_location.type() != "cuda" || local_location.index() < 0) {
        return Status::InvalidArgument(
            "Unable to resolve local CUDA device" LOC_MARK);
    }

    BufferDesc target_buffer;
    Status status = metadata_->segmentManager().withCachedSegment(
        request.target_id, [&](SegmentDesc* segment) {
            if (segment->type != SegmentType::Memory) {
                return Status::NeedsRefreshCache(
                    "NCCL target segment is not memory" LOC_MARK);
            }
            auto* buffer = segment->findBuffer(request.target_offset,
                                              target_length);
            if (!buffer) {
                return Status::NeedsRefreshCache(
                    "Requested address is not in registered buffer" LOC_MARK);
            }
            target_buffer = *buffer;
            ctx.remote_segment_name = segment->name;
            ctx.remote_rpc_addr = segment->rpc_server_addr;
            return Status::OK();
        });
    if (!status.ok()) return status;

    LocationParser remote_location(target_buffer.location);
    if (remote_location.type() != "cuda" || remote_location.index() < 0) {
        return Status::InvalidArgument(
            "NCCL host RMA target must be CUDA memory" LOC_MARK);
    }

    auto local_segment = metadata_->segmentManager().getLocal();
    if (!local_segment || local_segment->type != SegmentType::Memory) {
        return Status::InvalidArgument(
            "NCCL source segment is not memory" LOC_MARK);
    }
    const auto source_addr = reinterpret_cast<uint64_t>(request.source);
    auto* source_buffer = local_segment->findBuffer(source_addr, source_length);
    if (!source_buffer) {
        return Status::InvalidArgument(
            "NCCL source address is not in registered buffer" LOC_MARK);
    }

    ctx.target_id = request.target_id;
    ctx.target_pool_base = target_buffer.addr;
    ctx.target_pool_length = target_buffer.length;
    ctx.source_pool_base = source_buffer->addr;
    ctx.source_pool_length = source_buffer->length;
    ctx.target_base = target_buffer.addr;
    ctx.target_length = target_buffer.length;
    ctx.target_offset = request.target_offset - target_buffer.addr;
    ctx.source_base = source_buffer->addr;
    ctx.source_length = source_buffer->length;
    ctx.source_offset = source_addr - source_buffer->addr;
    ctx.local_device = local_location.index();
    ctx.remote_device = remote_location.index();
    ctx.session_key = makeSessionKey(local_segment_name_,
                                     ctx.remote_segment_name,
                                     ctx.local_device, ctx.remote_device);
    ctx.window_key = makeWindowKey(ctx.session_key, "target", ctx.target_base,
                                   ctx.target_length);
    ctx.source_window_key = makeWindowKey(ctx.session_key, "source",
                                          ctx.source_base, ctx.source_length);
    return Status::OK();
}

void NcclTransport::startBackground(std::function<void()> fn) {
    std::lock_guard<std::mutex> lock(background_mutex_);
    background_threads_.emplace_back(std::move(fn));
}

Status NcclTransport::waitForComm(const std::string& session_key,
                                  std::shared_ptr<CommState>& state) {
    {
        std::lock_guard<std::mutex> lock(comm_mutex_);
        auto it = comms_.find(session_key);
        if (it == comms_.end()) {
            return Status::InvalidArgument(
                "NCCL communicator session not found" LOC_MARK);
        }
        state = it->second;
    }

    std::unique_lock<std::mutex> lock(state->mu);
    state->cv.wait(lock, [&] { return state->ready || !state->status.ok(); });
    return state->status;
}

Status NcclTransport::ensureComm(const TransferContext& ctx,
                                 std::shared_ptr<CommState>& state) {
    bool should_init = false;
    {
        std::lock_guard<std::mutex> lock(comm_mutex_);
        auto& entry = comms_[ctx.session_key];
        if (!entry) entry = std::make_shared<CommState>();
        state = entry;
        std::lock_guard<std::mutex> state_lock(state->mu);
        if (!state->ready && !state->initializing) {
            state->initializing = true;
            state->device_index = ctx.local_device;
            state->local_rank = 0;
            state->peer_rank = 1;
            should_init = true;
        }
    }

    if (should_init) {
        const size_t lanes = params_.gin_lanes;
        ncclUniqueId unique_id{};
        Status status = ncclStatus(ncclGetUniqueId(&unique_id),
                                   "ncclGetUniqueId");
        if (status.ok()) {
            NcclBootstrapDesc request;
            request.session_key = ctx.session_key;
            request.unique_id = serializeUniqueId(unique_id);
            // For the single-communicator device-GIN path this field carries
            // the requested number of GIN contexts/stripe lanes.
            request.comm_count = static_cast<int>(lanes);
            request.device_index = ctx.remote_device;
            NcclBootstrapDesc response;
            if (pagedGinDiagEnabled()) {
                LOG(INFO) << "NCCL comm bootstrap request: session="
                          << ctx.session_key << " remote="
                          << ctx.remote_rpc_addr << " device="
                          << ctx.remote_device << " uid=0x" << std::hex
                          << uniqueIdFingerprint(unique_id) << std::dec;
            }
            status = ControlClient::bootstrapNccl(ctx.remote_rpc_addr,
                                                  request, response);
            if (pagedGinDiagEnabled()) {
                LOG(INFO) << "NCCL comm bootstrap response: session="
                          << ctx.session_key << " status=" << status.ToString();
            }
        }
        if (status.ok()) {
            int previous_device = 0;
            status = setCudaDevice(ctx.local_device, previous_device);
            if (status.ok()) {
                state->lanes = lanes;
                if (pagedGinDiagEnabled()) {
                    LOG(INFO) << "NCCL comm local init begin: session="
                              << ctx.session_key << " rank=0 device="
                              << ctx.local_device;
                }
                status = initCommBlocking(&state->comm, unique_id, 0,
                                          "ncclCommInitRankConfig(local)");
                if (pagedGinDiagEnabled()) {
                    LOG(INFO) << "NCCL comm local init end: session="
                              << ctx.session_key << " status=" << status.ToString();
                }
                if (status.ok()) {
                    // Rank 0 may return from communicator initialization ahead
                    // of rank 1. Keep device-communicator setup collective by
                    // releasing the remote side only after both ranks finish.
                    NcclBootstrapDesc ready_request;
                    ready_request.session_key = ctx.session_key;
                    ready_request.phase = 1;
                    NcclBootstrapDesc ready_response;
                    if (pagedGinDiagEnabled()) {
                        LOG(INFO) << "NCCL comm remote-ready request: session="
                                  << ctx.session_key;
                    }
                    status = ControlClient::bootstrapNccl(ctx.remote_rpc_addr,
                                                          ready_request,
                                                          ready_response);
                    if (pagedGinDiagEnabled()) {
                        LOG(INFO) << "NCCL comm remote-ready response: session="
                                  << ctx.session_key << " status="
                                  << status.ToString();
                    }
                }
                if (status.ok()) {
                    state->peer_in_lsa = peerInLsaTeam(state->comm,
                                                       state->peer_rank);
                }
                if (status.ok() && (!state->peer_in_lsa || params_.force_gin)) {
                    ncclDevCommRequirements_t reqs =
                        NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
                    reqs.ginForceEnable = true;
                    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
                    reqs.ginContextCount = static_cast<int>(lanes);
                    // Data completion uses [0, lanes); wait-ack uses
                    // [lanes, 2 * lanes).
                    reqs.ginSignalCount = static_cast<int>(lanes * 2);
                    status = ncclStatus(ncclGroupStart(),
                                        "ncclGroupStart(dev comm)");
                    if (status.ok()) {
                        Status create_status = ncclStatus(
                            ncclDevCommCreate(state->comm, &reqs,
                                              &state->dev_comm),
                            "ncclDevCommCreate");
                        Status group_status = ncclStatus(ncclGroupEnd(),
                                                        "ncclGroupEnd(dev comm)");
                        status = create_status.ok() ? group_status : create_status;
                    }
                    state->dev_comm_created = status.ok();
                }
                if (status.ok() && (!state->peer_in_lsa || params_.force_gin) &&
                    static_cast<size_t>(state->dev_comm.ginContextCount) <
                        lanes) {
                    status = Status::InternalError(
                        "NCCL dev comm did not provide requested GIN contexts" LOC_MARK);
                }
                if (status.ok()) {
                    status = cudaStatus(
                        cudaStreamCreateWithFlags(&state->completion_stream,
                                                  cudaStreamNonBlocking),
                        "cudaStreamCreateWithFlags(completion)");
                }
                if (status.ok()) {
                    if (state->peer_in_lsa) {
                        LOG(INFO) << "NCCL LSA communicator ready: lanes="
                                  << lanes << " peer_in_lsa=1";
                    } else {
                        LOG(INFO) << "NCCL GIN single communicator ready: lanes="
                                  << lanes << " gin_connections="
                                  << static_cast<int>(
                                         state->dev_comm.ginConnectionCount)
                                  << " gin_contexts="
                                  << state->dev_comm.ginContextCount
                                  << " peer_in_lsa=0";
                        if (static_cast<size_t>(
                                state->dev_comm.ginConnectionCount) < lanes) {
                            LOG(WARNING)
                                << "NCCL GIN has fewer connections than contexts; "
                                   "set NCCL_GIN_NCONNECTIONS="
                                << lanes << " for peak striped bandwidth";
                        }
                    }
                }
                cudaSetDevice(previous_device);
            }
        }

        {
            std::lock_guard<std::mutex> lock(state->mu);
            state->status = status;
            state->ready = status.ok();
            state->initializing = false;
        }
        state->cv.notify_all();
    }

    Status status = waitForComm(ctx.session_key, state);
    if (status.ok() && !should_init && pagedGinDiagEnabled()) {
        std::lock_guard<std::mutex> lock(state->mu);
        LOG(INFO) << "NCCL communicator reuse: session=" << ctx.session_key
                  << " local_rank=" << state->local_rank
                  << " peer_rank=" << state->peer_rank
                  << " device=" << state->device_index;
    }
    return status;
}

Status NcclTransport::ensureWindow(
    const TransferContext& ctx, const std::shared_ptr<CommState>& comm_state,
    std::shared_ptr<WindowState>& state) {
    bool should_init = false;
    {
        std::lock_guard<std::mutex> lock(window_mutex_);
        auto& entry = windows_[ctx.window_key];
        if (!entry) entry = std::make_shared<WindowState>();
        state = entry;
        std::lock_guard<std::mutex> state_lock(state->mu);
        if (!state->ready && !state->initializing) {
            state->initializing = true;
            state->length = ctx.target_length;
            state->device_index = ctx.local_device;
            state->session_key = ctx.session_key;
            should_init = true;
        }
    }

    if (should_init) {
        std::lock_guard<std::mutex> collective_lock(comm_state->collective_mu);
        Status status;
        NcclWindowDesc request;
        request.session_key = ctx.session_key;
        request.window_key = ctx.window_key;
        request.addr = ctx.target_base;
        request.length = ctx.target_length;
        request.device_index = ctx.remote_device;
        request.win_flags = NCCL_WIN_COLL_SYMMETRIC;
        request.allocate_local = false;
        NcclWindowDesc response;
        status = ControlClient::registerNcclWindow(ctx.remote_rpc_addr,
                                                   request, response);

        int previous_device = 0;
        bool device_changed = false;
        if (status.ok()) {
            status = setCudaDevice(ctx.local_device, previous_device);
            device_changed = status.ok();
        }
        if (status.ok() && ctx.use_paired_window_buffers) {
            state->local_buffer = reinterpret_cast<void*>(ctx.source_base);
        } else if (status.ok()) {
            status = ncclStatus(ncclMemAlloc(&state->local_buffer,
                                             ctx.target_length),
                                "ncclMemAlloc(window dummy)");
            state->owns_local_buffer = status.ok();
        }
        if (status.ok()) {
            status = ncclStatus(
                ncclCommWindowRegister(comm_state->comm, state->local_buffer,
                                       ctx.target_length, &state->window,
                                       NCCL_WIN_COLL_SYMMETRIC),
                "ncclCommWindowRegister");
        }
        if (device_changed) cudaSetDevice(previous_device);

        {
            std::lock_guard<std::mutex> lock(state->mu);
            state->status = status;
            state->ready = status.ok();
            state->initializing = false;
        }
        state->cv.notify_all();
    }

    std::unique_lock<std::mutex> lock(state->mu);
    state->cv.wait(lock, [&] { return state->ready || !state->status.ok(); });
    return state->status;
}

Status NcclTransport::ensureSourceWindow(
    const TransferContext& ctx, const std::shared_ptr<CommState>& comm_state,
    std::shared_ptr<WindowState>& state) {
    bool should_init = false;
    {
        std::lock_guard<std::mutex> lock(window_mutex_);
        auto& entry = windows_[ctx.source_window_key];
        if (!entry) entry = std::make_shared<WindowState>();
        state = entry;
        std::lock_guard<std::mutex> state_lock(state->mu);
        if (!state->ready && !state->initializing) {
            state->initializing = true;
            state->length = ctx.source_length;
            state->device_index = ctx.local_device;
            state->session_key = ctx.session_key;
            should_init = true;
        }
    }

    if (should_init) {
        std::lock_guard<std::mutex> collective_lock(comm_state->collective_mu);
        Status status;
        NcclWindowDesc request;
        request.session_key = ctx.session_key;
        request.window_key = ctx.source_window_key;
        request.addr = ctx.use_paired_window_buffers ? ctx.target_base : 0;
        request.length = ctx.source_length;
        request.device_index = ctx.remote_device;
        request.win_flags = NCCL_WIN_COLL_SYMMETRIC;
        request.allocate_local = !ctx.use_paired_window_buffers;
        NcclWindowDesc response;
        status = ControlClient::registerNcclWindow(ctx.remote_rpc_addr,
                                                   request, response);

        int previous_device = 0;
        bool device_changed = false;
        if (status.ok()) {
            status = setCudaDevice(ctx.local_device, previous_device);
            device_changed = status.ok();
        }
        if (status.ok()) {
            status = ncclStatus(
                ncclCommWindowRegister(
                    comm_state->comm, reinterpret_cast<void*>(ctx.source_base),
                    ctx.source_length, &state->window,
                    NCCL_WIN_COLL_SYMMETRIC),
                "ncclCommWindowRegister(source)");
        }
        if (device_changed) cudaSetDevice(previous_device);

        {
            std::lock_guard<std::mutex> lock(state->mu);
            state->status = status;
            state->ready = status.ok();
            state->initializing = false;
        }
        state->cv.notify_all();
    }

    std::unique_lock<std::mutex> lock(state->mu);
    state->cv.wait(lock, [&] { return state->ready || !state->status.ok(); });
    return state->status;
}

Status NcclTransport::postRemoteWaitSignal(
    const TransferContext& ctx, const std::shared_ptr<CommState>& comm_state,
    uint64_t signal_value) {
    NcclSignalDesc request;
    request.session_key = ctx.session_key;
    request.peer = comm_state->local_rank;
    request.op_count = static_cast<int>(signal_value);
    request.signal_value = signal_value;
    request.signal_index = 0;
    request.context = 0;
    request.device_index = ctx.remote_device;
    NcclSignalDesc response;
    return ControlClient::waitNcclSignal(ctx.remote_rpc_addr, request,
                                         response);
}

Status NcclTransport::onBootstrapNccl(const NcclBootstrapDesc& request,
                                      NcclBootstrapDesc& response) {
    response.session_key = request.session_key;
    response.unique_id = request.unique_id;
    response.unique_ids = request.unique_ids;
    response.comm_count = request.comm_count;
    response.device_index = request.device_index;
    response.phase = request.phase;

    if (request.phase == 1) {
        std::shared_ptr<CommState> state;
        {
            std::lock_guard<std::mutex> lock(comm_mutex_);
            auto it = comms_.find(request.session_key);
            if (it == comms_.end()) {
                return Status::InvalidArgument(
                    "NCCL communicator session not found" LOC_MARK);
            }
            state = it->second;
        }
        std::unique_lock<std::mutex> lock(state->mu);
        state->cv.wait(lock, [&] {
            return state->comm_init_ready || !state->initializing;
        });
        if (!state->comm_init_ready) {
            return state->status.ok()
                       ? Status::InternalError(
                             "Remote NCCL communicator initialization failed" LOC_MARK)
                       : state->status;
        }
        state->device_init_requested = true;
        lock.unlock();
        state->cv.notify_all();
        if (pagedGinDiagEnabled()) {
            LOG(INFO) << "NCCL comm remote device-init released: session="
                      << request.session_key;
        }
        return Status::OK();
    }

    bool should_init = false;
    std::shared_ptr<CommState> state;
    {
        std::lock_guard<std::mutex> lock(comm_mutex_);
        auto& entry = comms_[request.session_key];
        if (!entry) entry = std::make_shared<CommState>();
        state = entry;
        std::lock_guard<std::mutex> state_lock(state->mu);
        if (state->ready || state->initializing) return Status::OK();
        state->initializing = true;
        state->device_index = request.device_index;
        state->local_rank = 1;
        state->peer_rank = 0;
        should_init = true;
    }

    if (!should_init) return Status::OK();

    auto prepared = std::make_shared<NcclBootstrapPrepared>();
    startBackground([this, state, request, prepared]() {
        Status status;
        const size_t lanes = request.comm_count > 0
                                 ? static_cast<size_t>(request.comm_count)
                                 : size_t{1};
        ncclUniqueId unique_id{};
        const std::string& raw = request.unique_ids.empty()
                                     ? request.unique_id
                                     : request.unique_ids[0];
        status = deserializeUniqueId(raw, unique_id);
        int previous_device = 0;
        bool device_changed = false;
        if (status.ok()) {
            status = setCudaDevice(request.device_index, previous_device);
            device_changed = status.ok();
        }
        {
            std::lock_guard<std::mutex> lock(prepared->mu);
            prepared->status = status;
            prepared->ready = true;
        }
        prepared->cv.notify_all();
        if (status.ok()) {
            state->lanes = lanes;
            if (pagedGinDiagEnabled()) {
                LOG(INFO) << "NCCL comm remote init begin: session="
                          << request.session_key << " rank=1 device="
                          << request.device_index << " uid=0x" << std::hex
                          << uniqueIdFingerprint(unique_id) << std::dec;
            }
            status = initCommBlocking(&state->comm, unique_id, 1,
                                      "ncclCommInitRankConfig(remote)");
            if (pagedGinDiagEnabled()) {
                LOG(INFO) << "NCCL comm remote init end: session="
                          << request.session_key << " status=" << status.ToString();
            }
            {
                std::lock_guard<std::mutex> lock(state->mu);
                state->status = status;
                state->comm_init_ready = status.ok();
            }
            state->cv.notify_all();
            if (status.ok()) {
                std::unique_lock<std::mutex> lock(state->mu);
                state->cv.wait(lock, [&] {
                    return state->device_init_requested || !state->status.ok();
                });
                status = state->status;
            }
            if (status.ok()) {
                state->peer_in_lsa = peerInLsaTeam(state->comm,
                                                   state->peer_rank);
            }
            if (status.ok() && (!state->peer_in_lsa || params_.force_gin)) {
                ncclDevCommRequirements_t reqs =
                    NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
                reqs.ginForceEnable = true;
                reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
                reqs.ginContextCount = static_cast<int>(lanes);
                // Data completion uses [0, lanes); wait-ack uses
                // [lanes, 2 * lanes).
                reqs.ginSignalCount = static_cast<int>(lanes * 2);
                status = ncclStatus(ncclGroupStart(),
                                    "ncclGroupStart(remote dev comm)");
                if (status.ok()) {
                    Status create_status = ncclStatus(
                        ncclDevCommCreate(state->comm, &reqs,
                                          &state->dev_comm),
                        "ncclDevCommCreate(remote)");
                    Status group_status = ncclStatus(ncclGroupEnd(),
                                                    "ncclGroupEnd(remote dev comm)");
                    status = create_status.ok() ? group_status : create_status;
                }
                state->dev_comm_created = status.ok();
            }
            if (status.ok() && (!state->peer_in_lsa || params_.force_gin) &&
                static_cast<size_t>(state->dev_comm.ginContextCount) < lanes) {
                status = Status::InternalError(
                    "NCCL remote dev comm did not provide requested GIN contexts" LOC_MARK);
            }
            if (status.ok()) {
                status = cudaStatus(
                    cudaStreamCreateWithFlags(&state->completion_stream,
                                              cudaStreamNonBlocking),
                    "cudaStreamCreateWithFlags(remote completion)");
            }
            if (status.ok()) {
                if (state->peer_in_lsa) {
                    LOG(INFO) << "NCCL LSA remote communicator ready: lanes="
                              << lanes << " peer_in_lsa=1";
                } else {
                    LOG(INFO) << "NCCL GIN remote single communicator ready: lanes="
                              << lanes << " gin_connections="
                              << static_cast<int>(
                                     state->dev_comm.ginConnectionCount)
                              << " gin_contexts="
                              << state->dev_comm.ginContextCount
                              << " peer_in_lsa=0";
                    if (static_cast<size_t>(
                            state->dev_comm.ginConnectionCount) < lanes) {
                        LOG(WARNING)
                            << "NCCL GIN has fewer connections than contexts; "
                               "set NCCL_GIN_NCONNECTIONS="
                            << lanes << " for peak striped bandwidth";
                    }
                }
            }
        }
        if (device_changed) cudaSetDevice(previous_device);
        {
            std::lock_guard<std::mutex> lock(state->mu);
            state->status = status;
            state->ready = status.ok();
            state->initializing = false;
        }
        state->cv.notify_all();
    });
    std::unique_lock<std::mutex> lock(prepared->mu);
    prepared->cv.wait(lock, [&] { return prepared->ready; });
    return prepared->status;
}

Status NcclTransport::onRegisterNcclWindow(const NcclWindowDesc& request,
                                           NcclWindowDesc& response) {
    response.session_key = request.session_key;
    response.window_key = request.window_key;
    response.addr = request.addr;
    response.length = request.length;
    response.device_index = request.device_index;
    response.win_flags = request.win_flags;
    response.allocate_local = request.allocate_local;

    bool should_init = false;
    std::shared_ptr<WindowState> state;
    {
        std::lock_guard<std::mutex> lock(window_mutex_);
        auto& entry = windows_[request.window_key];
        if (!entry) entry = std::make_shared<WindowState>();
        state = entry;
        std::lock_guard<std::mutex> state_lock(state->mu);
        if (state->ready || state->initializing) return Status::OK();
        state->initializing = true;
        state->length = request.length;
        state->device_index = request.device_index;
        state->session_key = request.session_key;
        should_init = true;
    }

    if (!should_init) return Status::OK();

    startBackground([this, state, request]() {
        std::shared_ptr<CommState> comm_state;
        Status status = waitForComm(request.session_key, comm_state);
        std::unique_lock<std::mutex> collective_lock;
        if (status.ok()) {
            collective_lock = std::unique_lock<std::mutex>(
                comm_state->collective_mu);
        }
        int previous_device = 0;
        bool device_changed = false;
        if (status.ok()) {
            status = setCudaDevice(request.device_index, previous_device);
            device_changed = status.ok();
        }
        if (status.ok() && request.allocate_local) {
            status = ncclStatus(ncclMemAlloc(&state->local_buffer,
                                             request.length),
                                "ncclMemAlloc(remote window dummy)");
            state->owns_local_buffer = status.ok();
        }
        if (status.ok()) {
            void* buffer = request.allocate_local
                               ? state->local_buffer
                               : reinterpret_cast<void*>(request.addr);
            status = ncclStatus(
                ncclCommWindowRegister(comm_state->comm, buffer,
                                       request.length, &state->window,
                                       request.win_flags),
                "ncclCommWindowRegister(remote)");
        }
        if (device_changed) cudaSetDevice(previous_device);
        {
            std::lock_guard<std::mutex> lock(state->mu);
            state->status = status;
            state->ready = status.ok();
            state->initializing = false;
        }
        state->cv.notify_all();
    });
    return Status::OK();
}

Status NcclTransport::onWaitNcclSignal(const NcclSignalDesc& request,
                                       NcclSignalDesc& response) {
    response.session_key = request.session_key;
    response.peer = request.peer;
    response.op_count = request.op_count;
    response.signal_index = request.signal_index;
    response.context = request.context;
    response.device_index = request.device_index;

    startBackground([this, request]() {
        std::shared_ptr<CommState> comm_state;
        Status status = waitForComm(request.session_key, comm_state);
        std::shared_ptr<WindowState> window_state;
        std::shared_ptr<WindowState> source_window_state;
        if (status.ok() && request.put_signal) {
            {
                std::lock_guard<std::mutex> lock(window_mutex_);
                auto dst_it = windows_.find(request.window_key);
                auto src_it = windows_.find(request.source_window_key);
                if (dst_it == windows_.end() || src_it == windows_.end()) {
                    status = Status::InvalidArgument(
                        "NCCL device put window not found" LOC_MARK);
                } else {
                    window_state = dst_it->second;
                    source_window_state = src_it->second;
                }
            }
            if (status.ok()) {
                std::unique_lock<std::mutex> lock(window_state->mu);
                window_state->cv.wait(lock, [&] {
                    return window_state->ready || !window_state->status.ok();
                });
                status = window_state->status;
            }
            if (status.ok()) {
                std::unique_lock<std::mutex> lock(source_window_state->mu);
                source_window_state->cv.wait(lock, [&] {
                    return source_window_state->ready ||
                           !source_window_state->status.ok();
                });
                status = source_window_state->status;
            }
        }
        int previous_device = 0;
        bool device_changed = false;
        if (status.ok()) {
            status = setCudaDevice(request.device_index, previous_device);
            device_changed = status.ok();
        }
        const uint64_t signal_value =
            request.signal_value ? request.signal_value : request.op_count;
        if (pagedGinDiagEnabled())
        {
            LOG(INFO) << "NCCL remote signal op: session=" << request.session_key
                      << " signal=" << signal_value << " put_signal="
                      << request.put_signal << " device=" << request.device_index;
        }
        std::unique_lock<std::mutex> signal_order_lock;
        bool ordered_signal_turn = false;
        if (status.ok() && !request.put_signal) {
            signal_order_lock =
                std::unique_lock<std::mutex>(comm_state->remote_signal_mu);
            comm_state->remote_signal_cv.wait(signal_order_lock, [&] {
                return signal_value == comm_state->next_remote_signal ||
                       shutting_down_.load(std::memory_order_acquire);
            });
            if (shutting_down_.load(std::memory_order_acquire)) {
                status = Status::InvalidArgument(
                    "NCCL transport is shutting down" LOC_MARK);
            } else {
                ordered_signal_turn = true;
            }
        }
        if (status.ok() && request.put_signal) {
            auto err = tentNcclGinLaunchPut(
                comm_state->dev_comm, request.peer,
                static_cast<int>(comm_state->lanes), window_state->window,
                request.peer_window_offset, source_window_state->window,
                request.source_addr, request.length,
                static_cast<unsigned long long>(signal_value),
                comm_state->completion_stream);
            status = cudaStatus(err, "tentNcclGinLaunchPut(remote)");
        } else if (status.ok()) {
            auto err = tentNcclGinLaunchWaitAck(
                comm_state->dev_comm, request.peer,
                static_cast<int>(comm_state->lanes),
                static_cast<unsigned long long>(signal_value),
                comm_state->completion_stream);
            status = cudaStatus(err, "tentNcclGinLaunchWaitAck(remote)");
            if (pagedGinDiagEnabled())
            {
                LOG(INFO) << "NCCL remote signal op: wait-ack launch done: "
                          << status.ToString();
            }
        }
        if (ordered_signal_turn) {
            ++comm_state->next_remote_signal;
            signal_order_lock.unlock();
            comm_state->remote_signal_cv.notify_all();
        }
        if (device_changed) cudaSetDevice(previous_device);
        if (!status.ok()) {
            LOG(WARNING) << "NCCL remote signal operation failed: "
                         << status.ToString();
        }
    });
    return Status::OK();
}

void NcclTransport::startTransfer(NcclTask* task, NcclSubBatch* batch) {
    if (!task || !batch) return;
    if (shutting_down_.load(std::memory_order_acquire)) {
        markFailed(*task, "NCCL transport shutting down");
        return;
    }

    TransferContext ctx;
    auto status = buildTransferContext(task->request, ctx);
    const bool is_read = task->request.opcode == Request::READ;
    std::shared_ptr<CommState> comm_state;
    if (status.ok()) status = ensureComm(ctx, comm_state);
    std::shared_ptr<WindowState> window_state;
    if (status.ok()) status = ensureWindow(ctx, comm_state, window_state);
    const bool use_lsa =
        status.ok() && comm_state->peer_in_lsa && !params_.force_gin;
    std::shared_ptr<WindowState> source_window_state;
    if (status.ok() && !use_lsa) {
        status = ensureSourceWindow(ctx, comm_state, source_window_state);
    }
    const bool wait_ack =
        params_.wait_ack && status.ok() && !use_lsa && !is_read;
    const uint64_t signal_value =
        wait_ack ? comm_state->signal_epoch.fetch_add(
                       1, std::memory_order_acq_rel) +
                       1
                 : 0;
    if (status.ok() && wait_ack) {
        status = postRemoteWaitSignal(ctx, comm_state, signal_value);
    }
    if (status.ok()) {
        int previous_device = 0;
        status = setCudaDevice(ctx.local_device, previous_device);
        bool device_changed = status.ok();
        cudaEvent_t event = nullptr;
        if (status.ok()) {
            const int lanes = static_cast<int>(comm_state->lanes);
            if (use_lsa) {
                void* peer_ptr = nullptr;
                status = getLsaPeerPointer(window_state->window,
                                           ctx.target_offset,
                                           comm_state->peer_rank,
                                           &peer_ptr);
                if (status.ok()) {
                    void* local_ptr = reinterpret_cast<void*>(ctx.source_base);
                    auto err = cudaMemcpyAsync(
                        is_read ? local_ptr : peer_ptr,
                        is_read ? peer_ptr : local_ptr, task->request.length,
                        cudaMemcpyDeviceToDevice,
                        comm_state->completion_stream);
                    status = cudaStatus(err, "cudaMemcpyAsync(NCCL LSA)");
                }
            } else if (is_read) {
                auto err = tentNcclGinLaunchGet(
                    comm_state->dev_comm, comm_state->peer_rank, lanes,
                    window_state->window, ctx.target_offset,
                    source_window_state->window, 0, task->request.length,
                    comm_state->completion_stream);
                status = cudaStatus(err, "tentNcclGinLaunchGet");
            } else {
                auto err = tentNcclGinLaunchPut(
                    comm_state->dev_comm, comm_state->peer_rank, lanes,
                    window_state->window, ctx.target_offset,
                    source_window_state->window, 0, task->request.length,
                    static_cast<unsigned long long>(signal_value),
                    comm_state->completion_stream);
                status = cudaStatus(err, "tentNcclGinLaunchPut");
                if (status.ok() && wait_ack) {
                    err = tentNcclGinLaunchWaitSignal(
                        comm_state->dev_comm, lanes, lanes,
                        static_cast<unsigned long long>(signal_value),
                        comm_state->completion_stream);
                    status = cudaStatus(err, "tentNcclGinLaunchWaitSignal");
                }
            }
            if (status.ok()) {
                auto err = cudaEventCreateWithFlags(&event,
                                                    cudaEventDisableTiming);
                if (err != cudaSuccess) {
                    status = Status::InternalError(
                        std::string("cudaEventCreateWithFlags: ") +
                        cudaGetErrorString(err) + LOC_MARK);
                }
            }
            if (status.ok()) {
                auto err = cudaEventRecord(event,
                                           comm_state->completion_stream);
                if (err != cudaSuccess) {
                    status = Status::InternalError(
                        std::string("cudaEventRecord: ") +
                        cudaGetErrorString(err) + LOC_MARK);
                }
            }
        }
        if (device_changed) cudaSetDevice(previous_device);
        if (status.ok()) {
            task->completion_event.store(event, std::memory_order_release);
            event = nullptr;
        }
        if (event) cudaEventDestroy(event);
    }

    if (!status.ok()) markFailed(*task, status.ToString());
}

Status NcclTransport::transferPagedSync(
    const PagedTransferRequest& request) {
    if (!installed_ || !thread_pool_) {
        return Status::InternalError(
            "NCCL transport is not installed" LOC_MARK);
    }
    if (shutting_down_.load(std::memory_order_acquire)) {
        return Status::InvalidArgument(
            "NCCL transport is shutting down" LOC_MARK);
    }
    // Paged GIN needs receiver participation so the destination stream observes
    // remote writes before the source reports completion to its caller. Reuse
    // the regular NCCL wait-ack control path for this PoC.
    if (request.page_bytes == 0) {
        return Status::InvalidArgument(
            "Paged transfer page_bytes must be nonzero" LOC_MARK);
    }
    if (request.src_layer_ptrs.size() != request.dst_layer_ptrs.size()) {
        return Status::InvalidArgument(
            "Paged transfer source and destination layer counts differ"
            LOC_MARK);
    }
    if (request.src_page_indices.size() != request.dst_page_indices.size()) {
        return Status::InvalidArgument(
            "Paged transfer source and destination page tables differ"
            LOC_MARK);
    }

    const size_t layer_count = request.src_layer_ptrs.size();
    const size_t page_count = request.src_page_indices.size();
    if (layer_count == 0 || page_count == 0) return Status::OK();
    if (page_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return Status::InvalidArgument(
            "Paged transfer page table is too large" LOC_MARK);
    }
    if (page_count > std::numeric_limits<size_t>::max() / sizeof(int32_t)) {
        return Status::InvalidArgument(
            "Paged transfer page table byte size overflows" LOC_MARK);
    }

    bool has_valid_pages = false;
    int32_t min_src_page = 0;
    int32_t max_src_page = 0;
    int32_t min_dst_page = 0;
    int32_t max_dst_page = 0;
    for (size_t i = 0; i < page_count; ++i) {
        const int32_t src_page = request.src_page_indices[i];
        const int32_t dst_page = request.dst_page_indices[i];
        if (src_page < 0 || dst_page < 0) continue;
        if (!has_valid_pages) {
            min_src_page = src_page;
            max_src_page = src_page;
            min_dst_page = dst_page;
            max_dst_page = dst_page;
            has_valid_pages = true;
        } else {
            min_src_page = std::min(min_src_page, src_page);
            max_src_page = std::max(max_src_page, src_page);
            min_dst_page = std::min(min_dst_page, dst_page);
            max_dst_page = std::max(max_dst_page, dst_page);
        }
    }
    if (!has_valid_pages) return Status::OK();

    auto page_span = [&](int32_t min_page, int32_t max_page, const char* name,
                         size_t& span) -> Status {
        const size_t pages = static_cast<size_t>(max_page - min_page) + 1;
        if (pages > std::numeric_limits<size_t>::max() / request.page_bytes) {
            return Status::InvalidArgument(
                std::string(name) + " byte size overflows" LOC_MARK);
        }
        span = pages * request.page_bytes;
        return Status::OK();
    };

    size_t source_span = 0;
    size_t target_span = 0;
    CHECK_STATUS(page_span(min_src_page, max_src_page, "Paged source span",
                           source_span));
    CHECK_STATUS(page_span(min_dst_page, max_dst_page, "Paged target span",
                           target_span));

    const size_t source_base_offset = static_cast<size_t>(min_src_page)
        * request.page_bytes;
    const size_t target_base_offset = static_cast<size_t>(min_dst_page)
        * request.page_bytes;
    std::vector<int32_t> src_page_indices = request.src_page_indices;
    std::vector<int32_t> dst_page_indices = request.dst_page_indices;

    std::vector<TransferContext> contexts;
    contexts.reserve(layer_count);
    for (size_t layer = 0; layer < layer_count; ++layer) {
        if (!request.src_layer_ptrs[layer]) {
            return Status::InvalidArgument(
                "Paged transfer source layer pointer is null" LOC_MARK);
        }
        Request layer_request;
        layer_request.opcode = Request::WRITE;
        layer_request.source = static_cast<char*>(request.src_layer_ptrs[layer])
            + source_base_offset;
        layer_request.target_id = request.target_id;
        if (request.dst_layer_ptrs[layer]
            > std::numeric_limits<uint64_t>::max() - target_base_offset) {
            return Status::InvalidArgument(
                "Paged target base offset overflows" LOC_MARK);
        }
        layer_request.target_offset = request.dst_layer_ptrs[layer]
            + target_base_offset;
        layer_request.length = std::max(source_span, target_span);

        TransferContext ctx;
        CHECK_STATUS(buildTransferContext(layer_request, source_span,
                                          target_span, ctx));
        CHECK_STATUS(configurePoolWindow(ctx));
        const uint64_t source_pages =
            static_cast<uint64_t>(max_src_page) + 1;
        const uint64_t target_pages =
            static_cast<uint64_t>(max_dst_page) + 1;
        if (source_pages >
                ctx.source_visible_length / request.page_bytes ||
            target_pages > ctx.target_visible_length / request.page_bytes) {
            return Status::InvalidArgument(
                "Paged transfer page range exceeds the persistent symmetric "
                "pool window" LOC_MARK);
        }
        if (!contexts.empty()) {
            const auto& first = contexts.front();
            if (ctx.local_device != first.local_device ||
                ctx.remote_device != first.remote_device ||
                ctx.session_key != first.session_key) {
                return Status::InvalidArgument(
                    "Paged transfer layers must use one NCCL session"
                    LOC_MARK);
            }
        }
        contexts.push_back(std::move(ctx));
    }

    std::shared_ptr<CommState> comm_state;
    CHECK_STATUS(ensureComm(contexts.front(), comm_state));
    // Paged KV issues one transfer per page per layer. The LSA path becomes a
    // cudaMemcpyAsync page loop, which is much slower than fused GIN on GB200.
    // Keep LSA for regular contiguous transfers, but route paged transfers to GIN
    // even when the peer is in the local shared-address team.
    const bool use_lsa = false;
    if (!use_lsa && !comm_state->dev_comm_created) {
        return Status::InvalidArgument(
            "NCCL paged GIN requires a device communicator; set "
            "MC_NCCL_FORCE_GIN=1 and NCCL_CUMEM_ENABLE=1 for LSA peers"
            LOC_MARK);
    }
    std::unique_lock<std::mutex> enqueue_lock(comm_state->enqueue_mu);
    const int lanes = static_cast<int>(comm_state->lanes);
    const uint64_t signal_value = comm_state->signal_epoch.fetch_add(
                                      1, std::memory_order_acq_rel) +
        1;
    const char* wait_ack_env = std::getenv("TENT_NCCL_PAGED_WAIT_SOURCE_ACK");
    const bool wait_source_ack =
        !(wait_ack_env && std::string(wait_ack_env) == "0");
    const char* serial_sync_env = std::getenv("TENT_NCCL_PAGED_SERIAL_SYNC");
    const bool serial_sync =
        serial_sync_env && std::string(serial_sync_env) != "0";
    CHECK_STATUS(postRemoteWaitSignal(contexts.front(), comm_state,
                                      signal_value));
    if (pagedGinDiagEnabled())
    {
        LOG(INFO) << "NCCL paged sync posted remote wait-ack: signal="
                  << signal_value << " layers=" << layer_count
                  << " pages=" << page_count << " page_bytes="
                  << request.page_bytes << " wait_source_ack="
                  << wait_source_ack << " serial_sync=" << serial_sync;
    }

    int previous_device = 0;
    if (pagedGinDiagEnabled())
    {
        LOG(INFO) << "NCCL paged sync phase: set source device="
                  << contexts.front().local_device;
    }
    Status status = setCudaDevice(contexts.front().local_device,
                                  previous_device);
    bool device_changed = status.ok();
    int32_t* d_src_pages = nullptr;
    int32_t* d_dst_pages = nullptr;
    TentNcclPagedTransferJob* d_job = nullptr;
    cudaEvent_t event = nullptr;
    bool work_enqueued = false;

    auto cleanup_device_allocations = [&]() {
        auto free_one = [&](void* ptr, const char* name) {
            if (!ptr) return;
            auto err = cudaFree(ptr);
            if (err != cudaSuccess) {
                if (status.ok()) {
                    status = cudaStatus(err, name);
                } else {
                    LOG(WARNING) << name << ": " << cudaGetErrorString(err);
                }
            }
        };
        free_one(d_job, "cudaFree(paged job)");
        free_one(d_dst_pages, "cudaFree(paged dst pages)");
        free_one(d_src_pages, "cudaFree(paged src pages)");
    };

    if (status.ok() && !use_lsa) {
        const size_t page_table_bytes = page_count * sizeof(int32_t);
        if (pagedGinDiagEnabled())
        {
            LOG(INFO) << "NCCL paged sync phase: allocate/copy page tables bytes="
                      << page_table_bytes;
        }
        auto err = cudaMalloc(reinterpret_cast<void**>(&d_src_pages),
                              page_table_bytes);
        status = cudaStatus(err, "cudaMalloc(paged src pages)");
        if (status.ok()) {
            err = cudaMalloc(reinterpret_cast<void**>(&d_dst_pages),
                             page_table_bytes);
            status = cudaStatus(err, "cudaMalloc(paged dst pages)");
        }
        if (status.ok()) {
            err = cudaMalloc(reinterpret_cast<void**>(&d_job),
                             sizeof(TentNcclPagedTransferJob));
            status = cudaStatus(err, "cudaMalloc(paged job)");
        }
        if (status.ok()) {
            err = cudaMemcpyAsync(d_src_pages,
                                  src_page_indices.data(),
                                  page_table_bytes, cudaMemcpyHostToDevice,
                                  comm_state->completion_stream);
            status = cudaStatus(err, "cudaMemcpyAsync(paged src pages)");
            if (status.ok()) work_enqueued = true;
        }
        if (status.ok()) {
            err = cudaMemcpyAsync(d_dst_pages,
                                  dst_page_indices.data(),
                                  page_table_bytes, cudaMemcpyHostToDevice,
                                  comm_state->completion_stream);
            status = cudaStatus(err, "cudaMemcpyAsync(paged dst pages)");
            if (status.ok()) work_enqueued = true;
        }
    }

    TentNcclPagedKvLayout layout;
    layout.page_stride_bytes = request.page_bytes;

    for (size_t layer = 0; status.ok() && layer < contexts.size(); ++layer) {
        const auto& ctx = contexts[layer];
        if (pagedGinDiagEnabled())
        {
            LOG(INFO) << "NCCL paged sync phase: layer " << layer
                      << " ensure destination window target_base=0x"
                      << std::hex << ctx.target_base << std::dec
                      << " target_length=" << ctx.target_length;
        }
        std::shared_ptr<WindowState> window_state;
        status = ensureWindow(ctx, comm_state, window_state);
        if (pagedGinDiagEnabled())
        {
            LOG(INFO) << "NCCL paged sync phase: layer " << layer
                      << " ensure destination window done: "
                      << status.ToString();
        }
        if (!status.ok()) break;

        if (use_lsa) {
            void* peer_ptr = nullptr;
            status = getLsaPeerPointer(window_state->window,
                                       ctx.target_offset,
                                       comm_state->peer_rank, &peer_ptr);
            if (!status.ok()) break;
            const char* src_base = reinterpret_cast<const char*>(
                static_cast<uintptr_t>(ctx.source_base));
            char* dst_base = static_cast<char*>(peer_ptr);
            for (size_t page = 0; page < page_count; ++page) {
                const int32_t src_page = src_page_indices[page];
                const int32_t dst_page = dst_page_indices[page];
                if (src_page < 0 || dst_page < 0) continue;
                auto err = cudaMemcpyAsync(
                    dst_base + static_cast<size_t>(dst_page) *
                                   request.page_bytes,
                    src_base + static_cast<size_t>(src_page) *
                                   request.page_bytes,
                    request.page_bytes, cudaMemcpyDeviceToDevice,
                    comm_state->completion_stream);
                status = cudaStatus(err, "cudaMemcpyAsync(NCCL paged LSA)");
                if (!status.ok()) break;
                work_enqueued = true;
            }
            continue;
        }

        // The persistent symmetric pool window is valid for both handles
        // consumed by tentNcclGinLaunchPagedPut. Page IDs remain pool-relative;
        // the base offsets account only for alignment before each pool.
        std::shared_ptr<WindowState> source_window_state = window_state;

        TentNcclPagedTransferJob job;
        job.src_page_table = d_src_pages;
        job.dst_page_table = d_dst_pages;
        job.num_pages = static_cast<int>(page_count);
        job.layer_begin = 0;
        job.layer_end = 1;
        job.src_layer_stride = 0;
        job.dst_layer_stride = 0;
        job.src_base_offset = static_cast<size_t>(ctx.source_offset);
        job.dst_base_offset = static_cast<size_t>(ctx.target_offset);

        if (pagedGinDiagEnabled())
        {
            LOG(INFO) << "NCCL paged sync phase: layer " << layer
                      << " copy job src_base_offset=" << job.src_base_offset
                      << " dst_base_offset=" << job.dst_base_offset
                      << " num_pages=" << job.num_pages;
        }
        auto err = cudaMemcpyAsync(d_job, &job, sizeof(job),
                                   cudaMemcpyHostToDevice,
                                   comm_state->completion_stream);
        status = cudaStatus(err, "cudaMemcpyAsync(paged job)");
        if (status.ok()) work_enqueued = true;
        if (!status.ok()) break;

        const unsigned long long layer_signal =
            (layer + 1 == contexts.size())
            ? static_cast<unsigned long long>(signal_value)
            : 0;
        if (pagedGinDiagEnabled())
        {
            LOG(INFO) << "NCCL paged sync phase: layer " << layer
                      << " launch paged put layer_signal=" << layer_signal;
        }
        err = tentNcclGinLaunchPagedPut(
            comm_state->dev_comm, comm_state->peer_rank, lanes,
            window_state->window, source_window_state->window, layout, d_job,
            1, layer_signal, comm_state->completion_stream);
        status = cudaStatus(err, "tentNcclGinLaunchPagedPut");
        if (pagedGinDiagEnabled())
        {
            LOG(INFO) << "NCCL paged sync phase: layer " << layer
                      << " launch paged put done: " << status.ToString();
        }
        if (status.ok()) work_enqueued = true;
    }

    if (status.ok() && wait_source_ack) {
        if (pagedGinDiagEnabled())
        {
            LOG(INFO) << "NCCL paged sync phase: launch source ack wait signal="
                      << signal_value;
        }
        auto err = tentNcclGinLaunchWaitSignal(
            comm_state->dev_comm, lanes, lanes,
            static_cast<unsigned long long>(signal_value),
            comm_state->completion_stream);
        status = cudaStatus(err, "tentNcclGinLaunchWaitSignal(paged ack)");
        if (pagedGinDiagEnabled())
        {
            LOG(INFO) << "NCCL paged sync phase: launch source ack wait done: "
                      << status.ToString();
        }
        if (status.ok()) work_enqueued = true;
    } else if (status.ok()) {
        if (pagedGinDiagEnabled())
        {
            LOG(INFO) << "NCCL paged sync phase: source ack wait skipped";
        }
    }
    if (status.ok()) {
        if (pagedGinDiagEnabled())
        {
            LOG(INFO) << "NCCL paged sync phase: create event";
        }
        auto err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        status = cudaStatus(err, "cudaEventCreateWithFlags(paged sync)");
    }
    if (status.ok()) {
        if (pagedGinDiagEnabled())
        {
            LOG(INFO) << "NCCL paged sync phase: record event";
        }
        auto err = cudaEventRecord(event, comm_state->completion_stream);
        status = cudaStatus(err, "cudaEventRecord(paged sync)");
    }
    if (status.ok() && !serial_sync) {
        enqueue_lock.unlock();
    }
    if (status.ok()) {
        if (pagedGinDiagEnabled())
        {
            LOG(INFO) << "NCCL paged sync phase: synchronize event";
        }
        auto err = cudaEventSynchronize(event);
        status = cudaStatus(err, "cudaEventSynchronize(paged sync)");
        if (pagedGinDiagEnabled())
        {
            LOG(INFO) << "NCCL paged sync phase: synchronize event done: "
                      << status.ToString();
        }
    } else if (work_enqueued && comm_state) {
        auto err = cudaStreamSynchronize(comm_state->completion_stream);
        if (err != cudaSuccess) {
            LOG(WARNING) << "cudaStreamSynchronize(paged cleanup): "
                         << cudaGetErrorString(err);
        }
    }

    if (enqueue_lock.owns_lock()) enqueue_lock.unlock();
    if (event) cudaEventDestroy(event);
    cleanup_device_allocations();
    if (device_changed) cudaSetDevice(previous_device);
    return status;
}

Status NcclTransport::submitTransferTasks(
    SubBatchRef batch, const std::vector<Request>& request_list) {
    auto nccl_batch = dynamic_cast<NcclSubBatch*>(batch);
    if (!nccl_batch)
        return Status::InvalidArgument("Invalid NCCL sub-batch" LOC_MARK);
    if (!thread_pool_)
        return Status::InternalError("NCCL transport is not installed" LOC_MARK);
    if (request_list.size() + nccl_batch->task_list.size() >
        nccl_batch->max_size)
        return Status::TooManyRequests("Exceed batch capacity" LOC_MARK);

    std::vector<NcclTask*> new_tasks;
    new_tasks.reserve(request_list.size());
    for (const auto& request : request_list) {
        nccl_batch->task_list.emplace_back();
        auto& task = nccl_batch->task_list.back();
        task.request = request;
        task.status_word.store(TransferStatusEnum::PENDING,
                               std::memory_order_release);
        task.transferred_bytes.store(0, std::memory_order_release);
        task.completion_event.store(nullptr, std::memory_order_release);

        new_tasks.push_back(&task);
    }

    if (new_tasks.empty()) return Status::OK();

    auto task_ptrs = std::make_shared<std::vector<NcclTask*>>(
        std::move(new_tasks));
    try {
        // Communicator and window setup can perform RPCs; keep it off the
        // transfer-engine submission path.
        thread_pool_->enqueue([this, nccl_batch, task_ptrs]() {
            for (auto* task : *task_ptrs) {
                try {
                    startTransfer(task, nccl_batch);
                } catch (const std::exception& e) {
                    markFailed(
                        *task, std::string("NCCL transfer worker failed: ") +
                                   e.what());
                } catch (...) {
                    markFailed(*task, "NCCL transfer worker failed");
                }
            }
        });
    } catch (const std::exception& e) {
        for (auto* task : *task_ptrs) {
            markFailed(
                *task, std::string("NCCL submit worker enqueue failed: ") +
                           e.what());
        }
        return Status::InternalError(
            std::string("NCCL submit worker enqueue failed: ") + e.what() +
            LOC_MARK);
    }

    return Status::OK();
}

Status NcclTransport::getTransferStatus(SubBatchRef batch, int task_id,
                                        TransferStatus& status) {
    auto nccl_batch = dynamic_cast<NcclSubBatch*>(batch);
    if (!nccl_batch)
        return Status::InvalidArgument("Invalid NCCL sub-batch" LOC_MARK);
    if (task_id < 0 || task_id >= (int)nccl_batch->task_list.size()) {
        return Status::InvalidArgument("Invalid task id" LOC_MARK);
    }

    auto& task = nccl_batch->task_list[task_id];
    auto current = task.status_word.load(std::memory_order_acquire);
    auto event = task.completion_event.load(std::memory_order_acquire);
    if (current == TransferStatusEnum::PENDING && event) {
        auto err = cudaEventQuery(event);
        if (err == cudaSuccess) {
            task.transferred_bytes.store(task.request.length,
                                         std::memory_order_release);
            task.status_word.store(TransferStatusEnum::COMPLETED,
                                   std::memory_order_release);
            current = TransferStatusEnum::COMPLETED;
        } else if (err != cudaErrorNotReady) {
            task.status_word.store(TransferStatusEnum::FAILED,
                                   std::memory_order_release);
            current = TransferStatusEnum::FAILED;
        }
    }

    status.s = current;
    status.transferred_bytes =
        task.transferred_bytes.load(std::memory_order_acquire);
    return Status::OK();
}

Status NcclTransport::allocateLocalMemory(void** addr, size_t size,
                                          MemoryOptions& options) {
    LocationParser location(options.location);
    if (location.type() != "cuda") {
        return Status::InvalidArgument(
            "NCCL transport only allocates CUDA memory" LOC_MARK);
    }

    int previous_device = 0;
    CHECK_CUDA(cudaGetDevice(&previous_device));
    CHECK_CUDA(cudaSetDevice(location.index()));
    auto status = ncclStatus(ncclMemAlloc(addr, size), "ncclMemAlloc");
    cudaSetDevice(previous_device);
    if (!status.ok()) return status;

    std::lock_guard<std::mutex> lock(allocation_mutex_);
    nccl_allocations_.insert(reinterpret_cast<uint64_t>(*addr));
    return Status::OK();
}

Status NcclTransport::freeLocalMemory(void* addr, size_t size) {
    {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        auto it = nccl_allocations_.find(reinterpret_cast<uint64_t>(addr));
        if (it == nccl_allocations_.end()) {
            return Platform::getLoader().free(addr, size);
        }
        nccl_allocations_.erase(it);
    }
    CHECK_NCCL(ncclMemFree(addr));
    return Status::OK();
}

bool NcclTransport::isNcclAllocated(uint64_t addr) const {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    return nccl_allocations_.count(addr) != 0;
}

bool NcclTransport::isCudaLocation(const std::string& location) const {
    return LocationParser(location).type() == "cuda";
}

Status NcclTransport::addMemoryBuffer(BufferDesc& desc,
                                      const MemoryOptions& options) {
    if (!isCudaLocation(desc.location)) return Status::OK();

    const bool nccl_allocated = isNcclAllocated(desc.addr);
    if (!nccl_allocated && !allow_external_window_buffers_) {
        if (options.type == NCCL) {
            return Status::InvalidArgument(
                "NCCL host RMA requires ncclMemAlloc/VMM-compatible CUDA "
                "buffers; set transports/nccl/allow_external_window_buffers "
                "only if the caller guarantees window registration support" LOC_MARK);
        }
        return Status::OK();
    }

    desc.transports.push_back(TransportType::NCCL);
    desc.transport_attrs[TransportType::NCCL] =
        nccl_allocated ? "allocator=ncclMemAlloc;window=deferred"
                       : "allocator=external;window=deferred";
    return Status::OK();
}

Status NcclTransport::removeMemoryBuffer(BufferDesc& desc) {
    desc.transport_attrs.erase(TransportType::NCCL);
    return Status::OK();
}

}  // namespace tent
}  // namespace mooncake
