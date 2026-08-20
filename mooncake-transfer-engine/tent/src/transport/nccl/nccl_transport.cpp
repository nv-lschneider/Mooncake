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
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
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


bool pagedGinSummaryDiagEnabled() {
    return envFlagEnabled("TENT_NCCL_PAGED_SUMMARY_DIAG");
}

bool pagedGinAccumDiagEnabled() {
    return envFlagEnabled("TENT_NCCL_PAGED_ACCUM_DIAG");
}

uint64_t pagedGinAccumDiagInterval() {
    static const uint64_t interval = [] {
        constexpr uint64_t kDefaultInterval = 64;
        const char* value =
            std::getenv("TENT_NCCL_PAGED_ACCUM_DIAG_INTERVAL");
        if (!value || !*value) return kDefaultInterval;
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        if (end == value || *end != '\0' || parsed == 0) {
            LOG(WARNING)
                << "Invalid TENT_NCCL_PAGED_ACCUM_DIAG_INTERVAL='"
                << value << "'; using " << kDefaultInterval;
            return kDefaultInterval;
        }
        return static_cast<uint64_t>(parsed);
    }();
    return interval;
}

using PagedDiagClock = std::chrono::steady_clock;

int64_t elapsedMicros(PagedDiagClock::time_point start,
                      PagedDiagClock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
        .count();
}

std::atomic<uint64_t> pagedTransferDiagSequence{0};

struct PagedTimingMetric {
    std::atomic<uint64_t> sum_us{0};
    std::atomic<uint64_t> max_us{0};

    void add(int64_t value_us) {
        const uint64_t value =
            value_us > 0 ? static_cast<uint64_t>(value_us) : 0;
        sum_us.fetch_add(value, std::memory_order_relaxed);
        uint64_t observed = max_us.load(std::memory_order_relaxed);
        while (observed < value &&
               !max_us.compare_exchange_weak(
                   observed, value, std::memory_order_relaxed,
                   std::memory_order_relaxed)) {
        }
    }

    double average(uint64_t count) const {
        return count == 0
            ? 0.0
            : static_cast<double>(sum_us.load(std::memory_order_relaxed)) /
                  static_cast<double>(count);
    }

    uint64_t maximum() const {
        return max_us.load(std::memory_order_relaxed);
    }
};

struct PagedTimingAccumulator {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> failures{0};
    std::atomic<uint64_t> workspace_creates{0};
    std::atomic<uint64_t> workspace_grows{0};
    std::atomic<uint64_t> job_groups{0};
    std::atomic<uint64_t> layers{0};
    std::atomic<uint64_t> pages{0};
    PagedTimingMetric ensure_comm;
    PagedTimingMetric ensure_window;
    PagedTimingMetric set_device;
    PagedTimingMetric workspace_acquire;
    PagedTimingMetric metadata_prepare;
    PagedTimingMetric enqueue_lock_wait;
    PagedTimingMetric post_remote_signal;
    PagedTimingMetric malloc;
    PagedTimingMetric h2d_enqueue;
    PagedTimingMetric put_enqueue;
    PagedTimingMetric ack_enqueue;
    PagedTimingMetric event_record;
    PagedTimingMetric event_query_wait;
    std::atomic<uint64_t> event_queries{0};
    PagedTimingMetric total;
};
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

struct PagedWorkspace {
    void* device_buffer = nullptr;
    void* host_buffer = nullptr;
    size_t capacity = 0;
    cudaEvent_t completion_event = nullptr;
    bool in_use = false;
};

struct NcclTransport::PagedWorkspacePool {
    explicit PagedWorkspacePool(int device) : device_index(device) {}

    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::unique_ptr<PagedWorkspace>> workspaces;
    PagedTimingAccumulator paged_timing;
    int device_index = -1;
};

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
            if (comm) {
                comm->remote_signal_cv.notify_all();
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock(paged_workspace_pool_mutex_);
        for (auto& [_, pool] : paged_workspace_pools_) {
            if (pool) pool->cv.notify_all();
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
        std::lock_guard<std::mutex> lock(paged_workspace_pool_mutex_);
        for (auto& [device, pool] : paged_workspace_pools_) {
            if (!pool) continue;
            int previous_device = 0;
            auto status = setCudaDevice(device, previous_device);
            if (!status.ok()) {
                LOG(WARNING) << "Unable to select CUDA device " << device
                             << " while destroying paged workspace pool: "
                             << status.ToString();
                continue;
            }
            std::lock_guard<std::mutex> pool_lock(pool->mu);
            if (pagedGinAccumDiagEnabled()) {
                const auto& accum = pool->paged_timing;
                const uint64_t count =
                    accum.count.load(std::memory_order_relaxed);
                if (count > 0) {
                    LOG(INFO) << "NCCL paged transfer accumulated timing final: pool_device="
                              << pool->device_index
                              << " count=" << count
                              << " failures=" << accum.failures.load(std::memory_order_relaxed)
                              << " pool_workspaces=" << pool->workspaces.size()
                              << " workspace_creates=" << accum.workspace_creates.load(std::memory_order_relaxed)
                              << " workspace_grows=" << accum.workspace_grows.load(std::memory_order_relaxed)
                              << " event_queries=" << accum.event_queries.load(std::memory_order_relaxed)
                              << " avg_event_query_wait_us=" << accum.event_query_wait.average(count)
                              << " max_event_query_wait_us=" << accum.event_query_wait.maximum();
                }
            }
            for (auto& workspace : pool->workspaces) {
                if (!workspace) continue;
                if (workspace->in_use) {
                    LOG(WARNING)
                        << "Destroying an in-use NCCL paged workspace";
                }
                if (workspace->completion_event) {
                    auto err = cudaEventDestroy(workspace->completion_event);
                    if (err != cudaSuccess) {
                        LOG(WARNING) << "cudaEventDestroy(paged workspace): "
                                     << cudaGetErrorString(err);
                    }
                }
                if (workspace->device_buffer) {
                    auto err = cudaFree(workspace->device_buffer);
                    if (err != cudaSuccess) {
                        LOG(WARNING) << "cudaFree(paged workspace): "
                                     << cudaGetErrorString(err);
                    }
                }
                if (workspace->host_buffer) {
                    auto err = cudaFreeHost(workspace->host_buffer);
                    if (err != cudaSuccess) {
                        LOG(WARNING) << "cudaFreeHost(paged workspace): "
                                     << cudaGetErrorString(err);
                    }
                }
            }
            pool->workspaces.clear();
            cudaSetDevice(previous_device);
        }
        paged_workspace_pools_.clear();
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

Status NcclTransport::getReadyComm(
    const TransferContext& ctx, std::shared_ptr<CommState>& state) {
    {
        std::lock_guard<std::mutex> lock(comm_mutex_);
        auto it = comms_.find(ctx.session_key);
        if (it == comms_.end()) {
            return Status::InvalidArgument(
                "NCCL paged serving found no preconnected communicator for session "
                + ctx.session_key
                + "; communicator creation is forbidden after readiness" LOC_MARK);
        }
        state = it->second;
    }

    std::lock_guard<std::mutex> lock(state->mu);
    if (!state->status.ok()) return state->status;
    if (!state->ready) {
        return Status::InvalidArgument(
            "NCCL paged serving found an incomplete preconnected communicator for session "
            + ctx.session_key
            + "; communicator creation or waiting is forbidden after readiness" LOC_MARK);
    }
    return Status::OK();
}

Status NcclTransport::getReadyWindow(
    const TransferContext& ctx, std::shared_ptr<WindowState>& state) {
    {
        std::lock_guard<std::mutex> lock(window_mutex_);
        auto it = windows_.find(ctx.window_key);
        if (it == windows_.end()) {
            return Status::InvalidArgument(
                "NCCL paged serving found no preconnected exact pool window for key "
                + ctx.window_key
                + "; window registration is forbidden after readiness" LOC_MARK);
        }
        state = it->second;
    }

    std::lock_guard<std::mutex> lock(state->mu);
    if (!state->status.ok()) return state->status;
    if (!state->ready || state->window == nullptr) {
        return Status::InvalidArgument(
            "NCCL paged serving found an incomplete preconnected exact pool window for key "
            + ctx.window_key
            + "; window registration or waiting is forbidden after readiness" LOC_MARK);
    }
    return Status::OK();
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
    const bool diag = pagedGinSummaryDiagEnabled();
    const auto total_start = PagedDiagClock::now();
    int64_t lookup_us = 0;
    int64_t collective_wait_us = 0;
    int64_t rpc_us = 0;
    int64_t set_device_us = 0;
    int64_t alloc_us = 0;
    int64_t register_us = 0;
    int64_t state_wait_us = 0;
    bool should_init = false;
    bool cache_hit = false;
    bool waited_for_initializer = false;
    const auto lookup_start = PagedDiagClock::now();
    {
        std::lock_guard<std::mutex> lock(window_mutex_);
        auto& entry = windows_[ctx.window_key];
        if (!entry) entry = std::make_shared<WindowState>();
        state = entry;
        std::lock_guard<std::mutex> state_lock(state->mu);
        cache_hit = state->ready;
        waited_for_initializer = !state->ready && state->initializing;
        if (!state->ready && !state->initializing) {
            state->initializing = true;
            state->length = ctx.target_length;
            state->device_index = ctx.local_device;
            state->session_key = ctx.session_key;
            should_init = true;
        }
    }
    lookup_us = elapsedMicros(lookup_start, PagedDiagClock::now());

    if (should_init) {
        const auto collective_wait_start = PagedDiagClock::now();
        std::unique_lock<std::mutex> collective_lock(comm_state->collective_mu);
        collective_wait_us = elapsedMicros(collective_wait_start,
                                           PagedDiagClock::now());
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
        const auto rpc_start = PagedDiagClock::now();
        status = ControlClient::registerNcclWindow(ctx.remote_rpc_addr,
                                                   request, response);
        rpc_us = elapsedMicros(rpc_start, PagedDiagClock::now());

        int previous_device = 0;
        bool device_changed = false;
        if (status.ok()) {
            const auto set_device_start = PagedDiagClock::now();
            status = setCudaDevice(ctx.local_device, previous_device);
            set_device_us = elapsedMicros(set_device_start,
                                          PagedDiagClock::now());
            device_changed = status.ok();
        }
        if (status.ok() && ctx.use_paired_window_buffers) {
            state->local_buffer = reinterpret_cast<void*>(ctx.source_base);
        } else if (status.ok()) {
            const auto alloc_start = PagedDiagClock::now();
            status = ncclStatus(ncclMemAlloc(&state->local_buffer,
                                             ctx.target_length),
                                "ncclMemAlloc(window dummy)");
            alloc_us = elapsedMicros(alloc_start, PagedDiagClock::now());
            state->owns_local_buffer = status.ok();
        }
        if (status.ok()) {
            const auto register_start = PagedDiagClock::now();
            status = ncclStatus(
                ncclCommWindowRegister(comm_state->comm, state->local_buffer,
                                       ctx.target_length, &state->window,
                                       NCCL_WIN_COLL_SYMMETRIC),
                "ncclCommWindowRegister");
            register_us = elapsedMicros(register_start,
                                        PagedDiagClock::now());
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

    const auto state_wait_start = PagedDiagClock::now();
    std::unique_lock<std::mutex> lock(state->mu);
    state->cv.wait(lock, [&] { return state->ready || !state->status.ok(); });
    state_wait_us = elapsedMicros(state_wait_start, PagedDiagClock::now());
    Status result = state->status;
    lock.unlock();

    if (diag && (should_init || waited_for_initializer)) {
        LOG(INFO) << "NCCL paged ensure-window summary: key=" << ctx.window_key
                  << " cache_hit=" << cache_hit
                  << " initialized_here=" << should_init
                  << " waited_for_initializer=" << waited_for_initializer
                  << " paired_buffers=" << ctx.use_paired_window_buffers
                  << " lookup_us=" << lookup_us
                  << " collective_wait_us=" << collective_wait_us
                  << " rpc_us=" << rpc_us
                  << " set_device_us=" << set_device_us
                  << " alloc_us=" << alloc_us
                  << " register_us=" << register_us
                  << " state_wait_us=" << state_wait_us
                  << " total_us="
                  << elapsedMicros(total_start, PagedDiagClock::now())
                  << " status=" << result.ToString();
    }
    return result;
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
    if (layer_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return Status::InvalidArgument(
            "Paged transfer layer count is too large" LOC_MARK);
    }
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

    const bool summary_diag = pagedGinSummaryDiagEnabled();
    const bool accum_diag = pagedGinAccumDiagEnabled();
    const uint64_t diag_id = summary_diag
        ? pagedTransferDiagSequence.fetch_add(1, std::memory_order_relaxed) + 1
        : 0;
    const auto transfer_start = PagedDiagClock::now();
    int64_t ensure_comm_us = 0;
    int64_t set_device_us = 0;
    int64_t workspace_acquire_us = 0;
    int64_t metadata_prepare_us = 0;
    int64_t malloc_us = 0;
    int64_t page_copy_enqueue_us = 0;
    int64_t ensure_window_us = 0;
    int64_t job_copy_enqueue_us = 0;
    int64_t enqueue_lock_wait_us = 0;
    int64_t post_remote_signal_us = 0;
    int64_t put_enqueue_us = 0;
    int64_t ack_enqueue_us = 0;
    int64_t event_create_us = 0;
    int64_t event_record_us = 0;
    int64_t event_query_wait_us = 0;
    uint64_t event_queries = 0;
    int64_t event_destroy_us = 0;
    int64_t free_us = 0;
    size_t job_group_count = 0;
    size_t workspace_pool_size = 0;
    bool workspace_created = false;
    bool workspace_grew = false;

    std::shared_ptr<CommState> comm_state;
    const auto ensure_comm_start = PagedDiagClock::now();
    Status ensure_comm_status = getReadyComm(contexts.front(), comm_state);
    ensure_comm_us = elapsedMicros(ensure_comm_start, PagedDiagClock::now());
    CHECK_STATUS(ensure_comm_status);
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
    const int lanes = static_cast<int>(comm_state->lanes);
    const char* wait_ack_env = std::getenv("TENT_NCCL_PAGED_WAIT_SOURCE_ACK");
    const bool wait_source_ack =
        !(wait_ack_env && std::string(wait_ack_env) == "0");
    const char* serial_sync_env = std::getenv("TENT_NCCL_PAGED_SERIAL_SYNC");
    const bool serial_sync =
        serial_sync_env && std::string(serial_sync_env) != "0";
    uint64_t signal_value = 0;

    std::vector<std::shared_ptr<WindowState>> window_states;
    window_states.reserve(contexts.size());
    for (size_t layer = 0; layer < contexts.size(); ++layer) {
        const auto& ctx = contexts[layer];
        if (pagedGinDiagEnabled()) {
            LOG(INFO) << "NCCL paged sync phase: layer " << layer
                      << " lookup destination window target_base=0x"
                      << std::hex << ctx.target_base << std::dec
                      << " target_length=" << ctx.target_length;
        }
        std::shared_ptr<WindowState> window_state;
        const auto ensure_window_start = PagedDiagClock::now();
        Status window_status = getReadyWindow(ctx, window_state);
        ensure_window_us +=
            elapsedMicros(ensure_window_start, PagedDiagClock::now());
        if (!window_status.ok()) return window_status;
        window_states.push_back(std::move(window_state));
    }

    int previous_device = 0;
    if (pagedGinDiagEnabled())
    {
        LOG(INFO) << "NCCL paged sync phase: set source device="
                  << contexts.front().local_device;
    }
    const auto set_device_start = PagedDiagClock::now();
    Status status = setCudaDevice(contexts.front().local_device,
                                  previous_device);
    set_device_us = elapsedMicros(set_device_start, PagedDiagClock::now());
    bool device_changed = status.ok();
    bool work_enqueued = false;
    std::shared_ptr<PagedWorkspacePool> workspace_pool;
    PagedWorkspace* workspace = nullptr;
    int32_t* d_src_pages = nullptr;
    int32_t* d_dst_pages = nullptr;
    TentNcclPagedTransferJob* d_jobs = nullptr;
    TentNcclPagedTransferJob* h_jobs = nullptr;

    const size_t page_table_bytes = page_count * sizeof(int32_t);
    const size_t job_alignment = alignof(TentNcclPagedTransferJob);
    size_t jobs_offset = 0;
    size_t workspace_bytes = 0;
    if (page_table_bytes >
        (std::numeric_limits<size_t>::max() - (job_alignment - 1)) / 2) {
        status = Status::InvalidArgument(
            "Paged transfer workspace byte size overflows" LOC_MARK);
    } else {
        const size_t tables_bytes = page_table_bytes * 2;
        jobs_offset =
            (tables_bytes + job_alignment - 1) & ~(job_alignment - 1);
        if (layer_count >
            (std::numeric_limits<size_t>::max() - jobs_offset) /
                sizeof(TentNcclPagedTransferJob)) {
            status = Status::InvalidArgument(
                "Paged transfer job array byte size overflows" LOC_MARK);
        } else {
            workspace_bytes =
                jobs_offset + layer_count * sizeof(TentNcclPagedTransferJob);
        }
    }

    auto release_workspace = [&]() {
        if (!workspace) return;
        {
            std::lock_guard<std::mutex> lock(workspace_pool->mu);
            workspace->in_use = false;
        }
        workspace_pool->cv.notify_one();
        workspace = nullptr;
    };

    if (status.ok()) {
        {
            std::lock_guard<std::mutex> lock(
                paged_workspace_pool_mutex_);
            auto& pool = paged_workspace_pools_[contexts.front().local_device];
            if (!pool) {
                pool = std::make_shared<PagedWorkspacePool>(
                    contexts.front().local_device);
            }
            workspace_pool = pool;
        }
        const auto acquire_start = PagedDiagClock::now();
        const size_t workspace_limit =
            std::max<size_t>(1, params_.max_concurrent_tasks);
        {
            std::unique_lock<std::mutex> lock(workspace_pool->mu);
            workspace_pool->cv.wait(lock, [&] {
                if (shutting_down_.load(std::memory_order_acquire)) {
                    return true;
                }
                if (workspace_pool->workspaces.size() < workspace_limit) {
                    return true;
                }
                return std::any_of(
                    workspace_pool->workspaces.begin(),
                    workspace_pool->workspaces.end(),
                    [](const auto& candidate) {
                        return candidate && !candidate->in_use;
                    });
            });
            if (shutting_down_.load(std::memory_order_acquire)) {
                status = Status::InvalidArgument(
                    "NCCL transport is shutting down" LOC_MARK);
            } else {
                for (auto& candidate : workspace_pool->workspaces) {
                    if (candidate && !candidate->in_use) {
                        workspace = candidate.get();
                        break;
                    }
                }
                if (!workspace) {
                    auto candidate = std::make_unique<PagedWorkspace>();
                    workspace = candidate.get();
                    workspace_created = true;
                    workspace_pool->workspaces.push_back(
                        std::move(candidate));
                }
                workspace_pool_size = workspace_pool->workspaces.size();
                workspace->in_use = true;
            }
        }
        workspace_acquire_us =
            elapsedMicros(acquire_start, PagedDiagClock::now());

        if (status.ok() && !workspace->completion_event) {
            const auto event_create_start = PagedDiagClock::now();
            auto err = cudaEventCreateWithFlags(
                &workspace->completion_event, cudaEventDisableTiming);
            event_create_us =
                elapsedMicros(event_create_start, PagedDiagClock::now());
            status = cudaStatus(
                err, "cudaEventCreateWithFlags(paged workspace)");
        }

        if (status.ok() && !use_lsa &&
            workspace->capacity < workspace_bytes) {
            const auto malloc_start = PagedDiagClock::now();
            void* new_device_buffer = nullptr;
            void* new_host_buffer = nullptr;
            auto err = cudaMalloc(&new_device_buffer, workspace_bytes);
            status = cudaStatus(err, "cudaMalloc(paged workspace)");
            if (status.ok()) {
                err = cudaMallocHost(&new_host_buffer, workspace_bytes);
                status = cudaStatus(err, "cudaMallocHost(paged workspace)");
            }
            if (!status.ok()) {
                if (new_device_buffer) cudaFree(new_device_buffer);
                if (new_host_buffer) cudaFreeHost(new_host_buffer);
            } else {
                const auto free_start = PagedDiagClock::now();
                if (workspace->device_buffer) {
                    err = cudaFree(workspace->device_buffer);
                    if (err != cudaSuccess) {
                        LOG(WARNING) << "cudaFree(old paged workspace): "
                                     << cudaGetErrorString(err);
                    }
                }
                if (workspace->host_buffer) {
                    err = cudaFreeHost(workspace->host_buffer);
                    if (err != cudaSuccess) {
                        LOG(WARNING)
                            << "cudaFreeHost(old paged workspace): "
                            << cudaGetErrorString(err);
                    }
                }
                free_us +=
                    elapsedMicros(free_start, PagedDiagClock::now());
                workspace->device_buffer = new_device_buffer;
                workspace->host_buffer = new_host_buffer;
                workspace->capacity = workspace_bytes;
                workspace_grew = true;
            }
            malloc_us =
                elapsedMicros(malloc_start, PagedDiagClock::now());
        }

        if (status.ok() && !use_lsa) {
            const auto metadata_prepare_start = PagedDiagClock::now();
            auto* d_base = static_cast<char*>(workspace->device_buffer);
            auto* h_base = static_cast<char*>(workspace->host_buffer);
            d_src_pages = reinterpret_cast<int32_t*>(d_base);
            d_dst_pages =
                reinterpret_cast<int32_t*>(d_base + page_table_bytes);
            d_jobs = reinterpret_cast<TentNcclPagedTransferJob*>(
                d_base + jobs_offset);
            auto* h_src_pages = reinterpret_cast<int32_t*>(h_base);
            auto* h_dst_pages =
                reinterpret_cast<int32_t*>(h_base + page_table_bytes);
            h_jobs = reinterpret_cast<TentNcclPagedTransferJob*>(
                h_base + jobs_offset);
            std::memcpy(h_src_pages, src_page_indices.data(),
                        page_table_bytes);
            std::memcpy(h_dst_pages, dst_page_indices.data(),
                        page_table_bytes);
            for (size_t layer = 0; layer < contexts.size(); ++layer) {
                const auto& ctx = contexts[layer];
                TentNcclPagedTransferJob& job = h_jobs[layer];
                job = TentNcclPagedTransferJob{};
                job.src_page_table = d_src_pages;
                job.dst_page_table = d_dst_pages;
                job.num_pages = static_cast<int>(page_count);
                job.layer_begin = 0;
                job.layer_end = 1;
                job.src_base_offset =
                    static_cast<size_t>(ctx.source_offset);
                job.dst_base_offset =
                    static_cast<size_t>(ctx.target_offset);
            }
            metadata_prepare_us = elapsedMicros(
                metadata_prepare_start, PagedDiagClock::now());
        }
    }

    std::unique_lock<std::mutex> enqueue_lock;
    if (status.ok()) {
        const auto enqueue_lock_start = PagedDiagClock::now();
        enqueue_lock =
            std::unique_lock<std::mutex>(comm_state->enqueue_mu);
        enqueue_lock_wait_us = elapsedMicros(
            enqueue_lock_start, PagedDiagClock::now());
        signal_value = comm_state->signal_epoch.fetch_add(
                           1, std::memory_order_acq_rel) +
            1;
        const auto post_remote_signal_start = PagedDiagClock::now();
        status = postRemoteWaitSignal(contexts.front(), comm_state,
                                      signal_value);
        post_remote_signal_us = elapsedMicros(
            post_remote_signal_start, PagedDiagClock::now());
        if (pagedGinDiagEnabled()) {
            LOG(INFO) << "NCCL paged sync posted remote wait-ack: signal="
                      << signal_value << " layers=" << layer_count
                      << " pages=" << page_count << " page_bytes="
                      << request.page_bytes << " wait_source_ack="
                      << wait_source_ack << " serial_sync=" << serial_sync;
        }
    }
    if (status.ok() && !use_lsa) {
        const auto page_copy_start = PagedDiagClock::now();
        auto err = cudaMemcpyAsync(
            workspace->device_buffer, workspace->host_buffer,
            workspace_bytes, cudaMemcpyHostToDevice,
            comm_state->completion_stream);
        status = cudaStatus(err, "cudaMemcpyAsync(paged workspace)");
        page_copy_enqueue_us =
            elapsedMicros(page_copy_start, PagedDiagClock::now());
        if (status.ok()) work_enqueued = true;
    }

    TentNcclPagedKvLayout layout;
    layout.page_stride_bytes = request.page_bytes;

    if (use_lsa) {
        // Preserve the existing LSA fallback exactly; normal Paged GIN serving
        // forces the device-communicator path above.
        for (size_t layer = 0;
             status.ok() && layer < contexts.size(); ++layer) {
            const auto& ctx = contexts[layer];
            const auto& window_state = window_states[layer];
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
                status = cudaStatus(
                    err, "cudaMemcpyAsync(NCCL paged LSA)");
                if (!status.ok()) break;
                work_enqueued = true;
            }
        }
    } else {
        // Jobs can share one launch only when they use the same already-ready
        // persistent symmetric window. Differing handles remain separate
        // groups. Source and destination intentionally use the same symmetric
        // pool window, matching the previously validated per-layer launch.
        for (size_t group_begin = 0;
             status.ok() && group_begin < contexts.size();) {
            size_t group_end = group_begin + 1;
            while (group_end < contexts.size() &&
                   window_states[group_end]->window ==
                       window_states[group_begin]->window) {
                ++group_end;
            }
            const auto& window_state = window_states[group_begin];
            const int group_jobs =
                static_cast<int>(group_end - group_begin);
            ++job_group_count;
            const unsigned long long group_signal =
                (group_end == contexts.size())
                ? static_cast<unsigned long long>(signal_value)
                : 0;
            if (pagedGinDiagEnabled()) {
                LOG(INFO)
                    << "NCCL paged sync phase: launch job group begin="
                    << group_begin << " jobs=" << group_jobs
                    << " signal=" << group_signal;
            }
            const auto put_start = PagedDiagClock::now();
            auto err = tentNcclGinLaunchPagedPut(
                comm_state->dev_comm, comm_state->peer_rank, lanes,
                window_state->window, window_state->window, layout,
                d_jobs + group_begin, group_jobs, group_signal,
                comm_state->completion_stream);
            status = cudaStatus(err, "tentNcclGinLaunchPagedPut");
            put_enqueue_us +=
                elapsedMicros(put_start, PagedDiagClock::now());
            if (status.ok()) work_enqueued = true;
            group_begin = group_end;
        }
    }

    if (status.ok() && wait_source_ack) {
        if (pagedGinDiagEnabled())
        {
            LOG(INFO) << "NCCL paged sync phase: launch source ack wait signal="
                      << signal_value;
        }
        const auto ack_start = PagedDiagClock::now();
        auto err = tentNcclGinLaunchWaitSignal(
            comm_state->dev_comm, lanes, lanes,
            static_cast<unsigned long long>(signal_value),
            comm_state->completion_stream);
        status = cudaStatus(err, "tentNcclGinLaunchWaitSignal(paged ack)");
        ack_enqueue_us = elapsedMicros(ack_start, PagedDiagClock::now());
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
            LOG(INFO) << "NCCL paged sync phase: record event";
        }
        const auto event_record_start = PagedDiagClock::now();
        auto err = cudaEventRecord(workspace->completion_event,
                                   comm_state->completion_stream);
        status = cudaStatus(err, "cudaEventRecord(paged workspace)");
        event_record_us =
            elapsedMicros(event_record_start, PagedDiagClock::now());
    }
    if (status.ok() && !serial_sync) {
        enqueue_lock.unlock();
    }
    if (status.ok()) {
        if (pagedGinDiagEnabled())
        {
            LOG(INFO) << "NCCL paged sync phase: query completion event";
        }
        const auto event_query_start = PagedDiagClock::now();
        cudaError_t err = cudaErrorNotReady;
        while (err == cudaErrorNotReady) {
            err = cudaEventQuery(workspace->completion_event);
            ++event_queries;
            if (err == cudaErrorNotReady) std::this_thread::yield();
        }
        status = cudaStatus(err, "cudaEventQuery(paged workspace)");
        event_query_wait_us =
            elapsedMicros(event_query_start, PagedDiagClock::now());
        if (pagedGinDiagEnabled())
        {
            LOG(INFO) << "NCCL paged sync phase: completion event ready: "
                      << status.ToString()
                      << " queries=" << event_queries;
        }
    } else if (work_enqueued && comm_state) {
        auto err = cudaStreamSynchronize(comm_state->completion_stream);
        if (err != cudaSuccess) {
            LOG(WARNING) << "cudaStreamSynchronize(paged cleanup): "
                         << cudaGetErrorString(err);
        }
    }

    if (enqueue_lock.owns_lock()) enqueue_lock.unlock();
    release_workspace();
    if (device_changed) cudaSetDevice(previous_device);
    const int64_t total_us =
        elapsedMicros(transfer_start, PagedDiagClock::now());
    if (summary_diag) {
        LOG(INFO) << "NCCL paged transfer summary: id=" << diag_id
                  << " signal=" << signal_value
                  << " workspace_pool_device="
                  << (workspace_pool ? workspace_pool->device_index : -1)
                  << " session=" << contexts.front().session_key
                  << " layers=" << layer_count
                  << " pages=" << page_count
                  << " page_bytes=" << request.page_bytes
                  << " job_groups=" << job_group_count
                  << " workspace_pool_size=" << workspace_pool_size
                  << " workspace_reused=" << (!workspace_created)
                  << " workspace_created=" << workspace_created
                  << " workspace_grew=" << workspace_grew
                  << " workspace_acquire_us=" << workspace_acquire_us
                  << " metadata_prepare_us=" << metadata_prepare_us
                  << " enqueue_lock_wait_us=" << enqueue_lock_wait_us
                  << " post_remote_signal_us=" << post_remote_signal_us
                  << " ensure_comm_us=" << ensure_comm_us
                  << " set_device_us=" << set_device_us
                  << " malloc_us=" << malloc_us
                  << " page_copy_enqueue_us=" << page_copy_enqueue_us
                  << " ensure_window_us=" << ensure_window_us
                  << " job_copy_enqueue_us=" << job_copy_enqueue_us
                  << " put_enqueue_us=" << put_enqueue_us
                  << " ack_enqueue_us=" << ack_enqueue_us
                  << " event_create_us=" << event_create_us
                  << " event_record_us=" << event_record_us
                  << " event_query_wait_us=" << event_query_wait_us
                  << " event_queries=" << event_queries
                  << " event_destroy_us=" << event_destroy_us
                  << " free_us=" << free_us
                  << " total_us=" << total_us
                  << " status=" << status.ToString();
    }
    if (accum_diag && workspace_pool) {
        auto& accum = workspace_pool->paged_timing;
        accum.ensure_comm.add(ensure_comm_us);
        accum.ensure_window.add(ensure_window_us);
        accum.set_device.add(set_device_us);
        accum.workspace_acquire.add(workspace_acquire_us);
        accum.metadata_prepare.add(metadata_prepare_us);
        accum.enqueue_lock_wait.add(enqueue_lock_wait_us);
        accum.post_remote_signal.add(post_remote_signal_us);
        accum.malloc.add(malloc_us);
        accum.h2d_enqueue.add(page_copy_enqueue_us);
        accum.put_enqueue.add(put_enqueue_us);
        accum.ack_enqueue.add(ack_enqueue_us);
        accum.event_record.add(event_record_us);
        accum.event_query_wait.add(event_query_wait_us);
        accum.event_queries.fetch_add(event_queries,
                                      std::memory_order_relaxed);
        accum.total.add(total_us);
        accum.job_groups.fetch_add(
            static_cast<uint64_t>(job_group_count), std::memory_order_relaxed);
        accum.layers.fetch_add(
            static_cast<uint64_t>(layer_count), std::memory_order_relaxed);
        accum.pages.fetch_add(
            static_cast<uint64_t>(page_count), std::memory_order_relaxed);
        if (workspace_created) {
            accum.workspace_creates.fetch_add(1, std::memory_order_relaxed);
        }
        if (workspace_grew) {
            accum.workspace_grows.fetch_add(1, std::memory_order_relaxed);
        }
        if (!status.ok()) {
            accum.failures.fetch_add(1, std::memory_order_relaxed);
        }
        const uint64_t count =
            accum.count.fetch_add(1, std::memory_order_relaxed) + 1;
        const uint64_t interval = pagedGinAccumDiagInterval();
        if (count % interval == 0) {
            LOG(INFO) << "NCCL paged transfer accumulated timing: pool_device="
                      << workspace_pool->device_index
                      << " count=" << count
                      << " failures=" << accum.failures.load(std::memory_order_relaxed)
                      << " workspace_creates=" << accum.workspace_creates.load(std::memory_order_relaxed)
                      << " workspace_grows=" << accum.workspace_grows.load(std::memory_order_relaxed)
                      << " job_groups=" << accum.job_groups.load(std::memory_order_relaxed)
                      << " layers=" << accum.layers.load(std::memory_order_relaxed)
                      << " pages=" << accum.pages.load(std::memory_order_relaxed)
                      << " avg_ensure_comm_us=" << accum.ensure_comm.average(count)
                      << " max_ensure_comm_us=" << accum.ensure_comm.maximum()
                      << " avg_ensure_window_us=" << accum.ensure_window.average(count)
                      << " max_ensure_window_us=" << accum.ensure_window.maximum()
                      << " avg_set_device_us=" << accum.set_device.average(count)
                      << " max_set_device_us=" << accum.set_device.maximum()
                      << " avg_workspace_acquire_us=" << accum.workspace_acquire.average(count)
                      << " max_workspace_acquire_us=" << accum.workspace_acquire.maximum()
                      << " avg_metadata_prepare_us=" << accum.metadata_prepare.average(count)
                      << " max_metadata_prepare_us=" << accum.metadata_prepare.maximum()
                      << " avg_enqueue_lock_wait_us=" << accum.enqueue_lock_wait.average(count)
                      << " max_enqueue_lock_wait_us=" << accum.enqueue_lock_wait.maximum()
                      << " avg_post_remote_signal_us=" << accum.post_remote_signal.average(count)
                      << " max_post_remote_signal_us=" << accum.post_remote_signal.maximum()
                      << " avg_malloc_us=" << accum.malloc.average(count)
                      << " max_malloc_us=" << accum.malloc.maximum()
                      << " avg_h2d_enqueue_us=" << accum.h2d_enqueue.average(count)
                      << " max_h2d_enqueue_us=" << accum.h2d_enqueue.maximum()
                      << " avg_put_enqueue_us=" << accum.put_enqueue.average(count)
                      << " max_put_enqueue_us=" << accum.put_enqueue.maximum()
                      << " avg_ack_enqueue_us=" << accum.ack_enqueue.average(count)
                      << " max_ack_enqueue_us=" << accum.ack_enqueue.maximum()
                      << " avg_event_record_us=" << accum.event_record.average(count)
                      << " max_event_record_us=" << accum.event_record.maximum()
                      << " event_queries=" << accum.event_queries.load(std::memory_order_relaxed)
                      << " avg_event_query_wait_us=" << accum.event_query_wait.average(count)
                      << " max_event_query_wait_us=" << accum.event_query_wait.maximum()
                      << " avg_total_us=" << accum.total.average(count)
                      << " max_total_us=" << accum.total.maximum();
        }
    }
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
