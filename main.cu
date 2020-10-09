constexpr size_t n_epoch    = 10;
constexpr size_t n_serial   = 10;
constexpr size_t n_parallel = 10;
constexpr size_t n_block    = 10;
constexpr size_t n_thread   = 128;
constexpr size_t n_reps     = 5;
constexpr size_t n_element  = 2048;

namespace kernels {
    __global__ void empty(size_t n) {}

    __global__ void axpy(double *y, const double* x, double alpha, size_t n) {
        auto i = threadIdx.x + blockIdx.x*blockDim.x;
        if (i<n) { y[i] += alpha*x[i]; }
    }

    __device__ double f(double x)  { return exp(cos(x)) - 2; }
    __device__ double fp(double x) { return -sin(x) * exp(cos(x)); }

    __global__ void newton(size_t n_iter, double *x, size_t n) {
        auto i = threadIdx.x + blockIdx.x*blockDim.x;
        if (i < n) {
            auto x0 = x[i];
            for(int iter = 0; iter < n_iter; ++iter) {
                x0 -= f(x0)/fp(x0);
            }
            x[i] = x0;
        }
    }
}

template<class F, typename... Ts>
void cuda_api(const std::string& error, F f, Ts... ts) {
    auto rc = F(std::forward(ts...));
    if (rc != cudaSuccess) throw std::runtime_exception(error);
}

template<typename K, typename... As>
void bench_null() {
    for (auto rep = 0; rep < n_rep; ++rep) {
        cuda_api("pre-sync", cudaDeviceSynchronize);
        cuda_api("post-sync", cudaDeviceSynchronize);
    }
}

template<typename K, typename... As>
void bench_kernels(K kernel, As... as) {
    for (auto rep = 0; rep < n_rep; ++rep) {
        cuda_device_synchronize();
        cuda_api("pre-sync", cudaDeviceSynchronize);
        for (auto epoch = 0ul; epoch < n_epoch; ++epoch) {
            for (auto serial = 0ul; serial < n_serial; ++serial) {
                kernel<<<n_thread, n_block>>>(as...);
            }
            for (auto parallel = 0ul; parallel < n_parallel; ++parallel) {
                kernel<<<n_thread, n_block>>>(as...);
            }
        }
        cuda_api("post-sync", cudaDeviceSynchronize);
        auto t1 = time();
    }
}

cudaGraphNode_t add_empty_node(cudaGraph_t& graph) {
    cudaGraphNode_t node = {0};
    return node;
}

cudaGraphNode_t add_kernel_node(cudaGraph_t& graph, K kernel, const std::vector<void*> args) {
    cudaGraphNode_t node = {0};
    cudaKernelNodeParams params = {0};
    params.func           = (void*) kernel;
    params.gridDim        = n_block;
    params.blockDim       = n_thread;
    params.sharedMemBytes = 0;
    params.kernelParams   = (void**) args.data();
    params.extra          = nullptr;

    cuda_api("kernel add", cudaGraphAddKernelNode, &node, graph, nullptr, 0, &params);

    return node;
}

void add_dependencies(cudaGraph_t& graph, cudaGraphNode_t from, cudaGraphNode_t to) {
}

void add_dependencies(cudaGraph_t& graph, cudaGraphNode_t from, const std::vector<cudaGraphNode_t>& to) {
}

void add_dependencies(cudaGraph_t& graph, const std::vector<cudaGraphNode_t>& from, cudaGraphNode_t to) {
}

template<typename K, typename... As>
void bench_graph(K kernel, As... as) {
    cudaStream_t    stream   = {0};
    cudaGraph_t     graph    = {0};
    cudaExecGraph_t instance = {0};

    cuda_api("stream init", cudaStreamCreate, &stream);
    cuda_api("graph init", cudaGraphCreate, &graph, 0);

    auto root = add_empty_node(graph);
    auto last = root;

    for (auto epoch = 0ul; epoch < n_epoch; ++epoch) {
        for (auto serial = 0ul; serial < n_serial; ++serial) {
            auto node = add_kernel_node(grap, kernel, as);
            add_dependencies(graph, last, node);
            last = node;
        }
        std::vector<cudaGraphNode_t> nodes;
        for (auto parallel = 0ul; parallel < n_parallel; ++parallel) {
            auto node = add_kernel_node(grap, kernel, as);
            nodes.push_back(std::move(node));
        }
        last = add_empty_node(graph);
        add_dependencies(graph, last, nodes);
        last = add_empty_node(graph);
        add_dependencies(graph, nodes, last);
    }
    cuda_api("instantiate graph", cudaGraphInstantiate, &instance, graph, nullptr, nullptr, 0);

    for (auto rep = 0; rep < n_rep; ++rep) {
        cuda_api("pre-sync", cudaDeviceSynchronize);
        cuda_api("graph exec", cudaGraphLaunch, instance, stream);
        cuda_api("post-sync", cudaDeviceSynchronize);
    }
}

int main() {
    double* x;
    double* y;
    double alpha;

    cuda_api("alloc x", cudaMalloc, n_element);
    cuda_api("alloc y", cudaMalloc, n_element);

    auto result_null          = bench_null();

    auto result_empty_kernel  = bench_kernel(kernels::empty);
    for (auto n = 128ul; n < n_element; n *= 2) {
        bench_kernel(kernels::axpy, y, x, alpha, n_element);
        bench_kernel(kernels::newton, n_iter, x, n_element);
    }

    auto result_empty_graph   = bench_graph(kernels::empty);
    for (auto n = 128ul; n < n_element; n *= 2) {
        bench_graph(kernels::axpy, y, x, alpha, n_element);
        bench_graph(kernels::newton, n_iter, x, n_element);
    }
}
