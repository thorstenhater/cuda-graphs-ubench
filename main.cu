#include <memory>
#include <string>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime_api.h>

constexpr size_t n_block    = 64;
constexpr size_t n_thread   = 128;
constexpr size_t n_iter     = 5;
constexpr size_t n_rep      = 10;
constexpr size_t n_element_lo  =   512;
constexpr size_t n_element_hi  = 65536;
constexpr size_t n_epoch_lo = 128;
constexpr size_t n_epoch_hi = 128;
constexpr size_t n_chunk_lo = 32;
constexpr size_t n_chunk_hi = 512;

using timer = std::chrono::high_resolution_clock;
using nsecs = std::chrono::microseconds;

namespace kernels {
  __global__ void empty(size_t n) {}

  __global__ void axpy(double *y, double* x, double alpha, size_t n) {
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

using result = std::vector<double>;

#define cuda_api(error, f, ...)						\
  do {									\
    auto rc = f(__VA_ARGS__);						\
    if (rc != cudaSuccess) {						\
      throw std::runtime_error(std::string(error) + ": " +  cudaGetErrorString(rc)); \
    } else {								\
      /*std::cerr << error << '\n';*/					\
    }									\
  } while (0)

void print_result(result& res, 
		  size_t epoch, size_t chunk, size_t element, 
		  const std::string& kind, const std::string& kernel) {
  std::cout << kind    << "," 
	    << kernel  << ","
	    << epoch   << ","
	    << chunk   << ","
	    << element << ",";
  for (const auto& r: res)  std::cout << r << ",";    
  std::cout << std::endl;
  std::cout.flush();
}						

auto bench_null() {
  result res;
  for (auto rep = 0; rep < n_rep; ++rep) {
    auto t0 = timer::now();
    cuda_api("pre-sync", cudaDeviceSynchronize);
    cuda_api("post-sync", cudaDeviceSynchronize);
    auto t1 = timer::now();
    auto dt = (t1 - t0).count();
    res.push_back(dt);
  }
  return res;
}

template<typename K, typename... As>
auto bench_kernels(const std::string& tag, size_t n_epoch, size_t n_chunk, size_t n_element, K kernel, As... as) {
  result res;
  for (auto rep = 0; rep < n_rep; ++rep) {
    auto t0 = timer::now();
    cuda_api("pre-sync", cudaDeviceSynchronize);
    for (auto epoch = 0ul; epoch < n_epoch; ++epoch) {
      for (auto serial = 0ul; serial < n_chunk; ++serial) {
	kernel<<<n_thread, n_block>>>(as...);
      }
      for (auto parallel = 0ul; parallel < n_chunk; ++parallel) {
	kernel<<<n_thread, n_block>>>(as...);
      }
    }
    cuda_api("post-sync", cudaDeviceSynchronize);
    auto t1 = timer::now();
    auto dt = (t1 - t0).count();
    res.push_back(dt);
  }
  print_result(res, n_epoch, n_chunk, n_element, tag, "kernels");
}

cudaGraphNode_t add_empty_node(cudaGraph_t& graph) {
  cudaGraphNode_t node = {0};
  cuda_api("add empty", cudaGraphAddEmptyNode, &node, graph, nullptr, 0);
  return node;
}

template<class K>
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

template<int N, int ix, typename... Ts>
constexpr void set_args(std::unique_ptr<std::tuple<Ts...>>& values, std::vector<void*>& pointers) {
  if constexpr(ix >= N) {
      return;
    } else {
    pointers[ix] = &std::get<ix>(*values);
    set_args<N, ix + 1, Ts...>(values, pointers);
  }
}

void add_dependencies(cudaGraph_t& graph, const cudaGraphNode_t& from, const cudaGraphNode_t& to) {
  cuda_api("add edge", cudaGraphAddDependencies, graph, &from, &to, 1); 
}

void add_dependencies(cudaGraph_t& graph, const cudaGraphNode_t& from_, const std::vector<cudaGraphNode_t>& to) {
  auto n = to.size();
  std::vector<cudaGraphNode_t> from(n, from_);
  cuda_api("add edge", cudaGraphAddDependencies, graph, from.data(), to.data(), n); 
}

void add_dependencies(cudaGraph_t& graph, const std::vector<cudaGraphNode_t>& from, const cudaGraphNode_t& to_) {
  auto n = from.size();
  std::vector<cudaGraphNode_t> to(n, to_);
  cuda_api("add edge", cudaGraphAddDependencies, graph, from.data(), to.data(), n); 
}

template<typename K, typename... As>
auto bench_graph(const std::string& tag, size_t n_epoch, size_t n_chunk, size_t n_element, K kernel, As... as) {
  cudaStream_t    stream   = {0};
  cudaGraph_t     graph    = {0};
  cudaGraphExec_t instance = {0};

  std::vector<void*> args(sizeof...(As), nullptr);
  auto tmp = std::make_unique<std::tuple<As...>>(as...);
  set_args<sizeof...(As), 0, As...>(tmp, args);

  cuda_api("stream init", cudaStreamCreate, &stream);
  cuda_api("graph init", cudaGraphCreate, &graph, 0);

  auto root = add_empty_node(graph);
  auto last = root;

  for (auto epoch = 0ul; epoch < n_epoch; ++epoch) {
    for (auto serial = 0ul; serial < n_chunk; ++serial) {
      auto node = add_kernel_node(graph, kernel, args);
      add_dependencies(graph, last, node);
      last = node;
    }
    std::vector<cudaGraphNode_t> nodes;
    for (auto parallel = 0ul; parallel < n_chunk; ++parallel) {
      auto node = add_kernel_node(graph, kernel, args);
      nodes.push_back(std::move(node));
    }
    last = add_empty_node(graph);
    add_dependencies(graph, last, nodes);
    last = add_empty_node(graph);
    add_dependencies(graph, nodes, last);
  }
  cuda_api("instantiate graph", cudaGraphInstantiate, &instance, graph, nullptr, nullptr, 0);
  result res;
  for (auto rep = 0; rep < n_rep; ++rep) {
    auto t0 = timer::now();
    cuda_api("pre-sync", cudaDeviceSynchronize);
    cuda_api("graph exec", cudaGraphLaunch, instance, stream);
    cuda_api("post-sync", cudaDeviceSynchronize);
    auto t1 = timer::now();
    auto dt = (t1 - t0).count();
    res.push_back(dt);
  }
  print_result(res, n_epoch, n_chunk, n_element, tag, "graphs");
}

int main() {
  double* x;
  double* y;
  double alpha;

  cuda_api("malloc x", cudaMalloc, &x, n_element_hi);
  cuda_api("malloc y", cudaMalloc, &y, n_element_hi);

  for (auto n_epoch = n_epoch_lo; n_epoch <= n_epoch_hi; n_epoch *= 2) {
    for (auto n_chunk = n_chunk_lo; n_chunk <= n_chunk_hi; n_chunk *= 2) {
      for (auto n_element = n_element_lo; n_element <= n_element_hi; n_element *= 2) {
	bench_kernels("empty",  n_epoch, n_chunk, n_element, kernels::empty, n_element);
	bench_kernels("axpy",   n_epoch, n_chunk, n_element, kernels::axpy, y, x, alpha, n_element);
	bench_kernels("newton", n_epoch, n_chunk, n_element, kernels::newton, n_iter, x, n_element);
	bench_graph("empty",    n_epoch, n_chunk, n_element, kernels::empty, n_element);
	bench_graph("axpy",     n_epoch, n_chunk, n_element, kernels::axpy, y, x, alpha, n_element);
	bench_graph("newton",   n_epoch, n_chunk, n_element, kernels::newton, n_iter, x, n_element);
      }
    }
  }
}
