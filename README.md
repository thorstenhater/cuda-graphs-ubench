# Micro-benchmark for CUDA Graphs

Investigates the effect of using graphs for concurrent kernel execution.
Three kernels of increasing complexity are used

- empty
- axpy
- newton

The benchmark runs a number of _epochs_, each comprising a set of serial
kernel launches followed by a set of concurrent kernel launches.
This process is repeated a number of times to obtain statistically sound 
results.
