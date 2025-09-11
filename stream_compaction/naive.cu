#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        __global__ void addStride(int n, int stride, int* idata, int* odata) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx + stride < n) {
                odata[idx + stride] = idata[idx] + idata[idx + stride];
            }
        }

        __global__ void inclScanToExclScan(int n, int* idata, int* odata) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n - 1) {
                odata[idx + 1] = idata[idx];
            }
            if (idx == n - 1) {
                odata[0] = 0;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            
            unsigned blocksize = 128;

            // n rounded up to the nearest power of two
            int roundUpN = ilog2ceil(n);
            int totalN = pow(2, roundUpN);

            int *d_ping, *d_pong;
            cudaMalloc((void**)&d_ping, totalN * sizeof(int));
            checkCUDAError("cudaMalloc ping");
            cudaMalloc((void**)&d_pong, totalN * sizeof(int));
            checkCUDAError("cudaMalloc pong");

            cudaMemset(d_ping, 0, totalN * sizeof(int));
            checkCUDAError("cudaMemset ping");
            cudaMemcpy(d_ping, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy initial data to ping");

            cudaMemcpy(d_pong, d_ping, totalN * sizeof(int), cudaMemcpyDeviceToDevice);

            for (int i = 0; i < roundUpN; ++i) {

                addStride<<<divup(totalN, blocksize), blocksize>>>(n, pow(2, i), d_ping, d_pong);
                checkCUDAError("addStride failed");
                cudaDeviceSynchronize();

                cudaMemcpy(d_ping, d_pong, totalN * sizeof(int), cudaMemcpyDeviceToDevice);

            }

            inclScanToExclScan<<<divup(totalN, blocksize), blocksize>>>(n, d_ping, d_pong);

            cudaMemcpy(odata, d_pong, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy output data from pong");
            cudaFree(d_ping);
            cudaFree(d_pong);
            timer().endGpuTimer();
        }
    }
}
