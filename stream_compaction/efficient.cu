#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        void printArray(int n, int* a, bool abridged = false) {
            printf("    [ ");
            for (int i = 0; i < n; i++) {
                if (abridged && i + 2 == 15 && n > 16) {
                    i = n - 2;
                    printf("... ");
                }
                printf("%3d ", a[i]);
            }
            printf("]\n");
        }
        __global__ void kernUpsweepStep(int n, int exp, int* data) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            // Similar to array modifications when making a heap, we want 
            // our array to be 1-indexed instead of 0-indexed because 
            // 0 is divisible by all powers of two - but we want that position
            // on the right.
            idx += 1;
            unsigned lowerNeighbor = idx - (1 << (exp - 1));
            if (idx <= n && lowerNeighbor >= 1 && idx % (1 << exp) == 0) {
                //data[idx - 1] += data[idx - (1 << (exp - 1)) - 1];
                data[idx - 1] += data[lowerNeighbor - 1];
            }
        }

        __global__ void kernDownsweepStep(int n, int exp, int* data) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            // Similar to array modifications when making a heap, we want 
            // our array to be 1-indexed instead of 0-indexed because 
            // 0 is divisible by all powers of two - but we want that position
            // on the right.
            idx += 1;
            unsigned lowerNeighbor = idx - (1 << (exp - 1));
            if (idx <= n && lowerNeighbor >= 1 && idx % (1 << exp) == 0) {
                int temp = data[idx - 1];
                data[idx - 1] += data[lowerNeighbor - 1];
                data[lowerNeighbor - 1] = temp;
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
            
            int* d_data;
            cudaMalloc((void**)&d_data, totalN * sizeof(int));
            checkCUDAError("cudaMalloc d_data"); 

            cudaMemset(d_data, 0, totalN * sizeof(int));
            checkCUDAError("cudaMemset d_data");
            cudaMemcpy(d_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy initial data to d_data");

            // Up-Sweep
            for (int exp = 1; exp <= totalN; ++exp) {
                kernUpsweepStep<<<divup(totalN, blocksize), blocksize>>>(totalN, exp, d_data);
            }
            cudaDeviceSynchronize();
            // Down-Sweep
            cudaMemset(d_data + (totalN - 1), 0, 1 * sizeof(int));
            for (int exp = totalN; exp >= 1; --exp) {
                kernDownsweepStep<<<divup(totalN, blocksize), blocksize>>>(totalN, exp, d_data);
            }

            cudaMemcpy(odata, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy output data from d_data");
            cudaFree(d_data);

            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            unsigned blocksize = 128;

            // n rounded up to the nearest power of two
            int roundUpN = ilog2ceil(n);
            int totalN = pow(2, roundUpN);

            int *d_data, *d_bools, *d_indices, *d_output;
            cudaMalloc((void**)&d_data, totalN * sizeof(int));
            checkCUDAError("cudaMalloc d_data");
            cudaMalloc((void**)&d_bools, totalN * sizeof(int));
            checkCUDAError("cudaMalloc d_bools");
            cudaMalloc((void**)&d_indices, totalN * sizeof(int));
            checkCUDAError("cudaMalloc d_indices");
            cudaMalloc((void**)&d_output, totalN * sizeof(int));
            checkCUDAError("cudaMalloc d_output");

            cudaMemset(d_data, 0, totalN * sizeof(int));
            checkCUDAError("cudaMemset d_data");
            cudaMemcpy(d_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata into d_data");

            Common::kernMapToBoolean<<<divup(totalN, blocksize), blocksize>>>(totalN, d_bools, d_data);
            checkCUDAError("kernMapToBoolean");
            cudaDeviceSynchronize();
            // Up-Sweep
            for (int exp = 1; exp <= totalN; ++exp) {
                kernUpsweepStep << <divup(totalN, blocksize), blocksize >> > (totalN, exp, d_bools);
            }
            cudaDeviceSynchronize();
            // Down-Sweep
            cudaMemset(d_bools + (totalN - 1), 0, 1 * sizeof(int));
            for (int exp = totalN; exp >= 1; --exp) {
                kernDownsweepStep<<<divup(totalN, blocksize), blocksize>>>(totalN, exp, d_bools);
            }
            checkCUDAError("upsweep and downsweep scan");
            int sizePlusMaybeOne;
            cudaMemcpy(&sizePlusMaybeOne, d_bools + (totalN - 1), 1 * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy sizePlusMaybeOne");
            printf("%i", sizePlusMaybeOne);

            cudaMemset(d_output, 0, totalN * sizeof(int));
            checkCUDAError("cudaMemset d_output");

            Common::kernScatter<<<divup(totalN, blocksize), blocksize>>>(totalN, d_output, d_data, d_data, d_bools);
            checkCUDAError("kernScatter");
            cudaMemcpy(odata, d_output, std::min(sizePlusMaybeOne, n) * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to output");

            if (odata[sizePlusMaybeOne - 1] == 0) {
                sizePlusMaybeOne--;
            }

            cudaFree(d_data);
            cudaFree(d_bools);
            cudaFree(d_indices);
            cudaFree(d_output);
            checkCUDAError("cudaFree");


            timer().endGpuTimer();
            return sizePlusMaybeOne;
        }
    }
}
