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
                                                     // idx % powf(2, exp)
            if (idx <= n && lowerNeighbor >= 1 && (idx & (1 << exp) - 1) == 0) {
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

        __global__ void kernUpsweepBlock(int n, int* idata, int* odata) {
            // BLOCKSIZE must be power of two
            extern __shared__ int s[];

            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            //if (idx >= n) return;

            // If it is the last block, adjust its size accordingly.
            unsigned effectiveSize = blockDim.x;

            // load into shared memory
            if (idx < n) {
                s[threadIdx.x] = idata[idx];
            } else { // virtual padding
                s[threadIdx.x] = 0;
            }
            __syncthreads();

            // For implementation purposes, we will mirror the implementation
            // compared to the stepped methods so the biggest numbers will end
            // at the 0th index.
            for (int c = 2; c <= effectiveSize; c *= 2) {
                if (c * threadIdx.x + c / 2 < effectiveSize) {
                    s[c * threadIdx.x] += s[c * threadIdx.x + (c / 2)];
                }
                __syncthreads();
            }
            if (idx < n) idata[idx] = s[threadIdx.x];
            if (threadIdx.x == 0) {
                odata[blockIdx.x] = s[threadIdx.x];
            }
        }

        __global__ void kernDownsweepBlock(int n, int* idata, int* odata) {
            // BLOCKSIZE must be power of two
            extern __shared__ int s[];

            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

            // recover the downsweep result for this block from the recursive
            // layer above, remember, we are writing in reverse!
            if (threadIdx.x == 0) {
                //int totalBlocks = (n + blockDim.x - 1) / blockDim.x;
                s[threadIdx.x] = idata[blockIdx.x];
            // load into shared memory
            } else if (idx < n) {
                s[threadIdx.x] = odata[idx];
            }
            else { // virtual padding
                s[threadIdx.x] = 0;
            }
            __syncthreads();

            // The implementation is mirrored compared to downsweepStep
            for (int c = blockDim.x; c >= 2; c /= 2) {
                if (c * threadIdx.x + c / 2 < blockDim.x) {
                    int temp = s[c * threadIdx.x];
                    s[c * threadIdx.x] += s[c * threadIdx.x + (c / 2)];
                    s[c * threadIdx.x + (c / 2)] = temp;
                }
                __syncthreads();
            }
            if (idx < n) odata[idx] = s[threadIdx.x];
        }

        __global__ void kernReverse(int n, int* idata, int* odata) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) odata[idx] = idata[n - idx - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
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

            timer().startGpuTimer();
            // Up-Sweep
            for (int exp = 1; exp <= roundUpN; ++exp) {
                kernUpsweepStep<<<divup(totalN, blocksize), blocksize>>>(totalN, exp, d_data);
            }
            cudaDeviceSynchronize();
            // Down-Sweep
            cudaMemset(d_data + (totalN - 1), 0, 1 * sizeof(int));
            for (int exp = roundUpN; exp >= 1; --exp) {
                kernDownsweepStep<<<divup(totalN, blocksize), blocksize>>>(totalN, exp, d_data);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy output data from d_data");
            cudaFree(d_data);

        }


        void scanSharedMemory(int n, int* odata, const int* idata) {
            unsigned blocksize = 128;
            // We will just use one buffer, each cycle the number of 
            // elements we process is divided by blocksize.

            int roundArraySize = n;
            int sum = n;
            int* breakpoints = new int[20]; //TODO find through log
            breakpoints[0] = 0;
            breakpoints[1] = sum;
            int breakpointsSize = 2;
            printf("Breakpoints: 0, %i,", sum);
            while (roundArraySize > 1) {
                roundArraySize = divup(roundArraySize, blocksize);
                sum += roundArraySize;
                breakpoints[breakpointsSize] = sum;
                breakpointsSize++;
                printf(" %i,", roundArraySize);
            }
            printf("\n");


            int *d_data, *d_dataUnreversed;
            cudaMalloc((void**)&d_dataUnreversed, n * sizeof(int));
            checkCUDAError("cudaMalloc d_dataUnreversed");
            cudaMalloc((void**)&d_data, sum * sizeof(int));
            checkCUDAError("cudaMalloc d_data");

            cudaMemset(d_data, 0, sum * sizeof(int));
            checkCUDAError("cudaMemset d_data");
            
            cudaMemcpy(d_dataUnreversed, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy initial data to d_dataUnreversed");

            timer().startGpuTimer();

            kernReverse<<<divup(n, blocksize), blocksize>>>(n, d_dataUnreversed, d_data);
            cudaDeviceSynchronize();
            int* out = new int[sum];
            cudaMemcpy(out, d_data, sum * sizeof(int), cudaMemcpyDeviceToHost);
            printArray(n, out, false);
            checkCUDAError("kernReverse");

            for (int i = 0; i < breakpointsSize - 1; ++i) {
                printf("Interval is %i to %i\n", breakpoints[i], breakpoints[i + 1]);
                // interval is from breakpoints[i] to breakpoints[i+1]
                kernUpsweepBlock<<<divup(breakpoints[i+1] - breakpoints[i], blocksize), blocksize, blocksize * 1 * sizeof(int)>>>
                    (breakpoints[i + 1] - breakpoints[i], d_data + breakpoints[i], d_data + breakpoints[i + 1]);
                checkCUDAError("naiveScanSharedMem");
                cudaDeviceSynchronize();

            }
            
            cudaMemcpy(out, d_data, sum * sizeof(int), cudaMemcpyDeviceToHost);
            printArray(sum, out, false);

            cudaMemset(d_data + sum - 1, 0, 1 * sizeof(int));
            for (int i = breakpointsSize - 3; i >= 0; --i) {
                printf("Interval2 is %i to %i\n", breakpoints[i], breakpoints[i + 1]);
                // interval is from breakpoints[i] to breakpoints[i+1]

                kernDownsweepBlock<<<divup(breakpoints[i + 1] - breakpoints[i], blocksize), blocksize>>>
                    (breakpoints[i + 1] - breakpoints[i], d_data + breakpoints[i + 1], d_data + breakpoints[i]);
                checkCUDAError("addPrefix");
            }


            cudaMemcpy(out, d_data, sum * sizeof(int), cudaMemcpyDeviceToHost);
            printArray(sum, out, false);
            delete[] out;

            timer().endGpuTimer();

            kernReverse<<<divup(n, blocksize), blocksize>>>(n, d_data, d_dataUnreversed);
            checkCUDAError("kernReverse");

            cudaMemcpy(odata, d_dataUnreversed, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy output data from d_data");
            cudaFree(d_data);
            cudaFree(d_dataUnreversed);

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

            timer().startGpuTimer();
            Common::kernMapToBoolean<<<divup(totalN, blocksize), blocksize>>>(totalN, d_bools, d_data);
            checkCUDAError("kernMapToBoolean");
            cudaDeviceSynchronize();
            // Up-Sweep
            for (int exp = 1; exp <= roundUpN; ++exp) {
                kernUpsweepStep<<<divup(totalN, blocksize), blocksize>>>(totalN, exp, d_bools);
            }
            cudaDeviceSynchronize();
            // Down-Sweep
            cudaMemset(d_bools + (totalN - 1), 0, 1 * sizeof(int));
            for (int exp = roundUpN; exp >= 1; --exp) {
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
            timer().endGpuTimer();

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


            return sizePlusMaybeOne;
        }
    }
}
