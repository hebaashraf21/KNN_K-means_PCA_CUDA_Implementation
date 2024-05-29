
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define MAX_ERR 1e-6

__global__ void knn_kernel(float *reference_points, float *query_points, int *results, int n, int m, int k, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        float *min_distances = new float[k];
        int *min_indices = new int[k];

        for (int i = 0; i < k; ++i) {
            min_distances[i] = INFINITY;
            min_indices[i] = -1;
        }

        for (int i = 0; i < n; ++i) {
            float distance = 0;
            for (int j = 0; j < dim; ++j) {
                float diff = reference_points[i * dim + j] - query_points[idx * dim + j];
                distance += diff * diff;
            }

            for (int l = 0; l < k; ++l) {
                if (distance < min_distances[l]) {
                    for (int t = k - 1; t > l; --t) {
                        min_distances[t] = min_distances[t - 1];
                        min_indices[t] = min_indices[t - 1];
                    }
                    min_distances[l] = distance;
                    min_indices[l] = i;
                    break;
                }
            }
        }

        for (int i = 0; i < k; ++i) {
            results[idx * k + i] = min_indices[i];
        }

        delete[] min_distances;
        delete[] min_indices;
    }
}

void knn_with_cuda(float *reference_points, float *query_points, int *results, int n, int m, int k, int dim) {
    float *d_reference, *d_query;
    int *d_results;

    cudaMalloc((void **)&d_reference, sizeof(float) * dim * n);
    cudaMalloc((void **)&d_query, sizeof(float) * dim * m);
    cudaMalloc((void **)&d_results, sizeof(int) * m * k);

    cudaMemcpy(d_reference, reference_points, sizeof(float) * dim * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query_points, sizeof(float) * dim * m, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (m + blockSize - 1) / blockSize;

    knn_kernel<<<numBlocks, blockSize>>>(d_reference, d_query, d_results, n, m, k, dim);

    cudaMemcpy(results, d_results, sizeof(int) * m * k, cudaMemcpyDeviceToHost);

    cudaFree(d_reference);
    cudaFree(d_query);
    cudaFree(d_results);
}

int main() {
    int sizes[] = {100, 1000, 10000, 50000, 100000,1000000};
    int m = 1000;  // Number of query points
    int k = 5;      // Number of nearest neighbors
    int dim = 3;    // Number of dimensions

    for (int i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        int n = sizes[i]; // Number of reference points

        float *reference_points = (float *)malloc(sizeof(float) * dim * n);
        float *query_points = (float *)malloc(sizeof(float) * dim * m);
        int *results = (int *)malloc(sizeof(int) * m * k);

        // Initialize reference_points and query_points with random values
        for (int i = 0; i < dim * n; ++i) {
            reference_points[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        for (int i = 0; i < dim * m; ++i) {
            query_points[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        auto start = std::chrono::high_resolution_clock::now();
        knn_with_cuda(reference_points, query_points, results, n, m, k, dim);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float, std::milli> duration = end - start;
        std::cout << "Time taken with CUDA for n=" << n << ": " << duration.count() << " milliseconds" << std::endl;

        free(reference_points);
        free(query_points);
        free(results);
    }

    return 0;
}
