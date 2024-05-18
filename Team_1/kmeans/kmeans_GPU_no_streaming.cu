#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

#define TPB 128   // Threads per block
#define MAX_ITER 10
#define MAX_LINE_LENGTH 1024

// Function to read CSV data
// Reads data from a CSV file.
// Determines the number of points (num_points) and dimensions (num_dims).
// Allocates memory for the data array.
int readCSVData(const char *filename, float **data, int *num_points, int *num_dims) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Unable to open file");
        return -1;
    }

    char line[MAX_LINE_LENGTH];
    int n = 0, d = 0;

    // Read first line to determine the number of dimensions
    if (fgets(line, MAX_LINE_LENGTH, file)) {
        char *token = strtok(line, ",");
        while (token) {
            d++;
            token = strtok(NULL, ",");
        }
    }

    // Count number of points
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        n++;
    }

    // Allocate memory for data
    *data = (float *)malloc(n * d * sizeof(float));
    if (!*data) {
        perror("Unable to allocate memory");
        fclose(file);
        return -1;
    }

    rewind(file);

    // Read data into the array
    int point = 0;
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        char *token = strtok(line, ",");
        int dim = 0;
        while (token) {
            (*data)[point * d + dim] = atof(token);
            token = strtok(NULL, ",");
            dim++;
        }
        point++;
    }

    fclose(file);

    *num_points = n;
    *num_dims = d;

    return 0;
}

// Computes the Euclidean distance between two points in dims dimensions.
__device__ float distance(float *a, float *b, int dims) {
    float dist = 0;
    for (int i = 0; i < dims; ++i) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(dist);
}

// Assigns each data point to the nearest centroid.
__global__ void kMeansClusterAssignment(float *d_datapoints, int *d_clust_assn, float *d_centroids, int N, int K, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float min_dist = INFINITY;
    int closest_centroid = 0;

    for (int c = 0; c < K; ++c) {
        float dist = distance(&d_datapoints[idx * D], &d_centroids[c * D], D);
        if (dist < min_dist) {
            min_dist = dist;
            closest_centroid = c;
        }
    }
    d_clust_assn[idx] = closest_centroid;
}

// Updates centroids by averaging the assigned points.
__global__ void kMeansCentroidUpdate(float *d_datapoints, int *d_clust_assn, float *d_centroids, int *d_clust_sizes, int N, int K, int D) {
    // This declares shared memory for each block to hold partial centroid accumulators.
    extern __shared__ float s_centroids[];
    // to store the count of data points assigned to each centroid. (in the shared)
    int *s_counts = (int *)&s_centroids[K * D];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Each thread initializes its partial centroid accumulator (s_centroids) and cluster size counter (s_counts) to zero.
    if (tid < K) {
        for (int i = 0; i < D; ++i) {
            s_centroids[tid * D + i] = 0;
        }
        s_counts[tid] = 0;
    }
    // to ensure proper initialization
    __syncthreads();

    // Centroid Accumulation
    if (idx < N) {
        int cluster_id = d_clust_assn[idx];
        // The thread atomically accumulates the coordinates of the data point 
        // into the partial centroid accumulator (s_centroids) corresponding to its assigned cluster.
        for (int i = 0; i < D; ++i) {
            atomicAdd(&s_centroids[cluster_id * D + i], d_datapoints[idx * D + i]);
        }

        // increments the cluster size counter (s_counts) for the assigned cluster.
        atomicAdd(&s_counts[cluster_id], 1);
    }
    // to ensure all partial accumulators and cluster size counters are updated
    __syncthreads();

    // Centroid Update:
    if (tid < K) {
        for (int i = 0; i < D; ++i) {
          // Each thread atomically adds its partial centroid accumulator (s_centroids)
          // to the corresponding centroid in the global centroids array.
            atomicAdd(&d_centroids[tid * D + i], s_centroids[tid * D + i]);
        }
        // It also atomically adds its cluster size counter (s_counts)
        // to the corresponding cluster size in the global cluster size array.
        atomicAdd(&d_clust_sizes[tid], s_counts[tid]);
    }
}

// Normalizes centroids by dividing the sum of points by the number of points in each cluster.
__global__ void normalizeCentroids(float *d_centroids, int *d_clust_sizes, int K, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;

    for (int i = 0; i < D; ++i) {
        if (d_clust_sizes[idx] > 0) {
            d_centroids[idx * D + i] /= d_clust_sizes[idx];
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.csv> <K>\n", argv[0]);
        return -1;
    }

    const char *filename = argv[1];
    // number of clusters
    int K = atoi(argv[2]);

    float *datapoints;
    int num_points = 0;
    int num_dims = 0;

    if (readCSVData(filename, &datapoints, &num_points, &num_dims) != 0) {
        fprintf(stderr, "Error reading data from file\n");
        return -1;
    }

    float *d_datapoints, *d_centroids;
    int *d_clust_assn, *d_clust_sizes;

    cudaMalloc(&d_datapoints, num_points * num_dims * sizeof(float));
    cudaMalloc(&d_clust_assn, num_points * sizeof(int));
    cudaMalloc(&d_centroids, K * num_dims * sizeof(float));
    // keep track of the number of points assigned to each cluster.
    cudaMalloc(&d_clust_sizes, K * sizeof(int));

    float *h_centroids = (float *)malloc(K * num_dims * sizeof(float));
    // keep track of the number of points assigned to each cluster.
    int *h_clust_sizes = (int *)malloc(K * sizeof(int));

    srand(time(0));

    // It copies the feature values from a data point to initialize the corresponding centroid.
    for (int c = 0; c < K; ++c) {
        for (int d = 0; d < num_dims; ++d) {
            h_centroids[c * num_dims + d] = datapoints[c * num_dims + d];
        }
        h_clust_sizes[c] = 0;
    }

    cudaMemcpy(d_centroids, h_centroids, K * num_dims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_datapoints, datapoints, num_points * num_dims * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start_total, stop_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);

    cudaEventRecord(start_total);

    for (int cur_iter = 0; cur_iter < MAX_ITER; ++cur_iter) {
        kMeansClusterAssignment<<<(num_points + TPB - 1) / TPB, TPB>>>(d_datapoints, d_clust_assn, d_centroids, num_points, K, num_dims);

        // Reset centroids and sizes on device
        cudaMemset(d_centroids, 0, K * num_dims * sizeof(float));
        cudaMemset(d_clust_sizes, 0, K * sizeof(int));

        /* 
        K * num_dims * sizeof(float): This calculates the size of shared memory required to store 
        the accumulator for each centroid's dimensions.
        K * sizeof(int): This calculates the size of shared memory required to store 
        the array holding the size of each cluster.
        */
        size_t shared_mem_size = K * num_dims * sizeof(float) + K * sizeof(int);
        kMeansCentroidUpdate<<<(num_points + TPB - 1) / TPB, TPB, shared_mem_size>>>(d_datapoints, d_clust_assn, d_centroids, d_clust_sizes, num_points, K, num_dims);

        normalizeCentroids<<<(K + TPB - 1) / TPB, TPB>>>(d_centroids, d_clust_sizes, K, num_dims);

        cudaMemcpy(h_centroids, d_centroids, K * num_dims * sizeof(float), cudaMemcpyDeviceToHost);

        /*printf("Iteration %d centroids:\n", cur_iter + 1);
        for (int i = 0; i < K; ++i) {
            printf("Centroid %d: ", i);
            for (int j = 0; j < num_dims; ++j) {
                printf("%f ", h_centroids[i * num_dims + j]);
            }
            printf("\n");
        }*/
    }

    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    float total_milliseconds = 0;
    cudaEventElapsedTime(&total_milliseconds, start_total, stop_total);
    printf("Total time: %f seconds\n", total_milliseconds / 1000.0);

    cudaFree(d_datapoints);
    cudaFree(d_clust_assn);
    cudaFree(d_centroids);
    cudaFree(d_clust_sizes);

    free(h_centroids);
    free(datapoints);
    free(h_clust_sizes);

    return 0;
}
