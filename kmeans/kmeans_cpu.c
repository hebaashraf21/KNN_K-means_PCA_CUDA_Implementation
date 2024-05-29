#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define MAX_ITER 10
#define MAX_LINE_LENGTH 1024

// Function to read CSV data
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

float distance(float *a, float *b, int dims) {
    float dist = 0;
    for (int i = 0; i < dims; ++i) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(dist);
}

void kMeansClusterAssignment(float *datapoints, int *clust_assn, float *centroids, int N, int K, int D) {
    for (int p = 0; p < N; ++p) {
        float min_dist = INFINITY;
        int closest_centroid = 0;

        for (int c = 0; c < K; ++c) {
            float dist = distance(&datapoints[p * D], &centroids[c * D], D);
            if (dist < min_dist) {
                min_dist = dist;
                closest_centroid = c;
            }
        }
        clust_assn[p] = closest_centroid;
    }
}

void kMeansCentroidUpdate(float *datapoints, int *clust_assn, float *centroids, int *clust_sizes, int N, int K, int D) {
    // Reset centroids and sizes
    for (int c = 0; c < K; ++c) {
        clust_sizes[c] = 0;
        for (int d = 0; d < D; ++d) {
            centroids[c * D + d] = 0;
        }
    }

    // Sum all points in each cluster
    for (int p = 0; p < N; ++p) {
        int cluster_id = clust_assn[p];
        for (int d = 0; d < D; ++d) {
            centroids[cluster_id * D + d] += datapoints[p * D + d];
        }
        clust_sizes[cluster_id] += 1;
    }

    // Normalize the centroids
    for (int c = 0; c < K; ++c) {
        if (clust_sizes[c] > 0) {
            for (int d = 0; d < D; ++d) {
                centroids[c * D + d] /= clust_sizes[c];
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.csv> <K>\n", argv[0]);
        return -1;
    }

    const char *filename = argv[1];
    int K = atoi(argv[2]);

    float *datapoints;
    int *clust_assn;
    float *centroids;
    int *clust_sizes;
    int num_points = 0;
    int num_dims = 0;

    if (readCSVData(filename, &datapoints, &num_points, &num_dims) != 0) {
        fprintf(stderr, "Error reading data from file\n");
        return -1;
    }

    clust_assn = (int *)malloc(num_points * sizeof(int));
    centroids = (float *)malloc(K * num_dims * sizeof(float));
    clust_sizes = (int *)malloc(K * sizeof(int));

    // Initialize centroids (choose first K points as initial centroids)
    for (int c = 0; c < K; ++c) {
        for (int d = 0; d < num_dims; ++d) {
            centroids[c * num_dims + d] = datapoints[c * num_dims + d];
        }
        clust_sizes[c] = 0;
    }

    clock_t start_total = clock();

    for (int cur_iter = 0; cur_iter < MAX_ITER; ++cur_iter) {
        kMeansClusterAssignment(datapoints, clust_assn, centroids, num_points, K, num_dims);
        kMeansCentroidUpdate(datapoints, clust_assn, centroids, clust_sizes, num_points, K, num_dims);

       /* printf("Iteration %d centroids:\n", cur_iter + 1);
        for (int i = 0; i < K; ++i) {
            printf("Centroid %d: ", i);
            for (int j = 0; j < num_dims; ++j) {
                printf("%f ", centroids[i * num_dims + j]);
            }
            printf("\n");
        }*/
    }

    clock_t stop_total = clock();
    double total_time = (double)(stop_total - start_total) / CLOCKS_PER_SEC;
    printf("Total time : %f seconds\n", total_time);

    free(datapoints);
    free(clust_assn);
    free(centroids);
    free(clust_sizes);

    return 0;
}
