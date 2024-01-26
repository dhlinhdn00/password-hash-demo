#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include "sha256.cuh"

#define MAX_LEN 8 

__device__ int cuda_memcmp(const BYTE *a, const BYTE *b, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (a[i] != b[i]) {
            return -1;
        }
    }
    return 0;
}

__global__ void brute_force_kernel(char *charset, int charset_len, BYTE *target_hash, char *result, int *found) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    char current_string[MAX_LEN + 1];
    BYTE hash[SHA256_BLOCK_SIZE];
    SHA256_CTX ctx;

    // Cải tiến cách tạo chuỗi ở đây
    // Ví dụ, bạn có thể thêm mã để tạo các chuỗi với độ dài khác nhau
    // dựa trên giá trị của idx

    sha256_init(&ctx);
    sha256_update(&ctx, (BYTE *)current_string, strlen(current_string)); // Cập nhật để sử dụng strlen
    sha256_final(&ctx, hash);

    if (cuda_memcmp(hash, target_hash, SHA256_BLOCK_SIZE) == 0) {
        memcpy(result, current_string, MAX_LEN + 1);
        *found = 1;
    }

    // Mã gỡ lỗi: In ra chuỗi hiện tại và hash tương ứng
    printf("String: %s, Hash: ", current_string);
    for (int i = 0; i < SHA256_BLOCK_SIZE; i++) {
        printf("%02x", hash[i]);
    }
    printf("\n");
}

void convert_hex_to_bytes(const char *hex_str, BYTE *bytes) {
    for (int i = 0; i < SHA256_BLOCK_SIZE; i++) {
        sscanf(hex_str + 2 * i, "%2hhx", &bytes[i]);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <target_hash>\n", argv[0]);
        return 1;
    }

    char *target_hash_str = argv[1];
    BYTE target_hash[SHA256_BLOCK_SIZE];
    convert_hex_to_bytes(target_hash_str, target_hash);

    char charset[] = "abcdefghijklmnopqrstuvwxyz"; // Bảng ký tự
    int charset_len = strlen(charset);

    char *d_charset, *d_result;
    BYTE *d_target_hash;
    int *d_found;

    cudaMalloc((void **)&d_charset, charset_len);
    cudaMemcpy(d_charset, charset, charset_len, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_target_hash, SHA256_BLOCK_SIZE);
    cudaMemcpy(d_target_hash, target_hash, SHA256_BLOCK_SIZE, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_result, MAX_LEN + 1);
    cudaMemset(d_result, 0, MAX_LEN + 1);

    cudaMalloc((void **)&d_found, sizeof(int));
    cudaMemset(d_found, 0, sizeof(int));

    int numThreads = 256;
    int numBlocks = (charset_len ^ MAX_LEN + numThreads - 1) / numThreads;

    brute_force_kernel<<<numBlocks, numThreads>>>(d_charset, charset_len, d_target_hash, d_result, d_found);

    cudaDeviceSynchronize();

    char result[MAX_LEN + 1];
    cudaMemcpy(result, d_result, MAX_LEN + 1, cudaMemcpyDeviceToHost);

    int found;
    cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);

    if (found) {
        printf("Password found: %s\n", result);
    } else {
        printf("Password not found\n");
    }

    cudaFree(d_charset);
    cudaFree(d_target_hash);
    cudaFree(d_result);
    cudaFree(d_found);

    return 0;
}
