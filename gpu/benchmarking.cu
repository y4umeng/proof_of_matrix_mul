#include <iostream>     // For std::cout, std::endl
#include <vector>       // For std::vector
#include <string>       // For std::string
#include <iomanip>      // For std::hex, std::setw, std::setfill
#include <chrono>       // For timing
#include <numeric>      // For std::iota
#include <algorithm>    // For std::generate
#include <random>       // For std::mt19937 and std::uniform_int_distribution
#include <cuda_runtime.h> // For CUDA runtime API
#include <cuhash.hpp>   // Main header for cuHash from cuPQC SDK

// Helper macro to check for CUDA errors
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__,       \
                    __LINE__, cudaGetErrorString(err));                          \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// --- cuHash GPU Hashing Logic ---
using SHA256_Thread_Hasher = decltype(cupqc::SHA2_256() + cupqc::Thread());

__global__ void sha256_gpu_kernel(uint8_t* output_digest_gpu,
                                  const uint8_t* input_data_gpu,
                                  size_t input_length) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        SHA256_Thread_Hasher hasher;
        hasher.reset();
        hasher.update(input_data_gpu, input_length);
        hasher.finalize();
        hasher.digest(output_digest_gpu, SHA256_Thread_Hasher::digest_size);
    }
}

// --- Simple CPU SHA256 Implementation (for baseline comparison) ---
// Based on a common public domain implementation structure
namespace CPUSHA256 {
    constexpr uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
    inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
    inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
    inline uint32_t sigma0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
    inline uint32_t sigma1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
    inline uint32_t gamma0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
    inline uint32_t gamma1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

    void transform(uint32_t state[8], const uint8_t block[64]) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = ((uint32_t)block[i * 4 + 0] << 24) |
                   ((uint32_t)block[i * 4 + 1] << 16) |
                   ((uint32_t)block[i * 4 + 2] << 8)  |
                   ((uint32_t)block[i * 4 + 3] << 0);
        }
        for (int i = 16; i < 64; ++i) {
            w[i] = gamma1(w[i - 2]) + w[i - 7] + gamma0(w[i - 15]) + w[i - 16];
        }

        uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = sigma1(e);
            uint32_t ch_val = ch(e, f, g);
            uint32_t temp1 = h + S1 + ch_val + K[i] + w[i];
            uint32_t S0 = sigma0(a);
            uint32_t maj_val = maj(a, b, c);
            uint32_t temp2 = S0 + maj_val;

            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

    std::vector<uint8_t> hash(const std::vector<uint8_t>& input) {
        uint32_t state[8] = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        };

        std::vector<uint8_t> padded_input = input;
        uint64_t total_bits = (uint64_t)input.size() * 8;

        // Padding: append '1' bit (0x80 byte)
        padded_input.push_back(0x80);
        // Pad with 0s until length % 64 == 56
        while (padded_input.size() % 64 != 56) {
            padded_input.push_back(0x00);
        }
        // Append original length in bits as 64-bit big-endian integer
        for (int i = 7; i >= 0; --i) {
            padded_input.push_back((total_bits >> (i * 8)) & 0xFF);
        }

        for (size_t i = 0; i < padded_input.size(); i += 64) {
            transform(state, padded_input.data() + i);
        }

        std::vector<uint8_t> digest(32);
        for (int i = 0; i < 8; ++i) {
            digest[i * 4 + 0] = (state[i] >> 24) & 0xFF;
            digest[i * 4 + 1] = (state[i] >> 16) & 0xFF;
            digest[i * 4 + 2] = (state[i] >> 8)  & 0xFF;
            digest[i * 4 + 3] = (state[i] >> 0)  & 0xFF;
        }
        return digest;
    }
} // namespace CPUSHA256


// Function to generate random byte data
std::vector<uint8_t> generate_random_data(size_t num_bytes) {
    std::vector<uint8_t> data(num_bytes);
    // Using a fixed seed for reproducibility of data, not for cryptographic security
    std::mt19937 rng(12345); 
    std::uniform_int_distribution<int> dist(0, 255);
    std::generate(data.begin(), data.end(), [&]() { return static_cast<uint8_t>(dist(rng)); });
    return data;
}

// Function to print digest in hex format
void print_digest(const std::string& label, const std::vector<uint8_t>& digest) {
    std::cout << label << ": ";
    for (uint8_t byte : digest) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
    }
    std::cout << std::dec << std::endl;
}

// Function to benchmark hashing for a given data size
void benchmark_data_size(size_t data_size_bytes, int num_iterations = 10) {
    std::cout << "\n--- Benchmarking Data Size: " << data_size_bytes << " bytes (" 
              << data_size_bytes / (1024.0 * 1024.0) << " MB) ---" << std::endl;

    std::vector<uint8_t> host_input_data = generate_random_data(data_size_bytes);
    
    // --- GPU cuHash Benchmark ---
    uint8_t* device_input_data = nullptr;
    uint8_t* device_output_digest = nullptr;
    size_t digest_size_bytes = SHA256_Thread_Hasher::digest_size;

    CUDA_CHECK(cudaMalloc(&device_input_data, data_size_bytes));
    CUDA_CHECK(cudaMalloc(&device_output_digest, digest_size_bytes));
    CUDA_CHECK(cudaMemcpy(device_input_data, host_input_data.data(), data_size_bytes, cudaMemcpyHostToDevice));

    dim3 num_blocks(1);
    dim3 threads_per_block(1);
    
    // Warmup GPU run
    if (num_iterations > 1) { // Don't warmup if only 1 iteration
        sha256_gpu_kernel<<<num_blocks, threads_per_block>>>(device_output_digest, device_input_data, data_size_bytes);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<double> gpu_times_ms;
    std::vector<uint8_t> gpu_digest_result(digest_size_bytes);

    for (int i = 0; i < num_iterations; ++i) {
        CUDA_CHECK(cudaDeviceSynchronize()); // Ensure previous work is done
        auto start_gpu = std::chrono::high_resolution_clock::now();
        
        sha256_gpu_kernel<<<num_blocks, threads_per_block>>>(device_output_digest, device_input_data, data_size_bytes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to complete
        
        auto end_gpu = std::chrono::high_resolution_clock::now();
        gpu_times_ms.push_back(std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count());
        
        if (i == 0) { // Only copy and print digest once
            CUDA_CHECK(cudaMemcpy(gpu_digest_result.data(), device_output_digest, digest_size_bytes, cudaMemcpyDeviceToHost));
        }
    }
    
    double avg_gpu_time_ms = 0;
    if (!gpu_times_ms.empty()) {
        double sum_gpu_time_ms = 0;
        for(double t : gpu_times_ms) sum_gpu_time_ms += t;
        avg_gpu_time_ms = sum_gpu_time_ms / gpu_times_ms.size();
    }
    print_digest("GPU cuHash SHA256", gpu_digest_result);
    std::cout << "Avg GPU Hashing Time (kernel launch + exec + sync): " << avg_gpu_time_ms << " ms" << std::endl;
    if (data_size_bytes > 0 && avg_gpu_time_ms > 0) {
        double throughput_gb_s = (static_cast<double>(data_size_bytes) / (1024.0 * 1024.0 * 1024.0)) / (avg_gpu_time_ms / 1000.0);
        std::cout << "GPU Throughput: " << throughput_gb_s * 1000.0 << " MB/s" << std::endl;
    }

    CUDA_CHECK(cudaFree(device_input_data));
    CUDA_CHECK(cudaFree(device_output_digest));

    // --- CPU SHA256 Benchmark (using our simple C++ implementation) ---
    // Warmup CPU run
    if (num_iterations > 1) {
        CPUSHA256::hash(host_input_data);
    }

    std::vector<double> cpu_times_ms;
    std::vector<uint8_t> cpu_digest_result;

    for (int i = 0; i < num_iterations; ++i) {
        auto start_cpu = std::chrono::high_resolution_clock::now();
        cpu_digest_result = CPUSHA256::hash(host_input_data); // Recompute each time
        auto end_cpu = std::chrono::high_resolution_clock::now();
        cpu_times_ms.push_back(std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count());
    }

    double avg_cpu_time_ms = 0;
    if(!cpu_times_ms.empty()){
        double sum_cpu_time_ms = 0;
        for(double t : cpu_times_ms) sum_cpu_time_ms += t;
        avg_cpu_time_ms = sum_cpu_time_ms / cpu_times_ms.size();
    }
    print_digest("CPU C++ SHA256  ", cpu_digest_result);
    std::cout << "Avg CPU Hashing Time: " << avg_cpu_time_ms << " ms" << std::endl;
    if (data_size_bytes > 0 && avg_cpu_time_ms > 0) {
        double throughput_gb_s = (static_cast<double>(data_size_bytes) / (1024.0 * 1024.0 * 1024.0)) / (avg_cpu_time_ms / 1000.0);
        std::cout << "CPU Throughput: " << throughput_gb_s * 1000.0 << " MB/s" << std::endl;
    }
    
    // Verify digests match (optional, but good for sanity)
    if (gpu_digest_result == cpu_digest_result) {
        std::cout << "SUCCESS: GPU and CPU digests match." << std::endl;
    } else {
        std::cout << "ERROR: GPU and CPU digests DO NOT match!" << std::endl;
    }
}


int main() {
    std::cout << "Starting Rigorous Hashing Benchmark..." << std::endl;
    int iterations_per_size = 20; // Number of times to hash for averaging (includes one implicit warmup)

    // Test various data sizes
    benchmark_data_size(1 * 1024, iterations_per_size);         // 1 KB
    benchmark_data_size(128 * 1024, iterations_per_size);       // 128 KB
    benchmark_data_size(512 * 1024, iterations_per_size);       // 512 KB
    benchmark_data_size(1 * 1024 * 1024, iterations_per_size);  // 1 MB
    benchmark_data_size(2 * 1024 * 1024, iterations_per_size);  // 2 MB
    benchmark_data_size(4 * 1024 * 1024, iterations_per_size);  // 4 MB
    // benchmark_data_size(8 * 1024 * 1024, iterations_per_size);  // 8 MB - uncomment for larger tests

    // Test with the original "Hello, cuHash!" string for direct comparison
    std::string specific_test_str = "Hello, cuHash! This is a test message.";
    std::vector<uint8_t> specific_test_data(specific_test_str.begin(), specific_test_str.end());
    std::cout << "\n--- Benchmarking Specific String: \"" << specific_test_str << "\" (" << specific_test_data.size() << " bytes) ---" << std::endl;
    // For very small data, increase iterations to get more stable average, but 1 may be enough to see fixed overhead
    benchmark_data_size(specific_test_data.size(), 50); 


    std::cout << "\nAll benchmarks complete." << std::endl;
    return 0;
}

/*
nvcc -std=c++17 -O3 -arch=sm_89 \
     -I/path/to/your/cupqc-sdk/include/cupqc \
     -I/path/to/your/cupqc-sdk/include \
     benchmarking.cu -o benchmarking \
     -L/path/to/your/cupqc-sdk/lib -lcupqc -lcuhash
*/
