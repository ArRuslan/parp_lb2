#include <bitset>
#include <iostream>
#include <cpuid.h>
#include <cstdint>
#include <iomanip>
#include <omp.h>
#include <random>
#include <complex>
#include <cstring>

// ecx
#define SSE3 (1 << 0)
#define SSSE3 (1 << 9)
#define SSE41 (1 << 19)
#define SSE42 (1 << 20)
#define AVX_1 (1 << 26)
#define AVX_2 (1 << 27)
#define AVX_3 (1 << 28)
// edx
#define MMX (1 << 23)
#define SSE (1 << 25)
#define SSE2 (1 << 26)
#define HYPER_T (1 << 28)

#define E3DNOW (1 << 31)
#define E3DNOW2 (1 << 30)

#define SSE4A (1 << 6)
#define SSE5 (1 << 11)

constexpr uint32_t arr_size = 4096 * 4096;
static std::default_random_engine rand_eng;

void task_1() {
    uint32_t reg_eax, reg_ebx, reg_ecx, reg_edx;
    uint32_t max_func;
    __get_cpuid(0, &max_func, &reg_ebx, &reg_ecx, &reg_edx);
    std::cout << "Processor: " << std::string((char*)&reg_ebx, 4) << std::string((char*)&reg_edx, 4) << std::string((char*)&reg_ecx, 4) << "\n";

    __get_cpuid(1, &reg_eax, &reg_ebx, &reg_ecx, &reg_edx);
    if((reg_ecx & SSE3) == SSE3)
        std::cout << "  SSE3 is supported\n";
    if((reg_ecx & SSSE3) == SSSE3)
        std::cout << "  SSSE3 is supported\n";
    if((reg_ecx & SSE41) == SSE41)
        std::cout << "  SSE41 is supported\n";
    if((reg_ecx & SSE42) == SSE42)
        std::cout << "  SSE42 is supported\n";
    if((reg_ecx & AVX_1) == AVX_1 && (reg_ecx & AVX_2) == AVX_2 && (reg_ecx & AVX_3) == AVX_3)
        std::cout << "  AVX is supported\n";

    if((reg_edx & MMX) == MMX)
        std::cout << "  MMX is supported\n";
    if((reg_edx & SSE) == SSE)
        std::cout << "  SSE is supported\n";
    if((reg_edx & SSE2) == SSE2)
        std::cout << "  SSE2 is supported\n";
    if((reg_edx & HYPER_T) == HYPER_T)
        std::cout << "  HYPER Threading is supported\n";

    if(max_func >= 0x80000000) {
        __get_cpuid(0x80000000, &reg_eax, &reg_ebx, &reg_ecx, &reg_edx);
        if((reg_edx & E3DNOW) == E3DNOW)
            std::cout << "  3DNow! is supported\n";
        if((reg_edx & E3DNOW2) == E3DNOW2)
            std::cout << "  3DNow!2 is supported\n";
    }

    if(max_func >= 0x80000001) {
        __get_cpuid(0x80000001, &reg_eax, &reg_ebx, &reg_ecx, &reg_edx);
        if((reg_ecx & SSE4A) == SSE4A)
            std::cout << "  SSE4A is supported\n";
        if((reg_ecx & SSE5) == SSE5)
            std::cout << "  SSE5 is supported\n";
    }
}

template<typename T>
void task_2_func(T* x, T* y, T* z, uint32_t n) {
    for(uint32_t i = 0; i < n; i++)
        x[i] = (y[i] >= 0 ? y[i] : -y[i]) + (z[i] >= 0 ? z[i] : -z[i]);
}

#if defined(__AVX2__)
#include <immintrin.h>
#include <vaesintrin.h>
#define SIMD_BITS 256
#define SIMD_INT __m256i
#define SIMD_FLOAT __m256
#define SIMD_DOUBLE __m256d
#define SIMD_LOAD(data) _mm256_loadu_si256(data)
#define SIMD_LOADF(data) _mm256_loadu_ps(data)
#define SIMD_LOADD(data) _mm256_loadu_pd(data)
#define SIMD_ABS_I8(data) _mm256_abs_epi8(data)
#define SIMD_ABS_I16(data) _mm256_abs_epi16(data)
#define SIMD_ABS_I32(data) _mm256_abs_epi32(data)
#define SIMD_ABS_I64(data) _mm256_blendv_epi8(data, _mm256_sub_epi64(_mm256_setzero_si256(), data), _mm256_cmpgt_epi64(_mm256_setzero_si256(), data))
#define SIMD_ABS_FLOAT(data) _mm256_andnot_ps(_mm256_set1_ps(-0.0), data)
#define SIMD_ABS_DOUBLE(data) _mm256_andnot_pd(_mm256_set1_pd(-0.0), data)
#define SIMD_ADD_I8(a, b) _mm256_add_epi8(a, b)
#define SIMD_ADD_I16(a, b) _mm256_add_epi16(a, b)
#define SIMD_ADD_I32(a, b) _mm256_add_epi32(a, b)
#define SIMD_ADD_I64(a, b) _mm256_add_epi64(a, b)
#define SIMD_ADD_FLOAT(a, b) _mm256_add_ps(a, b)
#define SIMD_ADD_DOUBLE(a, b) _mm256_add_pd(a, b)
#define SIMD_SQRT_FLOAT(data) _mm256_sqrt_ps(data)
#define SIMD_SQRT_DOUBLE(data) _mm256_sqrt_pd(data)
#define SIMD_STORE(out, data) _mm256_storeu_si256(out, data)
#define SIMD_STORE_FLOAT(out, data) _mm256_storeu_ps(out, data)
#define SIMD_STORE_DOUBLE(out, data) _mm256_storeu_pd(out, data)

#define SIMD_SETZERO _mm256_setzero_si256
#define SIMD_SLLI(data, bits) _mm256_slli_epi64(data, bits)
#define SIMD_SRLI(data, bits) _mm256_srli_epi64(data, bits)
#define SIMD_OR(data_a, data_b) _mm256_or_si256(data_a, data_b)
#elif defined(__SSE2__) && defined(__SSE4_2__)
#include <emmintrin.h>
#include <smmintrin.h>
#define SIMD_BITS 128
#define SIMD_INT __m128i
#define SIMD_FLOAT __m128
#define SIMD_DOUBLE __m128d
#define SIMD_LOAD(data) _mm_loadu_si128(data)
#define SIMD_LOADF(data) _mm_loadu_ps(data)
#define SIMD_LOADD(data) _mm_loadu_pd(data)
#define SIMD_ABS_I8(data) _mm_abs_epi8(data)
#define SIMD_ABS_I16(data) _mm_abs_epi16(data)
#define SIMD_ABS_I32(data) _mm_abs_epi32(data)
#define SIMD_ABS_I64(data) _mm_sub_epi64(data, _mm_and_si128(_mm_cmpgt_epi64(_mm_setzero_si128(), data), _mm_add_epi64(data, _mm_set1_epi64x(1))))
#define SIMD_ABS_FLOAT(data) _mm_andnot_ps(_mm_set_ps1(-0.0), data)
#define SIMD_ABS_DOUBLE(data) _mm_andnot_pd(_mm_set_pd1(-0.0), data)
#define SIMD_ADD_I8(a, b) _mm_add_epi8(a, b)
#define SIMD_ADD_I16(a, b) _mm_add_epi16(a, b)
#define SIMD_ADD_I32(a, b) _mm_add_epi32(a, b)
#define SIMD_ADD_I64(a, b) _mm_add_epi64(a, b)
#define SIMD_ADD_FLOAT(a, b) _mm_add_ps(a, b)
#define SIMD_ADD_DOUBLE(a, b) _mm_add_pd(a, b)
#define SIMD_SQRT_FLOAT(data) _mm_sqrt_ps(data)
#define SIMD_SQRT_DOUBLE(data) _mm_sqrt_pd(data)
#define SIMD_SUB_FLOAT(a, b) _mm_sub_ps(a, b)
#define SIMD_MUL_FLOAT(a, b) _mm_mul_ps(a, b)
#define SIMD_STORE(out, data) _mm_storeu_si128(out, data)
#define SIMD_STORE_FLOAT(out, data) _mm_storeu_ps(out, data)
#define SIMD_STORE_DOUBLE(out, data) _mm_storeu_pd(out, data)

#define SIMD_SETZERO _mm_setzero_si128
#define SIMD_SLLI(data, bits) _mm_slli_epi64(data, bits)
#define SIMD_SRLI(data, bits) _mm_srli_epi64(data, bits)
#define SIMD_OR(data_a, data_b) _mm_or_si128(data_a, data_b)
#endif

void task_2_func_simdeeznuts(int8_t* x, int8_t* y, int8_t* z, uint32_t n) {
    constexpr uint32_t batch_size = SIMD_BITS / (sizeof(int8_t) * 8);
    uint32_t i;
    for (i = 0; i + batch_size <= n; i += batch_size) {
        SIMD_INT y_simd = SIMD_LOAD((const SIMD_INT*)&y[i]);
        SIMD_INT z_simd = SIMD_LOAD((const SIMD_INT*)&z[i]);
        y_simd = SIMD_ABS_I8(y_simd);
        z_simd = SIMD_ABS_I8(z_simd);
        SIMD_INT sum = SIMD_ADD_I8(y_simd, z_simd);
        SIMD_STORE((SIMD_INT*)&x[i], sum);
    }
    for (; i < n; i++)
        x[i] = (y[i] >= 0 ? y[i] : -y[i]) + (z[i] >= 0 ? z[i] : -z[i]);
}

void task_2_func_simdeeznuts(int16_t* x, int16_t* y, int16_t* z, uint32_t n) {
    constexpr uint32_t batch_size = SIMD_BITS / (sizeof(int16_t) * 8);
    uint32_t i;
    for (i = 0; i + batch_size <= n; i += batch_size) {
        SIMD_INT y_simd = SIMD_LOAD((const SIMD_INT*)&y[i]);
        SIMD_INT z_simd = SIMD_LOAD((const SIMD_INT*)&z[i]);
        y_simd = SIMD_ABS_I16(y_simd);
        z_simd = SIMD_ABS_I16(z_simd);
        SIMD_INT sum = SIMD_ADD_I16(y_simd, z_simd);
        SIMD_STORE((SIMD_INT*)&x[i], sum);
    }
    for (; i < n; i++)
        x[i] = (y[i] >= 0 ? y[i] : -y[i]) + (z[i] >= 0 ? z[i] : -z[i]);
}

void task_2_func_simdeeznuts(int32_t* x, int32_t* y, int32_t* z, uint32_t n) {
    constexpr uint32_t batch_size = SIMD_BITS / (sizeof(int32_t) * 8);
    uint32_t i;
    for (i = 0; i + batch_size <= n; i += batch_size) {
        SIMD_INT y_simd = SIMD_LOAD((const SIMD_INT*)&y[i]);
        SIMD_INT z_simd = SIMD_LOAD((const SIMD_INT*)&z[i]);
        y_simd = SIMD_ABS_I32(y_simd);
        z_simd = SIMD_ABS_I32(z_simd);
        SIMD_INT sum = SIMD_ADD_I32(y_simd, z_simd);
        SIMD_STORE((SIMD_INT*)&x[i], sum);
    }
    for (; i < n; i++)
        x[i] = (y[i] >= 0 ? y[i] : -y[i]) + (z[i] >= 0 ? z[i] : -z[i]);
}

void task_2_func_simdeeznuts(int64_t* x, int64_t* y, int64_t* z, uint32_t n) {
    constexpr uint32_t batch_size = SIMD_BITS / (sizeof(int64_t) * 8);
    uint32_t i;
    for (i = 0; i + batch_size <= n; i += batch_size) {
        SIMD_INT y_simd = SIMD_LOAD((const SIMD_INT*)&y[i]);
        SIMD_INT z_simd = SIMD_LOAD((const SIMD_INT*)&z[i]);

        y_simd = SIMD_ABS_I64(y_simd);
        z_simd = SIMD_ABS_I64(z_simd);

        SIMD_INT sum = SIMD_ADD_I64(y_simd, z_simd);
        SIMD_STORE((SIMD_INT*)&x[i], sum);
    }
    for (; i < n; i++)
        x[i] = (y[i] >= 0 ? y[i] : -y[i]) + (z[i] >= 0 ? z[i] : -z[i]);
}

void task_2_func_simdeeznuts(float* x, float* y, float* z, uint32_t n) {
    constexpr uint32_t batch_size = SIMD_BITS / (sizeof(float) * 8);

    uint32_t i;
    for (i = 0; i + batch_size <= n; i += batch_size) {
        SIMD_FLOAT y_simd = SIMD_LOADF(&y[i]);
        SIMD_FLOAT z_simd = SIMD_LOADF(&z[i]);
        y_simd = SIMD_ABS_FLOAT(y_simd);
        z_simd = SIMD_ABS_FLOAT(z_simd);
        SIMD_FLOAT sum = SIMD_ADD_FLOAT(y_simd, z_simd);
        SIMD_STORE_FLOAT(&x[i], sum);
    }
    for (; i < n; i++)
        x[i] = (y[i] >= 0 ? y[i] : -y[i]) + (z[i] >= 0 ? z[i] : -z[i]);
}

void task_2_func_simdeeznuts(double* x, double* y, double* z, uint32_t n) {
    constexpr uint32_t batch_size = SIMD_BITS / (sizeof(double) * 8);

    uint32_t i;
    for (i = 0; i + batch_size <= n; i += batch_size) {
        SIMD_DOUBLE y_simd = SIMD_LOADD(&y[i]);
        SIMD_DOUBLE z_simd = SIMD_LOADD(&z[i]);
        y_simd = SIMD_ABS_DOUBLE(y_simd);
        z_simd = SIMD_ABS_DOUBLE(z_simd);
        SIMD_DOUBLE sum = SIMD_ADD_DOUBLE(y_simd, z_simd);
        SIMD_STORE_DOUBLE(&x[i], sum);
    }
    for (; i < n; i++)
        x[i] = (y[i] >= 0 ? y[i] : -y[i]) + (z[i] >= 0 ? z[i] : -z[i]);
}

template<typename T>
void task_2_measure_simd_nosimd(const std::string& type_name, T* x, T* y, T* z, void(*simd_func)(T*, T*, T*, uint32_t)) {
    double start_time = omp_get_wtime();
    task_2_func<T>(x, y, z, arr_size);
    double nosimd_time = omp_get_wtime() - start_time;
    std::cout << "[" << type_name << "] Time (no simd): " << std::fixed << std::setprecision(6) << nosimd_time << " seconds\n";

    start_time = omp_get_wtime();
    simd_func(x, y, z, arr_size);
    double simd_time = omp_get_wtime() - start_time;
    std::cout << "[" << type_name << "] Time (simd): " << std::fixed << std::setprecision(6) << simd_time << " seconds\n";
    std::cout << "[" << type_name << "] SIMD is " << std::fixed << std::setprecision(2) << nosimd_time / simd_time << " times faster\n";
}

void task_2_int8() {
    auto* x = new int8_t[arr_size];
    auto* y = new int8_t[arr_size];
    auto* z = new int8_t[arr_size];
    for(int i = 0; i < arr_size; i++) {
        y[i] = rand() - RAND_MAX / 2;
        z[i] = rand() - RAND_MAX / 2;
    }

    task_2_measure_simd_nosimd<int8_t>("int8", x, y, z, task_2_func_simdeeznuts);

    delete[] x;
    delete[] y;
    delete[] z;
}

void task_2_int16() {
    auto* x = new int16_t[arr_size];
    auto* y = new int16_t[arr_size];
    auto* z = new int16_t[arr_size];
    for(int i = 0; i < arr_size; i++) {
        y[i] = rand() - RAND_MAX / 2;
        z[i] = rand() - RAND_MAX / 2;
    }

    task_2_measure_simd_nosimd<int16_t>("int16", x, y, z, task_2_func_simdeeznuts);

    delete[] x;
    delete[] y;
    delete[] z;
}

void task_2_int32() {
    auto* x = new int32_t[arr_size];
    auto* y = new int32_t[arr_size];
    auto* z = new int32_t[arr_size];
    for(int i = 0; i < arr_size; i++) {
        y[i] = rand() - RAND_MAX / 2;
        z[i] = rand() - RAND_MAX / 2;
    }

    task_2_measure_simd_nosimd<int32_t>("int32", x, y, z, task_2_func_simdeeznuts);

    delete[] x;
    delete[] y;
    delete[] z;
}

void task_2_int64() {
    auto* x = new int64_t[arr_size];
    auto* y = new int64_t[arr_size];
    auto* z = new int64_t[arr_size];
    for(int i = 0; i < arr_size; i++) {
        y[i] = rand() - RAND_MAX / 2;
        z[i] = rand() - RAND_MAX / 2;
    }

    task_2_measure_simd_nosimd<int64_t>("int64", x, y, z, task_2_func_simdeeznuts);

    delete[] x;
    delete[] y;
    delete[] z;
}

void task_2_float() {
    std::uniform_real_distribution<float> randfloat;

    auto* x = new float[arr_size];
    auto* y = new float[arr_size];
    auto* z = new float[arr_size];
    for(int i = 0; i < arr_size; i++) {
        y[i] = randfloat(rand_eng);
        z[i] = randfloat(rand_eng);
    }

    task_2_measure_simd_nosimd<float>("float", x, y, z, task_2_func_simdeeznuts);

    delete[] x;
    delete[] y;
    delete[] z;
}

void task_2_double() {
    std::uniform_real_distribution<double> randdouble;

    auto* x = new double[arr_size];
    auto* y = new double[arr_size];
    auto* z = new double[arr_size];
    for(int i = 0; i < arr_size; i++) {
        y[i] = randdouble(rand_eng);
        z[i] = randdouble(rand_eng);
    }

    task_2_measure_simd_nosimd<double>("double", x, y, z, task_2_func_simdeeznuts);

    delete[] x;
    delete[] y;
    delete[] z;
}

void task_2() {
    task_2_int8();
    task_2_int16();
    task_2_int32();
    task_2_int64();
    task_2_float();
    task_2_double();
}

template<typename T>
void task_6_func(T* x, T* y, uint32_t n) {
    for(uint32_t i = 0; i < n; i++)
        x[i] = std::sqrt(y[i]);
}

void task_6_func_simdeeznuts(float* x, float* y, uint32_t n) {
    constexpr uint32_t batch_size = SIMD_BITS / (sizeof(float) * 8);

    uint32_t i;
    for (i = 0; i + batch_size <= n; i += batch_size) {
        SIMD_FLOAT y_simd = SIMD_LOADF(&y[i]);
        SIMD_STORE_FLOAT(&x[i], SIMD_SQRT_FLOAT(y_simd));
    }
    for (; i < n; i++)
        x[i] = std::sqrt(y[i]);
}

void task_6_func_simdeeznuts(double* x, double* y, uint32_t n) {
    constexpr uint32_t batch_size = SIMD_BITS / (sizeof(double) * 8);

    uint32_t i;
    for (i = 0; i + batch_size <= n; i += batch_size) {
        SIMD_DOUBLE y_simd = SIMD_LOADD(&y[i]);
        SIMD_STORE_DOUBLE(&x[i], SIMD_SQRT_DOUBLE(y_simd));
    }
    for (; i < n; i++)
        x[i] = std::sqrt(y[i]);
}

template<typename T>
void task_6_measure_simd_nosimd(const std::string& type_name, T* x, T* y, void(*simd_func)(T* x, T* y, uint32_t n)) {
    double start_time = omp_get_wtime();
    task_6_func<T>(x, y, arr_size);
    double nosimd_time = omp_get_wtime() - start_time;
    std::cout << "[" << type_name << "] Time (no simd): " << std::fixed << std::setprecision(6) << nosimd_time << " seconds\n";

    start_time = omp_get_wtime();
    simd_func(x, y, arr_size);
    double simd_time = omp_get_wtime() - start_time;
    std::cout << "[" << type_name << "] Time (simd): " << std::fixed << std::setprecision(6) << simd_time << " seconds\n";
    std::cout << "[" << type_name << "] SIMD is " << std::fixed << std::setprecision(2) << nosimd_time / simd_time << " times faster\n";
}

void task_6_float() {
    std::uniform_real_distribution<float> randfloat;

    auto* x = new float[arr_size];
    auto* y = new float[arr_size];
    for(int i = 0; i < arr_size; i++) {
        y[i] = randfloat(rand_eng);
    }

    task_6_measure_simd_nosimd<float>("float", x, y, task_6_func_simdeeznuts);

    delete[] x;
    delete[] y;
}

void task_6_double() {
    std::uniform_real_distribution<double> randdouble;

    auto* x = new double[arr_size];
    auto* y = new double[arr_size];
    for(int i = 0; i < arr_size; i++) {
        y[i] = randdouble(rand_eng);
    }

    task_6_measure_simd_nosimd<double>("double", x, y, task_6_func_simdeeznuts);

    delete[] x;
    delete[] y;
}

void task_6() {
    task_6_float();
    task_6_double();
}

void task_8_func(std::complex<float>* x, std::complex<float>* y, std::complex<float>* z, uint32_t n) {
    for(uint32_t i = 0; i < n; i++)
        x[i] = y[i] * z[i];
}

void task_8_func_simdeeznuts(std::complex<float>* x, std::complex<float>* y, std::complex<float>* z, uint32_t n) {
    constexpr uint32_t batch_size = SIMD_BITS / (sizeof(float) * 8);

    uint32_t i;
    for (i = 0; i + batch_size <= n; i += batch_size) {
        const SIMD_FLOAT y_simd_real = SIMD_LOADF(reinterpret_cast<const float*>(&y[i]) + 0);
        const SIMD_FLOAT y_simd_imag = SIMD_LOADF(reinterpret_cast<const float*>(&y[i]) + 1);
        const SIMD_FLOAT z_simd_real = SIMD_LOADF(reinterpret_cast<const float*>(&z[i]) + 0);
        const SIMD_FLOAT z_simd_imag = SIMD_LOADF(reinterpret_cast<const float*>(&z[i]) + 1);

        const SIMD_FLOAT x_real = SIMD_SUB_FLOAT(SIMD_MUL_FLOAT(y_simd_real, z_simd_real), SIMD_MUL_FLOAT(y_simd_imag, z_simd_imag));
        const SIMD_FLOAT x_imag = SIMD_ADD_FLOAT(SIMD_MUL_FLOAT(y_simd_real, z_simd_imag), SIMD_MUL_FLOAT(y_simd_imag, z_simd_real));

        SIMD_STORE_FLOAT(reinterpret_cast<float*>(&x[i]) + 0, x_real);
        SIMD_STORE_FLOAT(reinterpret_cast<float*>(&x[i]) + 1, x_imag);
    }
    for (; i < n; i++)
        x[i] = y[i] * z[i];
}

void task_8() {
    std::uniform_real_distribution<float> randfloat;

    auto* x = new std::complex<float>[arr_size];
    auto* y = new std::complex<float>[arr_size];
    auto* z = new std::complex<float>[arr_size];
    for(int i = 0; i < arr_size; i++) {
        y[i] = {randfloat(rand_eng), randfloat(rand_eng)};
        z[i] = {randfloat(rand_eng), randfloat(rand_eng)};
    }

    memset(x, 0, arr_size * sizeof(std::complex<float>));
    double start_time = omp_get_wtime();
    task_8_func(x, y, z, arr_size);
    double nosimd_time = omp_get_wtime() - start_time;
    std::cout << "[float-complex] Time (no simd): " << std::fixed << std::setprecision(6) << nosimd_time << " seconds, x[0] = " << x[0] << "\n";

    memset(x, 0, arr_size * sizeof(std::complex<float>));
    start_time = omp_get_wtime();
    task_8_func_simdeeznuts(x, y, z, arr_size);
    double simd_time = omp_get_wtime() - start_time;
    std::cout << "[float-complex] Time (simd): " << std::fixed << std::setprecision(6) << simd_time << " seconds, x[0] = " << x[0] << "\n";
    std::cout << "[float-complex] SIMD is " << std::fixed << std::setprecision(2) << nosimd_time / simd_time << " times faster\n";

    delete x;
    delete y;
    delete z;
}

void task_9_print(SIMD_INT* simd_ints) {
    auto* ints = (uint64_t*)simd_ints;
    std::cout << "512-bit data: ";
    for(int i = 0; i < 8; i++) {
        std::cout << std::bitset<64>(ints[i]);
    }

    std::cout << "\n";
}

void task_9() {
    uint64_t data[8] = {(uint64_t)0b11 << 62, 0, 0, 0, 0, 0, 0};
    SIMD_INT simd_data[512/SIMD_BITS];

    task_9_print((SIMD_INT*)&data[0]);

    double start_time = omp_get_wtime();
    for(int i = 0; i < 100000000; i++) {
        for (int j = 0; j < 512/SIMD_BITS; j++) {
            simd_data[j] = SIMD_LOAD((SIMD_INT*)(&data[j * (SIMD_BITS / 64)]));
        }

        for (int j = 0; j < 512/SIMD_BITS; j++) {
            SIMD_INT shifted = SIMD_SLLI(simd_data[j], 1);
            SIMD_INT high_bits = SIMD_SRLI(simd_data[j], 63);

            if (j < (512/SIMD_BITS - 1)) {
                simd_data[j + 1] = SIMD_OR(simd_data[j + 1], high_bits);
            }

            simd_data[j] = shifted;
        }
    }

    double time_taken = omp_get_wtime() - start_time;
    std::cout << "[" << SIMD_BITS << "] Time: " << std::fixed << std::setprecision(6) << time_taken << " seconds\n";

    std::cout << "After 1-bit left shift:" << std::endl;
    task_9_print(&simd_data[0]);

}

int main() {
    std::cout << "Task 1\n";
    task_1();
    std::cout << "\n";

    std::cout << "Task 2\n";
    task_2();
    std::cout << "\n";

    std::cout << "Task 6\n";
    task_6();
    std::cout << "\n";

    std::cout << "Task 8\n";
    task_8();
    std::cout << "\n";

    std::cout << "Task 9\n";
    task_9();
    std::cout << "\n";

    return 0;
}
