#include <iostream>
#include <cpuid.h>
#include <cstdint>
#include <iomanip>
#include <omp.h>
#include <random>

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

#if defined(__SSE2__) && defined(__SSE4_2__)
#include <emmintrin.h>
#include <smmintrin.h>

void task_2_func_simdeeznuts(int8_t* x, int8_t* y, int8_t* z, uint32_t n) {
    constexpr uint32_t batch_size = 128 / (sizeof(int8_t) * 8);
    uint32_t i;
    for (i = 0; i + batch_size <= n; i += batch_size) {
        __m128i y_simd = _mm_loadu_si128((const __m128i*)&y[i]);
        __m128i z_simd = _mm_loadu_si128((const __m128i*)&z[i]);
        y_simd = _mm_abs_epi8(y_simd);
        z_simd = _mm_abs_epi8(z_simd);
        __m128i sum = _mm_add_epi8(y_simd, z_simd);
        _mm_storeu_si128((__m128i*)&x[i], sum);
    }
    for (; i < n; i++)
        x[i] = (y[i] >= 0 ? y[i] : -y[i]) + (z[i] >= 0 ? z[i] : -z[i]);
}

void task_2_func_simdeeznuts(int16_t* x, int16_t* y, int16_t* z, uint32_t n) {
    constexpr uint32_t batch_size = 128 / (sizeof(int16_t) * 8);
    uint32_t i;
    for (i = 0; i + batch_size <= n; i += batch_size) {
        __m128i y_simd = _mm_loadu_si128((const __m128i*)&y[i]);
        __m128i z_simd = _mm_loadu_si128((const __m128i*)&z[i]);
        y_simd = _mm_abs_epi16(y_simd);
        z_simd = _mm_abs_epi16(z_simd);
        __m128i sum = _mm_add_epi16(y_simd, z_simd);
        _mm_storeu_si128((__m128i*)&x[i], sum);
    }
    for (; i < n; i++)
        x[i] = (y[i] >= 0 ? y[i] : -y[i]) + (z[i] >= 0 ? z[i] : -z[i]);
}

void task_2_func_simdeeznuts(int32_t* x, int32_t* y, int32_t* z, uint32_t n) {
    constexpr uint32_t batch_size = 128 / (sizeof(int32_t) * 8);
    uint32_t i;
    for (i = 0; i + batch_size <= n; i += batch_size) {
        __m128i y_simd = _mm_loadu_si128((const __m128i*)&y[i]);
        __m128i z_simd = _mm_loadu_si128((const __m128i*)&z[i]);
        y_simd = _mm_abs_epi32(y_simd);
        z_simd = _mm_abs_epi32(z_simd);
        __m128i sum = _mm_add_epi32(y_simd, z_simd);
        _mm_storeu_si128((__m128i*)&x[i], sum);
    }
    for (; i < n; i++)
        x[i] = (y[i] >= 0 ? y[i] : -y[i]) + (z[i] >= 0 ? z[i] : -z[i]);
}

void task_2_func_simdeeznuts(int64_t* x, int64_t* y, int64_t* z, uint32_t n) {
    constexpr uint32_t batch_size = 128 / (sizeof(int64_t) * 8);
    uint32_t i;
    for (i = 0; i + batch_size <= n; i += batch_size) {
        __m128i y_simd = _mm_loadu_si128((const __m128i*)&y[i]);
        __m128i z_simd = _mm_loadu_si128((const __m128i*)&z[i]);

        y_simd = _mm_sub_epi64(y_simd, _mm_and_si128(_mm_cmpgt_epi64(_mm_setzero_si128(), y_simd), _mm_add_epi64(y_simd, _mm_set1_epi64x(1))));
        z_simd = _mm_sub_epi64(z_simd, _mm_and_si128(_mm_cmpgt_epi64(_mm_setzero_si128(), z_simd), _mm_add_epi64(z_simd, _mm_set1_epi64x(1))));

        __m128i sum = _mm_add_epi64(y_simd, z_simd);
        _mm_storeu_si128((__m128i*)&x[i], sum);
    }
    for (; i < n; i++)
        x[i] = (y[i] >= 0 ? y[i] : -y[i]) + (z[i] >= 0 ? z[i] : -z[i]);
}

void task_2_func_simdeeznuts(float* x, float* y, float* z, uint32_t n) {
    constexpr uint32_t batch_size = 128 / (sizeof(float) * 8);
    __m128 abs_mask = _mm_set_ps1(-0.0);

    uint32_t i;
    for (i = 0; i + batch_size <= n; i += batch_size) {
        __m128 y_simd = _mm_load_ps(&y[i]);
        __m128 z_simd = _mm_load_ps(&z[i]);
        y_simd = _mm_andnot_ps(abs_mask, y_simd);
        z_simd = _mm_andnot_ps(abs_mask, z_simd);
        __m128 sum = _mm_add_ps(y_simd, z_simd);
        _mm_store_ps(&x[i], sum);
    }
    for (; i < n; i++)
        x[i] = (y[i] >= 0 ? y[i] : -y[i]) + (z[i] >= 0 ? z[i] : -z[i]);
}

void task_2_func_simdeeznuts(double* x, double* y, double* z, uint32_t n) {
    constexpr uint32_t batch_size = 128 / (sizeof(double) * 8);
    __m128d abs_mask = _mm_set_pd1(-0.0);

    uint32_t i;
    for (i = 0; i + batch_size <= n; i += batch_size) {
        __m128d y_simd = _mm_load_pd(&y[i]);
        __m128d z_simd = _mm_load_pd(&z[i]);
        y_simd = _mm_andnot_pd(abs_mask, y_simd);
        z_simd = _mm_andnot_pd(abs_mask, z_simd);
        __m128d sum = _mm_add_pd(y_simd, z_simd);
        _mm_store_pd(&x[i], sum);
    }
    for (; i < n; i++)
        x[i] = (y[i] >= 0 ? y[i] : -y[i]) + (z[i] >= 0 ? z[i] : -z[i]);
}
#endif

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
    std::cout << "[" << type_name << "] SIMD (SSE) is " << std::fixed << std::setprecision(2) << nosimd_time / simd_time << " times faster\n";
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

#if defined(__SSE2__) && defined(__SSE4_2__)
#include <emmintrin.h>
#include <smmintrin.h>

void task_6_func_simdeeznuts(float* x, float* y, uint32_t n) {
    constexpr uint32_t batch_size = 128 / (sizeof(float) * 8);

    uint32_t i;
    for (i = 0; i + batch_size <= n; i += batch_size) {
        __m128 y_simd = _mm_load_ps(&y[i]);
        _mm_store_ps(&x[i], _mm_sqrt_ps(y_simd));
    }
    for (; i < n; i++)
        x[i] = std::sqrt(y[i]);
}

void task_6_func_simdeeznuts(double* x, double* y, uint32_t n) {
    constexpr uint32_t batch_size = 128 / (sizeof(double) * 8);

    uint32_t i;
    for (i = 0; i + batch_size <= n; i += batch_size) {
        __m128d y_simd = _mm_load_pd(&y[i]);
        _mm_store_pd(&x[i], _mm_sqrt_pd(y_simd));
    }
    for (; i < n; i++)
        x[i] = std::sqrt(y[i]);
}
#endif

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
    std::cout << "[" << type_name << "] SIMD (SSE) is " << std::fixed << std::setprecision(2) << nosimd_time / simd_time << " times faster\n";
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

int main() {
    task_1();
    std::cout << "\n";

    task_2();
    std::cout << "\n";

    task_6();
    std::cout << "\n";
    return 0;
}
