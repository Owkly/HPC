#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <getopt.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;

double cutoff = 500;
u64 seed = 0;
u64 size = 0;
char *filename = NULL;

#if defined(__x86_64__) || defined(__i386__)
#if defined(__AVX2__)
#include <immintrin.h>

/*****************************************************************************/
/*                    pseudo-random function (SPECK-like)                    */
/*****************************************************************************/

#define ROR(x, r) ((x >> r) | (x << (64 - r)))
#define ROL(x, r) ((x << r) | (x >> (64 - r)))
#define R(x, y, k) (x = ROR(x, 8), x += y, x ^= k, y = ROL(y, 3), y ^= x)
u64 PRF(u64 seed, u64 IV, u64 i)
{
    u64 y = i;
    u64 x = 0xBaadCafeDeadBeefULL;
    u64 b = IV;
    u64 a = seed;
    R(x, y, b);

    // #pragma omp parallel for (ralentit le code)
    #pragma omp simd
    for (int i = 0; i < 32; i++)
    {
        R(a, b, i);
        R(x, y, b);
    }
    return x + i;
}

/************************** Fast Fourier Transform ***************************/
/*              This code assumes that n is a power of two !!!               */
/*****************************************************************************/

void FFT_rec(u64 n, const double complex *X, double complex *Y, u64 stride)
{

    if (n == 1)
    {
        Y[0] = X[0];
        return;
    }
    double complex omega_n = cexp(-2 * I * M_PI / n);
    double complex omega = 1;

    if (n > 128)
    {
        #pragma omp task
        FFT_rec(n / 2, X, Y, 2 * stride);

        #pragma omp task
        FFT_rec(n / 2, X + stride, Y + n / 2, 2 * stride);
    }

    else
    {
        FFT_rec(n / 2, X, Y, 2 * stride);
        FFT_rec(n / 2, X + stride, Y + n / 2, 2 * stride);
    }

    #pragma omp taskwait
    // on prélève les parties réelles et imaginaires de omega_n et omega
    // _omega[0] = creal(omega_n)
    // _omega[1] = cimag(omega_n)
    // _omega[2] = creal(omega)
    // _omega[3] = cimag(omega)
    __m256d _omega = _mm256_setr_pd(creal(omega_n), cimag(omega_n), creal(omega), cimag(omega));
    // #pragma omp parallel for (ralentit le code)
    for (u64 i = 0; i < n / 2; i++)
    {
        // pour chaque i, on prélève p = Y[i]; et q = Y[i + n / 2] * omega;
        // Rappel :
        // z1 = a1 + b1i
        // z2 = a2 + b2i
        // z1 * z2 = (a1a2 - b1b2) + (a1b2 + a2b1)i
        // p_q[0] = creal(Y[i])
        // p_q[1] = cimag(Y[i])
        // p_q[2] = creal(Y[i + n / 2]) * _omega[2] - cimag(Y[i + n / 2]) * _omega[3]
        // p_q[3] = creal(Y[i + n / 2]) * _omega[3] + cimag(Y[i + n / 2]) * _omega[2]
        __m256d p_q = _mm256_setr_pd(creal(Y[i]), cimag(Y[i]),
                                     creal(Y[i + n / 2]) * _omega[2] - cimag(Y[i + n / 2]) * _omega[3],
                                     creal(Y[i + n / 2]) * _omega[3] + cimag(Y[i + n / 2]) * _omega[2]);

        // on calcule Y = p + q et  Yn2 = p - q
        // Y_Yn2[0] = creal(p + q)
        // Y_Yn2[1] = cimag(p + q)
        // Y_Yn2[2] = creal(p - q)
        // Y_Yn2[3] = cimag(p - q)
        __m256d Y_Yn2 = _mm256_setr_pd(p_q[0] + p_q[2], p_q[1] + p_q[3], p_q[0] - p_q[2], p_q[1] - p_q[3]);


        // Stockage de p + q dans Y[i] et p - q dans Y[i + n/2]
        Y[i] = Y_Yn2[0] + Y_Yn2[1] * I;
        Y[i + n / 2] = Y_Yn2[2] + Y_Yn2[3] * I;

        // Mise à jour de omega
        _omega = _mm256_setr_pd(_omega[0], _omega[1],
                                _omega[2] * _omega[0] - _omega[3] * _omega[1],
                                _omega[2] * _omega[1] + _omega[3] * _omega[0]);
    }
}

void FFT(u64 n, const double complex *X, double complex *Y)
{
    if ((n & (n - 1)) != 0)
        errx(1, "size is not a power of two (this code does not handle other cases)");
    #pragma omp parallel
    {
        #pragma omp single
        FFT_rec(n, X, Y, 1);
    }
}

void iFFT(u64 n, double complex *X, double complex *Y)
{
    // X est un tableau de double complex, en C en elle est représenté par un tableau de double
    // avec la partie réelle à l'indice pair et la partie imaginaire à l'indice impair
    #pragma omp parallel for
    for (u64 i = 0; i < n; i += 2)
    {
        // on charge 4 valeurs consécutives à partir de l'adresse (double *)(X + i) dans _X (on charge donc 2 valeurs complexes)
        __m256d _X = _mm256_loadu_pd((double *)(X + i));
        // On multiplie les 4 valeurs de _X par le vecteur _mm256_set_pd(1, -1, 1, -1)
        // ce qui revient à multiplier par -1 les valeurs des parties imaginaires (donc à calculer la conjuguée)
        _X = _mm256_mul_pd(_X, _mm256_set_pd(1, -1, 1, -1));
        // on stocke les 4 nouvelles valeurs de manière consécutive à l'adresse (double *)(X + i)
        _mm256_storeu_pd((double *)(X + i), _X);
    }

    FFT(n, X, Y);

    #pragma omp parallel for 
    for (u64 i = 0; i < n; i += 2)
    {
        // on charge 4 valeurs consécutives à partir de l'adresse (double *)(Y + i) dans _Y (on charge donc 2 valeurs complexes)
        __m256d _Y = _mm256_loadu_pd((double *)(Y + i));
        // On multiplie les 4 valeurs de _Y par le vecteur _mm256_set_pd(1, -1, 1, -1)
        // ce qui revient à multiplier par -1 les valeurs des parties imaginaires (donc à calculer la conjuguée)
        _Y = _mm256_mul_pd(_Y, _mm256_set_pd(1, -1, 1, -1));
        // on divise les 4 valeurs de _Y par n
        _Y = _mm256_div_pd(_Y, _mm256_set1_pd(n));
        // on stocke les 4 nouvelles valeurs de manière consécutive à l'adresse (double *)(Y + i)
        _mm256_storeu_pd((double *)(Y + i), _Y);
    }
}

/***************************** utility functions *************************************/
/*          Défini les des fonctions qui seront utile tel que un timer               */
/*        Une fonction qui véréfie les options de la commande d'éxécution            */
/* Une fonction qui enregistre les données d'un tableau complexe dans un fichier WAV */
/*************************************************************************************/

double wtime()
{
    struct timeval ts;
    gettimeofday(&ts, NULL);
    return (double)ts.tv_sec + ts.tv_usec / 1e6;
}

void process_command_line_options(int argc, char **argv)
{
    struct option longopts[5] = {
        {"size", required_argument, NULL, 'n'},
        {"seed", required_argument, NULL, 's'},
        {"output", required_argument, NULL, 'o'},
        {"cutoff", required_argument, NULL, 'c'},
        {NULL, 0, NULL, 0}};
    char ch;
    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1)
    {
        switch (ch)
        {
            case 'n': size = atoll(optarg); break;
            case 's': seed = atoll(optarg); break;
            case 'o': filename = optarg; break;
            case 'c': cutoff = atof(optarg); break;
            default: errx(1, "Unknown option\n");
        }
    }
    if (size == 0)
        errx(1, "missing --size argument");
}

void save_WAV(char *filename, u64 size, double complex *C)
{

    assert(size < 1000000000);
    FILE *f = fopen(filename, "w");
    if (f == NULL)
        err(1, "fopen");
    printf("Writing <= 10s of audio output in %s\n", filename);
    u32 rate = 44100;
    u32 frame_count = 10 * rate;
    if (size < frame_count)
        frame_count = size;
    u16 chan_num = 2;
    u16 bits = 16;
    u32 length = frame_count * chan_num * bits / 8;
    u16 byte;
    double multiplier = 32767;

    fwrite("RIFF", 1, 4, f);
    u32 chunk_size = length + 44 - 8;
    fwrite(&chunk_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    u32 subchunk1_size = 16;
    fwrite(&subchunk1_size, 4, 1, f);
    u16 fmt_type = 1;
    fwrite(&fmt_type, 2, 1, f);
    fwrite(&chan_num, 2, 1, f);
    fwrite(&rate, 4, 1, f);

    uint32_t byte_rate = rate * bits * chan_num / 8;
    fwrite(&byte_rate, 4, 1, f);
    uint16_t block_align = chan_num * bits / 8;
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits, 2, 1, f);

    fwrite("data", 1, 4, f);
    fwrite(&length, 4, 1, f);
    for (u32 i = 0; i < frame_count; i++)
    {
        byte = creal(C[i]) * multiplier;
        fwrite(&byte, 2, 1, f);
        byte = cimag(C[i]) * multiplier;
        fwrite(&byte, 2, 1, f);
    }
    fclose(f);
}

/***************************************************************************/
/*                           main function                                 */
/***************************************************************************/

int main(int argc, char **argv)
{   
    printf("Number of processors: %d\n", omp_get_num_procs());
    printf("Using %d threads\n", omp_get_max_threads());
    double t0 = wtime();
    process_command_line_options(argc, argv);

    double complex *A = malloc(size * sizeof(*A));
    double complex *B = malloc(size * sizeof(*B));
    double complex *C = malloc(size * sizeof(*C));

    printf("Generating white noise...\n");
    #pragma omp parallel for simd
    for (u64 i = 0; i < size; i++)
    {
        double real = 2 * (PRF(seed, 0, i) * 5.42101086242752217e-20) - 1;
        double imag = 2 * (PRF(seed, 1, i) * 5.42101086242752217e-20) - 1;
        A[i] = real + imag * I;
    }

    printf("Forward FFT...\n");
    FFT(size, A, B);

    printf("Adjusting Fourier coefficients...\n");
    #pragma omp parallel for simd
    for (u64 i = 0; i < size; i++)
    {
        double tmp = sin(i * 2 * M_PI / 44100);
        B[i] *= tmp * cexp(-i * 2 * I * M_PI / 4 / 44100);
        B[i] *= (i + 1) / exp((i * cutoff) / size);
    }

    printf("Inverse FFT...\n");
    iFFT(size, B, C);

    printf("Normalizing output...\n");
    double max = 0;
    #pragma omp parallel for reduction(max : max)
    for (u64 i = 0; i < size; i++)
        max = fmax(max, cabs(C[i]));
    printf("max = %g\n", max);

    // Normalisation avec AVX
    // charge 4 fois la valeur max dans un vecteur
    __m256d max_vec = _mm256_set1_pd(max);
    #pragma omp parallel for
    for (u64 i = 0; i < size; i += 2)
    {
        // sachant que C est un tableau de double complex, donc un tableau de double
        // avec la partie réelle à l'indice pair et la partie imaginaire à l'indice impair

        // charge 4 valeur consécutives à partir de l'adresse (double *)(C + i) dans c_val (on charge donc 2 valeurs complexes)
        __m256d c_val = _mm256_loadu_pd((double *)(C + i));
        // on divise les 4 valeurs de c_val par max_vec
        c_val = _mm256_div_pd(c_val, max_vec);
        // on stocke les 4 nouvelles valeurs de manière consécutive à l'adresse (double *)(C + i)
        _mm256_storeu_pd((double *)(C + i), c_val);
    }

    if (filename != NULL)
        save_WAV(filename, size, C);

    double t1 = wtime();
    printf("Time: %g\n", t1 - t0);
    exit(EXIT_SUCCESS);
}
#endif
#elif defined(__ARM_NEON) // Pour les architectures ARM avec NEON
    #include <arm_neon.h>
/*****************************************************************************/
/*                    pseudo-random function (SPECK-like)                    */
/*****************************************************************************/

#define ROR(x, r) ((x >> r) | (x << (64 - r)))
#define ROL(x, r) ((x << r) | (x >> (64 - r)))
#define R(x, y, k) (x = ROR(x, 8), x += y, x ^= k, y = ROL(y, 3), y ^= x)
u64 PRF(u64 seed, u64 IV, u64 i)
{
    u64 y = i;
    u64 x = 0xBaadCafeDeadBeefULL;
    u64 b = IV;
    u64 a = seed;
    R(x, y, b);

    for (int i = 0; i < 32; i++)
    {
        R(a, b, i);
        R(x, y, b);
    }
    return x + i;
}

/************************** Fast Fourier Transform ***************************/
/*              This code assumes that n is a power of two !!!               */
/*****************************************************************************/


void FFT_rec(u64 n, const double complex *X, double complex *Y, u64 stride) {
    if (n == 1) {
        Y[0] = X[0];
        return;
    }

    // Calcul des racines de l'unité
    double complex omega_n = cexp(-2 * I * M_PI / n);
    double complex omega = 1;

    // Diviser pour régner
    if (n > 128) {
        #pragma omp task
        FFT_rec(n / 2, X, Y, 2 * stride);
        #pragma omp task
        FFT_rec(n / 2, X + stride, Y + n / 2, 2 * stride);
        #pragma omp taskwait
    } else {
        FFT_rec(n / 2, X, Y, 2 * stride);
        FFT_rec(n / 2, X + stride, Y + n / 2, 2 * stride);
    }

    // Combinaison
    for (u64 i = 0; i < n / 2; i++) {
        double complex p = Y[i];
        double complex q = Y[i + n / 2] * omega;

        float64x2_t p_vector = vld1q_f64((double *)&p);
        float64x2_t q_vector = vld1q_f64((double *)&q);

        float64x2_t result_plus = vaddq_f64(p_vector, q_vector);
        float64x2_t result_minus = vsubq_f64(p_vector, q_vector);

        vst1q_f64((double *)(Y + i), result_plus);
        vst1q_f64((double *)(Y + i + n / 2), result_minus);

        omega *= omega_n;
    }
}



void FFT(u64 n, const double complex *X, double complex *Y)
{
    if ((n & (n - 1)) != 0)
        errx(1, "size is not a power of two (this code does not handle other cases)");
    #pragma omp parallel
    {
        #pragma omp single
        FFT_rec(n, X, Y, 1);
    }
}


void iFFT(u64 n, double complex *X, double complex *Y)
{
    #pragma omp parallel for
    for (u64 i = 0; i < n; i++)
        X[i] = conj(X[i]);

    FFT(n, X, Y);

    #pragma omp parallel for
    for (u64 i = 0; i < n; i++)
        Y[i] = conj(Y[i]) / n;
}






/***************************** utility functions *************************************/
/*          Défini les des fonctions qui seront utile tel que un timer               */
/*        Une fonction qui véréfie les options de la commande d'éxécution            */
/* Une fonction qui enregistre les données d'un tableau complexe dans un fichier WAV */
/*************************************************************************************/

double wtime()
{
    struct timeval ts;
    gettimeofday(&ts, NULL);
    return (double)ts.tv_sec + ts.tv_usec / 1e6;
}

void process_command_line_options(int argc, char **argv)
{
    struct option longopts[5] = {
        {"size", required_argument, NULL, 'n'},
        {"seed", required_argument, NULL, 's'},
        {"output", required_argument, NULL, 'o'},
        {"cutoff", required_argument, NULL, 'c'},
        {NULL, 0, NULL, 0}};
    char ch;
    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1)
    {
        switch (ch)
        {
            case 'n': size = atoll(optarg); break;
            case 's': seed = atoll(optarg); break;
            case 'o': filename = optarg; break;
            case 'c': cutoff = atof(optarg); break;
            default: errx(1, "Unknown option\n");
        }
    }
    if (size == 0)
        errx(1, "missing --size argument");
}

void save_WAV(char *filename, u64 size, double complex *C)
{

    assert(size < 1000000000);
    FILE *f = fopen(filename, "w");
    if (f == NULL)
        err(1, "fopen");
    printf("Writing <= 10s of audio output in %s\n", filename);
    u32 rate = 44100;
    u32 frame_count = 10 * rate;
    if (size < frame_count)
        frame_count = size;
    u16 chan_num = 2;
    u16 bits = 16;
    u32 length = frame_count * chan_num * bits / 8;
    u16 byte;
    double multiplier = 32767;

    fwrite("RIFF", 1, 4, f);
    u32 chunk_size = length + 44 - 8;
    fwrite(&chunk_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    u32 subchunk1_size = 16;
    fwrite(&subchunk1_size, 4, 1, f);
    u16 fmt_type = 1;
    fwrite(&fmt_type, 2, 1, f);
    fwrite(&chan_num, 2, 1, f);
    fwrite(&rate, 4, 1, f);

    uint32_t byte_rate = rate * bits * chan_num / 8;
    fwrite(&byte_rate, 4, 1, f);
    uint16_t block_align = chan_num * bits / 8;
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits, 2, 1, f);

    fwrite("data", 1, 4, f);
    fwrite(&length, 4, 1, f);
    for (u32 i = 0; i < frame_count; i++)
    {
        byte = creal(C[i]) * multiplier;
        fwrite(&byte, 2, 1, f);
        byte = cimag(C[i]) * multiplier;
        fwrite(&byte, 2, 1, f);
    }
    fclose(f);
}

/***************************************************************************/
/*                           main function                                 */
/***************************************************************************/

int main(int argc, char **argv)
{
    double t0 = wtime();
    process_command_line_options(argc, argv);

    double complex *A = malloc(size * sizeof(*A));
    double complex *B = malloc(size * sizeof(*B));
    double complex *C = malloc(size * sizeof(*C));

    printf("Generating white noise...\n");
#pragma omp parallel for
    for (u64 i = 0; i < size; i++)
    {
        double real = 2 * (PRF(seed, 0, i) * 5.42101086242752217e-20) - 1;
        double imag = 2 * (PRF(seed, 1, i) * 5.42101086242752217e-20) - 1;
        A[i] = real + imag * I;
    }

    printf("Forward FFT...\n");
    FFT(size, A, B);

    printf("Adjusting Fourier coefficients...\n");
    #pragma omp parallel for
    for (u64 i = 0; i < size; i++)
    {
        double tmp = sin(i * 2 * M_PI / 44100);
        B[i] *= tmp * cexp(-i * 2 * I * M_PI / 4 / 44100);
        B[i] *= (i + 1) / exp((i * cutoff) / size);
    }

    printf("Inverse FFT...\n");
    iFFT(size, B, C);

    printf("Normalizing output...\n");
    double max = 0;
    #pragma omp parallel for reduction(max : max)
    for (u64 i = 0; i < size; i++)
        max = fmax(max, cabs(C[i]));

    printf("max = %g\n", max);

    // Normalisation avec NEON
    #pragma omp parallel for
    for (u64 i = 0; i < size; i += 2)
    {
        float64x2_t c_val = vld1q_f64((double *)(C + i));

        // Calculer l'inverse de max
        float64x2_t inv_max = vrecpeq_f64(vdupq_n_f64(max));
        inv_max = vmulq_f64(vrecpsq_f64(vdupq_n_f64(max), inv_max), inv_max);

        // Multiplier par l'inverse de max pour effectuer la division
        c_val = vmulq_f64(c_val, inv_max);

        vst1q_f64((double *)(C + i), c_val);
    }


    if (filename != NULL)
        save_WAV(filename, size, C);

    double t1 = wtime();
    printf("Time: %g\n", t1 - t0);

    exit(EXIT_SUCCESS);
}
#endif