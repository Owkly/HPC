// #define  _POSIX_C_SOURCE 1
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <getopt.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#include <complex.h>

// uint64_t est un type entier non signé de 64 bits c'est à dire un entier positif
typedef uint64_t u64; // u64 est un alias pour uint64_t qui
typedef uint32_t u32;
typedef uint16_t u16;

double cutoff = 500; // cutoff frequency (Hz) ce qui veut dire que les fréquences supérieures à 500Hz seront atténuées
u64 seed = 0;
u64 size = 0;
char *filename = NULL;

/*****************************************************************************/
/*                    pseudo-random function (SPECK-like)                    */
/*****************************************************************************/

/**
 * ROR décale tout les bits à droite de r position
 * ROL décale tout les bits à gauche de r position
 * avec r = 3
 * x = 0000 0000 0000 1001
 *
 * Exemple sur 16 bit (sachant que les vrais fonction sont sur 64 bits)
 *      ROR(x,r) = 0010 0000 0000 0001
 *      ROL(x,r) = 0000 0000 0100 1000
 */
#define ROR(x, r) ((x >> r) | (x << (64 - r)))
#define ROL(x, r) ((x << r) | (x >> (64 - r)))

/**
 * Applique l'algo suivant pour modifier les entiers passé en paramètres (ce qui permet de faire un changement "pseudo aléatoire")
 *
 * x = 0000 0001 0000 0000                  (256 en base 10)
 * y = 0000 0000 0000 1000                  (  8 en base 10)
 * k = 2
 *
 * R(x,y,k)
 *      x = ROR(x, 8) = 0000 0000 0000 0001 (  1 en base 10)
 *      x = x + y     = 0000 0000 0000 1001 (  9 en base 10)
 *      x = x^k       = 0000 0000 0101 0001 ( 81 en base 10)
 *      y = ROL(y,3)  = 0000 0000 0000 0001 (  1 en base 10)
 *      y = y^x       = 0000 0000 0000 0001 (  1 en base 10)
 *
 */

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

/* Fonction qui calcul les coeff de fourrier avec la FFT récursif */
void FFT_rec(u64 n, const double complex *X, double complex *Y, u64 stride)
{
	if (n == 1)
	{
		Y[0] = X[0];
		return;
	}
	double complex omega_n = cexp(-2 * I * M_PI / n); /* n-th root of unity*/
	double complex omega = 1;						  /* twiddle factor */
	// prend les éléments du tableau sur les indice pair
	FFT_rec(n / 2, X, Y, 2 * stride);
	// prend les éléments du tableau sur les indice impair
	FFT_rec(n / 2, X + stride, Y + n / 2, 2 * stride);

	/**
	 * Exemple [0,1,2,3,4,5,6,7] -> [0,2,4,],   [1,3,5,7]
	 *                           -> [0,4],[2,6] [1,5],[3,7]
	 */

	for (u64 i = 0; i < n / 2; i++)
	{
		double complex p = Y[i];
		double complex q = Y[i + n / 2] * omega;
		Y[i] = p + q;
		Y[i + n / 2] = p - q;
		omega *= omega_n;
	}
}

/*  Ce code vérifie avant d'utiliser la FFT si n est une puissance de 2 */
void FFT(u64 n, const double complex *X, double complex *Y)
{
	/* sanity check */
	if ((n & (n - 1)) != 0)
		errx(1, "size is not a power of two (this code does not handle other cases)");
	FFT_rec(n, X, Y, 1); /* stride == 1 initially */
}

/**
 * Computes the inverse Fourier transform, but destroys the input
 * Calcul le conjugué de la FFT
 * IFFT = 1/(taille de X) * conj(FFT(conj(X)))
 */
void iFFT(u64 n, double complex *X, double complex *Y)
{
	for (u64 i = 0; i < n; i++)
		X[i] = conj(X[i]); // conjugue/change le signe de la partie imaginaire de X[i]
	FFT(n, X, Y);
	for (u64 i = 0; i < n; i++)
		Y[i] = conj(Y[i]) / n; // conjugue la partie imaginaire de la FFT Y[i]
}

/***************************** utility functions *************************************/
/*          Défini les des fonctions qui seront utile tel que un timer               */
/*        Une fonction qui véréfie les options de la commande d'éxécution            */
/* Une fonction qui enregistre les données d'un tableau complexe dans un fichier WAV */
/*************************************************************************************/

/*
	donne le temps actuel avec une précision jusqu'aux microsecondes
*/
double wtime()
{
	struct timeval ts;
	gettimeofday(&ts, NULL);
	return (double)ts.tv_sec + ts.tv_usec / 1e6;
}

/*
	Si vous exécuter le programme avec cette ligne de commande
		./mon_programme --size 100 --seed 42 --output result.txt --cutoff 0.5
	Cette fonction analysera les options et les mettra à jour dans le programme

	--size 100 :            Définit la taille à 100.
	--seed 42 :             Définit la graine de génération aléatoire à 42.
	--output result.txt :   Définit le fichier de sortie à "result.txt".
	--cutoff 0.5 :          Définit la valeur de seuil à 0.5.

	tout les options sont optionnel sauf size qui est obligatoire
*/
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
	/* validation */
	if (size == 0)
		errx(1, "missing --size argument");
}

/*
	save at most 10s of sound output in .WAV format

	La fonction effectue les opérations suivantes :
	1. Vérifie que la taille est inférieure à 1000000000.
	2. Ouvre le fichier WAV en mode écriture ("w").
	3. Affiche un message dans le terminal indiquant l'écriture de <= 10 secondes de sortie audio dans le fichier.
	4. Définit les paramètres WAV tels que le taux d'échantillonnage, le nombre de canaux, la profondeur de bits, etc.
	5. Écrit l'en-tête WAV dans le fichier (donc de définir le fichier comme étant un fichier wav)
	6. Écrit les échantillons audio convertis en entiers 16 bits dans le fichier.
	7. Ferme le fichier une fois l'écriture terminée.

	Note : La fonction génère un fichier WAV avec une taille d'échantillon d'au plus 10 secondes.
	Les échantillons audio sont obtenus à partir du tableau de nombres complexes C, en prenant
	les parties réelle et imaginaire, puis en les convertissant en entiers 16 bits.
*/

void save_WAV(char *filename, u64 size, double complex *C)
{
	/* Étape 1 */
	assert(size < 1000000000);
	/* Étape 2 */
	FILE *f = fopen(filename, "w");
	if (f == NULL)
		err(1, "fopen");
	/* Étape 3 */
	printf("Writing <= 10s of audio output in %s\n", filename);
	/* Étape 4 */
	u32 rate = 44100; // Sample rate
	u32 frame_count = 10 * rate;
	if (size < frame_count)
		frame_count = size;
	u16 chan_num = 2; // Number of channels
	u16 bits = 16;	  // Bit depth
	u32 length = frame_count * chan_num * bits / 8;
	u16 byte;
	double multiplier = 32767;

	/* Étape 5 */
	/* WAVE Header Data */
	fwrite("RIFF", 1, 4, f);
	u32 chunk_size = length + 44 - 8;
	fwrite(&chunk_size, 4, 1, f);
	fwrite("WAVE", 1, 4, f);
	fwrite("fmt ", 1, 4, f);
	u32 subchunk1_size = 16;
	fwrite(&subchunk1_size, 4, 1, f);
	u16 fmt_type = 1; // 1 = PCM
	fwrite(&fmt_type, 2, 1, f);
	fwrite(&chan_num, 2, 1, f);
	fwrite(&rate, 4, 1, f);
	// (Sample Rate * BitsPerSample * Channels) / 8
	uint32_t byte_rate = rate * bits * chan_num / 8;
	fwrite(&byte_rate, 4, 1, f);
	uint16_t block_align = chan_num * bits / 8;
	fwrite(&block_align, 2, 1, f);
	fwrite(&bits, 2, 1, f);

	/* Étape 6 */
	/* Marks the start of the data */
	fwrite("data", 1, 4, f);
	fwrite(&length, 4, 1, f); // Data size
	for (u32 i = 0; i < frame_count; i++)
	{
		byte = creal(C[i]) * multiplier;
		fwrite(&byte, 2, 1, f);
		byte = cimag(C[i]) * multiplier;
		fwrite(&byte, 2, 1, f);
	}
	/* Étape 7 */
	fclose(f);
}

/***************************************************************************/
/*                           main function                                 */
/***************************************************************************/

int main(int argc, char **argv)
{
	// Étape 1: Traitement des options de ligne de commande pour définir les paramètres du programme
	double t0 = wtime();
	process_command_line_options(argc, argv);

	// Étape 2 : Allocation de mémoire pour les tableaux de nombres complexes A, B et C
	double complex *A = malloc(size * sizeof(*A));
	double complex *B = malloc(size * sizeof(*B));
	double complex *C = malloc(size * sizeof(*C));

	// Étape 3 : Génération de bruit dans le tableau A en utilisant la fonction pseudo-aléatoire PRF
	printf("Generating white noise...\n");
	for (u64 i = 0; i < size; i++)
	{
		double real = 2 * (PRF(seed, 0, i) * 5.42101086242752217e-20) - 1;
		double imag = 2 * (PRF(seed, 1, i) * 5.42101086242752217e-20) - 1;
		A[i] = real + imag * I;
	}
    printf("Generating white noise time = %g\n", wtime() - t0);

	// Étape 4 : Calcul de la transformée de Fourier récursif (FFT) des données du tableau A dans le tableau B
	printf("Forward FFT...\n");
	FFT(size, A, B);
	printf("Foward FFT time = %g\n", wtime() - t0);

	// Étape 5 : Ajustement des coefficients de Fourier dans le tableau B.
	printf("Adjusting Fourier coefficients...\n");
	for (u64 i = 0; i < size; i++)
	{
		double tmp = sin(i * 2 * M_PI / 44100);
		B[i] *= tmp * cexp(-i * 2 * I * M_PI / 4 / 44100);
		B[i] *= (i + 1) / exp((i * cutoff) / size); // filtre passe bas
	}
    printf("Adjusting Fourier coefficients time = %g\n", wtime() - t0);

	// Étape 6 : Calcul l'inverse de la FFT (iFFT) des données ajustées dans le tableau B dans le tableau C
	printf("Inverse FFT...\n");
	iFFT(size, B, C);
    printf("Inverse FFT time = %g\n",wtime() - t0);

	// Étape 7 : Normalisation des données dans le tableau C en divisant chaque élément par la valeur maximale absolue trouvée
	printf("Normalizing output...\n");
	double max = 0;
	for (u64 i = 0; i < size; i++)
		max = fmax(max, cabs(C[i]));
	printf("max = %g\n", max);
	for (u64 i = 0; i < size; i++)
		C[i] /= max;
    printf("Normalizing output time = %g\n", wtime() - t0);

	// Étape 8 : Sauvegarde des données dans un fichier WAV si un nom de fichier est spécifié
	if (filename != NULL)
		save_WAV(filename, size, C);

	double t1 = wtime();
	printf("Time: %g\n", t1 - t0);

	exit(EXIT_SUCCESS);
}