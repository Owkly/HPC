# Compilateur
ifeq ($(ARCH),NEON)
  CC = gcc-13
else
  CC = gcc
#   CC = gcc -pg
#   CC = gcc -g
endif

# Flags de lien
LDFLAGS = -lm
# Flags de compilation pour le fichier séquentiel
SEQ_CFLAGS = -Wall
# Flags de compilation pour les fichiers parallèles avec OpenMP
OMP_CFLAGS = -fopenmp  -Wall
# Flags de compilation pour les fichiers avec vectorisation
VEC_CFLAGS = -O3 -ftree-vectorize -fopenmp -Wall
# Flags pour NEON et AVX
AVX_FLAGS = -mavx2 -mfma
NEON_FLAGS = -march=armv8-a+simd

# Nom du fichier source séquentiel et de son exécutable
SEQ_SRC = fft_sequential.c
SEQ_EXE = $(SEQ_SRC:%.c=%)
# Liste des fichiers sources utilisant OpenMP et de leurs exécutables
OMP_SRCS = $(wildcard *omp.c)
OMP_EXES = $(OMP_SRCS:%.c=%)
# Liste des fichiers sources avec vectorisation et de leurs exécutables
VEC_SRCS = $(wildcard *vec.c)
VEC_EXES = $(VEC_SRCS:%.c=%)

# Règle par défaut pour tout compiler
all: $(SEQ_EXE) $(OMP_EXES) $(VEC_EXES)

# Règle pour compiler le fichier source séquentiel en un exécutable
$(SEQ_EXE): $(SEQ_SRC)
	$(CC) $(SEQ_CFLAGS) -o $@ $< $(LDFLAGS)
# Règle pour compiler chaque fichier source OpenMP en un exécutable
$(OMP_EXES): %: %.c
	$(CC) $(OMP_CFLAGS) -o $@ $< $(LDFLAGS)
# Règle pour compiler chaque fichier source de vectorisation en un exécutable
$(VEC_EXES): %: %.c
ifeq ($(ARCH),NEON)
	$(CC) $(VEC_CFLAGS) $(NEON_FLAGS) -o $@ $< $(LDFLAGS)
else
	$(CC) $(VEC_CFLAGS) $(AVX_FLAGS) -o $@ $< $(LDFLAGS)
endif











# Variables communes pour les options de ligne de commande
# SIZE = 1048576
# SIZE = 2097152
# SIZE = 4194304
# SIZE = 8388608
# SIZE = 16777216
# SIZE = 33554432
SIZE = 67108864
# SIZE = 134217728

SEED = 0
CUTOFF = 500

# Règle pour exécuter le programme séquentiel
# Utilisez cette commande avec: make run-seq
run-seq:
	./$(SEQ_EXE) --size $(SIZE) --seed $(SEED) --output $(SEQ_EXE).WAV --cutoff $(CUTOFF)

# Règle pour exécuter un programme spécifique avec un nombre donné de threads
# Utilisez cette commande avec: make run TARGET=nom_du_programme THREADS=x
run:
	OMP_NUM_THREADS=$(THREADS) ./$(TARGET) --size $(SIZE) --seed $(SEED) --output $(TARGET).WAV --cutoff $(CUTOFF)
# Exemple d'utilisation : make run TARGET=fft_omp THREADS=4
# Ceci va lancer 'fft_omp' avec 4 threads.

# Règle pour supprimer les exécutables et les fichiers WAV, ainsi que les fichiers de profilage
clean:
	rm -f $(SEQ_EXE) $(OMP_EXES) $(VEC_EXES) *.WAV analysis*