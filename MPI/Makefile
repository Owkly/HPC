# Makefile pour compiler heatsink_1D, heatsink_2D et heatsink

# Compilateurs
MPICC = mpicc
GCC = gcc

# Options de compilation
CFLAGS = -Wall

# Bibliothèques à lier
LIBS = -lm

# Cibles par défaut
all: heatsink_1D heatsink_2D heatsink_sequential

# Règles de compilation pour heatsink_1D et heatsink_2D
heatsink_1D: heatsink_1D.c
	$(MPICC) $(CFLAGS) heatsink_1D.c -o heatsink_1D $(LIBS)

heatsink_2D: heatsink_2D.c
	$(MPICC) $(CFLAGS) heatsink_2D.c -o heatsink_2D $(LIBS)

# # Règle de compilation pour heatsink
heatsink_sequential: heatsink_sequential.c
	$(GCC) $(CFLAGS) heatsink_sequential.c -o heatsink_sequential $(LIBS)

# Nettoyage
clean:
	rm -f heatsink_1D heatsink_2D heatsink_sequential *.txt

# Sur ce PC max 6 processus
run:
	mpiexec -n $(NUM) ./$(TARGET) > steady.txt

# Règle pour générer les graphiques
# Valeurs par défaut (mode FAST)
GRAPHIC_PARAMS = 38 2 30

ifeq ($(MODE), FAST)
    GRAPHIC_PARAMS = 38 2 30
endif

ifeq ($(MODE), MEDIUM)
    GRAPHIC_PARAMS = 75 4 60
endif

ifeq ($(MODE), NORMAL)
    GRAPHIC_PARAMS = 150 8 120
endif

graphic:
	python3 rendu_picture_steady.py steady.txt $(GRAPHIC_PARAMS)






