#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <mpi.h>
#include <sys/time.h>

/* AUTHOR : Charles Bouillaguet <charles.bouillaguet@lip6.fr>
USAGE  : compile with -lm (and why not -O3)
            redirect the standard output to a text file
            gcc heatsink.c -O3 -lm -o heatsink
            ./heatsink > steady_state.txt
            then run the indicated python script for graphical rendering

DISCLAIMER : this code does not claim to an absolute realism.
                this code could be obviously improved, but has been written so as
                to make as clear as possible the physics principle of the simulation.
*/

/* one can change the matter of the heatsink, its size, the power of the CPU, etc. */
#define ALUMINIUM
#define FAST
#define DUMP_STEADY_STATE

const double L = 0.15;            /* length (x) of the heatsink (m) */
const double l = 0.12;            /* height (y) of the heatsink (m) */
const double E = 0.008;           /* width (z) of the heatsink (m) */
const double watercooling_T = 20; /* temperature of the fluid for water-cooling, (°C) */
const double CPU_TDP = 280;       /* power dissipated by the CPU (W) */

/* dl: "spatial step" for simulation (m) */
/* dt: "time step" for simulation (s) */
#ifdef FAST
double dl = 0.004;
double dt = 0.004;
#endif

#ifdef MEDIUM
double dl = 0.002;
double dt = 0.002;
#endif

#ifdef NORMAL
double dl = 0.001;
double dt = 0.001;
#endif

#ifdef CHALLENGE
double dl = 0.0001;
double dt = 0.00001;
#endif

/* sink_heat_capacity: specific heat capacity of the heatsink (J / kg / K) */
/* sink_density: density of the heatsink (kg / m^3) */
/* sink_conductivity: thermal conductivity of the heatsink (W / m / K) */
/* euros_per_kg: price of the matter by kilogram */
#ifdef ALUMINIUM
double sink_heat_capacity = 897;
double sink_density = 2710;
double sink_conductivity = 237;
double euros_per_kg = 1.594;
#endif

#ifdef COPPER
double sink_heat_capacity = 385;
double sink_density = 8960;
double sink_conductivity = 390;
double euros_per_kg = 5.469;
#endif

#ifdef GOLD
double sink_heat_capacity = 128;
double sink_density = 19300;
double sink_conductivity = 317;
double euros_per_kg = 47000;
#endif

#ifdef IRON
double sink_heat_capacity = 444;
double sink_density = 7860;
double sink_conductivity = 80;
double euros_per_kg = 0.083;
#endif

const double Stefan_Boltzmann = 5.6703e-8;   /* (W / m^2 / K^4), radiation of black body */
const double heat_transfer_coefficient = 10; /* coefficient of thermal convection (W / m^2 / K) */
double CPU_surface;

/*
 * Return True if the CPU is in contact with the heatsink at the point (x,y).
 * This describes an AMD EPYC "Rome".
 */
static inline bool CPU_shape(double x, double y)
{
    x -= (L - 0.0754) / 2;
    y -= (l - 0.0585) / 2;
    bool small_y_ok = (y > 0.015 && y < 0.025) || (y > 0.0337 && y < 0.0437);
    bool small_x_ok = (x > 0.0113 && x < 0.0186) || (x > 0.0193 && x < 0.0266) || (x > 0.0485 && x < 0.0558) || (x > 0.0566 && x < 0.0639);
    bool big_ok = (x > 0.03 && x < 0.045 && y > 0.0155 && y < 0.0435);
    return big_ok || (small_x_ok && small_y_ok);
}

/* returns the total area of the surface of contact between CPU and heatsink (in m^2) */
double CPU_contact_surface()
{
    double S = 0;
    for (double x = dl / 2; x < L; x += dl)
        for (double y = dl / 2; y < l; y += dl)
            if (CPU_shape(x, y))
                S += dl * dl;
    return S;
}

/* Returns the new temperature of the cell (i, j, k). For this, there is an access to neighbor
 * cells (left, right, top, bottom, front, back), except if (i, j, k) is on the external surface. */
static inline double update_temperature(const double *T, int u, int n, int m, int o, int i, int j, int k, int N, int O)
{
    /* quantity of thermal energy that must be brought to a cell to make it heat up by 1°C */
    const double cell_heat_capacity = sink_heat_capacity * sink_density * dl * dl * dl; /* J.K */
    const double dl2 = dl * dl;
    double thermal_flux = 0;

    if (i > 0)                                                      /* Je suis sur une face interne */
        thermal_flux += (T[u - 1] - T[u]) * sink_conductivity * dl; /* neighbor x-1, pour la convexion avec le voisin */
    else
    {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);                     /* radiation des voisins */
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T); /* convection avec le watercooling */
    }

    if (i < n - 1)
        thermal_flux += (T[u + 1] - T[u]) * sink_conductivity * dl; /* neighbor x+1 convexion */
    else
    {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);                     /* radiation */
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T); /* convection */
    }

    if (j > 0)
        thermal_flux += (T[u - N] - T[u]) * sink_conductivity * dl; /* neighbor y-1 */
    else
    {
        /* Bottom cell: does it receive it from the CPU ? */
        if (CPU_shape(i * dl, k * dl))
            thermal_flux += CPU_TDP / CPU_surface * dl2;
        else
        {
            thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);                     /* radiation */
            thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T); /* convection */
        }
    }

    if (j < m - 1)
        thermal_flux += (T[u + N] - T[u]) * sink_conductivity * dl; /* neighbor y+1 */
    else
    {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (k > 0)
        thermal_flux += (T[u - N * m] - T[u]) * sink_conductivity * dl; /* neighbor z-1 */
    else
    {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (k < o - 1)
        thermal_flux += (T[u + N * m] - T[u]) * sink_conductivity * dl; /* neighbor z+1 */
    else
    {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    /* adjust temperature depending on the heat flux */
    return T[u] + thermal_flux * dt / cell_heat_capacity; /* J.K / J.K = K */
}

/* Run the simulation on the k-th xy plane.
 * v is the index of the start of the k-th xy plane in the arrays T and R. */
// static inline void do_xy_plane(const double *T, double *R, int v, int n, int m, int o, int k)
// {
//     if (k == 0)
// 				// we do not modify the z = 0 plane: it is maintained at constant temperature via water-cooling
//         return;

//     for (int j = 0; j < m; j++) {   // y
//         for (int i = 0; i < n; i++) {   // x
//             int u = v + j * n + i;
//             R[u] = update_temperature(T, u, n, m, o, i, j, k);
//         }
//     }
// }

double wallclock_time()
{
    struct timeval tmp_time;
    gettimeofday(&tmp_time, NULL);
    return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6);
}

int main()
{
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    CPU_surface = CPU_contact_surface();
    double V = L * l * E;
    int n = ceil(L / dl);
    int m = ceil(E / dl);
    int o = ceil(l / dl);

    if (rank == 0)
    {
        fprintf(stderr, "HEATSINK\n");
        fprintf(stderr, "\tDimension (cm) [x,y,z] = %.1f x %.1f x %.1f\n", 100 * L, 100 * E, 100 * l);
        fprintf(stderr, "\tVolume = %.1f cm^3\n", V * 1e6);
        fprintf(stderr, "\tWeight = %.2f kg\n", V * sink_density);
        fprintf(stderr, "\tPrice = %.2f €\n", V * sink_density * euros_per_kg);
        fprintf(stderr, "SIMULATION\n");
        fprintf(stderr, "\tGrid (x,y,z) = %d x %d x %d (%.1fMo)\n", n, m, o, 7.6293e-06 * n * m * o);
        fprintf(stderr, "\tdt = %gs\n", dt);
        fprintf(stderr, "CPU\n");
        fprintf(stderr, "\tPower = %.0fW\n", CPU_TDP);
        fprintf(stderr, "\tArea = %.1f cm^2\n", CPU_surface * 10000);
    }

    // le nb de processus dans chaque axe x et z
    int num_procs_n = (int)sqrt((double)size);
    while (size % num_procs_n != 0)
    {
        num_procs_n--;
    }
    int num_procs_o = size / num_procs_n;

    // Taille de chaque sous-domaine dans chaque axe x et z et reste de la division euclidienne
    int local_n_size = n / num_procs_n;
    int local_o_size = o / num_procs_o;

    int rest_n = n % num_procs_n;
    int rest_o = o % num_procs_o;

    // s'il y a un reste on échange les valeurs de num_procs_n et num_procs_o pour voir si on a toujours un reste
    if (rest_n > 0 || rest_o > 0)
    {
        num_procs_n = num_procs_o;
        num_procs_o = size / num_procs_n;

        local_n_size = n / num_procs_n;
        local_o_size = o / num_procs_o;

        rest_n = n % num_procs_n;
        rest_o = o % num_procs_o;
    }

    // Calcul des indices de départ pour chaque processus dans chaque axe x et z
    // ainsi que la taille de pour chaque processus dans chaque axe x et z
    int tab_start_n[size];
    int tab_start_o[size];
    int tab_local_n_size[size];
    int tab_local_o_size[size];

    for (int i = 0; i < size; i++)
    {
        tab_start_n[i] = (i % num_procs_n) * local_n_size + ((i % num_procs_n) < rest_n ? (i % num_procs_n) : rest_n);
        if ((i % num_procs_n) < rest_n)
            tab_local_n_size[i] = local_n_size + 1;
        else
            tab_local_n_size[i] = local_n_size;

        tab_start_o[i] = (i / num_procs_n) * local_o_size + ((i / num_procs_n) < rest_o ? (i / num_procs_n) : rest_o);
        if ((i / num_procs_n) < rest_o)
            tab_local_o_size[i] = local_o_size + 1;
        else
            tab_local_o_size[i] = local_o_size;
    }

    int start_n = tab_start_n[rank];
    int start_o = tab_start_o[rank];
    local_n_size = tab_local_n_size[rank];
    local_o_size = tab_local_o_size[rank];

    // Vérification des calculs
    // if (rank == 0)
    // {
    //     printf("n = %d, m = %d, o = %d\n", n, m, o);
    //     printf("N = %d, m = %d, O = %d\n", local_n_size, m, local_o_size);
    //     printf("num_procs_n : %d, num_procs_o : %d\n", num_procs_n, num_procs_o);
    //     printf("rest_n : %d, rest_o : %d\n", rest_n, rest_o);
    //     if (rest_n > 0 || rest_o > 0)
    //         printf("il y a un reste\n");

    //     else
    //         printf("répartition ok\n");
    // }
    // printf("rank : %d, start_o : %d, start_n : %d, local_n_size : %d, local_o_size : %d\n", rank, start_o, start_n, local_n_size, local_o_size);

    // Permet de savoir combien de plan on doit ajouter pour chaque processus
    int column = rank % num_procs_n;
    int line = rank / num_procs_n;

    int voisin_N = 0;
    int voisin_gauche_compteur = 0, voisin_droite_compteur = 0;
    int voisin_O = 0;
    int voisin_bas_compteur = 0, voisin_haut_compteur = 0;

    int voisin_gauche = (column > 0) ? rank - 1 : MPI_PROC_NULL;
    if (voisin_gauche != MPI_PROC_NULL)
    {
        voisin_gauche_compteur++;
        voisin_N++;
    }

    int voisin_droite = (column < num_procs_n - 1) ? rank + 1 : MPI_PROC_NULL;
    if (voisin_droite != MPI_PROC_NULL)
    {
        voisin_droite_compteur++;
        voisin_N++;
    }
    int voisin_haut = (line < num_procs_o - 1) ? rank + num_procs_n : MPI_PROC_NULL;
    if (voisin_haut != MPI_PROC_NULL)
    {
        voisin_haut_compteur++;
        voisin_O++;
    }

    int voisin_bas = (line > 0) ? rank - num_procs_n : MPI_PROC_NULL;
    if (voisin_bas != MPI_PROC_NULL)
    {
        voisin_bas_compteur++;
        voisin_O++;
    }

    // // Vérification des calculs des voisins
    // assert(voisin_gauche == MPI_PROC_NULL || (voisin_gauche >= 0 && voisin_gauche < size));
    // assert(voisin_droite == MPI_PROC_NULL || (voisin_droite >= 0 && voisin_droite < size));
    // assert(voisin_haut == MPI_PROC_NULL || (voisin_haut >= 0 && voisin_haut < size));
    // assert(voisin_bas == MPI_PROC_NULL || (voisin_bas >= 0 && voisin_bas < size));
    // printf("Rank : %d, voisin_gauche : %d, voisin_droite : %d, voisin_haut : %d, voisin_bas : %d\n", rank, voisin_gauche, voisin_droite, voisin_haut, voisin_bas);

    // printf("rank : %d, start_o : %d, start_n : %d\n", rank, start_o, start_n);
    // printf("rank : %d, voisin_O : %d, voisin_N : %d, voisin_bas_compteur : %d, voisin_gauche_compteur : %d, voisin_haut_compteur : %d, voisin_droite_compteur : %d\n", rank, voisin_O, voisin_N, voisin_bas_compteur, voisin_gauche_compteur, voisin_haut_compteur, voisin_droite_compteur);

    /* initially the heatsink is at the temperature of the water-cooling fluid */
    int local_size = (local_n_size + voisin_N) * m * (local_o_size + voisin_O);
    double *T_local = malloc(local_size * sizeof(*T_local));
    double *R_local = malloc(local_size * sizeof(*R_local));
    for (int u = 0; u < local_size; u++)
    {
        R_local[u] = T_local[u] = watercooling_T + 273.15;
    }

    /* let's go! we switch the CPU on and launch the simulation until it reaches a stationary state. */
    double t = 0;
    int n_steps = 0;
    int convergence = 0;

    int count = (local_o_size + voisin_O) * m;
    int blocklength = 1;
    int stride = local_n_size + voisin_N;
    MPI_Datatype planche;
    MPI_Type_vector(count, blocklength, stride, MPI_DOUBLE, &planche);
    MPI_Type_commit(&planche);

    /* simulating time steps */
    double start_time = wallclock_time();
    while (convergence == 0)
    {
        /* Update all cells. xy planes are processed, for increasing values of z. */
        for (int k = 0; k < local_o_size; k++)
        { // z
            int v = k * (local_n_size + voisin_N) * m;
            for (int j = 0; j < m; j++)
            { // y
                for (int i = 0; i < local_n_size; i++)
                { // x
                    int u = v + j * (local_n_size + voisin_N) + i;
                    u += (local_n_size + voisin_N) * m * voisin_bas_compteur;
                    u += voisin_gauche_compteur;

                    if (k + start_o != 0)
                    {
                        R_local[u] = update_temperature(T_local, u, n, m, o, i + start_n, j, k + start_o, local_n_size + voisin_N, local_o_size + voisin_O);
                    }
                }
            }
        }

        MPI_Request send[4], recv[4];
        for (int i = 0; i < 4; i++)
        {
            send[i] = MPI_REQUEST_NULL;
            recv[i] = MPI_REQUEST_NULL;
        }

        if (size > 1)
        {
            #define TAG_GAUCHE_ENVOI 0
            #define TAG_GAUCHE_RECEPTION 1
            #define TAG_DROITE_ENVOI 1
            #define TAG_DROITE_RECEPTION 0
            #define TAG_HAUT_ENVOI 2
            #define TAG_HAUT_RECEPTION 3
            #define TAG_BAS_ENVOI 3
            #define TAG_BAS_RECEPTION 2

            // Taille du message
            int taille_message = m * (local_n_size + voisin_N);
            // Indices de départ pour l'envoi et la réception
            double *send_haut_indice = R_local + m * (local_n_size + voisin_N) * (local_o_size + voisin_O - 2);
            double *receive_haut_indice = R_local + m * (local_n_size + voisin_N) * (local_o_size + voisin_O - 1);
            double *send_bas_indice = R_local + m * (local_n_size + voisin_N);
            double *receive_bas_indice = R_local;

            // // Vérification des indices et de la taille du message
            // printf("début de test pour le rank : %d\n", rank);
            // assert(send_haut_indice >= R && send_haut_indice + taille_message <= R + total_size);
            // printf("test 1 réussi pour le rank : %d\n", rank);
            // assert(receive_haut_indice >= R && receive_haut_indice + taille_message <= R + total_size);
            // printf("test 2 réussi pour le rank : %d\n", rank);
            // assert(send_bas_indice >= R && send_bas_indice + taille_message <= R + total_size);
            // printf("test 3 réussi pour le rank : %d\n", rank);
            // assert(receive_bas_indice >= R && receive_bas_indice + taille_message <= R + total_size);
            // printf("test 4 réussi pour le rank : %d\n", rank);

            // Indices pour l'envoi et la réception
            double *send_gauche_indice = R_local + 1;
            double *recv_gauche_indice = R_local;
            double *send_droite_indice = R_local + local_n_size + voisin_N - 2;
            double *recv_droite_indice = R_local + local_n_size + voisin_N - 1;

            // // Vérifications pour les communications à gauche et à droite
            // assert(send_gauche_indice >= R && send_gauche_indice + m * (stride - 1) < R + total_size);
            // assert(recv_gauche_indice >= R - 1 && recv_gauche_indice + m * (stride - 1) < R + total_size);
            // assert(send_droite_indice >= R && send_droite_indice + m * (stride - 1) < R + total_size);
            // assert(recv_droite_indice >= R && recv_droite_indice + m * (stride - 1) < R + total_size + 1);
            // printf("test 5 réussi pour le rank : %d\n", rank);

            // Début des communications: Haut et bas se réfèrent à l'axe Z, Gauche et droite à l'axe X.
            if (voisin_haut != MPI_PROC_NULL)
            {
                // Envoi de la dernière couche de données internes au voisin du haut
                MPI_Isend(send_haut_indice, taille_message, MPI_DOUBLE, voisin_haut, TAG_HAUT_ENVOI, MPI_COMM_WORLD, &send[0]);
                // Réception dans la première couche de bordure depuis le voisin du haut
                MPI_Irecv(receive_haut_indice, taille_message, MPI_DOUBLE, voisin_haut, TAG_HAUT_RECEPTION, MPI_COMM_WORLD, &recv[0]);
            }

            if (voisin_bas != MPI_PROC_NULL)
            {
                // Envoi de la première couche de données internes au voisin du bas
                MPI_Isend(send_bas_indice, taille_message, MPI_DOUBLE, voisin_bas, TAG_BAS_ENVOI, MPI_COMM_WORLD, &send[1]);
                // Réception dans la dernière couche de bordure depuis le voisin du bas
                MPI_Irecv(receive_bas_indice, taille_message, MPI_DOUBLE, voisin_bas, TAG_BAS_RECEPTION, MPI_COMM_WORLD, &recv[1]);
            }

            if (voisin_gauche != MPI_PROC_NULL)
            {
                // Envoi de la première "colonne" interne au voisin de gauche
                MPI_Isend(send_gauche_indice, 1, planche, voisin_gauche, TAG_GAUCHE_ENVOI, MPI_COMM_WORLD, &send[2]);
                // Réception dans la "colonne" de bordure gauche depuis le voisin de gauche
                MPI_Irecv(recv_gauche_indice, 1, planche, voisin_gauche, TAG_GAUCHE_RECEPTION, MPI_COMM_WORLD, &recv[2]);
            }

            if (voisin_droite != MPI_PROC_NULL)
            {
                // Envoi de la dernière colonne de données internes au voisin de droite
                MPI_Isend(send_droite_indice, 1, planche, voisin_droite, TAG_DROITE_ENVOI, MPI_COMM_WORLD, &send[3]);
                // Réception dans la colonne de bordure droite depuis le voisin de droite
                MPI_Irecv(recv_droite_indice, 1, planche, voisin_droite, TAG_DROITE_RECEPTION, MPI_COMM_WORLD, &recv[3]);
            }

            // Attendre la fin de toutes les communications
            for (int i = 0; i < 4; i++)
            {
                if (send[i] != MPI_REQUEST_NULL)
                    MPI_Wait(&send[i], MPI_STATUS_IGNORE);
                if (recv[i] != MPI_REQUEST_NULL)
                    MPI_Wait(&recv[i], MPI_STATUS_IGNORE);
            }
        }

        /* each second, we test the convergence, and print a short progress report */
        if (n_steps % ((int)(1 / dt)) == 0)
        {
            double delta_T = 0;
            double max = -INFINITY;
            for (int k = 0; k < local_o_size; k++)
            { // z
                int v = k * (local_n_size + voisin_N) * m;
                for (int j = 0; j < m; j++)
                { // y
                    for (int i = 0; i < local_n_size; i++)
                    { // x
                        int u = v + j * (local_n_size + voisin_N) + i;
                        u += (local_n_size + voisin_N) * m * voisin_bas_compteur;
                        u += voisin_gauche_compteur;

                        delta_T += (R_local[u] - T_local[u]) * (R_local[u] - T_local[u]);
                        if (R_local[u] > max)
                            max = R_local[u];
                    }
                }
            }

            MPI_Allreduce(&delta_T, &delta_T, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&max, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            delta_T = sqrt(delta_T) / dt;
            if (rank == 0)
            {
                fprintf(stderr, "t = %.1fs ; T_max = %.1f°C ; convergence = %g\n", t, max - 273.15, delta_T);
            }
            if (delta_T < 0.1)
                convergence = 1;
        }

        /* the new temperatures are in R */
        double *tmp = R_local;
        R_local = T_local;
        T_local = tmp;

        t += dt;
        n_steps += 1;
    }
    double end_time = wallclock_time();

    // préparation de l'envoi des données : on fait un tableau de taille local_n_size * m * local_o_size pour récupérer que les données internes
    double *T_send = NULL;
    if (rank != 0)
    {
        T_send = malloc(local_n_size * m * local_o_size * sizeof(*T_send));
        for (int k = 0; k < local_o_size; k++)
        { // z
            int v = k * (local_n_size + voisin_N) * m;
            for (int j = 0; j < m; j++)
            { // y
                for (int i = 0; i < local_n_size; i++)
                { // x
                    int u = v + j * (local_n_size + voisin_N) + i;
                    u += (local_n_size + voisin_N) * m * voisin_bas_compteur;
                    u += voisin_gauche_compteur;

                    T_send[k * local_n_size * m + j * local_n_size + i] = T_local[u];
                }
            }
        }
    }

    // Réception et assemblage des données de température globales dans le processus principal
    double *T_final = NULL;
    double *T_recv = NULL;
    if (rank == 0)
    {
        // Copie des données locales de température du processus principal dans T_final
        T_final = malloc(n * m * o * sizeof(*T_final));
        for (int k = 0; k < local_o_size; k++)
        { // z
            int v = k * (local_n_size + voisin_N) * m;
            for (int j = 0; j < m; j++)
            { // y
                for (int i = 0; i < local_n_size; i++)
                { // x
                    int u = v + j * (local_n_size + voisin_N) + i;
                    u += (local_n_size + voisin_N) * m * voisin_bas_compteur;
                    u += voisin_gauche_compteur;
                    // Mapping des indices du processus principal vers les indices du processus actuel
                    T_final[(k + start_o) * n * m + j * n + (i + start_n)] = T_local[u];
                }
            }
        }

        // Réception des données des autres processus et intégration dans T_final
        T_recv = malloc(local_n_size * m * local_o_size * sizeof(*T_recv));
        for (int _rank = 1; _rank < size; _rank++)
        {
            MPI_Recv(T_recv, tab_local_n_size[_rank] * m * tab_local_o_size[_rank], MPI_DOUBLE, _rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int k = 0; k < tab_local_o_size[_rank]; k++)
            { // z
                for (int j = 0; j < m; j++)
                { // y
                    for (int l = 0; l < tab_local_n_size[_rank]; l++)
                    { // x
                        T_final[(k + tab_start_o[_rank]) * n * m + j * n + (l + tab_start_n[_rank])] = T_recv[k * tab_local_n_size[_rank] * m + j * tab_local_n_size[_rank] + l];
                    }
                }
            }
        }
    }
    else
    { // Envoi des données de température locales du processus actuel au processus principal
        MPI_Send(T_send, local_n_size * m * local_o_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
#ifdef DUMP_STEADY_STATE
        printf("###### STEADY STATE; t = %.1f\n", t);
        for (int k = 0; k < o; k++)
        { // z
            printf("# z = %g\n", k * dl);
            for (int j = 0; j < m; j++)
            { // y
                for (int i = 0; i < n; i++)
                { // x
                    printf("%.1f ", T_final[k * n * m + j * n + i] - 273.15);
                }
                printf("\n");
            }
        }
        printf("\n");
        fprintf(stderr, "Total computing time: %g sec\n", end_time - start_time);
        fprintf(stderr, "For graphical rendering: python3 rendu_picture_steady.py [filename.txt] %d %d %d\n", n, m, o);
#endif
    }

    free(T_local);
    free(R_local);
    free(T_send);
    if (rank == 0)
    {
        free(T_recv);
        free(T_final);
    }

    MPI_Finalize();
    exit(EXIT_SUCCESS);
}