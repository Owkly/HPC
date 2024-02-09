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
#define COPPER /* ALUMINIUM, COPPER, GOLD, IRON */
#define FAST /* MEDIUM is faster, and FAST is even faster (for debugging) */
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
static inline double update_temperature(const double *T, int u, int n, int m, int o, int i, int j, int k)
{
    /* quantity of thermal energy that must be brought to a cell to make it heat up by 1°C */
    const double cell_heat_capacity = sink_heat_capacity * sink_density * dl * dl * dl; /* J.K */
    const double dl2 = dl * dl;
    double thermal_flux = 0;

    if (i > 0)
        thermal_flux += (T[u - 1] - T[u]) * sink_conductivity * dl; /* neighbor x-1 */
    else
    {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (i < n - 1)
        thermal_flux += (T[u + 1] - T[u]) * sink_conductivity * dl; /* neighbor x+1 */
    else
    {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (j > 0)
        thermal_flux += (T[u - n] - T[u]) * sink_conductivity * dl; /* neighbor y-1 */
    else
    {
        /* Bottom cell: does it receive it from the CPU ? */
        if (CPU_shape(i * dl, k * dl))
            thermal_flux += CPU_TDP / CPU_surface * dl2;
        else
        {
            thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
            thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
        }
    }

    if (j < m - 1)
        thermal_flux += (T[u + n] - T[u]) * sink_conductivity * dl; /* neighbor y+1 */
    else
    {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (k > 0)
        thermal_flux += (T[u - n * m] - T[u]) * sink_conductivity * dl; /* neighbor z-1 */
    else
    {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (k < o - 1)
        thermal_flux += (T[u + n * m] - T[u]) * sink_conductivity * dl; /* neighbor z+1 */
    else
    {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    /* adjust temperature depending on the heat flux */
    return T[u] + thermal_flux * dt / cell_heat_capacity;
}

/* Run the simulation on the k-th xy plane.
 * v is the index of the start of the k-th xy plane in the arrays T and R. */
static inline void do_xy_plane(const double *T, double *R, int v, int n, int m, int o, int k)
{
    if (k == 0)
        // we do not modify the z = 0 plane: it is maintained at constant temperature via water-cooling
        return;

    for (int j = 0; j < m; j++)
    { // y
        for (int i = 0; i < n; i++)
        { // x
            int u = v + j * n + i;
            R[u] = update_temperature(T, u, n, m, o, i, j, k);
        }
    }
}

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
    int n = ceil(L / dl); // x
    int m = ceil(E / dl); // y
    int o = ceil(l / dl); // z

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

    // On vérifie que le nombre de processus est cohérent avec la taille du problème
    if (size >= o)
    {
        if (rank == 0)
        {
            fprintf(stderr, "Le nombre de processus est trop grand par rapport à la taille du problème\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    // Calcul de la répartition des plans par processus
    int local_o_size = o / size;
    int reste = o % size;
    int o_start, o_per_process;
    // Les processus de rang < reste ont un plan de plus à traiter
    if (rank < reste)
    {
        o_start = rank * (local_o_size + 1);
        o_per_process = local_o_size + 1;
    }
    else
    {
        o_start = rank * local_o_size + reste;
        o_per_process = local_o_size;
    }

    // o_per_process_and_border est le nombre de plan par processus avec les plans de bord
    int o_per_process_and_border = o_per_process;
    if (rank == 0 || rank == size - 1)
    {
        o_per_process_and_border++;
    }
    else
    {
        o_per_process_and_border += 2;
    }

    double *T_local = malloc(n * m * (o_per_process_and_border) * sizeof(*T_local));
    double *R_local = malloc(n * m * (o_per_process_and_border) * sizeof(*R_local));
    if (T_local == NULL || R_local == NULL)
    {
        perror("T_local or R_local could not be allocated");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* initially the heatsink is at the temperature of the water-cooling fluid */
    for (int u = 0; u < n * m * (o_per_process_and_border); u++)
        R_local[u] = T_local[u] = watercooling_T + 273.15;

    /* let's go! we switch the CPU on and launch the simulation until it reaches a stationary state. */
    double t = 0;
    int n_steps = 0;
    int convergence = 0;
    int local_size = n * m * o_per_process;

    /* simulating time steps */
    double start_time = wallclock_time();
    while (convergence == 0)
    {

        #define TAG_SEND_LAST 0
        #define TAG_RECV_LAST 0
        #define TAG_SEND_FIRST 1
        #define TAG_RECV_FIRST 1

        int rank_left = rank - 1;
        int rank_right = rank + 1;
        // Communication des plans de border
        MPI_Request send_request[2], recv_request[2];
        if (size > 1)
        {
            // On envoie les plans de bord aux processus voisins
            if (rank < size - 1) // on envoie le dernier plan et on reçoit le premier
            {
                if (rank == 0)
                {
                    MPI_Isend(R_local + n * m * (o_per_process - 1), n * m, MPI_DOUBLE, rank_right, TAG_SEND_LAST, MPI_COMM_WORLD, &send_request[0]);
                    MPI_Irecv(R_local + n * m * (o_per_process), n * m, MPI_DOUBLE, rank_right, TAG_RECV_FIRST, MPI_COMM_WORLD, &recv_request[0]);
                }
                else
                {
                    MPI_Isend(R_local + n * m * (o_per_process), n * m, MPI_DOUBLE, rank_right, TAG_SEND_LAST, MPI_COMM_WORLD, &send_request[0]);
                    MPI_Irecv(R_local + n * m * (o_per_process + 1), n * m, MPI_DOUBLE, rank_right, TAG_RECV_FIRST, MPI_COMM_WORLD, &recv_request[0]);
                }
            }
            if (rank > 0) // on envoie le premier plan et o reçoit le denier
            {
                MPI_Isend(R_local + n * m, n * m, MPI_DOUBLE, rank_left, TAG_SEND_FIRST, MPI_COMM_WORLD, &send_request[1]);
                MPI_Irecv(R_local, n * m, MPI_DOUBLE, rank_left, TAG_RECV_LAST, MPI_COMM_WORLD, &recv_request[1]);
            }

            /* Update all cells. xy planes are processed, for increasing values of z. */
            for (int k = 0; k < o_per_process; k++)
            {                                              // z
                int v = ((rank == 0) ? k : k + 1) * n * m; // on ajoute 1 pour pas update le plan de bord de gauche
                for (int j = 0; j < m; j++)
                { // y
                    for (int i = 0; i < n; i++)
                    { // x
                        int u = v + j * n + i;
                        if (k + o_start != 0)
                        {
                            R_local[u] = update_temperature(T_local, u, n, m, o, i, j, k + o_start);
                        }
                    }
                }
            }

            /* each second, we test the convergence, and print a short progress report */
            if (n_steps % ((int)(1 / dt)) == 0)
            {
                // Calcul de la somme de delta_T et de la température maximale locale
                double local_delta_T = 0;
                double local_max = -INFINITY;
                for (int u = 0; u < n * m * o_per_process; u++)
                {
                    local_delta_T += (R_local[u] - T_local[u]) * (R_local[u] - T_local[u]);
                    if (R_local[u] > local_max)
                        local_max = R_local[u];
                }

                // Calcul de la somme de delta_T et de la température maximale globale
                double global_delta_T = 0;
                double global_max = -INFINITY;
                MPI_Allreduce(&local_delta_T, &global_delta_T, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

                // Affichage des résultats par le processus 0
                global_delta_T = sqrt(global_delta_T) / dt;
                if (rank == 0)
                {
                    fprintf(stderr, "t = %.1fs ; T_max = %.1f°C ; convergence = %g\n", t, global_max - 273.15, global_delta_T);
                }
                if (global_delta_T < 0.1)
                    convergence = 1;
            }

            // Ensuite, on attend que la communication soit complète avant d'utiliser les données communiquées
            if (rank < size - 1)
            {
                MPI_Wait(&send_request[0], MPI_STATUS_IGNORE);
                MPI_Wait(&recv_request[0], MPI_STATUS_IGNORE);
            }
            if (rank > 0)
            {
                MPI_Wait(&send_request[1], MPI_STATUS_IGNORE);
                MPI_Wait(&recv_request[1], MPI_STATUS_IGNORE);
            }
        }

        /* the new temperatures are in R */
        double *tmp = R_local;
        R_local = T_local;
        T_local = tmp;

        t += dt;
        n_steps += 1;
    }
    // Récupération des données par le processus 0
    double end_time = wallclock_time();

    double *T = NULL;
    if (rank == 0)
    {
        T = malloc(n * m * o * sizeof(*T));

        int *recvcounts = malloc(size * sizeof(int));
        int *displs = malloc(size * sizeof(int));
        for (int i = 0; i < size; i++)
        {
            recvcounts[i] = n * m * ((i < reste) ? local_o_size + 1 : local_o_size);
            displs[i] = (i < reste) ? i * (local_o_size + 1) * n * m : (i * local_o_size + reste) * n * m;
        }

        MPI_Gatherv(&T_local[0], local_size, MPI_DOUBLE, T, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gatherv(&T_local[n * m], local_size, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
                    printf("%.1f ", T[k * n * m + j * n + i] - 273.15);
                }
                printf("\n");
            }
        }
        printf("\n");
        fprintf(stderr, "Total computing time: %g sec\n", end_time - start_time);
        fprintf(stderr, "For graphical rendering: python3 rendu_picture_steady.py [filename.txt] %d %d %d\n", n, m, o);
#endif
    }
    MPI_Finalize();

    free(T_local);
    free(R_local);
    if (rank == 0)
    {
        free(T);
    }

    exit(EXIT_SUCCESS);
}