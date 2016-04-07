/*
 * RRT.cuh
 *
 *  Created on: 28 Feb 2016
 *      Author: thenightling
 */

#ifndef RRT_CUH_
#define RRT_CUH_

////////////////////////////////////////////
// INCLUDES
////////////////////////////////////////////

// includes from CUDA
#include <curand.h>
#include <curand_kernel.h>

////////////////////////////////////////////
// MACROS
////////////////////////////////////////////

// grid macros
#define ANG_POS_0_MIN -M_PI		// minimum angular position
#define ANG_POS_0_MAX M_PI		// maximum angular position
#define ANG_VEL_0_MIN -10.0		// minimum angular velocity
#define ANG_VEL_0_MAX 10.0		// maximum angular velocity
#define ANG_POS_1_MIN -M_PI		// minimum angular position
#define ANG_POS_1_MAX M_PI		// maximum angular position
#define ANG_VEL_1_MIN -10.0		// minimum angular velocity
#define ANG_VEL_1_MAX 10.0		// maximum angular velocity
#define GRID_X 4
#define GRID_Y 4
#define GRID_Z 4
#define GRID_W 4
#define DELTA_X (ANG_POS_0_MAX-ANG_POS_0_MIN)/(2*GRID_X + 2)
#define DELTA_Y (ANG_VEL_0_MAX-ANG_VEL_0_MIN)/(2*GRID_Y + 2)
#define DELTA_Z (ANG_POS_1_MAX-ANG_POS_1_MIN)/(2*GRID_Z + 2)
#define DELTA_W (ANG_VEL_1_MAX-ANG_VEL_1_MIN)/(2*GRID_W*NUM_BLOCKS + 2)

// RRT algorithm macros
#define NUM_OF_ITERATIONS 300
#define LENGTH_OF_SOLN_PATH 20
#define DIMENSIONS 4
#define NUM_OF_GOAL_STATES 80

// thread macros
#define NUM_BLOCKS 8
#define NUM_THREADS GRID_X * GRID_Y * GRID_Z * GRID_W
#define NUM_RESULTS_PER_THREAD GRID_X * GRID_Y * GRID_Z * GRID_W * NUM_BLOCKS
#define NUM_RESULTS_PER_THREAD_2 DIMENSIONS * LENGTH_OF_SOLN_PATH * NUM_OF_GOAL_STATES

// for graph shortest path finding algorithm
#define NUM_NODES GRID_X * GRID_Y * GRID_Z * GRID_W * NUM_BLOCKS
#define MAX_COST 9999

////////////////////////////////////////////
// FUNCTION HEADERS
////////////////////////////////////////////
__device__ void euclidianDistSquare(double* A, double B[][4], int lengthOfB, double* listOfDistSq);
__device__ int findMin(double array[], int lengthOfArray);
__device__ void acrobotDynamics(double* x, double u, double* xd);
__device__ void matrixInverse(double M[2][2], double invM[2][2]);
__device__ void matrixMultiply(double A[2][2], double B[2], double C[2]);


////////////////////////////////////////////
// KERNEL HEADERS
////////////////////////////////////////////
__global__ void RNG_setup_kernel(curandState *state);
__global__ void init_adj_matrix_kernel(int * adjacency_matrix);
__global__ void RRT_kernel(curandState *my_curandstate, int *adjacency_matrix, double * path_solutions, double * control_solutions, double * tmp);



#endif /* RRT_CUH_ */
