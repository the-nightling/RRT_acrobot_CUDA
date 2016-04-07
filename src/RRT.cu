////////////////////////////////////////////
// INCLUDES
////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "RRT.cuh"


////////////////////////////////////////////
// CUDA KERNELS
////////////////////////////////////////////

/*
 * Initializes CUDA RNG
 */
__global__ void RNG_setup_kernel(curandState *state) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;		// thread id
	curand_init(1234, idx, 0, &state[idx]);	// using seed 1234 (change to time at a later stage)
}

/*
 * Initializes adjacent matrix
 */
__global__ void init_adj_matrix_kernel(int * adjacency_matrix){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for(int i=0; i < NUM_THREADS*NUM_BLOCKS; i++){
		int index = idx * NUM_THREADS*NUM_BLOCKS + i;
		if(index % (NUM_THREADS*NUM_BLOCKS + 1) == 0){
			adjacency_matrix[index] = 0;
		}else{
			adjacency_matrix[index] = 9999;
			//adjacency_matrix[index] = 0;
		}
	}
}

/*
 * Main kernel; Contains RRT algorithm
 */
__global__ void RRT_kernel(curandState *my_curandstate, int *adjacency_matrix,
		double * path_solutions, double * control_solutions, double* tmp) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;		// thread id

	// computing initial state
	double start_state[] = { ANG_POS_0_MIN, ANG_VEL_0_MIN, ANG_POS_1_MIN, ANG_VEL_1_MIN }; // initial state; angle position measured from x-axis
	start_state[0] += ((idx % GRID_X) * 2 * DELTA_X) + (2 * DELTA_X);
	start_state[1] += (((idx / GRID_X) % GRID_Y) * 2 * DELTA_Y) + (2 * DELTA_Y);
	start_state[2] += (((idx / (GRID_X*GRID_Y)) % GRID_Z) * 2 * DELTA_Z) + (2 * DELTA_Z);
	start_state[3] += (((idx / (GRID_X*GRID_Y*GRID_Z)) % (GRID_W*NUM_BLOCKS)) * 2 * DELTA_W) + (2 * DELTA_W);

	tmp[4*idx] = start_state[0];
	tmp[4*idx+1] = start_state[1];
	tmp[4*idx+2] = start_state[2];
	tmp[4*idx+3] = start_state[3];

	// TODO: automate goal placement around initial state
	double end_state[NUM_OF_GOAL_STATES][DIMENSIONS] = {{0}};
	int goal_idx;
	for(goal_idx = 0; goal_idx < pow(3,DIMENSIONS); goal_idx++)
	{
		end_state[goal_idx][0] = start_state[0] + ((goal_idx%3) - 1)*2*DELTA_X;
		end_state[goal_idx][1] = start_state[1] + (((goal_idx/3)%3) - 1)*2*DELTA_Y;
		end_state[goal_idx][2] = start_state[2] + (((goal_idx/(3*3))%3) - 1)*2*DELTA_Z;
		end_state[goal_idx][3] = start_state[3] + (((goal_idx/(3*3*3))%3) - 1)*2*DELTA_W;
	}


	//double state_limits[2][2] = { { start_state[0] - 3 * DELTA_X, start_state[0] + 3 * DELTA_X }, {start_state[1] - 3 * DELTA_Y, start_state[1] + 3 * DELTA_Y } }; // state limits; angular position between -pi & pi rad; angular velocity between -10 & 10 rad/s
	double state_limits[4][2] = {
			{start_state[0] - 3 * DELTA_X, start_state[0] + 3 * DELTA_X},
			{start_state[1] - 3 * DELTA_Y, start_state[1] + 3 * DELTA_Y},
			{start_state[2] - 3 * DELTA_Z, start_state[2] + 3 * DELTA_Z},
			{start_state[3] - 3 * DELTA_W, start_state[3] + 3 * DELTA_W},
	};  // state limits; angular position between -pi & pi rad; angular velocity between -10 & 10 rad/s

	// control torques to be used: linspace(-5,5,20)
	//*
	double discrete_control_torques[] = {-20,-10,0,10,20};
	//*/
	/*
	double discrete_control_torques[] = { -1.0000, -0.8947, -0.7895, -0.6842, -0.5789, -0.4737, -0.3684, -0.2632, -0.1579, -0.0526,
			1.0000, 0.8947, 0.7895, 0.6842, 0.5789, 0.4737, 0.3684, 0.2632, 0.1579, 0.0526};
	//*/
	int number_of_discrete_torques = (int) (sizeof(discrete_control_torques) / sizeof(discrete_control_torques[0]));

	double time_step = 0.04; // time interval between application of subsequent control torques

	// static memory allocation
	double random_state[DIMENSIONS];        // stores a state
	double next_state[DIMENSIONS];

	double RRT_tree[NUM_OF_ITERATIONS][DIMENSIONS];	// stores tree
	int x, y;
	for (x = 0; x < NUM_OF_ITERATIONS; x++) {	// initialize tree to initial state
		RRT_tree[x][0] = start_state[0];
		RRT_tree[x][1] = start_state[1];
		RRT_tree[x][2] = start_state[2];
		RRT_tree[x][3] = start_state[3];
	}

	//int adjMatrix[NUM_THREADS][NUM_THREADS];
	//memset(adjMatrix, 0, sizeof(int)*NUM_THREADS*NUM_THREADS);

	int parent_state_index[NUM_OF_ITERATIONS]; // stores index of parent state for each state in graph RRT_tree
	int control_action_index[NUM_OF_ITERATIONS]; // stores index of control actions in discrete_control_torques (each state will use a control action value in discrete_control_torques)
	double u_path[NUM_OF_GOAL_STATES][LENGTH_OF_SOLN_PATH]; // stores sequence of control actions (solution to problem)
	double x_path[NUM_OF_GOAL_STATES][LENGTH_OF_SOLN_PATH][DIMENSIONS];
	for (y = 0; y < NUM_OF_GOAL_STATES; y++) {
		for (x = 0; x < LENGTH_OF_SOLN_PATH; x++) {	// initialize tree to initial state
			x_path[y][x][0] = 0;
			x_path[y][x][1] = 0;
			x_path[y][x][2] = 0;
			x_path[y][x][3] = 0;
			u_path[y][x] = 0;
		}
	}
	int state_index = 0;    // stores sequence of states joining initial to goal state
	double temp_achievable_states[5][DIMENSIONS]; // stores temporary achievable states from a particular vertex; 20 is length of discrete_control_torques

	double distance_square_values[NUM_OF_ITERATIONS];  // stores distance square values

	int goal_index;
	int not_found[NUM_OF_GOAL_STATES] = {0};
	for(int i=0; i < NUM_OF_GOAL_STATES;i++)
		not_found[i] = 1;
	int weight = 0;

	double k1[4],k2[4],k3[4],k4[4],kTemp[4];                // for RK4 algorithm

	// keep growing RRT until goal found or run out of iterations
	int iteration;
	for (iteration = 1; iteration < NUM_OF_ITERATIONS; iteration++) {
		// get random state
		random_state[0] = curand_uniform(my_curandstate + idx) * (state_limits[0][1] - state_limits[0][0]) + state_limits[0][0];
		random_state[1] = curand_uniform(my_curandstate + idx) * (state_limits[1][1] - state_limits[1][0]) + state_limits[1][0];
		random_state[2] = curand_uniform(my_curandstate + idx) * (state_limits[2][1] - state_limits[2][0]) + state_limits[2][0];
		random_state[3] = curand_uniform(my_curandstate + idx) * (state_limits[3][1] - state_limits[3][0]) + state_limits[3][0];

		// find distances between that state point and every vertex in RRT
		euclidianDistSquare(random_state, RRT_tree, iteration, distance_square_values);

		// select RRT vertex closest to the state point
		int nearest_state_index = findMin(distance_square_values, iteration);

		// from the closest RRT vertex, compute all the states that can be reached,
		// given the pendulum dynamics and available torques
		int ui;
		for (ui = 0; ui < number_of_discrete_torques; ui++) {
			// using RK4 for dynamics
			            acrobotDynamics(RRT_tree[nearest_state_index],discrete_control_torques[ui],k1);

			            kTemp[0] = RRT_tree[nearest_state_index][0]+0.5*k1[0]*time_step;
			            kTemp[1] = RRT_tree[nearest_state_index][1]+0.5*k1[1]*time_step;
			            kTemp[2] = RRT_tree[nearest_state_index][2]+0.5*k1[2]*time_step;
			            kTemp[3] = RRT_tree[nearest_state_index][3]+0.5*k1[3]*time_step;
			            acrobotDynamics(kTemp,discrete_control_torques[ui],k2);

			            kTemp[0] = RRT_tree[nearest_state_index][0]+0.5*k2[0]*time_step;
			            kTemp[1] = RRT_tree[nearest_state_index][1]+0.5*k2[1]*time_step;
			            kTemp[2] = RRT_tree[nearest_state_index][2]+0.5*k2[2]*time_step;
			            kTemp[3] = RRT_tree[nearest_state_index][3]+0.5*k2[3]*time_step;
			            acrobotDynamics(kTemp,discrete_control_torques[ui],k3);

			            kTemp[0] = RRT_tree[nearest_state_index][0]+k3[0]*time_step;
			            kTemp[1] = RRT_tree[nearest_state_index][1]+k3[1]*time_step;
			            kTemp[2] = RRT_tree[nearest_state_index][2]+k3[2]*time_step;
			            kTemp[3] = RRT_tree[nearest_state_index][3]+k3[3]*time_step;
			            acrobotDynamics(kTemp,discrete_control_torques[ui],k4);

			            temp_achievable_states[ui][0] = RRT_tree[nearest_state_index][0] + time_step*(1.0/6.0)*(k1[0]+2*k2[0]+2*k3[0]+k4[0]);
			            temp_achievable_states[ui][1] = RRT_tree[nearest_state_index][1] + time_step*(1.0/6.0)*(k1[1]+2*k2[1]+2*k3[1]+k4[1]);
			            temp_achievable_states[ui][2] = RRT_tree[nearest_state_index][2] + time_step*(1.0/6.0)*(k1[2]+2*k2[2]+2*k3[2]+k4[2]);
			            temp_achievable_states[ui][3] = RRT_tree[nearest_state_index][3] + time_step*(1.0/6.0)*(k1[3]+2*k2[3]+2*k3[3]+k4[3]);
		}

		// select the closest reachable state point
		euclidianDistSquare(random_state, temp_achievable_states, number_of_discrete_torques, distance_square_values);
		ui = findMin(distance_square_values, number_of_discrete_torques);
		random_state[0] = temp_achievable_states[ui][0];
		random_state[1] = temp_achievable_states[ui][1];
		random_state[2] = temp_achievable_states[ui][2];
		random_state[3] = temp_achievable_states[ui][3];

        // if angular position is greater than pi rads, wrap around
        if(xn[0] > M_PI || xn[0] < -M_PI)
            xn[0] = fmod((xn[0]+M_PI), (2*M_PI)) - M_PI;

        if(xn[2] > M_PI || xn[2] < -M_PI)
            xn[2] = fmod((xn[2]+M_PI), (2*M_PI)) - M_PI;

		// link reachable state point to the nearest vertex in the tree
		RRT_tree[iteration][0] = random_state[0];
		RRT_tree[iteration][1] = random_state[1];
		RRT_tree[iteration][2] = random_state[2];
		RRT_tree[iteration][3] = random_state[3];
		parent_state_index[iteration] = nearest_state_index;
		control_action_index[iteration] = ui;

		// if tree has grown near enough to one of the surrounding goal states
		// set that particular goal state to 'found'
		// save path from initial state to that goal state
		for (goal_index = 0; goal_index < NUM_OF_GOAL_STATES; goal_index++) {
			if (not_found[goal_index] == 1
					&& (random_state[0] <= end_state[goal_index][0] + 0.05)
					&& (random_state[0] >= end_state[goal_index][0] - 0.05)) {
				if ((random_state[1] <= end_state[goal_index][1] + 0.25)
						&& (random_state[1] >= end_state[goal_index][1] - 0.25)) {

					not_found[goal_index] = 0;
					state_index = iteration;
					int length_of_soln = 0;
					while (state_index != 0) {
						u_path[goal_index][length_of_soln] = discrete_control_torques[control_action_index[state_index]];
						x_path[goal_index][length_of_soln][0] = RRT_tree[state_index][0];
						x_path[goal_index][length_of_soln][1] = RRT_tree[state_index][1];
						x_path[goal_index][length_of_soln][2] = RRT_tree[state_index][2];
						x_path[goal_index][length_of_soln][3] = RRT_tree[state_index][3];
						length_of_soln++;

						state_index = parent_state_index[state_index];
					}
				}
			}
		}
	}

	weight = 1;

	// Update adjacency matrix:
	// for each goal state surrounding an initial state,
	// if the goal state has been reached,
	// if tree is growing near border of phase space, check if tree is growing within state space limits
	// set respective flag in adjacency matrix to 1 (or to a weight)
	int k;
	for (k = 0; k < 8; k++) {
		if (not_found[k] == 0) {
			if (k == 0 && idx % GRID_X != 0) {
				if (idx + GRID_X - 1 <= NUM_THREADS * NUM_BLOCKS - 1) {
					adjacency_matrix[idx * NUM_THREADS * 8 + idx + GRID_X - 1] = weight;
				}
			} else if (k == 1) {
				if (idx + GRID_X <= NUM_THREADS * NUM_BLOCKS - 1) {
					adjacency_matrix[idx * NUM_THREADS * 8 + idx + GRID_X] = weight;
				}
			} else if (k == 2 && (idx + 1) % GRID_X != 0) {
				if (idx + GRID_X + 1 <= NUM_THREADS * NUM_BLOCKS - 1) {
					adjacency_matrix[idx * NUM_THREADS * 8 + idx + GRID_X + 1] = weight;
				}
			} else if (k == 3 && idx % GRID_X != 0) {
				if (idx - 1 >= 0) { // don't need that line
					adjacency_matrix[idx * NUM_THREADS * 8 + idx - 1] = weight;
				}
			} else if (k == 4 && (idx + 1) % GRID_X != 0) {
				if (idx + 1 <= NUM_THREADS * NUM_BLOCKS - 1) { // don't need that line
					adjacency_matrix[idx * NUM_THREADS * 8 + idx + 1] = weight;
				}
			} else if (k == 5 && idx % GRID_X != 0) {
				if (idx - GRID_X - 1 >= 0) {
					adjacency_matrix[idx * NUM_THREADS * 8 + idx - GRID_X - 1] = weight;
				}
			} else if (k == 6) {
				if (idx - GRID_X >= 0) {
					adjacency_matrix[idx * NUM_THREADS * 8 + idx - GRID_X] = weight;
				}
			} else if (k == 7 && (idx + 1) % GRID_X != 0) {
				if (idx - GRID_X + 1 >= 0) {
					adjacency_matrix[idx * NUM_THREADS * 8 + idx - GRID_X + 1] = weight;
				}
			}
		}
	}
	//*/

	//* copy path results of algorithm to device results array
	int i, j;
	int num_of_goals = NUM_OF_GOAL_STATES;
	for (j = 0; j < num_of_goals; j++) {
		for (i = 0; i < LENGTH_OF_SOLN_PATH; i++) {
			path_solutions[idx * 2 * num_of_goals * LENGTH_OF_SOLN_PATH + j * 2 * LENGTH_OF_SOLN_PATH + 2 * i] = x_path[j][i][0];
			path_solutions[idx * 2 * num_of_goals * LENGTH_OF_SOLN_PATH + j * 2 * LENGTH_OF_SOLN_PATH + 2 * i + 1] = x_path[j][i][1];
			control_solutions[idx * num_of_goals * LENGTH_OF_SOLN_PATH + j * LENGTH_OF_SOLN_PATH + i] = u_path[j][i];
			if (not_found[j] == 0) {
				if (i == LENGTH_OF_SOLN_PATH - 2) {
					path_solutions[idx * 2 * num_of_goals * LENGTH_OF_SOLN_PATH + j * 2 * LENGTH_OF_SOLN_PATH + 2 * i] = start_state[0];
					path_solutions[idx * 2 * num_of_goals * LENGTH_OF_SOLN_PATH + j * 2 * LENGTH_OF_SOLN_PATH + 2 * i + 1] = start_state[1];
				} else if (i == LENGTH_OF_SOLN_PATH - 1) {
					path_solutions[idx * 2 * num_of_goals * LENGTH_OF_SOLN_PATH + j * 2 * LENGTH_OF_SOLN_PATH + 2 * i] = end_state[j][0];
					path_solutions[idx * 2 * num_of_goals * LENGTH_OF_SOLN_PATH + j * 2 * LENGTH_OF_SOLN_PATH + 2 * i + 1] = end_state[j][1];
				}
			}
		}
	}
	//*/





	/*
	 int i;
	 for (i = 0; i < NUM_RESULTS_PER_THREAD; i++)
	 result[idx * NUM_RESULTS_PER_THREAD + i] = start_state[i];
	 //*/
	/*
	 result[idx * NUM_RESULTS_PER_THREAD + 0] = start_state[0];
	 result[idx * NUM_RESULTS_PER_THREAD + 1] = start_state[1];
	 //*/

}



////////////////////////////////////////////
// HELPER FUNCTIONS
////////////////////////////////////////////
/*
 * computes the euclidian distances squared from point A to every point in array B
 */
__device__ void euclidianDistSquare(double* A, double B[][4], int lengthOfB, double* listOfDistSq)
{
	int i;
	for(i = 0; i < lengthOfB; i++)
		listOfDistSq[i] = pow((B[i][0] - A[0]),2) + pow((B[i][1] - A[1]),2) + pow((B[i][2] - A[2]),2) + pow((B[i][3] - A[3]),2);
}

/*
 * finds the index of the minimum in an array
 */
__device__ int findMin(double array[], int lengthOfArray) {
	int minIndex = 0;

	int i;
	for (i = 0; i < lengthOfArray; i++) {
		if (array[i] < array[minIndex])
			minIndex = i;
	}

	return minIndex;
}

/*
 * Computes x_dot of the acrobot, given x and a control input u
 */
__device__ void acrobotDynamics(double* x, double u, double* xd)
{
	// acrobot parameters
	int m1 = 1;
	int m2 = 1;
	double l1 = 1;
	double l2 = 1;
	double lc1 = l1/2;
	double lc2 = l2/2;
	double Ic1 = (lc1*lc1)/3;
	double Ic2 = (lc2*lc2)/3;
	double I1 = Ic1+m1*lc1*lc1;
	double I2 = Ic2+m2*lc2*lc2;
	double b1 = 0.4;
	double b2 = 0.4;
	double g = 9.8;

	double H[2][2] = {{I1 + I2 + m2*l1*l1 + 2*m2*l1*lc2*cos(x[2]),
			I2 + m2*l1*lc2*cos(x[2])},
			{I2 + m2*l1*lc2*cos(x[2]),
					I2}};

	double C[2][2] = {{-2*m2*l1*lc2*sin(x[2])*x[3] + b1,
			-m2*l1*lc2*sin(x[2])*x[3]},
			{m2*l1*lc2*sin(x[2])*x[1],
					b2}};

	double G[2] = {m1*g*lc1*sin(x[0]) + m2*g*(l1*sin(x[0])+lc2*sin(x[0]+x[2])),
			m2*g*lc2*sin(x[0]+x[2])};

	double B[2] = {0,u};

	double invH[2][2];
	matrixInverse(H,invH);

	double C_qd[2];
	double qd[2] = {x[1],x[3]};
	matrixMultiply(C,qd,C_qd);

	double temp[2];
	temp[0] = B[0] - C_qd[0] - G[0];
	temp[1] = B[1] - C_qd[1] - G[1];

	double qdd[2];
	matrixMultiply(invH,temp,qdd);

	xd[0] = x[1];
	xd[1] = qdd[0];
	xd[2] = x[3];
	xd[3] = qdd[1];
}

/*
 * Computes the matrix inverse of a 2x2 matrix
 */
__device__ void matrixInverse(double M[2][2], double invM[2][2])
{
	double detInv = 1/(M[0][0]*M[1][1] - M[0][1]*M[1][0]);

	invM[0][0] = detInv*M[1][1];
	invM[0][1] = detInv*M[0][1]*-1;
	invM[1][0] = detInv*M[1][0]*-1;
	invM[1][1] = detInv*M[0][0];
}

/*
 * Computes the product between 2x2 and 2x1 matrices
 */
__device__ void matrixMultiply(double A[2][2], double B[2], double C[2])
{
	C[0] = A[0][0]*B[0] + A[0][1]*B[1];
	C[1] = A[1][0]*B[0] + A[1][1]*B[1];
}
