/******************************************************************************
 *
 *           XXXII Heidelberg Physics Graduate Days - GPU Computing
 *
 *                 Gruppe : gpu01
 *
 *                   File : main.cu
 *
 *                Purpose : n-Body Computation
 *
 ******************************************************************************/

#include <cmath>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <chCommandLine.h>
#include <chTimer.hpp>
#include <cstdio>
#include <iomanip>

const static int DEFAULT_NUM_ELEMENTS = 1024;
const static int DEFAULT_NUM_ITERATIONS = 100;
const static int DEFAULT_BLOCK_DIM = 128;

const static float TIMESTEP = 1e-6;	  // s
const static float GAMMA = 6.673e-11; // (Nm^2)/(kg^2)

// Loop unrolling factor for inner tile processing loop
// Set to 1 to disable unrolling, higher values (2, 4, 8) for unrolling
#define LOOP_UNROLL_FACTOR 4

//
// Structures
//
// Here with two AOS (arrays of structures).
//
// Structure of a single body, with position (x, y, z) and mass (w), as well as velocity (v_x, v_y, v_z)
/*
struct Body_t
{
	float4 posMass;   // x = x, y = y, z = z, w = mass
	float3 velocity;  // x = v_x, y = v_y, z = v_z

	Body_t() : posMass(make_float4(0, 0, 0, 0)), velocity(make_float3(0, 0, 0)) {}
};
*/


struct particles {
	float4* posMass;
	float3* velocity;

	particles(int numElements) : posMass(new float4[numElements]), velocity(new float3[numElements]) {}

	~particles()
	{
		delete[] posMass;
		delete[] velocity;
	}
};

//
// Function Prototypes
//
void printHelp(char *);
void printElement(float4 *posMass, float3 *velocity, int elementId, int iteration);

//
// Device Functions
//

//
// Calculate the Distance of two points
//
__device__ float
getDistance(float4 a, float4 b)
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

//
// Calculate the forces between two bodies
//


__device__ float3 normalize(float3 v)
{
	float length = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	if (length > 0.0f)
	{
		v.x /= length;
		v.y /= length;
		v.z /= length;
	}
	return v;
}

__device__ void
bodyBodyInteraction(float4 bodyA, float4 bodyB, float3 &force)
{
	float distance = getDistance(bodyA, bodyB);

	if (distance == 0)
		return;

	float forceMagnitude = (GAMMA * bodyA.w * bodyB.w) / (distance * distance);
	float3 forceDirection = make_float3(bodyB.x - bodyA.x, bodyB.y - bodyA.y, bodyB.z - bodyA.z);
	forceDirection = normalize(forceDirection);
	// Multiply force magnitude by direction (component-wise)
	force.x += forceMagnitude * forceDirection.x;
	force.y += forceMagnitude * forceDirection.y;
	force.z += forceMagnitude * forceDirection.z;
}

//
// Calculate the new velocity of one particle
//
__device__ void
calculateSpeed(float mass, float3 &currentSpeed, float3 force)
{
	float dt_over_mass = TIMESTEP / mass;
	currentSpeed.x += force.x * dt_over_mass;
	currentSpeed.y += force.y * dt_over_mass;
	currentSpeed.z += force.z * dt_over_mass;
}

//
// n-Body Kernel for the speed calculation
//


__global__ void
sharedNbody_Kernel(int numElements, float4 *bodyPos, float3 *bodyVelocity)
{
	// Use the packed values and SOA to optimize load and store operations
	extern __shared__ float4 shared_posMass[];
	
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (elementId < numElements)
	{
		float4 elementPosMass = bodyPos[elementId];  // SOA: direct array access
		float3 elementVelocity = bodyVelocity[elementId];  // SOA: direct array access
		float3 elementForce = make_float3(0, 0, 0);
		
		// Blocking: Process particles in chunks of blockDim.x
		for (int tileStart = 0; tileStart < numElements; tileStart += blockDim.x) {
			// Cooperative loading: each thread loads one particle into shared memory
			int loadIdx = tileStart + threadIdx.x;
			if (loadIdx < numElements) {
				shared_posMass[threadIdx.x] = bodyPos[loadIdx];
			}
			__syncthreads();
			
			// Process all particles in current tile
			int tileEnd = (tileStart + blockDim.x < numElements) ? tileStart + blockDim.x : numElements;
			int tileSize = tileEnd - tileStart;
			
#if LOOP_UNROLL_FACTOR > 1
			// Loop unrolling: process LOOP_UNROLL_FACTOR particles per iteration
			int k = 0;
			for (; k < tileSize - (LOOP_UNROLL_FACTOR - 1); k += LOOP_UNROLL_FACTOR) {
				#pragma unroll
				for (int u = 0; u < LOOP_UNROLL_FACTOR; u++) {
					int j = tileStart + k + u;
					if (elementId != j) {  // Don't interact with self
						bodyBodyInteraction(elementPosMass, shared_posMass[k + u], elementForce);
					}
				}
			}
			// Handle remaining particles
			for (; k < tileSize; k++) {
				int j = tileStart + k;
				if (elementId != j) {  // Don't interact with self
					bodyBodyInteraction(elementPosMass, shared_posMass[k], elementForce);
				}
			}
#else
			// Standard loop (no unrolling)
			for (int k = 0; k < tileSize; k++) {
				int j = tileStart + k;
				if (elementId != j) {  // Don't interact with self
					bodyBodyInteraction(elementPosMass, shared_posMass[k], elementForce);
				}
			}
#endif
			__syncthreads();
		}
		
		calculateSpeed(elementPosMass.w, elementVelocity, elementForce);
		bodyVelocity[elementId] = elementVelocity;  // SOA: direct array write
	}
}

//
// n-Body Kernel to update the position
// Neended to prevent write-after-read-hazards
//
__global__ void
updatePosition_Kernel(int numElements, float4 *bodyPos, float3 *bodyVelocity)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	if (elementId < numElements)
	{
		bodyPos[elementId].x += bodyVelocity[elementId].x * TIMESTEP;
		bodyPos[elementId].y += bodyVelocity[elementId].y * TIMESTEP;
		bodyPos[elementId].z += bodyVelocity[elementId].z * TIMESTEP;	
	}
}

//
// Main
//
int main(int argc, char *argv[])
{
	bool showHelp = chCommandLineGetBool("h", argc, argv);
	if (!showHelp)
	{
		showHelp = chCommandLineGetBool("help", argc, argv);
	}

	if (showHelp)
	{
		printHelp(argv[0]);
		exit(0);
	}

	std::cout << "***" << std::endl
			  << "*** Starting ..." << std::endl
			  << "***" << std::endl;

	ChTimer memCpyH2DTimer, memCpyD2HTimer;
	ChTimer kernelTimer;

	//
	// Allocate Memory
	//
	int numElements = 0;
	chCommandLineGet<int>(&numElements, "s", argc, argv);
	chCommandLineGet<int>(&numElements, "size", argc, argv);
	numElements = numElements != 0 ? numElements : DEFAULT_NUM_ELEMENTS;
	//
	// Host Memory
	//
	bool pinnedMemory = chCommandLineGetBool("p", argc, argv);
	if (!pinnedMemory)
	{
		pinnedMemory = chCommandLineGetBool("pinned-memory", argc, argv);
	}

	particles* h_particles;
	if (!pinnedMemory)
	{
		// Pageable
		h_particles = new particles(numElements);
	}
	else
	{
		// Pinned
		void* pinned_posMass = nullptr;
		void* pinned_velocity = nullptr;
		cudaMallocHost(&pinned_posMass, sizeof(float4) * numElements);
		cudaMallocHost(&pinned_velocity, sizeof(float3) * numElements);

		// Allocate struct without allocating arrays (we use pinned memory instead)
		h_particles = static_cast<particles*>(malloc(sizeof(particles)));
		if (h_particles == nullptr || pinned_posMass == nullptr || pinned_velocity == nullptr)
		{
			if (pinned_posMass) cudaFreeHost(pinned_posMass);
			if (pinned_velocity) cudaFreeHost(pinned_velocity);
			std::cout << "\033[31m***" << std::endl
					  << "*** Error - Pinned memory allocation failed" << std::endl
					  << "***\033[0m" << std::endl;
			exit(-1);
		}
		h_particles->posMass = reinterpret_cast<float4*>(pinned_posMass);
		h_particles->velocity = reinterpret_cast<float3*>(pinned_velocity);
	}

	// Init Particles
	//	srand(static_cast<unsigned>(time(0)));
	srand(0); // Always the same random numbers
	for (int i = 0; i < numElements; i++)
	{
		h_particles->posMass[i].x = 1e-8 * static_cast<float>(rand()); // Modify the random values to
		h_particles->posMass[i].y = 1e-8 * static_cast<float>(rand()); // increase the position changes
		h_particles->posMass[i].z = 1e-8 * static_cast<float>(rand()); // and the velocity
		h_particles->posMass[i].w = 1e4 * static_cast<float>(rand());
		h_particles->velocity[i].x = 0.0f;
		h_particles->velocity[i].y = 0.0f;
		h_particles->velocity[i].z = 0.0f;
	}

	printElement(h_particles->posMass, h_particles->velocity, 0, 0);

	// Device Memory
	float4* d_posMass;
	float3* d_velocity;
	cudaMalloc((void**)&d_posMass, sizeof(float4) * numElements);
	cudaMalloc((void**)&d_velocity, sizeof(float3) * numElements);

	if (h_particles == NULL || d_posMass == NULL || d_velocity == NULL)
	{
		std::cout << "\033[31m***" << std::endl
				  << "*** Error - Memory allocation failed" << std::endl
				  << "***\033[0m" << std::endl;

		exit(-1);
	}

	//
	// Copy Data to the Device
	//
	memCpyH2DTimer.start();

	// Copy the raw arrays posMass and velocity to device
	cudaMemcpy(d_posMass, h_particles->posMass, sizeof(float4) * numElements, cudaMemcpyHostToDevice);
	cudaMemcpy(d_velocity, h_particles->velocity, sizeof(float3) * numElements, cudaMemcpyHostToDevice);

	memCpyH2DTimer.stop();

	//
	// Get Kernel Launch Parameters
	//
	int blockSize = 0,
		gridSize = 0,
		numIterations = 0;

	// Number of Iterations
	chCommandLineGet<int>(&numIterations, "i", argc, argv);
	chCommandLineGet<int>(&numIterations, "num-iterations", argc, argv);
	numIterations = numIterations != 0 ? numIterations : DEFAULT_NUM_ITERATIONS;

	// Block Dimension / Threads per Block
	chCommandLineGet<int>(&blockSize, "t", argc, argv);
	chCommandLineGet<int>(&blockSize, "threads-per-block", argc, argv);
	blockSize = blockSize != 0 ? blockSize : DEFAULT_BLOCK_DIM;

	if (blockSize > 1024)
	{
		std::cout << "\033[31m***" << std::endl
				  << "*** Error - The number of threads per block is too big" << std::endl
				  << "***\033[0m" << std::endl;

		exit(-1);
	}

	// Grid size: minimum threads needed to cover all particles
	// Since kernel uses loop pattern, this ensures all particles are processed
	gridSize = (numElements + blockSize - 1) / blockSize;

	dim3 grid_dim = dim3(gridSize);
	dim3 block_dim = dim3(blockSize);

	std::cout << "***" << std::endl;
	std::cout << "*** Grid: " << gridSize << std::endl;
	std::cout << "*** Block: " << blockSize << std::endl;
	std::cout << "***" << std::endl;

	bool silent = chCommandLineGetBool("silent", argc, argv);

	kernelTimer.start();

	// Calculate shared memory size needed per block
	size_t sharedMemSize = blockSize * sizeof(float4);
	
	for (int i = 0; i < numIterations; i++)
	{
		sharedNbody_Kernel<<<grid_dim, block_dim, sharedMemSize>>>(numElements, d_posMass, d_velocity);
		cudaDeviceSynchronize(); // Ensure velocity update completes
		updatePosition_Kernel<<<grid_dim, block_dim>>>(numElements, d_posMass, d_velocity);
		cudaDeviceSynchronize(); // Ensure position update completes

		if (!silent)
		{
			// Copy via device pointers
			cudaMemcpy(h_particles->posMass, d_posMass, numElements * sizeof(float4), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_particles->velocity, d_velocity, numElements * sizeof(float3), cudaMemcpyDeviceToHost);
			printElement(h_particles->posMass, h_particles->velocity, 0, i + 1);
		}
	}

	// Synchronize
	cudaDeviceSynchronize();

	// Check for Errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		std::cout << "\033[31m***" << std::endl
				  << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
				  << std::endl
				  << "***\033[0m" << std::endl;

		return -1;
	}

	kernelTimer.stop();

	//
	// Copy Back Data
	//
	memCpyD2HTimer.start();

	cudaMemcpy(h_particles->posMass, d_posMass, numElements * sizeof(float4), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_particles->velocity, d_velocity, numElements * sizeof(float3), cudaMemcpyDeviceToHost);

	memCpyD2HTimer.stop();

	// Free Memory
	if (!pinnedMemory)
	{
		delete h_particles;
	}
	else
	{
		// Free pinned memory arrays
		cudaFreeHost(h_particles->posMass);
		cudaFreeHost(h_particles->velocity);
		// Free the struct itself (allocated with malloc)
		free(h_particles);
	}

	cudaFree(d_posMass);
	cudaFree(d_velocity);

	// Print Meassurement Results
	std::cout << "***" << std::endl
			  << "*** Results:" << std::endl
			  << "***    Num Elements: " << numElements << std::endl
			  << "***    Num Iterations: " << numIterations << std::endl
			  << "***    Threads per block: " << blockSize << std::endl
			  << "***    Time to Copy to Device: " << 1e3 * memCpyH2DTimer.getTime()
			  << " ms" << std::endl
		<< "***    Copy Bandwidth: "
		<< 1e-9 * memCpyH2DTimer.getBandwidth(numElements * (sizeof(float4) + sizeof(float3)))
		<< " GB/s" << std::endl
		<< "***    Time to Copy from Device: " << 1e3 * memCpyD2HTimer.getTime()
		<< " ms" << std::endl
		<< "***    Copy Bandwidth: "
		<< 1e-9 * memCpyD2HTimer.getBandwidth(numElements * (sizeof(float4) + sizeof(float3)))
			  << " GB/s" << std::endl
			  << "***    Time for n-Body Computation: " << 1e3 * kernelTimer.getTime()
			  << " ms" << std::endl
			  << "***" << std::endl;

	return 0;
}

void printHelp(char *argv)
{
	std::cout << "Help:" << std::endl
			  << "  Usage: " << std::endl
			  << "  " << argv << " [-p] [-s <num-elements>] [-t <threads_per_block>]"
			  << std::endl
			  << "" << std::endl
			  << "  -p|--pinned-memory" << std::endl
			  << "    Use pinned Memory instead of pageable memory" << std::endl
			  << "" << std::endl
			  << "  -s <num-elements>|--size <num-elements>" << std::endl
			  << "    Number of elements (particles)" << std::endl
			  << "" << std::endl
			  << "  -i <num-iterations>|--num-iterations <num-iterations>" << std::endl
			  << "    Number of iterations" << std::endl
			  << "" << std::endl
			  << "  -t <threads_per_block>|--threads-per-block <threads_per_block>"
			  << std::endl
			  << "    The number of threads per block" << std::endl
			  << "" << std::endl
			  << "  --silent"
			  << std::endl
			  << "    Suppress print output during iterations (useful for benchmarking)" << std::endl
			  << "" << std::endl;
}

//
// Print one element
//
void printElement(float4 *posMass, float3 *velocity, int elementId, int iteration)
{
	float4 posMass_val = posMass[elementId];
	float3 velocity_val = velocity[elementId];

	std::cout << "***" << std::endl
			  << "*** Printing Element " << elementId << " in iteration " << iteration << std::endl
			  << "***" << std::endl
			  << "*** Position: <"
			  << std::setw(11) << std::setprecision(9) << posMass_val.x << "|"
			  << std::setw(11) << std::setprecision(9) << posMass_val.y << "|"
			  << std::setw(11) << std::setprecision(9) << posMass_val.z << "> [m]" << std::endl
			  << "*** velocity: <"
			  << std::setw(11) << std::setprecision(9) << velocity_val.x << "|"
			  << std::setw(11) << std::setprecision(9) << velocity_val.y << "|"
			  << std::setw(11) << std::setprecision(9) << velocity_val.z << "> [m/s]" << std::endl
			  << "*** Mass: <"
			  << std::setw(11) << std::setprecision(9) << posMass_val.w << "> [kg]" << std::endl
			  << "***" << std::endl;
}
