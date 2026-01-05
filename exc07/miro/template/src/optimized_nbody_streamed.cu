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
const static size_t DEFAULT_SEGMENT_SIZE = 1024 * 1024;
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
sharedNbody_Kernel(int numTargetElements, float4 *bodytargetPos, float3 *bodytargetVelocity,
	int numSourceElements, float4 *bodysourcePos, int targetOffset, int sourceOffset)
{
	// Use the packed values and SOA to optimize load and store operations
	extern __shared__ float4 shared_posMass[];
	
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (elementId < numTargetElements)
	{
		float4 elementPosMass = bodytargetPos[elementId];  // SOA: direct array access
		float3 elementVelocity = bodytargetVelocity[elementId];  // SOA: direct array access
		float3 elementForce = make_float3(0, 0, 0);
		
		// Blocking: Process source particles in chunks of blockDim.x
		for (int tileStart = 0; tileStart < numSourceElements; tileStart += blockDim.x) {
			// Cooperative loading: each thread loads one particle into shared memory
			int loadIdx = tileStart + threadIdx.x;
			if (loadIdx < numSourceElements) {
				shared_posMass[threadIdx.x] = bodysourcePos[loadIdx];
			}
			__syncthreads();
			
			// Process all particles in current tile
			int tileEnd = (tileStart + blockDim.x < numSourceElements) ? tileStart + blockDim.x : numSourceElements;
			int tileSize = tileEnd - tileStart;
			
			// Calculate global indices for self-interaction check
			int globalTargetId = targetOffset + elementId;
			
#if LOOP_UNROLL_FACTOR > 1
			// Loop unrolling: process LOOP_UNROLL_FACTOR particles per iteration
			int k = 0;
			for (; k < tileSize - (LOOP_UNROLL_FACTOR - 1); k += LOOP_UNROLL_FACTOR) {
				#pragma unroll
				for (int u = 0; u < LOOP_UNROLL_FACTOR; u++) {
					int globalSourceId = sourceOffset + tileStart + k + u;
					if (globalTargetId != globalSourceId) {  // Don't interact with self
						bodyBodyInteraction(elementPosMass, shared_posMass[k + u], elementForce);
					}
				}
			}
			// Handle remaining particles
			for (; k < tileSize; k++) {
				int globalSourceId = sourceOffset + tileStart + k;
				if (globalTargetId != globalSourceId) {  // Don't interact with self
					bodyBodyInteraction(elementPosMass, shared_posMass[k], elementForce);
				}
			}
#else
			// Standard loop (no unrolling)
			for (int k = 0; k < tileSize; k++) {
				int globalSourceId = sourceOffset + tileStart + k;
				if (globalTargetId != globalSourceId) {  // Don't interact with self
					bodyBodyInteraction(elementPosMass, shared_posMass[k], elementForce);
				}
			}
#endif
			__syncthreads();
		}
		
		// Update velocity incrementally: v_new = v_old + (F_segment/m) * dt
		calculateSpeed(elementPosMass.w, elementVelocity, elementForce);
		bodytargetVelocity[elementId] = elementVelocity;  // Store updated velocity
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

	ChTimer totalTimer;

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
	// Segment size (default 1MB = 1024*1024 bytes)
	int segmentSizeInt = 0;
	chCommandLineGet<int>(&segmentSizeInt, "segment-size", argc, argv);
	size_t segmentSize = segmentSizeInt != 0 ? static_cast<size_t>(segmentSizeInt) : DEFAULT_SEGMENT_SIZE;

	float4* d_target_posMass;
	float3* d_target_velocity;
	cudaMalloc((void**)&d_target_posMass, segmentSize);
	cudaMalloc((void**)&d_target_velocity, segmentSize);

	float4* d_source_posMass;
	float3* d_source_velocity;
	cudaMalloc((void**)&d_source_posMass, segmentSize);
	cudaMalloc((void**)&d_source_velocity, segmentSize);

	if (h_particles == NULL || d_target_posMass == NULL || d_target_velocity == NULL || d_source_posMass == NULL || d_source_velocity == NULL)
	{
		std::cout << "\033[31m***" << std::endl
				  << "*** Error - Memory allocation failed" << std::endl
				  << "***\033[0m" << std::endl;

		exit(-1);
	}

	// Streams
	// stream0: target pos H2D, force computation kernels, position update kernel, target pos D2H
	// stream1: target velocity H2D, target velocity D2H
	// stream2: source pos H2D
	cudaStream_t stream0, stream1, stream2;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	// Calculate number of segments based on concurrent memory requirements
	// We need 3 segments concurrently on GPU: target_pos (float4), target_vel (float3), source_pos (float4)
	// Accounting for overhead: 32 bytes per element
	// Available memory per segment set: 3 * segmentSize (for 3 concurrent segments)
	// Number of segment sets needed: ceil((32 * numElements) / (3 * segmentSize))
	int numSegments = (32 * numElements + 3 * segmentSize - 1) / (3 * segmentSize);
	int numElementsPerSegment = segmentSize / sizeof(float4);

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
	gridSize = (numElementsPerSegment + blockSize - 1) / blockSize;

	dim3 grid_dim = dim3(gridSize);
	dim3 block_dim = dim3(blockSize);
	size_t sharedMemSize = sizeof(float4) * blockSize;

	std::cout << "***" << std::endl;
	std::cout << "*** Grid: " << gridSize << std::endl;
	std::cout << "*** Block: " << blockSize << std::endl;
	std::cout << "***" << std::endl;

	bool silent = chCommandLineGetBool("silent", argc, argv);

	totalTimer.start();
	
	// Process in segments: for each target segment, compute forces against all source segments
	for (int i = 0; i < numIterations; i++) {
		for (int targetSeg = 0; targetSeg < numSegments; targetSeg++) {
			// Calculate actual segment sizes (handle partial last segment)
			int targetStart = targetSeg * numElementsPerSegment;
			int targetSize = (targetStart + numElementsPerSegment <= numElements) 
				? numElementsPerSegment 
				: numElements - targetStart;
			size_t targetBytes = targetSize * sizeof(float4);
			size_t targetVelocityBytes = targetSize * sizeof(float3);
			
			// Grid size for this segment
			int targetGridSize = (targetSize + blockSize - 1) / blockSize;
			dim3 targetGridDim = dim3(targetGridSize);
			
			// Copy target segment to device (overlap with previous computation)
			cudaMemcpyAsync(d_target_posMass, h_particles->posMass + targetStart, 
				targetBytes, cudaMemcpyHostToDevice, stream0);
			cudaMemcpyAsync(d_target_velocity, h_particles->velocity + targetStart, 
				targetVelocityBytes, cudaMemcpyHostToDevice, stream1); // was stream1 in version with streaming, but brought no performance improvement
			
			// For each source segment, compute forces and update velocity incrementally
			for (int sourceSeg = 0; sourceSeg < numSegments; sourceSeg++) {
				int sourceStart = sourceSeg * numElementsPerSegment;
				int sourceSize = (sourceStart + numElementsPerSegment <= numElements) 
					? numElementsPerSegment 
					: numElements - sourceStart;
				size_t sourceBytes = sourceSize * sizeof(float4);
				
				// Copy source segment to device (use stream2 - can start immediately, overlaps with target copies)
				cudaMemcpyAsync(d_source_posMass, h_particles->posMass + sourceStart, 
					sourceBytes, cudaMemcpyHostToDevice, stream2); // was stream2 in version with streaming, but brought no performance improvement
				
				// Wait for target and source data to be ready before kernel
				// Only synchronize when we actually need the data (before kernel launch)
				cudaStreamSynchronize(stream0);
				cudaStreamSynchronize(stream1); // was stream1 in version with streaming, but brought no performance improvement
				cudaStreamSynchronize(stream1); //was stream2 in version with streaming, but brought no performance improvement
				
				// Launch kernel: target segment interacts with source segment
				// Kernel computes forces from this source segment and immediately updates velocity
				sharedNbody_Kernel<<<targetGridDim, block_dim, sharedMemSize, stream0>>>(
					targetSize, d_target_posMass, d_target_velocity,
					sourceSize, d_source_posMass, targetStart, sourceStart);
			}
			
			// Wait for all force computations and velocity updates to complete
			cudaStreamSynchronize(stream0);
			
			// Update positions (on stream0, same as target_pos H2D) - only reads velocities, doesn't modify them
			updatePosition_Kernel<<<targetGridDim, block_dim, 0, stream0>>>(
				targetSize, d_target_posMass, d_target_velocity);
			
			// Copy results back to host
			cudaMemcpyAsync(h_particles->velocity + targetStart, d_target_velocity, 
				targetVelocityBytes, cudaMemcpyDeviceToHost, stream1); // was stream1 in version with streaming, but brought no performance improvement
			cudaMemcpyAsync(h_particles->posMass + targetStart, d_target_posMass, 
				targetBytes, cudaMemcpyDeviceToHost, stream0);
		}
	}

	cudaDeviceSynchronize();
	totalTimer.stop();
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		std::cout << "\033[31m***" << std::endl
				<< "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
				<< std::endl
				<< "***\033[0m" << std::endl;

		return -1;
	}

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

	cudaFree(d_target_posMass);
	cudaFree(d_target_velocity);
	cudaFree(d_source_posMass);
	cudaFree(d_source_velocity);
	
	// Destroy streams
	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

	// Print Meassurement Results
	std::cout << "***" << std::endl
			  << "*** Results:" << std::endl
			  << "***    Num Elements: " << numElements << std::endl
			  << "***    Num Iterations: " << numIterations << std::endl
			  << "***    Threads per block: " << blockSize << std::endl
			  << "***    Time to Finnish: " << 1e3 * totalTimer.getTime()
			  << " ms" << std::endl
			  << "***" << std::endl;
	return 0;
}

void printHelp(char *argv)
{
	std::cout << "Help:" << std::endl
			  << "  Usage: " << std::endl
			  << "  " << argv << " [-p] [-s <num-elements>] [-t <threads_per_block>] [--segment-size <size>]"
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
			  << "  --segment-size <size>" << std::endl
			  << "    Segment size in bytes for streaming (default: 1MB)" << std::endl
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
