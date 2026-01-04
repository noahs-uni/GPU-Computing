/******************************************************************************
 *
 *           XXXII Heidelberg Physics Graduate Days - GPU Computing
 *
 *                 Gruppe : 01
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
const static int DEFAULT_NUM_ITERATIONS = 5;
const static int DEFAULT_BLOCK_DIM = 128;

const static float TIMESTEP = 1e-6;	  // s
const static float GAMMA = 6.673e-11; // (Nm^2)/(kg^2)

//
// Structures
//
// Here with two AOS (arrays of structures).
//
struct Body_t
{
	float4 posMass;	 /* x = x */
					 /* y = y */
					 /* z = z */
					 /* w = Mass */
	float3 velocity; /* x = v_x*/
					 /* y = v_y */
					 /* z= v_z */

	Body_t() : posMass(make_float4(0, 0, 0, 0)), velocity(make_float3(0, 0, 0)) {}
};

struct Body_SOA
{
	float4 *posMass;
	float3 *velocity;
};


//
// Function Prototypes
//
void printHelp(char *);
void printElement(Body_t *, int, int);
void sharedPrintElement(Body_SOA *, int, int);

//
// Device Functions
//

//
// Calculate the Distance of two points
//
__device__ float
getDistance(float4 a, float4 b)
{
	float dx = b.x - a.x;
	float dy = b.y - a.y;
	float dz = b.z - a.z;

	return sqrtf(dx * dx + dy * dy + dz * dz);
}

//
// Calculate unit vector from bodyA to bodyB
//
__device__ float3
getUnitVector(float4 bodyA, float4 bodyB, float distance)
{
	float3 unitVector;

	if (distance == 0)
	{
		unitVector.x = 0;
		unitVector.y = 0;
		unitVector.z = 0;
	}
	else
	{
		unitVector.x = (bodyB.x - bodyA.x) / distance;
		unitVector.y = (bodyB.y - bodyA.y) / distance;
		unitVector.z = (bodyB.z - bodyA.z) / distance;
	}

	return unitVector;
}

//
// Calculate the forces between two bodies
//
__device__ void
bodyBodyInteraction(float4 bodyA, float4 bodyB, float3 &force)
{
	float distance = getDistance(bodyA, bodyB);

	if (distance == 0)
		return;

	force = getUnitVector(bodyB, bodyA, distance);

	float forceValue = - GAMMA * bodyA.w * bodyB.w / (distance * distance);
	
	force.x *= forceValue;
	force.y *= forceValue;
	force.z *= forceValue;
}

//
// Calculate the new velocity of one particle
//
__device__ void
calculateSpeed(float mass, float3 &currentSpeed, float3 force)
{
	currentSpeed.x += (force.x / mass) * TIMESTEP;
	currentSpeed.y += (force.y / mass) * TIMESTEP;
	currentSpeed.z += (force.z / mass) * TIMESTEP;
}

//
// n-Body Kernel for the speed calculation
//
__global__ void
simpleNbody_Kernel(int numElements, Body_t *body)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	float4 elementPosMass;
	float3 elementForce;
	float3 elementSpeed;

	if (elementId < numElements)
	{
		elementPosMass = body[elementId].posMass;
		elementSpeed = body[elementId].velocity;
		elementForce = make_float3(0, 0, 0);

		for (int i = 0; i < numElements; i++)
		{
			if (i != elementId)
			{
				bodyBodyInteraction(elementPosMass, body[i].posMass, elementForce);
			}
		}

		calculateSpeed(elementPosMass.w, elementSpeed, elementForce);

		body[elementId].velocity = elementSpeed;
	}
}

__global__ void
sharedNbody_Kernel(int numElements, float4 *bodyPos, float3 *bodySpeed)
{
	// Use the packed values and SOA to optimize load and store operations

	extern __shared__ float4 shPosMass[];
	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements; i += blockDim.x * gridDim.x){
		float3 acc_cumulated[3] = {0};	
		float3 acc[3] = {0};
		float4 myPosMass = ((float4 *) bodyPos)[i];
		float3 mySpeed = ((float3 *) bodySpeed)[i];
		for(int j = 0; j < numElements; j+= blockDim.x){
			shPosMass[threadIdx.x] = ((float4 *) bodyPos)[j + threadIdx.x];
			__syncthreads();
			for(int k = 0; k < blockDim.x; k++){
				float4 otherPosMass = shPosMass[k];
				bodyBodyInteraction(myPosMass, otherPosMass, *((float3 *) acc));
				acc_cumulated->x += acc->x;
				acc_cumulated->y += acc->y;
				acc_cumulated->z += acc->z;
			}
			__syncthreads();
		}
		calculateSpeed(myPosMass.w, mySpeed, *((float3 *) acc_cumulated));
		((float3 *) bodySpeed)[i] = mySpeed;
	}
}

//
// n-Body Kernel to update the position
// Needed to prevent write-after-read-hazards
//
__global__ void
updatePosition_Kernel(int numElements, Body_t *bodies)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	if (elementId < numElements)
	{
		float4 elementPosMass = bodies[elementId].posMass;
		float3 elementSpeed = bodies[elementId].velocity;

		elementPosMass.x += elementSpeed.x * TIMESTEP;
		elementPosMass.y += elementSpeed.y * TIMESTEP;
		elementPosMass.z += elementSpeed.z * TIMESTEP;
		
		bodies[elementId].posMass = elementPosMass;
	}
}

__global__ void
sharedUpdatePosition_Kernel(int numElements, Body_SOA *bodies)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	if (elementId < numElements)
	{
		float4 elementPosMass = bodies->posMass[elementId];
		float3 elementSpeed = bodies->velocity[elementId];

		elementPosMass.x += elementSpeed.x * TIMESTEP;
		elementPosMass.y += elementSpeed.y * TIMESTEP;
		elementPosMass.z += elementSpeed.z * TIMESTEP;
		
		bodies->posMass[elementId] = elementPosMass;
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

	bool verbose = chCommandLineGetBool("v", argc, argv);
	if (!verbose)
	{
		verbose = chCommandLineGetBool("verbose", argc, argv);
	}
	if(verbose)
	{
		std::cout << "***" << std::endl
				<< "*** Starting ..." << std::endl
				<< "***" << std::endl;
	}

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

	bool optimized = chCommandLineGetBool("o", argc, argv);
	if (!optimized)
	{
		optimized = chCommandLineGetBool("optimized", argc, argv);
	}

	if(optimized)
	{
		Body_SOA *h_particles;
		if (!pinnedMemory)
		{
			// Pageable
			h_particles = static_cast<Body_SOA *>(malloc(sizeof(Body_SOA)));
			h_particles->posMass = static_cast<float4 *>(malloc(static_cast<size_t>(numElements * sizeof(float4))));
			h_particles->velocity = static_cast<float3 *>(malloc(static_cast<size_t>(numElements * sizeof(float3))));
		}
		else
		{
			// Pinned
			h_particles = static_cast<Body_SOA *>(malloc(sizeof(Body_SOA)));
			cudaMallocHost(&h_particles->posMass, static_cast<size_t>(numElements * sizeof(float4)));
			cudaMallocHost(&h_particles->velocity, static_cast<size_t>(numElements * sizeof(float3)));
		}
		srand(0);
		for (int i = 0; i < numElements; i++)
		{
			h_particles->posMass[i].x = 1e-8 * static_cast<float>(rand());
			h_particles->posMass[i].y = 1e-8 * static_cast<float>(rand());
			h_particles->posMass[i].z = 1e-8 * static_cast<float>(rand());
			h_particles->posMass[i].w = 1e4 * static_cast<float>(rand());
			h_particles->velocity[i].x = 0.0f;
			h_particles->velocity[i].y = 0.0f;
			h_particles->velocity[i].z = 0.0f;
		}

		Body_SOA *d_particles;
		cudaMalloc(&d_particles, sizeof(Body_SOA));
		cudaMalloc(&d_particles->posMass, static_cast<size_t>(numElements * sizeof(float4)));
		cudaMalloc(&d_particles->velocity, static_cast<size_t>(numElements * sizeof(float3)));
		if (h_particles == NULL || d_particles == NULL)
		{
			std::cout << "\033[31m***" << std::endl
					  << "*** Error - Memory allocation failed" << std::endl
					  << "***\033[0m" << std::endl;

			exit(-1);
		}

		memCpyH2DTimer.start();
		cudaMemcpy(d_particles->posMass, h_particles->posMass, static_cast<size_t>(numElements * sizeof(float4)), cudaMemcpyHostToDevice);
		cudaMemcpy(d_particles->velocity, h_particles->velocity, static_cast<size_t>(numElements * sizeof(float3)), cudaMemcpyHostToDevice);
		memCpyH2DTimer.stop();

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

		gridSize = ceil(static_cast<float>(numElements) / static_cast<float>(blockSize));

		dim3 grid_dim = dim3(gridSize);
		dim3 block_dim = dim3(blockSize);

		if (verbose)
		{	
			std::cout << "***" << std::endl;
			std::cout << "*** Grid: " << gridSize << std::endl;
			std::cout << "*** Block: " << blockSize << std::endl;
			std::cout << "***" << std::endl;
		}

		bool silent = chCommandLineGetBool("silent", argc, argv);

		kernelTimer.start();

		for (int i = 0; i < numIterations; i++)
		{
			sharedNbody_Kernel<<<grid_dim, block_dim>>>(numElements, d_particles->posMass, d_particles->velocity);
			sharedUpdatePosition_Kernel<<<grid_dim, block_dim>>>(numElements, d_particles);

			
			if (!silent)
			{
				cudaMemcpy(h_particles->posMass, d_particles->posMass, static_cast<size_t>(numElements * sizeof(float4)), cudaMemcpyDeviceToHost);
				cudaMemcpy(h_particles->velocity, d_particles->velocity, static_cast<size_t>(numElements * sizeof(float3)), cudaMemcpyDeviceToHost);
				for (int j = 0; j < numElements; j++){
					sharedPrintElement(h_particles, j, i + 1);
				}
				
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

		cudaMemcpy(h_particles, d_particles, static_cast<size_t>(numElements * sizeof(*d_particles)), cudaMemcpyDeviceToHost);

		memCpyD2HTimer.stop();

		// Free Memory
		if (!pinnedMemory)
		{
			free(h_particles);
		}
		else
		{
			cudaFreeHost(h_particles);
		}

		cudaFree(d_particles);

		float interactionsPerSecond = static_cast<float>(numElements) * static_cast<float>(numElements) * static_cast<float>(numIterations) / kernelTimer.getTime();

		// Print Meassurement Results
		if (verbose)
		{
			std::cout << "***" << std::endl
					<< "*** Results:" << std::endl
					<< "***    Num Elements: " << numElements << std::endl
					<< "***    Num Iterations: " << numIterations << std::endl
					<< "***    Threads per block: " << blockSize << std::endl
					<< "***    Time to Copy to Device: " << 1e3 * memCpyH2DTimer.getTime()
					<< " ms" << std::endl
					<< "***    Copy Bandwidth: "
					<< 1e-9 * memCpyH2DTimer.getBandwidth(numElements * sizeof(h_particles))
					<< " GB/s" << std::endl
					<< "***    Time to Copy from Device: " << 1e3 * memCpyD2HTimer.getTime()
					<< " ms" << std::endl
					<< "***    Copy Bandwidth: "
					<< 1e-9 * memCpyD2HTimer.getBandwidth(numElements * sizeof(h_particles))
					<< " GB/s" << std::endl
					<< "***    Time for n-Body Computation: " << 1e3 * kernelTimer.getTime()
					<< " ms" << std::endl
					<< "***" << std::endl;
		} else {
			std::cout << numElements << ","
					<< blockSize << ","
					<< numIterations << ","
					//<< 1e3 * memCpyH2DTimer.getTime() << ","
					//<< 1e-9 * memCpyH2DTimer.getBandwidth(numElements * sizeof(h_particles)) << ","
					//<< 1e3 * memCpyD2HTimer.getTime() << ","
					//<< 1e-9 * memCpyD2HTimer.getBandwidth(numElements * sizeof(h_particles)) << ","
					<< 1e3 * kernelTimer.getTime() << ","
					<< interactionsPerSecond
					<< std::endl;
		}


		return 0;
	}
	Body_t *h_particles;
	if (!pinnedMemory)
	{
		// Pageable
		h_particles = static_cast<Body_t *>(malloc(static_cast<size_t>(numElements * sizeof(*h_particles))));
	}
	else
	{
		// Pinned
		cudaMallocHost(&h_particles, static_cast<size_t>(numElements * sizeof(*h_particles)));
	}

	// Init Particles
	//	srand(static_cast<unsigned>(time(0)));
	srand(0); // Always the same random numbers
	for (int i = 0; i < numElements; i++)
	{
		h_particles[i].posMass.x = 1e-8 * static_cast<float>(rand()); // Modify the random values to
		h_particles[i].posMass.y = 1e-8 * static_cast<float>(rand()); // increase the position changes
		h_particles[i].posMass.z = 1e-8 * static_cast<float>(rand()); // and the velocity
		h_particles[i].posMass.w = 1e4 * static_cast<float>(rand());
		h_particles[i].velocity.x = 0.0f;
		h_particles[i].velocity.y = 0.0f;
		h_particles[i].velocity.z = 0.0f;
	}

	// Device Memory
	Body_t *d_particles;
	cudaMalloc(&d_particles, static_cast<size_t>(numElements * sizeof(*d_particles)));

	if (h_particles == NULL || d_particles == NULL)
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

	cudaMemcpy(d_particles, h_particles, static_cast<size_t>(numElements * sizeof(*d_particles)), cudaMemcpyHostToDevice);

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

	gridSize = ceil(static_cast<float>(numElements) / static_cast<float>(blockSize));

	dim3 grid_dim = dim3(gridSize);
	dim3 block_dim = dim3(blockSize);

	if (verbose)
	{	
		std::cout << "***" << std::endl;
		std::cout << "*** Grid: " << gridSize << std::endl;
		std::cout << "*** Block: " << blockSize << std::endl;
		std::cout << "***" << std::endl;
	}

	bool silent = chCommandLineGetBool("silent", argc, argv);

	kernelTimer.start();

	for (int i = 0; i < numIterations; i++)
	{
		simpleNbody_Kernel<<<grid_dim, block_dim>>>(numElements, d_particles);
		updatePosition_Kernel<<<grid_dim, block_dim>>>(numElements, d_particles);

		
		if (!silent)
		{
			cudaMemcpy(h_particles, d_particles, static_cast<size_t>(numElements * sizeof(*h_particles)), cudaMemcpyDeviceToHost);
			for (int j = 0; j < numElements; j++){
				printElement(h_particles, j, i + 1);
			}
			
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

	cudaMemcpy(h_particles, d_particles, static_cast<size_t>(numElements * sizeof(*d_particles)), cudaMemcpyDeviceToHost);

	memCpyD2HTimer.stop();

	// Free Memory
	if (!pinnedMemory)
	{
		free(h_particles);
	}
	else
	{
		cudaFreeHost(h_particles);
	}

	cudaFree(d_particles);

	float interactionsPerSecond = static_cast<float>(numElements) * static_cast<float>(numElements) * static_cast<float>(numIterations) / kernelTimer.getTime();

	// Print Meassurement Results
	if (verbose)
	{
		std::cout << "***" << std::endl
				<< "*** Results:" << std::endl
				<< "***    Num Elements: " << numElements << std::endl
				<< "***    Num Iterations: " << numIterations << std::endl
				<< "***    Threads per block: " << blockSize << std::endl
				<< "***    Time to Copy to Device: " << 1e3 * memCpyH2DTimer.getTime()
				<< " ms" << std::endl
				<< "***    Copy Bandwidth: "
				<< 1e-9 * memCpyH2DTimer.getBandwidth(numElements * sizeof(h_particles))
				<< " GB/s" << std::endl
				<< "***    Time to Copy from Device: " << 1e3 * memCpyD2HTimer.getTime()
				<< " ms" << std::endl
				<< "***    Copy Bandwidth: "
				<< 1e-9 * memCpyD2HTimer.getBandwidth(numElements * sizeof(h_particles))
				<< " GB/s" << std::endl
				<< "***    Time for n-Body Computation: " << 1e3 * kernelTimer.getTime()
				<< " ms" << std::endl
				<< "***" << std::endl;
	} else {
		std::cout << numElements << ","
				  << blockSize << ","
				  << numIterations << ","
				  //<< 1e3 * memCpyH2DTimer.getTime() << ","
				  //<< 1e-9 * memCpyH2DTimer.getBandwidth(numElements * sizeof(h_particles)) << ","
				  //<< 1e3 * memCpyD2HTimer.getTime() << ","
				  //<< 1e-9 * memCpyD2HTimer.getBandwidth(numElements * sizeof(h_particles)) << ","
				  << 1e3 * kernelTimer.getTime() << ","
				  << interactionsPerSecond
				  << std::endl;
	}

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
			  << "" << std::endl
			  << "  -v|--verbose" << std::endl
			  << "    Verbose output" << std::endl
			  << "" << std::endl
			  << "  -h|--help" << std::endl
			  << "    Show this help" << std::endl
			  << std::endl
			  << "  -o|--optimized" << std::endl
			  << "    Use optimized version with shared memory" << std::endl;
}

//
// Print one element
//
void printElement(Body_t *particles, int elementId, int iteration)
{
	float4 posMass = particles[elementId].posMass;
	float3 velocity = particles[elementId].velocity;

	std::cout << "***" << std::endl
			  << "*** Printing Element " << elementId << " in iteration " << iteration << std::endl
			  << "***" << std::endl
			  << "*** Position: <"
			  << std::setw(11) << std::setprecision(9) << posMass.x << "|"
			  << std::setw(11) << std::setprecision(9) << posMass.y << "|"
			  << std::setw(11) << std::setprecision(9) << posMass.z << "> [m]" << std::endl
			  << "*** velocity: <"
			  << std::setw(11) << std::setprecision(9) << velocity.x << "|"
			  << std::setw(11) << std::setprecision(9) << velocity.y << "|"
			  << std::setw(11) << std::setprecision(9) << velocity.z << "> [m/s]" << std::endl
			  << "*** Mass: <"
			  << std::setw(11) << std::setprecision(9) << posMass.w << "> [kg]" << std::endl
			  << "***" << std::endl;
}

void sharedPrintElement(Body_SOA *particles, int elementId, int iteration)
{
	float4 posMass = particles->posMass[elementId];
	float3 velocity = particles->velocity[elementId];

	std::cout << "***" << std::endl
			  << "*** Printing Element " << elementId << " in iteration " << iteration << std::endl
			  << "***" << std::endl
			  << "*** Position: <"
			  << std::setw(11) << std::setprecision(9) << posMass.x << "|"
			  << std::setw(11) << std::setprecision(9) << posMass.y << "|"
			  << std::setw(11) << std::setprecision(9) << posMass.z << "> [m]" << std::endl
			  << "*** velocity: <"
			  << std::setw(11) << std::setprecision(9) << velocity.x << "|"
			  << std::setw(11) << std::setprecision(9) << velocity.y << "|"
			  << std::setw(11) << std::setprecision(9) << velocity.z << "> [m/s]" << std::endl
			  << "*** Mass: <"
			  << std::setw(11) << std::setprecision(9) << posMass.w << "> [kg]" << std::endl
			  << "***" << std::endl;
}
