#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
using namespace std;

__constant__ int sharpen[9] = {
    0, -1, 0, 
    -1, 5, -1, 
    0, -1, 0
};
__constant__ int ridge[9] = {
	-1, -1, -1,
	-1, 8, -1,
	-1, -1, -1
};


__global__ void kernel(unsigned char* input, unsigned char* output,
	int width, int height, int offset, int total_pixels)
{
	int index = threadIdx.x + offset;
	int filter_index = 0;
	int pixel_value = 0;
	int x = index % width;
	int y = index / width;
	if (index < total_pixels) {
		if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
			output[index] = input[index];
			return;
		}
		for (int i = -1; i<= 1; i++)
			for (int j = -1; j <= 1; j++)
			{
				int neighbor_index = (y + j) * width + (x + i);
				pixel_value += input[neighbor_index] * sharpen[filter_index];
				filter_index++;
			}
		if (pixel_value <= 0)
		{
			output[index] = 0;
		}
			
		else if (pixel_value > 255)
		{
			output[index] = 255;
		}
		else
		{
			output[index] = pixel_value;
		}
			
	}
}

int main()
{
	int width, height, channels;
	unsigned char* input_cpu = stbi_load("image.png", &width, &height, &channels, 1);
	int total_pixels = width * height;
	unsigned char* input_gpu, * output_gpu;
	cudaMalloc(&input_gpu, total_pixels);
	cudaMalloc(&output_gpu, total_pixels);
	cudaMemcpy(input_gpu, input_cpu, total_pixels, cudaMemcpyHostToDevice);
	
	cout << total_pixels << endl;
	int temp = 1;
	for (int i = 0; i < total_pixels; i += 1024) {
		kernel << <1, 1024 >> > (input_gpu, output_gpu, width, height, i, total_pixels);
		cudaDeviceSynchronize();
	}
	unsigned char* output_cpu = new unsigned char[total_pixels];
	cudaMemcpy(output_cpu, output_gpu, total_pixels, cudaMemcpyDeviceToHost);
	stbi_write_png("output_image.png", width, height, 1, output_cpu, width);
	return 0;
}