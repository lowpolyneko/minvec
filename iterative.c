#include <immintrin.h>
#include <limits.h>

#include"min.h"

int arraymin(int* array, int size) {
	int min = INT_MAX;
	int vsize = size & ~0xF;
	unsigned i = 0;
	
	// batch compare 16 int blocks
	for (; i < vsize; i += 16) {
		__m512i a = _mm512_load_si512(array + i);
		int res = _mm512_reduce_min_epi32(a);

		if (res < min)
			min = res;
	}

	// leftovers
	for (; i < size; ++i)
		if (array[i] < min)
			min = array[i];

	return min;
}

int arraymin64(int* array) {
	return arraymin(array,64);
}

int minindex(int* array, int size) {
	int min = INT_MAX;
	int index = -1; 
	int vsize = size & ~0xF;
	unsigned i = 0, min_block = 0;
 
	// batch compare, keeping track of what block the minimum was found
	for (; i < vsize; i += 16) {
		__m512i a = _mm512_load_si512(array + i);
		int res = _mm512_reduce_min_epi32(a);

		if (res < min) {
			min = res;
			min_block = i;
		}
	}

	// leftovers, setting the index as needed
	for (; i < size; ++i) {
		if (array[i] < min) {
			min = array[i];
			index = i;
		}
	}

	if (index != -1)
		return index;

	// find index from block if not found in leftovers
	for (i = min_block; i < min_block + 15; ++i)
		if (array[i] == min)
			return i;

	return i;
}
