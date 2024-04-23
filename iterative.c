#include <immintrin.h>
#include <limits.h>

#include"min.h"

int arraymin(int* array, int size) {
	int vsize = size & ~0xF;
	unsigned i = 0;
	int min = INT_MAX;
	__m512i min_vec = _mm512_set1_epi32(min);
	
	// batch compare 16 int blocks
	for (; i < vsize; i += 16) {
		__m512i a = _mm512_load_si512(array + i);
		__mmask16 res = _mm512_cmplt_epi32_mask(a, min_vec);

		if (res) {
			// minimum exists, find it!
			min = _mm512_reduce_min_epi32(a);
			min_vec = _mm512_set1_epi32(min);
		}
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
	int index = -1; 
	int vsize = size & ~0xF;
	unsigned i = 0, min_block = 0;
	int min = INT_MAX;
	__m512i min_vec = _mm512_set1_epi32(min);
 
	// batch compare, keeping track of what block the minimum was found
	for (; i < vsize; i += 16) {
		__m512i a = _mm512_load_si512(array + i);
		__mmask16 res = _mm512_cmplt_epi32_mask(a, min_vec);

		if (res) {
			// minimum exists, find it!
			min = _mm512_reduce_min_epi32(a);
			min_vec = _mm512_set1_epi32(min);
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
	__m512i a = _mm512_load_si512(array + min_block);
	__m512i b = _mm512_set1_epi32(min);
	__mmask16 c = _mm512_cmpeq_epi32_mask(a, b);

	return min_block + __tzcnt_u32(c);
}
