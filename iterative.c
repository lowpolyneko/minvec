#include <immintrin.h>
#include <limits.h>

#include"min.h"

int arraymin(int* array, int size) {
	int min = INT_MAX;
	int vsize = size & ~0xF;
	unsigned i = 0;
	
	for (; i < vsize; i += 16) {
		__m512i a = _mm512_load_si512(array + i);
		int res = _mm512_reduce_min_epi32(a);

		if (res < min)
			min = res;
	}

	for (; i < size; ++i) {
		if (array[i] < min)
			min = array[i];
	}

	return min;
}

int arraymin64(int* array) {
	return arraymin(array,64);
}

int minindex(int* array, int size) {
	int min=array[0];
	int index=0;
	for(unsigned i=1;i<size;i++) {
		if(array[i]<min) {
			min=array[i];
			index=i;
		}
	}
	return index;
}
