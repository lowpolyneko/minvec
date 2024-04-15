CC := gcc
LD := gcc

ROOT ?= ../..

MAKE := make

CFLAGS += -g -O3 -mavx2 -Winline -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl  -Wall

.PHONY: all clean

BINS = test_vec test_novec test_iterative test_fixed benchmark_vec benchmark_novec benchmark_iterative benchmark_fixed

all: $(BINS)

benchmark_vec: benchmark.c cmin.c
	$(CC) $(CFLAGS) -o $@ $^
benchmark_novec: benchmark.c cmin.c
	$(CC) $(CFLAGS) -fno-tree-vectorize -o $@ $^
benchmark_iterative: benchmark.c iterative.c
	$(CC) $(CFLAGS) -o $@ $^
benchmark_fixed: benchmark.c fixed.c
	$(CC) $(CFLAGS) -DARRAY256 -o $@ $^


test_vec: test.c cmin.c
	$(CC) $(CFLAGS) -o $@ $^
test_novec: test.c cmin.c
	$(CC) $(CFLAGS) -fno-tree-vectorize -o $@ $^
test_iterative: test.c iterative.c
	$(CC) $(CFLAGS) -o $@ $^
test_fixed: test.c fixed.c
	$(CC) $(CFLAGS) -DARRAY256 -o $@ $^

mtbench: mtbench.c cmin.c
	$(CC) $(CFLAGS) -o $@ $^ -lpthread

report.pdf: report.tex
	pdflatex report.tex
	pdflatex report.tex

clean:
	rm -f $(BINS) *.o *.so
