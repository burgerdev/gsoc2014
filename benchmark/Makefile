
CC=g++
CCOLD=g++-4.6

CFLAGS=-O3 -Winline -Wall -Wextra -pedantic #-finline-limit=100000
IFLAGS=-I../include -I/usr/include/python2.7
LFLAGS=-lpython2.7

HEADER=../include/mergeLabels.hxx

all: bench46 bench49

bench49: benchmark_merge_labels.cxx $(HEADER) Makefile
	$(CC) $(CFLAGS) $< $(IFLAGS) $(LFLAGS) -o $@

bench46: benchmark_merge_labels.cxx $(HEADER) Makefile
	$(CCOLD) $(CFLAGS) $< $(IFLAGS) $(LFLAGS) -o $@



.PHONY: all

