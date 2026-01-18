CC=gcc
CFLAGS=-O2 -Wall -Wextra

all: gen_vector

gen_vector: gen_vector.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f gen_vector

.PHONY: all clean
