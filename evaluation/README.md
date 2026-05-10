## CUDA configuration

Set CUDA settings in the shell before running the evaluation Makefiles:

```sh
export CUDA_PATH=/path/to/cuda
export CUDA_ARCH=sm_90
```

`CUDA_HOME` is also accepted as the CUDA root. Each Makefile uses:

```make
CUDA_PATH ?= $(CUDA_HOME)
CUDA_ARCH ?= sm_90
```

You can still override per command when needed:

```sh
make CUDA_PATH=/path/to/cuda CUDA_ARCH=sm_80
```
