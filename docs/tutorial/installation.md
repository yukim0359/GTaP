# Installation

This page prepares the GTaP source tree and builds the GTaP-enabled Clang.

## Tested environment

GTaP has been tested on an NVIDIA GH200 node of the Miyabi-G supercomputer.

| Component | Tested configuration |
| --- | --- |
| GPU | NVIDIA GH200 |
| Compute capability | 9.0 (`sm_90`) |
| GTaP-enabled Clang | Based on LLVM 21.1.8 |
| CUDA Toolkit | 12.9 |
| Linux kernel | `5.14.0-427.13.1.el9_4.aarch64` |

Other NVIDIA GPU and CUDA configurations may work, but this is the currently
verified environment.

## 1. Clone GTaP

Clone the repository and its compiler submodule:

```bash
git clone https://github.com/yukim0359/GTaP.git --recursive
cd GTaP
```

If the repository was cloned without `--recursive`, initialize the submodule
separately:

```bash
git submodule update --init --recursive
```

## 2. Build the GTaP-enabled Clang

GTaP programs are compiled with the GTaP-enabled Clang fork in
[`clang-gtap`](https://github.com/yukim0359/clang-gtap). The following commands
assume that the current directory is the root of the cloned GTaP repository.

### Install the build requirements

The compiler build requires:

- CMake 3.20 or later
- Ninja
- a host C/C++ compiler, such as Clang or GCC

The CUDA Toolkit is required later to compile and run GTaP programs, but not to
build the GTaP-enabled Clang itself.

Check that the build tools are available:

```bash
cmake --version
ninja --version
clang --version
```

### Select the LLVM host target

GTaP needs the LLVM backend for the host CPU and the `NVPTX` backend for CUDA
device code. Check the host architecture:

```bash
uname -m
```

Set `LLVM_HOST_TARGET` from the result:

| `uname -m` | LLVM target |
| --- | --- |
| `aarch64` | `AArch64` |
| `x86_64` | `X86` |

For example, use the following setting on an NVIDIA GH200 system:

```bash
export LLVM_HOST_TARGET=AArch64
```

### Configure Clang

Configure an optimized build in `clang-gtap/build`:

```bash
cmake -G Ninja \
  -S clang-gtap/llvm \
  -B clang-gtap/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DLLVM_TARGETS_TO_BUILD="${LLVM_HOST_TARGET};NVPTX"
```

::: tip Using GCC as the host compiler
`CMAKE_C_COMPILER` and `CMAKE_CXX_COMPILER` select the compiler used to build
the GTaP-enabled Clang. They do not change the resulting compiler: the output
is still Clang with GTaP support. Replace `clang` and `clang++` above with
`gcc` and `g++` when Clang is not available on the host.
:::

### Build and verify Clang

Build the `clang` target:

```bash
ninja -C clang-gtap/build clang
```

The resulting compiler is placed at `clang-gtap/build/bin/clang`, which is also
the default compiler path used by the example Makefiles. Verify the build with:

```bash
./clang-gtap/build/bin/clang --version
```

The command should report a Clang version based on LLVM 21.1.8. The installation
is now ready; continue to [Quickstart](./quickstart).
