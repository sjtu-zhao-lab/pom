
# POM: An Optimizing Framework on MLIR for Efficient FPGA-based Accelerator Generation
![GitHub License](https://img.shields.io/github/license/sjtu-zhao-lab/pom) ![GitHub Repo stars](https://img.shields.io/github/stars/sjtu-zhao-lab/pom)
## 1. Introduction


POM is an end-to-end optimizing framework on MLIR for efficient FPGA-based accelerator generation.  POM has the following technical contributions:  
- **Programmability**: POM provides a decoupled DSL that enables concise descriptions of functions, loops, and arrays. A rich collection of scheduling primitives is provided for flexible customization, leading to much fewer lines of code while maintaining high performance.
- **Extensibility**: POM explicitly introduces three layers of IR to perform operations at suitable abstraction levels in a unified framework, streamlining the implementation and debugging process and reducing the effort of supporting various optimization methods.
- **Quality**: POM provides a rich set of optimization methods and performs FPGA-oriented schedule operations at proper levels, relieving tight loop-carried dependence, exploiting parallelism, and improving overall performance.
- **Automation**: POM contains a design space exploration (DSE) engine to search for high-performance schedule schemes automatically and efficiently, while also allowing designers to set user-specified schedules.

Please refer to our [HPCA' 24 ](https://arxiv.org/abs/2401.05154)paper for more details: 
```
@inproceedings{zhanghpca2024pom,
  title={An Optimizing Framework on MLIR for Efficient FPGA-based Accelerator Generation},
  author={Weichuang Zhang and Jieru Zhao and Guan Shen and Quan Chen and Chen Chen and Minyi Guo},
  booktitle={2024 IEEE International Symposium on High-Performance Computer Architecture (HPCA)},
  year={2024}
}
```
***
## 2. Installation
### 2.1 Install Prerequisite: isl 
```
git clone git://repo.or.cz/isl.git  
cd isl 
git pull  
git submodule init  
git submodule update  
./autogen.sh  
./configure --with-int=imath  
make  
make check  
make install  
```
More details of isl installation: https://compsys-tools.ens-lyon.fr/iscc/isl.pdf

### 2.2 Install POM
```
git clone --recursive git@github.com:sjtu-zhao-lab/pom.git 
cd pom
```

### 2.3 Code structure
```
pom/
├── scalehls/
│    ├── polygeist /
│    │    ├──  llvm-project/ 
```

## 3. Build 
### 3.1 Build scalehls

```
# Go to scalehls/  
./build-scalehls.sh
```
### 3.2 Build POM

```
# Go to pom/  
./build-pom.sh
```
***
## 4. Getting Started with a GEMM kernel

```
# Go to pom/build/
cmake --build . --target gemm
```

You can run the following instruction to generate an optimized MLIR affine dialect:
```
./bin/gemm
```
The optimized IR is stored at pom/samples/gemm/test_gemm_4096.mlir .  
You can further translate the optimized IR into HLS C code with the following instruction:

```
../scalehls/build/bin/scalehls-opt ../samples/gemm/test_gemm_4096.mlir\
    --scalehls-func-preprocess="top-func=gemm" \
    --scalehls-qor-estimation="target-spec=../samples/config.json" \
    | ../scalehls/build/bin/scalehls-translate -emit-hlscpp >  ../samples/gemm/test_gemm_4096.cpp
```
## Repository Layout
- `include` and `lib` : Compiler implementation
- `scalehls` : the HLS C code generation
- `testbench`: Kernels and applications described with POM DSL
- `samples`: The generated designs

## Related Projects
- [ScaleHLS](https://github.com/hanchenye/scalehls)
- [Tiramisu](https://github.com/Tiramisu-Compiler/tiramisu)
- [MLIR](https://mlir.llvm.org/)
