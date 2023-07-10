# Auto-generating Quantized DNN Kernels using TVM

In this project, we will look at TVM, a compiler for deep neural networks (DNNs),
and try out its capability to automatically compile [quantized convolutions](https://medium.com/@joel_34050/quantization-in-deep-learning-478417eab72b) for CPU or GPU.

## Setup

We will need to compile and install TVM, for which you can find instructions [here](https://tvm.apache.org/docs/install/from_source.html).
This repo already contains the source of TVM v0.9 so there is *no need to download/clone TVM source* code again.
In short:

- Install dependencies of TVM as specified in the instructions.
  - If your machine *does not have an Nvidia GPU*, you can choose to not use CUDA by changing `tvm/build/config.cmake`: change `set(USE_CUDA ON)` to `set(USE_CUDA OFF)` in that file.

- `cd tvm/build` (relative to the root of this repo), and compile the C++ part of TVM:

  ```bash
  cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
  make -j16
  ```

  (Change `-j16` to any number of threads as needed)

- `cd ../python` (relative to `tvm/build`), and install the Python part of TVM:

    ```bash
    pip install -e .
    ```

## Experiments

Our goal is to use TVM compile a quantized ResNet50 down to a exectuable function;
for brevity, it's not yet necessary to test the accuracy of the compiled model.

- No usage of dataset is necessary at any point in these experiments;
  when TVM calls for inputs to the model for compilation / benchmarking purposes,
  feel free to use *randomized dummy inputs with the correct shape*.

Gather your own ideas for a workflow and write down your implementation.
The [TVM how-to guides](https://tvm.apache.org/docs/how_to/) contains
instructions and code examples on basic operations in TVM,
and the [TVM discussion board](https://discuss.tvm.apache.org) may have more advanced information.

There are roughly the following few major steps:

1. Get an instance of a ResNet50 model implemented in PyTorch.
   It's available in the `torchvision` package and as easy to get as a function call (remember to install the package first).

2. It may be a good idea to try using TVM on plain (un-quantized) DNN first.
   Give TVM the network and a sample input the network takes, to compile the network into a function object that can be called from Python side and gives outputs.

   The TVM how-to guides has complete tutorials on how to do this step.
   Pay attention to *which hardware (CPU? GPU?) the model is being compiled for* and how to specify it.

3. Now, quantize the model down to **int8** precision.
   TVM itself has utilities to quantize a DNN before compilation;
   you can find how-to in the guides and forum.

   Do this for the GPU (if you have one), or CPU otherwise.
   Use TVM utils to benchmark the inference time of the quantized model vs. the un-quantized model.

   We're not (yet) looking to maximize the performance of the DNN with quantization,
   but if there is no speedup, you should look into it and form your own guess.

    - Hint: TVM may print the following only for the quantized case, or for both -- what does it mean?
      > One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.

4. [Bonus] If you used `qconfig` for the previous part, look into how to change the quantization precision,
   which is the number of bits of quantization (the $n$ in int-$n$),
   by looking at the source code of `class QConfig` or search on forum.

   Go down `int8` -> `int4` -> `int2` -> `int1 (bool)`, then followed by non-power-of-2 bits (`int7`, `int6`...),
   and investigate what is supported by TVM and what is failing when it doesn't work.
