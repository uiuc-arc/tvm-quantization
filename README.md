# Auto-generating Quantized DNN Kernels using TVM

In this project, we will look at TVM, a compiler for deep neural networks (DNNs),
and try out its capability to automatically compile [quantized convolutions](https://medium.com/@joel_34050/quantization-in-deep-learning-478417eab72b) for CPU or GPU.

## Setup

We will need to compile and install TVM, for which you can find instructions [here](https://tvm.apache.org/docs/install/from_source.html).
This repo already contains the source of TVM v0.15 so there is *no need to download/clone TVM source* code again.
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

### Basic Quantization

Our goal is to use TVM compile a quantized ResNet50 down to a exectuable function;
for simplicity, it's not yet necessary to test the accuracy of the compiled model.

- No usage of dataset is necessary at any point in these experiments;
  when TVM calls for inputs to the model for compilation / benchmarking purposes,
  feel free to use *randomized dummy inputs with the correct shape*.

Gather your own ideas for a workflow and write down your implementation.
The [TVM how-to guides](https://tvm.apache.org/docs/how_to/) contains
instructions and code examples on basic operations in TVM,
and the [TVM discussion board](https://discuss.tvm.apache.org) may have more advanced information.

There are roughly the following few major steps:

1. Get an instance of a ResNet50 model implemented in PyTorch. It's available in the `torchvision` package.

2. It's a good idea to try TVM on an un-quantized DNN first.
   Give TVM the network and a sample input to the network,
   and compile the network into a function object that can be called from Python side to produce DNN outputs.

   The TVM how-to guides has complete tutorials on how to do this step.
   Pay attention to the compilation **target**:
   which hardware (CPU? GPU?) the model is being compiled for, and understand how to specify it.
   Compile for GPU, if you have one, or CPU otherwise.

3. Now, quantize the model down to **int8** precision.
   TVM itself has utilities to quantize a DNN before compilation;
   you can find how-tos in the guides and forum.
   Again, you should get a function object that can be called from Python side.

   **Hint**: there is a namespace `tvm.relay.quantize` and everything you need is somewhere in there.

4. Just for your own check -- how can you see the TVM code in the compiled module?
   Did the quantization actually happen, for example, did the datatypes in the code change?

5. Use TVM's utility functions to benchmark the inference time of the quantized model vs. the un-quantized model.

   In this task we will not try to maximize the performance of the quantized DNN,
   but if there is no speedup, you should try to understand it and formulate a guess.
   **Hint**: TVM may print the following when you compile the DNN -- what does it mean?
   > One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.

6. In your quantization setup, how did TVM know that you wanted to quantize to int8?
   Look into that, and vary the number of bits of quantization (the $n$ in int-$n$).
   Searching in forum and peeking the source code of the quantizer class will both help.

   Try out `int8` -> `int4` -> `int2` -> `int1`; note which precisions work.
   When it doesn't work, note exactly which part is failing.
