#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/interfaces.hpp>

#include "exception_handler.hpp"

using ValueT = int;
// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class LambdaRegisterMapControlIP;
class LambdaStreamingControlIP;

// offloaded computation
ValueT SomethingComplicated(ValueT val) { return (ValueT)(val * (val + 1)); }

/////////////////////////////////////////

struct FunctorStreamingControlIP {
  // Use the 'conduit' annotation on a kernel argument to specify it to be
  // a streaming kernel argument.
  conduit ValueT *input;
  conduit ValueT *output;
  // Without the annotations, kernel arguments will be inferred to be streaming
  // kernel arguments if the kernel control interface is streaming, and
  // vise-versa.
  size_t n;
  FunctorStreamingControlIP(ValueT *in_, ValueT *out_, size_t N_)
      : input(in_), output(out_), n(N_) {}
  // Use the 'streaming_interface' annotation on a kernel to specify it to be
  // a kernel with streaming kernel control signals.
  streaming_interface void operator()() const {
    for (int i = 0; i < n; i++) {
      output[i] = SomethingComplicated(input[i]);
    }
  }
};

struct FunctorRegisterMapControlIP {
  // Use the 'register_map' annotation on a kernel argument to specify it to be
  // a register map kernel argument.
  register_map ValueT *input;
  // Without the annotations, kernel arguments will be inferred to be register map
  // kernel arguments if the kernel control interface is register map, and
  // vise-versa.
  ValueT *output;
  // A kernel with register map control can also independently have streaming
  // kernel arguments, when annotated by 'conduit'.
  conduit size_t n;
  FunctorRegisterMapControlIP(ValueT *in_, ValueT *out_, size_t N_)
      : input(in_), output(out_), n(N_) {}
  register_map_interface void operator()() const {
    for (int i = 0; i < n; i++) {
      output[i] = SomethingComplicated(input[i]);
    }
  }
};

void TestLambdaRegisterMapControlKernel(sycl::queue &q, ValueT *in, ValueT *out, size_t count) {
  // In the Lambda programming model, all kernel arguments will have the same interface as the 
  // kernel control interface.
  q.single_task<LambdaRegisterMapControlIP>([=] register_map_interface  {
    for (int i = 0; i < count; i++) {
      out[i] = SomethingComplicated(in[i]);
    }
  }).wait();

  std::cout << "\t Done" << std::endl;
}

void TestLambdaStreamingControlKernel(sycl::queue &q, ValueT *in, ValueT *out, size_t count) {
  // In the Lambda programming model, all kernel arguments will have the same interface as the 
  // kernel control interface.
  q.single_task<LambdaStreamingControlIP>([=] streaming_interface  {
    for (int i = 0; i < count; i++) {
      out[i] = SomethingComplicated(in[i]);
    }
  }).wait();

  std::cout << "\t Done" << std::endl;
}

template <typename KernelType>
void TestFunctorKernel(sycl::queue &q, ValueT *in, ValueT *out, size_t count) {
  q.single_task(KernelType{in, out, count}).wait();

  std::cout << "\t Done" << std::endl;
}

int main(int argc, char *argv[]) {
#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#elif defined(FPGA_SIMULATOR)
  sycl::ext::intel::fpga_simulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif

  bool passed = true;

  size_t count = 16;
  if (argc > 1)
    count = atoi(argv[1]);

  if (count <= 0) {
    std::cerr << "ERROR: 'count' must be positive" << std::endl;
    return 1;
  }

  try {
    // create the device queue
    sycl::queue q(device_selector, fpga_tools::exception_handler);

    // make sure the device supports USM device allocations
    sycl::device d = q.get_device();
    if (!d.has(sycl::aspect::usm_host_allocations)) {
      std::cerr << "ERROR: The selected device does not support USM host"
                << " allocations" << std::endl;
      return 1;
    }

    ValueT *in = sycl::malloc_host<ValueT>(count, q);
    ValueT *functorStreamingOut = sycl::malloc_host<ValueT>(count, q);
    ValueT *functorRegisterMapOut = sycl::malloc_host<ValueT>(count, q);
    ValueT *LambdaStreamingOut = sycl::malloc_host<ValueT>(count, q);
    ValueT *LambdaRegisterMapOut = sycl::malloc_host<ValueT>(count, q);
    ValueT *golden = sycl::malloc_host<ValueT>(count, q);

    // create input and golden output data
    for (int i = 0; i < count; i++) {
      in[i] = rand() % 77;
      golden[i] = SomethingComplicated(in[i]);
      functorStreamingOut[i] = 0;
      functorRegisterMapOut[i] = 0;
      LambdaStreamingOut[i] = 0;
      LambdaRegisterMapOut[i] = 0;
    }

    // validation lambda
    auto validate = [](auto &in, auto &out, size_t size) {
      for (int i = 0; i < size; i++) {
        if (out[i] != in[i]) {
          std::cout << "out[" << i << "] != in[" << i << "]"
                    << " (" << out[i] << " != " << in[i] << ")" << std::endl;
          return false;
        }
      }
      return true;
    };

    // Launch the kernel with streaming control implemented in the functor programming model
    std::cout << "Running the kernel with streaming control implemented in the functor programming model" << std::endl;
    TestFunctorKernel<FunctorStreamingControlIP>(q, in, functorStreamingOut, count);
    passed &= validate(golden, functorStreamingOut, count);
    std::cout << std::endl;

    // Launch the kernel with register map control implemented in the functor programming model
    std::cout << "Running the kernel with register map control implemented in the functor programming model" << std::endl;
    TestFunctorKernel<FunctorRegisterMapControlIP>(q, in, functorRegisterMapOut, count);
    passed &= validate(golden, functorRegisterMapOut, count);
    std::cout << std::endl;

    // Launch the kernel with streaming control implemented in the lambda programming model
    std::cout << "Running kernel with streaming control implemented in the lambda programming model" << std::endl;
    TestLambdaStreamingControlKernel(q, in, LambdaStreamingOut, count);
    passed &= validate(golden, LambdaStreamingOut, count);
    std::cout << std::endl;

    // Launch the kernel with register map control implemented in the lambda programming model
    std::cout << "Running kernel with register map control implemented in the lambda programming model" << std::endl;
    TestLambdaRegisterMapControlKernel(q, in, LambdaRegisterMapOut, count);
    passed &= validate(golden, LambdaRegisterMapOut, count);
    std::cout << std::endl;

    sycl::free(in, q);
    sycl::free(functorStreamingOut, q);
    sycl::free(functorRegisterMapOut, q);
    sycl::free(LambdaStreamingOut, q);
    sycl::free(LambdaRegisterMapOut, q);
    sycl::free(golden, q);
  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";
    std::terminate();
  }

  if (passed) {
    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 1;
  }
}
