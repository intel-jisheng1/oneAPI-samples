// ==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// This agreement shall be governed in all respects by the laws of the State of
// California and by the laws of the United States of America.

////////////////////////////////////////////////////////////////////////////////
//
// CRRSolver CPU/FPGA Accelerator Demo Program
//
////////////////////////////////////////////////////////////////////////////////
//
// This design implements a simple Cox-Ross-Rubinstein(CRR) binomial tree model
// with Greeks for American exercise options.
//
////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <iomanip>

#include "CRR_common.hpp"

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace std;
using namespace sycl;

class CRRSolver;

void CRRCompute(const InputData &inp, const CRRInParams &in_params, CRRResParams &res_params, bool optionType0) {

   double pvalue[kMaxNSteps3];

   int n_steps = in_params.n_steps;
   if (optionType0) n_steps += kExtraStepsForOptionType0;

   double umin = in_params.umin;
   // option value computed at each final node
   for (int i = 0; i <= n_steps; i++, umin *= in_params.u2) {
     pvalue[i] = sycl::fmax(inp.cp * (umin - inp.strike), 0.0);
   }
   
   umin = in_params.umin;
   // backward recursion to evaluate option price
   for (int i = n_steps - 1; i >= 0; i--) {
     umin *= in_params.u;
     for (int j = 0; j <= n_steps - 1; j++, umin *= in_params.u2) {
        double temp = pvalue[j];
        double value1 = in_params.c1 * temp + in_params.c2 * pvalue[j + 1];
        double value2 = inp.cp * (umin - inp.strike);
        double result = (j <= i) ? sycl::fmax(value1, value2) : temp;
        pvalue[j] = result;

        if(j == 0) {
          res_params.val = result;
        }
        if(i == 4 && j == 2) {
          // pgreek will only be used if this option is of type 0
          res_params.pgreek[3] = result;
        }
        if(i == 2 && j <= 2) {
          res_params.pgreek[j] = result;
        }
     }
   }
}

double CrrSolver(const int n_crrs,
                 const vector<InputData> &inp,
                 const vector<CRRInParams> &in_params,
                 vector<CRRResParams> &res_params,
                 queue &q) {
  dpc_common::TimeInterval timer;

  buffer inp_params(inp);
  buffer i_params(in_params);
  buffer r_params(res_params);

  q.submit([&](handler &h) {
    accessor accessor_inp(inp_params, h, read_only);
    accessor accessor_i(i_params, h, read_only);
    accessor accessor_r(r_params, h,  write_only, no_init);

    h.single_task<CRRSolver>([=]() [[intel::kernel_args_restrict]] {
      [[intel::loop_coalesce(2)]]
      for (int i = 0; i < n_crrs; ++i) {
        InputData inp = accessor_inp[i];

        for (int j = 0; j < kNumOptionTypes; j++) {
          int k = i*kNumOptionTypes + j;
          CRRInParams vals = accessor_i[k];

          CRRResParams res_params;
          CRRCompute(inp, vals, res_params, (j == 0) /* option type 0 */);

          accessor_r[k] = res_params;
        }
      }
    });
  });

  double diff = timer.Elapsed();
  return diff;
}

void ReadInputFromFile(ifstream &input_file, vector<InputData> &inp) {
  string line_of_args;
  while (getline(input_file, line_of_args)) {
    InputData temp;
    istringstream line_of_args_ss(line_of_args);
    line_of_args_ss >> temp.n_steps;
    line_of_args_ss.ignore(1, ',');
    line_of_args_ss >> temp.cp;
    line_of_args_ss.ignore(1, ',');
    line_of_args_ss >> temp.spot;
    line_of_args_ss.ignore(1, ',');
    line_of_args_ss >> temp.fwd;
    line_of_args_ss.ignore(1, ',');
    line_of_args_ss >> temp.strike;
    line_of_args_ss.ignore(1, ',');
    line_of_args_ss >> temp.vol;
    line_of_args_ss.ignore(1, ',');
    line_of_args_ss >> temp.df;
    line_of_args_ss.ignore(1, ',');
    line_of_args_ss >> temp.t;

    inp.push_back(temp);
  }
}

static string ToStringWithPrecision(const double value, const int p = 6) {
  ostringstream out;
  out.precision(p);
  out << std::fixed << value;
  return out.str();
}

void WriteOutputToFile(ofstream &output_file, const vector<OutputRes> &outp) {
  size_t n = outp.size();
  for (size_t i = 0; i < n; ++i) {
    OutputRes temp;
    temp = outp[i];
    string line = ToStringWithPrecision(temp.value, 12) + " " +
                  ToStringWithPrecision(temp.delta, 12) + " " +
                  ToStringWithPrecision(temp.gamma, 12) + " " +
                  ToStringWithPrecision(temp.vega, 12) + " " +
                  ToStringWithPrecision(temp.theta, 12) + " " +
                  ToStringWithPrecision(temp.rho, 12) + "\n";

    output_file << line;
  }
}

void ReadGoldenResultFromFile(ifstream &input_file, vector<OutputRes> &golden) {
  string line_of_res;
  while (getline(input_file, line_of_res)) {
    OutputRes temp;
    istringstream line_of_res_ss(line_of_res);
    line_of_res_ss >> temp.value;
    line_of_res_ss.ignore(1, ' ');
    line_of_res_ss >> temp.delta;
    line_of_res_ss.ignore(1, ' ');
    line_of_res_ss >> temp.gamma;
    line_of_res_ss.ignore(1, ' ');
    line_of_res_ss >> temp.vega;
    line_of_res_ss.ignore(1, ' ');
    line_of_res_ss >> temp.theta;
    line_of_res_ss.ignore(1, ' ');
    line_of_res_ss >> temp.rho;
    golden.push_back(temp);
  }
}

// Used in parsing of command-line arguments.
// Precondition: out_val is a buffer of maxchars elements.
// Upon successful return, out_val contains the string following 'in_name'
// in 'in_arg'.
bool GetArgValue(const char *in_arg, const char *in_name, char *out_val,
                      size_t maxchars) {
  string arg_string(in_arg);
  size_t found = arg_string.find(in_name, 0, strlen(in_name));
  if (found != string::npos) {
    const char *sptr = &in_arg[strlen(in_name)];
    for (int i = 0; i < maxchars - 1; i++) {
      char ch = sptr[i];
      switch (ch) {
        case ' ':
        case '\t':
        case '\0':
          out_val[i] = 0;
          return true;
          break;
        default:
          out_val[i] = ch;
          break;
      }
    }
    return true;
  }
  return false;
}

// Generate probability of increase, increase factor, etc.
// from the option's description in the input file
CRRInParams PrepareOptionType(int type, const InputData &inp) {
  CRRInParams in_params;
  in_params.n_steps = inp.n_steps;
  double d_df = exp(-inp.t * kEpsilon);

  double r;
  if (type == 0 || type == 2)
    r =  pow(inp.df, 1.0 / inp.n_steps);
  else
    r = pow(inp.df * d_df, 1.0 / inp.n_steps);

  if (type == 0 || type == 1)
    in_params.u = exp(inp.vol * sqrt(inp.t / inp.n_steps));
  else
    in_params.u = exp((inp.vol + kEpsilon) * sqrt(inp.t / inp.n_steps));
  in_params.u2 = in_params.u * in_params.u;

  if (type == 0)
    in_params.umin = inp.spot * pow(1 / in_params.u, inp.n_steps + kExtraStepsForOptionType0);
  else
    in_params.umin = inp.spot * pow(1 / in_params.u, inp.n_steps);

  if (type == 0 || type == 2)
    in_params.c1 =
      r * (in_params.u - pow(inp.fwd / inp.spot, 1.0 / inp.n_steps)) /
      (in_params.u - 1 / in_params.u);
  else
    in_params.c1 =
      r *(in_params.u - pow((inp.fwd / d_df) / inp.spot, 1.0 / inp.n_steps)) /
      (in_params.u - 1 / in_params.u);

  in_params.c2 = r - in_params.c1;

  return in_params;
}

// Computes the Premium and Greeks
vector<OutputRes> ComputeOutput(const vector<InputData> &inp,
                                const vector<CRRInParams> &in_params,
                                const vector<CRRResParams> &res_params) {
  assert(in_params.size() == res_params.size());
  assert(in_params.size() == (inp.size() * kNumOptionTypes));
  vector<OutputRes> output_res(inp.size());

  for (int i = 0; i < inp.size(); i++) {
    int j = i*kNumOptionTypes;
    double spot = inp[i].spot;
    double t = inp[i].t;
    int n_steps = inp[i].n_steps;
    double h;
    OutputRes res;
    double u2_0 = in_params[j].u2;
    h = spot * (u2_0 - 1 / u2_0);
    res.value = res_params[j].pgreek[1];
    res.delta = (res_params[j].pgreek[2] - res_params[j].pgreek[0]) / h;
    res.gamma = 2 / h *
                ((res_params[j].pgreek[2] - res_params[j].pgreek[1]) / spot /
                     (u2_0 - 1) -
                 (res_params[j].pgreek[1] - res_params[j].pgreek[0]) / spot /
                     (1 - (1 / u2_0)));
    res.theta =
        (res_params[j].val - res_params[j].pgreek[3]) / 4 / t * n_steps;
    res.rho = (res_params[j+1].val - res.value) / kEpsilon;
    res.vega = (res_params[j+2].val - res.value) / kEpsilon;
    output_res[i] = res;
  }
  return output_res;
}

vector<CRRResParams> ComputeGoldenResult(int n_crrs, const vector<InputData> &inp, const vector<CRRInParams> &vals) {

  vector<CRRResParams> crr_res_params(n_crrs * kNumOptionTypes);
  for (int i = 0; i < n_crrs; ++i) {
    for (int j = 0; j < kNumOptionTypes; j++) {
      int k = i*kNumOptionTypes + j;

      CRRCompute(inp[i], vals[k], crr_res_params[k], (j == 0) /* option type 0 */);
    }
  }
  return crr_res_params;
}

// Perform CRR solving using the CPU and compare FPGA resutls with CPU results
// to test correctness.
bool TestCorrectness(size_t n_crrs, vector<OutputRes> &fpga_res, vector<OutputRes> &cpu_res) {

  // This CRR benchmark ensures a minimum 4 decimal points match
  // between FPGA and CPU "threshold" is chosen to enforce this guarantee
  float threshold = 0.00001;

  std::cout << "\n============= Correctness Test ============= \n";
  std::cout << "Running analytical correctness checks... \n";

  bool pass = true;
  for (int i = 0; i < n_crrs; i++) {
    if (abs(cpu_res[i].value - fpga_res[i].value) > threshold) {
      pass = false;
      std::cout << "fpga_res[" << i << "].value " << " = " << std::fixed
                << std::setprecision(20) << fpga_res[i].value << "\n";
      std::cout << "cpu_res[" << i << "].value " << " = " << std::fixed
                << std::setprecision(20) << cpu_res[i].value << "\n";
      std::cout << "Mismatch detected for value of crr " << i << "\n";
    }
    if (abs(cpu_res[i].delta - fpga_res[i].delta) > threshold) {
      pass = false;
      std::cout << "fpga_res[" << i << "].delta " << " = " << std::fixed
                << std::setprecision(20) << fpga_res[i].delta << "\n";
      std::cout << "cpu_res[" << i << "].delta " << " = " << std::fixed
                << std::setprecision(20) << cpu_res[i].delta << "\n";
      std::cout << "Mismatch detected for value of crr " << i << "\n";
    }
    if (abs(cpu_res[i].gamma - fpga_res[i].gamma) > threshold) {
      pass = false;
      std::cout << "fpga_res[" << i << "].gamma " << " = " << std::fixed
                << std::setprecision(20) << fpga_res[i].gamma << "\n";
      std::cout << "cpu_res[" << i << "].gamma " << " = " << std::fixed
                << std::setprecision(20) << cpu_res[i].gamma << "\n";
      std::cout << "Mismatch detected for value of crr " << "\n";
    }
    if (abs(cpu_res[i].vega - fpga_res[i].vega) > threshold) {
      pass = false;
      std::cout << "fpga_res[" << i << "].vega " << " = " << std::fixed
                << std::setprecision(20) << fpga_res[i].vega << "\n";
      std::cout << "cpu_res[" << i << "].vega " << " = " << std::fixed
                << std::setprecision(20) << cpu_res[i].vega << "\n";
      std::cout << "Mismatch detected for value of crr " << "\n";
    }
    if (abs(cpu_res[i].theta - fpga_res[i].theta) > threshold) {
      pass = false;
      std::cout << "fpga_res[" << i << "].theta " << " = " << std::fixed
                << std::setprecision(20) << fpga_res[i].theta << "\n";
      std::cout << "cpu_res[" << i << "].theta " << " = " << std::fixed
                << std::setprecision(20) << cpu_res[i].theta << "\n";
      std::cout << "Mismatch detected for value of crr " << "\n";
    }
    if (abs(cpu_res[i].rho - fpga_res[i].rho) > threshold) {
      pass = false;
      std::cout << "fpga_res[" << i << "].rho " << " = " << std::fixed
                << std::setprecision(20) << fpga_res[i].rho << "\n";
      std::cout << "cpu_res[" << i << "].rho " << " = " << std::fixed
                << std::setprecision(20) << cpu_res[i].rho << "\n";
      std::cout << "Mismatch detected for value of crr " << "\n";
    }
  }

  std::cout << "CPU-FPGA Equivalence: " << (pass ? "PASS" : "FAIL") << "\n";
  return pass;
}

// Print out the achieved CRR throughput
void TestThroughput(const double &time, const int &n_crrs) {
  std::cout << "\n============= Throughput Test =============\n";

  std::cout << "   Avg throughput:   " << std::fixed << std::setprecision(1)
            << (n_crrs / time) << " assets/s\n";
}

bool ValidateFilename(string filename) {
  std::size_t found = filename.find_last_of(".");
  return (filename.substr(found + 1).compare("csv") == 0);
}

int main(int argc, char *argv[]) {
  string infilename = "";
  string outfilename = "";

  const string default_ifile = "src/data/ordered_inputs.csv";
  const string default_ofile = "src/data/ordered_outputs.csv";
  const string golden_ifile = "src/data/golden_result.csv";

  char *user_infile = nullptr;
  char user_outfile[kMaxFilenameLen] = {0};
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      GetArgValue(argv[i], "-o=", user_outfile, kMaxFilenameLen);
      GetArgValue(argv[i], "--output-file=", user_outfile, kMaxFilenameLen);
    } else {
      user_infile = argv[i];
    }
  }

  infilename = (user_infile == nullptr) ? default_ifile : user_infile;
  // Check input file format
  if (!ValidateFilename(infilename)) {
    std::cerr << "Input file must be a .csv file.\n";
    return 1;
  }
  outfilename = strlen(user_outfile) ? user_outfile : default_ofile;
  if (!ValidateFilename(outfilename)) {
    std::cerr << "Output file must be a .csv file.\n";
    return 1;
  }

  vector<InputData> inp;
  vector<OutputRes> golden_res_from_file;

  ifstream inputFile(infilename);
  if (!inputFile.is_open()) {
    std::cerr << "Input file doesn't exist \n";
    return 1;
  }
  // Read inputs data from input file
  ReadInputFromFile(inputFile, inp);

  ifstream goldenFile(golden_ifile);
  if (!goldenFile.is_open()) {
    std::cerr << "Golden result file doesn't exist\n";
    return 1;
  }
  ReadGoldenResultFromFile(goldenFile, golden_res_from_file);

  const int n_crrs = inp.size();

  vector<CRRInParams> in_params(n_crrs * kNumOptionTypes);

  for (int j = 0; j < n_crrs; ++j) {
    for (int k = 0; k < kNumOptionTypes; k++) {
      in_params[j*kNumOptionTypes + k] = PrepareOptionType(k, inp[j]);
    }
  }

  vector<CRRResParams> fpga_res_params(n_crrs * kNumOptionTypes);
  vector<CRRResParams> res_dummy(n_crrs * kNumOptionTypes);

  double kernel_time = 0;

  try {
#if defined(FPGA_EMULATOR)
    ext::intel::fpga_emulator_selector device_selector;
#else
    ext::intel::fpga_selector device_selector;
#endif

    queue q(device_selector, dpc_common::exception_handler);

    std::cout << "Running on device:  "
              << q.get_device().get_info<info::device::name>().c_str() << "\n";

    device device = q.get_device();
    std::cout << "Device name: "
              << device.get_info<info::device::name>().c_str() << "\n \n \n";

    // warmup run - use this run to warmup accelerator
    CrrSolver(n_crrs, inp, in_params, res_dummy, q);
    // Timed run - profile performance
    kernel_time = CrrSolver(n_crrs, inp, in_params, fpga_res_params, q);

  } catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what() << "\n";
    std::cerr << "   If you are targeting an FPGA hardware, "
                 "ensure that your system is plugged to an FPGA board that is "
                 "set up correctly\n";
    std::cerr << "   If you are targeting the FPGA emulator, compile with "
                 "-DFPGA_EMULATOR\n";
    return 1;
  }

  // Compute the premium and greeks
  vector<OutputRes> fpga_res = ComputeOutput(inp, in_params, fpga_res_params);

  // Compute golden result on CPU
  vector<CRRResParams> cpu_res_params = ComputeGoldenResult(n_crrs, inp, in_params);
  vector<OutputRes> golden = ComputeOutput(inp, in_params, cpu_res_params);

  bool pass = TestCorrectness(n_crrs, fpga_res, golden);

  // Guard against changes creeping into the CPU golden calculation
  bool passes_file_test = TestCorrectness(n_crrs, fpga_res, golden_res_from_file);
  if (!passes_file_test) {
    std::cerr << "FPGA result does not match data/golden_result.csv\n" << std::endl;
  }

  // Write output data to output file
  ofstream outputFile(outfilename);

  WriteOutputToFile(outputFile, fpga_res);

  TestThroughput(kernel_time, n_crrs);
  if ((!pass) || (!passes_file_test)) {
    return 1;
  }

 return 0;
}
