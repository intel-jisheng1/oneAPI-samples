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

#ifndef __CRR_COMMON_H__
#define __CRR_COMMON_H__

constexpr int kMaxFilenameLen = 1024;

// Maximum number of time steps in the binomial tree
constexpr size_t kMaxNSteps  = 8189;
// Increments of kMaxNSteps
constexpr size_t kMaxNSteps1 = kMaxNSteps + 1;
constexpr size_t kMaxNSteps2 = kMaxNSteps + 2;
constexpr size_t kMaxNSteps3 = kMaxNSteps + 3;

// Increment by a small epsilon in order to compute derivative
// of option price with respect to Vol or Interest. The derivatives
// are then used to compute Vega and Rho.
constexpr double kEpsilon  = 0.0001;

// Whenever calculations are made for Option Price 0, need to increment
// nsteps by 2 to ensure all the required derivative prices are calculated.
constexpr size_t kExtraStepsForOptionType0 = 2;

// Each CRR problem is split into 3 subproblems to calculate
// each required option price separately
// Calculate 3 binomial trees, each used for different greeks
// Three different option prices are required to solve each CRR problem
// The following lists why each option price is required:
// [0] : Used to compute Premium, Delta, Gamma and Theta
// [1] : Used to compute Rho
// [2] : Used to compute Vega
constexpr size_t kNumOptionTypes = 3;

// Data structure for original input data.
// The order of fields in this struct matches the order of columns in the
// input csv file.
typedef struct {
  double n_steps; /* n_steps = number of time steps in the binomial tree. */
  int cp;         /* cp = -1 or 1 for Put & Call respectively. */
  double spot;    /* spot = spot price of the underlying. */
  double fwd;     /* fwd = forward price of the underlying. */
  double strike;  /* strike = exercise price of option. */
  double vol;     /* vol = per cent volatility, input as a decimal. */
  double df;      /* df = discount factor to option expiry. */
  double t;       /* t = time in years to the maturity of the option. */

} InputData;

// Pre-processed input for one option type
typedef struct {
  double n_steps; /* n_steps = number of time steps in the binomial tree. */
  double u;       /* u = the increase factor of a up movement in the binomial tree,
                         same for each time step. */
  double u2;      /* u2 = the square of increase factor. */
  double c1;      /* c1 = the probality of a down movement in the binomial tree,
                          same for each time step. */
  double c2;      /* c2 = the probality of a up movement in the binomial tree. */
  double umin;    /* umin = minimum price of the underlying at the maturity. */

} CRRInParams;

// Output of the kernel
typedef struct {
  double pgreek[4];               /* Stores the 4 derivative prices in the binomial tree
                                     required to compute the Premium and Greeks. */
  double val;                     /* option price calculated */

} CRRResParams;

typedef struct {
  double value; /* value = option price. */
  double delta;
  double gamma;
  double vega;
  double theta;
  double rho;
} OutputRes;

#endif
