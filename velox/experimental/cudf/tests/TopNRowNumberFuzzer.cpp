/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/// cuDF TopNRowNumber Fuzzer Runner (CPU fallback enabled)
///
/// Runs the standard TopNRowNumberFuzzer with cuDF registered and CPU fallback
/// enabled. Operators that can run on GPU are routed there by the
/// DriverAdapter; unsupported operators fall back to CPU. It takes the same
/// flags as velox_topn_row_number_fuzzer, except that plans with spilling are
/// tested only with --enable_spill=true.
///
/// Usage:
///   ./velox_cudf_topn_row_number_fuzzer --steps 100 --v=1

#include "velox/exec/fuzzer/TopNRowNumberFuzzer.h"
#include "velox/experimental/cudf/tests/utils/CudfFuzzerUtil.h"

#include "velox/common/memory/SharedArbitrator.h"
#include "velox/exec/fuzzer/DuckQueryRunner.h"
#include "velox/exec/fuzzer/FuzzerUtil.h"
#include "velox/exec/fuzzer/ReferenceQueryRunner.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"
#include "velox/parse/TypeResolver.h"

#include <folly/init/Init.h>
#include <gflags/gflags.h>

DEFINE_int64(
    seed,
    0,
    "Initial seed for random number generator used to reproduce previous "
    "results (0 means start with random seed).");

DEFINE_string(
    presto_url,
    "",
    "Presto coordinator URI along with port. If set, we use Presto "
    "source of truth. Otherwise, use DuckDB. Example: "
    "--presto_url=http://127.0.0.1:8080");

DEFINE_uint32(
    req_timeout_ms,
    1000,
    "Timeout in milliseconds for HTTP requests made to reference DB, "
    "such as Presto. Example: --req_timeout_ms=2000");

DEFINE_int64(allocator_capacity, 8L << 30, "Allocator capacity in bytes.");

DEFINE_int64(arbitrator_capacity, 6L << 30, "Arbitrator capacity in bytes.");

DECLARE_bool(enable_spill);

using namespace facebook::velox;

int main(int argc, char** argv) {
  // cuDF TopNRowNumber and Window do not spill, so spilling runs repeat the
  // non-spilling GPU runs. Disable them unless requested on the command line.
  FLAGS_enable_spill = false;
  folly::Init init(&argc, &argv);
  functions::prestosql::registerAllScalarFunctions();
  parse::registerTypeResolver();
  window::prestosql::registerAllWindowFunctions();
  exec::test::setupMemory(FLAGS_allocator_capacity, FLAGS_arbitrator_capacity);
  cudf_velox::test::registerCudfAndCountOperators();
  std::shared_ptr<memory::MemoryPool> rootPool{
      memory::memoryManager()->addRootPool()};
  auto referenceQueryRunner = exec::test::setupReferenceQueryRunner(
      rootPool.get(),
      FLAGS_presto_url,
      "topn_row_number_fuzzer",
      FLAGS_req_timeout_ms);
  const size_t initialSeed = FLAGS_seed == 0 ? std::time(nullptr) : FLAGS_seed;
  exec::topNRowNumberFuzzer(initialSeed, std::move(referenceQueryRunner));
  cudf_velox::test::logOperatorCountsAndUnregisterCudf();
}
