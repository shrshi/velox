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

#pragma once

namespace facebook::velox::cudf_velox::test {

/// Registers cuDF with CPU fallback enabled, so operators that cuDF cannot run
/// stay on CPU, and starts counting, for each operator type, the tasks that
/// finished successfully with that operator. All tasks in the process are
/// counted, including those of verifiers and alternative plans.
void registerCudfAndCountOperators();

/// Logs the counts collected since registerCudfAndCountOperators(), then stops
/// counting and unregisters cuDF. cuDF operator types (e.g. CudfTopNRowNumber)
/// differ from the CPU ones, so the counts show which operators ran on GPU and
/// which fell back to CPU.
void logOperatorCountsAndUnregisterCudf();

} // namespace facebook::velox::cudf_velox::test
