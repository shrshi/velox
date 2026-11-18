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

#include "velox/common/Casts.h"
#include "velox/exec/JoinBridge.h"
#include "velox/exec/Operator.h"
#include "velox/exec/Task.h"

#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox {

class CudaEvent;

/// The build result a cuDF join build publishes once to its probe drivers.
template <typename Payload>
struct CudfJoinBuildResult {
  /// Each probe host-waits for its own reads before it drops its reference, so
  /// the payload can be freed in any stream's order.
  Payload payload;
  /// Recorded on the build's stream as soon as the payload is ready, not
  /// lazily by a probe, so it captures that point before cudfGlobalStreamPool()
  /// can recycle the stream. Every probe stream waits on this one event.
  std::shared_ptr<CudaEvent> buildReadyEvent;
};

/// All build-side tables, batched when the build data exceeds cudf::size_type
/// limits, paired with the hash join objects constructed from them.
using CudfHashJoinBuildData = std::pair<
    std::vector<std::shared_ptr<cudf::table>>,
    std::vector<std::shared_ptr<cudf::hash_join>>>;

/// The single concatenated build-side table. The nested loop join does not
/// batch its build side because the cross-join output is probe_rows x
/// build_rows however the build is split, so a build that exceeds
/// cudf::size_type limits fails early.
struct CudfNestedLoopJoinBuildData {
  std::shared_ptr<cudf::table> table;
  // cudf::table cannot represent the row count of a zero-column table.
  cudf::size_type rowCount;
};

/// Hands the hash tables and hash join objects to the hash probe drivers.
class CudfHashJoinBridge : public exec::BuildResultBridge<
                               CudfJoinBuildResult<CudfHashJoinBuildData>> {};

/// Hands the build table to the nested loop join probe drivers.
class CudfNestedLoopJoinBridge
    : public exec::BuildResultBridge<
          CudfJoinBuildResult<CudfNestedLoopJoinBuildData>> {};

/// Returns the TBridge of the operator's plan node and split group. Throws if
/// the task has no such bridge or it is not a TBridge.
template <typename TBridge>
std::shared_ptr<TBridge> getCudfJoinBridge(
    const exec::OperatorCtx& operatorCtx) {
  return checkedPointerCast<TBridge>(operatorCtx.task()->getCustomJoinBridge(
      operatorCtx.driverCtx()->splitGroupId, operatorCtx.planNodeId()));
}

} // namespace facebook::velox::cudf_velox
