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

#include "velox/experimental/cudf/exec/CudfJoin.h"
#include "velox/experimental/cudf/exec/CudfJoinBridge.h"
#include "velox/experimental/cudf/exec/CudfJoinBuild.h"
#include "velox/experimental/cudf/exec/CudfJoinProbe.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/AstExpressionUtils.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/JoinBridge.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>

#include <cuda/stream>

#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

class CudaEvent;
class CudfExpression;

/**
 * @brief Build operator that constructs CUDF hash tables from build-side input
 * data.
 *
 * Builds the hash tables from the inputs of all build drivers, split into
 * batches when the data exceeds cudf::size_type limits. The constructed hash
 * tables are transferred to probe operators via CudfHashJoinBridge.
 */
class CudfHashJoinBuild : public CudfJoinBuild {
 public:
  CudfHashJoinBuild(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::HashJoinNode> joinNode);

 protected:
  void recordInputStats(const CudfVector& input) override;
  void buildAndPublish(std::vector<CudfVectorPtr> inputs) override;

 private:
  std::shared_ptr<const core::HashJoinNode> joinNode_;
};

/**
 * @brief Probe operator that performs CUDF hash join operations on probe-side
 * input.
 *
 * This operator receives hash tables from CudfHashJoinBuild via the bridge and
 * performs join operations on probe-side input data. It supports all standard
 * join types (inner, left, right, anti, semi) with optional filter conditions.
 * The operator handles stream synchronization between build and probe phases,
 * manages right join state across multiple drivers, and supports batched
 * processing for large datasets.
 */
class CudfHashJoinProbe : public CudfOperatorBase {
 public:
  using hash_type = CudfHashJoinBuildData;

  CudfHashJoinProbe(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::HashJoinNode> joinNode);

  void initialize() override;

  bool needsInput() const override;

  bool skipProbeOnEmptyBuild() const;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  /// Returns true if the join type is supported by cudf hash join.
  /// Supported types:
  /// - Inner, Left, Right, Full joins
  /// - Left/Right Semi Filter joins
  /// - Left Semi Project join (excluding null-aware join with filter)
  /// - Anti join (non-null-aware, or null-aware without filter)
  /// Note: Right Semi Project, and null-aware left semi-project join with
  /// filter not yet supported.
  static bool isSupportedJoinType(core::JoinType joinType) {
    return joinType == core::JoinType::kInner ||
        joinType == core::JoinType::kLeft ||
        joinType == core::JoinType::kAnti ||
        joinType == core::JoinType::kLeftSemiFilter ||
        joinType == core::JoinType::kLeftSemiProject ||
        joinType == core::JoinType::kRight ||
        joinType == core::JoinType::kRightSemiFilter ||
        joinType == core::JoinType::kFull;
  }

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
  void doClose() override;

 private:
  std::shared_ptr<const core::HashJoinNode> joinNode_;
  /** @brief Hash tables and join objects received from build operator */
  std::optional<hash_type> hashObject_;

  // Filter related members
  /** @brief Whether to use AST-based filtering (false if filter spans both
   * sides or if filter deals with decimal types) */
  bool useAstFilter_{true};
  /** @brief CUDF AST tree for join filter evaluation */
  cudf::ast::tree tree_;
  /** @brief Scalar values used in filter expressions */
  std::vector<std::unique_ptr<cudf::scalar>> scalars_;
  /** @brief Precompute instructions for left (probe) table columns */
  std::vector<PrecomputeInstruction> leftPrecomputeInstructions_;
  /** @brief Precompute instructions for right (build) table columns */
  std::vector<PrecomputeInstruction> rightPrecomputeInstructions_;
  /** @brief Row type for probe table (needed for precomputation) */
  RowTypePtr probeType_;
  /** @brief Row type for build table (needed for precomputation) */
  RowTypePtr buildType_;
  /** @brief Cached evaluator for post-join filter column */
  std::shared_ptr<CudfExpression> filterEvaluator_;

  bool rightPrecomputed_{false};

  // Batched probe inputs needed for right join
  std::vector<CudfVectorPtr> inputs_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};

  /** @brief Column indices for join keys in left (probe) table */
  std::vector<cudf::size_type> leftKeyIndices_;
  /** @brief Column indices for join keys in right (build) table */
  std::vector<cudf::size_type> rightKeyIndices_;
  CudfJoinOutputLayout outputLayout_;
  bool finished_{false};

  /// True if any build table has NULL values in join key columns.
  /// Used for null-aware LEFT SEMI PROJECT to determine match column
  /// nullability.
  bool buildSideHasNullKeys_{false};

  // Copied from HashProbe.h
  // Indicates whether to skip probe input data processing or not. It only
  // applies for a specific set of join types (see skipProbeOnEmptyBuild()), and
  // the build table is empty and the probe input is read from non-spilled
  // source. This ensures the hash probe operator keeps running until all the
  // probe input from the sources have been processed. It prevents the exchange
  // hanging problem at the producer side caused by the early query finish.
  bool skipInput_{false};

  /** @brief Event recorded after build-side CUDA work is ready for probes */
  std::shared_ptr<CudaEvent> buildReadyEvent_;

  // Streaming right join state
  // Per-build-table flags indicating whether a build row has had at least one
  // left match.
  /** @brief Flags tracking which build rows have been matched (for right joins)
   */
  std::vector<CudfJoinMatchTracker> rightMatchedFlags_;

  /// Cached precomputed columns for right (build) tables
  std::vector<std::vector<ColumnOrView>> cachedRightPrecomputed_;
  /// Cached extended views for right tables (original + precomputed columns)
  std::vector<cudf::table_view> cachedExtendedRightViews_;

  // For Right joins, only one driver collects the unmatched rows mask and
  // emits. This value is set true only for that driver. See noMoreInput
  bool isLastDriver_{false};

  /// CUDA stream used during the last getOutput() probe operation. Set only
  /// for right/full joins, and only for drivers that process at least one probe
  /// batch. Used in noMoreInput() to synchronize GPU streams across drivers
  /// before combining rightMatchedFlags_. Drivers with no probe input are safe
  /// to skip: the driver loop guarantees all addInput batches are consumed by
  /// getOutput() before noMoreInput() fires, and unset flags remain in their
  /// host-synchronized all-false init state with no pending GPU work.
  std::optional<cuda::stream_ref> lastProbeStream_;

  /// Streams that read hashObject_, rightMatchedFlags_, scalars_, tree_, or
  /// filterEvaluator_. Waited for before isFinished()/doClose() release them.
  CudfJoinReadTracker readTracker_;

  static constexpr auto oobPolicy = cudf::out_of_bounds_policy::NULLIFY;

  struct JoinOutput {
    std::unique_ptr<cudf::table> table;
    vector_size_t numRows;
  };

  /**
   * @brief Performs inner join between probe table and all build tables.
   * @param leftTable Probe-side table to join
   * @param stream CUDA stream for operations
   * @return Vector of result tables (multiple if build data was batched)
   */
  std::vector<JoinOutput> innerJoin(
      cudf::table_view leftTableView,
      cuda::stream_ref stream);
  /**
   * @brief Performs left join between probe table and all build tables.
   * @param leftTableView Probe-side table view to join
   * @param stream CUDA stream for operations
   * @return Vector of result tables (multiple if build data was batched)
   */
  std::vector<JoinOutput> leftJoin(
      cudf::table_view leftTableView,
      cuda::stream_ref stream);
  /**
   * @brief Performs right join between probe table and all build tables.
   * @param leftTableView Probe-side table view to join
   * @param stream CUDA stream for operations
   * @return Vector of result tables (multiple if build data was batched)
   */
  std::vector<JoinOutput> rightJoin(
      cudf::table_view leftTableView,
      cuda::stream_ref stream);
  /**
   * @brief Performs full outer join between probe table and all build tables.
   * @param leftTableView Probe-side table view to join
   * @param stream CUDA stream for operations
   * @return Vector of result tables (multiple if build data was batched)
   */
  std::vector<JoinOutput> fullJoin(
      cudf::table_view leftTableView,
      cuda::stream_ref stream);
  /**
   * @brief Performs left semi filter join between probe table and all build
   * tables.
   * @param leftTableView Probe-side table view to join
   * @param stream CUDA stream for operations
   * @return Vector of result tables (multiple if build data was batched)
   */
  std::vector<JoinOutput> leftSemiFilterJoin(
      cudf::table_view leftTableView,
      cuda::stream_ref stream);
  /**
   * @brief Performs left semi project join between probe table and all build
   * tables. Returns all probe rows with a boolean match column indicating
   * whether each row has a match on the build side.
   * @param leftTableView Probe-side table view to join
   * @param stream CUDA stream for operations
   * @return Vector of result tables (multiple if build data was batched)
   */
  std::vector<JoinOutput> leftSemiProjectJoin(
      cudf::table_view leftTableView,
      cuda::stream_ref stream);
  /**
   * @brief Performs right semi filter join between probe table and all build
   * tables.
   * @param leftTableView Probe-side table view to join
   * @param stream CUDA stream for operations
   * @return Vector of result tables (multiple if build data was batched)
   */
  std::vector<JoinOutput> rightSemiFilterJoin(
      cudf::table_view leftTableView,
      cuda::stream_ref stream);
  /**
   * @brief Performs anti join between probe table and all build tables.
   * @param leftTableView Probe-side table view to join
   * @param stream CUDA stream for operations
   * @return Vector of result tables (multiple if build data was batched)
   */
  std::vector<JoinOutput> antiJoin(
      cudf::table_view leftTableView,
      cuda::stream_ref stream);
  /**
   * @brief Constructs join output table without applying filter conditions.
   * @param leftTableView Input probe table view
   * @param leftIndicesCol Column of indices into left table
   * @param rightTableView Input build table view
   * @param rightIndicesCol Column of indices into right table
   * @param stream CUDA stream for operations
   * @return Join result table with its logical row count
   */
  JoinOutput unfilteredOutput(
      cudf::table_view leftTableView,
      cudf::column_view leftIndicesCol,
      cudf::table_view rightTableView,
      cudf::column_view rightIndicesCol,
      cuda::stream_ref stream);
  /**
   * @brief Constructs join output table with filter condition applied.
   * @param leftTableView Input probe table view
   * @param leftIndicesCol Column of indices into left table
   * @param rightTableView Input build table view
   * @param rightIndicesCol Column of indices into right table
   * @param tracker If not null, updated with the trackedIndicesCol rows
   * that pass the filter
   * @param trackedIndicesCol leftIndicesCol or rightIndicesCol, the side
   * tracker tracks
   * @param stream CUDA stream for operations
   * @return Filtered join result table with its logical row count
   */
  JoinOutput filteredOutput(
      cudf::table_view leftTableView,
      cudf::column_view leftIndicesCol,
      cudf::table_view rightTableView,
      cudf::column_view rightIndicesCol,
      CudfJoinMatchTracker* tracker,
      cudf::column_view trackedIndicesCol,
      cuda::stream_ref stream);

  /**
   * @brief Appends the output of the probe rows at probeIndices with NULL
   * build columns to cudfOutputs. Does nothing if probeIndices is empty.
   * @param anyBuildTableView Any build table view, for the build column types
   */
  void appendProbeRowsOutput(
      std::vector<JoinOutput>& cudfOutputs,
      cudf::table_view leftTableView,
      std::unique_ptr<cudf::column> probeIndices,
      cudf::table_view anyBuildTableView,
      cuda::stream_ref stream);
};

/**
 * @brief Factory for creating CUDF hash join operators and bridges from plan
 * nodes.
 *
 * This translator converts HashJoinNode plan nodes into the appropriate
 * CUDF-specific operators and bridges. It creates CudfHashJoinProbe operators
 * for probe-side processing, CudfHashJoinBuild operators for build-side
 * processing, and CudfHashJoinBridge instances for coordinating between them.
 */
class CudfHashJoinBridgeTranslator : public exec::Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<exec::Operator>
  toOperator(exec::DriverCtx* ctx, int32_t id, const core::PlanNodePtr& node);

  std::unique_ptr<exec::JoinBridge> toJoinBridge(const core::PlanNodePtr& node);

  exec::OperatorSupplier toOperatorSupplier(const core::PlanNodePtr& node);
};

} // namespace facebook::velox::cudf_velox
