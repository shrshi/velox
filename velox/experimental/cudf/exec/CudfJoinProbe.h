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

#include "velox/common/memory/MemoryPool.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Returns the INT32 column [0, 1, ..., numRows - 1].
std::unique_ptr<cudf::column> rowIndexSequence(
    cudf::size_type numRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Returns a BOOL8 column of numRows rows where row i is true if i appears in
/// matchedIndices. Duplicate indices are allowed.
std::unique_ptr<cudf::column> matchedRowFlags(
    cudf::column_view matchedIndices,
    cudf::size_type numRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Returns a BOOL8 column of numRows false values.
std::unique_ptr<cudf::column> makeAllFalseColumn(
    cudf::size_type numRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

// Selects whether getMaskedIndices keeps rows where the mask is true (kMatched)
// or where the mask is false (kUnmatched).
enum class MaskType { kMatched, kUnmatched };

// Returns row indices selected by mask as a column of size_type. When maskType
// is kMatched, indices where the mask is true are returned; when kUnmatched,
// indices where the mask is false are returned.
std::unique_ptr<cudf::column> getMaskedIndices(
    cudf::column_view mask,
    MaskType maskType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Tracks which rows of one join side have matched across join batches, as a
/// BOOL8 flag per row that is OR-accumulated by update() and merge().
class CudfJoinMatchTracker {
 public:
  /// Starts with all numRows rows unmatched. The flags are allocated from mr.
  CudfJoinMatchTracker(
      cudf::size_type numRows,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  /// Marks the rows present in matchedIndices as matched. The new flags are
  /// allocated from mr. Synchronizes stream before replacing the old flags,
  /// which may have been allocated on another stream.
  void update(
      cudf::column_view matchedIndices,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  /// Marks the rows matched in peer as matched, like update().
  void merge(
      const CudfJoinMatchTracker& peer,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  cudf::column_view flags() const {
    return flags_->view();
  }

  /// Moves the flags out of the tracker, which must not be used afterwards.
  std::unique_ptr<cudf::column> releaseFlags() {
    return std::move(flags_);
  }

  /// Returns the indices of the rows that matched at least once.
  std::unique_ptr<cudf::column> matchedIndices(
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const;

  /// Returns the indices of the rows that never matched.
  std::unique_ptr<cudf::column> unmatchedIndices(
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const;

 private:
  void orFlags(
      cudf::column_view otherFlags,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  std::unique_ptr<cudf::column> flags_;
};

/// Makes stream wait for the work submitted so far on each probe stream that
/// is set. Does nothing if none is set.
void joinProbeStreams(
    const std::vector<std::optional<cuda::stream_ref>>& probeStreams,
    cuda::stream_ref stream);

/// Returns a non-owning INT32 column view of join indices.
cudf::column_view indicesView(
    const rmm::device_uvector<cudf::size_type>& indices);

/// Keeps the (leftIndices[i], rightIndices[i]) pairs for which filter is true
/// on rows of left and right (cudf::filter_join_indices with INNER_JOIN).
std::pair<
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
filterJoinIndices(
    cudf::table_view left,
    cudf::table_view right,
    cudf::column_view leftIndices,
    cudf::column_view rightIndices,
    const cudf::ast::expression& filter,
    cuda::stream_ref stream);

/// Wraps a join output table of size rows in a CudfVector. Returns nullptr if
/// size is 0.
RowVectorPtr makeJoinOutput(
    memory::MemoryPool* pool,
    const RowTypePtr& outputType,
    vector_size_t size,
    std::unique_ptr<cudf::table> table,
    cuda::stream_ref stream);

} // namespace facebook::velox::cudf_velox
