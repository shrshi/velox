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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfJoinProbe.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/join.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/utilities/span.hpp>

#include <algorithm>

namespace facebook::velox::cudf_velox {

std::unique_ptr<cudf::column> rowIndexSequence(
    cudf::size_type numRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  return cudf::sequence(
      numRows,
      cudf::numeric_scalar<cudf::size_type>(0, true, stream, mr),
      cudf::numeric_scalar<cudf::size_type>(1, true, stream, mr),
      stream,
      mr);
}

std::unique_ptr<cudf::column> matchedRowFlags(
    cudf::column_view matchedIndices,
    cudf::size_type numRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto rowIndices = rowIndexSequence(numRows, stream, get_temp_mr());
  return cudf::contains(matchedIndices, rowIndices->view(), stream, mr);
}

std::unique_ptr<cudf::column> makeAllFalseColumn(
    cudf::size_type numRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto falseScalar = cudf::numeric_scalar<bool>(false, true, stream, mr);
  return cudf::make_column_from_scalar(falseScalar, numRows, stream, mr);
}

std::unique_ptr<cudf::column> getMaskedIndices(
    cudf::column_view mask,
    MaskType maskType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto seq = rowIndexSequence(mask.size(), stream, mr);

  auto indicesTable = maskType == MaskType::kMatched
      ? cudf::apply_retention_mask(
            cudf::table_view{{seq->view()}}, mask, stream, mr)
      : cudf::apply_deletion_mask(
            cudf::table_view{{seq->view()}}, mask, stream, mr);

  return std::move(indicesTable->release().front());
}

CudfJoinMatchTracker::CudfJoinMatchTracker(
    cudf::size_type numRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr)
    : flags_(makeAllFalseColumn(numRows, stream, mr)) {}

void CudfJoinMatchTracker::update(
    cudf::column_view matchedIndices,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto matchedInBatch =
      matchedRowFlags(matchedIndices, flags_->size(), stream, get_temp_mr());
  orFlags(matchedInBatch->view(), stream, mr);
}

void CudfJoinMatchTracker::merge(
    const CudfJoinMatchTracker& peer,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  orFlags(peer.flags(), stream, mr);
}

std::unique_ptr<cudf::column> CudfJoinMatchTracker::matchedIndices(
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  return getMaskedIndices(flags(), MaskType::kMatched, stream, mr);
}

std::unique_ptr<cudf::column> CudfJoinMatchTracker::unmatchedIndices(
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  return getMaskedIndices(flags(), MaskType::kUnmatched, stream, mr);
}

void CudfJoinMatchTracker::orFlags(
    cudf::column_view otherFlags,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto updatedFlags = cudf::binary_operation(
      flags_->view(),
      otherFlags,
      cudf::binary_operator::BITWISE_OR,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  // binary_operation is async on `stream`; the old column destructs via
  // cudaFreeAsync on its allocation stream (not `stream`), so the free
  // can race the kernel. Drain `stream` before the move-assign.
  stream.sync();
  flags_ = std::move(updatedFlags);
}

void waitForBuildReady(
    const std::shared_ptr<CudaEvent>& buildReadyEvent,
    cuda::stream_ref stream) {
  if (buildReadyEvent != nullptr) {
    buildReadyEvent->waitOn(stream);
  }
}

CudfJoinReadTracker::ReadStream* CudfJoinReadTracker::findReadStream(
    cuda::stream_ref stream) {
  auto it = std::find_if(
      readStreams_.begin(), readStreams_.end(), [&](const ReadStream& rs) {
        return rs.stream.get() == stream.get();
      });
  return it == readStreams_.end() ? nullptr : &*it;
}

void CudfJoinReadTracker::registerStream(cuda::stream_ref stream) {
  auto* readStream = findReadStream(stream);
  if (readStream == nullptr) {
    readStream = &readStreams_.emplace_back(
        ReadStream{
            stream, std::make_unique<CudaEvent>(cudaEventDisableTiming)});
  }
  readStream->pending = true;
}

void CudfJoinReadTracker::recordCompletion(cuda::stream_ref stream) noexcept {
  auto* readStream = findReadStream(stream);
  if (readStream == nullptr) {
    return;
  }
  try {
    readStream->completion->recordFrom(stream);
    readStream->pending = false;
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to record cuDF join probe read completion; "
                 << "cleanup will synchronize the stream instead: " << e.what();
  }
}

void CudfJoinReadTracker::waitForCompletion() {
  for (auto& readStream : readStreams_) {
    if (readStream.pending) {
      readStream.stream.sync();
    } else {
      readStream.completion->synchronize();
    }
  }
  readStreams_.clear();
}

void joinProbeStreams(
    const std::vector<std::optional<cuda::stream_ref>>& probeStreams,
    cuda::stream_ref stream) {
  std::vector<cuda::stream_ref> inputStreams;
  for (const auto& probeStream : probeStreams) {
    if (probeStream.has_value()) {
      inputStreams.push_back(probeStream.value());
    }
  }
  if (!inputStreams.empty()) {
    cudf::detail::join_streams(inputStreams, stream);
  }
}

cudf::column_view indicesView(
    const rmm::device_uvector<cudf::size_type>& indices) {
  return cudf::column_view{cudf::device_span<cudf::size_type const>{indices}};
}

std::pair<
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
filterJoinIndices(
    cudf::table_view left,
    cudf::table_view right,
    cudf::column_view leftIndices,
    cudf::column_view rightIndices,
    const cudf::ast::expression& filter,
    cuda::stream_ref stream) {
  return cudf::filter_join_indices(
      left,
      right,
      leftIndices,
      rightIndices,
      filter,
      cudf::join_kind::INNER_JOIN,
      std::nullopt,
      stream,
      get_temp_mr());
}

RowVectorPtr makeJoinOutput(
    memory::MemoryPool* pool,
    const RowTypePtr& outputType,
    vector_size_t size,
    std::unique_ptr<cudf::table> table,
    cuda::stream_ref stream) {
  if (size == 0) {
    return nullptr;
  }
  return std::make_shared<CudfVector>(
      pool, outputType, size, std::move(table), stream);
}

} // namespace facebook::velox::cudf_velox
