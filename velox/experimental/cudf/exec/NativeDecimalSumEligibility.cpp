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
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfAggregation.h"
#include "velox/experimental/cudf/exec/CudfPlanNodes.h"
#include "velox/experimental/cudf/exec/NativeDecimalSumEligibility.h"

namespace facebook::velox::cudf_velox {

namespace {

template <typename Aggregation>
std::vector<bool> candidates(const Aggregation& aggregation) {
  std::vector<bool> eligible(aggregation.aggregates().size(), false);
  using Step = core::AggregationNode::Step;
  const auto step = aggregation.step();
  if (step != Step::kPartial && step != Step::kIntermediate &&
      step != Step::kFinal && step != Step::kSingle) {
    return eligible;
  }
  const auto sumName = CudfConfig::getInstance().functionNamePrefix + "sum";
  for (size_t i = 0; i < eligible.size(); ++i) {
    const auto& aggregate = aggregation.aggregates()[i];
    if (aggregate.call->name() != sumName ||
        aggregate.rawInputTypes.size() != 1 ||
        !aggregate.rawInputTypes[0]->isShortDecimal() ||
        aggregate.call->inputs().size() != 1 || aggregate.mask ||
        aggregate.distinct || !aggregate.sortingKeys.empty()) {
      continue;
    }
    const auto* field = dynamic_cast<const core::FieldAccessTypedExpr*>(
        aggregate.call->inputs()[0].get());
    if (!field || !field->isInputColumn()) {
      continue;
    }
    const auto& rawType = aggregate.rawInputTypes[0];
    const bool rawInput = step == Step::kPartial || step == Step::kSingle;
    const bool partialOutput =
        step == Step::kPartial || step == Step::kIntermediate;
    const auto resultType = partialOutput
        ? VARBINARY()
        : DECIMAL(38, getDecimalPrecisionScale(*rawType).second);
    eligible[i] =
        field->type()->equivalent(*(rawInput ? rawType : VARBINARY())) &&
        aggregate.call->type()->equivalent(*resultType) &&
        aggregation.outputType()
            ->childAt(aggregation.groupingKeys().size() + i)
            ->equivalent(*resultType);
  }
  return eligible;
}

} // namespace

std::vector<bool> nativeDecimalSumEligibility(
    const core::AggregationNode& aggregation) {
  return candidates(aggregation);
}

std::vector<bool> nativeDecimalSumEligibility(
    const CudfAggregationNode& aggregation) {
  return candidates(aggregation);
}

} // namespace facebook::velox::cudf_velox
