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

#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::cudf_velox {

/// Returns one eligibility flag per aggregate in 'aggregation'. Only direct,
/// unshared grouped PARTIAL -> single-source hash LocalPartition -> grouped
/// FINAL paths qualify. Both aggregations must be supported by cuDF and have
/// identical layouts containing only ordinary DECIMAL64 SUMs. This proves plan
/// compatibility, not operator replacement; unsupported runtime consumers must
/// still materialize native state.
std::vector<bool> nativeDecimalSumEligibility(
    const core::AggregationNode& aggregation,
    const core::PlanNode& planRoot,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool);

} // namespace facebook::velox::cudf_velox
