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
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfGroupby.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/NativeDecimalSumEligibility.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/tests/utils/CudfPlanTestUtils.h"
#include "velox/experimental/cudf/tests/utils/ExpressionTestUtil.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/DecimalUtil.h"

#include <unordered_set>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <limits>
#include <optional>
#include <type_traits>

namespace facebook::velox::cudf_velox {
namespace {

using exec::test::AssertQueryBuilder;
using test::rewriteToCudfPlan;

int64_t computeAvgRaw(const std::vector<int64_t>& values) {
  int128_t sum = 0;
  for (auto value : values) {
    sum += value;
  }
  int128_t avg = 0;
  facebook::velox::DecimalUtil::computeAverage(avg, sum, values.size(), 0);
  return static_cast<int64_t>(avg);
}

constexpr int kBitsPerWord = 8 * sizeof(cudf::bitmask_type);

std::pair<cuda::device_buffer<std::byte>, cudf::size_type> makeNullMask(
    const std::vector<bool>& valid,
    cuda::stream_ref stream) {
  auto numBits = static_cast<cudf::size_type>(valid.size());
  if (numBits == 0) {
    return {
        cuda::device_buffer<std::byte>{
            stream, cudf::get_current_device_resource_ref()},
        0};
  }
  auto maskBytes = cudf::bitmask_allocation_size_bytes(numBits);
  auto numWords = maskBytes / sizeof(cudf::bitmask_type);
  std::vector<cudf::bitmask_type> host(numWords, 0);
  cudf::size_type nullCount = 0;
  for (cudf::size_type i = 0; i < numBits; ++i) {
    if (valid[i]) {
      auto word = i / kBitsPerWord;
      auto bit = i % kBitsPerWord;
      host[word] |= (cudf::bitmask_type{1} << bit);
    } else {
      ++nullCount;
    }
  }
  auto mask =
      cudf::create_null_mask(numBits, cudf::mask_state::UNINITIALIZED, stream);
  if (!host.empty()) {
    auto status = cudaMemcpyAsync(
        mask.data(),
        host.data(),
        host.size() * sizeof(cudf::bitmask_type),
        cudaMemcpyHostToDevice,
        stream.get());
    VELOX_CHECK_EQ(0, static_cast<int>(status));
    stream.sync();
  }
  return {std::move(mask), nullCount};
}

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* key, const char* value) : key_(key) {
    const char* existing = std::getenv(key);
    if (existing) {
      oldValue_ = std::string(existing);
    }
    if (value) {
      setenv(key, value, 1);
    } else {
      unsetenv(key);
    }
  }

  ~ScopedEnvVar() {
    if (oldValue_) {
      setenv(key_.c_str(), oldValue_->c_str(), 1);
    } else {
      unsetenv(key_.c_str());
    }
  }

 private:
  std::string key_;
  std::optional<std::string> oldValue_;
};

template <typename T>
std::unique_ptr<cudf::column> makeFixedWidthColumn(
    cudf::data_type type,
    const std::vector<T>& values,
    const std::vector<bool>* valid,
    cuda::stream_ref stream) {
  auto col = cudf::make_fixed_width_column(
      type,
      static_cast<cudf::size_type>(values.size()),
      cudf::mask_state::UNALLOCATED,
      stream);
  if (!values.empty()) {
    auto status = cudaMemcpyAsync(
        col->mutable_view().data<T>(),
        values.data(),
        values.size() * sizeof(T),
        cudaMemcpyHostToDevice,
        stream.get());
    VELOX_CHECK_EQ(0, static_cast<int>(status));
    stream.sync();
  }
  if (valid) {
    auto [mask, nullCount] = makeNullMask(*valid, stream);
    col->set_null_mask(std::move(mask), nullCount);
  }
  return col;
}

template <typename T>
std::unique_ptr<cudf::column> makeDecimalColumn(
    const std::vector<T>& values,
    int32_t scale,
    const std::vector<bool>* valid,
    cuda::stream_ref stream) {
  cudf::type_id typeId = std::is_same_v<T, int64_t> ? cudf::type_id::DECIMAL64
                                                    : cudf::type_id::DECIMAL128;
  cudf::data_type type{typeId, -scale};
  return makeFixedWidthColumn(type, values, valid, stream);
}

std::unique_ptr<cudf::column> makeInt64Column(
    const std::vector<int64_t>& values,
    const std::vector<bool>* valid,
    cuda::stream_ref stream) {
  return makeFixedWidthColumn(
      cudf::data_type{cudf::type_id::INT64}, values, valid, stream);
}

template <typename T>
std::vector<T> copyColumnData(
    const cudf::column_view& view,
    cuda::stream_ref stream) {
  std::vector<T> host(view.size());
  if (view.size() == 0) {
    return host;
  }
  auto status = cudaMemcpyAsync(
      host.data(),
      view.data<T>(),
      view.size() * sizeof(T),
      cudaMemcpyDeviceToHost,
      stream.get());
  VELOX_CHECK_EQ(0, static_cast<int>(status));
  stream.sync();
  return host;
}

std::vector<cudf::bitmask_type> copyNullMask(
    const cudf::column_view& view,
    cuda::stream_ref stream) {
  auto numWords = cudf::num_bitmask_words(view.size());
  std::vector<cudf::bitmask_type> host(numWords, 0);
  if (!view.nullable() || numWords == 0) {
    return host;
  }
  auto status = cudaMemcpyAsync(
      host.data(),
      view.null_mask(),
      host.size() * sizeof(cudf::bitmask_type),
      cudaMemcpyDeviceToHost,
      stream.get());
  VELOX_CHECK_EQ(0, static_cast<int>(status));
  stream.sync();
  return host;
}

bool isValidAt(const std::vector<cudf::bitmask_type>& mask, size_t idx) {
  if (mask.empty()) {
    return true;
  }
  auto word = idx / kBitsPerWord;
  auto bit = idx % kBitsPerWord;
  return (mask[word] >> bit) & 1;
}

class CudfDecimalTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    exec::test::OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    CudfConfig::getInstance().allowCpuFallback = false;
    // Ensure a CUDA device is selected and initialized (RMM asserts otherwise).
    int deviceCount = 0;
    auto status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess) {
      GTEST_SKIP() << "cudaGetDeviceCount failed: " << static_cast<int>(status)
                   << " (" << cudaGetErrorString(status) << ")";
    }
    if (deviceCount == 0) {
      GTEST_SKIP() << "No CUDA devices visible (check CUDA_VISIBLE_DEVICES)";
    }
    VELOX_CHECK_EQ(0, static_cast<int>(cudaSetDevice(0)));
    VELOX_CHECK_EQ(0, static_cast<int>(cudaFree(nullptr)));
    registerCudf();
  }

  void TearDown() override {
    unregisterCudf();
    exec::test::OperatorTestBase::TearDown();
  }

  bool hasStreamingGroupbyStat(
      const std::shared_ptr<exec::Task>& task,
      const core::PlanNodeId& planNodeId) {
    const auto planStats = exec::toPlanStats(task->taskStats());
    const auto it = planStats.find(planNodeId);
    return it != planStats.end() &&
        it->second.customStats.count(
            std::string{kStreamingGroupbyUsedStat}) > 0;
  }

  void assertNativeSumPlan(const core::PlanNodePtr& plan, bool expectNative) {
    size_t nativeAggregates = 0;
    const auto inspect = [&](const auto& self,
                             const core::PlanNodePtr& node) -> void {
      if (auto aggregate =
              std::dynamic_pointer_cast<const CudfAggregationNode>(node)) {
        const auto& states = aggregate->nativeDecimalSumStates();
        ASSERT_EQ(states.size(), aggregate->aggregates().size());
        for (size_t i = 0; i < states.size(); ++i) {
          const auto& state = states[i];
          const bool selected = state.input || state.output || state.buffer;
          nativeAggregates += selected;
          if (!expectNative) {
            EXPECT_FALSE(selected);
          }
          const auto& call = aggregate->aggregates()[i].call;
          if (selected) {
            const auto type = DECIMAL(
                38,
                getDecimalPrecisionScale(
                    *aggregate->aggregates()[i].rawInputTypes[0])
                    .second);
            EXPECT_TRUE(state.buffer);
            const auto step = aggregate->step();
            EXPECT_EQ(
                state.input,
                step == core::AggregationNode::Step::kIntermediate ||
                    step == core::AggregationNode::Step::kFinal);
            EXPECT_EQ(
                state.output,
                step == core::AggregationNode::Step::kPartial ||
                    step == core::AggregationNode::Step::kIntermediate);
            if (state.input) {
              const auto field =
                  core::TypedExprs::asFieldAccess(call->inputs()[0]);
              ASSERT_NE(field, nullptr);
              EXPECT_EQ(*field->type(), *type);
              EXPECT_EQ(
                  *aggregate->sources()[0]->outputType()->findChild(
                      field->name()),
                  *type);
            }
            if (state.output) {
              EXPECT_EQ(*call->type(), *type);
              EXPECT_EQ(
                  *aggregate->outputType()->childAt(
                      aggregate->groupingKeys().size() + i),
                  *type);
            }
          }
        }
      }
      for (const auto& source : node->sources()) {
        self(self, source);
      }
    };
    inspect(inspect, plan);
    EXPECT_EQ(nativeAggregates > 0, expectNative);
  }

  void assertNativeSumStats(const std::shared_ptr<exec::Task>& task) {
    int64_t produced = 0;
    int64_t consumed = 0;
    uint64_t partialInputVectors = 0;
    uint64_t partialDrivers = 0;
    for (const auto& pipeline : task->taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        if (op.operatorType == "CudfGroupbyPARTIAL") {
          partialInputVectors += op.inputVectors;
          partialDrivers += op.numDrivers;
        }
        for (const auto& [name, metric] : op.runtimeStats) {
          if (name == "nativeDecimalSumProducedRows") {
            produced += metric.sum;
          } else if (name == "nativeDecimalSumConsumedRows") {
            consumed += metric.sum;
          }
        }
      }
    }
    EXPECT_GT(produced, 0);
    EXPECT_GT(consumed, 0);
    EXPECT_GT(partialDrivers, 0);
    EXPECT_GT(partialInputVectors, partialDrivers);
  }
};

TEST_F(CudfDecimalTest, mixedWidthDecimalDivision) {
  const auto rowType = ROW({
      {"short_decimal", DECIMAL(7, 2)},
      {"long_decimal", DECIMAL(20, 3)},
  });
  auto queryCtx = core::QueryCtx::create();
  core::ExecCtx execCtx(pool(), queryCtx.get());
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  const std::vector<bool> shortValid{true, true, true, false, true};
  const std::vector<bool> longValid{true, true, true, true, false};
  // Narrowing this denominator to int64_t would turn it into zero.
  const int128_t largeDenominator = int128_t{1} << 64;
  auto shortDecimal = makeDecimalColumn<int64_t>(
      {1'000, -1'000, 1'000, 1'000, 0}, 2, &shortValid, stream);
  auto longDecimal = makeDecimalColumn<int128_t>(
      {2'000, 2'000, largeDenominator, 2'000, 2'000}, 3, &longValid, stream);
  const std::vector<cudf::column_view> inputs{
      shortDecimal->view(), longDecimal->view()};

  auto assertDivision =
      [&](const std::string& sql,
          const TypePtr& expectedType,
          const std::vector<std::optional<int128_t>>& expected) {
        SCOPED_TRACE(sql);
        auto expression = test_utils::optimizeTypedExpr(
            sql, rowType, queryCtx.get(), &execCtx);
        ASSERT_TRUE(expression->type()->equivalent(*expectedType));
        auto evaluator = createCudfExpression(expression, rowType, pool());
        auto result = evaluator->eval(inputs, stream, mr);
        const auto view = asView(result);
        ASSERT_EQ(view.type(), veloxToCudfDataType(expectedType));
        ASSERT_EQ(view.size(), expected.size());
        auto values = [&]() -> std::vector<int128_t> {
          if (expectedType->isShortDecimal()) {
            const auto shortValues = copyColumnData<int64_t>(view, stream);
            return {shortValues.begin(), shortValues.end()};
          }
          return copyColumnData<int128_t>(view, stream);
        }();
        const auto nullMask = copyNullMask(view, stream);
        cudf::size_type expectedNulls = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
          ASSERT_EQ(isValidAt(nullMask, i), expected[i].has_value());
          if (expected[i]) {
            EXPECT_EQ(values[i], *expected[i]);
          } else {
            ++expectedNulls;
          }
        }
        EXPECT_EQ(view.null_count(), expectedNulls);
      };

  assertDivision(
      "short_decimal / long_decimal",
      DECIMAL(11, 3),
      {5'000, -5'000, 0, std::nullopt, std::nullopt});
  assertDivision(
      "long_decimal / short_decimal",
      DECIMAL(22, 3),
      {200, -200, 1'844'674'407'370'955'162, std::nullopt, std::nullopt});
  assertDivision(
      "short_decimal / CAST('2.000' AS DECIMAL(20, 3))",
      DECIMAL(11, 3),
      {5'000, -5'000, 5'000, std::nullopt, 0});
  assertDivision(
      "long_decimal / CAST('10.00' AS DECIMAL(7, 2))",
      DECIMAL(22, 3),
      {200, 200, 1'844'674'407'370'955'162, 200, std::nullopt});
  assertDivision(
      "CAST('10.00' AS DECIMAL(7, 2)) / long_decimal",
      DECIMAL(11, 3),
      {5'000, 5'000, 0, 5'000, std::nullopt});
  assertDivision(
      "CAST('2.000' AS DECIMAL(20, 3)) / short_decimal",
      DECIMAL(22, 3),
      {200, -200, 200, std::nullopt, std::nullopt});
  assertDivision(
      "short_decimal / CAST('18446744073709551.616' AS DECIMAL(20, 3))",
      DECIMAL(11, 3),
      {0, 0, 0, std::nullopt, 0});
}

TEST_F(CudfDecimalTest, decimalAvgDecimalInput) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeFlatVector<int64_t>(
          {100, 200, 300, 400}, // 1.00, 2.00, 3.00, 4.00
          DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS avg_d"})
                  .planNode();

  auto expected = makeRowVector(
      {"avg_d"}, {makeFlatVector<int64_t>({250}, DECIMAL(12, 2))}); // 2.50

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgDecimalInputRounds) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  // Sum = 1.60, count = 7 => 0.22857..., rounds to 0.23 at scale 2.
  std::vector<int64_t> rawValues = {100, 10, 10, 10, 10, 10, 10};
  auto input = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>(rawValues, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS avg_d"})
                  .planNode();

  auto expected = makeRowVector(
      {"avg_d"},
      {makeFlatVector<int64_t>({computeAvgRaw(rawValues)}, DECIMAL(12, 2))});

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgPartialFinalVarbinaryRounds) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  std::vector<int32_t> keys = {1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3};
  std::vector<int64_t> values = {100, 10, 10, 10, 10, 10, 10, 100, 1, -100, -1};

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>(keys),
          makeFlatVector<int64_t>(values, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"avg(d) AS a"})
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  std::vector<std::pair<int32_t, std::vector<int64_t>>> groups = {
      {1, {100, 10, 10, 10, 10, 10, 10}},
      {2, {100, 1}},
      {3, {-100, -1}},
  };

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>(
              {computeAvgRaw(groups[0].second),
               computeAvgRaw(groups[1].second),
               computeAvgRaw(groups[2].second)},
              DECIMAL(12, 2)),
      });

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgIntermediateVarbinaryRounds) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 3}),
          makeFlatVector<int64_t>({100, 10, 100, -100}, DECIMAL(12, 2)),
      });
  auto input2 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 1, 1, 2, 3}),
          makeFlatVector<int64_t>({10, 10, 10, 10, 10, 1, -1}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"avg(d) AS a"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  std::vector<std::pair<int32_t, std::vector<int64_t>>> groups = {
      {1, {100, 10, 10, 10, 10, 10, 10}},
      {2, {100, 1}},
      {3, {-100, -1}},
  };

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>(
              {computeAvgRaw(groups[0].second),
               computeAvgRaw(groups[1].second),
               computeAvgRaw(groups[2].second)},
              DECIMAL(12, 2)),
      });

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalPartialFinalVarbinaryRounds) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({100, 10, 10}, DECIMAL(12, 2))});
  auto input2 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({10, 10, 10, 10}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"avg(d) AS a"})
                  .finalAggregation()
                  .planNode();

  std::vector<int64_t> allValues = {100, 10, 10, 10, 10, 10, 10};
  auto expected = makeRowVector(
      {"a"},
      {makeFlatVector<int64_t>({computeAvgRaw(allValues)}, DECIMAL(12, 2))});

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalIntermediateVarbinaryRounds) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({100, 10, 10}, DECIMAL(12, 2))});
  auto input2 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({10, 10, 10, 10}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"avg(d) AS a"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  std::vector<int64_t> allValues = {100, 10, 10, 10, 10, 10, 10};
  auto expected = makeRowVector(
      {"a"},
      {makeFlatVector<int64_t>({computeAvgRaw(allValues)}, DECIMAL(12, 2))});

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalSingleRounds) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeFlatVector<int64_t>({100, 10, 10, 10, 10, 10, 10}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS a"})
                  .planNode();

  std::vector<int64_t> allValues = {100, 10, 10, 10, 10, 10, 10};
  auto expected = makeRowVector(
      {"a"},
      {makeFlatVector<int64_t>({computeAvgRaw(allValues)}, DECIMAL(12, 2))});

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalSingleDecimal64Overflow) {
  // 12 values of 9e17 (DECIMAL(18,0)) sum to 1.08e19, past 2^63. The sum must
  // accumulate in 128 bits or a DECIMAL64 accumulator wraps; avg is 9e17.
  constexpr int64_t kBig = 900'000'000'000'000'000;
  constexpr int kNumRows = 12;
  std::vector<int64_t> values(kNumRows, kBig);

  auto input =
      makeRowVector({"d"}, {makeFlatVector<int64_t>(values, DECIMAL(18, 0))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS a"})
                  .planNode();

  auto expected =
      makeRowVector({"a"}, {makeFlatVector<int64_t>({kBig}, DECIMAL(18, 0))});

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGroupbySingleDecimal64Overflow) {
  // Same overflow within a single group, exercising the groupby raw sum path.
  constexpr int64_t kBig = 900'000'000'000'000'000;
  constexpr int kNumRows = 12;
  std::vector<int32_t> keys(kNumRows, 1);
  std::vector<int64_t> values(kNumRows, kBig);

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>(keys),
          makeFlatVector<int64_t>(values, DECIMAL(18, 0)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({"k"}, {"avg(d) AS a"})
                  .planNode();

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1}),
          makeFlatVector<int64_t>({kBig}, DECIMAL(18, 0)),
      });

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalSingleAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS a"})
                  .planNode();

  auto expected = makeRowVector(
      {"a"}, {makeNullableFlatVector<int64_t>({std::nullopt}, DECIMAL(12, 2))});

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalPartialFinalVarbinaryAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"avg(d) AS a"})
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"a"}, {makeNullableFlatVector<int64_t>({std::nullopt}, DECIMAL(12, 2))});

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalIntermediateVarbinaryAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"avg(d) AS a"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"a"}, {makeNullableFlatVector<int64_t>({std::nullopt}, DECIMAL(12, 2))});

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgPartialFinalVarbinaryNullGroup) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 2, 3, 3}),
          makeNullableFlatVector<int64_t>(
              {100, 200, std::nullopt, std::nullopt, 400, std::nullopt},
              DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"avg(d) AS a"})
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {150, std::nullopt, 400}, DECIMAL(12, 2)),
      });

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgIntermediateVarbinaryNullGroup) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {100, std::nullopt, 400}, DECIMAL(12, 2)),
      });
  auto input2 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {200, std::nullopt, std::nullopt}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"avg(d) AS a"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {150, std::nullopt, 400}, DECIMAL(12, 2)),
      });

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumPartialFinalVarbinary) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 2, 2}),
          makeFlatVector<int64_t>(
              {12345, -2500, 10000, 200, -300}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();

  auto task = facebook::velox::exec::test::AssertQueryBuilder(
                  rewriteToCudfPlan(plan), duckDbQueryRunner_)
                  .assertResults("SELECT k, sum(d) AS s FROM tmp GROUP BY k");
  const auto stats = exec::toPlanStats(task->taskStats());
  EXPECT_GT(
      stats.at(plan->id())
          .customStats.count(
              std::string{kDirectGroupbyFinalizationStat}),
      0);
}

TEST_F(CudfDecimalTest, decimalGroupbyReleasesRequestTemporaries) {
  rmm::cuda_stream streamOwner;
  const cuda::stream_ref stream = streamOwner;
  rmm::mr::tracking_resource_adaptor tracking{
      rmm::mr::cuda_async_memory_resource{}};
  auto previousResource = cudf::set_current_device_resource(tracking);
  SCOPE_EXIT {
    stream.sync();
    cudf::set_current_device_resource(std::move(previousResource));
  };
  const rmm::device_async_resource_ref mr{tracking};
  const auto decimalType = DECIMAL(12, 2);
  auto input = makeRowVector(
      {makeFlatVector<int64_t>({0}),
       makeFlatVector<int64_t>({100}, decimalType)});

  for (const auto& function : {"sum", "avg"}) {
    SCOPED_TRACE(function);
    auto builder = exec::test::PlanBuilder().values({input}).partialAggregation(
        {"c0"}, {fmt::format("{}(c1)", function)});
    auto partialNode =
        CudfPlanRewriter::translateForAdapterAs<CudfAggregationNode>(
            builder.planNode());
    auto finalNode =
        CudfPlanRewriter::translateForAdapterAs<CudfAggregationNode>(
            builder.finalAggregation().planNode());
    auto partial = toGroupbyAggregators(
        *partialNode,
        partialNode->step(),
        partialNode->outputType(),
        {nullptr},
        {});
    auto final = toGroupbyAggregators(
        *finalNode, finalNode->step(), finalNode->outputType(), {nullptr}, {});

    // Reuse both adapters, but do not let the next request hide a retained
    // temporary by replacing it. Check allocation ownership after each release.
    for (int64_t batch = 1; batch <= 2; ++batch) {
      SCOPED_TRACE(batch);
      {
        auto keys = makeInt64Column({0, 0, 0}, nullptr, stream);
        const std::vector<bool> valid{true, false, true};
        auto values = makeDecimalColumn<int64_t>(
            {100 * batch, 0, 300 * batch}, 2, &valid, stream);
        const cudf::table_view rawInput{{keys->view(), values->view()}};
        cudf::groupby::groupby partialGroupby(cudf::table_view{{keys->view()}});
        std::vector<cudf::groupby::aggregation_request> requests;
        partial[0]->addGroupbyRequest(rawInput, requests, stream, mr);
        auto* castData =
            const_cast<int128_t*>(requests[0].values.data<int128_t>());
        ASSERT_EQ(tracking.get_outstanding_allocations().count(castData), 1);
        auto [partialKeys, partialResults] =
            partialGroupby.aggregate(requests, stream, mr);
        requests.clear();
        partial[0]->releaseInput();
        EXPECT_EQ(tracking.get_outstanding_allocations().count(castData), 0);
        auto state = partial[0]->makeOutputColumn(partialResults, stream, mr);

        const cudf::table_view stateInput{
            {partialKeys->view().column(0), state->view()}};
        cudf::groupby::groupby finalGroupby(partialKeys->view());
        final[0]->addGroupbyRequest(stateInput, requests, stream, mr);
        std::vector<void*> decodedData;
        for (const auto& request : requests) {
          decodedData.push_back(const_cast<void*>(request.values.head()));
          ASSERT_EQ(
              tracking.get_outstanding_allocations().count(decodedData.back()),
              1);
        }
        auto [finalKeys, finalResults] =
            finalGroupby.aggregate(requests, stream, mr);
        requests.clear();
        final[0]->releaseInput();
        for (auto* data : decodedData) {
          EXPECT_EQ(tracking.get_outstanding_allocations().count(data), 0);
        }
        auto result = final[0]->makeOutputColumn(finalResults, stream, mr);
        ASSERT_EQ(result->size(), 1);
        EXPECT_EQ(result->null_count(), 0);
        if (std::string_view(function) == "sum") {
          EXPECT_EQ(
              copyColumnData<int128_t>(result->view(), stream),
              std::vector<int128_t>{400 * batch});
        } else {
          EXPECT_EQ(
              copyColumnData<int64_t>(result->view(), stream),
              std::vector<int64_t>{200 * batch});
        }
      }
      EXPECT_EQ(tracking.get_allocated_bytes(), 0);
    }
  }
}

TEST_F(CudfDecimalTest, streamingDecimalSumReleasesDecodedInput) {
  rmm::cuda_stream streamOwner;
  const cuda::stream_ref stream = streamOwner;
  rmm::mr::tracking_resource_adaptor tracking{
      rmm::mr::cuda_async_memory_resource{}};
  auto previousResource = cudf::set_current_device_resource(tracking);
  SCOPE_EXIT {
    stream.sync();
    cudf::set_current_device_resource(std::move(previousResource));
  };
  const rmm::device_async_resource_ref mr{tracking};
  auto input = makeRowVector(
      {makeFlatVector<int64_t>({0}),
       makeFlatVector<int64_t>({100}, DECIMAL(12, 2))});
  const auto plan = exec::test::PlanBuilder()
                        .values({input})
                        .partialAggregation({"c0"}, {"sum(c1)"})
                        .finalAggregation()
                        .planNode();
  // Adapter translation preserves the serialized contract for this decode
  // lifetime regression; the full rewriter would select a native input.
  const auto node =
      CudfPlanRewriter::translateForAdapterAs<CudfAggregationNode>(plan);
  auto adapters = toStreamingGroupbyAggregators(
      *node,
      node->sources()[0]->outputType(),
      {0, 1},
      node->outputType(),
      {nullptr},
      {});
  ASSERT_TRUE(adapters.has_value());
  auto& adapter = *adapters->at(0);
  std::unique_ptr<cudf::groupby::streaming_groupby> groupby;
  for (int64_t batch = 1; batch <= 2; ++batch) {
    auto keys = makeInt64Column({0}, nullptr, stream);
    auto sums = makeDecimalColumn<int128_t>({100 * batch}, 2, nullptr, stream);
    auto counts = makeInt64Column({1}, nullptr, stream);
    auto state =
        serializeDecimalSumState(sums->view(), counts->view(), stream, mr);
    const cudf::table_view stateInput{{keys->view(), state->view()}};
    std::vector<cudf::column_view> prepared{keys->view()};
    adapter.prepareInput(stateInput, prepared, stream);
    auto* decodedData = const_cast<void*>(prepared.back().head());
    ASSERT_EQ(tracking.get_outstanding_allocations().count(decodedData), 1);
    if (!groupby) {
      std::vector<cudf::groupby::streaming_aggregation_request> requests;
      adapter.addStreamingRequest(requests);
      groupby = std::make_unique<cudf::groupby::streaming_groupby>(
          std::vector<cudf::size_type>{0},
          requests,
          4096,
          cudf::null_policy::INCLUDE,
          mr);
    }
    groupby->aggregate(cudf::table_view{prepared}, stream);
    prepared.clear();
    adapter.releaseInput();
    EXPECT_EQ(tracking.get_outstanding_allocations().count(decodedData), 0);
  }
  const auto bytesBeforeFinalize = tracking.get_allocated_bytes();
  bool allocatedOutput = false;
  auto outputUpstream = mr;
  rmm::mr::callback_memory_resource outputResource{
      [&](std::size_t bytes, cuda::stream_ref outputStream, void*) {
        // Key locations (8 bytes/slot) and hash slots (4 bytes at load factor
        // 0.5) must be released before the first output allocation.
        if (!allocatedOutput) {
          EXPECT_LE(
              tracking.get_allocated_bytes() + 4096 * 16, bytesBeforeFinalize);
          allocatedOutput = true;
        }
        return outputUpstream.allocate(
            outputStream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
      },
      [&](void* ptr, std::size_t bytes, cuda::stream_ref outputStream, void*) {
        outputUpstream.deallocate(
            outputStream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
      }};
  auto [keys, results] = groupby->finalize_and_release(stream, outputResource);
  EXPECT_TRUE(allocatedOutput);
  EXPECT_EQ(groupby->distinct_keys(), 0);
  EXPECT_THROW(groupby->finalize(stream, mr), cudf::logic_error);
  EXPECT_THROW(groupby->finalize_and_release(stream, mr), cudf::logic_error);
  auto result = adapter.makeOutputColumn(results, stream, mr);
  EXPECT_EQ(
      copyColumnData<int128_t>(result->view(), stream),
      std::vector<int128_t>{300});
}

TEST_F(CudfDecimalTest, decimalSumFinalUsesStreamingGroupby) {
  auto& config = CudfConfig::getInstance();
  const auto savedStreamingGroupbyEnabled = config.streamingGroupbyEnabled;
  const auto savedCapacityMultiplier =
      config.streamingGroupbyCapacityMultiplier;
  const auto savedConcatEnabled = config.concatOptimizationEnabled;
  const auto savedBatchSizeMin = config.batchSizeMinThreshold;
  config.streamingGroupbyEnabled = true;
  config.streamingGroupbyCapacityMultiplier = 2.0;
  config.concatOptimizationEnabled = true;
  config.batchSizeMinThreshold = 1;
  SCOPE_EXIT {
    config.streamingGroupbyEnabled = savedStreamingGroupbyEnabled;
    config.streamingGroupbyCapacityMultiplier = savedCapacityMultiplier;
    config.concatOptimizationEnabled = savedConcatEnabled;
    config.batchSizeMinThreshold = savedBatchSizeMin;
  };

  const auto decimalType = DECIMAL(18, 2);
  std::vector<RowVectorPtr> batches;
  for (int32_t batch = 0; batch < 8; ++batch) {
    batches.push_back(makeRowVector(
        {"k", "d"},
        {makeNullableFlatVector<int32_t>(
             {0, 2 * batch + 1, 2 * batch + 2, std::nullopt}),
         makeNullableFlatVector<int64_t>(
             {100, -200, std::nullopt, 400}, decimalType)}));
  }
  auto builder = exec::test::PlanBuilder()
                     .values(batches, true)
                     .partialAggregation({"k"}, {"sum(d) AS s"});
  const auto plan = builder.finalAggregation().planNode();
  const auto finalAggregationId = plan->id();

  unregisterCudf();
  auto expected =
      exec::test::AssertQueryBuilder(plan).maxDrivers(2).copyResults(pool());
  registerCudf();
  auto task = exec::test::AssertQueryBuilder(rewriteToCudfPlan(plan, 2))
                  .maxDrivers(2)
                  .config(CudfFromVelox::kGpuBatchSizeRows, "4")
                  .config(core::QueryConfig::kMaxPartialAggregationMemory, "1")
                  .assertResults(expected);
  EXPECT_TRUE(hasStreamingGroupbyStat(task, finalAggregationId));
  const auto stats = exec::toPlanStats(task->taskStats());
  EXPECT_GT(
      stats.at(finalAggregationId)
          .customStats.at(std::string{kStreamingGroupbyRebuildsStat})
          .sum,
      0);
}

TEST_F(CudfDecimalTest, nativeDecimalSumDirectAndLocalExchange) {
  constexpr int64_t maxDecimal = 999'999'999'999'999'999;
  const auto decimalType = DECIMAL(18, 2);
  auto input = makeRowVector(
      {"k", "d"},
      {makeFlatVector<int32_t>({0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5}),
       makeNullableFlatVector<int64_t>(
           {100,
            200,
            -100,
            -200,
            maxDecimal,
            -maxDecimal,
            std::nullopt,
            std::nullopt,
            maxDecimal,
            std::nullopt,
            -maxDecimal,
            0},
           decimalType)});
  const std::vector<RowVectorPtr> batches{input, input, input, input};
  for (const auto& [exchange, partialMemory] :
       std::vector<std::pair<bool, std::string>>{
           {false, "1"}, {true, "1"}, {true, "1048576"}}) {
    SCOPED_TRACE(
        fmt::format("exchange={}, memory={}", exchange, partialMemory));
    auto builder = exec::test::PlanBuilder()
                       .values(batches, true)
                       .partialAggregation({"k"}, {"sum(d) AS s"});
    const auto partial = std::dynamic_pointer_cast<const core::AggregationNode>(
        builder.planNode());
    if (exchange) {
      builder.localPartition({"k"});
    }
    const auto plan = builder.finalAggregation().planNode();
    const auto final =
        std::dynamic_pointer_cast<const core::AggregationNode>(plan);
    EXPECT_EQ(nativeDecimalSumEligibility(*partial), std::vector<bool>{true});
    EXPECT_EQ(nativeDecimalSumEligibility(*final), std::vector<bool>{true});

    unregisterCudf();
    auto expected =
        exec::test::AssertQueryBuilder(plan).maxDrivers(2).copyResults(pool());
    registerCudf();
    const auto gpuPlan = rewriteToCudfPlan(plan, 2);
    assertNativeSumPlan(gpuPlan, true);
    auto task =
        exec::test::AssertQueryBuilder(gpuPlan)
            .maxDrivers(2)
            .config(CudfFromVelox::kGpuBatchSizeRows, "1")
            .config(
                core::QueryConfig::kMaxPartialAggregationMemory, partialMemory)
            .assertResults(expected);
    assertNativeSumStats(task);
  }
}

TEST_F(CudfDecimalTest, nativeDecimalSumIntermediateAndMultipleSources) {
  const auto decimalType = DECIMAL(18, 2);
  auto input = makeRowVector(
      {makeNullableFlatVector<int32_t>({0, 0, 1, std::nullopt}),
       makeNullableFlatVector<int64_t>(
           {100, -50, std::nullopt, 200}, decimalType)});
  for (bool intermediate : {false, true}) {
    SCOPED_TRACE(intermediate);
    auto ids = std::make_shared<core::PlanNodeIdGenerator>();
    auto left = exec::test::PlanBuilder(ids)
                    .values({input, input})
                    .partialAggregation({"c0"}, {"sum(c1) AS s"})
                    .planNode();
    auto right = exec::test::PlanBuilder(ids)
                     .values({input, input})
                     .partialAggregation({"c0"}, {"sum(c1) AS s"})
                     .planNode();
    auto builder =
        exec::test::PlanBuilder(ids).localPartition({"c0"}, {left, right});
    if (intermediate) {
      builder.intermediateAggregation();
    }
    const auto plan = builder.finalAggregation().planNode();
    unregisterCudf();
    auto expected =
        exec::test::AssertQueryBuilder(plan).maxDrivers(2).copyResults(pool());
    registerCudf();
    const auto gpuPlan = rewriteToCudfPlan(plan, 2);
    assertNativeSumPlan(gpuPlan, true);
    for (const auto& source : {left, right}) {
      const auto partial = dynamic_cast<const CudfAggregationNode*>(
          core::PlanNode::findNodeById(gpuPlan.get(), source->id()));
      ASSERT_NE(partial, nullptr);
      EXPECT_TRUE(partial->nativeDecimalSumStates()[0].output);
      EXPECT_EQ(*partial->outputType()->childAt(1), *DECIMAL(38, 2));
    }
    exec::test::AssertQueryBuilder(gpuPlan).maxDrivers(2).assertResults(
        expected);
  }
}

TEST_F(CudfDecimalTest, nativeDecimalSumGlobalStages) {
  constexpr int64_t value = 999'999'999'999'999'999;
  for (const bool allNull : {false, true}) {
    for (const bool intermediate : {false, true}) {
      SCOPED_TRACE(
          fmt::format("allNull={}, intermediate={}", allNull, intermediate));
      const std::optional<int64_t> inputValue =
          allNull ? std::nullopt : std::optional<int64_t>{value};
      auto batch = makeRowVector({makeNullableFlatVector<int64_t>(
          {inputValue, std::nullopt}, DECIMAL(18, 2))});
      auto builder = exec::test::PlanBuilder()
                         .values({batch, batch, batch})
                         .partialAggregation({}, {"sum(c0) AS s"});
      std::vector<core::PlanNodeId> aggregateIds{builder.planNode()->id()};
      if (intermediate) {
        builder.intermediateAggregation();
        aggregateIds.push_back(builder.planNode()->id());
      }
      const auto plan = builder.finalAggregation().planNode();
      aggregateIds.push_back(plan->id());
      const auto gpuPlan = rewriteToCudfPlan(plan);
      assertNativeSumPlan(gpuPlan, true);
      for (const auto& id : aggregateIds) {
        const auto* node = dynamic_cast<const CudfAggregationNode*>(
            core::PlanNode::findNodeById(gpuPlan.get(), id));
        ASSERT_NE(node, nullptr);
        EXPECT_TRUE(node->nativeDecimalSumStates()[0].buffer);
        EXPECT_EQ(*node->outputType()->childAt(0), *DECIMAL(38, 2));
      }
      const std::optional<int128_t> sum =
          allNull ? std::nullopt : std::optional<int128_t>{int128_t{value} * 3};
      auto expected = makeRowVector(
          {makeNullableFlatVector<int128_t>({sum}, DECIMAL(38, 2))});
      auto task = AssertQueryBuilder(gpuPlan)
                      .maxDrivers(1)
                      .config(CudfFromVelox::kGpuBatchSizeRows, "1")
                      .assertResults(expected);
      std::unordered_set<core::PlanNodeId> gpuAggregates;
      for (const auto& pipeline : task->taskStats().pipelineStats) {
        for (const auto& op : pipeline.operatorStats) {
          if (op.operatorType.starts_with("CudfReduce")) {
            gpuAggregates.insert(op.planNodeId);
            EXPECT_GT(op.inputVectors, 0);
            if (op.planNodeId == aggregateIds.front()) {
              EXPECT_GT(op.inputVectors, 1);
            }
          }
        }
      }
      for (const auto& id : aggregateIds) {
        EXPECT_EQ(gpuAggregates.count(id), 1);
      }
    }
  }
}

TEST_F(CudfDecimalTest, nativeDecimalSumSingleBuffer) {
  constexpr int64_t value = 999'999'999'999'999'999;
  auto batch = makeRowVector(
      {makeFlatVector<int32_t>({0, 0, 1}),
       makeNullableFlatVector<int64_t>(
           {value, std::nullopt, std::nullopt}, DECIMAL(18, 2))});
  const auto plan = exec::test::PlanBuilder()
                        .values({batch, batch, batch})
                        .singleAggregation({"c0"}, {"sum(c1) AS s"})
                        .planNode();
  const auto gpuPlan = rewriteToCudfPlan(plan);
  assertNativeSumPlan(gpuPlan, true);
  const auto* node = dynamic_cast<const CudfAggregationNode*>(
      core::PlanNode::findNodeById(gpuPlan.get(), plan->id()));
  ASSERT_NE(node, nullptr);
  const auto& state = node->nativeDecimalSumStates()[0];
  EXPECT_FALSE(state.input);
  EXPECT_FALSE(state.output);
  EXPECT_TRUE(state.buffer);
  EXPECT_EQ(*node->sources()[0]->outputType()->childAt(1), *DECIMAL(18, 2));
  EXPECT_EQ(*node->outputType()->childAt(1), *DECIMAL(38, 2));
  EXPECT_EQ(*getBufferedResultType(*node)->childAt(1), *DECIMAL(38, 2));
  auto expected = makeRowVector(
      {makeFlatVector<int32_t>({0, 1}),
       makeNullableFlatVector<int128_t>(
           {int128_t{value} * 3, std::nullopt}, DECIMAL(38, 2))});
  auto task = AssertQueryBuilder(gpuPlan)
                  .maxDrivers(1)
                  .config(CudfFromVelox::kGpuBatchSizeRows, "1")
                  .assertResults(expected);
  uint64_t inputVectors = 0;
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& op : pipeline.operatorStats) {
      if (op.planNodeId == plan->id() &&
          op.operatorType == "CudfGroupbySINGLE") {
        inputVectors += op.inputVectors;
      }
    }
  }
  EXPECT_GT(inputVectors, 1);
  const auto stats = exec::toPlanStats(task->taskStats());
  EXPECT_GT(
      stats.at(plan->id())
          .customStats.count(std::string{kDirectGroupbyFinalizationStat}),
      0);
}

TEST_F(CudfDecimalTest, nativeDecimalSumSharedSubplanKeepsSerializedState) {
  auto input = makeRowVector(
      {makeFlatVector<int32_t>({0}),
       makeFlatVector<int64_t>({100}, DECIMAL(18, 2))});
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  const auto partial = exec::test::PlanBuilder(ids)
                           .values({input})
                           .partialAggregation({"c0"}, {"sum(c1) AS s"})
                           .planNode();
  const auto plan = exec::test::PlanBuilder(ids)
                        .localPartition({"c0"}, {partial, partial})
                        .finalAggregation()
                        .planNode();
  const auto gpuPlan = rewriteToCudfPlan(plan);
  assertNativeSumPlan(gpuPlan, false);
  const auto gpuPartial = dynamic_cast<const CudfAggregationNode*>(
      core::PlanNode::findNodeById(gpuPlan.get(), partial->id()));
  ASSERT_NE(gpuPartial, nullptr);
  EXPECT_TRUE(gpuPartial->outputType()->childAt(1)->isVarbinary());
  const auto gpuFinal = dynamic_cast<const CudfAggregationNode*>(
      core::PlanNode::findNodeById(gpuPlan.get(), plan->id()));
  ASSERT_NE(gpuFinal, nullptr);
  EXPECT_TRUE(
      gpuFinal->aggregates()[0].call->inputs()[0]->type()->isVarbinary());
}

TEST_F(CudfDecimalTest, nativeDecimalSumStatePartitionKeyKeepsSerializedState) {
  auto input = makeRowVector(
      {makeFlatVector<int32_t>({0}),
       makeFlatVector<int64_t>({100}, DECIMAL(18, 2))});
  auto builder = exec::test::PlanBuilder().values({input}).partialAggregation(
      {"c0"}, {"sum(c1) AS s"});
  const auto partialId = builder.planNode()->id();
  const auto plan = builder.localPartition({"s"}).finalAggregation().planNode();
  const auto gpuPlan = rewriteToCudfPlan(plan);
  assertNativeSumPlan(gpuPlan, false);
  const auto gpuPartial = dynamic_cast<const CudfAggregationNode*>(
      core::PlanNode::findNodeById(gpuPlan.get(), partialId));
  ASSERT_NE(gpuPartial, nullptr);
  EXPECT_TRUE(gpuPartial->outputType()->childAt(1)->isVarbinary());
  const auto gpuFinal = dynamic_cast<const CudfAggregationNode*>(
      core::PlanNode::findNodeById(gpuPlan.get(), plan->id()));
  ASSERT_NE(gpuFinal, nullptr);
  EXPECT_TRUE(
      gpuFinal->aggregates()[0].call->inputs()[0]->type()->isVarbinary());
}

TEST_F(CudfDecimalTest, nativeDecimalSumModifiersAreIneligible) {
  auto input = makeRowVector(
      {makeFlatVector<int32_t>({0}),
       makeFlatVector<int64_t>({100}, DECIMAL(18, 2)),
       makeFlatVector<bool>({true})});
  // DISTINCT and ORDER BY are valid on SINGLE, not decomposed aggregation.
  for (const auto& expression :
       {"sum(c1)", "sum(DISTINCT c1)", "sum(c1 ORDER BY c0)"}) {
    SCOPED_TRACE(expression);
    const bool masked = std::string_view(expression) == "sum(c1)";
    const auto plan = exec::test::PlanBuilder()
                          .values({input})
                          .singleAggregation(
                              {"c0"},
                              {expression},
                              masked ? std::vector<std::string>{"c2"}
                                     : std::vector<std::string>{})
                          .planNode();
    const auto aggregate =
        std::dynamic_pointer_cast<const core::AggregationNode>(plan);
    ASSERT_NE(aggregate, nullptr);
    const auto& sum = aggregate->aggregates()[0];
    EXPECT_TRUE(sum.mask || sum.distinct || !sum.sortingKeys.empty());
    EXPECT_EQ(
        nativeDecimalSumEligibility(*aggregate), std::vector<bool>{false});
    assertNativeSumPlan(rewriteToCudfPlan(plan), false);
  }
}

TEST_F(CudfDecimalTest, nativeDecimalSumProjectionBridges) {
  const auto decimalType = DECIMAL(18, 2);
  const auto otherDecimalType = DECIMAL(12, 3);
  auto input = makeRowVector(
      {"k", "d", "e"},
      {makeFlatVector<int32_t>({0, 0, 1, 1, 2, 2, 3, 3}),
       makeNullableFlatVector<int64_t>(
           {100, -100, std::nullopt, std::nullopt, 200, std::nullopt, -50, -20},
           decimalType),
       makeNullableFlatVector<int64_t>(
           {1000, 2000, std::nullopt, std::nullopt, -3000, 1000, 0, 0},
           otherDecimalType)});
  for (const auto& partialMemory : {"1", "1048576"}) {
    SCOPED_TRACE(partialMemory);
    auto builder =
        exec::test::PlanBuilder()
            .values({input, input, input, input}, true)
            .partialAggregation(
                {"k"}, {"sum(d) AS s", "count(d) AS n", "sum(e) AS t"});
    auto partial = std::dynamic_pointer_cast<const core::AggregationNode>(
        builder.planNode());
    EXPECT_EQ(
        nativeDecimalSumEligibility(*partial),
        (std::vector<bool>{true, false, true}));
    auto plan =
        builder.project({"t AS t0", "k AS key0", "n AS n0", "s AS s0"})
            .project({"s0 AS s1", "n0 AS n1", "key0 AS key1", "t0 AS t1"})
            .filter("key1 > 0")
            .localPartition({"key1"})
            .project({"n1 AS n2", "t1 AS t2", "s1 AS s2", "key1 AS key2"})
            .project({"t2 AS t3", "key2 AS key3", "n2 AS n3", "s2 AS s3"})
            .finalAggregation(
                {"key3"},
                {"sum(t3) AS total_e", "sum(s3) AS total_d", "count(n3) AS n"},
                {{otherDecimalType}, {decimalType}, {decimalType}})
            .planNode();
    auto final = std::dynamic_pointer_cast<const core::AggregationNode>(plan);
    EXPECT_EQ(
        nativeDecimalSumEligibility(*final),
        (std::vector<bool>{true, true, false}));
    unregisterCudf();
    auto expected =
        exec::test::AssertQueryBuilder(plan).maxDrivers(2).copyResults(pool());
    registerCudf();
    const auto gpuPlan = rewriteToCudfPlan(plan, 2);
    assertNativeSumPlan(gpuPlan, true);
    const auto gpuFinal = dynamic_cast<const CudfAggregationNode*>(
        core::PlanNode::findNodeById(gpuPlan.get(), final->id()));
    ASSERT_NE(gpuFinal, nullptr);
    EXPECT_TRUE(gpuFinal->nativeDecimalSumStates()[0].input);
    EXPECT_TRUE(gpuFinal->nativeDecimalSumStates()[1].input);
    EXPECT_FALSE(gpuFinal->nativeDecimalSumStates()[2].buffer);
    auto task =
        exec::test::AssertQueryBuilder(gpuPlan)
            .maxDrivers(2)
            .config(CudfFromVelox::kGpuBatchSizeRows, "1")
            .config(
                core::QueryConfig::kMaxPartialAggregationMemory, partialMemory)
            .assertResults(expected);
    assertNativeSumStats(task);
  }
}

TEST_F(CudfDecimalTest, nativeDecimalSumMixedAggregateSelection) {
  const auto shortType = DECIMAL(18, 2);
  const auto longType = DECIMAL(20, 2);
  auto input = makeRowVector(
      {makeFlatVector<int32_t>({0, 0, 1}),
       makeNullableFlatVector<int64_t>({100, -50, std::nullopt}, shortType),
       makeNullableFlatVector<int128_t>({1000, -500, std::nullopt}, longType)});
  const auto plan =
      exec::test::PlanBuilder()
          .values({input, input})
          .partialAggregation(
              {"c0"}, {"sum(c1)", "avg(c1)", "sum(c2)", "count(c1)"})
          .finalAggregation()
          .planNode();
  unregisterCudf();
  auto expected = exec::test::AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();
  const auto gpuPlan = rewriteToCudfPlan(plan);
  assertNativeSumPlan(gpuPlan, true);
  const auto final = dynamic_cast<const CudfAggregationNode*>(
      core::PlanNode::findNodeById(gpuPlan.get(), plan->id()));
  ASSERT_NE(final, nullptr);
  EXPECT_TRUE(final->nativeDecimalSumStates()[0].input);
  for (size_t i = 1; i < 4; ++i) {
    EXPECT_FALSE(final->nativeDecimalSumStates()[i].input);
    EXPECT_FALSE(final->nativeDecimalSumStates()[i].output);
    EXPECT_FALSE(final->nativeDecimalSumStates()[i].buffer);
  }
  exec::test::AssertQueryBuilder(gpuPlan).assertResults(expected);
}

TEST_F(CudfDecimalTest, nativeDecimalSumStateExpressionsKeepSerializedPlan) {
  const auto decimalType = DECIMAL(18, 2);
  auto input = makeRowVector(
      {"k", "d", "e"},
      {makeFlatVector<int32_t>({0, 0, 1, 1, 2, 2}),
       makeNullableFlatVector<int64_t>(
           {100, -100, std::nullopt, std::nullopt, 200, std::nullopt},
           decimalType),
       makeNullableFlatVector<int64_t>(
           {500, -200, std::nullopt, std::nullopt, -50, 20}, decimalType)});
  for (bool filterState : {false, true}) {
    SCOPED_TRACE(filterState);
    auto builder =
        exec::test::PlanBuilder()
            .values({input, input, input, input}, true)
            .partialAggregation({"k"}, {"sum(d) AS s", "sum(e) AS t"});
    if (filterState) {
      builder.filter("s IS NOT NULL");
    } else {
      // A state-reading expression prevents native-state selection even
      // when the expression itself is supported on GPU.
      builder.project({"k", "s", "t", "s IS NULL AS missing"});
    }
    auto plan = builder.localPartition({"k"})
                    .finalAggregation(
                        {"k"},
                        {"sum(t) AS total_e", "sum(s) AS total_d"},
                        {{decimalType}, {decimalType}})
                    .planNode();
    unregisterCudf();
    auto expected =
        exec::test::AssertQueryBuilder(plan).maxDrivers(2).copyResults(pool());
    registerCudf();
    const auto gpuPlan = rewriteToCudfPlan(plan, 2);
    assertNativeSumPlan(gpuPlan, false);
    auto task =
        exec::test::AssertQueryBuilder(gpuPlan)
            .maxDrivers(2)
            .config(CudfFromVelox::kGpuBatchSizeRows, "1")
            .config(core::QueryConfig::kMaxPartialAggregationMemory, "1")
            .assertResults(expected);
    int64_t produced = 0;
    for (const auto& pipeline : task->taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        for (const auto& [name, metric] : op.runtimeStats) {
          if (name == "nativeDecimalSumProducedRows") {
            produced += metric.sum;
          }
        }
      }
    }
    EXPECT_EQ(produced, 0);
  }
}

TEST_F(CudfDecimalTest, serializedDecimalSumStateToCpu) {
  RowVectorPtr states;
  {
    const auto stream = cudf::get_default_stream();
    const auto mr = cudf::get_current_device_resource_ref();
    const auto stateType = ROW({{"k", BIGINT()}, {"s", VARBINARY()}});
    const std::vector<bool> valid{true, true, false, false, true, true};
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(makeInt64Column({0, 0, 1, 1, 2, 2}, nullptr, stream));
    auto sums = makeDecimalColumn<int128_t>(
        {300, -300, 123, 456, -500, 200}, 2, &valid, stream);
    auto inputCounts = makeInt64Column({1, 1, 0, 0, 1, 1}, nullptr, stream);
    columns.push_back(serializeDecimalSumState(
        sums->view(), inputCounts->view(), stream, mr));
    auto state = std::make_shared<CudfVector>(
        pool(),
        stateType,
        6,
        std::make_unique<cudf::table>(std::move(columns)),
        stream);
    auto decoded =
        deserializeDecimalSumState(state->getTableView().column(1), 2, stream);
    const auto counts = copyColumnData<int64_t>(decoded.count->view(), stream);
    for (size_t i = 0; i < valid.size(); ++i) {
      if (valid[i]) {
        EXPECT_EQ(counts[i], 1);
      }
    }
    states = with_arrow::toVeloxColumn(
        state->getTableView(), pool(), stateType, stream, mr);
    stream.sync();
  }
  auto expected = makeRowVector(
      {"k", "s"},
      {makeFlatVector<int64_t>({0, 1, 2}),
       makeNullableFlatVector<int128_t>(
           {0, std::nullopt, -300}, DECIMAL(38, 2))});
  unregisterCudf();
  for (bool intermediate : {false, true}) {
    SCOPED_TRACE(intermediate);
    auto plan =
        exec::test::PlanBuilder()
            .values({states})
            .finalAggregation({"k"}, {"sum(s) AS s"}, {{DECIMAL(18, 2)}})
            .planNode();
    if (intermediate) {
      const auto final =
          std::dynamic_pointer_cast<const core::AggregationNode>(plan);
      auto aggregates = final->aggregates();
      const auto& call = aggregates[0].call;
      aggregates[0].call = std::make_shared<core::CallTypedExpr>(
          VARBINARY(), call->inputs(), call->name());
      auto merge = std::make_shared<core::AggregationNode>(
          "merge",
          core::AggregationNode::Step::kIntermediate,
          final->groupingKeys(),
          final->preGroupedKeys(),
          final->aggregateNames(),
          aggregates,
          false,
          false,
          final->sources()[0]);
      plan = std::make_shared<core::AggregationNode>(
          final->id(),
          core::AggregationNode::Step::kFinal,
          final->groupingKeys(),
          final->preGroupedKeys(),
          final->aggregateNames(),
          final->aggregates(),
          false,
          false,
          merge);
    }
    exec::test::AssertQueryBuilder(plan).assertResults(expected);
  }
  registerCudf();
}

TEST_F(CudfDecimalTest, nativeDecimalSumResultRange) {
  const auto stream = cudf::get_default_stream();
  const int128_t limit = DecimalUtil::kPowersOfTen[38];
  auto valid = makeDecimalColumn<int128_t>(
      {limit - 1, -limit + 1, 0}, 2, nullptr, stream);
  EXPECT_NO_THROW(validateDecimalSumResult(valid->view(), stream));
  for (auto value :
       {limit,
        -limit,
        std::numeric_limits<int128_t>::min(),
        std::numeric_limits<int128_t>::max()}) {
    auto overflow = makeDecimalColumn<int128_t>({value}, 2, nullptr, stream);
    VELOX_ASSERT_THROW(
        validateDecimalSumResult(overflow->view(), stream), "Decimal overflow");
  }
  const std::vector<bool> validity{true, false, false, true};
  auto nullable = makeDecimalColumn<int128_t>(
      {limit, limit, -limit, -limit + 1}, 2, &validity, stream);
  auto slices = cudf::slice(nullable->view(), {1, 4}, stream);
  EXPECT_NO_THROW(validateDecimalSumResult(slices[0], stream));
  auto empty = makeDecimalColumn<int128_t>({}, 2, nullptr, stream);
  EXPECT_NO_THROW(validateDecimalSumResult(empty->view(), stream));
}

TEST_F(CudfDecimalTest, nativeDecimalSumTypedConcat) {
  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  const auto type = ROW({{"s", DECIMAL(38, 2)}});
  const std::vector<bool> valid{true, false};
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(makeDecimalColumn<int128_t>({123, 456}, 2, &valid, stream));
  auto native = std::make_shared<CudfVector>(
      pool(),
      type,
      2,
      std::make_unique<cudf::table>(std::move(columns)),
      stream);
  auto concatenated = getConcatenatedCudfVectorsBatched(
      pool(), {native, native}, type, stream, mr);
  ASSERT_EQ(concatenated.size(), 1);
  EXPECT_EQ(*concatenated[0]->type(), *type);
  auto column = concatenated[0]->getTableView().column(0);
  const auto sums = copyColumnData<int128_t>(column, stream);
  ASSERT_EQ(sums.size(), 4);
  EXPECT_EQ(sums[0], 123);
  EXPECT_EQ(sums[2], 123);
  const auto mask = copyNullMask(column, stream);
  EXPECT_TRUE(isValidAt(mask, 0));
  EXPECT_FALSE(isValidAt(mask, 1));
  EXPECT_TRUE(isValidAt(mask, 2));
  EXPECT_FALSE(isValidAt(mask, 3));
  EXPECT_EQ(column.type(), cudf::data_type(cudf::type_id::DECIMAL128, -2));
}

TEST_F(CudfDecimalTest, nativeDecimalSumUnsupportedBridgeKeepsSerializedState) {
  const auto decimalType = DECIMAL(18, 2);
  auto input = makeRowVector(
      {makeFlatVector<int32_t>({0, 0, 1}),
       makeNullableFlatVector<int64_t>(
           {100, -100, std::nullopt}, decimalType)});
  // Limit is not in the native-state proof's supported bridge set.
  const auto plan = exec::test::PlanBuilder()
                        .values({input})
                        .partialAggregation({"c0"}, {"sum(c1) AS s"})
                        .limit(0, 1000, true)
                        .finalAggregation({"c0"}, {"sum(s)"}, {{decimalType}})
                        .planNode();
  unregisterCudf();
  auto expected = exec::test::AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();
  const auto gpuPlan = rewriteToCudfPlan(plan);
  assertNativeSumPlan(gpuPlan, false);
  exec::test::AssertQueryBuilder(gpuPlan).assertResults(expected);
}

TEST_F(CudfDecimalTest, nativeDecimalSumBoundaryKeepsSerializedState) {
  const auto decimalType = DECIMAL(18, 2);
  auto input = makeRowVector(
      {makeFlatVector<int32_t>({0, 0, 1}),
       makeNullableFlatVector<int64_t>(
           {100, -100, std::nullopt}, decimalType)});
  const auto partial = exec::test::PlanBuilder()
                           .values({input})
                           .partialAggregation({"c0"}, {"sum(c1) AS s"})
                           .planNode();
  const auto gpuPartial = rewriteToCudfPlan(partial);
  assertNativeSumPlan(gpuPartial, false);
  EXPECT_EQ(*gpuPartial->outputType(), *partial->outputType());
  EXPECT_TRUE(gpuPartial->outputType()->childAt(1)->isVarbinary());
  auto states = exec::test::AssertQueryBuilder(gpuPartial).copyResults(pool());

  // The worker rewrites the source of its remote output, not the external
  // schema. No consumer in this fragment proves a native state contract.
  auto output = exec::test::PlanBuilder()
                    .values({input})
                    .partialAggregation({"c0"}, {"sum(c1) AS s"})
                    .partitionedOutput({}, 1)
                    .planNode();
  const auto rewrittenSource = rewriteToCudfPlan(output->sources()[0]);
  assertNativeSumPlan(rewrittenSource, false);
  EXPECT_EQ(*rewrittenSource->outputType(), *output->outputType());

  unregisterCudf();
  auto expected = makeRowVector(
      {makeFlatVector<int32_t>({0, 1}),
       makeNullableFlatVector<int128_t>({0, std::nullopt}, DECIMAL(38, 2))});
  auto final = exec::test::PlanBuilder()
                   .values({states})
                   .finalAggregation({"c0"}, {"sum(s)"}, {{decimalType}})
                   .planNode();
  exec::test::AssertQueryBuilder(final).assertResults(expected);
  registerCudf();
  // Serialized state arriving from another fragment is not reinterpreted as
  // decimal merely because the final SUM's raw input type was short decimal.
  const auto gpuFinal = rewriteToCudfPlan(final);
  assertNativeSumPlan(gpuFinal, false);
  exec::test::AssertQueryBuilder(gpuFinal).assertResults(expected);
}

TEST_F(CudfDecimalTest, decimalPartialSumVarbinaryToVeloxRoundTrip) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({100, 200, 300}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .planNode();

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  VELOX_CHECK_NOT_NULL(result);
  ASSERT_GT(result->size(), 0);
  ASSERT_EQ(result->childAt(0)->type()->kind(), TypeKind::VARBINARY);
}

TEST_F(CudfDecimalTest, decimalSumPartialFinalEmptyInput) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>({100, 200, 300}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .filter("k < 0")
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();

  AssertQueryBuilder(rewriteToCudfPlan(plan), duckDbQueryRunner_)
      .assertResults("SELECT k, sum(d) AS s FROM tmp WHERE k < 0 GROUP BY k");
}

TEST_F(CudfDecimalTest, decimalSumIntermediateVarbinary) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2}),
          makeFlatVector<int64_t>({12345, -2500, 10000}, DECIMAL(12, 2)),
      });
  auto input2 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({2, 3}),
          makeFlatVector<int64_t>({200, -300}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input1, input2};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  AssertQueryBuilder(rewriteToCudfPlan(plan), duckDbQueryRunner_)
      .assertResults("SELECT k, sum(d) AS s FROM tmp GROUP BY k");
}

TEST_F(CudfDecimalTest, decimalSumGlobalPartialFinalVarbinary) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({12345, -2500, 10000}, DECIMAL(12, 2))});
  auto input2 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({200, -300}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input1, input2};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();

  AssertQueryBuilder(rewriteToCudfPlan(plan), duckDbQueryRunner_)
      .assertResults("SELECT sum(d) AS s FROM tmp");
}

TEST_F(CudfDecimalTest, decimalSumGlobalIntermediateVarbinary) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({12345, -2500, 10000}, DECIMAL(12, 2))});
  auto input2 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({200, -300}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input1, input2};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  AssertQueryBuilder(rewriteToCudfPlan(plan), duckDbQueryRunner_)
      .assertResults("SELECT sum(d) AS s FROM tmp");
}

TEST_F(CudfDecimalTest, decimalSumGlobalSingle) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeFlatVector<int64_t>(
          {12345, -2500, 10000, 200, -300}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"sum(d) AS s"})
                  .planNode();

  AssertQueryBuilder(rewriteToCudfPlan(plan), duckDbQueryRunner_)
      .assertResults("SELECT sum(d) AS s FROM tmp");
}

// Masked groupby sum: the mask null-injects raw input so cuDF sum excludes
// masked rows. Group 3 is fully masked out -> NULL. Runs on GPU (fallback off).
TEST_F(CudfDecimalTest, decimalSumMaskedGroupbySingle) {
  auto input = makeRowVector(
      {"k", "d", "m"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 2, 3}),
          makeFlatVector<int64_t>(
              {12345, -2500, 10000, 200, -300}, DECIMAL(12, 2)),
          makeFlatVector<bool>({true, false, true, true, false}),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({"k"}, {"sum(d) AS s"}, {"m"})
                  .planNode();

  AssertQueryBuilder(rewriteToCudfPlan(plan), duckDbQueryRunner_)
      .assertResults(
          "SELECT k, sum(d) FILTER (WHERE m) AS s FROM tmp GROUP BY k");
}

// Masked groupby sum across partial + final steps: the mask applies only at the
// raw partial step and propagates through the serialized intermediate state.
TEST_F(CudfDecimalTest, decimalSumMaskedPartialFinal) {
  auto input = makeRowVector(
      {"k", "d", "m"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 2, 3}),
          makeFlatVector<int64_t>(
              {12345, -2500, 10000, 200, -300}, DECIMAL(12, 2)),
          makeFlatVector<bool>({true, false, true, true, false}),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"}, {"m"})
                  .finalAggregation()
                  .planNode();

  AssertQueryBuilder(rewriteToCudfPlan(plan), duckDbQueryRunner_)
      .assertResults(
          "SELECT k, sum(d) FILTER (WHERE m) AS s FROM tmp GROUP BY k");
}

// Masked global (reduce) sum, including a NULL mask entry which is excluded.
TEST_F(CudfDecimalTest, decimalSumMaskedGlobalSingle) {
  auto input = makeRowVector(
      {"d", "m"},
      {
          makeFlatVector<int64_t>(
              {12345, -2500, 10000, 200, -300}, DECIMAL(12, 2)),
          makeNullableFlatVector<bool>({true, false, true, std::nullopt, true}),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"sum(d) AS s"}, {"m"})
                  .planNode();

  AssertQueryBuilder(rewriteToCudfPlan(plan), duckDbQueryRunner_)
      .assertResults("SELECT sum(d) FILTER (WHERE m) AS s FROM tmp");
}

// Masked global (reduce) sum where every row is masked out: the input reduces
// to the empty set, so the result is NULL. Runs on GPU (fallback off).
TEST_F(CudfDecimalTest, decimalSumMaskedGlobalAllMasked) {
  auto input = makeRowVector(
      {"d", "m"},
      {
          makeFlatVector<int64_t>(
              {12345, -2500, 10000, 200, -300}, DECIMAL(12, 2)),
          makeFlatVector<bool>({false, false, false, false, false}),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"sum(d) AS s"}, {"m"})
                  .planNode();

  AssertQueryBuilder(rewriteToCudfPlan(plan), duckDbQueryRunner_)
      .assertResults("SELECT sum(d) FILTER (WHERE m) AS s FROM tmp");
}

TEST_F(CudfDecimalTest, decimalSumGroupbySingleDecimal64Overflow) {
  // One group of 12 values of 9e17 (DECIMAL(18,0)) sums to 1.08e19, past 2^63.
  // sum(decimal(18,0)) -> decimal(38,0), computed in 128 bits, no wrap.
  constexpr int64_t kBig = 900'000'000'000'000'000;
  constexpr int kNumRows = 12;
  std::vector<int32_t> keys(kNumRows, 1);
  std::vector<int64_t> values(kNumRows, kBig);

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>(keys),
          makeFlatVector<int64_t>(values, DECIMAL(18, 0)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  const int128_t expectedSum = static_cast<int128_t>(kBig) * kNumRows;
  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({"k"}, {"sum(d) AS s"})
                  .planNode();

  auto expected = makeRowVector(
      {"k", "s"},
      {
          makeFlatVector<int32_t>({1}),
          makeFlatVector<int128_t>({expectedSum}, DECIMAL(38, 0)),
      });

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumGlobalPartialFinalDecimal64Overflow) {
  // Global SUM whose total overflows DECIMAL64; exercises the partial raw sum
  // (serialized to VARBINARY) and the final merge, both in 128 bits.
  constexpr int64_t kBig = 900'000'000'000'000'000;
  constexpr int kNumRows = 12;
  std::vector<int64_t> values(kNumRows, kBig);

  auto input =
      makeRowVector({"d"}, {makeFlatVector<int64_t>(values, DECIMAL(18, 0))});

  std::vector<RowVectorPtr> vectors = {input};

  const int128_t expectedSum = static_cast<int128_t>(kBig) * kNumRows;
  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"s"}, {makeFlatVector<int128_t>({expectedSum}, DECIMAL(38, 0))});

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumPartialFinalVarbinaryNullGroup) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 2, 3, 3}),
          makeNullableFlatVector<int64_t>(
              {100, 200, std::nullopt, std::nullopt, 400, std::nullopt},
              DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"k", "s"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int128_t>(
              {static_cast<int128_t>(300),
               std::nullopt,
               static_cast<int128_t>(400)},
              DECIMAL(38, 2)),
      });

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumIntermediateVarbinaryNullGroup) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {100, std::nullopt, 400}, DECIMAL(12, 2)),
      });
  auto input2 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {200, std::nullopt, std::nullopt}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"k", "s"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int128_t>(
              {static_cast<int128_t>(300),
               std::nullopt,
               static_cast<int128_t>(400)},
              DECIMAL(38, 2)),
      });

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumGlobalPartialFinalVarbinaryAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"s"},
      {makeNullableFlatVector<int128_t>({std::nullopt}, DECIMAL(38, 2))});

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumGlobalIntermediateVarbinaryAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"s"},
      {makeNullableFlatVector<int128_t>({std::nullopt}, DECIMAL(38, 2))});

  auto result = AssertQueryBuilder(rewriteToCudfPlan(plan)).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalDeserializeSumStateDecimal64) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<int64_t> sums = {100, -200, 300};
  std::vector<int64_t> counts = {1, 2, 0};
  std::vector<bool> sumValid = {true, false, true};
  std::vector<bool> countValid = {true, true, true};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);
  auto sumAndCount = deserializeDecimalSumState(stateCol->view(), 2, stream);
  auto stateMask = copyNullMask(stateCol->view(), stream);
  auto sumMask = copyNullMask(sumAndCount.sum->view(), stream);
  EXPECT_EQ(stateMask, sumMask);

  auto outSum = copyColumnData<__int128_t>(sumAndCount.sum->view(), stream);
  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(sumMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], static_cast<__int128_t>(sums[i]));
    }
  }
}

TEST_F(CudfDecimalTest, decimalDeserializeSumStateDecimal128) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> sums = {
      static_cast<__int128_t>(123450),
      static_cast<__int128_t>(-25000),
      static_cast<__int128_t>(100000),
  };
  std::vector<int64_t> counts = {2, 1, 0};
  std::vector<bool> sumValid = {true, true, true};
  std::vector<bool> countValid = {true, false, true};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 3, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);
  auto sumAndCount = deserializeDecimalSumState(stateCol->view(), 3, stream);
  auto stateMask = copyNullMask(stateCol->view(), stream);
  auto sumMask = copyNullMask(sumAndCount.sum->view(), stream);
  EXPECT_EQ(stateMask, sumMask);

  auto outSum = copyColumnData<__int128_t>(sumAndCount.sum->view(), stream);
  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(sumMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], sums[i]);
    }
  }
}

// Reproduces the Q18 failure scenario: serializes a decimal sum state with
// partial nulls, round-trips through Arrow (which compacts null rows to 0-byte
// payloads), then deserializes. Without the null-count fix in
// deserializeDecimalSumState, the payload size check would fire because
// chars_size == (numRows - nullCount) * 32, not numRows * 32.
TEST_F(CudfDecimalTest, decimalDeserializeSumStatePartialNullCompact) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  // Serialize a 3-row state where row 1 is null (sum-null, count == 0).
  std::vector<int64_t> sums = {100, 0, 300};
  std::vector<int64_t> counts = {1, 0, 2};
  std::vector<bool> sumValid = {true, false, true};
  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);

  // Round-trip through Arrow (cuDF -> Velox VARBINARY -> cuDF STRING).
  // Arrow stores null rows with 0-byte payloads, so the resulting cuDF STRING
  // column has chars_size == (numRows - nullCount) * 32.
  auto expectedType = ROW({{"s", VARBINARY()}});
  auto veloxRow = with_arrow::toVeloxColumn(
      cudf::table_view{{stateCol->view()}},
      pool(),
      expectedType,
      "s",
      stream,
      mr);
  auto compactTable = with_arrow::toCudfTable(veloxRow, pool(), stream, mr);
  auto compactStateView = compactTable->view().column(0);

  // Verify the column now has the compact layout.
  cudf::strings_column_view strings(compactStateView);
  EXPECT_LT(
      strings.chars_size(stream),
      static_cast<int64_t>(sums.size()) * 32); // 32 == kDecimalSumStateSize

  auto result = deserializeDecimalSumState(compactStateView, 2, stream);

  auto outSum = copyColumnData<__int128_t>(result.sum->view(), stream);
  auto outCount = copyColumnData<int64_t>(result.count->view(), stream);
  auto outMask = copyNullMask(result.sum->view(), stream);

  EXPECT_TRUE(isValidAt(outMask, 0));
  EXPECT_FALSE(isValidAt(outMask, 1));
  EXPECT_TRUE(isValidAt(outMask, 2));
  EXPECT_EQ(outSum[0], static_cast<__int128_t>(100));
  EXPECT_EQ(outCount[0], 1);
  EXPECT_EQ(outSum[2], static_cast<__int128_t>(300));
  EXPECT_EQ(outCount[2], 2);
}

// Reproduces the TPC-DS Q1 failure in CudfGroupbyFINAL: the streaming final
// aggregation concatenates its buffered result -- which came straight from
// serializeDecimalSumState and so keeps a 32-byte payload for null rows -- with
// each newly arrived batch, which was round tripped through velox and so has
// its null rows compacted to 0 bytes. The concatenated state column mixes both
// encodings, so its payload size is neither numRows * 32 nor
// (numRows - nullCount) * 32 and a two-way equality check rejects it.
TEST_F(CudfDecimalTest, decimalDeserializeSumStateMixedNullEncodings) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  auto serialize = [&](const std::vector<int64_t>& sums,
                       const std::vector<int64_t>& counts,
                       const std::vector<bool>& sumValid) {
    auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
    auto countCol = makeInt64Column(counts, nullptr, stream);
    return serializeDecimalSumState(
        sumCol->view(), countCol->view(), stream, mr);
  };

  // Buffered side: row 1 is null and keeps its 32-byte payload.
  auto fullState = serialize({100, 0, 300}, {1, 0, 2}, {true, false, true});

  // Incoming side: same shape, but the velox round trip compacts its null row
  // to a 0-byte payload.
  auto incomingState = serialize({400, 0, 600}, {3, 0, 4}, {true, false, true});
  auto veloxRow = with_arrow::toVeloxColumn(
      cudf::table_view{{incomingState->view()}},
      pool(),
      ROW({{"s", VARBINARY()}}),
      "s",
      stream,
      mr);
  auto compactTable = with_arrow::toCudfTable(veloxRow, pool(), stream, mr);

  auto mixed = cudf::concatenate(
      std::vector<cudf::column_view>{
          fullState->view(), compactTable->view().column(0)},
      stream,
      mr);

  // Four non-null rows, plus the one null payload the buffered side kept: the
  // payload size sits strictly between the compact and full extremes.
  cudf::strings_column_view mixedStrings(mixed->view());
  auto const numRows = static_cast<int64_t>(mixed->size());
  auto const nullCount = static_cast<int64_t>(mixed->null_count());
  EXPECT_EQ(numRows, 6);
  EXPECT_EQ(nullCount, 2);
  EXPECT_GT(
      mixedStrings.chars_size(stream),
      (numRows - nullCount) * 32); // 32 == kDecimalSumStateSize
  EXPECT_LT(mixedStrings.chars_size(stream), numRows * 32);

  auto result = deserializeDecimalSumState(mixed->view(), 2, stream);

  auto outSum = copyColumnData<__int128_t>(result.sum->view(), stream);
  auto outCount = copyColumnData<int64_t>(result.count->view(), stream);
  auto outMask = copyNullMask(result.sum->view(), stream);

  EXPECT_FALSE(isValidAt(outMask, 1));
  EXPECT_FALSE(isValidAt(outMask, 4));
  for (auto row : {0, 2, 3, 5}) {
    EXPECT_TRUE(isValidAt(outMask, row));
  }
  EXPECT_EQ(outSum[0], static_cast<__int128_t>(100));
  EXPECT_EQ(outCount[0], 1);
  EXPECT_EQ(outSum[2], static_cast<__int128_t>(300));
  EXPECT_EQ(outCount[2], 2);
  EXPECT_EQ(outSum[3], static_cast<__int128_t>(400));
  EXPECT_EQ(outCount[3], 3);
  EXPECT_EQ(outSum[5], static_cast<__int128_t>(600));
  EXPECT_EQ(outCount[5], 4);
}

// Trailing null: the offset for the last row equals chars_size, so the kernel
// would read 32 bytes past the buffer end without the null-mask guard.
TEST_F(CudfDecimalTest, decimalDeserializeSumStateTrailingNullCompact) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  // 3-row state: [valid, valid, null].
  std::vector<int64_t> sums = {100, 200, 0};
  std::vector<int64_t> counts = {1, 2, 0};
  std::vector<bool> sumValid = {true, true, false};
  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);

  auto expectedType = ROW({{"s", VARBINARY()}});
  auto veloxRow = with_arrow::toVeloxColumn(
      cudf::table_view{{stateCol->view()}},
      pool(),
      expectedType,
      "s",
      stream,
      mr);
  auto compactTable = with_arrow::toCudfTable(veloxRow, pool(), stream, mr);
  auto compactStateView = compactTable->view().column(0);

  cudf::strings_column_view strings(compactStateView);
  EXPECT_LT(strings.chars_size(stream), static_cast<int64_t>(sums.size()) * 32);

  auto result = deserializeDecimalSumState(compactStateView, 2, stream);

  auto outSum = copyColumnData<__int128_t>(result.sum->view(), stream);
  auto outCount = copyColumnData<int64_t>(result.count->view(), stream);
  auto outMask = copyNullMask(result.sum->view(), stream);

  EXPECT_TRUE(isValidAt(outMask, 0));
  EXPECT_TRUE(isValidAt(outMask, 1));
  EXPECT_FALSE(isValidAt(outMask, 2));
  EXPECT_EQ(outSum[0], static_cast<__int128_t>(100));
  EXPECT_EQ(outCount[0], 1);
  EXPECT_EQ(outSum[1], static_cast<__int128_t>(200));
  EXPECT_EQ(outCount[1], 2);
}

// Leading null: offset 0 overlaps the next valid row's data so the read is
// in-bounds, but verify the null mask propagates correctly.
TEST_F(CudfDecimalTest, decimalDeserializeSumStateLeadingNullCompact) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  // 3-row state: [null, valid, valid].
  std::vector<int64_t> sums = {0, 200, 300};
  std::vector<int64_t> counts = {0, 2, 3};
  std::vector<bool> sumValid = {false, true, true};
  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);

  auto expectedType = ROW({{"s", VARBINARY()}});
  auto veloxRow = with_arrow::toVeloxColumn(
      cudf::table_view{{stateCol->view()}},
      pool(),
      expectedType,
      "s",
      stream,
      mr);
  auto compactTable = with_arrow::toCudfTable(veloxRow, pool(), stream, mr);
  auto compactStateView = compactTable->view().column(0);

  cudf::strings_column_view strings(compactStateView);
  EXPECT_LT(strings.chars_size(stream), static_cast<int64_t>(sums.size()) * 32);

  auto result = deserializeDecimalSumState(compactStateView, 2, stream);

  auto outSum = copyColumnData<__int128_t>(result.sum->view(), stream);
  auto outCount = copyColumnData<int64_t>(result.count->view(), stream);
  auto outMask = copyNullMask(result.sum->view(), stream);

  EXPECT_FALSE(isValidAt(outMask, 0));
  EXPECT_TRUE(isValidAt(outMask, 1));
  EXPECT_TRUE(isValidAt(outMask, 2));
  EXPECT_EQ(outSum[1], static_cast<__int128_t>(200));
  EXPECT_EQ(outCount[1], 2);
  EXPECT_EQ(outSum[2], static_cast<__int128_t>(300));
  EXPECT_EQ(outCount[2], 3);
}

TEST_F(CudfDecimalTest, decimalDeserializeSumStateAllNull) {
  auto stream = cudf::get_default_stream();
  constexpr cudf::size_type numRows = 4;

  auto offsetsCol = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT32},
      numRows + 1,
      cudf::mask_state::UNALLOCATED,
      stream);
  auto* offsetsPtr = offsetsCol->mutable_view().data<int32_t>();
  auto status = cudaMemsetAsync(
      offsetsPtr,
      0,
      static_cast<size_t>(numRows + 1) * sizeof(int32_t),
      stream.get());
  VELOX_CHECK_EQ(0, static_cast<int>(status));
  stream.sync();

  std::vector<bool> valid(numRows, false);
  auto [nullMask, nullCount] = makeNullMask(valid, stream);
  rmm::device_buffer charsBuf(0, stream);
  auto stateCol = cudf::make_strings_column(
      numRows,
      std::move(offsetsCol),
      std::move(charsBuf),
      nullCount,
      std::move(nullMask));

  auto sumAndCount = deserializeDecimalSumState(stateCol->view(), 2, stream);
  auto outSumView = sumAndCount.sum->view();
  auto outCountView = sumAndCount.count->view();

  EXPECT_EQ(outSumView.size(), numRows);
  EXPECT_EQ(outCountView.size(), numRows);
  EXPECT_EQ(outSumView.null_count(), numRows);
  EXPECT_EQ(outCountView.null_count(), numRows);

  auto outSumMask = copyNullMask(outSumView, stream);
  auto outCountMask = copyNullMask(outCountView, stream);
  for (size_t i = 0; i < static_cast<size_t>(numRows); ++i) {
    EXPECT_FALSE(isValidAt(outSumMask, i));
    EXPECT_FALSE(isValidAt(outCountMask, i));
  }
}

TEST_F(CudfDecimalTest, decimalSerializeSumStateUsesInt64OffsetsWhenEnabled) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  ScopedEnvVar enableLargeStrings("LIBCUDF_LARGE_STRINGS_ENABLED", "1");
  ScopedEnvVar threshold("LIBCUDF_LARGE_STRINGS_THRESHOLD", "1");

  std::vector<int64_t> sums = {100, -200};
  std::vector<int64_t> counts = {1, 1};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, nullptr, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);

  cudf::strings_column_view strings(stateCol->view());
  EXPECT_EQ(strings.offsets().type().id(), cudf::type_id::INT64);
}

TEST_F(CudfDecimalTest, decimalSumStateRoundTripUsesInt64Offsets) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  ScopedEnvVar enableLargeStrings("LIBCUDF_LARGE_STRINGS_ENABLED", "1");
  ScopedEnvVar threshold("LIBCUDF_LARGE_STRINGS_THRESHOLD", "1");

  std::vector<int64_t> sums = {100, -200, 300, 400};
  std::vector<int64_t> counts = {1, 0, 2, 3};
  std::vector<bool> sumValid = {true, true, false, true};
  std::vector<bool> countValid = {true, true, true, false};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);

  cudf::strings_column_view strings(stateCol->view());
  EXPECT_EQ(strings.offsets().type().id(), cudf::type_id::INT64);

  auto sumAndCount = deserializeDecimalSumState(stateCol->view(), 2, stream);
  auto outSumView = sumAndCount.sum->view();
  auto outCountView = sumAndCount.count->view();
  auto outSum = copyColumnData<__int128_t>(outSumView, stream);
  auto outCount = copyColumnData<int64_t>(outCountView, stream);
  auto outSumMask = copyNullMask(outSumView, stream);
  auto outCountMask = copyNullMask(outCountView, stream);

  EXPECT_EQ(outSumView.size(), sums.size());
  EXPECT_EQ(outCountView.size(), counts.size());
  EXPECT_EQ(outSumMask, outCountMask);

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(outSumMask, i), expectedValid);
    EXPECT_EQ(isValidAt(outCountMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], static_cast<__int128_t>(sums[i]));
      EXPECT_EQ(outCount[i], counts[i]);
    }
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal64) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<int64_t> sums = {100, 105, 250, -125};
  std::vector<int64_t> counts = {4, 2, 0, 2};
  std::vector<bool> sumValid = {true, true, true, true};
  std::vector<bool> countValid = {true, false, true, true};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto avgCol =
      computeDecimalAverage(sumCol->view(), countCol->view(), stream, mr);

  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<int64_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return static_cast<int64_t>(out);
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(avgMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i]));
    }
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal128) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> sums = {
      static_cast<__int128_t>(123450),
      static_cast<__int128_t>(-25000),
      static_cast<__int128_t>(100000),
  };
  std::vector<int64_t> counts = {3, 2, 0};
  std::vector<bool> sumValid = {true, true, true};
  std::vector<bool> countValid = {true, true, true};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 3, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto avgCol =
      computeDecimalAverage(sumCol->view(), countCol->view(), stream, mr);

  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<__int128_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return out;
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(avgMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i]));
    }
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal64MostNegativeSum) {
  // Negating INT64_MIN in the signed type is overflow UB; the magnitude and
  // sign must be handled in the unsigned domain. avg of one INT64_MIN is
  // itself.
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  std::vector<int64_t> sums = {kMin};
  std::vector<int64_t> counts = {1};
  std::vector<bool> valid = {true};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 0, &valid, stream);
  auto countCol = makeInt64Column(counts, &valid, stream);
  auto avgCol =
      computeDecimalAverage(sumCol->view(), countCol->view(), stream, mr);

  auto outAvg = copyColumnData<int64_t>(avgCol->view(), stream);
  EXPECT_EQ(outAvg[0], kMin);
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal128MostNegativeSum) {
  // Same regression at the __int128 boundary. avg of one -2^127 is itself.
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  const __int128_t kMin =
      static_cast<__int128_t>(static_cast<unsigned __int128>(1) << 127);
  std::vector<__int128_t> sums = {kMin};
  std::vector<int64_t> counts = {1};
  std::vector<bool> valid = {true};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 0, &valid, stream);
  auto countCol = makeInt64Column(counts, &valid, stream);
  auto avgCol =
      computeDecimalAverage(sumCol->view(), countCol->view(), stream, mr);

  auto outAvg = copyColumnData<__int128_t>(avgCol->view(), stream);
  EXPECT_EQ(outAvg[0], kMin);
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal64AllValid) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<int64_t> sums = {100, 200, -150};
  std::vector<int64_t> counts = {4, 5, 3};
  std::vector<bool> sumValid = {true, true, true};
  std::vector<bool> countValid = {true, true, true};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto avgCol =
      computeDecimalAverage(sumCol->view(), countCol->view(), stream, mr);

  EXPECT_EQ(avgCol->view().null_count(), 0);
  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<int64_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return static_cast<int64_t>(out);
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    if (avgCol->view().nullable()) {
      EXPECT_TRUE(isValidAt(avgMask, i)) << "row " << i;
    }
    EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i])) << "row " << i;
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal128AllValid) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> sums = {
      static_cast<__int128_t>(90000),
      static_cast<__int128_t>(-5000),
      static_cast<__int128_t>(1),
  };
  std::vector<int64_t> counts = {3, 5, 1};
  std::vector<bool> sumValid = {true, true, true};
  std::vector<bool> countValid = {true, true, true};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 3, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto avgCol =
      computeDecimalAverage(sumCol->view(), countCol->view(), stream, mr);

  EXPECT_EQ(avgCol->view().null_count(), 0);
  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<__int128_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return out;
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    if (avgCol->view().nullable()) {
      EXPECT_TRUE(isValidAt(avgMask, i)) << "row " << i;
    }
    EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i])) << "row " << i;
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal64NonNullableInputs) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<int64_t> sums = {80, -40, 1000};
  std::vector<int64_t> counts = {2, 4, 10};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, nullptr, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  ASSERT_FALSE(sumCol->nullable());
  ASSERT_FALSE(countCol->nullable());

  auto avgCol =
      computeDecimalAverage(sumCol->view(), countCol->view(), stream, mr);

  EXPECT_EQ(avgCol->view().null_count(), 0);
  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<int64_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return static_cast<int64_t>(out);
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    if (avgCol->view().nullable()) {
      EXPECT_TRUE(isValidAt(avgMask, i)) << "row " << i;
    }
    EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i])) << "row " << i;
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal128NonNullableInputs) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> sums = {
      static_cast<__int128_t>(600),
      static_cast<__int128_t>(-99),
  };
  std::vector<int64_t> counts = {3, 9};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 4, nullptr, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  ASSERT_FALSE(sumCol->nullable());
  ASSERT_FALSE(countCol->nullable());

  auto avgCol =
      computeDecimalAverage(sumCol->view(), countCol->view(), stream, mr);

  EXPECT_EQ(avgCol->view().null_count(), 0);
  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<__int128_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return out;
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    if (avgCol->view().nullable()) {
      EXPECT_TRUE(isValidAt(avgMask, i)) << "row " << i;
    }
    EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i])) << "row " << i;
  }
}

TEST_F(CudfDecimalTest, decimalSumStateRoundTripDecimal64) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<int64_t> sums = {100, -200, 300, 400};
  std::vector<int64_t> counts = {1, 0, 2, 3};
  std::vector<bool> sumValid = {true, true, false, true};
  std::vector<bool> countValid = {true, true, true, false};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);
  auto stateMask = copyNullMask(stateCol->view(), stream);

  auto sumAndCount = deserializeDecimalSumState(stateCol->view(), 2, stream);
  auto outSumView = sumAndCount.sum->view();
  auto outCountView = sumAndCount.count->view();
  auto outSum = copyColumnData<__int128_t>(outSumView, stream);
  auto outCount = copyColumnData<int64_t>(outCountView, stream);
  auto outSumMask = copyNullMask(outSumView, stream);
  auto outCountMask = copyNullMask(outCountView, stream);

  EXPECT_EQ(stateMask, outSumMask);
  EXPECT_EQ(stateMask, outCountMask);

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(stateMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], static_cast<__int128_t>(sums[i]));
      EXPECT_EQ(outCount[i], counts[i]);
    }
  }
}

TEST_F(CudfDecimalTest, decimalSumStateRoundTripDecimal128) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> sums = {
      static_cast<__int128_t>(123450),
      static_cast<__int128_t>(-25000),
      static_cast<__int128_t>(100000),
  };
  std::vector<int64_t> counts = {2, 1, 0};
  std::vector<bool> sumValid = {true, false, true};
  std::vector<bool> countValid = {true, true, true};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 3, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);
  auto stateMask = copyNullMask(stateCol->view(), stream);

  auto sumAndCount = deserializeDecimalSumState(stateCol->view(), 3, stream);
  auto outSumView = sumAndCount.sum->view();
  auto outCountView = sumAndCount.count->view();
  auto outSum = copyColumnData<__int128_t>(outSumView, stream);
  auto outCount = copyColumnData<int64_t>(outCountView, stream);
  auto outSumMask = copyNullMask(outSumView, stream);
  auto outCountMask = copyNullMask(outCountView, stream);

  EXPECT_EQ(stateMask, outSumMask);
  EXPECT_EQ(stateMask, outCountMask);

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(stateMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], sums[i]);
      EXPECT_EQ(outCount[i], counts[i]);
    }
  }
}

TEST_F(CudfDecimalTest, cudfVarbinaryArrowRoundTrip) {
  auto input = makeRowVector(
      {"bin"},
      {makeNullableFlatVector<std::string>(
          {std::string("abc"), std::nullopt, std::string("xyz")},
          VARBINARY())});

  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto cudfTable = with_arrow::toCudfTable(input, pool(), stream, mr);
  auto roundTrip =
      with_arrow::toVeloxColumn(cudfTable->view(), pool(), "rt_", stream, mr);

  ASSERT_EQ(roundTrip->childAt(0)->type()->kind(), TypeKind::VARCHAR);
  VELOX_ASSERT_THROW(
      roundTrip->setType(ROW({{"rt_0", VARBINARY()}})),
      "Cannot change vector type");

  auto expected = makeRowVector(
      {"rt_0"},
      {makeNullableFlatVector<std::string>(
          {std::string("abc"), std::nullopt, std::string("xyz")}, VARCHAR())});

  facebook::velox::test::assertEqualVectors(expected, roundTrip);
}

TEST_F(CudfDecimalTest, cudfVarbinaryArrowRoundTripWithExpectedType) {
  auto input = makeRowVector(
      {"bin"},
      {makeNullableFlatVector<std::string>(
          {std::string("abc"), std::nullopt, std::string("xyz")},
          VARBINARY())});

  auto expectedType = ROW({{"bin", VARBINARY()}});

  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto cudfTable = with_arrow::toCudfTable(input, pool(), stream, mr);
  auto roundTrip = with_arrow::toVeloxColumn(
      cudfTable->view(), pool(), expectedType, "rt_", stream, mr);

  ASSERT_EQ(roundTrip->childAt(0)->type()->kind(), TypeKind::VARBINARY);

  auto expected = makeRowVector(
      {"rt_0"},
      {makeNullableFlatVector<std::string>(
          {std::string("abc"), std::nullopt, std::string("xyz")},
          VARBINARY())});

  facebook::velox::test::assertEqualVectors(expected, roundTrip);
}

TEST_F(CudfDecimalTest, cudfVarbinaryRowTypeMismatch) {
  auto input = makeRowVector(
      {"l_returnflag",
       "l_linestatus",
       "avg_51",
       "avg_52",
       "avg_53",
       "count_54",
       "sum_47",
       "sum_48",
       "sum_49",
       "sum_50"},
      {makeFlatVector<std::string>({"A", "B"}, VARCHAR()),
       makeFlatVector<std::string>({"F", "O"}, VARCHAR()),
       makeFlatVector<std::string>({"x", "y"}, VARBINARY()),
       makeFlatVector<std::string>({"p", "q"}, VARBINARY()),
       makeFlatVector<std::string>({"m", "n"}, VARBINARY()),
       makeFlatVector<int64_t>({10, 20}, BIGINT()),
       makeFlatVector<std::string>({"u", "v"}, VARBINARY()),
       makeFlatVector<std::string>({"r", "s"}, VARBINARY()),
       makeFlatVector<std::string>({"t", "w"}, VARBINARY()),
       makeFlatVector<std::string>({"c", "d"}, VARBINARY())});

  auto expectedType = ROW({
      {"l_returnflag", VARCHAR()},
      {"l_linestatus", VARCHAR()},
      {"avg_51", VARBINARY()},
      {"avg_52", VARBINARY()},
      {"avg_53", VARBINARY()},
      {"count_54", BIGINT()},
      {"sum_47", VARBINARY()},
      {"sum_48", VARBINARY()},
      {"sum_49", VARBINARY()},
      {"sum_50", VARBINARY()},
  });

  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto cudfTable = with_arrow::toCudfTable(input, pool(), stream, mr);
  auto roundTrip =
      with_arrow::toVeloxColumn(cudfTable->view(), pool(), "rt_", stream, mr);

  ASSERT_EQ(roundTrip->childAt(2)->type()->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(roundTrip->childAt(3)->type()->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(roundTrip->childAt(4)->type()->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(roundTrip->childAt(6)->type()->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(roundTrip->childAt(7)->type()->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(roundTrip->childAt(8)->type()->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(roundTrip->childAt(9)->type()->kind(), TypeKind::VARCHAR);

  VELOX_ASSERT_THROW(
      roundTrip->setType(expectedType), "Cannot change vector type");
}

} // namespace
} // namespace facebook::velox::cudf_velox
