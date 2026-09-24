# Native GPU State for Decimal64 SUM

## Goal

Avoid materializing the 32-byte `VARBINARY` decimal SUM state between compatible GPU partial and final aggregations.

This first version is deliberately narrow:

- aggregate: decimal `SUM` only;
- raw input: `DECIMAL64` only;
- producer: `CudfGroupbyPARTIAL`;
- transport: GPU `CudfLocalPartition`;
- consumer: `CudfGroupbyFINAL`;
- fallback: materialize the existing CPU-compatible `VARBINARY` state at every unsupported boundary.

The motivating Q18 pipeline is:

```text
CudfGroupbyPARTIAL(l_orderkey, sum(l_quantity DECIMAL64))
    -> CudfLocalPartition HASH(l_orderkey)
    -> CudfGroupbyFINAL(l_orderkey, sum(partial state))
```

The plan's logical intermediate type remains `VARBINARY`. Only its physical representation inside a compatible GPU pipeline changes.

## Why the existing representation is expensive

The existing intermediate state mirrors `LongDecimalWithOverflowState`:

```text
count:     INT64       8 bytes
overflow:  INT64       8 bytes
sum:       INT128     16 bytes
                       --------
payload:               32 bytes per group
```

On GPU this payload is represented as a cuDF STRING column, which also needs string offsets and a validity mask. Q18 emits 1.5 billion partial groups, so both the payload and the temporary decode/encode columns are large.

The final incremental group-by may repeatedly:

1. decode the incoming STRING into sum and count columns;
2. concatenate it with buffered state;
3. group the combined state;
4. encode the result back into STRING.

The proposed representation avoids these conversions while the state remains in the GPU pipeline.

## Native representation

Use one nullable `DECIMAL128` cuDF column:

```text
physical data:  DECIMAL128 partial sum
physical null:  group has no non-null input
logical type:   VARBINARY
encoding tag:   NativeDecimal64SumState
```

No empty or placeholder `VARBINARY` values are produced. The native column is the physical value. A real `VARBINARY` column is created only when an unsupported consumer requires the standard intermediate representation.

A STRUCT column is unnecessary for this first version because the state has only one value plus validity.

## Why sum plus validity is sufficient

For a SUM whose raw input type is `DECIMAL64`:

- the exact input count is not needed to merge sums;
- validity distinguishes an empty/all-null group from a valid sum of zero;
- a `DECIMAL64` value widens exactly to signed 128 bits;
- the aggregate's global row count is represented by signed 64-bit counters;
- therefore, even a conservative bound using an `INT64` value and `INT64_MAX` rows is below the signed 128-bit range.

In particular:

```text
(2^63 - 1) * (2^63 - 1) < 2^126 < 2^127
```

Actual DECIMAL64 values have decimal precision at most 18 and are more tightly bounded. Combining any valid set of DECIMAL64 inputs therefore does not require the long-decimal overflow counter used for DECIMAL128 raw inputs.

The final result must still perform the normal `DECIMAL(38, scale)` range validation. This optimization does not relax SQL overflow checks.

This reasoning does not apply to:

- raw `DECIMAL128` inputs;
- AVG, which needs the count;
- a state whose raw input width cannot be proven to be DECIMAL64;
- arbitrary CPU-produced `VARBINARY` states, which may contain an overflow count.

Those cases continue to use the existing representation.

## Encoding metadata

Do not infer native state solely from a mismatch between the logical Velox type and cuDF type. Add explicit per-column physical-encoding metadata to `CudfVector`, for example:

```cpp
enum class CudfPhysicalEncoding {
  kDefault,
  kNativeDecimal64SumState,
};
```

The metadata must:

- have one entry per logical top-level column;
- survive selection, slicing, splitting, concatenation, and local partitioning;
- be rejected or materialized by operators that do not understand it;
- participate in validation and debugging output;
- not change the logical `RowType` seen by the plan.

For `kNativeDecimal64SumState`, validate that the physical column is nullable `DECIMAL128` with the expected scale.

## Producer behavior

When `CudfGroupbyPARTIAL` handles decimal SUM with raw `DECIMAL64` input and the downstream path supports native state:

1. aggregate into a nullable `DECIMAL128` result per group;
2. use result validity to represent whether the group observed a non-null value;
3. emit that column in the logical `VARBINARY` output position;
4. mark the column `kNativeDecimal64SumState`;
5. do not call `serializeDecimalSumState`.

The initial implementation should enable this only when compatibility of the immediate path is known. Otherwise, retain the existing `VARBINARY` output.

## Local exchange behavior

`CudfLocalPartition` already passes `CudfVector` objects through local exchange queues and uses `cudf::hash_partition` on their physical tables. A nullable fixed-width `DECIMAL128` column can be partitioned along with its grouping key without serialization.

Required changes:

1. preserve physical-encoding metadata when constructing each partition vector;
2. keep the native state column out of the partition-key set unless it is logically a key;
3. account for its actual retained GPU bytes;
4. ensure queueing, splitting, and stream ordering preserve the column and its null mask.

There is no HTTP or page serialization in this Q18 local-exchange path.

## Consumer behavior

When `CudfGroupbyFINAL` receives a column tagged `kNativeDecimal64SumState`:

1. validate that the aggregate is decimal SUM whose raw input type was DECIMAL64;
2. consume the physical `DECIMAL128` column directly;
3. group and sum it using null exclusion;
4. preserve the resulting validity;
5. retain the same native encoding for any internal incremental buffered result;
6. cast or validate the final result as `DECIMAL(38, scale)` when producing final query output.

The final operator must not call `deserializeDecimalSumState` for native input.

For incremental final aggregation, both the incoming and buffered state should remain native. Concatenation then operates directly on nullable fixed-width columns instead of STRING payloads.

## Materialization boundary

Provide one explicit conversion:

```text
NativeDecimal64SumState -> standard 32-byte VARBINARY state
```

Materialization is required before:

- CPU execution or fallback;
- remote/HTTP exchange;
- generic spill or row serialization that does not preserve the encoding;
- conversion to an ordinary Velox vector;
- an expression or operator that treats the value as actual VARBINARY;
- mixing with an untagged or CPU-produced decimal SUM state;
- any unsupported local-exchange implementation.

To produce the existing state for each valid native row:

```text
count = 1
overflow = 0
sum = native DECIMAL128 sum
```

For SUM, `count = 1` is sufficient: downstream merging only needs zero versus nonzero to preserve NULL semantics. A null native row becomes a null intermediate state. This conversion must be verified against CPU partial, intermediate, and final aggregate paths.

The reverse conversion from standard `VARBINARY` to native state is safe only after validating that the state originated from DECIMAL64 semantics and has no nonzero overflow. It is not required for the first version.

## Eligibility and fallback

Use native state only if all of the following are true:

- function is decimal SUM;
- original raw input is `DECIMAL64`;
- aggregation step is partial or an internal intermediate merge;
- immediate consumers preserve or consume the encoding;
- no remote exchange, spill boundary, CPU conversion, or unsupported operator intervenes.

If eligibility cannot be proven, produce the existing standard `VARBINARY` state. Correct fallback is more important than maximizing coverage.

A mixed batch sequence must not silently combine native and serialized representations. Either materialize native batches before mixing or keep the entire edge on one representation.

## Initial implementation stages

### Stage 1: Native state plumbing

- Add physical-encoding metadata to `CudfVector`.
- Preserve it through the required table and local-exchange operations.
- Add strict physical/logical validation.
- Add explicit native-to-VARBINARY materialization.

### Stage 2: Decimal SUM producer and consumer

- Emit nullable `DECIMAL128` from eligible partial group-by.
- Consume it directly in final group-by.
- Keep incremental buffered state native.
- Preserve the existing path as fallback.

### Stage 3: Plan-path eligibility

- Enable native output only for a fully compatible GPU edge.
- Materialize before unsupported boundaries.
- Add runtime counters for native production, consumption, and materialization.

### Stage 4: Validation and profiling

Test:

- positive, negative, and zero sums;
- cancellation to zero;
- null and all-null inputs;
- mixed null/non-null inputs;
- maximum and minimum DECIMAL64 values;
- final DECIMAL(38) range checks;
- multiple partial batches and drivers;
- hash local exchange;
- forced materialization followed by CPU final aggregation;
- unsupported-path fallback;
- metadata preservation through split, concatenate, and partition.

Profile Q18 with the same two-driver managed-memory configuration and compare:

- partial output retained bytes;
- final `addInput` peak by iteration;
- STRING payload and offset allocations;
- decode/encode temporary allocations;
- runtime and Unified Memory migration behavior.

## Expected Q18 impact

The native state uses approximately:

```text
DECIMAL128 data: 16 bytes per group
validity:         1 bit per group
```

It replaces a 32-byte packed payload plus STRING offsets and avoids repeated decode/encode materialization. At 1.5 billion groups, eliminating 16 payload bytes alone represents about 24 GB decimal (22.35 GiB) per full materialized copy, with additional savings from removed offsets and temporary columns.

The exact peak reduction may be larger because the current incremental final group-by can retain incoming, buffered, concatenated, decoded, and output state concurrently. Hash-table and grouping-key allocations remain and must be measured separately.

## Non-goals for the first version

- native state for DECIMAL128 raw input;
- native decimal AVG state;
- changing the planner-visible intermediate type;
- remote exchange support for native state;
- generic native state support for all aggregates;
- replacing the existing CPU-compatible serialized format.
