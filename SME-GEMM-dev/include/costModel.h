#pragma once

#include "IR.h"
#include "cost.h"
#include "descriptor.h"
#include "frontend.h"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__linux__) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

using Block = Frontend::Block;

inline int div_up(int a, int b) { return (a + b - 1) / b; }

inline long long div_up_ll(long long a, long long b) { return (a + b - 1) / b; }

inline double overlap_merge(double a, double b, double overlap)
{
    const double clamped = std::clamp(overlap, 0.0, 1.0);
    const double mn = std::min(a, b);
    const double mx = std::max(a, b);
    return mx + (1.0 - clamped) * mn;
}

inline bool host_supports_sme2()
{
#if defined(__linux__) && defined(__aarch64__) && defined(AT_HWCAP2) && defined(HWCAP2_SME2)
    return (getauxval(AT_HWCAP2) & HWCAP2_SME2) != 0;
#else
    return false;
#endif
}

inline bool costmodel_enable_sme2_runtime()
{
    if (!COSTMODEL_ENABLE_SME2)
    {
        return false;
    }

    static int cached = -1;
    if (cached >= 0)
    {
        return cached != 0;
    }

    if (const char *force = std::getenv("SME_GEMM_COSTMODEL_ENABLE_SME2"))
    {
        if (force[0] == '1')
        {
            cached = 1;
            return true;
        }
        if (force[0] == '0')
        {
            cached = 0;
            return false;
        }
    }

    cached = host_supports_sme2() ? 1 : 0;
    return cached != 0;
}

struct InstructionCounts
{
    long long sme_compute = 0;
    long long sme2_compute = 0;
    long long sve_compute = 0;
    long long scalar_compute = 0;
    long long sme_move = 0;
    long long sve_concat = 0;
    long long mem_load = 0;
    long long mem_store = 0;

    InstructionCounts &operator+=(const InstructionCounts &rhs)
    {
        sme_compute += rhs.sme_compute;
        sme2_compute += rhs.sme2_compute;
        sve_compute += rhs.sve_compute;
        scalar_compute += rhs.scalar_compute;
        sme_move += rhs.sme_move;
        sve_concat += rhs.sve_concat;
        mem_load += rhs.mem_load;
        mem_store += rhs.mem_store;
        return *this;
    }

    double total_cost() const
    {
        const long long hidden_sve_by_mopa =
            std::min(sve_compute, sme_compute * static_cast<long long>(SVE_INSNS_COVERED_PER_MOPA));
        const long long hidden_move_by_mopa =
            std::min(sme_move, sme_compute * static_cast<long long>(SME_MOVE_INSNS_COVERED_PER_MOPA));
        const long long hidden_load_by_mopa =
            std::min(mem_load, sme_compute * static_cast<long long>(SVE_LOAD_INSNS_COVERED_PER_MOPA));
        const long long hidden_store_by_mopa =
            std::min(mem_store, sme_compute * static_cast<long long>(SVE_STORE_INSNS_COVERED_PER_MOPA));
        const long long hidden_concat_by_mopa =
            std::min(sve_concat, sme_compute * static_cast<long long>(SVE_CONCAT_INSNS_COVERED_PER_MOPA));

        const long long remain_sve_compute = sve_compute - hidden_sve_by_mopa;
        const long long remain_sme_move = sme_move - hidden_move_by_mopa;
        const long long remain_mem_load = mem_load - hidden_load_by_mopa;
        const long long remain_mem_store = mem_store - hidden_store_by_mopa;
        const long long remain_sve_concat = sve_concat - hidden_concat_by_mopa;

        long long effective_mopa_slots = 0;
        if (sme_compute > 0)
        {
            const long long head = std::min<long long>(sme_compute, MOPA_CHAIN_HEAD);
            const long long tail = std::max<long long>(0, sme_compute - head);
            const long long hidden_mopa = tail / std::max(1, MOPA_CHAIN_WIDTH);
            effective_mopa_slots = head + (tail - hidden_mopa);
        }

        const double sve_compute_cost = remain_sve_compute * SVE_FMLA_COST;
        const double sme_compute_cost = effective_mopa_slots * SME_MOPA_COST;
        const double sme2_compute_cost = sme2_compute * SME2_FMLA_COST;
        const double scalar_compute_cost = scalar_compute * SCALAR_COMPUTE_COST;
        const double sme_move_cost = remain_sme_move * SME_MOVA_COST;
        const double mem_cost =
            remain_mem_load * SVE_LOAD_COST + remain_mem_store * SVE_STORE_COST + remain_sve_concat * SVE_CONCAT_COST;

        const double compute_side =
            sme_compute_cost + sme2_compute_cost + sme_move_cost + sve_compute_cost + scalar_compute_cost;

        return overlap_merge(compute_side, mem_cost, MEM_COMPUTE_OVERLAP);
    }
};

enum class FusionStrategy
{
    UNFUSED,
    FUSED
};

class KernelEvaluator
{
public:
    explicit KernelEvaluator(int K)
    : K(K)
    {
    }

    virtual ~KernelEvaluator() = default;
    virtual std::string getKernelName() const = 0;

    InstructionCounts evaluate(const std::vector<Block> &blocks, FusionStrategy fusion_strategy) const
    {
        if (blocks.empty())
        {
            return {};
        }
        if (fusion_strategy == FusionStrategy::FUSED)
        {
            return calculateFusedBlocks(blocks);
        }

        InstructionCounts total_counts;
        for (const auto &block : blocks)
        {
            total_counts += calculateSingleBlock(block);
        }
        return total_counts;
    }

protected:
    int K;
    virtual InstructionCounts calculateSingleBlock(const Block &block) const = 0;
    virtual InstructionCounts calculateFusedBlocks(const std::vector<Block> &blocks) const = 0;
};

class SmeMopaEvaluator : public KernelEvaluator
{
public:
    explicit SmeMopaEvaluator(int K)
    : KernelEvaluator(K)
    {
    }

    std::string getKernelName() const override { return "SME_MOPA"; }

    InstructionCounts calculateSingleBlock(const Block &block) const override
    {
        return calculateMopa(
            block.trans_type, block.dtype, 1, block.row_step, block.column_step, block.row_step, block.column_step);
    }

    InstructionCounts calculateFusedBlocks(const std::vector<Block> &blocks) const override
    {
        if (blocks.empty())
        {
            return {};
        }

        int row_sum = 0;
        int col_sum = 0;
        for (const auto &b : blocks)
        {
            row_sum += b.row_step;
            col_sum += b.column_step;
        }

        return calculateMopa(
            blocks[0].trans_type, blocks[0].dtype, static_cast<int>(blocks.size()), row_sum, col_sum, row_sum, col_sum);
    }

private:
    InstructionCounts calculateMopa(TilePrimitiveDescriptor::TRANS_TYPE trans_type,
                                    TilePrimitiveDescriptor::DTYPE dtype,
                                    int num_elems,
                                    int row_sum,
                                    int col_sum,
                                    int store_rows,
                                    int /*store_cols*/) const
    {
        InstructionCounts counts;

        const bool transA =
            (trans_type == TilePrimitiveDescriptor::GEMM_NN || trans_type == TilePrimitiveDescriptor::GEMM_NT);
        const bool transB =
            (trans_type == TilePrimitiveDescriptor::GEMM_NT || trans_type == TilePrimitiveDescriptor::GEMM_TT);

        const int vl = IR::svl(dtype);
        const int k_unroll = (transA || transB) ? vl : 1;
        const int k_iters = div_up(K, k_unroll);

        counts.sme_compute = 1LL * k_iters * k_unroll;

        if (transA)
        {
            counts.mem_load += 1LL * k_iters * row_sum;
            counts.sme_move += 1LL * k_iters * (row_sum + k_unroll);
        }
        else
        {
            counts.mem_load += 1LL * k_iters * k_unroll * num_elems;
            counts.sve_concat += 1LL * k_iters * k_unroll * std::max(0, num_elems - 1);
        }

        if (transB)
        {
            counts.mem_load += 1LL * k_iters * col_sum;
            counts.sme_move += 1LL * k_iters * (col_sum + k_unroll);
        }
        else
        {
            counts.mem_load += 1LL * k_iters * k_unroll * num_elems;
            counts.sve_concat += 1LL * k_iters * k_unroll * std::max(0, num_elems - 1);
        }

        counts.mem_store += store_rows;
        counts.sme_move += store_rows;

        return counts;
    }
};

class SveEvaluator : public KernelEvaluator
{
public:
    explicit SveEvaluator(int K)
    : KernelEvaluator(K)
    {
    }

    std::string getKernelName() const override { return "SVE_ONLY"; }

    InstructionCounts calculateSingleBlock(const Block &block) const override
    {
        InstructionCounts counts;

        if (block.trans_type == TilePrimitiveDescriptor::GEMM_NT)
        {
            throw std::runtime_error("GEMM_NT is not supported in SVE_ONLY strategy");
        }

        if (block.trans_type == TilePrimitiveDescriptor::GEMM_NN ||
            block.trans_type == TilePrimitiveDescriptor::GEMM_TN)
        {
            counts.sve_compute = 1LL * block.row_step * K;
            counts.mem_load = 1LL * 2 * block.row_step * K;
            counts.mem_store = block.row_step;
            return counts;
        }

        counts.sve_compute = 1LL * block.column_step * K;
        counts.mem_load = 1LL * 2 * block.column_step * K;
        counts.mem_store = block.row_step;
        counts.sme_move = block.column_step + block.row_step;
        return counts;
    }

    InstructionCounts calculateFusedBlocks(const std::vector<Block> &blocks) const override
    {
        if (blocks.empty())
        {
            return {};
        }

        const Block &ref = blocks[0];
        const int n = static_cast<int>(blocks.size());
        InstructionCounts counts;

        if (ref.trans_type == TilePrimitiveDescriptor::GEMM_NT)
        {
            throw std::runtime_error("GEMM_NT is not supported in SVE_ONLY strategy");
        }

        if (ref.trans_type == TilePrimitiveDescriptor::GEMM_NN || ref.trans_type == TilePrimitiveDescriptor::GEMM_TN)
        {
            counts.sve_compute = 1LL * ref.row_step * K;
            counts.mem_load = 1LL * 2 * ref.row_step * n * K;
            counts.mem_store = 1LL * ref.row_step * n;
            counts.sve_concat = 1LL * 2 * ref.row_step * std::max(0, n - 1) * K;
            return counts;
        }

        counts.sve_compute = 1LL * ref.column_step * K;
        counts.mem_load = 1LL * 2 * ref.column_step * n * K;
        counts.mem_store = 1LL * ref.row_step * n;
        counts.sve_concat = 1LL * 2 * ref.column_step * std::max(0, n - 1) * K;
        counts.sme_move = ref.column_step + ref.row_step;
        return counts;
    }
};

class Sme2FmlaEvaluator : public KernelEvaluator
{
public:
    explicit Sme2FmlaEvaluator(int K)
    : KernelEvaluator(K)
    {
    }

    std::string getKernelName() const override { return "SME2_FMLA"; }

    InstructionCounts calculateSingleBlock(const Block &block) const override
    {
        if (!costmodel_enable_sme2_runtime())
        {
            InstructionCounts inf;
            inf.scalar_compute = std::numeric_limits<long long>::max() / 4;
            return inf;
        }

        InstructionCounts counts;

        const int rows = block.row_step;
        const int cols = block.column_step;
        const int short_step = std::min(rows, cols);
        const int long_step = std::max(rows, cols);
        const bool short_is_row = rows <= cols;
        const int vl = IR::svl(block.dtype);

        long long mla_groups_per_k = 0;
        long long tuple_create_per_k = 0;

        for (int start = 0; start < short_step;)
        {
            const int chunk = std::min(vl, short_step - start);
            const int pack = (chunk % 4 == 0) ? 4 : 2;
            const int groups = div_up(chunk, pack);
            mla_groups_per_k += groups;
            tuple_create_per_k += 2LL * groups; // create tuple for both A and B
            start += chunk;
        }

        counts.sme2_compute = mla_groups_per_k * K;
        counts.sve_concat = tuple_create_per_k * K;

        // IR::buildMlaSME uses one scalar load and one vector load for each vec_id.
        counts.mem_load = 1LL * short_step * (long_step + 1) * K;

        // Result extraction is through one ZA read per vec_id, then store differs by layout.
        counts.sme_move = short_step;
        counts.mem_store = short_is_row ? rows : (1LL * rows * cols);

        return counts;
    }

    InstructionCounts calculateFusedBlocks(const std::vector<Block> &blocks) const override
    {
        if (!costmodel_enable_sme2_runtime())
        {
            InstructionCounts inf;
            inf.scalar_compute = std::numeric_limits<long long>::max() / 4;
            return inf;
        }

        InstructionCounts counts;
        for (const auto &block : blocks)
        {
            counts += calculateSingleBlock(block);
        }
        return counts;
    }
};

class ScalarEvaluator : public KernelEvaluator
{
public:
    explicit ScalarEvaluator(int K)
    : KernelEvaluator(K)
    {
    }

    std::string getKernelName() const override { return "SCALAR"; }

    InstructionCounts calculateSingleBlock(const Block &block) const override
    {
        InstructionCounts counts;
        counts.mem_load = 1LL * (block.row_step + block.column_step) * K;
        counts.mem_store = 1LL * block.row_step * block.column_step;
        counts.scalar_compute = 1LL * block.row_step * block.column_step * K;
        return counts;
    }

    InstructionCounts calculateFusedBlocks(const std::vector<Block> &blocks) const override
    {
        InstructionCounts counts;
        for (const auto &block : blocks)
        {
            counts += calculateSingleBlock(block);
        }
        return counts;
    }
};
