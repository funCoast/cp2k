#include "block2Tile.h"
#include "costModel.h"
#include "descriptor.h"
#include "frontend.h"
#include "IR.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>

std::pair<std::vector<TilePrimitiveDescriptor>, int> Frontend::TileGenerator::build_strategy_mopa(Frontend &frontend)
{
    std::vector<TilePrimitiveDescriptor> descriptors;

    build_mopa(frontend, frontend.full_blocks, descriptors);
    build_mopa(frontend, frontend.right_partial_blocks, descriptors);
    build_mopa(frontend, frontend.bottom_partial_blocks, descriptors);
    build_mopa(frontend, frontend.noalign_blocks, descriptors);

    return {descriptors, 1};
}

std::pair<std::vector<TilePrimitiveDescriptor>, int> Frontend::TileGenerator::build_strategy_scalar(Frontend &frontend)
{
    std::vector<TilePrimitiveDescriptor> descriptors;
    build_scalar(frontend, frontend.full_blocks, descriptors);
    build_scalar(frontend, frontend.right_partial_blocks, descriptors);
    build_scalar(frontend, frontend.bottom_partial_blocks, descriptors);
    build_scalar(frontend, frontend.noalign_blocks, descriptors);
    return {descriptors, 1};
}

void Frontend::TileGenerator::build_merge_mopa(const Frontend &frontend,
                                               const std::vector<Block> &blocks,
                                               std::vector<TilePrimitiveDescriptor> &descriptors)
{
    TilePrimitiveDescriptor desc(TilePrimitiveDescriptor::SME_MOPA, frontend.trans_type, frontend.dtype, frontend.K);
    VectorDescriptor vec_a;
    VectorDescriptor vec_b;
    for (auto &block : blocks)
    {
        vec_a.elements.push_back({block.batch, block.row, block.row_step});
        vec_b.elements.push_back({block.batch, block.column, block.column_step});
    }
    MY_ASSERT(vec_a.element_offest(255) <= IR::svl(desc.dtype));
    desc.vec_a.push_back(vec_a);
    desc.vec_b.push_back(vec_b);
    descriptors.push_back(desc);
};
void Frontend::TileGenerator::build_inter_lane_sme_mla(const Frontend &frontend,
                                                       const std::vector<Block> &blocks,
                                                       std::vector<TilePrimitiveDescriptor> &descriptors)
{
    TilePrimitiveDescriptor desc(TilePrimitiveDescriptor::SME2_MLA, frontend.trans_type, frontend.dtype, frontend.K);
    VectorDescriptor vec_a;
    VectorDescriptor vec_b;

    int count = 0;
    auto submit = [&]() {
        MY_ASSERT(vec_a.element_offest(255) <= IR::svl(desc.dtype));
        MY_ASSERT(vec_b.element_offest(255) <= IR::svl(desc.dtype));
        descriptors.push_back(desc);
        desc.vec_a.clear();
        desc.vec_b.clear();
        count = 0;
    };
    for (int i = 0; i < blocks.size(); ++i)
    {
        auto get_short = [](const Block &block) {
            int short_step = std::min(block.row_step, block.column_step);
            bool is_A = (block.row_step < block.column_step);
            return std::tuple{short_step, is_A};
        };
        auto [short_step, short_is_A] = get_short(blocks[i]);

        auto add_vec = [&](int a_pos, int a_step, int a_dup, int b_pos, int b_step, int b_dup) {
            vec_a.elements.push_back({blocks[i].batch, a_pos, a_step, a_dup});
            vec_b.elements.push_back({blocks[i].batch, b_pos, b_step, b_dup});
            desc.vec_a.push_back(vec_a);
            desc.vec_b.push_back(vec_b);
            vec_a.elements.clear();
            vec_b.elements.clear();
            count++;
        };
        for (int p = 0; p < short_step; ++p)
        {
            int a_pos = short_is_A ? blocks[i].row + p : blocks[i].row;
            int b_pos = short_is_A ? blocks[i].column : blocks[i].column + p;
            int a_step = short_is_A ? 1 : blocks[i].row_step;
            int b_step = short_is_A ? blocks[i].column_step : 1;
            int a_dup = short_is_A ? blocks[i].column_step : 1;
            int b_dup = short_is_A ? 1 : blocks[i].row_step;
            add_vec(a_pos, a_step, a_dup, b_pos, b_step, b_dup);
            if (count == IR::svl(desc.dtype))
            {
                submit();
            }
        }
    }
    if (count > 0)
    {
        submit();
    }
};
std::pair<std::vector<TilePrimitiveDescriptor>, int> Frontend::TileGenerator::build_strategy1(Frontend &frontend)
{
    std::vector<TilePrimitiveDescriptor> descriptors;

    build_mopa(frontend, frontend.full_blocks, descriptors);
    build_inter_lane_sme_mla(frontend, frontend.right_partial_blocks, descriptors);
    build_inter_lane_sme_mla(frontend, frontend.bottom_partial_blocks, descriptors);
    build_mopa(frontend, frontend.noalign_blocks, descriptors);

    return {descriptors, 1};
}

std::pair<std::vector<TilePrimitiveDescriptor>, int>
Frontend::TileGenerator::build_strategy_costmodel(Frontend &frontend)
{
    std::vector<TilePrimitiveDescriptor> descriptors;
    SmeMopaEvaluator sme_eval(frontend.K);
    Sme2FmlaEvaluator sme2_eval(frontend.K);
    SveEvaluator sve_eval(frontend.K);
    ScalarEvaluator scalar_eval(frontend.K);
    auto batch_step = search_batch_step(frontend);

    auto sve_available = [&](const Block &b) { return b.trans_type != TilePrimitiveDescriptor::GEMM_NT; };

    auto build_best_single = [&](Block block, bool allow_sme2) {
        const double mopa_cost = sme_eval.calculateSingleBlock(block).total_cost();
        const double scalar_cost = scalar_eval.calculateSingleBlock(block).total_cost();

        double best_cost = mopa_cost;
        enum Choice
        {
            MOPA,
            SME2,
            SVE,
            SCALAR
        } best_choice = MOPA;

        if (allow_sme2)
        {
            const double sme2_cost = sme2_eval.calculateSingleBlock(block).total_cost();
            if (sme2_cost < best_cost)
            {
                best_cost = sme2_cost;
                best_choice = SME2;
            }
        }

        if (sve_available(block))
        {
            const double sve_cost = sve_eval.calculateSingleBlock(block).total_cost();
            if (sve_cost < best_cost)
            {
                best_cost = sve_cost;
                best_choice = SVE;
            }
        }

        if (scalar_cost < best_cost)
        {
            best_choice = SCALAR;
        }

        if (best_choice == SME2)
        {
            build_inter_lane_sme_mla(frontend, std::vector<Block>{block}, descriptors);
        }
        else if (best_choice == SVE)
        {
            build_mla(frontend, block, descriptors, TilePrimitiveDescriptor::SVE_MLA);
        }
        else if (best_choice == SCALAR)
        {
            build_scalar(frontend, std::vector<Block>{block}, descriptors);
        }
        else
        {
            build_mopa(frontend, block, descriptors);
        }
    };

    // fit well with za
    if (!frontend.full_blocks.empty())
    {
        for (int i = 0; i < batch_step; ++i)
        {
            for (auto &block : frontend.full_blocks)
            {
                block.batch = i;
                build_best_single(block, false);
            }
        }
    }

    if (!frontend.noalign_blocks.empty())
    {
        bool enable_merge_mopa = false;
        int vl = IR::svl(frontend.dtype);
        if (frontend.noalign_blocks[0].row_step * batch_step <= vl &&
            frontend.noalign_blocks[0].column_step * batch_step <= vl)
        {
            enable_merge_mopa = true;
        }
        auto &block = frontend.noalign_blocks[0];
        std::vector<Block> merged_blocks;
        for (int i = 0; i < batch_step; ++i)
        {
            block.batch = i;
            merged_blocks.push_back(block);
        }

        double unfused_mopa_cost = sme_eval.evaluate(merged_blocks, FusionStrategy::UNFUSED).total_cost();
        double merge_mopa_cost = enable_merge_mopa ? sme_eval.calculateFusedBlocks(merged_blocks).total_cost()
                                                   : std::numeric_limits<double>::max();

        double fused_sve_cost = std::numeric_limits<double>::max();
        double unfused_sve_cost = std::numeric_limits<double>::max();
        if (sve_available(block))
        {
            fused_sve_cost = sve_eval.calculateFusedBlocks(merged_blocks).total_cost();
            unfused_sve_cost = sve_eval.evaluate(merged_blocks, FusionStrategy::UNFUSED).total_cost();
        }

        double scalar_cost = scalar_eval.evaluate(merged_blocks, FusionStrategy::UNFUSED).total_cost();

        double min_cost = std::min({unfused_mopa_cost, merge_mopa_cost, unfused_sve_cost, fused_sve_cost, scalar_cost});
        if (fused_sve_cost == min_cost)
        {
            build_mla(frontend, merged_blocks, descriptors, TilePrimitiveDescriptor::SVE_MLA);
        }
        else if (merge_mopa_cost == min_cost)
        {
            build_merge_mopa(frontend, merged_blocks, descriptors);
        }
        else if (unfused_sve_cost == min_cost)
        {
            for (auto b : merged_blocks)
            {
                build_mla(frontend, b, descriptors, TilePrimitiveDescriptor::SVE_MLA);
            }
        }
        else if (scalar_cost == min_cost)
        {
            build_scalar(frontend, merged_blocks, descriptors);
        }
        else
        {
            for (auto b : merged_blocks)
            {
                build_mopa(frontend, b, descriptors);
            }
        }
    }

    {
        auto &right_blocks = frontend.right_partial_blocks;
        auto &bottom_blocks = frontend.bottom_partial_blocks;

        if (!right_blocks.empty())
        {
            for (int i = 0; i < batch_step; ++i)
            {
                for (auto b : right_blocks)
                {
                    b.batch = i;
                    build_best_single(b, true);
                }
            }
        }
        if (!bottom_blocks.empty())
        {
            for (int i = 0; i < batch_step; ++i)
            {
                for (auto b : bottom_blocks)
                {
                    b.batch = i;
                    build_best_single(b, true);
                }
            }
        }
    }

    return {descriptors, batch_step};
}

void Frontend::TileGenerator::build_scalar(const Frontend &frontend,
                                           const std::vector<Block> &blocks,
                                           std::vector<TilePrimitiveDescriptor> &descriptors)
{
    TilePrimitiveDescriptor desc(TilePrimitiveDescriptor::SCALAR, frontend.trans_type, frontend.dtype, frontend.K);
    for (const auto &block : blocks)
    {
        VectorDescriptor vec_a;
        VectorDescriptor vec_b;
        vec_a.elements.push_back({block.batch, block.row, block.row_step});
        vec_b.elements.push_back({block.batch, block.column, block.column_step});
        desc.vec_a.push_back(vec_a);
        desc.vec_b.push_back(vec_b);
        descriptors.push_back(desc);
        desc.vec_a.clear();
        desc.vec_b.clear();
    }
}
void Frontend::TileGenerator::build_mla(const Frontend &frontend,
                                        const std::vector<Block> &blocks,
                                        std::vector<TilePrimitiveDescriptor> &descriptors,
                                        TilePrimitiveDescriptor::OP_TYPE op_type)
{
    if (blocks.empty())
    {
        return;
    }
    TilePrimitiveDescriptor desc(op_type, frontend.trans_type, frontend.dtype, frontend.K);
    if (frontend.trans_type == TilePrimitiveDescriptor::GEMM_TN ||
        frontend.trans_type == TilePrimitiveDescriptor::GEMM_NN)
    {
        for (int i = 0; i < blocks[0].row_step; ++i)
        {
            VectorDescriptor vec_a;
            VectorDescriptor vec_b;
            for (const auto &block : blocks)
            {
                vec_a.elements.push_back({block.batch, block.row + i, 1, block.column_step});
                vec_b.elements.push_back({block.batch, block.column, block.column_step, 1});
            }
            desc.vec_a.push_back(vec_a);
            desc.vec_b.push_back(vec_b);
        }
    }
    else if (frontend.trans_type == TilePrimitiveDescriptor::GEMM_TT)
    {
        for (int i = 0; i < blocks[0].column_step; ++i)
        {
            VectorDescriptor vec_a;
            VectorDescriptor vec_b;
            for (const auto &block : blocks)
            {
                vec_a.elements.push_back({block.batch, block.row, block.row_step, 1});
                vec_b.elements.push_back({
                    block.batch,
                    block.column + i,
                    1,
                    block.row_step,
                });
            }
            desc.vec_a.push_back(vec_a);
            desc.vec_b.push_back(vec_b);
        }
    }
    else if (frontend.trans_type == TilePrimitiveDescriptor::GEMM_NT)
    {
        throw std::runtime_error("Unsupported transpose type NT for SVE strategy");
    }
    else
    {
        throw std::runtime_error("Unknown transpose type for SVE strategy");
    }
    descriptors.push_back(desc);
}
void Frontend::TileGenerator::build_mla(const Frontend &frontend,
                                        const Block &block,
                                        std::vector<TilePrimitiveDescriptor> &descriptors,
                                        TilePrimitiveDescriptor::OP_TYPE op_type)
{
    TilePrimitiveDescriptor desc(op_type, frontend.trans_type, frontend.dtype, frontend.K);
    if (frontend.trans_type == TilePrimitiveDescriptor::GEMM_TN ||
        frontend.trans_type == TilePrimitiveDescriptor::GEMM_NN)
    {
        for (int i = 0; i < block.row_step; ++i)
        {
            VectorDescriptor vec_a;
            vec_a.elements.push_back({block.batch, block.row + i, 1, block.column_step});
            desc.vec_a.push_back(vec_a);
            VectorDescriptor vec_b;
            vec_b.elements.push_back({block.batch, block.column, block.column_step, 1});
            desc.vec_b.push_back(vec_b);
        }
    }
    else if (frontend.trans_type == TilePrimitiveDescriptor::GEMM_TT)
    {
        for (int i = 0; i < block.column_step; ++i)
        {
            VectorDescriptor vec_a;
            vec_a.elements.push_back({block.batch, block.row, block.row_step, 1});
            desc.vec_a.push_back(vec_a);
            VectorDescriptor vec_b;
            vec_b.elements.push_back({block.batch, block.column + i, 1, block.row_step});
            desc.vec_b.push_back(vec_b);
        }
    }
    else
    {
        throw std::runtime_error("Unsupported transpose type for SVE strategy");
    }
    descriptors.push_back(desc);
}
void Frontend::TileGenerator::build_mopa(const Frontend &frontend,
                                         const Block &block,
                                         std::vector<TilePrimitiveDescriptor> &descriptors)
{
    TilePrimitiveDescriptor desc(TilePrimitiveDescriptor::SME_MOPA, frontend.trans_type, frontend.dtype, frontend.K);

    VectorDescriptor vec_a;
    vec_a.elements.push_back({block.batch, block.row, block.row_step});
    desc.vec_a.push_back(vec_a);

    VectorDescriptor vec_b;
    vec_b.elements.push_back({block.batch, block.column, block.column_step});
    desc.vec_b.push_back(vec_b);

    descriptors.push_back(desc);
};
int Frontend::TileGenerator::search_batch_step(Frontend &frontend)
{
    int step = 1;
    if (frontend.noalign_blocks.empty() && frontend.right_partial_blocks.empty() &&
        frontend.bottom_partial_blocks.empty())
    {
        step = 1;
    }
    else if (!frontend.noalign_blocks.empty())
    {
        step = IR::svl(frontend.dtype) /
               std::max(frontend.noalign_blocks[0].row_step, frontend.noalign_blocks[0].column_step);
    }
    else if (!frontend.right_partial_blocks.empty())
    {
        step = std::accumulate(
            frontend.right_partial_blocks.begin(),
            frontend.right_partial_blocks.end(),
            size_t(1),
            [&](size_t acc, const Block &block) { return std::max(acc, IR::svl(frontend.dtype) / block.column_step); });
    }
    else if (!frontend.bottom_partial_blocks.empty())
    {
        step = std::accumulate(
            frontend.bottom_partial_blocks.begin(),
            frontend.bottom_partial_blocks.end(),
            size_t(1),
            [&](size_t acc, const Block &block) { return std::max(acc, IR::svl(frontend.dtype) / block.row_step); });
    }
    else
    {
        throw std::runtime_error("Unexpected case in search_batch_step");
    }
    if (step >= 4)
    {
        step = 4;
    }
    return step;
}
std::pair<std::vector<TilePrimitiveDescriptor>, int>
Frontend::TileGenerator::build_strategy_mla(Frontend &frontend, TilePrimitiveDescriptor::OP_TYPE op_type)
{
    // use build_sve for every block seperately; always set batch_step = 1 and do not search
    std::vector<TilePrimitiveDescriptor> descriptors;
    int batch_per_step = 1;
    for (auto &block : frontend.full_blocks)
    {
        build_mla(frontend, block, descriptors, op_type);
    }
    for (auto &block : frontend.right_partial_blocks)
    {
        build_mla(frontend, block, descriptors, op_type);
    }
    for (auto &block : frontend.bottom_partial_blocks)
    {
        build_mla(frontend, block, descriptors, op_type);
    }
    for (auto &block : frontend.noalign_blocks)
    {
        build_mla(frontend, block, descriptors, op_type);
    }
    return {descriptors, batch_per_step};
}
std::pair<std::vector<TilePrimitiveDescriptor>, int>
Frontend::TileGenerator::build_strategy_fuse_mla(Frontend &frontend, TilePrimitiveDescriptor::OP_TYPE op_type)
{
    std::vector<TilePrimitiveDescriptor> descriptors;
    size_t batch_per_step = 1;
    bool fuse_right =
        (!frontend.right_partial_blocks.empty() || !frontend.noalign_blocks.empty()) && !frontend.is_b_transpose();
    bool fuse_bottom =
        (!frontend.bottom_partial_blocks.empty() || !frontend.noalign_blocks.empty()) && frontend.is_b_transpose();
    if (fuse_right)
    {
        auto lane_width_for_right = [&](const Block &block) { return block.column_step; };
        if (!frontend.right_partial_blocks.empty())
        {
            batch_per_step = std::max(batch_per_step,
                                      IR::svl(frontend.dtype) / lane_width_for_right(frontend.right_partial_blocks[0]));
        }
        else
        {
            batch_per_step =
                std::max(batch_per_step, IR::svl(frontend.dtype) / lane_width_for_right(frontend.noalign_blocks[0]));
        }
    }

    if (fuse_bottom)
    {
        auto lane_width_for_bottom = [&](const Block &block) { return block.row_step; };
        if (!frontend.bottom_partial_blocks.empty())
        {
            batch_per_step = std::max(
                batch_per_step, IR::svl(frontend.dtype) / lane_width_for_bottom(frontend.bottom_partial_blocks[0]));
        }
        else
        {
            batch_per_step =
                std::max(batch_per_step, IR::svl(frontend.dtype) / lane_width_for_bottom(frontend.noalign_blocks[0]));
        }
    }
    for (int i = 0; i < batch_per_step; ++i)
    {
        for (auto &block : frontend.full_blocks)
        {
            block.batch = i;
            build_mla(frontend, block, descriptors, op_type);
        }
    }

    auto build_fused_blocks = [&](const std::vector<Block> &blocks) {
        std::vector<Block> fused_blocks;
        int fused_lanes = 0;
        auto lane_width = [&](const Block &block) {
            // The fused vector side must fit in one SVL. Otherwise element offsets wrap
            // in Predicate::range and different blocks overwrite each other.
            return frontend.is_b_transpose() ? block.row_step : block.column_step;
        };
        for (int i = 0; i < batch_per_step; ++i)
        {
            for (auto block : blocks)
            {
                int width = lane_width(block);
                if (!fused_blocks.empty() && fused_lanes + width > IR::svl(frontend.dtype))
                {
                    build_mla(frontend, fused_blocks, descriptors, op_type);
                    fused_blocks.clear();
                    fused_lanes = 0;
                }
                block.batch = i;
                fused_blocks.push_back(block);
                fused_lanes += width;
            }
        }
        if (!fused_blocks.empty())
        {
            build_mla(frontend, fused_blocks, descriptors, op_type);
        }
    };

    auto build_seperate = [&](const std::vector<Block> &blocks) {
        for (int i = 0; i < batch_per_step; ++i)
        {
            for (auto block : blocks)
            {
                block.batch = i;
                build_mla(frontend, block, descriptors, op_type);
            }
        }
    };

    if (fuse_right)
    {
        build_fused_blocks(frontend.right_partial_blocks);
    }
    else
    {
        build_seperate(frontend.right_partial_blocks);
    }

    if (fuse_bottom)
    {
        build_fused_blocks(frontend.bottom_partial_blocks);
    }
    else
    {
        build_seperate(frontend.bottom_partial_blocks);
    }

    if (fuse_bottom || fuse_right)
    {
        build_fused_blocks(frontend.noalign_blocks);
    }
    else
    {
        build_seperate(frontend.noalign_blocks);
    }
    return {descriptors, batch_per_step};
}

// only noalign blocks can be fused.
std::pair<std::vector<TilePrimitiveDescriptor>, int>
Frontend::TileGenerator::build_strategy_fuse_mopa(Frontend &frontend)
{
    std::vector<TilePrimitiveDescriptor> descriptors;
    int batch_per_step = 1;
    if (!frontend.noalign_blocks.empty())
    {
        batch_per_step = IR::svl(frontend.dtype) /
                         std::max(frontend.noalign_blocks[0].column_step, frontend.noalign_blocks[0].row_step);
        if (batch_per_step >= 4)
        {
            batch_per_step = 4;
        }
    }
    auto build_fused_blocks = [&](const std::vector<Block> &blocks) {
        std::vector<Block> fused_blocks;
        for (int i = 0; i < batch_per_step; ++i)
        {
            for (auto block : blocks)
            {
                block.batch = i;
                fused_blocks.push_back(block);
            }
        }
        build_merge_mopa(frontend, fused_blocks, descriptors);
    };
    auto build_seperate = [&](const std::vector<Block> &blocks) {
        for (int i = 0; i < batch_per_step; ++i)
        {
            for (auto block : blocks)
            {
                block.batch = i;
                build_mopa(frontend, block, descriptors);
            }
        }
    };
    if (batch_per_step > 1)
    {
        build_fused_blocks(frontend.noalign_blocks);
    }
    else
    {
        build_seperate(frontend.noalign_blocks);
    }
    build_seperate(frontend.full_blocks);
    build_seperate(frontend.right_partial_blocks);
    build_seperate(frontend.bottom_partial_blocks);
    return {descriptors, batch_per_step};
}
