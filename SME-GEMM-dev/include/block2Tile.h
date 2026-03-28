#pragma once
#include "descriptor.h"
#include "frontend.h"
#include "macro.h"
namespace Frontend
{
class TileGenerator
{
private:
    void build_mopa(const Frontend &frontend,
                    const std::vector<Block> &blocks,
                    std::vector<TilePrimitiveDescriptor> &descriptors)
    {
        for (auto &block : blocks)
        {
            TilePrimitiveDescriptor desc(
                TilePrimitiveDescriptor::SME_MOPA, frontend.trans_type, frontend.dtype, frontend.K);

            VectorDescriptor vec_a;
            vec_a.elements.push_back({block.batch, block.row, block.row_step});
            desc.vec_a.push_back(vec_a);

            VectorDescriptor vec_b;
            vec_b.elements.push_back({block.batch, block.column, block.column_step});
            desc.vec_b.push_back(vec_b);

            descriptors.push_back(desc);

            // 清空以便下次使用
            desc.vec_a.clear();
            desc.vec_b.clear();
        }
    };

    // blocks must fit in a za tile
    void build_merge_mopa(const Frontend &frontend,
                          const std::vector<Block> &blocks,
                          std::vector<TilePrimitiveDescriptor> &descriptors);
    void build_inter_lane_sme_mla(const Frontend &frontend,
                                  const std::vector<Block> &blocks,
                                  std::vector<TilePrimitiveDescriptor> &descriptors);

    void build_scalar(const Frontend &frontend,
                      const std::vector<Block> &blocks,
                      std::vector<TilePrimitiveDescriptor> &descriptors);
    void build_mla(const Frontend &frontend,
                   const std::vector<Block> &blocks,
                   std::vector<TilePrimitiveDescriptor> &descriptors,
                   TilePrimitiveDescriptor::OP_TYPE op_type = TilePrimitiveDescriptor::SVE_MLA);
    void build_mla(const Frontend &frontend,
                   const Block &block,
                   std::vector<TilePrimitiveDescriptor> &descriptors,
                   TilePrimitiveDescriptor::OP_TYPE op_type);
    void build_sme2_mla(const Frontend &frontend,
                        const std::vector<Block> &blocks,
                        std::vector<TilePrimitiveDescriptor> &descriptors);
    void
    build_sme2_mla(const Frontend &frontend, const Block &block, std::vector<TilePrimitiveDescriptor> &descriptors);
    void build_mopa(const Frontend &frontend, const Block &block, std::vector<TilePrimitiveDescriptor> &descriptors);

    int search_batch_step(Frontend &frontend);

public:
    std::pair<std::vector<TilePrimitiveDescriptor>, int> build_strategy_costmodel(Frontend &frontend);
    std::pair<std::vector<TilePrimitiveDescriptor>, int> build_strategy_scalar(Frontend &frontend);
    std::pair<std::vector<TilePrimitiveDescriptor>, int>
    build_strategy_mla(Frontend &frontend, TilePrimitiveDescriptor::OP_TYPE op_type = TilePrimitiveDescriptor::SVE_MLA);
    std::pair<std::vector<TilePrimitiveDescriptor>, int>
    build_strategy_fuse_mla(Frontend &frontend,
                            TilePrimitiveDescriptor::OP_TYPE op_type = TilePrimitiveDescriptor::SVE_MLA);
    std::pair<std::vector<TilePrimitiveDescriptor>, int> build_strategy_mopa(Frontend &frontend);
    std::pair<std::vector<TilePrimitiveDescriptor>, int> build_strategy_fuse_mopa(Frontend &frontend);
    std::pair<std::vector<TilePrimitiveDescriptor>, int> build_strategy1(Frontend &frontend);
};
} // namespace Frontend
