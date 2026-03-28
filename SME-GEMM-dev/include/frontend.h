#pragma once
#include <vector>
#include "IR.h"
#include "descriptor.h"
namespace Frontend
{
struct Block
{
    int batch;
    int row;
    int column;
    int row_step;
    int column_step;
    TilePrimitiveDescriptor::DTYPE dtype;
    TilePrimitiveDescriptor::TRANS_TYPE trans_type;
};
class BlockGroup
{
    std::vector<Block> tiles;

public:
    bool fit(const Block &tile, int vl)
    {
        int row_sum = 0;
        int col_sum = 0;
        for (const auto &t : tiles)
        {
            row_sum += t.row;
            col_sum += t.column;
        }
        if (row_sum + tile.row < vl && col_sum + tile.column < vl)
        {
            return true;
        }
        return false;
    }
    void insert(const Block &tile) { tiles.push_back(tile); }
    BlockGroup() = default;
    BlockGroup(const Block &tile) { tiles.push_back(tile); }
};
class Frontend
{
    int M;
    int N;
    int K;
    int batch;
    TilePrimitiveDescriptor::DTYPE dtype;
    TilePrimitiveDescriptor::TRANS_TYPE trans_type;
    friend class TileGenerator;

public:
    std::vector<Block> full_blocks;
    std::vector<Block> right_partial_blocks;
    std::vector<Block> bottom_partial_blocks;
    std::vector<Block> noalign_blocks;
    Frontend(int M,
             int N,
             int K,
             int batch,
             TilePrimitiveDescriptor::DTYPE dtype,
             TilePrimitiveDescriptor::TRANS_TYPE trans_type)
    : M(M)
    , N(N)
    , K(K)
    , batch(batch)
    , dtype(dtype)
    , trans_type(trans_type)
    {
    }
    void build();
    bool is_b_transpose() const
    {
        return trans_type == TilePrimitiveDescriptor::GEMM_NT || trans_type == TilePrimitiveDescriptor::GEMM_TT;
    }
    bool is_a_transpose() const
    {
        return trans_type == TilePrimitiveDescriptor::GEMM_TN || trans_type == TilePrimitiveDescriptor::GEMM_TT;
    }
};

} // namespace Frontend