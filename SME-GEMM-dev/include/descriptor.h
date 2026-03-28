#pragma once
#include <vector>
#include "macro.h"
class VectorElementDescriptor
{
public:
    // use SME layout as standard layout. T for A or N for B.
    int batch;
    int column;
    int column_step;
    // only for scalar. for vector, dup_num must be 1.
    int dup_num;
    bool is_scalar() const { return column_step == 1; }
    VectorElementDescriptor(int batch, int column, int column_step, int dup_num = 1)
    : batch(batch)
    , column(column)
    , column_step(column_step)
    , dup_num(dup_num)
    {
        MY_ASSERT((column_step > 0) && "column_step must be positive");
        MY_ASSERT((dup_num > 0) && "dup_num must be positive");
        MY_ASSERT((dup_num == 1 || is_scalar()) && "only scalar can have dup_num > 1");
    }
};
class VectorDescriptor
{
public:
    auto begin() { return elements.begin(); }
    auto end() { return elements.end(); }
    auto begin() const { return elements.begin(); }
    auto end() const { return elements.end(); }
    size_t seg_size() const { return elements.size(); }
    size_t element_offest(size_t idx) const
    {
        size_t res = 0;

        for (auto &elem : elements)
        {
            if (idx == 0)
            {
                break;
            }
            res += elem.column_step * elem.dup_num;
            idx--;
        }
        return res;
    }
    auto &operator[](size_t i) { return elements[i]; }

    // private:
    std::vector<VectorElementDescriptor> elements;
};
class TilePrimitiveDescriptor
{
public:
    enum DTYPE
    {
        DTYPE_FP64,
        DTYPE_FP32
    };
    enum OP_TYPE
    {
        SME_MOPA,
        SVE_MLA,
        SME2_MLA,
        SCALAR
    };
    enum TRANS_TYPE
    {
        UNDEF,
        GEMM_NN,
        GEMM_NT,
        GEMM_TN,
        GEMM_TT
    };

    TilePrimitiveDescriptor(OP_TYPE op_type, TRANS_TYPE trans_type, DTYPE dtype, int K)
    : op_type(op_type)
    , trans_type(trans_type)
    , dtype(dtype)
    , K(K)
    {
    }

    OP_TYPE op_type;
    TRANS_TYPE trans_type;
    DTYPE dtype;
    std::vector<VectorDescriptor> vec_a;
    std::vector<VectorDescriptor> vec_b;
    int K;
};