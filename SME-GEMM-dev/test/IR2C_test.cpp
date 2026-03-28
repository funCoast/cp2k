#include "IR.h"
#include "descriptor.h"
#include <iostream>
#include <chrono>
#include <ratio>
void dump(IR::Function &func, const std::string &res)
{
    FILE *fkernel = fopen("kernel.cpp", "w");
    fprintf(fkernel, "%s", res.c_str());
    fclose(fkernel);
    FILE *ftest = fopen("test.sh", "w");
    fprintf(ftest, "#!/bin/bash\n");
    fprintf(ftest, "../../out/build/llvm/test/IR2C_test\n");
    fprintf(ftest, "set -e\n");
    fprintf(ftest, "make\n");
    fprintf(ftest, "./verify %d %d\n", func.batch() * 4, 32);
    fclose(ftest);
    FILE *fshape = fopen("shape.h", "w");
    fprintf(fshape, "constexpr int64_t M = %d;\n", func.M);
    fprintf(fshape, "constexpr int64_t N = %d;\n", func.N);
    if (func.getTransType() == TilePrimitiveDescriptor::GEMM_NT)
    {
        fprintf(fshape, "#define TRANS_TYPE nt");
    }
    else if (func.getTransType() == TilePrimitiveDescriptor::GEMM_NN)
    {
        fprintf(fshape, "#define TRANS_TYPE nn");
    }
    else if (func.getTransType() == TilePrimitiveDescriptor::GEMM_TN)
    {
        fprintf(fshape, "#define TRANS_TYPE tn");
    }
    else if (func.getTransType() == TilePrimitiveDescriptor::GEMM_TT)
    {
        fprintf(fshape, "#define TRANS_TYPE tt");
    }
    fclose(fshape);
}
void test_mopa()
{
    TilePrimitiveDescriptor desc(
        TilePrimitiveDescriptor::SME_MOPA, TilePrimitiveDescriptor::GEMM_NN, TilePrimitiveDescriptor::DTYPE_FP64, -1);
    VectorDescriptor vec_a;
    vec_a.elements.push_back({0, 0, 4, 1});
    vec_a.elements.push_back({1, 0, 4, 1});
    desc.vec_a.push_back(vec_a);
    VectorDescriptor vec_b;
    vec_b.elements.push_back({0, 0, 4, 1});
    vec_b.elements.push_back({1, 0, 4, 1});
    desc.vec_b.push_back(vec_b);
    IR::Function func(4, 4, 2, TilePrimitiveDescriptor::DTYPE_FP64, TilePrimitiveDescriptor::GEMM_NN);
    func.build(desc);
    auto res = IR::LowerToC(func);
    dump(func, res);
}

void test_sve()
{
    TilePrimitiveDescriptor desc(
        TilePrimitiveDescriptor::SVE_MLA, TilePrimitiveDescriptor::GEMM_NN, TilePrimitiveDescriptor::DTYPE_FP64, -1);
    for (int i = 0; i < 4; ++i)
    {
        VectorDescriptor vec_a;
        vec_a.elements.push_back({0, i, 1, 3});
        // vec_a.elements.push_back({1, i, 1, 3});
        desc.vec_a.push_back(vec_a);
        VectorDescriptor vec_b;
        vec_b.elements.push_back({0, 0, 3, 1});
        // vec_b.elements.push_back({1, 0, 3, 1});
        desc.vec_b.push_back(vec_b);
        // VectorDescriptor vec_a;
        // vec_a.elements.push_back({0, 0, 4, 1});
        // vec_a.elements.push_back({1, 0, 4, 1});
        // desc.vec_a.push_back(vec_a);
        // VectorDescriptor vec_b;
        // vec_b.elements.push_back({0, i, 1, 4});
        // vec_b.elements.push_back({1, i, 1, 4});
        // desc.vec_b.push_back(vec_b);
    }
    IR::Function func(4, 3, 1, TilePrimitiveDescriptor::DTYPE_FP64, desc.trans_type);
    func.build(desc);
    auto res = IR::LowerToC(func);
    dump(func, res);
}
void test_5x9()
{
    auto dtype = TilePrimitiveDescriptor::DTYPE_FP64;
    auto trans_type = TilePrimitiveDescriptor::GEMM_NT;
    auto st = std::chrono::high_resolution_clock::now();
    IR::Function func(5, 9, 1, TilePrimitiveDescriptor::DTYPE_FP64, trans_type);
    TilePrimitiveDescriptor tile0(TilePrimitiveDescriptor::SME_MOPA, trans_type, dtype, -1);
    {
        VectorDescriptor vec_a;
        vec_a.elements.push_back({0, 0, 5, 1});
        tile0.vec_a.push_back(vec_a);
        VectorDescriptor vec_b;
        vec_b.elements.push_back({0, 0, 8, 1});
        tile0.vec_b.push_back(vec_b);
        func.build(tile0);
    }
    TilePrimitiveDescriptor tile1(TilePrimitiveDescriptor::SME2_MLA, trans_type, dtype, -1);
    {
        VectorDescriptor vec_a;
        vec_a.elements.push_back({0, 0, 5});
        tile1.vec_a.push_back(vec_a);
        VectorDescriptor vec_b;
        vec_b.elements.push_back({0, 8, 1, 5});
        tile1.vec_b.push_back(vec_b);
        func.build(tile1);
    }
    auto ed = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = ed - st;
    std::cout << "Time taken: " << diff.count() << " ms\n";
    func.allocate_za();
    func.kLoopMerge();
    func.rearrange();
    auto res = IR::LowerToC(func);
    dump(func, res);
}
void test_5x9_scalar()
{
    TilePrimitiveDescriptor::TRANS_TYPE trans_type = TilePrimitiveDescriptor::GEMM_NT;
    TilePrimitiveDescriptor::OP_TYPE op_type = TilePrimitiveDescriptor::OP_TYPE::SCALAR;
    TilePrimitiveDescriptor::DTYPE dtype = TilePrimitiveDescriptor::DTYPE_FP64;

    auto st = std::chrono::high_resolution_clock::now();
    IR::Function func(5, 9, 1, TilePrimitiveDescriptor::DTYPE_FP64, trans_type);
    TilePrimitiveDescriptor tile0(op_type, trans_type, dtype, -1);
    {
        VectorDescriptor vec_a;
        vec_a.elements.push_back({0, 0, 5, 1});
        tile0.vec_a.push_back(vec_a);
        VectorDescriptor vec_b;
        vec_b.elements.push_back({0, 0, 8, 1});
        tile0.vec_b.push_back(vec_b);
        func.build(tile0);
    }
    TilePrimitiveDescriptor tile1(op_type, trans_type, dtype, -1);
    {
        VectorDescriptor vec_a;
        vec_a.elements.push_back({0, 0, 5, 1});
        tile1.vec_a.push_back(vec_a);
        VectorDescriptor vec_b;
        vec_b.elements.push_back({0, 8, 1, 1});
        tile1.vec_b.push_back(vec_b);
        func.build(tile1);
    }
    auto ed = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = ed - st;
    std::cout << "Time taken: " << diff.count() << " ms\n";
    func.allocate_za();
    func.kLoopMerge();
    // func.rearrange();
    auto res = IR::LowerToC(func);
    dump(func, res);
}
void test_mopa_8x8_nn()
{
    IR::Function func(8, 8, 4, TilePrimitiveDescriptor::DTYPE_FP64, TilePrimitiveDescriptor::GEMM_TN);
    for (int b = 0; b < 4; ++b)
    {
        auto op_type = TilePrimitiveDescriptor::OP_TYPE::SME_MOPA;
        auto dtype = TilePrimitiveDescriptor::DTYPE_FP64;
        auto trans_type = TilePrimitiveDescriptor::GEMM_TN;
        TilePrimitiveDescriptor desc(op_type, trans_type, dtype, -1);
        VectorDescriptor vec_a;
        vec_a.elements.push_back({b, 0, 8, 1});
        desc.vec_a.push_back(vec_a);
        VectorDescriptor vec_b;
        vec_b.elements.push_back({b, 0, 8, 1});
        desc.vec_b.push_back(vec_b);
        func.build(desc);
    }
    func.allocate_za();
    func.kLoopMerge();
    func.rearrange();
    auto res = IR::LowerToC(func);
    dump(func, res);
}
void test_sme_mla_8x8_nn()
{
    auto op_type = TilePrimitiveDescriptor::OP_TYPE::SVE_MLA;
    auto dtype = TilePrimitiveDescriptor::DTYPE_FP64;
    auto trans_type = TilePrimitiveDescriptor::GEMM_NN;
    TilePrimitiveDescriptor desc(op_type, trans_type, dtype, -1);
    desc.K = 8;
    for (int b = 0; b < 2; ++b)
    {
        for (int i = 0; i < 8; ++i)
        {
            VectorDescriptor vec_a;
            vec_a.elements.push_back({b, i, 1, 8});
            desc.vec_a.push_back(vec_a);
        }
        for (int i = 0; i < 8; ++i)
        {
            VectorDescriptor vec_b;
            vec_b.elements.push_back({b, 0, 8, 1});
            desc.vec_b.push_back(vec_b);
        }
    }
    IR::Function func(8, 8, 2, TilePrimitiveDescriptor::DTYPE_FP64, trans_type);
    func.build(desc);
    func.allocate_za();
    auto res = IR::LowerToC(func);
    dump(func, res);
}

void test_sve_mla_lane_tn()
{
    auto op_type = TilePrimitiveDescriptor::OP_TYPE::SVE_MLA;
    auto dtype = TilePrimitiveDescriptor::DTYPE_FP64;
    auto trans_type = TilePrimitiveDescriptor::GEMM_TN;
    auto K = 8;
    TilePrimitiveDescriptor desc(op_type, trans_type, dtype, K);
    VectorDescriptor vec_a;
    VectorDescriptor vec_b;
    vec_a.elements.push_back({0, 0, 8, 1});
    vec_b.elements.push_back({0, 0, 8, 1});
    desc.vec_a.push_back(vec_a);
    desc.vec_b.push_back(vec_b);
    IR::Function func(8, 8, 1, TilePrimitiveDescriptor::DTYPE_FP64, trans_type);
    func.build(desc);
    auto res = IR::LowerToC(func);
    dump(func, res);
}

void test_sve_mla_tt()
{
    auto op_type = TilePrimitiveDescriptor::OP_TYPE::SVE_MLA;
    auto dtype = TilePrimitiveDescriptor::DTYPE_FP64;
    auto trans_type = TilePrimitiveDescriptor::GEMM_TT;
    auto K = 8;
    TilePrimitiveDescriptor desc(op_type, trans_type, dtype, K);
    for (int i = 0; i < 4; ++i)
    {
        VectorDescriptor vec_a;
        VectorDescriptor vec_b;
        vec_a.elements.push_back({0, 0, 4, 1});
        vec_a.elements.push_back({1, 0, 4, 1});
        vec_b.elements.push_back({0, i, 1, 4});
        vec_b.elements.push_back({1, i, 1, 4});
        desc.vec_a.push_back(vec_a);
        desc.vec_b.push_back(vec_b);
    }
    IR::Function func(4, 4, 2, TilePrimitiveDescriptor::DTYPE_FP64, trans_type);
    func.build(desc);
    auto res = IR::LowerToC(func);
    dump(func, res);
}
int main()
{
    // test_mopa();
    // test_sme_mla_8x8_nn();
    // test_mopa_8x8_nn();
    // test_sve_mla_lane_tn();
    // test_5x9_scalar();
    test_sve_mla_tt();
    return 0;
}