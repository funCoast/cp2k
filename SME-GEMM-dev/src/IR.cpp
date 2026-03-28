#include "IR.h"
#include "descriptor.h"
#include "enumerate.h"
#include "macro.h"
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <algorithm>

template <class T>
inline void hash_combine(std::size_t &seed, const T &v)
{
    seed ^= std::hash<T>{}(v) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

template <typename... Args>
struct TupleHashGeneric
{
    size_t operator()(const std::tuple<Args...> &t) const noexcept
    {
        size_t seed = 0;
        std::apply([&seed](const auto &...args) { (hash_combine(seed, args), ...); }, t);
        return seed;
    }
    using tuple_type = std::tuple<Args...>;
};
void IR::Function::build(TilePrimitiveDescriptor &desc)
{
    if (this->trans_type == TilePrimitiveDescriptor::UNDEF)
    {
        this->trans_type = desc.trans_type;
    }
    else if (this->trans_type != desc.trans_type)
    {
        throw std::runtime_error("Transpose type mismatch in Function::build");
    }
    switch (desc.op_type)
    {
    case TilePrimitiveDescriptor::OP_TYPE::SME_MOPA:
        buildMopa(desc);
        break;
    case TilePrimitiveDescriptor::OP_TYPE::SVE_MLA:
        buildMlaSVE(desc);
        break;
    case TilePrimitiveDescriptor::OP_TYPE::SME2_MLA:
        buildMlaSME(desc);
        break;
    case TilePrimitiveDescriptor::OP_TYPE::SCALAR:
        buildScalar(desc);
        break;
    default:
        // Unsupported operation type
        break;
    }
}
// 对于SVE，要求各组标量×向量的向量必须一致，因为融合不同的向量无法获得收益。
void IR::Function::buildMlaSVE(TilePrimitiveDescriptor &desc)
{
    Primitive *primitive = new Primitive();
    Predicate ptrue = Predicate::ptrue(Type::getSVType(desc.dtype));
    auto op_dtype = Type::getSVType(desc.dtype);
    auto scalar_dtype = Type::getType(desc.dtype);
    bool row_mode = !desc.vec_a.empty() && std::all_of(desc.vec_a[0].elements.begin(),
                                                       desc.vec_a[0].elements.end(),
                                                       [](const auto &elem) { return elem.is_scalar(); });
    auto &vecs = row_mode ? desc.vec_b : desc.vec_a;
    auto &scalars = row_mode ? desc.vec_a : desc.vec_b;
    std::vector<Instruction *> accumulators;
    for (int i = 0; i < vecs.size(); ++i)
    {
        SVDupInst *zeros = SVDupInst::create(Constant::create(0., scalar_dtype, primitive), primitive);
        accumulators.push_back(zeros);
    }

    using TupleHash = TupleHashGeneric<Type *, MemoryTarget, int, int, int, int>;
    using KeyType = TupleHash::tuple_type;
    std::unordered_map<KeyType, LoadSVEInst *, TupleHash> load_map;
    auto get_load = [&](Type *, MemoryTarget target, int column, int step, int batch, int k_offset, int elem_offset) {
        auto key = std::make_tuple(Type::getType(desc.dtype), target, column, step, batch, k_offset);
        if (load_map.find(key) == load_map.end())
        {
            Predicate predicate = Predicate::range(elem_offset, elem_offset + step, op_dtype);
            load_map[key] = LoadSVEInst::create(op_dtype, predicate, target, column, step, batch, k_offset, primitive);
        }
        return load_map[key];
    };

    std::map<std::vector<LoadSVEInst *>, Instruction *> sel_map;

    KLoopBeginInst::create(primitive);
    for (int vec_id = 0; vec_id < vecs.size(); ++vec_id)
    {
        auto &vecDesc = vecs[vec_id];
        auto &scalarDesc = scalars[vec_id];
        // Create LoadSVEInst for vector
        std::vector<LoadSVEInst *> loadVecInsts;
        for (const auto &[i, elems] : enumerate(vecDesc))
        {
            const auto elem_offset = static_cast<int>(vecDesc.element_offest(i));
            const auto target = row_mode ? MemoryTarget::GEMM_B : MemoryTarget::GEMM_A;

            // For layouts lowered as k + column * K, subtracting element_offset from
            // column is incorrect because it becomes -offset * K. Keep column intact
            // and apply lane alignment via k_offset instead.
            const bool column_scaled_by_k =
                (target == MemoryTarget::GEMM_A && (desc.trans_type == TilePrimitiveDescriptor::GEMM_NN ||
                                                    desc.trans_type == TilePrimitiveDescriptor::GEMM_NT)) ||
                (target == MemoryTarget::GEMM_B && (desc.trans_type == TilePrimitiveDescriptor::GEMM_NT ||
                                                    desc.trans_type == TilePrimitiveDescriptor::GEMM_TT));

            const int st = column_scaled_by_k ? elems.column : elems.column - elem_offset;
            const int k_offset = column_scaled_by_k ? -elem_offset : 0;

            LoadSVEInst *loadVec =
                get_load(op_dtype, target, st, elems.column_step, elems.batch, k_offset, elem_offset);
            loadVecInsts.push_back(loadVec);
        }
        Instruction *vec_inst = loadVecInsts[0];
        if (auto iter = sel_map.find(loadVecInsts); iter != sel_map.end())
        {
            vec_inst = iter->second;
        }
        else
        {
            for (int i = 1; i < loadVecInsts.size(); ++i)
            {
                vec_inst = SelInst::Create(loadVecInsts[i]->predicate, loadVecInsts[i], vec_inst, primitive);
            }
            sel_map[loadVecInsts] = vec_inst;
        }
        // for (int i = 1; i < loadVecInsts.size(); ++i)
        // {
        //     vec_inst = SelInst::Create(loadVecInsts[i]->predicate, loadVecInsts[i], vec_inst, primitive);
        // }

        // Create LoadSVEInst for scalar
        std::vector<LoadInst *> loadScalarInsts;
        for (const auto &[i, elems] : enumerate(scalarDesc))
        {
            auto st = elems.column;
            auto ed = st + elems.column_step;
            LoadInst *loadScalar = LoadInst::create(
                scalar_dtype, row_mode ? MemoryTarget::GEMM_A : MemoryTarget::GEMM_B, st, elems.batch, primitive);
            loadScalarInsts.push_back(loadScalar);
        }
        Instruction *scalar_inst = loadScalarInsts[0];
        if (loadScalarInsts.size() != 1)
        {
            scalar_inst = SVDupInst::create(scalar_inst, primitive);
            for (int i = 1; i < loadScalarInsts.size(); ++i)
            {
                scalar_inst = SVDupMInst::create(
                    loadScalarInsts[i],
                    Predicate::range(vecDesc.element_offest(i), vecDesc.element_offest(i + 1), op_dtype),
                    scalar_inst,
                    primitive);
            }
        }
        // Create MlaSVEInst
        MlaSVEInst *mla = MlaSVEInst::create(ptrue, accumulators[vec_id], vec_inst, scalar_inst, primitive);
        accumulators[vec_id] = mla;
    }

    KLoopEndInst::create(primitive);
    // Store accumulators back
    if (row_mode)
    {
        for (int i = 0; i < vecs.size(); ++i)
        {
            auto &vecDesc = vecs[i];
            for (const auto &[j, elems] : enumerate(vecDesc))
            {
                auto st = elems.column - vecDesc.element_offest(j);
                auto ed = st + elems.column_step;
                auto elem_offset = vecDesc.element_offest(j);
                Predicate predicate = Predicate::range(elem_offset, elem_offset + elems.column_step, op_dtype);
                StoreSVEInst *store = StoreSVEInst::create(
                    predicate, accumulators[i], scalars[i].elements[j].column, st, elems.batch, primitive);
            }
        }
    }
    else
    {
        ZA *tarnspose_za = ZA::create(primitive);
        for (const auto &[i, accumulator] : enumerate(accumulators))
        {
            WriteZAInst *write_inst =
                WriteZAInst::create(tarnspose_za, i, ReadZAInst::HORIZONTAL, ptrue, op_dtype, accumulator, primitive);
        }
        std::vector<ReadZAInst *> transposed_result;
        auto &vecDesc = vecs[0];
        for (int j = 0; j < vecDesc.elements.size(); ++j)
        {
            for (int jj = 0; jj < vecDesc.elements[j].column_step; ++jj)
            {
                int lane_offset = vecDesc.element_offest(j) + jj;
                if (transposed_result.size() <= lane_offset)
                {
                    transposed_result.resize(lane_offset + 1, nullptr);
                }
                if (transposed_result[lane_offset] == nullptr)
                {
                    transposed_result[lane_offset] = ReadZAInst::create(tarnspose_za,
                                                                        lane_offset,
                                                                        ReadZAInst::VERTICAL,
                                                                        ptrue,
                                                                        op_dtype,
                                                                        SVUndefInst::create(op_dtype, primitive),
                                                                        primitive);
                }
                int valid_lanes = std::min<int>(accumulators.size(), this->N - scalars[0].elements[0].column);
                if (valid_lanes <= 0)
                {
                    continue;
                }
                Predicate predicate = Predicate::range(0, valid_lanes, op_dtype);
                StoreSVEInst *store = StoreSVEInst::create(predicate,
                                                           transposed_result[lane_offset],
                                                           vecDesc.elements[j].column + jj,
                                                           scalars[0].elements[0].column,
                                                           vecDesc.elements[j].batch,
                                                           primitive);
            }
        }
    }

    this->primitives.push_back(primitive);
}

void IR::Function::buildScalar(TilePrimitiveDescriptor &desc)
{
    Primitive *primitive = new Primitive();
    auto scalar_dtype = Type::getType(desc.dtype);
    MY_ASSERT(desc.vec_a.size() == 1);
    MY_ASSERT(desc.vec_b.size() == 1);
    MY_ASSERT(desc.vec_a[0].elements.size() == 1);
    MY_ASSERT(desc.vec_b[0].elements.size() == 1);
    std::vector<Instruction *> accumulators;
    for (int i = 0; i < desc.vec_a[0].elements[0].column_step * desc.vec_b[0].elements[0].column_step; ++i)
    {
        accumulators.push_back(Constant::create(0, scalar_dtype, primitive, true));
    }
    using TupleHash = TupleHashGeneric<MemoryTarget, int, int>;
    using KeyType = TupleHash::tuple_type;
    std::unordered_map<KeyType, LoadInst *, TupleHash> load_map;
    auto get_load = [&](MemoryTarget target, int column, int batch) {
        auto key = std::make_tuple(target, column, batch);
        if (load_map.find(key) == load_map.end())
        {
            load_map[key] = LoadInst::create(scalar_dtype, target, column, batch, primitive);
        }
        return load_map[key];
    };
    KLoopBeginInst::create(primitive);
    for (int vec_id = 0; vec_id < desc.vec_a.size(); ++vec_id)
    {
        auto aDesc = desc.vec_a[vec_id].elements[0];
        auto bDesc = desc.vec_b[vec_id].elements[0];
        std::vector<Instruction *> loadAInsts;
        std::vector<Instruction *> loadBInsts;

        for (int i = 0; i < aDesc.column_step; ++i)
        {
            // LoadInst *load =
            // LoadInst::create(scalar_dtype, MemoryTarget::GEMM_A, aDesc.column + i, aDesc.batch, primitive);
            LoadInst *load = get_load(GEMM_A, aDesc.column + i, aDesc.batch);
            loadAInsts.push_back(load);
        }
        for (int i = 0; i < bDesc.column_step; ++i)
        {
            // LoadInst *load =
            // LoadInst::create(scalar_dtype, MemoryTarget::GEMM_B, bDesc.column + i, bDesc.batch, primitive);
            LoadInst *load = get_load(GEMM_B, bDesc.column + i, bDesc.batch);
            loadBInsts.push_back(load);
        }
        for (int i = 0; i < loadAInsts.size(); ++i)
        {
            for (int j = 0; j < loadBInsts.size(); ++j)
            {
                auto &accu = accumulators[i * loadBInsts.size() + j];
                accu = MlaScalarInst::create(accu, loadAInsts[i], loadBInsts[j], primitive);
            }
        }
    }
    KLoopEndInst::create(primitive);
    for (int vec_id = 0; vec_id < desc.vec_a.size(); ++vec_id)
    {
        auto aDesc = desc.vec_a[vec_id].elements[0];
        auto bDesc = desc.vec_b[vec_id].elements[0];

        for (int i = 0; i < aDesc.column_step; ++i)
        {
            for (int j = 0; j < bDesc.column_step; ++j)
            {
                MY_ASSERT(aDesc.batch == bDesc.batch);
                auto &accu = accumulators[i * bDesc.column_step + j];
                StoreInst::create(accu, aDesc.column + i, bDesc.column + j, aDesc.batch, primitive);
            }
        }
    }

    this->primitives.push_back(primitive);
}
void IR::Function::buildMlaSME(TilePrimitiveDescriptor &desc)
{
    Primitive *primitive = new Primitive();
    Predicate ptrue = Predicate::ptrue(Type::getSVType(desc.dtype));
    auto op_dtype = Type::getSVType(desc.dtype);
    auto scalar_dtype = Type::getType(desc.dtype);
    KLoopBeginInst::create(primitive);
    std::vector<std::pair<Instruction *, Instruction *>> mla_input_insts;
    for (int vec_id = 0; vec_id < desc.vec_a.size(); ++vec_id)
    {
        auto aDesc = desc.vec_a[vec_id];
        auto bDesc = desc.vec_b[vec_id];
        std::vector<Instruction *> loadAInsts;
        std::vector<Instruction *> loadBInsts;

        auto loadFn = [&](const VectorDescriptor &vecDesc, MemoryTarget target, std::vector<Instruction *> &loadInsts) {
            for (const auto &[i, elems] : enumerate(vecDesc))
            {
                if (elems.is_scalar())
                {
                    LoadInst *load = LoadInst::create(scalar_dtype, target, elems.column, elems.batch, primitive);
                    loadInsts.push_back(load);
                    continue;
                }
                else
                {
                    auto lane_st = vecDesc.element_offest(i);
                    auto lane_ed = lane_st + elems.column_step;
                    auto mem_st = elems.column - lane_st;
                    Predicate predicate = Predicate::range(lane_st, lane_ed, op_dtype);
                    LoadSVEInst *load = LoadSVEInst::create(
                        op_dtype, predicate, target, mem_st, elems.column_step, elems.batch, 0, primitive);
                    loadInsts.push_back(load);
                }
            }
        };
        loadFn(aDesc, MemoryTarget::GEMM_A, loadAInsts);
        loadFn(bDesc, MemoryTarget::GEMM_B, loadBInsts);
        Instruction *vec_a =
            IR::isa<LoadInst>(loadAInsts[0]) ? SVDupInst::create(loadAInsts[0], primitive) : loadAInsts[0];
        Instruction *vec_b =
            IR::isa<LoadInst>(loadBInsts[0]) ? SVDupInst::create(loadBInsts[0], primitive) : loadBInsts[0];
        auto lane_range = [&](const VectorDescriptor &vecDesc, int idx) {
            auto elems = vecDesc.elements[idx];
            if (elems.is_scalar())
            {
                return std::pair<int, int>{vecDesc.element_offest(idx), vecDesc.element_offest(idx + 1)};
            }
            auto st = vecDesc.element_offest(idx);
            auto ed = st + elems.column_step;
            return std::pair<int, int>{st, ed};
        };

        auto combineFn = [&](std::vector<Instruction *> &loadInsts,
                             Instruction *&vec_inst,
                             VectorDescriptor &desc,
                             VectorDescriptor &peer_desc) {
            for (int i = 1; i < loadInsts.size(); ++i)
            {
                if (IR::dyn_cast<LoadSVEInst>(loadInsts[i]))
                {
                    auto loadSVE = IR::dyn_cast<LoadSVEInst>(loadInsts[i]);
                    vec_inst = SelInst::Create(loadSVE->predicate, loadSVE, vec_inst, primitive);
                }
                else
                {
                    auto [st, ed] = lane_range(desc, i);
                    if (!peer_desc.elements[i].is_scalar())
                    {
                        std::tie(st, ed) = lane_range(peer_desc, i);
                    }
                    vec_inst =
                        SVDupMInst::create(loadInsts[i], Predicate::range(st, ed, op_dtype), vec_inst, primitive);
                }
            }
        };
        combineFn(loadAInsts, vec_a, aDesc, bDesc);
        combineFn(loadBInsts, vec_b, bDesc, aDesc);
        mla_input_insts.push_back({vec_a, vec_b});
    }

    std::vector<int> vec_lane_map(mla_input_insts.size(), -1);
    ZA *za = nullptr;
    int pack_in_current_za = 2;
    int vl = svl(op_dtype);
    for (int i = 0; i < mla_input_insts.size();)
    {
        int remaining = mla_input_insts.size() - i;
        if (i % vl == 0)
        {
            za = ZA::create(primitive);

            // Keep a stable pack width inside one ZA. Mixing pack=4 and pack=2 in the
            // same ZA causes lane collisions (e.g. for 7 vectors), which corrupts rows.
            int vectors_in_this_za = std::min(vl, remaining);
            pack_in_current_za = (vectors_in_this_za % 4 == 0) ? 4 : 2;
        }

        int remaining_in_this_za = vl - (i % vl);
        int pack = pack_in_current_za;
        if (pack == 4 && (remaining < 4 || remaining_in_this_za < 4))
        {
            pack = 2;
        }

        auto getInstFn = [&](int idx, bool get_a) -> Instruction * {
            if (idx < mla_input_insts.size())
            {
                if (get_a)
                {
                    return mla_input_insts[idx].first;
                }
                else
                {
                    return mla_input_insts[idx].second;
                }
            }
            else
            {
                return SVUndefInst::create(op_dtype, primitive);
            }
        };
        Instruction *a;
        Instruction *b;
        if (pack == 2)
        {
            a = SVCreate2Inst::create(getInstFn(i, true), getInstFn(i + 1, true), primitive);
            b = SVCreate2Inst::create(getInstFn(i, false), getInstFn(i + 1, false), primitive);
        }
        else
        {
            a = SVCreate4Inst::create(
                getInstFn(i, true), getInstFn(i + 1, true), getInstFn(i + 2, true), getInstFn(i + 3, true), primitive);
            b = SVCreate4Inst::create(getInstFn(i, false),
                                      getInstFn(i + 1, false),
                                      getInstFn(i + 2, false),
                                      getInstFn(i + 3, false),
                                      primitive);
        }

        int lane_base = (i % vl) / pack;
        MlaSMEInst::create(za, lane_base, a, b, primitive);

        int lane_stride = vl / pack;
        int mapped_count = std::min(pack, remaining);
        for (int local = 0; local < mapped_count; ++local)
        {
            vec_lane_map[i + local] = lane_base + local * lane_stride;
        }

        i += pack;
    }

    KLoopEndInst::create(primitive);
    // Store back
    for (int vec_id = 0; vec_id < desc.vec_b.size(); ++vec_id)
    {
        auto aDesc = desc.vec_a[vec_id];
        auto bDesc = desc.vec_b[vec_id];
        auto za = primitive->getZA(vec_id / svl(op_dtype));
        int lane = vec_lane_map[vec_id];
        MY_ASSERT(lane >= 0);
        ReadZAInst::DIRECTION direction = ReadZAInst::HORIZONTAL;
        Predicate predicate = ptrue;
        Instruction *ori = SVUndefInst::create(op_dtype, primitive);
        Instruction *val = ReadZAInst::create(za, lane, direction, predicate, op_dtype, ori, primitive);
        for (const auto &[j, elems] : enumerate(bDesc))
        {
            if (elems.column_step > 1)
            {
                auto st = bDesc.element_offest(j);
                auto ed = st + elems.column_step;
                Predicate predicate = Predicate::range(st, ed, op_dtype);
                StoreSVEInst *store =
                    StoreSVEInst::create(predicate, val, aDesc[j].column, elems.column - st, elems.batch, primitive);
                continue;
            }
            else
            {
                for (int jj = 0; jj < elems.dup_num; ++jj)
                {
                    auto st = bDesc.element_offest(j) + jj;
                    Predicate predicate = Predicate::range(st, st + 1, op_dtype);
                    StoreSVEInst *store = StoreSVEInst::create(
                        predicate, val, aDesc[j].column + jj, elems.column - st, elems.batch, primitive);
                }
            }
        }
    }
    this->primitives.push_back(primitive);
}

void IR::Function::buildMopa(TilePrimitiveDescriptor &desc)
{
    Primitive *primitive = new Primitive();
    Predicate ptrue = Predicate::ptrue(Type::getSVType(desc.dtype));
    auto op_dtype = Type::getSVType(desc.dtype);

    // here transA and transB is for standard SME outer-product layout, i.e. A is column-major and B is row-major.
    // When transA/transB is true, it means the actual layout of A/B is the opposite, i.e. A is row-major and B is column-major.
    // This is different from is_a_transpose in frontend and other parts.
    bool transA =
        (desc.trans_type == TilePrimitiveDescriptor::GEMM_NN || desc.trans_type == TilePrimitiveDescriptor::GEMM_NT);
    bool transB =
        (desc.trans_type == TilePrimitiveDescriptor::GEMM_NT || desc.trans_type == TilePrimitiveDescriptor::GEMM_TT);

    auto &vecADesc = desc.vec_a[0];
    auto &vecBDesc = desc.vec_b[0];

    // 当转置存在时，K loop步长由转置时加载进ZA的行数/列数决定
    int k_unroll = 1;
    if (transA || transB)
    {
        k_unroll = svl(dtype);
    }

    // 1指定修改为实际的 K loop 的展开 step
    KLoopBeginInst::create(primitive, k_unroll);
    primitive->set_k_unroll(k_unroll);

    // =====================================
    // 构建 vector A (沿 K 展开)
    // =====================================
    std::vector<Instruction *> vec_a_unrolled(k_unroll, nullptr);
    if (transA)
    {
        ZA *za_a_trans = ZA::create(primitive);
        int lane = 0;
        for (const auto &[i, elems] : enumerate(vecADesc))
        {
            int elem_cnt = elems.column_step * elems.dup_num;
            // Transposed path loads directly from each batch pointer. The lane mapping is handled by ZA lanes,
            // so applying element_offset to memory address here is incorrect and can produce negative indices.
            auto base_st = elems.column;
            for (int k = 0; k < elem_cnt; ++k)
            {
                // A 转置时连续内存跨越K，读取完整SVL向量。
                // 此时 k 代表空间维度 M 上的偏移，K 维度无需 k_offset (由 K Loop 整体推进)
                LoadSVEInst *loadA =
                    LoadSVEInst::create(op_dtype, ptrue, MemoryTarget::GEMM_A, base_st + k, 1, elems.batch, 0);
                primitive->insert(loadA);
                // 逐个向量水平写入 ZA 中
                Instruction *writeA =
                    WriteZAInst::create(za_a_trans, lane, WriteZAInst::HORIZONTAL, ptrue, op_dtype, loadA);
                primitive->insert(writeA);
                lane++;
            }
        }
        Instruction *ori_a = SVUndefInst::create(op_dtype);
        primitive->insert(ori_a);

        // 代替单次读取，沿 K 维度展开垂直读出 `k_unroll` 个向量
        for (int ku = 0; ku < k_unroll; ++ku)
        {
            vec_a_unrolled[ku] = ReadZAInst::create(za_a_trans, ku, ReadZAInst::VERTICAL, ptrue, op_dtype, ori_a);
            primitive->insert(vec_a_unrolled[ku]);
        }
    }
    else
    {
        for (int ku = 0; ku < k_unroll; ++ku)
        {
            std::vector<LoadSVEInst *> loadAInsts;
            for (const auto &[i, elems] : enumerate(vecADesc))
            {
                auto elem_offset = vecADesc.element_offest(i);
                auto st = elems.column - elem_offset;
                auto ed = st + elems.column_step;
                Predicate predicate = Predicate::range(elem_offset, elem_offset + elems.column_step, op_dtype);
                // 不转置的数据加载，必须通过添加 ku 作为 k_offset 同步沿 K 展开
                LoadSVEInst *loadA = LoadSVEInst::create(
                    op_dtype, predicate, MemoryTarget::GEMM_A, st, elems.column_step, elems.batch, ku);
                loadAInsts.push_back(loadA);
                primitive->insert(loadA);
            }
            Instruction *vec_a = loadAInsts[0];
            for (size_t i = 1; i < loadAInsts.size(); ++i)
            {
                vec_a = SelInst::Create(loadAInsts[i]->predicate, loadAInsts[i], vec_a);
                primitive->insert(vec_a);
            }
            vec_a_unrolled[ku] = vec_a;
        }
    }

    // =====================================
    // 构建 vector B (沿 K 展开)
    // =====================================
    std::vector<Instruction *> vec_b_unrolled(k_unroll, nullptr);
    if (transB)
    {
        ZA *za_b_trans = ZA::create(primitive);
        int lane = 0;
        for (const auto &[i, elems] : enumerate(vecBDesc))
        {
            int elem_cnt = elems.column_step * elems.dup_num;
            // Same as A transpose: do not offset memory base by vector lane offset in fused-batch mode.
            auto base_st = elems.column;
            for (int k = 0; k < elem_cnt; ++k)
            {
                LoadSVEInst *loadB =
                    LoadSVEInst::create(op_dtype, ptrue, MemoryTarget::GEMM_B, base_st + k, 1, elems.batch, 0);
                primitive->insert(loadB);
                // B 作为转置源时垂直写入
                Instruction *writeB =
                    WriteZAInst::create(za_b_trans, lane, WriteZAInst::VERTICAL, ptrue, op_dtype, loadB);
                primitive->insert(writeB);
                lane++;
            }
        }
        Instruction *ori_b = SVUndefInst::create(op_dtype);
        primitive->insert(ori_b);

        // 展开水平读出 `k_unroll` 个向量
        for (int ku = 0; ku < k_unroll; ++ku)
        {
            vec_b_unrolled[ku] = ReadZAInst::create(za_b_trans, ku, ReadZAInst::HORIZONTAL, ptrue, op_dtype, ori_b);
            primitive->insert(vec_b_unrolled[ku]);
        }
    }
    else
    {
        for (int ku = 0; ku < k_unroll; ++ku)
        {
            std::vector<LoadSVEInst *> loadBInsts;
            for (const auto &[i, elems] : enumerate(vecBDesc))
            {
                auto elem_offset = vecBDesc.element_offest(i);
                auto st = elems.column - elem_offset;
                auto ed = st + elems.column_step;
                Predicate predicate = Predicate::range(elem_offset, elem_offset + elems.column_step, op_dtype);
                // 同理，沿 K 维度的偏移追加 ku 作为 k_offset
                LoadSVEInst *loadB = LoadSVEInst::create(
                    op_dtype, predicate, MemoryTarget::GEMM_B, st, elems.column_step, elems.batch, ku);
                loadBInsts.push_back(loadB);
                primitive->insert(loadB);
            }
            Instruction *vec_b = loadBInsts[0];
            for (size_t i = 1; i < loadBInsts.size(); ++i)
            {
                vec_b = SelInst::Create(loadBInsts[i]->predicate, loadBInsts[i], vec_b);
                primitive->insert(vec_b);
            }
            vec_b_unrolled[ku] = vec_b;
        }
    }

    // =====================================
    // 创建 MopaInst 计算及结尾释放
    // =====================================
    ZA *za = ZA::create(primitive);
    // 相应的，mopa 也要展开 k_unroll 次
    for (int ku = 0; ku < k_unroll; ++ku)
    {
        MopaInst *mopa = MopaInst::create(vec_a_unrolled[ku], vec_b_unrolled[ku], za, ptrue, ptrue);
        primitive->insert(mopa);
    }
    primitive->insert(KLoopEndInst::create());

    // 写回 ZA 计算结果到内存（逻辑保持不变）
    for (size_t j = 0; j < vecBDesc.seg_size(); ++j)
    {
        auto elems_b = vecBDesc[j];
        for (int i = 0; i < vecADesc[j].column_step; ++i)
        {
            int lane = i + vecADesc.element_offest(j);
            ReadZAInst::DIRECTION direction = ReadZAInst::HORIZONTAL;
            Predicate predicate = Predicate::range(
                0 + vecBDesc.element_offest(j), elems_b.column_step + vecBDesc.element_offest(j), op_dtype);
            Type *type = op_dtype;
            Instruction *ori = SVUndefInst::create(type);
            Instruction *val = ReadZAInst::create(za, lane, direction, predicate, type, ori);
            StoreSVEInst *store = StoreSVEInst::create(
                predicate, val, i + vecADesc[j].column, elems_b.column - vecBDesc.element_offest(j), elems_b.batch);
            primitive->insert(ori);
            primitive->insert(val);
            primitive->insert(store);
        }
    }
    this->primitives.push_back(primitive);
}

void IR::Function::allocate_za()
{
    int za_id = 0;
    for (auto prim : this->primitives)
    {
        for (auto za : prim->ZAs())
        {
            za->setId(za_id);
            za_id = (za_id + 1) % za_num(Type::getSVType(this->dtype));
        }
    }
}

void IR::Function::kLoopMerge()
{
    if (primitives.empty())
    {
        return;
    }

    std::set<int> za_used;
    std::vector<Primitive *> new_primitives;
    Primitive *new_prim = nullptr;
    std::vector<Primitive *> prim_group;
    auto contains = [](std::set<int> &s, Primitive *prim) {
        for (auto za : prim->ZAs())
        {
            if (s.find(za->id) != s.end())
            {
                return true;
            }
        }
        return false;
    };
    auto merge = [&]() {
        new_prim = new Primitive();
        int k_unroll = 1;
        for (auto prim_in_group : prim_group)
        {
            bool is_in_kloop = false;
            for (auto inst : prim_in_group->insts())
            {
                if (isa<KLoopBeginInst>(inst))
                {
                    is_in_kloop = true;
                    k_unroll = IR::dyn_cast<KLoopBeginInst>(inst)->step;
                    break;
                }
                else
                {
                    new_prim->insts().push_back(inst);
                }
            }
        }
        KLoopBeginInst::create(new_prim, k_unroll);
        for (auto prim_in_group : prim_group)
        {
            for (auto za : prim_in_group->ZAs())
            {
                new_prim->insert(za);
            }
            bool is_in_kloop = false;
            for (auto inst : prim_in_group->insts())
            {
                if (isa<KLoopBeginInst>(inst))
                {
                    is_in_kloop = true;
                    continue;
                }
                if (isa<KLoopEndInst>(inst))
                {
                    break;
                }
                if (is_in_kloop)
                {
                    new_prim->insts().push_back(inst);
                }
            }
        }
        KLoopEndInst::create(new_prim);

        for (auto prim_in_group : prim_group)
        {
            bool is_after_kloop = false;
            for (auto inst : prim_in_group->insts())
            {
                if (isa<KLoopEndInst>(inst))
                {
                    is_after_kloop = true;
                    continue;
                }
                if (is_after_kloop)
                {
                    new_prim->insts().push_back(inst);
                }
            }
        }
        for (auto prim_in_group : prim_group)
        {
            delete prim_in_group;
        }
        prim_group.clear();
        za_used.clear();
        new_primitives.push_back(new_prim);
    };
    int last_k_unroll = primitives[0]->get_k_unroll();
    for (auto prim : primitives)
    {
        if (contains(za_used, prim) || prim->get_k_unroll() != last_k_unroll || prim_group.size() >= 4)
        {
            merge();
            prim_group.push_back(prim);
            for (auto za : prim->ZAs())
            {
                za_used.insert(za->id);
            }
            last_k_unroll = prim->get_k_unroll();
        }
        else
        {
            prim_group.push_back(prim);
            for (auto za : prim->ZAs())
            {
                za_used.insert(za->id);
            }
        }
    }
    if (!prim_group.empty())
    {
        merge();
    }
    this->primitives = new_primitives;
}

void IR::Function::rearrange()
{
    for (auto prim : this->primitives)
    {
        std::vector<Instruction *> calculate_insts;
        for (auto inst : prim->insts())
        {
            if (isa<MlaSVEInst, MlaSVEInst, MopaInst>(inst))
            {
                calculate_insts.push_back(inst);
            }
        }
        if (calculate_insts.size() > 16)
        {
            continue;
        }
        for (auto calculate_inst : calculate_insts)
        {
            prim->insts().erase(std::remove(prim->insts().begin(), prim->insts().end(), calculate_inst),
                                prim->insts().end());
        }
        for (auto inst : prim->insts())
        {
            if (isa<KLoopEndInst>(inst))
            {
                for (auto calculate_inst : calculate_insts)
                {
                    prim->insts().insert(std::find(prim->insts().begin(), prim->insts().end(), inst), calculate_inst);
                }
                break;
            }
        }
    }
}