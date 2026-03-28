#include "IR.h"
#include "instructions.h"
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <format>
#include <iomanip>
#include <memory>
#include <optional>
#include <iostream>
#include <ratio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "vectorLinkedList.h"

#ifdef __APPLE__
#define APPLE_SIG_PREFIX "_"
#else
#define APPLE_SIG_PREFIX ""
#endif

namespace IR
{

class AGenerator
{
private:
    struct PredicateKey
    {
        std::int64_t mask;
        bool is_b64;

        bool operator==(const PredicateKey &other) const = default;
    };

    struct PredicateKeyHash
    {
        std::size_t operator()(const PredicateKey &key) const
        {
            return std::hash<std::int64_t>()((key.mask << 1) ^ static_cast<std::int64_t>(key.is_b64));
        }
    };

    struct LoopContext
    {
        std::string loop_label;
        std::string end_label;
        int step;
        size_t init_insert_pos;
    };

    struct ScalarUseReg
    {
        int reg = -1;
        bool temporary = false;
    };

    struct ScalarSpillHome
    {
        int slot = -1;
        bool is_fp64 = false;
        bool is_accumulator = false;
    };

    struct VectorSpillHome
    {
        int slot = -1;
        bool is_fp64 = false;
    };

    struct BranchFixup
    {
        std::size_t binary_index = 0;
        int pc = 0;
        ARM::Instruction *branch = nullptr;
        std::string target;
    };

    static constexpr const char *fp64KernelSymbol = APPLE_SIG_PREFIX "_Z15gemm_kernel_optxPPdS0_S0_iii";
    static constexpr const char *fp32KernelSymbol = APPLE_SIG_PREFIX "_Z15gemm_kernel_optxPPfS0_S0_iii";

    TilePrimitiveDescriptor::TRANS_TYPE trans_type = TilePrimitiveDescriptor::TRANS_TYPE::GEMM_NN;
    bool kernel_is_fp64 = false;
    int M = 0;
    int N = 0;
    int loop_depth = 0;
    int next_p_reg = 1;
    int next_transient_p_reg = 4;
    int spill_slot_count = 0;
    int scalar_spill_slot_reserve = 0;
    int vector_spill_slot_count = 0;
    bool in_k_loop = false;

    std::unordered_map<Instruction *, int> z_reg_map;
    std::unordered_map<Instruction *, std::vector<int>> tuple_map;
    std::unordered_map<Instruction *, ScalarSpillHome> scalar_spill_map;
    std::unordered_map<Instruction *, VectorSpillHome> vector_spill_map;
    std::unordered_map<Instruction *, int> remaining_uses;
    std::unordered_map<Instruction *, std::vector<Instruction *>> user_map;
    std::unordered_map<PredicateKey, int, PredicateKeyHash> predicate_map;
    std::unordered_map<std::string, int> label_address;
    std::unordered_set<Instruction *> initialized_scalar_roots;
    std::unordered_map<Instruction *, size_t> accumulator_insts;
    std::unordered_map<Instruction *, std::pair<size_t, int>> accumulator_to_mla_inst; // inst -> <mla pos, reg>
    std::optional<size_t> insert_point = std::nullopt;

    std::vector<int> free_z_regs;
    std::vector<LoopContext> loop_labels;
    VectorLinkedList<std::unique_ptr<ARM::Instruction>> arm_insts;
    bool output_prepared = false;
    bool binary_prepared = false;
    std::vector<std::uint32_t> cached_binary;
    bool p0_initialized = false;
    bool p0_is_b64 = true;
    Primitive *current_primitive = nullptr;
    std::size_t current_inst_index = 0;
    Instruction *current_debug_inst = nullptr;

    static constexpr int BatchCountReg = 0;
    static constexpr int AArrayReg = 1;
    static constexpr int BArrayReg = 2;
    static constexpr int CArrayReg = 3;
    static constexpr int BatchIndexReg = 4;
    static constexpr int KIndexReg = 5;
    static constexpr int KReg = 6;
    static constexpr int SpillBaseReg = 7;
    static constexpr int PredicateTmp0 = 9;
    static constexpr int PredicateTmp1 = 10;
    static constexpr int SpillSlotTmp = 11;
    static constexpr int ImmTmp0 = 16;
    static constexpr int ImmTmp1 = 17;
    // x18 is reserved by the Darwin arm64 ABI, so do not use it as a scratch GPR.
    static constexpr int AddrBaseReg = 13;
    static constexpr int CachedPredicateLimit = 4;
    static constexpr int PredicateTransient0 = 4;
    static constexpr int PredicateTransient1 = 5;
    static constexpr int PredicateScratchUpper = 14;
    static constexpr int PredicateScratchLower = 15;

    bool A_is_transpose() const
    {
        return trans_type == TilePrimitiveDescriptor::TRANS_TYPE::GEMM_TN ||
               trans_type == TilePrimitiveDescriptor::TRANS_TYPE::GEMM_TT;
    }

    bool B_is_transpose() const
    {
        return trans_type == TilePrimitiveDescriptor::TRANS_TYPE::GEMM_NT ||
               trans_type == TilePrimitiveDescriptor::TRANS_TYPE::GEMM_TT;
    }

    static bool is_full_true(std::int64_t mask, bool is_b64)
    {
        return is_b64 ? mask == 0x0101010101010101LL : mask == 0x1111111111111111LL;
    }

    static bool is_empty(std::int64_t mask) { return mask == 0; }

    static std::pair<int, int> predicate_range(std::int64_t mask, bool is_b64)
    {
        const int shift = is_b64 ? 8 : 4;
        const int lanes = is_b64 ? 8 : 16;
        int first = -1;
        int last = -1;
        for (int i = 0; i < lanes; ++i)
        {
            if (mask & (1LL << (i * shift)))
            {
                if (first == -1)
                {
                    first = i;
                }
                last = i;
            }
        }
        if (first == -1)
        {
            return {-1, -1};
        }
        for (int i = first; i <= last; ++i)
        {
            if (!(mask & (1LL << (i * shift))))
            {
                throw std::runtime_error("non-contiguous predicate mask is unsupported in AGenerator");
            }
        }
        return {first, last + 1};
    }

    static int align_up(int value, int align) { return ((value + align - 1) / align) * align; }

    int scalar_slot_bytes() const { return kernel_is_fp64 ? 8 : 4; }

    static int vector_slot_bytes() { return 64; }

    int vector_spill_base_bytes() const { return scalar_spill_slot_reserve * scalar_slot_bytes(); }

    static int vector_slot_stride_elements(bool is_fp64) { return is_fp64 ? 8 : 16; }

    size_t appendArmInst(std::unique_ptr<ARM::Instruction> inst)
    {
        if (insert_point.has_value())
        {
            auto new_pos = arm_insts.insert_after(insert_point.value(), std::move(inst));
            insert_point.value() = new_pos;
            return new_pos;
        }
        else
        {
            return arm_insts.push_back(std::move(inst));
        }
    }

    void appendLabel(const std::string &label) { appendArmInst(std::make_unique<ARM::LabelInst>(label, 0)); }

    void emit_mov_imm(int reg, int imm) { appendArmInst(std::make_unique<ARM::MOVImmInst>(reg, imm)); }

    void emit_add_imm(int xd, int xn, int imm)
    {
        while (imm > 0xFFF)
        {
            int chunk = std::min(imm, 0xFFF);
            appendArmInst(std::make_unique<ARM::ADDImmInst>(xd, xn, chunk));
            imm -= chunk;
        }
        appendArmInst(std::make_unique<ARM::ADDImmInst>(xd, xn, imm));
    }

    void emit_sub_imm(int xd, int xn, int imm)
    {
        while (imm > 0xFFF)
        {
            int chunk = std::min(imm, 0xFFF);
            appendArmInst(std::make_unique<ARM::SUBImmInst>(xd, xn, chunk));
            imm -= chunk;
        }
        appendArmInst(std::make_unique<ARM::SUBImmInst>(xd, xn, imm));
    }

    void emit_add_reg(int xd, int xn, int xm, int shift = 0)
    {
        appendArmInst(std::make_unique<ARM::ADDRegInst>(xd, xn, xm, shift));
    }

    void release_simd_reg(int reg)
    {
        if (reg < 0)
        {
            return;
        }
        free_z_regs.push_back(reg);
        std::sort(free_z_regs.begin(), free_z_regs.end(), std::greater<int>());
    }

    void spill_vector_value(Instruction *inst, int reg)
    {
        const bool is_fp64 = inst->type->is_fp64();
        VectorSpillHome home;
        auto it = vector_spill_map.find(inst);
        if (it != vector_spill_map.end())
        {
            home = it->second;
        }
        else
        {
            home.slot = vector_spill_slot_count++;
            home.is_fp64 = is_fp64;
            vector_spill_map[inst] = home;
        }

        if (home.is_fp64 != is_fp64)
        {
            throw std::runtime_error("vector spill dtype mismatch in AGenerator");
        }
        bool spill_accumulator = in_k_loop && accumulator_insts.contains(inst);
        if (spill_accumulator)
        {
            // If the value to spill is an accumulator that is still being updated in the k loop, we need to insert a spill right after the instruction that updates it, instead of spilling the live register here. This is because the register allocator may choose to reuse the same register for another value after this point, which would cause incorrect behavior if we spill it here.
            insert_point = accumulator_insts[inst];
        }

        int pg = ensure_full_true_predicate(is_fp64);
        const int elem_bytes = is_fp64 ? 8 : 4;
        const int base_elements = vector_spill_base_bytes() / elem_bytes;
        const int index_elements = base_elements + home.slot * vector_slot_stride_elements(is_fp64);
        auto insert_spill_insts = [&](int _reg) {
            materialize_constant_index(SpillSlotTmp, index_elements);
            if (is_fp64)
            {
                appendArmInst(std::make_unique<ARM::ST1DInst>(_reg, pg, SpillBaseReg, SpillSlotTmp));
            }
            else
            {
                appendArmInst(std::make_unique<ARM::ST1WInst>(_reg, pg, SpillBaseReg, SpillSlotTmp));
            }
        };

        auto insert_restore_insts = [&](int _reg) {
            materialize_constant_index(SpillSlotTmp, index_elements);
            if (is_fp64)
            {
                appendArmInst(std::make_unique<ARM::LD1DInst>(_reg, pg, SpillBaseReg, SpillSlotTmp));
            }
            else
            {
                appendArmInst(std::make_unique<ARM::LD1WInst>(_reg, pg, SpillBaseReg, SpillSlotTmp));
            }
        };
        insert_spill_insts(reg);
        if (spill_accumulator)
        {
            if (auto it = accumulator_to_mla_inst.find(inst); it != accumulator_to_mla_inst.end())
            {
                const auto &[mla_pos, mla_reg] = it->second;
                insert_point = mla_pos;
                insert_spill_insts(mla_reg);
                insert_point = arm_insts.prev(mla_pos);
                insert_restore_insts(mla_reg);
            }

            insert_point = std::nullopt;
        }
    }

    bool try_make_one_simd_reg_free(const std::unordered_set<int> &pinned)
    {
        for (auto it = z_reg_map.begin(); it != z_reg_map.end(); ++it)
        {
            Instruction *inst = it->first;
            int reg = it->second;
            if (pinned.contains(reg) || reg_is_live_in_other_tuple(inst, reg))
            {
                continue;
            }
            bool aliased = false;
            for (const auto &[other_inst, other_reg] : z_reg_map)
            {
                if (other_inst != inst && other_reg == reg)
                {
                    aliased = true;
                    break;
                }
            }
            if (aliased)
            {
                continue;
            }

            spill_vector_value(inst, reg);
            z_reg_map.erase(it);
            release_simd_reg(reg);
            return true;
        }

        return false;
    }

    int allocate_simd_reg(const std::unordered_set<int> &pinned = {})
    {
        if (!free_z_regs.empty())
        {
            int reg = free_z_regs.back();
            free_z_regs.pop_back();
            return reg;
        }

        if (!try_make_one_simd_reg_free(pinned) || free_z_regs.empty())
        {
            throw std::runtime_error(std::format(
                "ran out of SIMD registers in AGenerator (opcode={}, live_z={}, live_tuple={}, live_scalar={})",
                current_debug_inst ? static_cast<int>(current_debug_inst->opcode()) : -1,
                z_reg_map.size(),
                tuple_map.size(),
                scalar_spill_map.size()));
        }

        int reg = free_z_regs.back();
        free_z_regs.pop_back();
        return reg;
    }

    int allocate_p_reg()
    {
        if (next_p_reg >= CachedPredicateLimit)
        {
            throw std::runtime_error("ran out of temporary predicate registers in AGenerator");
        }
        return next_p_reg++;
    }

    int allocate_transient_p_reg()
    {
        int reg = next_transient_p_reg;
        next_transient_p_reg =
            (next_transient_p_reg == PredicateTransient0) ? PredicateTransient1 : PredicateTransient0;
        return reg;
    }

    int allocate_z_reg(const std::unordered_set<int> &pinned = {}) { return allocate_simd_reg(pinned); }

    std::vector<int> allocate_contiguous_z_regs(int count, const std::unordered_set<int> &pinned = {})
    {
        for (int attempt = 0; attempt <= 64; ++attempt)
        {
            std::vector<int> sorted = free_z_regs;
            std::sort(sorted.begin(), sorted.end());
            for (std::size_t i = 0; i + count <= sorted.size(); ++i)
            {
                if ((sorted[i] % count) != 0)
                {
                    continue;
                }
                bool contiguous = true;
                for (int j = 1; j < count; ++j)
                {
                    if (sorted[i + j] != sorted[i] + j)
                    {
                        contiguous = false;
                        break;
                    }
                }
                if (!contiguous)
                {
                    continue;
                }

                std::vector<int> regs;
                for (int j = 0; j < count; ++j)
                {
                    regs.push_back(sorted[i] + j);
                }
                for (int reg : regs)
                {
                    auto it = std::find(free_z_regs.begin(), free_z_regs.end(), reg);
                    if (it == free_z_regs.end())
                    {
                        throw std::runtime_error("failed to reserve contiguous Z register block");
                    }
                    free_z_regs.erase(it);
                }
                std::sort(free_z_regs.begin(), free_z_regs.end(), std::greater<int>());
                return regs;
            }

            if (!try_make_one_simd_reg_free(pinned))
            {
                break;
            }
        }

        throw std::runtime_error(std::format(
            "ran out of contiguous Z register blocks in AGenerator (opcode={}, count={}, live_z={}, live_tuple={})",
            current_debug_inst ? static_cast<int>(current_debug_inst->opcode()) : -1,
            count,
            z_reg_map.size(),
            tuple_map.size()));
    }

    static bool is_contiguous_aligned_block(const std::vector<int> &regs)
    {
        if (regs.empty())
        {
            return false;
        }
        if ((regs.front() % static_cast<int>(regs.size())) != 0)
        {
            return false;
        }
        for (std::size_t i = 1; i < regs.size(); ++i)
        {
            if (regs[i] != regs[i - 1] + 1)
            {
                return false;
            }
        }
        return true;
    }

    bool reg_is_live_in_other_tuple(Instruction *inst, int reg) const
    {
        for (const auto &[other_inst, other_regs] : tuple_map)
        {
            if (other_inst == inst)
            {
                continue;
            }
            if (std::find(other_regs.begin(), other_regs.end(), reg) != other_regs.end())
            {
                return true;
            }
        }
        return false;
    }

    void copy_into_z_reg(int zd, int zs, bool is_fp64)
    {
        if (zd == zs)
        {
            return;
        }
        int pg_full = ensure_full_true_predicate(is_fp64);
        appendArmInst(std::make_unique<ARM::SELInst>(zd, pg_full, zs, zs, is_fp64));
    }

    void free_z_reg_if_unused(Instruction *inst)
    {
        auto it = z_reg_map.find(inst);
        if (it == z_reg_map.end())
        {
            return;
        }
        int reg = it->second;
        bool aliased = false;
        for (const auto &[other_inst, other_reg] : z_reg_map)
        {
            if (other_inst != inst && other_reg == reg)
            {
                aliased = true;
                break;
            }
        }
        z_reg_map.erase(it);
        if (!aliased && !reg_is_live_in_other_tuple(inst, reg))
        {
            release_simd_reg(reg);
        }
    }

    void free_tuple_if_unused(Instruction *inst)
    {
        auto it = tuple_map.find(inst);
        if (it == tuple_map.end())
        {
            return;
        }
        for (int reg : it->second)
        {
            bool aliased = false;
            for (const auto &[other_inst, other_reg] : z_reg_map)
            {
                if (other_inst != inst && other_reg == reg)
                {
                    aliased = true;
                    break;
                }
            }
            if (!aliased && !reg_is_live_in_other_tuple(inst, reg))
            {
                release_simd_reg(reg);
            }
        }
        tuple_map.erase(it);
    }

    int allocate_scalar_temp_reg() { return allocate_simd_reg(); }

    void free_scalar_temp_reg(int reg)
    {
        if (reg < 0)
        {
            return;
        }
        release_simd_reg(reg);
    }

    ScalarSpillHome &ensure_scalar_spill_home(Instruction *inst, bool is_accumulator)
    {
        auto it = scalar_spill_map.find(inst);
        if (it != scalar_spill_map.end())
        {
            it->second.is_accumulator = it->second.is_accumulator || is_accumulator;
            return it->second;
        }

        if (spill_slot_count >= scalar_spill_slot_reserve)
        {
            throw std::runtime_error("scalar spill slot reserve exceeded in AGenerator");
        }

        ScalarSpillHome home;
        home.slot = spill_slot_count++;
        home.is_fp64 = inst->type->is_fp64();
        home.is_accumulator = is_accumulator;
        auto [insert_it, _] = scalar_spill_map.emplace(inst, home);
        return insert_it->second;
    }

    void materialize_predicate(std::int64_t mask, bool is_b64, int pd)
    {
        if (is_full_true(mask, is_b64))
        {
            appendArmInst(std::make_unique<ARM::PTUREInst>(pd, is_b64));
        }
        else if (is_empty(mask))
        {
            appendArmInst(std::make_unique<ARM::PFALSEInst>(pd));
        }
        else
        {
            const auto [first, end] = predicate_range(mask, is_b64);
            emit_mov_imm(PredicateTmp0, 0);
            emit_mov_imm(PredicateTmp1, end);
            if (first == 0)
            {
                appendArmInst(std::make_unique<ARM::WHILELTInst>(pd, PredicateTmp0, PredicateTmp1, is_b64));
            }
            else
            {
                appendArmInst(
                    std::make_unique<ARM::WHILELTInst>(PredicateScratchUpper, PredicateTmp0, PredicateTmp1, is_b64));
                emit_mov_imm(PredicateTmp1, first);
                appendArmInst(
                    std::make_unique<ARM::WHILELTInst>(PredicateScratchLower, PredicateTmp0, PredicateTmp1, is_b64));
                int all = ensure_full_true_predicate(is_b64);
                appendArmInst(std::make_unique<ARM::EORInst>(pd, all, PredicateScratchUpper, PredicateScratchLower));
            }
        }
    }

    int ensure_full_true_predicate(bool is_b64)
    {
        if (!p0_initialized || p0_is_b64 != is_b64)
        {
            appendArmInst(std::make_unique<ARM::PTUREInst>(0, is_b64));
            p0_initialized = true;
            p0_is_b64 = is_b64;
        }
        return 0;
    }

    int ensure_predicate(std::int64_t mask, bool is_b64)
    {
        if (is_full_true(mask, is_b64))
        {
            return ensure_full_true_predicate(is_b64);
        }

        PredicateKey key{mask, is_b64};
        auto it = predicate_map.find(key);
        if (it != predicate_map.end())
        {
            return it->second;
        }

        if (next_p_reg < CachedPredicateLimit)
        {
            int pd = allocate_p_reg();
            materialize_predicate(mask, is_b64, pd);
            predicate_map[key] = pd;
            return pd;
        }

        int pd = allocate_transient_p_reg();
        materialize_predicate(mask, is_b64, pd);
        return pd;
    }

    void load_batch_base(MemoryTarget target, int batch)
    {
        emit_mov_imm(ImmTmp0, batch);
        emit_add_reg(ImmTmp0, BatchIndexReg, ImmTmp0);
        int array_reg = target == GEMM_A ? AArrayReg : (target == GEMM_B ? BArrayReg : CArrayReg);
        appendArmInst(std::make_unique<ARM::LDRScalarInst>(AddrBaseReg, array_reg, ImmTmp0, true));
    }

    void materialize_constant_index(int reg, int value)
    {
        if (value < 0 || value > 0xffff)
        {
            throw std::runtime_error("constant index is out of MOVZ range in AGenerator");
        }
        emit_mov_imm(reg, value);
    }

    void materialize_c_index(int linear_index, bool is_fp64)
    {
        if (linear_index >= 0)
        {
            materialize_constant_index(ImmTmp1, linear_index);
            return;
        }

        const int byte_offset = -linear_index * (is_fp64 ? 8 : 4);
        if (byte_offset > 0xfff)
        {
            throw std::runtime_error("negative C offset is out of SUB immediate range in AGenerator");
        }
        emit_sub_imm(AddrBaseReg, AddrBaseReg, byte_offset);
        emit_mov_imm(ImmTmp1, 0);
    }

    void emit_mul_by_const(int dst, int src, int factor)
    {
        if (factor < 0)
        {
            throw std::runtime_error("negative multiply factor is unsupported in AGenerator");
        }

        emit_mov_imm(dst, 0);
        int bit = 0;
        unsigned int u = static_cast<unsigned int>(factor);
        while (u != 0)
        {
            if (u & 1U)
            {
                emit_add_reg(dst, dst, src, bit);
            }
            u >>= 1U;
            ++bit;
        }
    }

    void compute_ab_index(MemoryTarget target, int column, int k_offset)
    {
        if (k_offset >= 0)
        {
            emit_mov_imm(ImmTmp0, k_offset);
            emit_add_reg(ImmTmp0, KIndexReg, ImmTmp0);
        }
        else
        {
            const int neg = -k_offset;
            if (neg > 0xfff)
            {
                throw std::runtime_error("negative k_offset is out of SUB immediate range in AGenerator");
            }
            emit_sub_imm(ImmTmp0, KIndexReg, neg);
        }

        if (target == GEMM_A)
        {
            if (A_is_transpose())
            {
                emit_mul_by_const(ImmTmp1, ImmTmp0, M);
                if (column != 0)
                {
                    if (column > 0)
                    {
                        emit_add_imm(ImmTmp1, ImmTmp1, column);
                    }
                    else
                    {
                        emit_sub_imm(ImmTmp1, ImmTmp1, -column);
                    }
                }
            }
            else
            {
                emit_mul_by_const(ImmTmp1, KReg, column);
                emit_add_reg(ImmTmp1, ImmTmp1, ImmTmp0);
            }
            return;
        }

        if (B_is_transpose())
        {
            emit_mul_by_const(ImmTmp1, KReg, column);
            emit_add_reg(ImmTmp1, ImmTmp1, ImmTmp0);
        }
        else
        {
            emit_mul_by_const(ImmTmp1, ImmTmp0, N);
            if (column != 0)
            {
                if (column > 0)
                {
                    emit_add_imm(ImmTmp1, ImmTmp1, column);
                }
                else
                {
                    emit_sub_imm(ImmTmp1, ImmTmp1, -column);
                }
            }
        }
    }

    void emit_scalar_load_bits(LoadInst *load, int reg)
    {
        load_batch_base(load->target, load->batch);
        compute_ab_index(load->target, load->column, 0);
        appendArmInst(std::make_unique<ARM::LDRScalarInst>(reg, AddrBaseReg, ImmTmp1, load->type->is_fp64()));
    }

    void spill_scalar_home(int fp_reg, int slot)
    {
        emit_mov_imm(SpillSlotTmp, slot);
        appendArmInst(std::make_unique<ARM::FMovFPToWXInst>(ImmTmp0, fp_reg, kernel_is_fp64));
        appendArmInst(std::make_unique<ARM::STRScalarInst>(ImmTmp0, SpillBaseReg, SpillSlotTmp, kernel_is_fp64));
    }

    void reload_scalar_home(int slot, int fp_reg)
    {
        emit_mov_imm(SpillSlotTmp, slot);
        appendArmInst(std::make_unique<ARM::LDRScalarInst>(ImmTmp0, SpillBaseReg, SpillSlotTmp, kernel_is_fp64));
        appendArmInst(std::make_unique<ARM::FMovWXToFPInst>(fp_reg, ImmTmp0, kernel_is_fp64));
    }

    ScalarUseReg acquire_scalar_fp(Instruction *inst)
    {
        if (auto *constant = dyn_cast<Constant>(inst))
        {
            if (constant->val != 0.0)
            {
                throw std::runtime_error("only zero constants are supported in AGenerator scalar lowering");
            }
            int fp_reg = allocate_scalar_temp_reg();
            emit_mov_imm(ImmTmp0, 0);
            appendArmInst(std::make_unique<ARM::FMovWXToFPInst>(fp_reg, ImmTmp0, kernel_is_fp64));
            return {fp_reg, true};
        }

        if (auto *load = dyn_cast<LoadInst>(inst))
        {
            int fp_reg = allocate_scalar_temp_reg();
            emit_scalar_load_bits(load, ImmTmp0);
            appendArmInst(std::make_unique<ARM::FMovWXToFPInst>(fp_reg, ImmTmp0, kernel_is_fp64));
            return {fp_reg, true};
        }

        if (auto *mla = dyn_cast<MlaScalarInst>(inst))
        {
            auto it = scalar_spill_map.find(mla);
            if (it == scalar_spill_map.end())
            {
                throw std::runtime_error("scalar source must be materialized before use in AGenerator");
            }

            int fp_reg = allocate_scalar_temp_reg();
            reload_scalar_home(it->second.slot, fp_reg);
            return {fp_reg, true};
        }

        throw std::runtime_error("unsupported scalar FP source in AGenerator");
    }

    int acquire_scalar_bits_reg(Instruction *inst)
    {
        if (auto *constant = dyn_cast<Constant>(inst))
        {
            if (constant->val != 0.0)
            {
                throw std::runtime_error("only zero constants are supported in AGenerator scalar lowering");
            }
            emit_mov_imm(ImmTmp0, 0);
            return ImmTmp0;
        }

        if (auto *load = dyn_cast<LoadInst>(inst))
        {
            emit_scalar_load_bits(load, ImmTmp0);
            return ImmTmp0;
        }

        throw std::runtime_error("unsupported scalar bit-pattern source in AGenerator");
    }

    void release_scalar_use(const ScalarUseReg &use)
    {
        if (use.temporary)
        {
            free_scalar_temp_reg(use.reg);
        }
    }

    int acquire_z_reg(Instruction *inst, const std::unordered_set<int> &pinned = {})
    {
        auto it = z_reg_map.find(inst);
        if (it != z_reg_map.end())
        {
            return it->second;
        }

        auto spill_it = vector_spill_map.find(inst);
        if (spill_it != vector_spill_map.end())
        {
            const bool is_fp64 = spill_it->second.is_fp64;
            const int elem_bytes = is_fp64 ? 8 : 4;
            const int base_elements = vector_spill_base_bytes() / elem_bytes;
            const int index_elements = base_elements + spill_it->second.slot * vector_slot_stride_elements(is_fp64);

            int zd = allocate_z_reg(pinned);
            int pg = ensure_full_true_predicate(is_fp64);
            materialize_constant_index(SpillSlotTmp, index_elements);
            if (is_fp64)
            {
                appendArmInst(std::make_unique<ARM::LD1DInst>(zd, pg, SpillBaseReg, SpillSlotTmp));
            }
            else
            {
                appendArmInst(std::make_unique<ARM::LD1WInst>(zd, pg, SpillBaseReg, SpillSlotTmp));
            }

            z_reg_map[inst] = zd;
            return zd;
        }

        throw std::runtime_error(
            std::format("vector source must be materialized before use (producer_opcode={}, consumer_opcode={})",
                        static_cast<int>(inst->opcode()),
                        current_debug_inst ? static_cast<int>(current_debug_inst->opcode()) : -1));
    }

    void emit_svundef(SVUndefInst *inst)
    {
        if (z_reg_map.contains(inst))
        {
            return;
        }
        int zd = allocate_z_reg();
        emit_mov_imm(ImmTmp0, 0);
        appendArmInst(std::make_unique<ARM::DUPGPRInst>(zd, ImmTmp0, inst->type->is_fp64()));
        z_reg_map[inst] = zd;
    }

    int emit_load_sve(LoadSVEInst *load)
    {
        auto it = z_reg_map.find(load);
        if (it != z_reg_map.end())
        {
            return it->second;
        }
        int zd = allocate_z_reg();
        load_batch_base(load->target, load->batch);
        compute_ab_index(load->target, load->column, load->k_offset);
        int pg = ensure_predicate(load->predicate.active, load->type->is_fp64());
        if (load->type->is_fp64())
        {
            appendArmInst(std::make_unique<ARM::LD1DInst>(zd, pg, AddrBaseReg, ImmTmp1));
        }
        else
        {
            appendArmInst(std::make_unique<ARM::LD1WInst>(zd, pg, AddrBaseReg, ImmTmp1));
        }
        z_reg_map[load] = zd;
        return zd;
    }

    void emit_sel(SelInst *sel)
    {
        std::unordered_set<int> pinned;
        int zfalse = dyn_cast<SVUndefInst>(sel->falseValue) ? -1 : acquire_z_reg(sel->falseValue, pinned);
        if (zfalse >= 0)
        {
            pinned.insert(zfalse);
        }
        int ztrue = acquire_z_reg(sel->trueValue, pinned);
        pinned.insert(ztrue);
        int zd = allocate_z_reg(pinned);
        int pg = ensure_predicate(sel->predicate.active, sel->type->is_fp64());
        if (zfalse < 0)
        {
            zfalse = zd;
        }
        appendArmInst(std::make_unique<ARM::SELInst>(zd, pg, ztrue, zfalse, sel->type->is_fp64()));
        z_reg_map[sel] = zd;
    }

    void emit_svdup(SVDupInst *dup)
    {
        int zd = allocate_z_reg();
        int scalar = acquire_scalar_bits_reg(dup->val);
        size_t pos = appendArmInst(std::make_unique<ARM::DUPGPRInst>(zd, scalar, dup->type->is_fp64()));
        z_reg_map[dup] = zd;
        if (!in_k_loop)
        {
            accumulator_insts[dup] = pos;
        }
    }

    void emit_svdupm(SVDupMInst *dup)
    {
        int zori = dyn_cast<SVUndefInst>(dup->ori) ? -1 : acquire_z_reg(dup->ori);
        std::unordered_set<int> pinned;
        if (zori >= 0)
        {
            pinned.insert(zori);
        }
        int zd = allocate_z_reg(pinned);
        int pg = ensure_predicate(dup->predicate.active, dup->type->is_fp64());
        if (zori >= 0)
        {
            copy_into_z_reg(zd, zori, dup->type->is_fp64());
        }
        int scalar = acquire_scalar_bits_reg(dup->val);
        pinned.insert(zd);
        int tmp = allocate_z_reg(pinned);
        appendArmInst(std::make_unique<ARM::DUPGPRInst>(tmp, scalar, dup->type->is_fp64()));
        appendArmInst(std::make_unique<ARM::SELInst>(zd, pg, tmp, zd, dup->type->is_fp64()));
        release_simd_reg(tmp);
        z_reg_map[dup] = zd;
    }

    void emit_read_za(ReadZAInst *read)
    {
        int zsrc = dyn_cast<SVUndefInst>(read->source) ? -1 : acquire_z_reg(read->source);
        std::unordered_set<int> pinned;
        if (zsrc >= 0)
        {
            pinned.insert(zsrc);
        }
        int zd = allocate_z_reg(pinned);
        if (zsrc >= 0)
        {
            copy_into_z_reg(zd, zsrc, read->type->is_fp64());
        }
        int pg = ensure_predicate(read->predicate.active, read->type->is_fp64());
        constexpr int SliceIndexReg = 12;
        emit_mov_imm(SliceIndexReg, read->lane);
        appendArmInst(std::make_unique<ARM::MOVV2TInst>(
            zd, pg, SliceIndexReg, read->za->id, read->type->is_fp64(), read->direction == ReadZAInst::VERTICAL));
        z_reg_map[read] = zd;
    }

    void emit_mla_sve(MlaSVEInst *mla)
    {
        int pg = ensure_predicate(mla->predicate.active, mla->type->is_fp64());
        std::unordered_set<int> pinned;
        int zn = mla->a->type->is_svtype() ? acquire_z_reg(mla->a, pinned) : allocate_z_reg(pinned);
        bool zn_tmp = !mla->a->type->is_svtype();
        if (zn_tmp)
        {
            appendArmInst(std::make_unique<ARM::DUPGPRInst>(zn, acquire_scalar_bits_reg(mla->a), kernel_is_fp64));
        }
        pinned.insert(zn);
        int zm = mla->b->type->is_svtype() ? acquire_z_reg(mla->b, pinned) : allocate_z_reg(pinned);
        bool zm_tmp = !mla->b->type->is_svtype();
        if (zm_tmp)
        {
            appendArmInst(std::make_unique<ARM::DUPGPRInst>(zm, acquire_scalar_bits_reg(mla->b), kernel_is_fp64));
        }
        pinned.insert(zm);
        int zda = acquire_z_reg(mla->c, pinned);
        size_t pos = appendArmInst(std::make_unique<ARM::FMLASVEInst>(zda, pg, zn, zm, mla->type->is_fp64()));
        accumulator_to_mla_inst[mla->c] = {pos, zda};
        if (zn_tmp)
        {
            release_simd_reg(zn);
        }
        if (zm_tmp)
        {
            release_simd_reg(zm);
        }
        if (auto it = vector_spill_map.find(mla->c); it != vector_spill_map.end())
        {
            vector_spill_map[mla] = it->second; // update spill mapping to new value
            vector_spill_map.erase(it);
            spill_vector_value(mla, zda);
        }
        else
        {
            z_reg_map.erase(mla->c);
            z_reg_map[mla] = zda;
        }
        accumulator_insts[mla] = accumulator_insts[mla->c];
        accumulator_to_mla_inst[mla] = accumulator_to_mla_inst[mla->c];
        accumulator_insts.erase(mla->c);
    }

    const std::vector<int> &materialize_tuple(Instruction *inst)
    {
        auto it = tuple_map.find(inst);
        if (it != tuple_map.end())
        {
            return it->second;
        }

        std::vector<Instruction *> sources;
        if (auto *create2 = dyn_cast<SVCreate2Inst>(inst))
        {
            sources = {create2->v0, create2->v1};
        }
        else if (auto *create4 = dyn_cast<SVCreate4Inst>(inst))
        {
            sources = {create4->v0, create4->v1, create4->v2, create4->v3};
        }
        else
        {
            throw std::runtime_error("tuple source is not a tuple-producing instruction");
        }

        std::vector<int> existing_regs;
        existing_regs.reserve(sources.size());
        std::unordered_set<int> pinned;
        for (Instruction *source : sources)
        {
            int reg = acquire_z_reg(source, pinned);
            existing_regs.push_back(reg);
            pinned.insert(reg);
        }

        std::vector<int> regs = is_contiguous_aligned_block(existing_regs)
                                    ? existing_regs
                                    : allocate_contiguous_z_regs(static_cast<int>(sources.size()), pinned);
        int pg = -1;
        for (std::size_t i = 0; i < sources.size(); ++i)
        {
            int source_reg = existing_regs[i];
            if (regs[i] != source_reg)
            {
                if (pg < 0)
                {
                    pg = ensure_full_true_predicate(inst->type->is_fp64());
                }
                appendArmInst(
                    std::make_unique<ARM::SELInst>(regs[i], pg, source_reg, source_reg, inst->type->is_fp64()));
            }
        }

        auto [insert_it, _] = tuple_map.emplace(inst, regs);
        return insert_it->second;
    }

    const std::vector<int> &require_existing_tuple(Instruction *inst) const
    {
        auto it = tuple_map.find(inst);
        if (it != tuple_map.end())
        {
            return it->second;
        }

        throw std::runtime_error(
            std::format("tuple source must be materialized before use (producer_opcode={}, consumer_opcode={})",
                        static_cast<int>(inst->opcode()),
                        current_debug_inst ? static_cast<int>(current_debug_inst->opcode()) : -1));
    }

    void emit_vector_store(StoreSVEInst *store)
    {
        load_batch_base(GEMM_C, store->batch);
        materialize_c_index(store->row * N + store->column, store->val->type->is_fp64());
        int pg = ensure_predicate(store->predicate.active, store->val->type->is_fp64());
        int zt = acquire_z_reg(store->val);
        if (store->val->type->is_fp64())
        {
            appendArmInst(std::make_unique<ARM::ST1DInst>(zt, pg, AddrBaseReg, ImmTmp1));
        }
        else
        {
            appendArmInst(std::make_unique<ARM::ST1WInst>(zt, pg, AddrBaseReg, ImmTmp1));
        }
    }

    void emit_scalar_mla(MlaScalarInst *mla)
    {
        bool src_is_accumulator = false;
        bool src_is_accumulator_root_constant = false;
        int src_slot = -1;
        ScalarSpillHome *src_home_ptr = nullptr;

        if (auto *prev = dyn_cast<MlaScalarInst>(mla->c))
        {
            auto prev_it = scalar_spill_map.find(prev);
            if (prev_it == scalar_spill_map.end())
            {
                throw std::runtime_error("scalar accumulator source must have spill home in AGenerator");
            }
            src_slot = prev_it->second.slot;
            src_is_accumulator = prev_it->second.is_accumulator;
            src_home_ptr = &prev_it->second;
        }
        else if (auto *constant = dyn_cast<Constant>(mla->c))
        {
            src_is_accumulator = constant->is_accumulator;
            if (src_is_accumulator)
            {
                src_is_accumulator_root_constant = true;
            }
        }

        ScalarSpillHome *dst_home_ptr = nullptr;
        if (src_is_accumulator && src_home_ptr != nullptr)
        {
            auto [it, _] = scalar_spill_map.emplace(mla, *src_home_ptr);
            it->second.is_accumulator = true;
            dst_home_ptr = &it->second;
        }
        else
        {
            auto &home = ensure_scalar_spill_home(mla, src_is_accumulator);
            dst_home_ptr = &home;
        }

        if (src_is_accumulator_root_constant)
        {
            src_slot = dst_home_ptr->slot;
            if (initialized_scalar_roots.insert(mla).second)
            {
                auto saved_insert_point = insert_point;
                if (in_k_loop && !loop_labels.empty())
                {
                    insert_point = loop_labels.back().init_insert_pos;
                }
                // Accumulator constants are zero-initialized for scalar kernels.
                emit_mov_imm(ImmTmp0, 0);
                emit_mov_imm(SpillSlotTmp, dst_home_ptr->slot);
                appendArmInst(
                    std::make_unique<ARM::STRScalarInst>(ImmTmp0, SpillBaseReg, SpillSlotTmp, kernel_is_fp64));
                if (in_k_loop && !loop_labels.empty() && insert_point.has_value())
                {
                    loop_labels.back().init_insert_pos = insert_point.value();
                }
                insert_point = saved_insert_point;
            }
        }

        if (src_slot < 0)
        {
            src_slot = dst_home_ptr->slot;
        }

        ScalarUseReg c;
        if (src_slot >= 0)
        {
            c = {allocate_scalar_temp_reg(), true};
            reload_scalar_home(src_slot, c.reg);
        }
        else
        {
            c = acquire_scalar_fp(mla->c);
        }
        ScalarUseReg a = acquire_scalar_fp(mla->a);
        ScalarUseReg b = acquire_scalar_fp(mla->b);

        appendArmInst(std::make_unique<ARM::FMADDScalarInst>(c.reg, a.reg, b.reg, c.reg, kernel_is_fp64));
        spill_scalar_home(c.reg, dst_home_ptr->slot);

        release_scalar_use(a);
        release_scalar_use(b);
        release_scalar_use(c);
    }

    void emit_scalar_store(StoreInst *store)
    {
        load_batch_base(GEMM_C, store->batch);
        materialize_c_index(store->row * N + store->column, store->val->type->is_fp64());
        ScalarUseReg value = acquire_scalar_fp(store->val);
        appendArmInst(std::make_unique<ARM::FMovFPToWXInst>(ImmTmp0, value.reg, store->val->type->is_fp64()));
        appendArmInst(std::make_unique<ARM::STRScalarInst>(ImmTmp0, AddrBaseReg, ImmTmp1, store->val->type->is_fp64()));
        release_scalar_use(value);
    }

    void emit_mla_sme(MlaSMEInst *mla)
    {
        const auto &a_tuple = require_existing_tuple(mla->a);
        const auto &b_tuple = require_existing_tuple(mla->b);
        bool is_vg2 = a_tuple.size() == 2;
        if ((is_vg2 && b_tuple.size() != 2) || (!is_vg2 && a_tuple.size() != 4) || b_tuple.size() != a_tuple.size())
        {
            throw std::runtime_error("tuple arity mismatch in MLA_SME lowering");
        }

        int wv = 8 + (mla->lane % 4);
        emit_mov_imm(wv, mla->za->id + mla->lane * static_cast<int>(za_num(mla->a->type)));
        appendArmInst(std::make_unique<ARM::FMLASMEInst>(
            wv, a_tuple.front(), a_tuple.back(), b_tuple.front(), b_tuple.back(), mla->a->type->is_fp64(), is_vg2));
    }

    std::vector<Instruction *> operands_of(Instruction *inst) const
    {
        std::vector<Instruction *> operands;
        switch (inst->opcode())
        {
        case OPCODE::SEL:
            {
                auto *sel = dyn_cast<SelInst>(inst);
                operands = {sel->trueValue, sel->falseValue};
                break;
            }
        case OPCODE::SVDUP:
            operands = {dyn_cast<SVDupInst>(inst)->val};
            break;
        case OPCODE::SVDUPM:
            {
                auto *dup = dyn_cast<SVDupMInst>(inst);
                operands = {dup->val, dup->ori};
                break;
            }
        case OPCODE::MLA_SVE:
            {
                auto *mla = dyn_cast<MlaSVEInst>(inst);
                operands = {mla->c, mla->a, mla->b};
                break;
            }
        case OPCODE::MLA_SCALAR:
            {
                auto *mla = dyn_cast<MlaScalarInst>(inst);
                operands = {mla->c, mla->a, mla->b};
                break;
            }
        case OPCODE::MOPA:
            {
                auto *mopa = dyn_cast<MopaInst>(inst);
                operands = {mopa->va, mopa->vb};
                break;
            }
        case OPCODE::MLA_SME:
            {
                auto *mla = dyn_cast<MlaSMEInst>(inst);
                operands = {mla->a, mla->b};
                break;
            }
        case OPCODE::READ_ZA:
            operands = {dyn_cast<ReadZAInst>(inst)->source};
            break;
        case OPCODE::WRITE_ZA:
            operands = {dyn_cast<WriteZAInst>(inst)->source};
            break;
        case OPCODE::STORE_SVE:
            operands = {dyn_cast<StoreSVEInst>(inst)->val};
            break;
        case OPCODE::STORE:
            operands = {dyn_cast<StoreInst>(inst)->val};
            break;
        case OPCODE::SVCREATE2:
            {
                auto *create2 = dyn_cast<SVCreate2Inst>(inst);
                operands = {create2->v0, create2->v1};
                break;
            }
        case OPCODE::SVCREATE4:
            {
                auto *create4 = dyn_cast<SVCreate4Inst>(inst);
                operands = {create4->v0, create4->v1, create4->v2, create4->v3};
                break;
            }
        default:
            break;
        }
        return operands;
    }

    void collect_use_counts(Function &func)
    {
        remaining_uses.clear();
        user_map.clear();
        scalar_spill_map.clear();
        initialized_scalar_roots.clear();
        spill_slot_count = 0;
        scalar_spill_slot_reserve = 0;
        for (auto *prim : func.primitives)
        {
            for (auto *inst : prim->instructions)
            {
                if (inst->opcode() == OPCODE::MLA_SCALAR)
                {
                    ++scalar_spill_slot_reserve;
                }
                for (Instruction *operand : operands_of(inst))
                {
                    if (operand)
                    {
                        remaining_uses[operand]++;
                        user_map[operand].push_back(inst);
                    }
                }
            }
        }
    }

    void release_if_dead(Instruction *inst)
    {
        auto it = remaining_uses.find(inst);
        if (it == remaining_uses.end())
        {
            return;
        }
        it->second--;
        if (it->second > 0)
        {
            return;
        }

        free_z_reg_if_unused(inst);
        vector_spill_map.erase(inst);
        if (auto it = scalar_spill_map.find(inst); it != scalar_spill_map.end())
        {
            if (!in_k_loop)
            {
                scalar_spill_map.erase(it);
            }
        }
        free_tuple_if_unused(inst);
    }

    void release_operands(Instruction *inst)
    {
        for (Instruction *operand : operands_of(inst))
        {
            if (operand)
            {
                release_if_dead(operand);
            }
        }
    }

    void build_binary_with_fixups()
    {
        label_address.clear();
        cached_binary.clear();
        cached_binary.reserve(arm_insts.size());
        std::vector<BranchFixup> fixups;

        int pc = 0;
        for (const auto &inst : arm_insts)
        {
            if (auto *label = ARM::dyn_cast<ARM::LabelInst>(inst.get()))
            {
                label_address[label->name()] = pc;
                continue;
            }

            const std::size_t binary_index = cached_binary.size();
            cached_binary.push_back(static_cast<std::uint32_t>(inst->to_binary()));

            if (auto *bge = ARM::dyn_cast<ARM::BGEInst>(inst.get()))
            {
                fixups.push_back({binary_index, pc, bge, bge->target()});
            }
            else if (auto *blt = ARM::dyn_cast<ARM::BLTInst>(inst.get()))
            {
                fixups.push_back({binary_index, pc, blt, blt->target()});
            }

            pc += 4;
        }

        for (const auto &fixup : fixups)
        {
            const int target = label_address.at(fixup.target);
            const int imm19 = (target - fixup.pc) / 4;

            if (auto *bge = ARM::dyn_cast<ARM::BGEInst>(fixup.branch))
            {
                bge->set_imm19(imm19);
            }
            else if (auto *blt = ARM::dyn_cast<ARM::BLTInst>(fixup.branch))
            {
                blt->set_imm19(imm19);
            }
            else
            {
                throw std::runtime_error("unexpected branch fixup instruction type in AGenerator");
            }

            cached_binary[fixup.binary_index] = static_cast<std::uint32_t>(fixup.branch->to_binary());
        }

        binary_prepared = true;
    }

    void ensure_binary_prepared()
    {
        if (!binary_prepared)
        {
            build_binary_with_fixups();
        }
    }

    void finalize_function_frame()
    {
        const int scalar_bytes = scalar_spill_slot_reserve * scalar_slot_bytes();
        const int vector_bytes = vector_spill_slot_count * vector_slot_bytes();
        const int stack_size = align_up(scalar_bytes + vector_bytes, 16);
        if (stack_size > 0)
        {
            auto pos = arm_insts.push_front(std::make_unique<ARM::ADDImmInst>(SpillBaseReg, 31, 0));
            insert_point = pos;
            emit_sub_imm(31, 31, stack_size);
            emit_add_imm(SpillBaseReg, 31, 0);
            insert_point = std::nullopt;
            emit_add_imm(31, 31, stack_size);
        }
        appendArmInst(std::make_unique<ARM::RETInst>());
    }

    void prepare_output()
    {
        if (output_prepared)
        {
            return;
        }
        finalize_function_frame();
        binary_prepared = false;
        cached_binary.clear();
        output_prepared = true;
    }

    std::string render_text(bool output_asm)
    {
        prepare_output();
        if (!output_asm)
        {
            ensure_binary_prepared();
        }
        std::size_t binary_cursor = 0;
        std::stringstream ss;
        ss << "\t.text\n";
        ss << "\t.p2align 2\n";
        if (kernel_is_fp64)
        {
            ss << "\t.globl " << fp64KernelSymbol << "\n";
            ss << fp64KernelSymbol << ":\n";
        }
        else
        {
            ss << "\t.globl " << fp32KernelSymbol << "\n";
            ss << fp32KernelSymbol << ":\n";
        }
        for (const auto &inst : arm_insts)
        {
            if (auto *label = ARM::dyn_cast<ARM::LabelInst>(inst.get()))
            {
                ss << label->to_asm() << "\n";
                continue;
            }

            if (output_asm)
            {
                ss << "\t" << inst->to_asm() << "\n";
            }
            else
            {
                if (binary_cursor >= cached_binary.size())
                {
                    throw std::runtime_error("binary cursor out of range in render_text");
                }
                ss << "\t.inst 0x" << std::hex << std::setw(8) << std::setfill('0') << cached_binary[binary_cursor++]
                   << std::dec << std::setfill(' ') << "\n";
            }
        }
        return ss.str();
    }

    std::vector<std::uint32_t> render_binary()
    {
        prepare_output();
        ensure_binary_prepared();
        return cached_binary;
    }

public:
    AGenerator()
    {
        for (int reg = 31; reg >= 0; --reg)
        {
            free_z_regs.push_back(reg);
        }
    }

    void generate(Function &func)
    {
        trans_type = func.getTransType();
        kernel_is_fp64 = func.dtype == TilePrimitiveDescriptor::DTYPE_FP64;
        M = func.M;
        N = func.N;
        p0_initialized = false;
        p0_is_b64 = kernel_is_fp64;
        collect_use_counts(func);

        const std::string batch_loop = ".Lbatch_loop";
        const std::string batch_end = ".Lbatch_end";

        emit_mov_imm(BatchIndexReg, 0);
        appendLabel(batch_loop);
        appendArmInst(std::make_unique<ARM::CMPInst>(BatchIndexReg, BatchCountReg));
        appendArmInst(std::make_unique<ARM::BGEInst>(0, batch_end));
        (void)ensure_full_true_predicate(kernel_is_fp64);

        for (auto *prim : func.primitives)
        {
            current_primitive = prim;
            for (std::size_t i = 0; i < prim->instructions.size(); ++i)
            {
                current_inst_index = i;
                auto *inst = prim->instructions[i];
                emitInstruction(inst);
            }
        }
        current_primitive = nullptr;

        emit_add_imm(BatchIndexReg, BatchIndexReg, func.batch_per_step);
        appendArmInst(std::make_unique<ARM::CMPInst>(BatchIndexReg, BatchCountReg));
        appendArmInst(std::make_unique<ARM::BLTInst>(0, batch_loop));
        appendLabel(batch_end);
    }

    void emitInstruction(Instruction *inst)
    {
        current_debug_inst = inst;
        switch (inst->opcode())
        {
        case OPCODE::KLOOP_BEGIN:
            {
                in_k_loop = true;
                auto *loop = dyn_cast<KLoopBeginInst>(inst);
                const std::string loop_label = std::format(".Lk_loop_{}", loop_depth);
                const std::string end_label = std::format(".Lk_end_{}", loop_depth);
                ++loop_depth;

                appendArmInst(std::make_unique<ARM::ZeroInst>(0xff));
                size_t init_insert_pos = appendArmInst(std::make_unique<ARM::MOVImmInst>(KIndexReg, 0));
                loop_labels.push_back({loop_label, end_label, loop->step, init_insert_pos});
                appendLabel(loop_label);
                appendArmInst(std::make_unique<ARM::CMPInst>(KIndexReg, KReg));
                appendArmInst(std::make_unique<ARM::BGEInst>(0, end_label));
                break;
            }

        case OPCODE::KLOOP_END:
            {
                in_k_loop = false;
                if (loop_labels.empty())
                {
                    throw std::runtime_error("KLOOP_END without matching KLOOP_BEGIN");
                }
                auto loop_ctx = loop_labels.back();
                loop_labels.pop_back();
                emit_add_imm(KIndexReg, KIndexReg, loop_ctx.step);
                appendArmInst(std::make_unique<ARM::CMPInst>(KIndexReg, KReg));
                appendArmInst(std::make_unique<ARM::BLTInst>(0, loop_ctx.loop_label));
                appendLabel(loop_ctx.end_label);
                break;
            }

        case OPCODE::LOAD_SVE:
            (void)emit_load_sve(dyn_cast<LoadSVEInst>(inst));
            break;

        case OPCODE::SEL:
            (void)emit_sel(dyn_cast<SelInst>(inst));
            break;

        case OPCODE::SVDUPM:
            (void)emit_svdupm(dyn_cast<SVDupMInst>(inst));
            break;

        case OPCODE::SVDUP:
            (void)emit_svdup(dyn_cast<SVDupInst>(inst));
            break;

        case OPCODE::READ_ZA:
            (void)emit_read_za(dyn_cast<ReadZAInst>(inst));
            break;

        case OPCODE::SVUNDEF:
            (void)emit_svundef(dyn_cast<SVUndefInst>(inst));
            break;

        case OPCODE::WRITE_ZA:
            {
                auto *write = dyn_cast<WriteZAInst>(inst);
                int pg = ensure_predicate(write->predicate.active, write->type->is_fp64());
                int zn = acquire_z_reg(write->source);
                constexpr int SliceIndexReg = 12;
                emit_mov_imm(SliceIndexReg, write->lane);
                appendArmInst(std::make_unique<ARM::MOVT2VInst>(zn,
                                                                pg,
                                                                SliceIndexReg,
                                                                write->za->id,
                                                                write->type->is_fp64(),
                                                                write->direction == WriteZAInst::VERTICAL));
                break;
            }

        case OPCODE::LOAD:
        case OPCODE::CONSTANT:
            break;

        case OPCODE::SVCREATE2:
        case OPCODE::SVCREATE4:
            (void)materialize_tuple(inst);
            break;

        case OPCODE::MLA_SME:
            emit_mla_sme(dyn_cast<MlaSMEInst>(inst));
            break;

        case OPCODE::MOPA:
            {
                auto *mopa = dyn_cast<MopaInst>(inst);
                int pn = ensure_predicate(mopa->pa.active, mopa->va->type->is_fp64());
                int pm = ensure_predicate(mopa->pb.active, mopa->vb->type->is_fp64());
                std::unordered_set<int> pinned;
                int zn = acquire_z_reg(mopa->va, pinned);
                pinned.insert(zn);
                int zm = acquire_z_reg(mopa->vb, pinned);
                appendArmInst(
                    std::make_unique<ARM::FMOPAInst>(mopa->za->id, pn, pm, zn, zm, mopa->va->type->is_fp64()));
                break;
            }

        case OPCODE::MLA_SVE:
            emit_mla_sve(dyn_cast<MlaSVEInst>(inst));
            break;

        case OPCODE::MLA_SCALAR:
            emit_scalar_mla(dyn_cast<MlaScalarInst>(inst));
            break;

        case OPCODE::STORE_SVE:
            emit_vector_store(dyn_cast<StoreSVEInst>(inst));
            break;

        case OPCODE::STORE:
            emit_scalar_store(dyn_cast<StoreInst>(inst));
            break;

        default:
            throw std::runtime_error(
                std::format("Unsupported Opcode in AGenerator: {}", static_cast<int>(inst->opcode())));
        }

        release_operands(inst);
        current_debug_inst = nullptr;
    }
    LoweredAResult lower(Function &func, void *binary_addr)
    {
        auto st = std::chrono::high_resolution_clock::now();
        generate(func);
        auto ed = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = ed - st;
        std::cout << std::format("Code generation took {:.2f} microseconds\n", elapsed.count());

        LoweredAResult result;
        result.asm_text = render_text(true);
        st = std::chrono::high_resolution_clock::now();
        result.binary = render_binary();
        ed = std::chrono::high_resolution_clock::now();
        if (binary_addr != nullptr)
        {
            result.write_binary_to(binary_addr);
        }
        result.inst_text = render_text(false);
        std::cout << std::format("Render binary took {:.2f} microseconds\n", elapsed.count());
        return result;
    }
};

LoweredAResult LowerToA(Function &func, void *binary_addr)
{
    AGenerator gen;
    return gen.lower(func, binary_addr);
}

} // namespace IR
