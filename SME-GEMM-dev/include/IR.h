#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>
#include <stdexcept>
#include "descriptor.h"
namespace IR
{
class CGenerator;
class AGenerator;
enum MemoryTarget
{
    GEMM_A,
    GEMM_B,
    GEMM_C
};
enum OPCODE
{
    MOPA,
    MLA_SME,
    MLA_SVE,
    MLA_SCALAR,
    LOAD,
    STORE,
    LOAD_SVE,
    STORE_SVE,
    SEL,
    READ_ZA,
    WRITE_ZA,
    KLOOP_BEGIN,
    KLOOP_END,
    SVUNDEF,
    SVDUP,
    SVDUPM,
    CONSTANT,
    SVCREATE2,
    SVCREATE4,
};

class Type
{
public:
    enum TYPE_ID
    {
        TYPE_VOID,
        TYPE_FP32,
        TYPE_SVFP32,
        TYPE_FP64,
        TYPE_SVFP64,
        TYPE_SVFP32x2,
        TYPE_SVFP32x4,
        TYPE_SVFP64x2,
        TYPE_SVFP64x4,
        TYPE_PTR
    };
    TYPE_ID type_id;
    Type *ptr_dst = nullptr;

    Type *get_sv_type()
    {
        if (is_svtype())
        {
            return this;
        }
        if (type_id == TYPE_FP32)
        {
            return getSVFP32Type();
        }
        else if (type_id == TYPE_FP64)
        {
            return getSVFP64Type();
        }
        else
        {
            throw std::runtime_error("Unsupported type for get_sv_type");
        }
    }
    int bits()
    {
        if (is_svtype())
        {
            return 512;
        }
        else if (type_id == TYPE_FP32)
        {
            return 32;
        }
        else if (type_id == TYPE_FP64)
        {
            return 64;
        }
        else
        {
            throw std::runtime_error("Unsupported type for bits");
        }
    }
    bool is_fp64()
    {
        return type_id == TYPE_FP64 || type_id == TYPE_SVFP64 || type_id == TYPE_SVFP64x2 || type_id == TYPE_SVFP64x4;
    }

    bool is_svtype() { return type_id == TYPE_SVFP32 || type_id == TYPE_SVFP64; }

    static Type *getVoidType()
    {
        static Type voidType = {TYPE_VOID};
        return &voidType;
    }
    static Type *getFP32Type()
    {
        static Type fp32Type = {TYPE_FP32};
        return &fp32Type;
    }
    static Type *getFP64Type()
    {
        static Type fp64Type = {TYPE_FP64};
        return &fp64Type;
    }
    static Type *getSVFP32Type()
    {
        static Type svfp32Type = {TYPE_SVFP32};
        return &svfp32Type;
    }
    static Type *getSVFP64Type()
    {
        static Type svfp64Type = {TYPE_SVFP64};
        return &svfp64Type;
    }
    static Type *getSVFP32x2Type()
    {
        static Type svfp32x2Type = {TYPE_SVFP32x2};
        return &svfp32x2Type;
    }
    static Type *getSVFP32x4Type()
    {
        static Type svfp32x4Type = {TYPE_SVFP32x4};
        return &svfp32x4Type;
    }
    static Type *getSVFP64x2Type()
    {
        static Type svfp64x2Type = {TYPE_SVFP64x2};
        return &svfp64x2Type;
    }
    static Type *getSVFP64x4Type()
    {
        static Type svfp64x4Type = {TYPE_SVFP64x4};
        return &svfp64x4Type;
    }
    static Type *getType(TYPE_ID type_id)
    {
        switch (type_id)
        {
        case TYPE_FP32:
            return getFP32Type();
        case TYPE_FP64:
            return getFP64Type();
        case TYPE_SVFP32:
            return getSVFP32Type();
        case TYPE_SVFP64:
            return getSVFP64Type();
        case TYPE_SVFP32x2:
            return getSVFP32x2Type();
        case TYPE_SVFP32x4:

            return getSVFP32x4Type();
        case TYPE_SVFP64x2:
            return getSVFP64x2Type();
        case TYPE_SVFP64x4:
            return getSVFP64x4Type();
        default:
            throw std::runtime_error("Unsupported TYPE_ID for Type");
        }
    }
    static Type *getType(TilePrimitiveDescriptor::DTYPE dtype)
    {
        switch (dtype)
        {
        case TilePrimitiveDescriptor::DTYPE::DTYPE_FP32:
            return getFP32Type();
        case TilePrimitiveDescriptor::DTYPE::DTYPE_FP64:
            return getFP64Type();
        default:
            throw std::runtime_error("Unsupported DTYPE for Type");
        }
    }
    static Type *getSVType(TilePrimitiveDescriptor::DTYPE dtype)
    {
        switch (dtype)
        {
        case TilePrimitiveDescriptor::DTYPE::DTYPE_FP32:
            return getSVFP32Type();
        case TilePrimitiveDescriptor::DTYPE::DTYPE_FP64:
            return getSVFP64Type();
        default:
            throw std::runtime_error("Unsupported DTYPE for SV Type");
        }
    }
};
inline size_t za_num(Type *type)
{
    switch (type->type_id)
    {
    case Type::TYPE_SVFP32:
    case Type::TYPE_SVFP32x2:
    case Type::TYPE_SVFP32x4:
    case Type::TYPE_FP32:
        return 4; // Assuming 4 ZA registers for FP32
    case Type::TYPE_SVFP64:
    case Type::TYPE_SVFP64x2:
    case Type::TYPE_SVFP64x4:
    case Type::TYPE_FP64:
        return 8; // Assuming 8 ZA registers for FP64
    default:
        throw std::runtime_error("Unsupported type for za_num");
    }
}
inline size_t svl(Type *type)
{
    switch (type->type_id)
    {
    case Type::TYPE_SVFP32:
        return 16; // Assuming 512-bit SVE vector
    case Type::TYPE_SVFP64:
        return 8; // Assuming 512-bit SVE vector
    default:
        throw std::runtime_error("Unsupported type for svl");
    }
}
inline size_t svl(TilePrimitiveDescriptor::DTYPE dtype)
{
    switch (dtype)
    {
    case TilePrimitiveDescriptor::DTYPE::DTYPE_FP32:
        return 16; // Assuming 512-bit SVE vector
    case TilePrimitiveDescriptor::DTYPE::DTYPE_FP64:
        return 8; // Assuming 512-bit SVE vector
    default:
        throw std::runtime_error("Unsupported DTYPE for svl");
    }
}
class Instruction
{
protected:
    Instruction(OPCODE opcode, Type *type)
    : _opcode(opcode)
    , type(type)
    {
    }

public:
    OPCODE opcode() const { return _opcode; }

    Type *type;

private:
    OPCODE _opcode;
};

class ZA;
class Primitive
{
    friend class CGenerator;
    friend class AGenerator;
    std::vector<Instruction *> instructions;
    std::vector<ZA *> zas;
    int k_unroll = 1;

public:
    void insert(Instruction *inst) { instructions.push_back(inst); }
    void insert(ZA *za) { zas.push_back(za); }
    ZA *getZA(int idx) const { return zas.at(idx); }
    auto &ZAs() { return zas; }
    auto &insts() { return instructions; }
    void set_k_unroll(int unroll) { k_unroll = unroll; }
    int get_k_unroll() const { return k_unroll; }
};

template <typename T>
bool isa(Instruction *inst)
{
    return inst->opcode() == T::StaticOpcode;
}
template <typename T, typename T2, typename... Ts>
bool isa(Instruction *inst)
{
    return isa<T>(inst) || isa<T2, Ts...>(inst);
}
template <typename T>
T *dyn_cast(Instruction *inst)
{
    static_assert(std::is_base_of_v<Instruction, T>, "T must be derived from Instruction");
    if (isa<T>(inst))
    {
        return static_cast<T *>(inst);
    }
    return nullptr;
}

class ZA
{
    ZA() { }

public:
    int id = 0;
    static ZA *create(Primitive *primitive)
    {
        auto za = new ZA();
        primitive->insert(za);
        return za;
    }
    void setId(int _id) { id = _id; }
};
class Predicate
{
    Predicate(int64_t active)
    : active(active)
    {
    }

public:
    const int64_t active;
    static Predicate ptrue_d() { return Predicate(0x0101010101010101LL); }
    static Predicate ptrue_s() { return Predicate(0x1111111111111111LL); }
    static Predicate ptrue(Type *type)
    {
        if (type->type_id == Type::TYPE_SVFP32)
        {
            return ptrue_s();
        }
        else if (type->type_id == Type::TYPE_SVFP64)
        {
            return ptrue_d();
        }
        else
        {
            throw std::runtime_error("Unsupported type for Predicate ptrue");
        }
    }
    static Predicate range(int start, int end, Type *type)
    {
        end -= (start / svl(type)) * svl(type);
        start %= svl(type);
        int shift_step = 0;
        if (type->type_id == Type::TYPE_SVFP32)
        {
            shift_step = 4;
        }
        else if (type->type_id == Type::TYPE_SVFP64)
        {
            shift_step = 8;
        }
        else
        {
            throw std::runtime_error("Unsupported type for Predicate range");
        }
        int64_t mask = 0;
        for (int i = start; i < end; ++i)
        {
            mask |= (1LL << (i * shift_step));
        }
        return Predicate(mask);
    }
};
class MopaInst : public Instruction
{
public:
    Instruction *va;
    Instruction *vb;
    ZA *za;
    Predicate pa;
    Predicate pb;

private:
    MopaInst(Instruction *va, Instruction *vb, ZA *za, Predicate pa, Predicate pb)
    : va(va)
    , vb(vb)
    , za(za)
    , pa(pa)
    , pb(pb)
    , Instruction(StaticOpcode, Type::getVoidType())
    {
    }

public:
    static const OPCODE StaticOpcode = OPCODE::MOPA;
    static MopaInst *
    create(Instruction *va, Instruction *vb, ZA *za, Predicate pa, Predicate pb, Primitive *primitive = nullptr)
    {
        auto res = new MopaInst(va, vb, za, pa, pb);
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }
};
class MlaSMEInst : public Instruction
{
public:
    enum MODE
    {
        VG1x4,
        VG1x2,
        VG2x2,
        VG4x4
    };
    static const OPCODE StaticOpcode = OPCODE::MLA_SME;

    ZA *za;
    int lane;
    Instruction *a;
    Instruction *b;

private:
    MlaSMEInst(ZA *za, int lane, Instruction *a, Instruction *b)
    : za(za)
    , lane(lane)
    , a(a)
    , b(b)
    , Instruction(StaticOpcode, Type::getVoidType())
    {
    }

public:
    static MlaSMEInst *create(ZA *za, int lane, Instruction *a, Instruction *b, Primitive *primitive = nullptr)
    {
        auto res = new MlaSMEInst(za, lane, a, b);
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }
};
class MlaScalarInst : public Instruction
{
    MlaScalarInst(Instruction *a, Instruction *b, Instruction *c)
    : a(a)
    , b(b)
    , c(c)
    , Instruction(StaticOpcode, c->type)
    {
    }

public:
    Instruction *a;
    Instruction *b;
    Instruction *c;
    static const OPCODE StaticOpcode = OPCODE::MLA_SCALAR;
    static MlaScalarInst *create(Instruction *c, Instruction *a, Instruction *b, Primitive *primitive = nullptr)
    {
        auto res = new MlaScalarInst(a, b, c);
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }
};
class MlaSVEInst : public Instruction
{
    MlaSVEInst(Predicate predicate, Instruction *a, Instruction *b, Instruction *c)
    : predicate(predicate)
    , a(a)
    , b(b)
    , c(c)
    , Instruction(StaticOpcode, c->type)
    {
    }

public:
    Predicate predicate;
    Instruction *a;
    Instruction *b;
    Instruction *c;
    static const OPCODE StaticOpcode = OPCODE::MLA_SVE;
    static MlaSVEInst *
    create(Predicate pg, Instruction *c, Instruction *a, Instruction *b, Primitive *primitive = nullptr)
    {
        auto res = new MlaSVEInst(pg, a, b, c);
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }
};
class LoadSVEInst : public Instruction
{
    LoadSVEInst(Type *type, Predicate predicate, MemoryTarget target, int column, int batch, int k_offset)
    : predicate(predicate)
    , target(target)
    , column(column)
    , batch(batch)
    , k_offset(k_offset)
    , Instruction(StaticOpcode, type)
    {
    }

public:
    Predicate predicate;
    MemoryTarget target;
    int column;
    int batch;
    int k_offset;
    static const OPCODE StaticOpcode = OPCODE::LOAD_SVE;
    static LoadSVEInst *create(Type *type,
                               Predicate predicate,
                               MemoryTarget target,
                               int column,
                               int step,
                               int batch,
                               int k_offset = 0,
                               Primitive *primitive = nullptr)
    {
        MY_ASSERT(type->is_svtype() && "LoadSVEInst only supports SVE types");
        auto res = new LoadSVEInst(type, predicate, target, column, batch, k_offset);
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }

    static LoadSVEInst *
    create(Type *type, Predicate predicate, MemoryTarget target, int column, int step, int batch, Primitive *primitive)
    {
        return create(type, predicate, target, column, step, batch, 0, primitive);
    }
};
class KLoopBeginInst : public Instruction
{
    KLoopBeginInst(int step)
    : step(step)
    , Instruction(StaticOpcode, Type::getVoidType())
    {
    }

public:
    int step;
    static const OPCODE StaticOpcode = OPCODE::KLOOP_BEGIN;
    static KLoopBeginInst *create(Primitive *primitive = nullptr, int step = 1)
    {
        auto res = new KLoopBeginInst(step);
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }
    void set_step(int new_step) { step = new_step; }
};

class KLoopEndInst : public Instruction
{
    KLoopEndInst()
    : Instruction(StaticOpcode, Type::getVoidType())
    {
    }

public:
    static const OPCODE StaticOpcode = OPCODE::KLOOP_END;
    static KLoopEndInst *create(Primitive *primitive = nullptr)
    {
        auto res = new KLoopEndInst();
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }
};
class StoreSVEInst : public Instruction
{
public:
    int row;
    int column;
    int batch;
    Predicate predicate;
    Instruction *val;

private:
    StoreSVEInst(Predicate predicate, Instruction *val, int row, int column, int batch)
    : row(row)
    , column(column)
    , predicate(predicate)
    , val(val)
    , batch(batch)
    , Instruction(StaticOpcode, Type::getVoidType())
    {
    }

public:
    static const OPCODE StaticOpcode = OPCODE::STORE_SVE;
    static StoreSVEInst *
    create(Predicate predicate, Instruction *val, int row, int column, int batch, Primitive *primitive = nullptr)
    {
        auto res = new StoreSVEInst(predicate, val, row, column, batch);
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }
};
class LoadInst : public Instruction
{
public:
    MemoryTarget target;
    int column;
    int batch;

private:
    LoadInst(MemoryTarget target, int column, int batch, Type *type)
    : target(target)
    , column(column)
    , batch(batch)
    , Instruction(StaticOpcode, type)
    {
    }

public:
    static const OPCODE StaticOpcode = OPCODE::LOAD;
    static LoadInst *create(Type *type, MemoryTarget target, int column, int batch, Primitive *primitive = nullptr)
    {
        MY_ASSERT(!type->is_svtype() && "LoadInst only supports non-SVE types");
        auto res = new LoadInst(target, column, batch, type);
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }
};
class StoreInst : public Instruction
{
    StoreInst(Instruction *val, int row, int column, int batch)
    : row(row)
    , column(column)
    , val(val)
    , batch(batch)
    , Instruction(StaticOpcode, Type::getVoidType())
    {
    }

public:
    int row;
    int column;
    int batch;
    Instruction *val;
    static const OPCODE StaticOpcode = OPCODE::STORE;
    static StoreInst *create(Instruction *val, int row, int column, int batch, Primitive *primitive = nullptr)
    {
        auto res = new StoreInst(val, row, column, batch);
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }
};
class SelInst : public Instruction
{
    SelInst(Predicate predicate, Instruction *trueValue, Instruction *falseValue)
    : predicate(predicate)
    , trueValue(trueValue)
    , falseValue(falseValue)
    , Instruction(StaticOpcode, trueValue->type)
    {
    }

public:
    static const OPCODE StaticOpcode = OPCODE::SEL;
    Predicate predicate;
    Instruction *trueValue;
    Instruction *falseValue;
    static SelInst *
    Create(Predicate predicate, Instruction *trueValue, Instruction *falseValue, Primitive *primitive = nullptr)
    {
        auto res = new SelInst(predicate, trueValue, falseValue);
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }
};
class ZAInstruction : public Instruction
{
public:
    ZAInstruction(OPCODE opcode, Type *type)
    : Instruction(opcode, type)
    {
    }

    enum DIRECTION
    {
        VERTICAL,
        HORIZONTAL
    };
};
class ReadZAInst : public ZAInstruction
{
public:
    ZA *za;
    int lane;
    DIRECTION direction;
    Predicate predicate;
    Instruction *source;

    ReadZAInst(ZA *za, int lane, DIRECTION direction, Predicate predicate, Type *type, Instruction *source)
    : za(za)
    , lane(lane)
    , direction(direction)
    , predicate(predicate)
    , source(source)
    , ZAInstruction(StaticOpcode, type)
    {
    }

public:
    static const OPCODE StaticOpcode = OPCODE::READ_ZA;
    static ReadZAInst *create(ZA *za,
                              int lane,
                              DIRECTION direction,
                              Predicate predicate,
                              Type *type,
                              Instruction *source,
                              Primitive *primitive = nullptr)
    {
        auto res = new ReadZAInst(za, lane, direction, predicate, type, source);
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }
};
class WriteZAInst : public ZAInstruction
{
public:
    using DIRECTION = ReadZAInst::DIRECTION;
    ZA *za;
    int lane;
    DIRECTION direction;
    Predicate predicate;
    Instruction *source;

    WriteZAInst(ZA *za, int lane, DIRECTION direction, Predicate predicate, Type *type, Instruction *source)
    : za(za)
    , lane(lane)
    , direction(direction)
    , predicate(predicate)
    , source(source)
    , ZAInstruction(StaticOpcode, type)
    {
    }

public:
    static const OPCODE StaticOpcode = OPCODE::WRITE_ZA;
    static WriteZAInst *create(ZA *za,
                               int lane,
                               DIRECTION direction,
                               Predicate predicate,
                               Type *type,
                               Instruction *source,
                               Primitive *primitive = nullptr)
    {
        auto res = new WriteZAInst(za, lane, direction, predicate, type, source);
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }
};

class SVDupMInst : public Instruction
{
    SVDupMInst(Instruction *val, Predicate predicate, Instruction *ori, Type *type)
    : val(val)
    , predicate(predicate)
    , ori(ori)
    , Instruction(StaticOpcode, type)
    {
    }

public:
    Instruction *val;
    Predicate predicate;
    Instruction *ori;
    static const OPCODE StaticOpcode = OPCODE::SVDUPM;
    static SVDupMInst *create(Instruction *val, Predicate predicate, Instruction *ori, Primitive *primitive = nullptr)
    {
        auto res = new SVDupMInst(val, predicate, ori, val->type->get_sv_type());
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }
};
class SVDupInst : public Instruction
{
    SVDupInst(Instruction *val, Type *type)
    : val(val)
    , Instruction(StaticOpcode, type)
    {
    }

public:
    Instruction *val;
    static const OPCODE StaticOpcode = OPCODE::SVDUP;
    static SVDupInst *create(Instruction *val, Primitive *primitive = nullptr)
    {
        auto res = new SVDupInst(val, val->type->get_sv_type());
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }
};
class Constant : public Instruction
{
    Constant(double val, Type *type, bool is_accumulator = false)
    : val(val)
    , is_accumulator(is_accumulator)
    , Instruction(StaticOpcode, type)
    {
    }

public:
    double val;
    bool is_accumulator;
    static const OPCODE StaticOpcode = OPCODE::CONSTANT;
    static Constant *create(double val, Type *type, Primitive *primitive = nullptr, bool is_accumulator = false)
    {
        auto res = new Constant(val, type, is_accumulator);
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }
};
class SVCreate4Inst : public Instruction
{
    SVCreate4Inst(Instruction *v0, Instruction *v1, Instruction *v2, Instruction *v3, Type *type)
    : v0(v0)
    , v1(v1)
    , v2(v2)
    , v3(v3)
    , Instruction(StaticOpcode, type)
    {
    }

public:
    Instruction *v0;
    Instruction *v1;
    Instruction *v2;
    Instruction *v3;
    static const OPCODE StaticOpcode = OPCODE::SVCREATE4;
    static SVCreate4Inst *
    create(Instruction *v0, Instruction *v1, Instruction *v2, Instruction *v3, Primitive *primitive = nullptr)
    {
        Type *type = nullptr;
        if (v0->type->type_id == Type::TYPE_SVFP32)
        {
            type = Type::getType(Type::TYPE_SVFP32x4);
        }
        else if (v0->type->type_id == Type::TYPE_SVFP64)
        {
            type = Type::getType(Type::TYPE_SVFP64x4);
        }
        else
        {
            throw std::runtime_error("Unsupported type for SVCreate4Inst");
        }
        auto res = new SVCreate4Inst(v0, v1, v2, v3, type);
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }
};

class SVCreate2Inst : public Instruction
{
    SVCreate2Inst(Instruction *v0, Instruction *v1, Type *type)
    : v0(v0)
    , v1(v1)
    , Instruction(StaticOpcode, type)
    {
    }

public:
    Instruction *v0;
    Instruction *v1;
    static const OPCODE StaticOpcode = OPCODE::SVCREATE2;
    static SVCreate2Inst *create(Instruction *v0, Instruction *v1, Primitive *primitive = nullptr)
    {
        Type *type = nullptr;
        if (v0->type->type_id == Type::TYPE_SVFP32)
        {
            type = Type::getType(Type::TYPE_SVFP32x2);
        }
        else if (v0->type->type_id == Type::TYPE_SVFP64)
        {
            type = Type::getType(Type::TYPE_SVFP64x2);
        }
        else
        {
            throw std::runtime_error("Unsupported type for SVCreate2Inst");
        }
        auto res = new SVCreate2Inst(v0, v1, type);
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }
};

class SVUndefInst : public Instruction
{
    SVUndefInst(Type *type)
    : Instruction(StaticOpcode, type)
    {
    }

public:
    static const OPCODE StaticOpcode = OPCODE::SVUNDEF;
    static SVUndefInst *create(Type *type, Primitive *primitive = nullptr)
    {
        auto res = new SVUndefInst(type);
        if (primitive)
        {
            primitive->insert(res);
        }
        return res;
    }
};

class Function
{
    friend class CGenerator;
    friend class AGenerator;
    std::vector<Primitive *> primitives;
    void buildMopa(TilePrimitiveDescriptor &desc);
    void buildMlaSVE(TilePrimitiveDescriptor &desc);
    void buildMlaLaneSVE(TilePrimitiveDescriptor &desc);
    void buildMlaSME(TilePrimitiveDescriptor &desc);
    void buildScalar(TilePrimitiveDescriptor &desc);
    int batch_per_step;
    TilePrimitiveDescriptor::DTYPE dtype;
    TilePrimitiveDescriptor::TRANS_TYPE trans_type = TilePrimitiveDescriptor::TRANS_TYPE::UNDEF;

public:
    Function(int M,
             int N,
             int batch_per_step,
             TilePrimitiveDescriptor::DTYPE dtype,
             TilePrimitiveDescriptor::TRANS_TYPE trans_type)
    : M(M)
    , N(N)
    , batch_per_step(batch_per_step)
    , dtype(dtype)
    , trans_type(trans_type)
    {
    }
    void build(TilePrimitiveDescriptor &desc);
    int M;
    int N;

    int batch() const { return batch_per_step; }
    void allocate_za();
    void kLoopMerge();
    void rearrange();
    TilePrimitiveDescriptor::TRANS_TYPE getTransType() const { return trans_type; }
    TilePrimitiveDescriptor::DTYPE getDtype() const { return dtype; }
};

struct LoweredAResult
{
    std::string asm_text;
    std::string inst_text;
    std::vector<std::uint32_t> binary;

    void write_binary_to(void *dst) const
    {
        if (dst == nullptr || binary.empty())
        {
            return;
        }
        std::memcpy(dst, binary.data(), binary.size() * sizeof(std::uint32_t));
    }
};

std::string LowerToC(Function &func);
LoweredAResult LowerToA(Function &func, void *binary_addr = nullptr);
} // namespace IR
