#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "IR.h"
#include "descriptor.h"

llvm::Function *lowerToLLVM(IR::Function &func)
{
    llvm::LLVMContext context;
    llvm::Function *llvmFunc = llvm::Function::Create(
        llvm::FunctionType::get(
            llvm::Type::getVoidTy(context),
            {
                llvm::PointerType::getUnqual(llvm::PointerType::getUnqual(llvm::Type::getDoubleTy(context))), // A
                llvm::PointerType::getUnqual(llvm::PointerType::getUnqual(llvm::Type::getDoubleTy(context))), // B
                llvm::PointerType::getUnqual(llvm::PointerType::getUnqual(llvm::Type::getDoubleTy(context))), // C
                llvm::Type::getInt32Ty(context),                                                              // batch
                llvm::Type::getInt32Ty(context)                                                               // K
            },
            false),
        llvm::Function::ExternalLinkage,
        "gemm_kernel_opt");
    llvm::BasicBlock *entry = llvm::BasicBlock::Create(context, "entry", llvmFunc);
    llvm::IRBuilder<> builder(entry);

    return llvmFunc;
}