#include <isl/aff.h>
#include <isl/set.h>
#include <isl/constraint.h>
#include <isl/space.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <string>
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include <numeric>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/space.h>
#include <isl/constraint.h>

#include <map>
#include <string.h>
#include <stdint.h>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <queue>
#include <variant>
#include "expr.h"
#include "type.h"
#include "codegen.h"
#include "function.h"

using llvm::SmallVector;
using llvm::ArrayRef;
namespace polyfp{
class function;
class MLIRGenImpl {
friend function;
friend compute;

private:
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;
    
public:
  
    MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) 
    {
        theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    }

    mlir::ModuleOp mlirGen1(const polyfp::function &fct, isl_ast_node *isl_node, int &level, bool flag, bool flag2, bool if_flag);

    //contains all loops 
    std::vector<mlir::AffineForOp> ops;
    std::vector<int> start_loops_position;


    // std::map<std::string, mlir::Type > argument_list;
    std::vector<std::string> argument_list;
    std::map<std::string,mlir::Value> argument_map;
    std::map<std::string,int> array_map;
    std::map<std::string,mlir::Value> get_argument_map();
    std::map<std::string,int> get_array_map();
    std::vector<mlir::Value> values;
    
    std::vector<mlir::memref::AllocOp> allocs;
    std::vector<mlir::FuncOp> funcs;
    std::vector<mlir::FuncOp> get_funcs();
    std::map<int, std::string > name_map;

    
    mlir::ModuleOp getModule();

    void a_print_expr(polyfp::expr polyfp_expr, polyfp::compute *comp, int level);

    // std::vector<mlir::Value> index_values;
    // SmallVector<mlir::AffineExpr> index_args;
    int get_iterator_location_from_name(polyfp::compute *comp,polyfp::expr polyfp_expr, std::vector<mlir::Value> &index_values);
    mlir::AffineExpr a_print_index(polyfp::expr polyfp_expr, polyfp::compute *comp, std::vector<mlir::Value> &index_values,int level);
    // std::vector<mlir::arith::AddFOp> add_op;
    // // std::vector<mlir::Op<>> sum_op;
    // std::vector<mlir::arith::MulFOp> mul_op;
    

    // std::vector<mlir::arith::AddFOp> all_add_op;
    // std::vector<mlir::arith::MulFOp> all_mul_op;
    using value = std::variant<mlir::arith::AddFOp, mlir::arith::MulFOp,mlir::arith::SubFOp,mlir::arith::DivFOp,mlir::arith::MaxFOp>;
    std::vector<value> current_op;
    std::vector<value> all_current_op;

    using AffineLoopBand = SmallVector<mlir::AffineForOp, 6>;
    using TileList = SmallVector<unsigned, 8>;

};
}