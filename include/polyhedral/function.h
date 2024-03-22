#ifndef _H_polyfp_function_
#define _H_polyfp_function_

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

#include "scalehls/Transforms/Passes.h"
#include "scalehls/Transforms/Utils.h"
#include "scalehls/Transforms/Estimator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MemoryBuffer.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/BuiltinOps.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

#include "expr.h"
#include "type.h"
#include "codegen.h"
#include "generator_isl.h"
#include "placeholder.h"

namespace polyfp{


class constant;
class compute;
class generator;
class placeholder;

isl_ast_node *for_code_generator_after_for(isl_ast_node *node, isl_ast_build *build, void *user);
void gen_mlir(polyfp::function *fct, isl_ast_node *node, int level);


class function{

    friend constant;
    friend compute;
    friend generator;
    friend placeholder;

private:

    std::string name;
    std::vector<polyfp::constant> invariants;
    std::map<std::string, polyfp::constant *> constant_list;
    std::vector<polyfp::placeholder *> function_arguments;

    std::map<std::string, polyfp::placeholder *> placeholders_list;
    std::map<std::string, polyfp::placeholder *> fct_argument_list;
    std::map<std::string, polyfp::placeholder *> global_argument_list;
    bool fct_argument_added = false;

    std::vector<std::tuple<std::string, std::vector<int>, std::vector<std::string>>> partition_map;

    // The isl context of the function.
    isl_ctx *ctx;

      
    // The isl AST generated by gen_isl_ast().
    isl_ast_node *ast;

    // Contains all the computes of the function
    std::vector<compute *> body;

    /**
     * TODO: Extend
     * Derived from Tiramisu:
     * The context set of the function, i.e. a set representing the
     * constraints over the parameters.
     * The parameters of a function are the function invariants (constants).
     */
    isl_set *context_set;

    std::vector<std::string> iterator_names;

    isl_union_set *get_trimmed_time_processor_domain() const;

    /**
      * Derived from Tiramisu:
      * This function iterates over the computes of the function.
      * It modifies the identity schedule of each computes in order to
      * make all the identity schedules have the same number of dimensions
      * in their ranges.
      * This is done by adding dimensions equal to 0 to the range of each
      * identity schedule that does not have enough dimensions.
      */
    isl_union_map *get_aligned_identity_schedules() const;

    /**
      * Derived from Tiramisu:
      * This function first computes the identity schedules,
      * then it computes the maximal dimension among the dimensions
      * of the ranges of all the identity schedules.
      */
    int get_max_identity_schedules_range_dim() const;

    void rename_computations();
  
    // Recursive function to perform the DFS step of dump_sched_graph.
    void dump_sched_graph_dfs(polyfp::compute *,
                              std::unordered_set<polyfp::compute *> &);

    // Recursive function to perform the DFS step of is_sched_graph_tree.
    bool is_sched_graph_tree_dfs(polyfp::compute *,
                                 std::unordered_set<polyfp::compute *> &);

protected:

    void dfs(int pos, int top, int end, int map[500][500], int n, int v[500],int stack[500]);
    polyfp::compute * update_latency();
    int get_longest_path();
    int get_longest_node(std::vector<long> path);
    void add_computation(compute *cpt);

    void add_invariant(std::pair<std::string, polyfp::constant *>  param);
    void add_placeholder(std::pair<std::string, polyfp::placeholder *> buf);

    const std::vector<std::string> &get_iterator_names() const;

    // void add_iterator_name(const std::string &it_name);

    const std::vector<compute *> &get_computations() const;

    /** TODO: remove
      * Derived from Tiramisu:
      * Return a set that represents the parameters of the function
      * (an ISL set that represents the parameters and constraints over
      * the parameters of the functions,  a parameter is an invariant
      * of the function). This set is also known as the context of
      * the program.
      * An example of a context set is the following:
      *          "[N,M]->{: M>0 and N>0}"
      * This context set indicates that the two parameters N and M
      * are strictly positive.
      */
    isl_set *get_program_context() const;

    std::vector<compute *> get_computation_by_name(std::string str) const;

    isl_ctx *get_isl_ctx() const;

    /**
      * Return the union of all the schedules
      * of the compute of the function.
      */
    isl_union_map *get_schedule() const;

    /**
      * Return the union of all the iteration domains
      * of the computes of the function.
      */
    isl_union_set *get_iteration_domain() const;

    /**
     * Return true if the usage of high level scheduling comments is valid; i.e. if
     * the scheduling relations formed using before, after, compute_at, etc.. form a tree.
     *
     * More specifically, it verifies that:
     *     - There should be exactly one compute with no compute scheduled before it.
     *     - Each other compute should have exactly one compute scheduled before it.
     */
    bool is_sched_graph_tree();

    /**
     * Modify the schedules of the computes of this function to reflect
     * the order specified using the high level scheduling commands.
     *
     * Commands like .after() do not directly modify the schedules
     * but rather modify the sched_graph.
     */
    void gen_ordering_schedules();

    /**
     * This functions iterates over the schedules of the function (the schedule
     * of each compute in the function) and computes the maximal dimension
     * among the dimensions of the ranges of all the schedules.
     */
    int get_max_schedules_range_dim() const;

    /**
      * Stores all high level scheduling instructions between computes; i.e. if a user calls
      * for example c2.after(c1, L), sched_graph[&c1] would contain the key &c2, and
      * sched_graph[&c1][&c2] = L.
      */
    std::unordered_map<polyfp::compute *,std::unordered_map<polyfp::compute *, int>> sched_graph;

    std::unordered_map<polyfp::compute *,
    std::unordered_map<polyfp::compute *, int>> sched_graph_reversed;

    /**
      * Return an ISL AST that represents this function.
      * The function gen_isl_ast() should be called before calling
      * this function.
      */
    isl_ast_node *get_isl_ast() const;
  

    // Generate a mlir stmt that represents the function.
    void gen_mlir_stmt();

public:

    bool is_dataflowed = false;
    void evaluate_func();
    std::unordered_set<polyfp::compute *> starting_computations;
    std::vector<polyfp::compute *> leader_computations;
    std::vector<polyfp::compute *> leaf_computations;
    std::map<polyfp::compute *,int> leader_computation_index;
    std::map<int,long> latency_map;
    std::map<int,long> all_latency_map;
    std::map<int,int> resource_map;
    std::map<int,std::vector<int>> path_map;
    std::vector<std::vector<long>> paths;
    std::vector<std::string> finish_list;
    bool consistent_flag = true;
    bool refused = false;


    void add_fct_argument(std::pair<std::string, polyfp::placeholder *> buf);

    void add_fct_argument();

    void add_global_argument(std::pair<std::string, polyfp::placeholder *> buf);

    void check_loop_fusion();

    int get_global_location(){
        return global_location;
    }

    void set_global_location(int new_location){
        this->global_location = new_location;
    }

    void dump_schedule(std::string path);

    long longest_path;
    long longest_node;
    long dsp_max;
    long dsp_usage;
    long best_dsp_usage = dsp_max;
    long best_latency;
    long current_latency;
    bool new_strategy = true;
    polyfp::compute * current_opt_comp;


    int global_location;
    bool one_compute;

    function(std::string name);

    /**
      * Derived from Tiramisu:
      * This method applies to the schedule of each compute
      * in the function.  It makes the dimensions of the ranges of
      * all the schedules equal.  This is done by adding dimensions
      * equal to 0 to the range of each schedule.
      * This function is called automatically when gen_isl_ast()
      * or gen_time_processor_domain() are called.
      */
    void align_schedules();

    const std::vector<compute *> &get_body() const;

    const std::map<std::string, polyfp::placeholder *> &get_placeholders() const;
    const std::map<std::string, polyfp::placeholder *> &get_fct_arguments() const;
    const std::map<std::string, polyfp::placeholder *> &get_global_arguments() const;

    const std::map<std::string, polyfp::constant *> &get_invariants() const;
    const std::vector<std::string> get_invariant_names() const;

    std::vector<std::tuple<std::string, std::vector<int>, std::vector<std::string>>> get_partition_map();

    void set_partition(std::string name, std::vector<int> tile_factors, std::vector<std::string> types);
    
    void dump_sched_graph();

    isl_ast_node *get_isl_ast1() const;


    /**
      * Compute the graph of dependences between the computes of the function. 
      * C[0] = 0
      * D[1] = C[0]
      * D[2] = C[0]
      * {C[0] -> D[1]; C[0]->D[2]}
      */
    isl_union_map *compute_dep_graph();
    void gen_isl_ast();

    /**
      * Generate the time-space domain of the function.
      *
      * In this representation, the logical time of execution and the
      * processor where the compute will be executed are both
      * specified.
      */
    void gen_time_space_domain();

    void gen_loop_location();
    std::string get_name();

    void collect_accesses();
    std::map<int, std::map<std::string, std::vector<polyfp::expr> > > map_loadstores;

    void codegen();
    void auto_DSE_loop_transformation();
    void auto_DSE(std::string path);
    void auto_DSE_tile_size(polyfp::compute* comp, int factor,std::string path);
    void dependence_analysis();
    void compute_dependency_graph();

    /**
      * Dump the function on standard output (dump most of the fields of
      * polyfp::function).This is mainly useful for debugging.
      */
    void dump(bool exhaustive) const;
    void gen_c_code() const;

    void trans();


};

}

#endif