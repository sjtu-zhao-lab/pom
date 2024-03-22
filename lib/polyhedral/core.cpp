#include <isl/ctx.h>
#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/id.h>
#include <isl/constraint.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>

#include "core.h"

#ifdef _WIN32
#include <iso646.h>
#endif

namespace polyfp
{
// Used for the generation of new variable names.
int id_counter = 0;
static int next_dim_name = 0;

primitive_t global::loop_iterator_type = p_int32;
function *global::implicit_fct;
std::unordered_map<std::string, var> var::declared_vars;

std::string generate_new_variable_name();

polyfp::expr traverse_expr_and_replace_non_affine_accesses(polyfp::compute *comp,
                                                             const polyfp::expr &exp);

void init(std::string fct_name)
{
    function *fct = new function(fct_name);
    global::set_implicit_function(fct);
    init();
}


void init()
{
    global::set_default_polyfp_options();
}

void codegen()
{
    function *fct = global::get_implicit_function();
    fct->codegen();
}



/**
 * Derived from Tiramisu:
 * Transform the loop level into its corresponding dynamic schedule
 * dimension.
 *
 * In the example below, the dynamic dimension that corresponds
 * to the loop level 0 is 2, and to 1 it is 4, ...
 *
 * The first dimension is the duplication dimension, the following
 * dimensions are static, dynamic, static, dynamic, ...
 *
 * Loop level               :    -1         0      1      2
 * Schedule dimension number:        0, 1   2  3   4  5   6  7
 * Schedule:                        [0, 0, i1, 0, i2, 0, i3, 0]
 */
int loop_level_into_dynamic_dimension(int level)
{
    return 1 + (level * 2 + 1);
}

/**
 * Derived from Tiramisu:
 * Transform the loop level into the first static schedule
 * dimension after its corresponding dynamic dimension.
 *
 * In the example below, the first static dimension that comes
 * after the corresponding dynamic dimension for
 * the loop level 0 is 3, and to 1 it is 5, ...
 *
 * Loop level               :    -1         0      1      2
 * Schedule dimension number:        0, 1   2  3   4  5   6  7
 * Schedule:                        [0, 0, i1, 0, i2, 0, i3, 0]
 */
int loop_level_into_static_dimension(int level)
{
    return loop_level_into_dynamic_dimension(level) + 1;
}

/**
 * Derived from Tiramisu:
 * Transform a dynamic schedule dimension into the corresponding loop level.
 *
 * In the example below, the loop level that corresponds
 * to the dynamic dimension 2 is 0, and to the dynamic dimension 4 is 1, ...
 *
 * The first dimension is the duplication dimension, the following
 * dimensions are static, dynamic, static, dynamic, ...
 *
 * Loop level               :    -1         0      1      2
 * Schedule dimension number:        0, 1   2  3   4  5   6  7
 * Schedule:                        [0, 0, i1, 0, i2, 0, i3, 0]
 */
int dynamic_dimension_into_loop_level(int dim)
{
    assert(dim % 2 == 0);
    int level = (dim - 2)/2;
    return level;
}

std::string generate_new_variable_name()
{
    return "t" + std::to_string(id_counter++);
}

std::string generate_new_computation_name()
{
    return "C" + std::to_string(id_counter++);
}

std::string str_from_polyfp_type_expr(polyfp::expr_t type)
{
    switch (type)
    {
    case polyfp::e_val:
        return "val";
    case polyfp::e_op:
        return "op";
    case polyfp::e_var:
        return "var";
    default:
        ERROR("polyfp type not supported.", true);
        return "";
    }
}


std::string str_from_polyfp_type_primitive(polyfp::primitive_t type)
{
    switch (type)
    {
    case polyfp::p_uint8:
        return "uint8";
    case polyfp::p_int8:
        return "int8";
    case polyfp::p_uint16:
        return "uint16";
    case polyfp::p_int16:
        return "int16";
    case polyfp::p_uint32:
        return "uint32";
    case polyfp::p_int32:
        return "int32";
    case polyfp::p_uint64:
        return "uint64";
    case polyfp::p_int64:
        return "int64";
    case polyfp::p_float32:
        return "float32";
    case polyfp::p_float64:
        return "float64";
    default:
        ERROR("polyfp type not supported.", true);
        return "";
    }
}
std::string str_polyfp_type_op(polyfp::op_t type)
{
    switch (type)
    {
    case polyfp::o_max:
        return "max";
    case polyfp::o_min:
        return "min";
    case polyfp::o_add:
        return "add";
    case polyfp::o_sub:
        return "sub";
    case polyfp::o_mul:
        return "mul";
    case polyfp::o_div:
        return "div";
    case polyfp::o_mod:
    case polyfp::o_access:
        return "access";
    default:
        // ERROR("polyfp op not supported.", true);
        return "";
    }
}


isl_map *add_eq_to_schedule_map(int dim0, int in_dim_coefficient, int out_dim_coefficient,
                                int const_conefficient, isl_map *sched)
{
    // isl_map_to_str(sched);
    // std::to_string(const_conefficient);
    isl_map *identity = isl_set_identity(isl_map_range(isl_map_copy(sched)));
    identity = isl_map_universe(isl_map_get_space(identity));
    isl_space *sp = isl_map_get_space(identity);
    isl_local_space *lsp = isl_local_space_from_space(isl_space_copy(sp));

    // Create a transformation map that transforms the schedule.
    for (int i = 0; i < isl_map_dim (identity, isl_dim_out); i++)
        if (i == dim0)
        {
            isl_constraint *cst = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_in, dim0, in_dim_coefficient);
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, dim0, -out_dim_coefficient);
            // TODO: this should be inverted into const_conefficient.
            cst = isl_constraint_set_constant_si(cst, -const_conefficient);
            identity = isl_map_add_constraint(identity, cst);
            // isl_map_to_str(identity);
        }
        else
        {
            // Set equality constraints for dimensions
            isl_constraint *cst2 = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_in, i, 1);
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_out, i, -1);
            identity = isl_map_add_constraint(identity, cst2);
        }

    isl_map *final_identity = identity;
    // isl_map_to_str(final_identity);
    sched = isl_map_apply_range (sched, final_identity);
    // isl_map_to_str(sched);

    return sched;
}


}

