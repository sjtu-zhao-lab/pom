#include "codegen.h"
namespace polyfp
{

std::vector<compute *> function::get_computation_by_name(std::string name) const
{
    assert(!name.empty());

    std::vector<polyfp::compute *> res_comp;

    for (const auto &comp : this->get_computations())
    {
        if (name == comp->get_name())
        {
            res_comp.push_back(comp);
        }
    }

    if (res_comp.empty())
    {
        polyfp::str_dump("Computation not found.");
    }
    else
    {
        // polyfp::str_dump("Computation found.");
    }

    return res_comp;
}


bool access_is_affine(const polyfp::expr &exp)
{

    // We assume that the access is affine until we find the opposite.
    bool affine = true;

    // Traverse the expression tree and try to find expressions that are non-affine.
    if (exp.get_expr_type() == polyfp::e_val ||
        exp.get_expr_type() == polyfp::e_var)
    {
        affine = true;
    }
    else if (exp.get_expr_type() == polyfp::e_op)
    {
        switch (exp.get_op_type())
        {
            case polyfp::o_access:
            case polyfp::o_placeholder:
                affine = false;
                break;
            case polyfp::o_add:
            case polyfp::o_sub:
                affine = access_is_affine(exp.get_operand(0)) && access_is_affine(exp.get_operand(1));
                break;
            case polyfp::o_max:
            case polyfp::o_min:
            case polyfp::o_mul:
            case polyfp::o_div:
            case polyfp::o_mod:
                break;
            default:
                ERROR("Unsupported polyfp expression passed to access_is_affine().", 1);
        }
    }

    return affine;
}


isl_ast_node *for_code_generator_after_for(isl_ast_node *node, isl_ast_build *build, void *user)
{
    return node;
}








}
