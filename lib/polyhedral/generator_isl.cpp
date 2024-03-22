#include "generator_isl.h"

namespace polyfp{

polyfp::expr polyfp_expr_from_isl_ast_expr(isl_ast_expr *isl_expr);

void generator::get_rhs_accesses(const polyfp::function *func, const polyfp::compute *comp,
                                 std::vector<isl_map *> &accesses, bool return_buffer_accesses)
{

    const polyfp::expr &rhs = comp->get_expr();
    generator::traverse_expr_and_extract_accesses(func, comp, rhs, accesses, return_buffer_accesses);

}

isl_map *create_map_from_domain_and_range(isl_set *domain, isl_set *range)
{

    // polyfp::str_dump("Domain:", isl_set_to_str(domain));
    // polyfp::str_dump("Range:", isl_set_to_str(range));
    // Extracting the spaces and aligning them
    isl_space *sp1 = isl_set_get_space(domain);
    isl_space *sp2 = isl_set_get_space(range);
    sp1 = isl_space_align_params(sp1, isl_space_copy(sp2));
    sp2 = isl_space_align_params(sp2, isl_space_copy(sp1));
    // Create the space access_domain -> sched_range.
    isl_space *sp = isl_space_map_from_domain_and_range(
            isl_space_copy(sp1), isl_space_copy(sp2));
    isl_map *adapter = isl_map_universe(sp);
    polyfp::str_dump("Transformation map:", isl_map_to_str(adapter));
    isl_space *sp_map = isl_map_get_space(adapter);
    isl_local_space *l_sp = isl_local_space_from_space(sp_map);
    // Add equality constraints.
    for (int i = 0; i < isl_space_dim(sp1, isl_dim_set); i++)
    {
        if (isl_space_has_dim_id(sp1, isl_dim_set, i) == true)
        {
            for (int j = 0; j < isl_space_dim (sp2, isl_dim_set); j++)
            {
                if (isl_space_has_dim_id(sp2, isl_dim_set, j) == true)
                {
                    isl_id *id1 = isl_space_get_dim_id(sp1, isl_dim_set, i);
                    isl_id *id2 = isl_space_get_dim_id(sp2, isl_dim_set, j);
                    if (strcmp(isl_id_get_name(id1), isl_id_get_name(id2)) == 0)
                    {
                        isl_constraint *cst = isl_equality_alloc(
                                isl_local_space_copy(l_sp));
                        cst = isl_constraint_set_coefficient_si(cst,
                                                                isl_dim_in,
                                                                i, 1);
                        cst = isl_constraint_set_coefficient_si(
                                cst, isl_dim_out, j, -1);
                        adapter = isl_map_add_constraint(adapter, cst);
                    }
                    isl_id_free(id1);
                    isl_id_free(id2);
                }
            }
        }
    }

    isl_space_free(sp1);
    isl_space_free(sp2);
    isl_local_space_free(l_sp);

    // polyfp::str_dump("Transformation map after adding equality constraints:",
    //         isl_map_to_str(adapter)));


    return adapter;
}

isl_constraint *generator::get_constraint_for_access(int access_dimension,
                                                     const polyfp::expr &access_expression,
                                                     isl_map *access_relation,
                                                     isl_constraint *cst,
                                                     int coeff,
                                                     const polyfp::function *fct)
{
    if (access_expression.get_expr_type() == polyfp::e_val)
    {
        int64_t val = coeff * access_expression.get_int_val() -
                      isl_val_get_num_si(isl_constraint_get_constant_val(cst));
        cst = isl_constraint_set_constant_si(cst, -val);
    //  polyfp::str_dump("Assigning -(coeff * access_expression.get_int_val() - isl_val_get_num_si(isl_constraint_get_constant_val(cst))) to the cst dimension. The value assigned is : "
    //                                 + std::to_string(-val));
    }
    else if (access_expression.get_expr_type() == polyfp::e_var)
    {
        assert(!access_expression.get_name().empty());

        int dim0 = isl_space_find_dim_by_name(isl_map_get_space(access_relation),
                                              isl_dim_in,
                                              access_expression.get_name().c_str());
        if (dim0 >= 0)
        {
            int current_coeff = -isl_val_get_num_si(isl_constraint_get_coefficient_val(cst, isl_dim_in, dim0));
            coeff = current_coeff + coeff;
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_in, dim0, -coeff);
           
        }
        else
        {
            access_relation = isl_map_add_dims(access_relation, isl_dim_param, 1);
            int pos = isl_map_dim(access_relation, isl_dim_param);
            isl_id *param_id = isl_id_alloc(fct->get_isl_ctx(), access_expression.get_name().c_str (), NULL);
            access_relation = isl_map_set_dim_id(access_relation, isl_dim_param, pos - 1, param_id);
            isl_local_space *ls2 = isl_local_space_from_space(isl_map_get_space(access_relation));
            cst = isl_constraint_alloc_equality(ls2);
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_param, pos - 1, -coeff);
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, access_dimension, 1);         
        }
    }
    else if (access_expression.get_expr_type() == polyfp::e_op)
    {
        if (access_expression.get_op_type() == polyfp::o_add)
        {
            polyfp::expr op0 = access_expression.get_operand(0);
            polyfp::expr op1 = access_expression.get_operand(1);
            cst = generator::get_constraint_for_access(access_dimension, op0, access_relation, cst, coeff, fct);
            isl_constraint_dump(cst);
            cst = generator::get_constraint_for_access(access_dimension, op1, access_relation, cst, coeff, fct);
            isl_constraint_dump(cst);
        }
        else if (access_expression.get_op_type() == polyfp::o_sub)
        {
            polyfp::expr op0 = access_expression.get_operand(0);
            polyfp::expr op1 = access_expression.get_operand(1);
            cst = generator::get_constraint_for_access(access_dimension, op0, access_relation, cst, coeff, fct);
            cst = generator::get_constraint_for_access(access_dimension, op1, access_relation, cst, -coeff, fct);
        }
        else if (access_expression.get_op_type() == polyfp::o_mul)
        {
            polyfp::expr op0 = access_expression.get_operand(0);
            polyfp::expr op1 = access_expression.get_operand(1);
            if (op0.get_expr_type() == polyfp::e_val)
            {
                coeff = coeff * op0.get_int_val();
                cst = generator::get_constraint_for_access(access_dimension, op1, access_relation, cst, coeff, fct);
            }
            else if (op1.get_expr_type() == polyfp::e_val)
            {
                coeff = coeff * op1.get_int_val();
                cst = generator::get_constraint_for_access(access_dimension, op0, access_relation, cst, coeff, fct);
            }
        }
        else
        {
            ERROR("Currently only Add, Sub, Minus, and Mul operations for accesses are supported for now.", true);
        }
    }

    return cst;
}

void generator::traverse_expr_and_extract_accesses(const polyfp::function *fct,
                                                   const polyfp::compute *comp,
                                                   const polyfp::expr &exp,
                                                   std::vector<isl_map *> &accesses,
                                                   bool return_buffer_accesses)
{
    assert(fct != NULL);
    assert(comp != NULL);

    if ((exp.get_expr_type() == polyfp::e_op) && ((exp.get_op_type() == polyfp::o_access) ||                                                 
                                                    (exp.get_op_type() == polyfp::o_placeholder)))
    {
       
        std::vector<polyfp::compute *> computations_vector = fct->get_computation_by_name(exp.get_name());

        if (computations_vector.size() == 0)
        {
            // Search for update computations.
            computations_vector = fct->get_computation_by_name("_" + exp.get_name() + "_update_0");
            assert((computations_vector.size() > 0) && "Computation not found.");
        }
        polyfp::compute *access_op_comp = computations_vector[0];

        isl_set *lhs_comp_domain = isl_set_universe(isl_set_get_space(comp->get_iteration_domain()));
        isl_set *rhs_comp_domain = isl_set_universe(isl_set_get_space(
                access_op_comp->get_iteration_domain()));
        isl_map *access_map = create_map_from_domain_and_range(lhs_comp_domain, rhs_comp_domain);
        isl_set_free(lhs_comp_domain);
        isl_set_free(rhs_comp_domain);

        isl_map *access_to_comp = isl_map_universe(isl_map_get_space(access_map));
        isl_map_free(access_map);

        // The dimension_number is a counter that indicates to which dimension
        // is the access associated.
        int access_dimension = 0;
        for (const auto &access : exp.get_access())
        {
            
            isl_constraint *cst = isl_constraint_alloc_equality(isl_local_space_from_space(isl_map_get_space(
                    access_to_comp)));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, access_dimension, 1);
            cst = generator::get_constraint_for_access(access_dimension, access, access_to_comp, cst, 1, fct);
            access_to_comp = isl_map_add_constraint(access_to_comp, cst);
            access_dimension++;
        }

        if (return_buffer_accesses)
        {
            isl_map *access_to_buff = isl_map_copy(access_op_comp->get_access_relation());

            access_to_buff = isl_map_apply_range(isl_map_copy(access_to_comp), access_to_buff);
            accesses.push_back(access_to_buff);
            isl_map_free(access_to_comp);
        }
        else
        {
            accesses.push_back(access_to_comp);
        }
    }
    else if (exp.get_expr_type() == polyfp::e_op)
    {
        switch (exp.get_op_type())
        {
            case polyfp::o_max:
            case polyfp::o_min:
            case polyfp::o_add:
            case polyfp::o_sub:
            case polyfp::o_mul:
            case polyfp::o_div:
            case polyfp::o_mod:
            default:
                ERROR("Extracting access function from an unsupported polyfp expression.", 1);
        }
    }
}

polyfp::expr utility::get_bound(isl_set *set, int dim, int upper)
{
    assert(set != NULL);
    assert(dim >= 0);
    assert(dim < isl_space_dim(isl_set_get_space(set), isl_dim_set));
    assert(isl_set_is_empty(set) == isl_bool_false);

    polyfp::expr e = polyfp::expr();
    isl_ast_build *ast_build;
    isl_ctx *ctx = isl_set_get_ctx(set);
    ast_build = isl_ast_build_alloc(ctx);

    // Create identity map for set.
    isl_space *sp = isl_set_get_space(set);
    isl_map *sched = isl_map_identity(isl_space_copy(isl_space_map_from_set(sp)));
    sched = isl_map_set_tuple_name(sched, isl_dim_out, "");

    // Generate the AST.
    isl_options_set_ast_build_atomic_upper_bound(ctx, 1);
    isl_options_get_ast_build_exploit_nested_bounds(ctx);
    isl_options_set_ast_build_group_coscheduled(ctx, 1);
    isl_options_set_ast_build_allow_else(ctx, 1);
    isl_options_set_ast_build_detect_min_max(ctx, 1);

    // Intersect the iteration domain with the domain of the schedule.
    isl_map *map =
        isl_map_intersect_domain(
            isl_map_copy(sched),
            isl_set_copy(set));

    // Set iterator names
    int length = isl_map_dim(map, isl_dim_out);
    isl_id_list *iterators = isl_id_list_alloc(ctx, length);

    for (int i = 0; i < length; i++)
    {
        std::string name;
        if (isl_set_has_dim_name(set, isl_dim_set, i) == true)
            name = isl_set_get_dim_name(set, isl_dim_set, i);
        else
            name = generate_new_variable_name();
        isl_id *id = isl_id_alloc(ctx, name.c_str(), NULL);
        iterators = isl_id_list_add(iterators, id);
    }

    ast_build = isl_ast_build_set_iterators(ast_build, iterators);

    isl_ast_node *node = isl_ast_build_node_from_schedule_map(ast_build, isl_union_map_from_map(map));
    e = utility::extract_bound_expression(node, dim, upper);
    isl_ast_build_free(ast_build);

    assert(e.is_defined() && "The computed bound expression is undefined.");

    return e;
}

polyfp::expr utility::extract_bound_expression(isl_ast_node *node, int dim, bool upper)
{
    assert(node != NULL);
    assert(dim >= 0);

    polyfp::expr result;

    if (isl_ast_node_get_type(node) == isl_ast_node_block)
    {
        ERROR("Currently Tiramisu does not support extracting bounds from blocks.", true);
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_for)
    {
        isl_ast_expr *init_bound = isl_ast_node_for_get_init(node);
        isl_ast_expr *upper_bound = isl_ast_node_for_get_cond(node);
      
        if (dim == 0)
        {
            if (upper)
            {
                isl_ast_expr *cond = isl_ast_node_for_get_cond(node);

                if (isl_ast_expr_get_op_type(cond) == isl_ast_op_lt)
                {
                    // Create an expression of "1".
                    isl_val *one = isl_val_one(isl_ast_node_get_ctx(node));
                    // Add 1 to the ISL ast upper bound to transform it into a strinct bound.
                    result = polyfp_expr_from_isl_ast_expr(isl_ast_expr_sub(isl_ast_expr_get_op_arg(cond, 1),
                                                             isl_ast_expr_from_val(one)));
                }
                else if (isl_ast_expr_get_op_type(cond) == isl_ast_op_le)
                {
                    result = polyfp_expr_from_isl_ast_expr(isl_ast_expr_get_op_arg(cond, 1));
                }
            }
            else
            {
                isl_ast_expr *init = isl_ast_node_for_get_init(node);
                result = polyfp_expr_from_isl_ast_expr(init);
            }
        }
        else
        {
            isl_ast_node *body = isl_ast_node_for_get_body(node);
            result = utility::extract_bound_expression(body, dim-1, upper);
            isl_ast_node_free(body);
        }

        assert(result.is_defined());
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_user)
    {
        ERROR("Cannot extract bounds from a isl_ast_user node.", true);
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_if)
    {
        // polyfp::expr cond_bound = polyfp_expr_from_isl_ast_expr(isl_ast_node_if_get_cond(node));
        polyfp::expr then_bound = utility::extract_bound_expression(isl_ast_node_if_get_then(node), dim, upper);

        polyfp::expr else_bound;
        if (isl_ast_node_if_has_else(node))
        {
            // else_bound = utility::extract_bound_expression(isl_ast_node_if_get_else(node), dim, upper);
            // result = polyfp::expr(polyfp::o_s, cond_bound, then_bound, else_bound);
            ERROR("If Then Else is unsupported in bound extraction.", true);
        }
        else
            result = then_bound; //polyfp::expr(polyfp::o_cond, cond_bound, then_bound);
    }

    return result;
}

std::string utility::get_parameters_list(isl_set *set)
{
    std::string list = "";

    assert(set != NULL);

    for (int i = 0; i < isl_set_dim(set, isl_dim_param); i++)
    {
        list += isl_set_get_dim_name(set, isl_dim_param, i);
        if ((i != isl_set_dim(set, isl_dim_param) - 1))
        {
            list += ",";
        }
    }

    return list;
}

polyfp::expr polyfp_expr_from_isl_ast_expr(isl_ast_expr *isl_expr)
{
    polyfp::expr result;

    if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_int)
    {
        isl_val *init_val = isl_ast_expr_get_val(isl_expr);
        result = value_cast(polyfp::global::get_loop_iterator_data_type(), isl_val_get_num_si(init_val));
        isl_val_free(init_val);
    }
    else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_id)
    {
        isl_id *identifier = isl_ast_expr_get_id(isl_expr);
        std::string name_str(isl_id_get_name(identifier));
        isl_id_free(identifier);
        // TODO
        // result = polyfp::var(polyfp::global::get_loop_iterator_data_type(), name_str);
    }
    else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_op)
    {
        polyfp::expr op0, op1, op2;
        std::vector<polyfp::expr> new_arguments;

        isl_ast_expr *expr0 = isl_ast_expr_get_op_arg(isl_expr, 0);
        op0 = polyfp_expr_from_isl_ast_expr(expr0);
        isl_ast_expr_free(expr0);

        if (isl_ast_expr_get_op_n_arg(isl_expr) > 1)
        {
            isl_ast_expr *expr1 = isl_ast_expr_get_op_arg(isl_expr, 1);
            op1 = polyfp_expr_from_isl_ast_expr(expr1);
            isl_ast_expr_free(expr1);
        }

        if (isl_ast_expr_get_op_n_arg(isl_expr) > 2)
        {
            isl_ast_expr *expr2 = isl_ast_expr_get_op_arg(isl_expr, 2);
            op2 = polyfp_expr_from_isl_ast_expr(expr2);
            isl_ast_expr_free(expr2);
        }

        switch (isl_ast_expr_get_op_type(isl_expr))
        {
            case isl_ast_op_max:
                result = polyfp::expr(polyfp::o_max, op0, op1);
                break;
            case isl_ast_op_min:
                result = polyfp::expr(polyfp::o_min, op0, op1);
                break;
            case isl_ast_op_add:
                result = polyfp::expr(polyfp::o_add, op0, op1);
                break;
            case isl_ast_op_sub:
                result = polyfp::expr(polyfp::o_sub, op0, op1);
                break;
            case isl_ast_op_mul:
                result = polyfp::expr(polyfp::o_mul, op0, op1);
                break;
            case isl_ast_op_div:
                result = polyfp::expr(polyfp::o_div, op0, op1);
                break;
            default:
                polyfp::str_dump("Transforming the following expression",
                                   (const char *)isl_ast_expr_to_C_str(isl_expr));
                polyfp::str_dump("\n");
                ERROR("Translating an unsupported ISL expression into a Tiramisu expression.", 1);
        }
    }
    else
    {
        polyfp::str_dump("Transforming the following expression",
                           (const char *)isl_ast_expr_to_C_str(isl_expr));
        polyfp::str_dump("\n");
        ERROR("Translating an unsupported ISL expression into a Tiramisu expression.", 1);
    }


    return result;
}
}