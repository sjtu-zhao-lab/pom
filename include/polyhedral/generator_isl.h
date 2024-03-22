#ifndef _H_polyfp_function1_
#define _H_polyfp_function1_

#include "expr.h"
#include "type.h"
#include "compute.h"
#include "placeholder.h"
namespace polyfp{
class fucntion;
class generator
{
    friend function;
    friend compute;
    // friend placeholder;


protected:

    /**
     * Compute the accesses of the RHS of the compute
     * \p comp and store them in the accesses vector.
     * If \p return_buffer_accesses is set to true, this function returns access functions to
     * buffers. Otherwise it returns access functions to computes.
     */
    static void get_rhs_accesses(const polyfp::function *func, const polyfp::compute *comp,
                          std::vector<isl_map *> &accesses, bool return_buffer_accesses);

    /**
     * Derived from Tiramisu:
     * Analyze the \p access_expression and return a set of constraints
     * that correspond to the access pattern of the access_expression.
     *
     * access_dimension:
     *      The dimension of the access. For example, the access
     *      C0(i0, i1, i2) have three access dimensions: i0, i1 and i2.
     * access_expression:
     *      The expression of the access.
     *      This expression is parsed recursively (by calling get_constraint_for_access)
     *      and is gradually used to update the constraint.
     * access_relation:
     *      The access relation that represents the access.
     * cst:
     *      The constraint that defines the access and that is being constructed.
     *      Different calls to get_constraint_for_access modify this constraint
     *      gradually until the final constraint is created. Only the final constraint
     *      is added to the access_relation.
     * coeff:
     *      The coefficient in which all the dimension coefficients of the constraint
     *      are going to be multiplied. This coefficient is used to implement o_minus,
     *      o_mul and o_sub.
     */
    static isl_constraint *get_constraint_for_access(int access_dimension,
                                                     const polyfp::expr &access_expression,
                                                     isl_map *access_relation,
                                                     isl_constraint *cst,
                                                     int coeff,
                                                     const polyfp::function *fct);

    /**
     * Derived from Tiramisu:
     * Traverse a polyfp expression (\p exp) and extract the access relations
     * from the access operation passed in \p exp.  The access relations are added
     * to the vector \p accesses.
     * The access relation is from the domain of the compute \p comp to the
     * domain of the compute accessed by the access operation.
     * If \p return_buffer_accesses = true, an access to a buffer is created
     * instead of an access to computes.
     */
    static void traverse_expr_and_extract_accesses(const polyfp::function *fct,
                                            const polyfp::compute *comp,
                                            const polyfp::expr &exp,
                                            std::vector<isl_map *> &accesses,
                                            bool return_buffer_accesses);

public:
    // TODO
};

/**
 * A class containing utility functions.
 */
class utility
{
public:
    /**
     * Derived from Tiramisu:
     * Traverse recursively the ISL AST tree
     * \p node represents the root of the tree to be traversed.
     * \p dim is the dimension of the loop from which the bounds have to be
     * extracted.
     * \p upper is a boolean that should be set to true to extract
     * the upper bound and false to extract the lower bound.
     */
     static expr extract_bound_expression(isl_ast_node *ast, int dim, bool upper);

    /**
     * Derived from Tiramisu:
     * Return a polyfp::expr representing the bound of
     * the dimension \p dim in \p set.  If \p upper is true
     * then this function returns the upper bound otherwise
     * it returns the lower bound.
     *
     * For example, assuming that
     *
     * S = {S[i,j]: 0<=i<N and 0<=j<N and i<M}
     *
     * then
     *
     * get_upper_bound(S, 1)
     *
     * would return N-1, while
     *
     * get_upper_bound(S, 0)
     *
     * would return min(N-1,M-1)
     */
    static polyfp::expr get_bound(isl_set *set, int dim, int upper);

    /**
     * Create a comma separated string that represents the list
     * of the parameters of \p set.
     *
     * For example, if the set is
     *
     * [N,M,K]->{S[i]}
     *
     * this function returns the string "N,M,K".
     */
    static std::string get_parameters_list(isl_set *set);
};


}

#endif