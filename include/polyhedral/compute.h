#ifndef _H_polyfp_COMPUTE_
#define _H_polyfp_COMPUTE_

#include <isl/ctx.h>
#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/id.h>
#include <isl/constraint.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/space.h>
#include <isl/constraint.h>
#include <unordered_map>
#include<algorithm>
#include <isl/val.h>
#include <map>
#include "expr.h"
#include "placeholder.h"
#include "debug.h"
namespace polyfp{
std::string generate_new_computation_name();
class var;
class scheduler;
class function;
class placeholder;
class compute
{
  friend function;
  friend placeholder;
private:

    // The data access map
    isl_map *access;

    // The isl context of the function.
    isl_ctx *ctx;

    // The placeholder that stores the results
    polyfp::placeholder *plhd;
    polyfp::expr plhd_expr;

    polyfp::primitive_t data_type;


    // the expression (or statement) of the function
    polyfp::expr expression;

    polyfp::function *fct;

    /**
     * TODO: 
     */
    std::map<std::string, std::string > access_map;


    // The iteration domain of the compute(nested loop)
    isl_set *iteration_domain;


    // The name of the compute(nested loop)
    std::string name;

    // The number of dimensions in the original definition of the compute
    int number_of_dims;

    /**
     * TODO: Add predicates to the nested loops
     * Derived from Tiramisu:
     * A predicate around the compute. The compute is executed
     * only if this predicate is true. This is useful to insert a non-affine
     * condition around the compute.
     */
    polyfp::expr predicate;


    // The schedule of the compute.
    isl_map * schedule;

    /**
      * Derived from Tiramisu:
      * Time-processor domain of the compute.
      * In this representation, the logical time of execution and the
      * processor where the compute will be executed are both
      * specified.
      */
    isl_set *time_processor_domain;

    
    // The iteration variables(iterators) of the compute
    std::vector<polyfp::var> iteration_variables;

    /**
     * TODO: 
     */
    std::vector<polyfp::expr> placeholder_dims;
    std::vector<polyfp::expr > placeholder_accessmap;
    
    /**
    * TODO: add predicate
    * \p predicate is an expression that represents constraints on the iteration domain
    * (for example (i != j). The predicate has to be an affine
    * expression.
    */
    std::string construct_iteration_domain(std::string name, std::vector<var> iterator_variables, 
                                          polyfp::expr predicate);

    // Return the names of iteration domain dimensions.
    std::vector<std::string> get_iteration_domain_dimension_names();

    void check_dimensions_validity(std::vector<int> dimensions);

    // Get the number of dimensions of the compute
    int get_iteration_domain_dimensions_number();

    // Check that the names used in \p dimensions are not already in use.
    void assert_names_not_assigned(std::vector<std::string> dimensions);



    /**
      * Generate an identity schedule for the compute
      * Derived from Tiramisu:
      * This identity schedule is an identity relation created from the iteration domain.
      */
    isl_map *gen_identity_schedule_for_iteration_domain();

    /**
      * Generate an identity schedule for the compute.
      * Derived from Tiramisu:
      * This identity schedule is an identity relation created from the time-processor domain.
      */
    isl_map *gen_identity_schedule_for_time_space_domain();

    // Assign a name to iteration domain dimensions that do not have a name.
    void name_unnamed_iteration_domain_dimensions();

    // Assign a name to iteration domain dimensions that do not have a name.
    void name_unnamed_time_space_dimensions();


    /**
      * Set an identity schedule for the compute.
      * Derived from Tiramisu:
      * This identity schedule is an identity relation created from the iteration domain.
      */
    void set_identity_schedule_based_on_iteration_domain();

    // Set the iteration domain of the compute
    void set_iteration_domain(isl_set *domain);

    // Set the names of loop levels dimensions.
    void set_loop_level_names(std::vector<int> loop_levels, std::vector<std::string> names);
    void set_loop_level_names(std::vector<std::string> names);

    // Set the names of the dimensions of the schedule domain.
    void set_schedule_domain_dim_names(std::vector<int> loop_levels, std::vector<std::string> names);

    // Return the function where the compute is declared.
    polyfp::function *get_function() const;


    /**
      * Derived from Tiramisu:
      * Search the time-space domain (the range of the schedule) and
      * return the loop level numbers that correspond to the dimensions
      * named \p dim.
      */
    std::vector<int> get_loop_level_numbers_from_dimension_names(std::vector<std::string> dim_names);


    // Intersect set with the context of the compute.
    isl_set *intersect_set_with_context(isl_set *set);

    /**
      * Derived from Tiramisu:
      * Return the time-processor domain of the compute.
      * In this representation, the logical time of execution and the
      * processor where the compute will be executed are both specified.
      */
    isl_set *get_time_processor_domain() const;

    /**
      * Derived from Tiramisu:
      * Return the trimmed time-processor domain.
      * TODO: The first dimension of the time-processor domain is used
      * to indicate redundancy of the compute. In POM there is no redundancy 
      * of the compute. This feature will be removed soon.
      * The trimmed time-processor domain is the time-processor domain
      * without the dimension that represents the redundancy. We simply
      * take the time-processor domain and remove the first dimension.
      */
    isl_set *get_trimmed_time_processor_domain();

    /**
      * Derived from Tiramisu:
      * Update loop level names. This function should be called after each scheduling operation
      * because scheduling usually changes the original loop level names.
      * This function erases \p nb_loop_levels_to_erase loop level names starting from the
      * loop level \p start_erasing. It then inserts the loop level names \p new_names in
      * \p start_erasing. In other words, it replaces the names of loop levels from
      * \p start_erasing to \p start_erasing + \p nb_loop_levels_to_erase with the loop levels
      * indicated by \p new_names.  This function sets the non erased loop levels to be equal to the
      * original loop level names.
      *
      * \p original_loop_level_names : a vector containing the original loop level names (loop level
      * names before scheduling).
      *
      * \p new_names : the new loop level names.
      *
      * \p start_erasing : start erasing loop levels from this loop level.
      *
      * \p nb_loop_levels_to_erase : number of loop levels to erase.
      *
      * Example. Assuming the original loop levels are {i0, i1, i2, i3}
      *
      * Calling this->update_names({i0, i1, i2, i3}, {x0, x1}, 1, 2) updates the loop levels to become
      * {i0, x0, x1, i3}.
      */
    void update_names(std::vector<std::string> original_loop_level_names, std::vector<std::string> new_names,
                      int start_erasing, int nb_loop_levels_to_erase);

protected:

    isl_ctx *get_ctx() const;

    polyfp::expr get_predicate();

    /**
      * Return a unique name of compute; made of the following pattern:
      * [compute name]@[compute address in memory]
      */
    const std::string get_unique_name() const;

    // Set the name of the compute.
    void set_name(const std::string &n);

    void init_computation(std::string iteration_space_str,
                          polyfp::function *fct,
                          const polyfp::expr &e,
                          polyfp::primitive_t t, expr p);


    void set_schedule(isl_map *map);
    void set_schedule(std::string map_str);

    compute(std::string name,std::vector<var> iterator_variables, polyfp::expr e, primitive_t t, expr p);


public:
    compute();
    compute(std::string iteration_domain, polyfp::expr e,
                polyfp::primitive_t t,
                polyfp::function *fct, expr p);

    
    int II;
    bool is_unrolled;
    long latency;
    long best_latency = LLONG_MAX;
    int dsp;
    int minII;

    std::vector<polyfp::var> get_iteration_variables();

    isl_map * original_schedule;
    std::map<std::string, std::string > tile_map;
    std::map<std::string, int > tile_size_map;
    std::map<std::string, std::string > directive_map;
    std::map<std::string, std::string > directive_tool_map;
    std::vector<std::string> original_loop_level_name;
    std::vector<std::string> final_loop_level_names;
    std::vector<std::string> final_loop_level_names_reserved;
    std::vector<int> unroll_factor;
    std::vector<polyfp::expr> unroll_dimension;

    bool refused = false;
    std::map<std::string, std::string > temp_access_map;


    isl_map * best_schedule;
    std::map<std::string, std::string > best_tile_map;
    std::map<std::string, int > best_tile_size_map;
    std::map<std::string, std::string > best_directive_map;
    std::map<std::string, std::string > best_directive_tool_map;
    std::vector<std::string> best_loop_level_names;
    std::vector<int> best_unroll_factor;
    std::vector<polyfp::expr> best_unroll_dimension;

    std::map<std::string, int>iterators_location_map;
    int after_level;
    int ori_after_level;

    compute(std::string name, std::vector<var> iterator_variables, polyfp::expr e, expr p);
    compute(std::string name, std::vector<var> iterator_variables, int a, expr p);
    isl_map *get_access_relation() const;


    bool is_tiled = false ;
    bool is_skewed = false;
    bool is_optimized = false;
    bool is_pipelined = false;
    // bool is_first_opt = false;

    // TODO: Config file
    int current_factor = 1;
    int largest_factor = 2;

    std::string iterator_to_skew;
    std::string iterator_to_modify;
    int skew_factor;

    std::vector<std::string> get_loop_level_names();

    int get_loop_level_number_from_dimension_name(std::string dim_name)
    {
        return this->get_loop_level_numbers_from_dimension_names({dim_name})[0];
    }

    // Debug
    void dump_iteration_domain() const;

    // Debug
    void dump_schedule() const;

    // Debug
    void dump() const;

    void gen_time_space_domain();

    primitive_t get_data_type() const;

    const polyfp::expr &get_expr() const;

    std::vector<polyfp::expr> get_placeholder_dims();

    void set_placeholder_dims(std::vector<polyfp::expr> temp);

    int get_loop_levels_number();

    isl_set *get_iteration_domain() const;


    std::vector<polyfp::expr> compute_buffer_size();
    std::map<std::string, std::string > get_access_map();
    std::map<std::string, std::string > get_tile_map();
    std::map<std::string, int > get_tile_size_map();
    std::map<std::string, std::string > get_directive_map();
    std::map<std::string, std::string > get_directive_tool_map();
    void update_leader_components(polyfp::compute *comp);
    void delete_leader_components(polyfp::compute *comp);


    // DSE components
    std::map<polyfp::compute *, int> components;
    std::map<int, polyfp::compute *> component_level_map;
    polyfp::compute *leader;

    std::unordered_map<int, polyfp::compute *> childern;
    std::vector<polyfp::compute * > parents;

    bool is_leader;
    bool has_a_leader;
    bool is_top_parent;
    bool is_leaf;

    void dump_components();
    void dump_loads_stores();

    const std::string &get_name() const;

    isl_map *get_schedule() const;

    void set_expression(const polyfp::expr &e);

    void set_access(std::string access_str);
    void set_access(isl_map *access);

    placeholder *get_placeholder();
    expr get_placeholder_expr();


    // OPT 
    virtual void interchange(var L0, var L1);
    virtual void interchange(int L0, int L1);

    virtual void split(var L0, int sizeX);
    virtual void split(var L0, int sizeX, var L0_outer, var L0_inner);
    virtual void split(int L0, int sizeX);

    virtual void tile(var L0, var L1, int sizeX, int sizeY);
    virtual void tile(var L0, var L1, int sizeX, int sizeY,
                      var L0_outer, var L1_outer, var L0_inner, var L1_inner);
    virtual void tile(var L0, var L1, var L2, int sizeX, int sizeY, int sizeZ);
    virtual void tile(var L0, var L1, var L2, int sizeX, int sizeY, int sizeZ,
                      var L0_outer, var L1_outer, var L2_outer, var L0_inner,
                      var L1_inner, var L2_inner);
    virtual void tile(int L0, int L1, int sizeX, int sizeY);
    virtual void tile(int L0, int L1, int L2, int sizeX, int sizeY, int sizeZ);

    virtual void skew(var i, var j, int a , int b, var ni, var nj);
    virtual void skew(int i, int j, int a, int b); 

    void after(compute &comp, polyfp::var iterator);
    void after(compute &comp, int level);
    void after(compute *comp, polyfp::var iterator);
    void after(compute *comp, int level);

    void after_low_level(compute &comp, int level);
    void after_low_level(compute &comp, std::vector<int> levels);

    void pipeline(polyfp::expr dim, int II);
    void unroll(polyfp::expr dim, int factor);

    std::map<int, std::map<std::string, std::vector<polyfp::expr> > > map_loadstores;
    std::vector<polyfp::expr> get_loads();
    void get_loads_stores();
    void get_all_loadstores();
    void auto_loop_transformation();
    void compute_dependence_vectors();
    std::unordered_map<std::string, polyfp::expr *> load_map;
    std::unordered_map<std::string, polyfp::expr *> store_map;
    std::vector<polyfp::expr *> load_vector;
    std::vector<polyfp::expr *> store_vector;
    std::map<polyfp::expr *, std::vector<std::vector<int> > >  map_dependence_vectors;

    void dump_all_loadstores();
    void check_loop_interchange();
    void check_loop_skewing();
    void apply_opt_strategy(std::vector<int>);

    bool opt_finished = false;
    bool is_skewed_inDSE = false;
    std::vector<int> final_strategy;
    std::vector<int> current_strategy;
    std::vector<int> temp_strategy;

    const static int root_dimension = -1;
    
    template<typename... Args> polyfp::expr operator()(Args... args)
    {
        std::vector<polyfp::expr> access_expressions{std::forward<Args>(args)...};
        if (access_expressions.size() != this->number_of_dims)
        {
            polyfp::str_dump("Error - Incorrect access: " + this->get_name() + "(");
            for (int i = 0; i < access_expressions.size(); i++)
            {
                polyfp::expr e = access_expressions[i];
                e.dump(false);
                if (i != access_expressions.size() - 1)
                    polyfp::str_dump(", ");
            }
            polyfp::str_dump(").\n");
            polyfp::str_dump("The number of access dimensions does not match that used in the declaration of " + this->get_name() + ".\n\n");
            exit(1);
        }
        return polyfp::expr(polyfp::o_access,
                              this->get_name(),
                              access_expressions,
                              this->get_data_type());
    }
    operator expr();
};


}


#endif
