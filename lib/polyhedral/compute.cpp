#include "compute.h"
#include "core.h"
#include <algorithm>
namespace polyfp{

isl_ctx *polyfp::compute::get_ctx() const
{
    return ctx;
}

isl_set *polyfp::compute::get_iteration_domain() const
{
    assert(iteration_domain != NULL);
    return iteration_domain;
}

void polyfp::compute::set_iteration_domain(isl_set *domain)
{
    this->iteration_domain = domain;
}

int polyfp::compute::get_iteration_domain_dimensions_number()
{
    assert(iteration_domain != NULL);
    return isl_set_n_dim(this->iteration_domain);
}

isl_map *compute::get_schedule() const
{
    return this->schedule;
}

isl_set *polyfp::compute::get_trimmed_time_processor_domain()
{
    isl_set *tp_domain = isl_set_copy(this->get_time_processor_domain());
    const char *name = isl_set_get_tuple_name(isl_set_copy(tp_domain));
    isl_set *tp_domain_without_duplicate_dim =
        isl_set_project_out(isl_set_copy(tp_domain), isl_dim_set, 0, 1);
    tp_domain_without_duplicate_dim = isl_set_set_tuple_name(tp_domain_without_duplicate_dim, name);
    return tp_domain_without_duplicate_dim ;
}

void compute::name_unnamed_iteration_domain_dimensions()
{
    isl_set *iter = this->get_iteration_domain();
    assert(iter != NULL);
    for (int i = 0; i < this->get_iteration_domain_dimensions_number(); i++)
    {
        if (isl_set_has_dim_name(iter, isl_dim_set, i) == isl_bool_false)
            iter = isl_set_set_dim_name(iter, isl_dim_set, i,
                                        generate_new_variable_name().c_str());
    }
    this->set_iteration_domain(iter);
}

void compute::name_unnamed_time_space_dimensions()
{
    isl_map *sched = this->get_schedule();
    assert(sched != NULL);
    for (int i = 0; i < this->get_loop_levels_number(); i++)
    {
        if (isl_map_has_dim_name(sched, isl_dim_out, loop_level_into_dynamic_dimension(i)) == isl_bool_false)
            sched = isl_map_set_dim_name(sched, isl_dim_out, loop_level_into_dynamic_dimension(i), generate_new_variable_name().c_str());
    }
    this->set_schedule(sched);
}

isl_map *isl_map_add_dim_and_eq_constraint(isl_map *map, int dim_pos, int constant)
{
    assert(map != NULL);
    assert(dim_pos >= 0);
    assert(dim_pos <= (signed int) isl_map_dim(map, isl_dim_out));

    map = isl_map_insert_dims(map, isl_dim_out, dim_pos, 1);
    map = isl_map_set_tuple_name(map, isl_dim_out, isl_map_get_tuple_name(map, isl_dim_in));

    isl_space *sp = isl_map_get_space(map);
    isl_local_space *lsp =
        isl_local_space_from_space(isl_space_copy(sp));
    isl_constraint *cst = isl_constraint_alloc_equality(lsp);
    cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, dim_pos, 1);
    cst = isl_constraint_set_constant_si(cst, (-1) * constant);
    map = isl_map_add_constraint(map, cst);

    return map;
}

isl_map *polyfp::compute::gen_identity_schedule_for_iteration_domain()
{

    isl_space *sp = isl_set_get_space(this->get_iteration_domain());
    isl_map *sched = isl_map_identity(isl_space_map_from_set(sp));
    sched = isl_map_intersect_domain(sched, isl_set_copy(this->get_iteration_domain()));
    sched = isl_map_coalesce(sched);

    for (int i = 0; i < isl_space_dim(sp, isl_dim_out) + 1; i++)
    {
        sched = isl_map_add_dim_and_eq_constraint(sched, 2 * i, 0);
    }

    sched = isl_map_add_dim_and_eq_constraint(sched, 0, 0);
    return sched;
}

void compute::set_schedule(isl_map *map)
{
    this->schedule = map;
}
void compute::set_schedule(std::string map_str)
{
    assert(!map_str.empty());
    assert(this->ctx != NULL);

    isl_map *map = isl_map_read_from_str(this->ctx, map_str.c_str());
    assert(map != NULL);

    this->set_schedule(map);
}

void compute::dump_iteration_domain() const
{
    isl_set_dump(this->get_iteration_domain());

}

void compute::dump_schedule() const
{
    polyfp::str_dump("Dumping the schedule of the computation " + this->get_name() + " : ");
    std::flush(std::cout);
    isl_map_dump(this->get_schedule());

}

const polyfp::expr &polyfp::compute::get_expr() const
{
    return expression;
}

void compute::dump() const
{
    std::cout << std::endl << "Dumping the computation \"" + this->get_name() + "\" :" << std::endl;
    std::cout << "Iteration domain of the computation \"" << this->name << "\" : ";
    std::flush(std::cout);
    isl_set_dump(this->get_iteration_domain());
    std::flush(std::cout);
    this->dump_schedule();

    std::flush(std::cout);
    std::cout << "Expression of the computation : "; std::flush(std::cout);
    this->get_expr().dump(true);
    std::cout << std::endl; std::flush(std::cout);

    std::cout << "Access relation of the computation : "; std::flush(std::cout);
    isl_map_dump(this->get_access_relation());
    if (this->get_access_relation() == NULL)
    {
        std::cout << "\n";
    }
    std::flush(std::cout);

    if (this->get_time_processor_domain() != NULL)
    {
        std::cout << "Time-space domain " << std::endl; std::flush(std::cout);
        isl_set_dump(this->get_time_processor_domain());
    }
    else
    {
        std::cout << "Time-space domain : NULL." << std::endl;
    }

    polyfp::str_dump("\n");
    polyfp::str_dump("\n");
}

void polyfp::compute::set_identity_schedule_based_on_iteration_domain()
{
    isl_map *sched = this->gen_identity_schedule_for_iteration_domain();
    this->set_schedule(sched);
}

std::vector<std::string> compute::get_iteration_domain_dimension_names()
{

    isl_set *iter = this->get_iteration_domain();
    assert(iter != NULL);
    std::vector<std::string> result;
    for (int i = 0; i < this->get_iteration_domain_dimensions_number(); i++)
    {
        if (isl_set_has_dim_name(iter, isl_dim_set, i))
            result.push_back(std::string(isl_set_get_dim_name(iter,
                                                              isl_dim_set, i)));
        else
        {
            ERROR("All iteration domain dimensions must have "
                            "a name.", true);
        }
    }
    assert(result.size() == this->get_iteration_domain_dimensions_number());
    return result;
}

void compute::update_names(std::vector<std::string> original_loop_level_names, std::vector<std::string> new_names,
                               int erase_from, int nb_loop_levels_to_erase)
{
    this->final_loop_level_names.clear();
    this->final_loop_level_names = this->final_loop_level_names_reserved;
    // // std::cout<<"original names: "<<std::endl;
    // for (auto n: original_loop_level_names)
    // {
    //     polyfp::str_dump(n + " ");
    // }
    // // std::cout<<"finial names: "<<std::endl;
    // for (auto n: final_loop_level_names)
    // {
    //     polyfp::str_dump(n + " ");
    // }
    // polyfp::str_dump("Start erasing from: " + std::to_string(erase_from));
    // polyfp::str_dump("Number of loop levels to erase: " + std::to_string(nb_loop_levels_to_erase));

    original_loop_level_names.erase(original_loop_level_names.begin() + 
        erase_from, original_loop_level_names.begin() + erase_from + nb_loop_levels_to_erase);
    final_loop_level_names.erase(final_loop_level_names.begin() + 
        erase_from, final_loop_level_names.begin() + erase_from + nb_loop_levels_to_erase);

    original_loop_level_names.insert(original_loop_level_names.begin() + 
        erase_from, new_names.begin(), new_names.end());
    final_loop_level_names.insert(final_loop_level_names.begin() + 
        erase_from, new_names.begin(), new_names.end());
    
    // // std::cout<<"original names: "<<std::endl;
    // for (auto n: original_loop_level_names)
    // {
    //     polyfp::str_dump(n + " ");
    // }
    // // std::cout<<"finial names: "<<std::endl;
    // for (auto n: final_loop_level_names)
    // {
    //     polyfp::str_dump(n + " ");
    // }
    // this->final_loop_level_names.clear();
    // this->final_loop_level_names = original_loop_level_names;
    this->set_loop_level_names(original_loop_level_names);

}


void polyfp::compute::set_expression(const polyfp::expr &e)
{
    // polyfp::expr modified_e = traverse_expr_and_replace_non_affine_accesses(this, e);
    // polyfp::str_dump("The original expression is: "); modified_e.dump(false);
    this->expression = e.copy();
}

std::vector<std::string> compute::get_loop_level_names()
{
    // polyfp::str_dump("Collecting names of loop levels from the range of the schedule: ", isl_map_to_str(this->get_schedule()));

    std::vector<std::string> names;
    std::string names_to_print_for_debugging = "";

    for (int i = 0; i < this->get_loop_levels_number(); i++)
    {
        std::string dim_name = isl_map_get_dim_name(this->get_schedule(), 
            isl_dim_out, loop_level_into_dynamic_dimension(i));
        names.push_back(dim_name);
        names_to_print_for_debugging += dim_name + " ";
    }
    // polyfp::str_dump("Names of loop levels: " + names_to_print_for_debugging);

    return names;
}

std::vector<polyfp::expr> polyfp::compute::get_placeholder_dims()
{
    return placeholder_dims;
}

void polyfp::compute::set_placeholder_dims(std::vector<polyfp::expr> temp)
{
    this->placeholder_dims = temp ;
}

polyfp::function *polyfp::compute::get_function() const
{
    return fct;
}

int compute::get_loop_levels_number()
{
    assert(this->get_schedule() != NULL);
    int loop_levels_number = ((isl_map_dim(this->get_schedule(), isl_dim_out)) - 2)/2;

    return loop_levels_number;
}
void compute::set_loop_level_names(std::vector<std::string> names)
{
    assert(names.size() > 0);

    // polyfp::str_dump("Number of loop levels: " + std::to_string(this->get_loop_levels_number()));
    // polyfp::str_dump("Number of names to be set: " + std::to_string(names.size()));

    for (int i = 0; i < names.size(); i++)
    {
        if (isl_map_has_dim_name(this->get_schedule(), isl_dim_out, loop_level_into_dynamic_dimension(i)) == isl_bool_true)
        {
            this->schedule = isl_map_set_dim_name(this->get_schedule(),
                                                  isl_dim_out,
                                                  loop_level_into_dynamic_dimension(i),
                                                  names[i].c_str());
        //    polyfp::str_dump("Setting the name of loop level " + std::to_string(i) + " into " + names[i].c_str());
        }
    }

    // polyfp::str_dump("The schedule after renaming: ", isl_map_to_str(this->get_schedule()));

}

void compute::set_loop_level_names(std::vector<int> loop_levels,
        std::vector<std::string> names)
{
    this->check_dimensions_validity(loop_levels);
    assert(names.size() > 0);
    assert(names.size() == loop_levels.size());

    for (int i = 0; i < loop_levels.size(); i++)
    {
        if (loop_level_into_static_dimension(loop_levels[i]) <= isl_map_dim(this->get_schedule(), isl_dim_out))
        {
            this->schedule = isl_map_set_dim_name(this->get_schedule(),
                                                  isl_dim_out,
                                                  loop_level_into_dynamic_dimension(loop_levels[i]),
                                                  names[i].c_str());
            // polyfp::str_dump("Setting the name of loop level " + std::to_string(loop_levels[i]) + " into " + names[i].c_str());
           
        }
    }

    // polyfp::str_dump("The schedule after renaming: ", isl_map_to_str(this->get_schedule()));
}

isl_set *polyfp::compute::get_time_processor_domain() const
{
    return time_processor_domain;
}

void polyfp::compute::set_access(isl_map *access)
{
    assert(access != NULL);
    this->set_access(isl_map_to_str(access));
}


void polyfp::compute::set_access(std::string access_str)
{
    this->access = isl_map_read_from_str(this->ctx, access_str.c_str());

    std::vector<polyfp::compute *> same_name_computations =
        this->get_function()->get_computation_by_name(this->get_name());

    // TODO: Delete
    if (same_name_computations.size() > 1)
        for (auto c : same_name_computations)
        {
            c->access = isl_map_read_from_str(this->ctx, access_str.c_str());
        }

    std::vector<polyfp::compute *> computations =
        this->get_function()->get_computation_by_name(this->get_name());
    for (auto c : computations)
        if (isl_map_is_equal(this->get_access_relation(), c->get_access_relation()) == isl_bool_false)
        {
            ERROR("Computations that have the same name should also have the same access relation.",
                            true);
        }

    assert(this->access != nullptr && "Set access failed");
}

isl_map *polyfp::compute::get_access_relation() const
{
    return access;
}

polyfp::placeholder *polyfp::compute::get_placeholder()
{
    return this->plhd;
}
polyfp::expr polyfp::compute::get_placeholder_expr()
{
    return this->plhd_expr;
}
std::vector<polyfp::expr> compute::compute_buffer_size()
{

    std::vector<polyfp::expr> dim_sizes;

    // If the computation has an update, we first compute the union of all the
    // updates, then we compute the bounds of the union.
    for (int i = 0; i < this->get_iteration_domain_dimensions_number(); i++)
    {
        isl_set *union_iter_domain = isl_set_copy(this->get_iteration_domain());


        // polyfp::str_dump("Extracting bounds of the following set:", isl_set_to_str(union_iter_domain));
        polyfp::expr lower = utility::get_bound(union_iter_domain, i, false);
        polyfp::expr upper = utility::get_bound(union_iter_domain, i, true);
        polyfp::expr diff = (upper - lower + 1);
        dim_sizes.push_back(diff);
    }
    return dim_sizes;
}
std::map<std::string, std::string > compute::get_access_map(){
    return this->access_map;
}
std::map<std::string, std::string > compute::get_tile_map(){
    return this->tile_map;
}
std::map<std::string, int > compute::get_tile_size_map(){
    return this->tile_size_map;
}
std::map<std::string, std::string > compute::get_directive_map(){
    return this->directive_map;;
}
std::map<std::string, std::string > compute::get_directive_tool_map(){
    return this->directive_tool_map;;
}

void polyfp::compute::set_name(const std::string &n)
{
    this->name = n;
}

isl_map *polyfp::compute::gen_identity_schedule_for_time_space_domain()
{
    isl_set *tp_domain = this->get_trimmed_time_processor_domain();
    isl_space *sp = isl_set_get_space(tp_domain);
    isl_map *sched = isl_map_identity(isl_space_map_from_set(sp));
    sched = isl_map_intersect_domain(
                sched, isl_set_copy(this->get_trimmed_time_processor_domain()));
    sched = isl_map_set_tuple_name(sched, isl_dim_out, "");
    sched = isl_map_coalesce(sched);

    return sched;
}

void compute::assert_names_not_assigned(std::vector<std::string> dimensions)
{
    for (auto const dim: dimensions)
    {
        int d = isl_map_find_dim_by_name(this->get_schedule(), isl_dim_out,
                                         dim.c_str());
        if (d >= 0)
        {
            ERROR("Dimension " + dim + " is already in use.", true);
        }

        d = isl_map_find_dim_by_name(this->get_schedule(), isl_dim_in,
                                     dim.c_str());
        if (d >= 0)
        {
            ERROR("Dimension " + dim + " is already in use.", true);
        }
    }
}

void compute::check_dimensions_validity(std::vector<int> dimensions)
{
    assert(dimensions.size() > 0);

    for (auto const dim: dimensions)
    {
        assert(dim >= compute::root_dimension);

        if (loop_level_into_dynamic_dimension(dim) >=
            isl_space_dim(isl_map_get_space(this->get_schedule()),
                          isl_dim_out))
        {
            ERROR("The dynamic dimension " +
                            std::to_string(loop_level_into_dynamic_dimension(dim)) +
                            " is not less than the number of dimensions of the "
                            "time-space domain " +
                            std::to_string(isl_space_dim(isl_map_get_space(
                                    this->get_schedule()), isl_dim_out)), true);
        }
    }
}

void compute::set_schedule_domain_dim_names(std::vector<int> loop_levels,
        std::vector<std::string> names)
{
    this->check_dimensions_validity(loop_levels);
    assert(names.size() > 0);
    assert(names.size() == loop_levels.size());

    for (int i = 0; i < loop_levels.size(); i++)
    {
        assert(loop_levels[i] <= isl_map_dim(this->get_schedule(), isl_dim_in));
        this->schedule = isl_map_set_dim_name(this->get_schedule(),
                                              isl_dim_in, loop_levels[i], names[i].c_str());
    }

    // polyfp::str_dump("The schedule after renaming: ", isl_map_to_str(this->get_schedule()));
}

void polyfp::compute::init_computation(std::string iteration_space_str,
        polyfp::function *fction,
        const polyfp::expr &e,
        polyfp::primitive_t t, polyfp::expr p)
{
    // polyfp::str_dump("Constructing the computation: " + iteration_space_str);
    assert(iteration_space_str.length() > 0 && ("Empty iteration space"));
    access = NULL;
    time_processor_domain = NULL;
    predicate = polyfp::expr();
    this->data_type = t;
    this->ctx = fction->get_isl_ctx();
    //todo
    for(auto &kv : fction->get_placeholders()){
        if(kv.first == p.get_name())
            this->plhd = kv.second;
    }
    this->plhd_expr = p;
    this->plhd_expr.owner = this;

    placeholder_dims = p.get_access();
    iteration_domain = isl_set_read_from_str(ctx, iteration_space_str.c_str());
    //TODO
    name = std::string(isl_space_get_tuple_name(isl_set_get_space(iteration_domain),
                       isl_dim_type::isl_dim_set));

    number_of_dims = isl_set_dim(iteration_domain, isl_dim_type::isl_dim_set);

    // for (unsigned i = 0; i < number_of_dims; i++) {
    //     if (isl_set_has_dim_name(iteration_domain, isl_dim_type::isl_dim_set, i)) {
    //         std::string dim_name(isl_set_get_dim_name(iteration_domain, isl_dim_type::isl_dim_set, i));
    //         this->access_variables.push_back(make_pair(i, dim_name));
           
    //     }
    // }
    // for(auto &kv: access_variables){
    //     // std::cout<<std::to_string(kv.first)<<kv.second<<std::endl;
    // }
    fct = fction;
    this->is_leader = true;
    this->is_top_parent = true;
    this->is_leaf = true;
    this->has_a_leader = false;
    fct->leader_computations.push_back(this);
    this->after_level = -2;
    fct->add_computation(this);
    this->set_identity_schedule_based_on_iteration_domain();
    this->set_expression(e);

    std::vector<std::string> nms = this->get_iteration_domain_dimension_names();

    for (int i = 0; i< this->get_iteration_domain_dimensions_number(); i++)
        this->set_schedule_domain_dim_names({i}, {generate_new_variable_name()});
    for (int i = 0; i< nms.size(); i++){
        this->set_loop_level_names({i}, {nms[i]});
        this->final_loop_level_names.push_back(nms[i]);
        this->final_loop_level_names_reserved.push_back(nms[i]);
        // if(fct->get_body().size() == 1){
        //     this->iterators_location_map.insert(std::make_pair(nms[i],i));
        //     fct->global_location = nms.size();
        // }  
    }

}

compute::compute(std::string name, std::vector<polyfp::var> iterator_variables, polyfp::expr e, primitive_t t, expr p)
{   
    this->iteration_variables = iterator_variables;
    std::string iteration_space_str = construct_iteration_domain(name, iterator_variables, predicate);
    // std::cout<<iteration_space_str<<std::endl;
    init_computation(iteration_space_str, global::get_implicit_function(), e, t, p);
}

compute::compute(std::string iteration_domain_str, polyfp::expr e,
                         polyfp::primitive_t t,
                         polyfp::function *fct, expr p)
{
    init_computation(iteration_domain_str, fct, e, t, p);
}

compute::compute(std::string name, std::vector<polyfp::var> iterator_variables, polyfp::expr e, expr p)
    : compute(name, iterator_variables, e, p_float64, p) {}
compute::compute(std::string name, std::vector<polyfp::var> iterator_variables, int a, expr p)
    : compute(name, iterator_variables, expr((uint16_t) a), p_float64, p) {}

std::vector<polyfp::var> compute::get_iteration_variables()
{
    return this->iteration_variables;
}

std::string polyfp::compute::construct_iteration_domain(std::string name, std::vector<var> iterator_variables, polyfp::expr predicate)
{
    polyfp::function *function = global::get_implicit_function();

    std::string iteration_space_str = "";
    std::string comp_name = name;

    iteration_space_str += "{" + comp_name + "[";
    if (iterator_variables.size() == 0)
        iteration_space_str += "0";
    else
        for (int i = 0; i < iterator_variables.size(); i++)
        {
            var iter = iterator_variables[i];
            iteration_space_str += iter.get_name();
            if (i < iterator_variables.size() - 1)
                iteration_space_str += ", ";
        }

    iteration_space_str += "] ";

    if (iterator_variables.size() != 0)
        iteration_space_str += ": ";

    if (predicate.is_defined())
	 iteration_space_str += predicate.to_str() + " and ";

    bool insert_and = false;
    for (int i = 0; i < iterator_variables.size(); i++)
    {
        var iter = iterator_variables[i];

        if ((insert_and == true && (iter.lower.is_defined() || iter.upper.is_defined())))
        {
            iteration_space_str += " and ";
            insert_and = false;
        }

        if (iter.lower.is_defined() || iter.upper.is_defined())
        {
            iteration_space_str += iter.lower.to_str() + "<=" + iter.get_name() + "<" + iter.upper.to_str();
            insert_and = true;
        }
    }

    iteration_space_str += "}";

    return iteration_space_str;
}


const std::string &polyfp::compute::get_name() const
{
    return name;
}

polyfp::primitive_t polyfp::compute::get_data_type() const
{
    return data_type;
}

std::vector<int> compute::get_loop_level_numbers_from_dimension_names(
        std::vector<std::string> dim_names)
{
    assert(dim_names.size() > 0);
    std::vector<int> dim_numbers;

    for (auto const dim: dim_names)
    {
        assert(dim.size()>0);
        if (dim == "root")
        {
            int d = compute::root_dimension;
            dim_numbers.push_back(d);
        }
        else
        {
            int d = isl_map_find_dim_by_name(this->get_schedule(), isl_dim_out,
                                             dim.c_str());
            // polyfp::str_dump("Searching in the range of ", isl_map_to_str(this->get_schedule()));
            if (d < 0)
            {
                ERROR("Dimension " + dim + " not found.", true);
            }

            // polyfp::str_dump("Corresponding loop level is " + std::to_string(dynamic_dimension_into_loop_level(d)));
            dim_numbers.push_back(dynamic_dimension_into_loop_level(d));
        }
    }
    this->check_dimensions_validity(dim_numbers);
    return dim_numbers;
}

struct param_pack_1
{
    int in_dim;
    int out_constant;
};

/**
 * Derived from Tiramisu:
 * Take a basic map as input, go through all of its constraints,
 * identifies the constraint of the static dimension param_pack_1.in_dim
 * (passed in user) and replace the value of param_pack_1.out_constant if
 * the static dimension is bigger than that value.
 */
isl_stat extract_static_dim_value_from_bmap(__isl_take isl_basic_map *bmap, void *user)
{
    struct param_pack_1 *data = (struct param_pack_1 *) user;

    isl_constraint_list *list = isl_basic_map_get_constraint_list(bmap);
    int n_constraints = isl_constraint_list_n_constraint(list);

    for (int i = 0; i < n_constraints; i++)
    {
        isl_constraint *cst = isl_constraint_list_get_constraint(list, i);
        isl_val *val = isl_constraint_get_coefficient_val(cst, isl_dim_out, data->in_dim);
        if (isl_val_is_one(val)) // i.e., the coefficient of the dimension data->in_dim is 1
        {
            isl_val *val2 = isl_constraint_get_constant_val(cst);
            int const_val = (-1) * isl_val_get_num_si(val2);
            data->out_constant = const_val;
        }
    }

    return isl_stat_ok;
}

// Derived from Tiramisu:
// if multiple const values exist, choose the maximal value among them because we
// want to use this value to know by how much we shift the computations back.
// so we need to figure out the maximal const value and use it to shift the iterations
// backward so that that iteration runs before the consumer.
isl_stat extract_constant_value_from_bset(__isl_take isl_basic_set *bset, void *user)
{
    struct param_pack_1 *data = (struct param_pack_1 *) user;

    isl_constraint_list *list = isl_basic_set_get_constraint_list(bset);
    int n_constraints = isl_constraint_list_n_constraint(list);

    for (int i = 0; i < n_constraints; i++)
    {
        isl_constraint *cst = isl_constraint_list_get_constraint(list, i);
        if (isl_constraint_is_equality(cst) &&
                isl_constraint_involves_dims(cst, isl_dim_set, data->in_dim, 1))
        {
            isl_val *val = isl_constraint_get_coefficient_val(cst, isl_dim_out, data->in_dim);
            assert(isl_val_is_one(val));
            // assert that the coefficients of all the other dimension spaces are zero.
            isl_val *val2 = isl_constraint_get_constant_val(cst);
            int const_val = (-1) * isl_val_get_num_si(val2);
            data->out_constant = std::max(data->out_constant, const_val);
        }
    }
    return isl_stat_ok;
}


/**
 * Derived from Tiramisu:
 * Return the value of the static dimension.
 * For example, if we have a map M = {S0[i,j]->[0,0,i,1,j,2]; S0[i,j]->[1,0,i,1,j,3]}
 * and call isl_map_get_static_dim(M, 5, 1), it will return 3.
 */
int isl_map_get_static_dim(isl_map *map, int dim_pos)
{
    assert(map != NULL);
    assert(dim_pos >= 0);
    assert(dim_pos <= (signed int) isl_map_dim(map, isl_dim_out));

    // polyfp::str_dump("Getting the constant coefficient of ", isl_map_to_str(map));
    // polyfp::str_dump(" at dimension ");
    // polyfp::str_dump(std::to_string(dim_pos));
    struct param_pack_1 *data = (struct param_pack_1 *) malloc(sizeof(struct param_pack_1));
    data->out_constant = 0;
    data->in_dim = dim_pos;
    isl_map_foreach_basic_map(isl_map_copy(map),
                              &extract_static_dim_value_from_bmap,
                              data);
    // polyfp::str_dump("The constant is: ");
    // polyfp::str_dump(std::to_string(data->out_constant));
    return data->out_constant;
}

std::vector<polyfp::expr> compute::get_loads(){
    auto expr = this->get_expr();
    std::vector<polyfp::expr > loads;
    expr.get_access_vector(loads);  
    return loads;
}

void compute::get_loads_stores()
{
    auto s_loads =this->get_loads();
    std::map<std::string, std::vector<polyfp::expr>> s_single_ls;
    std::vector<polyfp::expr> s_stores;

    s_stores.push_back(this->get_placeholder_expr());
    s_single_ls.insert(std::pair("load", s_loads));
    s_single_ls.insert(std::pair("store", s_stores));
    this->map_loadstores.insert(std::pair(-1,s_single_ls));

    for (auto &edge: this->components)
    {   
        // std::cout<<"dump_component:"+edge.first->get_name()<<std::endl;
        auto loads = edge.first->get_loads();
        std::map<std::string, std::vector<polyfp::expr>> single_ls;
        std::vector<polyfp::expr> stores;
        stores.push_back(edge.first->get_placeholder_expr());
        // std::cout<<"component:"+std::to_string(loads.size())<<std::endl;
        // std::cout<<"component:"+edge.first->get_placeholder_expr().get_name()<<std::endl;
        single_ls.insert(std::pair("load", loads));
        single_ls.insert(std::pair("store", stores));
        this->map_loadstores.insert(std::pair(edge.second,single_ls));
    }
}

void compute::get_all_loadstores()
{
    this->get_loads_stores();
    for(auto &level: this->map_loadstores){
        for(auto &map: level.second){
            if(map.first == "load"){
                for(auto &op: map.second){
                    if(this->load_map.find(op.get_name()) == this->load_map.end()){
                        this->load_map.insert(std::pair(op.get_name(), &op));
                    }
                    this->load_vector.push_back(&op);
                    // // std::cout<<"load_vector:"+op.get_name()<<std::endl;            
                }
                
            }else if(map.first == "store"){
                for(auto &op: map.second){
                    if(this->store_map.find(op.get_name()) == this->store_map.end()){
                        this->store_map.insert(std::pair(op.get_name(), &op));
                    }
                    this->store_vector.push_back(&op);
                }
            }
        }
    }
}


void compute::check_loop_interchange(){
    bool is_legal = true;
    // std::map<int, polyfp::expr> new_order_map;
    std::map<polyfp::expr * , std::vector<polyfp::expr *> > new_order_map ;
    std::unordered_map<std::string, int> final_dim_order;
    std::vector<std::string> dims_no_dp;

    // std::cout<<"check_loop_interchange:"<<std::endl;
    for(auto &vector_list : this->map_dependence_vectors)
    {
        std::vector<std::string> dims_no_dp;
        auto vectors = vector_list.second;
        auto dim_list = vector_list.first->get_access();
        std::unordered_map<polyfp::expr *, int> new_vector_map;
        for(auto &vector: vectors)
        {
            int dims = vector.size();
            bool has_zero = false;
            int zero_number = 0;
            for(int i=0; i<dims; i++){
                if(vector[i] == 0)
                {
                    has_zero = true;
                    zero_number += 1;
                    if (dims_no_dp.size()==0)
                    {
                        if(dim_list[i].get_expr_type() == polyfp::e_op)
                        {
                            if (dim_list[i].get_operand(0).get_expr_type() == polyfp::e_var){
                                dims_no_dp.push_back(dim_list[i].get_operand(0).get_name());
                            }else if(dim_list[i].get_operand(1).get_expr_type() == polyfp::e_var){
                                dims_no_dp.push_back(dim_list[i].get_operand(1).get_name());
                            }
                        }
                        else{
                            dims_no_dp.push_back(dim_list[i].get_name());
                        }
                    }
                    else{                       
                        std::string name;
                        if(dim_list[i].get_expr_type()==polyfp::e_op)
                        {
                            if (dim_list[i].get_operand(0).get_expr_type() == polyfp::e_var){
                                name = dim_list[i].get_operand(0).get_name();
                                // dims_no_dp.push_back(dim_list[i].get_operand(0).get_name());
                            }else if(dim_list[i].get_operand(1).get_expr_type() == polyfp::e_var){
                                name = dim_list[i].get_operand(0).get_name();
                                // dims_no_dp.push_back(dim_list[i].get_operand(1).get_name());
                            }
                            std::vector<std::string>::iterator iter=find(dims_no_dp.begin(),dims_no_dp.end(),name);
                            if ( iter==dims_no_dp.end()){
                                dims_no_dp.push_back(name);
                            }
                        }else{
                            std::vector<std::string>::iterator iter=find(dims_no_dp.begin(),dims_no_dp.end(),dim_list[i].get_name());
                            if ( iter==dims_no_dp.end()){
                                dims_no_dp.push_back(dim_list[i].get_name());
                            }
                        }
                    }                  
                }
                if(vector[i] < 0)
                {
                    has_zero = true;
                    zero_number += 1;
                    is_legal = false;
                }
                if(vector[i] > 0)
                {
                    has_zero = true;
                    zero_number += 1;
                    std::string dim_name;
                    polyfp::expr temp_dim;
                    if(dim_list[i].get_expr_type()==polyfp::e_op)
                    {
                        if (dim_list[i].get_operand(0).get_expr_type() == polyfp::e_var)
                        {
                            dim_name = dim_list[i].get_operand(0).get_name();
                            temp_dim = dim_list[i].get_operand(0);
                        }else if(dim_list[i].get_operand(1).get_expr_type() == polyfp::e_var)
                        {
                            dim_name = dim_list[i].get_operand(1).get_name();
                            temp_dim = dim_list[i].get_operand(1);
                        }
                    }else
                    {
                        dim_name = dim_list[i].get_name();
                        temp_dim = dim_list[i];
                    }
                    if (new_vector_map.find(&temp_dim) == new_vector_map.end())
                    {
                        new_vector_map[&temp_dim] = vector[i];
                    }else if(new_vector_map[&temp_dim]>vector[i]){
                        new_vector_map[&temp_dim] = vector[i];
                    }
                    std::vector<std::string>::iterator iter=find(dims_no_dp.begin(),dims_no_dp.end(),dim_name);
                    if ( iter!=dims_no_dp.end())
                    {
                        iter = dims_no_dp.erase(iter);
                    }
                }
                
            }
            
        }
        std::vector<std::pair<polyfp::expr *, int>> tmp;
        std::vector<polyfp::expr *> dim_order;
        for (auto& i : new_vector_map)
            tmp.push_back(i);
        
        std::sort(tmp.begin(), tmp.end(), [=](std::pair<polyfp::expr *, int>& a, std::pair<polyfp::expr *, int>& b) { return a.second < b.second; });
        // other dims should be moved to outer level first.
        // std::cout<<"new_vector_map:"+ std::to_string(new_vector_map.size())<<std::endl;
        for(auto &other_dim:this->get_iteration_variables())
        {
            std::vector<std::string>::iterator iter=find(dims_no_dp.begin(),dims_no_dp.end(),other_dim.get_name());
            if(new_vector_map.find(&other_dim) == new_vector_map.end()&& iter==dims_no_dp.end())
            {
                dim_order.push_back(&other_dim);
            }
        }
        // move dims that have loop carried dependencies.
        // remove all elements with value val
        for(auto &dim: tmp)
        {
            dim_order.push_back(dim.first);
        }

        new_order_map[vector_list.first] = dim_order;

        for(auto &kv :new_vector_map)
        {
            if (final_dim_order.find(kv.first->get_name()) == final_dim_order.end()&&final_dim_order.size()<this->get_iteration_domain_dimensions_number())
            {
                final_dim_order[kv.first->get_name()] = kv.second;
            }else if(final_dim_order[kv.first->get_name()]>kv.second)
            {
                final_dim_order[kv.first->get_name()] = kv.second;
            }

        }
    }
    // Decide a common order and detect conflicts
    // Define confilct: for all comps in the nested loop, 
    // number of dims that need to be interchanged should not exceed total dims number-1
    // gradually add computes until conflicts occur
    std::vector<std::string> waiting_list;
    bool need_split = false;
    polyfp::compute *comp_to_split ;
    
    for(auto &dvectors: new_order_map)
    {
        for(auto &dvector:dvectors.second)
        {
            std::vector<std::string>::iterator iter=find(waiting_list.begin(),waiting_list.end(),dvector->get_name());
            if ( iter==waiting_list.end())
            {
                waiting_list.push_back(dvector->get_name());
            }
            if(waiting_list.size() == this->get_iteration_domain_dimensions_number())
            {
                need_split = true;
                comp_to_split = dvectors.first->owner;
            }
        }
    }

    if(need_split == true && is_legal == true)
    {
        // TODO: if there is no dependency between comp_to_split and 
        // other comps(its leader and component), split it from the nested loop
        int top_level = 0;
        // for(auto &dim: waiting_list){
        //     int level = this->get_loop_level_number_from_dimension_name(dim);
        //     if(level!=0){
        //          comp_to_split->interchange(top_level,level);
        //     }
        // }
        // comp_to_split->after(comp_to_split->leader,comp_to_split->leader->get_iteration_domain_dimensions_number()-1);
        comp_to_split->after(comp_to_split->leader, -1);
        comp_to_split->get_all_loadstores();
        // comp->dump_components();
        // comp->dump_loads_stores();
        comp_to_split->dump_all_loadstores();
        comp_to_split->compute_dependence_vectors();
        comp_to_split->check_loop_interchange();

    }
    if(need_split == false)
    {
        int top_level = 0;
        for(auto &dim: waiting_list)
        {
            int level = this->get_loop_level_number_from_dimension_name(dim);
            this->interchange(top_level,level);
            int count = level-top_level-1;
            if(count!=0){
                for(int i=0; i<count;i++){
                    this->interchange(top_level+1+i,level);
                }
            }
            top_level+=1;
        }
        for(auto &map: this->components)
        {
            int dims = map.first->get_iteration_variables().size();
            if(map.first->after_level==dims-1)
            {
                int top_level2 = 0;
                for(auto &dim: waiting_list)
                {
                    int level = map.first->get_loop_level_number_from_dimension_name(dim);
                    //TODO: Potential bugs here
                    map.first->interchange(top_level2,level);
                    int count = level-top_level2-1;
                    if(count!=0)
                    {
                        for(int i=0; i<count;i++){
                            map.first->interchange(top_level2+1+i,level);
                        }
                    }
                    map.first->after(map.first->leader,this->get_iteration_domain_dimensions_number()-1);
                    top_level2+=1;
                }
            }
        }
    }
}

void compute::check_loop_skewing()
{
    bool is_legal = false;
    int factor=1;
    if(this->map_dependence_vectors.size() == 1)
    {
        for(auto &vector_list : this->map_dependence_vectors)
        {
            std::vector<std::string> dims_no_dp;
            auto vectors = vector_list.second;
            auto dim_list = vector_list.first->get_access();
            std::unordered_map<polyfp::expr *, int> new_vector_map;
            // TODO: factors
            if(8>=vectors.size()&&vectors.size()>=2)
            {
                is_legal = true;
                factor = 1;
            }
            else if(8<=vectors.size()){
                is_legal = true;
                factor = 2;
            }else{
                // TODO
            }
        }

        auto iterators = this->get_iteration_variables();
        int size = iterators.size();
        std::map<int,polyfp::var> iterator_map;

        for(auto &iter: iterators)
        {
            int loc = this->get_loop_level_number_from_dimension_name(iter.get_name());
            // std::cout<<iter.get_name()<<std::endl;
            iterator_map[loc] = iter;
        }
        var i0("i0"), j0("j0"),k0("k0"), i1("i1"), j1("j1"),k1("k1");
        
        if(is_legal == true)
        {
            if(size==3)
            {
                this->skew(iterator_map[1],iterator_map[2],1,factor,i0,j0);
            }else if(size==2)
            {
                this->skew(iterator_map[0],iterator_map[1],1,factor,i0,j0);
            }
            this->is_skewed_inDSE = true;
        }
    }
}

void compute::auto_loop_transformation()
{
    this->check_loop_interchange();
    this->check_loop_skewing();
}

void compute::apply_opt_strategy(std::vector<int> tile_size){
    std::map<int,polyfp::var> iterator_map;
    
    this->set_schedule(this->original_schedule);
    this->set_loop_level_names(this->original_loop_level_name);
    this->directive_map.clear();
    this->is_unrolled = false;
    this->unroll_factor.clear();
    this->unroll_dimension.clear();
    this->tile_map.clear();
    this->tile_size_map.clear();
    this->access_map.clear();
    auto iterators = this->get_iteration_variables();
    int size = iterators.size();
    //TODO: SKEW MAP
    for(auto &iter: iterators)
    {
        int loc = this->get_loop_level_number_from_dimension_name(iter.get_name());
        iterator_map[loc] = iter;
    }

    if(size >= 3)
    {
        var i0("i0"), j0("j0"),k0("k0"), i1("i1"), j1("j1"),k1("k1");
        // TODO: Config file
        if(tile_size[0]<=64 && tile_size[1]<64 && tile_size[2]<64)
        {
            int temp_index = this->get_iteration_variables().size()-3;

            if(tile_size[2]==1 && tile_size[1]==1 && tile_size[0]==1)
            {  
                // TODO   
            }else
            {
                this->tile(iterator_map[temp_index],iterator_map[temp_index+1],
                    iterator_map[temp_index+2],tile_size[0],tile_size[1],tile_size[2],i0, j0, k0, i1, j1, k1);
            }
            // std::cout<<iterator_map[temp_index].get_name()<<std::endl;
            if(tile_size[2]!=1 && tile_size[1]!=1 && tile_size[0]!=1)
            {
                this->pipeline(k0,1);
                this->unroll(k1,-1);
                this->unroll(j1,-1);
                this->unroll(i1,-1);
            }
            if(tile_size[2]!=1 && tile_size[1]!=1 && tile_size[0]==1)
            {
                this->pipeline(k0,1);
                this->unroll(k1,-1);
                this->unroll(j1,-1);
            }
            if(tile_size[2]==1 && tile_size[1]==1 && tile_size[0]!=1)
            {
                this->pipeline(iterator_map[temp_index+2],1);
                // comp->unroll(k1,-1);
                // comp->unroll(j1,-1);
                this->unroll(i1,-1);
            }
            if(tile_size[2]!=1 && tile_size[1]==1 && tile_size[0]!=1)
            {
                this->pipeline(k0,1);
                this->unroll(k1,-1);
                this->unroll(i1,-1);
            }
            if(tile_size[2]==1 && tile_size[1]==1 && tile_size[0]==1)
            {
                int lower = stoi(iterator_map[temp_index+2].get_lower().to_str());
                int upper = stoi(iterator_map[temp_index+2].get_upper().to_str());
                int range = upper-lower;
                // TODO: Config
                if(range<=7)
                {
                    this->pipeline(iterator_map[temp_index+1],1);
                    this->unroll(iterator_map[temp_index+2],-1);
                }
            }
            if(tile_size[2]!=1 && tile_size[1]==1 && tile_size[0]==1)
            {
                this->pipeline(k0,1);
                this->unroll(k1,-1);
            }
            if(tile_size[2]==1 && tile_size[1]!=1 && tile_size[0]!=1)
            {
                int lower = stoi(iterator_map[temp_index+2].get_lower().to_str());
                int upper = stoi(iterator_map[temp_index+2].get_upper().to_str());
                int range = upper-lower;
                if(range<=6)
                {
                    this->pipeline(j0,1);
                    this->unroll(j1,-1);
                    this->unroll(i1,-1);
                    this->unroll(iterator_map[temp_index+2],-1);
                }else
                {
                    this->pipeline(iterator_map[temp_index+2],1);
                    this->unroll(j1,-1);
                    this->unroll(i1,-1);
                }
            }
            for(auto &part:this->components)
            {
                part.first->set_schedule(part.first->original_schedule);
                part.first->set_loop_level_names(part.first->original_loop_level_name);
                part.first->tile(iterator_map[temp_index+0],iterator_map[temp_index+1],
                    iterator_map[temp_index+2],tile_size[0],tile_size[1],tile_size[2],i0, j0, k0, i1, j1, k1);

                if(tile_size[2]==1 && tile_size[1]!=1 && tile_size[0]!=1)
                {
                    if(part.first->after_level == 2)
                    {
                        part.first->after(this,j1);
                    }
                    else if(part.first->after_level == 0)
                    {
                        part.first->after(this,i0);
                        part.first->pipeline(iterator_map[temp_index+2],1);   
                    }
                }else
                {
                    if(part.first->after_level == 2)
                    {
                        part.first->after(this,k1);
                    }
                    else if
                    (part.first->after_level == 0){
                        part.first->after(this,iterator_map[temp_index+0]);
                        part.first->pipeline(iterator_map[temp_index+2],1);   
                        //TODO
                        part.first->unroll(k1,-1);
                        part.first->unroll(j1,-1);
                    }
                }
            }
        }
    }
    else if(size == 2)
    {
        var i0("i0"), j0("j0"), i1("i1"), j1("j1");
        // TODO: Config file
        if(tile_size[0]<64 && tile_size[1]<64)
        {
            this->tile(iterator_map[0],iterator_map[1],tile_size[0],tile_size[1],i0, j0, i1, j1);
            if(tile_size[1]!=1&&tile_size[0]!=1)
            {
                this->pipeline(j0,1);
                this->unroll(j1,-1);
                this->unroll(i1,-1);
            }else if(tile_size[1]==1&&tile_size[0]!=1)
            {
                this->pipeline(iterator_map[1],1);
                this->unroll(i1,-1);
            }else if(tile_size[0]==1&&tile_size[1]!=1)
            {
                this->pipeline(j0,1);
                this->unroll(j1,-1);
            }
            for(auto &part:this->components)
            {
                part.first->set_schedule(part.first->original_schedule);
                part.first->set_loop_level_names(part.first->original_loop_level_name);
                part.first->directive_map.clear();
                part.first->is_unrolled = false;
                part.first->unroll_factor.clear();
                part.first->unroll_dimension.clear();
                part.first->tile_map.clear();
                part.first->tile_size_map.clear();
                part.first->access_map.clear();
                part.first->tile(iterator_map[0],iterator_map[1],tile_size[0],tile_size[1],i0, j0, i1, j1);

                if(tile_size[1]!=1&&tile_size[0]!=1)
                {
                    if(part.first->after_level == 1)
                    {
                        part.first->after(this,j1);
                    }
                    else if(part.first->after_level == 0)
                    {
                        part.first->after(this,i0);
                        part.first->pipeline(j0,1);
                    }
                    
                }
                else if(tile_size[1]==1&&tile_size[0]!=1)
                {
                    if(part.first->after_level == 1)
                    {
                        part.first->after(this,i1);
                    }
                    else if(part.first->after_level == 0)
                    {
                        part.first->pipeline(iterator_map[1],1);
                        part.first->after(this,i0);
                    }                  
                }
                else if(tile_size[0]==1&&tile_size[1]!=1)
                {
                    if(part.first->after_level == 1)
                    {
                        part.first->after(this,j1);                                  
                    }
                    else if(part.first->after_level == 0)
                    {
                        part.first->after(this,iterator_map[0]);
                        part.first->pipeline(j0,1);
                        part.first->unroll(j1,-1);
                    }
                    else if(part.first->after_level == 2)
                    {
                        part.first->after(this,j1);
                    }
                }
            }
        } 
    }
}



void compute::compute_dependence_vectors()
{
    for(auto &store: this->store_vector)
    {
        auto store_index = store->get_access();
        auto dims = store_index.size();
        for(auto &load: this->load_vector)
        {
            auto load_index = load->get_access();
            if(store->get_name() == load->get_name())
            {
                // std::cout<<"array " + store->get_name()<<std::endl;
                std::vector<int> vector_set;
                for(int i = 0; i < dims; i++)
                {
                    auto vector_element = store_index[i].get_dependence_vector()-load_index[i].get_dependence_vector();
                    // std::cout<<"vector of dimension " + std::to_string(i)+"is: "+std::to_string(vector_element)<<std::endl;
                    vector_set.push_back(vector_element);

                }
                this->map_dependence_vectors[store].push_back(vector_set);
            }
        }
    }
}

void compute::dump_all_loadstores()
{
    std::string result1 = "loads:";
    std::string result2 = "stores:";
    for(auto &op: this->load_map){
        result1 += op.first +" ";
    }
    for(auto &op: this->store_map){
        result2 += op.first +" ";
    }
}

void compute::interchange(polyfp::var L0_var, polyfp::var L1_var)
{
    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name(), L1_var.get_name()});

    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];
    int L1 = dimensions[1];

    this->interchange(L0, L1);
}

void compute::interchange(int L0, int L1)
{
    int inDim0 = loop_level_into_dynamic_dimension(L0);
    int inDim1 = loop_level_into_dynamic_dimension(L1);

    assert(inDim0 >= 0);
    assert(inDim0 < isl_space_dim(isl_map_get_space(this->get_schedule()),
                                  isl_dim_out));
    assert(inDim1 >= 0);
    assert(inDim1 < isl_space_dim(isl_map_get_space(this->get_schedule()),
                                  isl_dim_out));

    isl_map *schedule = this->get_schedule();

    // polyfp::str_dump("Original schedule: ", isl_map_to_str(schedule));
    // polyfp::str_dump("Interchanging the dimensions " + std::to_string(
    //                                 L0) + " and " + std::to_string(L1));

    int n_dims = isl_map_dim(schedule, isl_dim_out);

    std::string inDim0_str = isl_map_get_dim_name(schedule, isl_dim_out, inDim0);
    std::string inDim1_str = isl_map_get_dim_name(schedule, isl_dim_out, inDim1);

    std::vector<isl_id *> dimensions;

    // Create a map for the duplicate schedule.
    std::string map = "{ " + this->get_name() + "[";

    for (int i = 0; i < n_dims; i++)
    {
        if (i == 0)
        {
            int duplicate_ID = isl_map_get_static_dim(schedule, 0);
            map = map + std::to_string(duplicate_ID);
        }
        else
        {
            if (isl_map_get_dim_name(schedule, isl_dim_out, i) == NULL)
            {
                isl_id *new_id = isl_id_alloc(this->get_ctx(), generate_new_variable_name().c_str(), NULL);
                schedule = isl_map_set_dim_id(schedule, isl_dim_out, i, new_id);
            }

            map = map + isl_map_get_dim_name(schedule, isl_dim_out, i);
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "] ->" + this->get_name() + "[";

    for (int i = 0; i < n_dims; i++)
    {
        if (i == 0)
        {
            int duplicate_ID = isl_map_get_static_dim(schedule, 0);
            map = map + std::to_string(duplicate_ID);
        }
        else
        {
            if ((i != inDim0) && (i != inDim1))
            {
                map = map + isl_map_get_dim_name(schedule, isl_dim_out, i);
                dimensions.push_back(isl_map_get_dim_id(schedule, isl_dim_out, i));
            }
            else if (i == inDim0)
            {
                map = map + inDim1_str;
                isl_id *id1 = isl_id_alloc(this->get_ctx(), inDim1_str.c_str(), NULL);
                dimensions.push_back(id1);
            }
            else if (i == inDim1)
            {
                map = map + inDim0_str;
                isl_id *id1 = isl_id_alloc(this->get_ctx(), inDim0_str.c_str(), NULL);
                dimensions.push_back(id1);
            }
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }
    map = map + "]}";
    // polyfp::str_dump(map.c_str());

    isl_map *transformation_map = isl_map_read_from_str(this->get_ctx(), map.c_str());

    transformation_map = isl_map_set_tuple_id(
                             transformation_map, isl_dim_in, isl_map_get_tuple_id(isl_map_copy(schedule), isl_dim_out));
    isl_id *id_range = isl_id_alloc(this->get_ctx(), this->get_name().c_str(), NULL);
    transformation_map = isl_map_set_tuple_id(
                             transformation_map, isl_dim_out, id_range);

    // Check that the names of each dimension is well set
    for (int i = 1; i < isl_map_dim(transformation_map, isl_dim_in); i++)
    {
        isl_id *dim_id = isl_id_copy(dimensions[i - 1]);
        transformation_map = isl_map_set_dim_id(transformation_map, isl_dim_out, i, dim_id);
        assert(isl_map_has_dim_name(transformation_map, isl_dim_in, i));
        assert(isl_map_has_dim_name(transformation_map, isl_dim_out, i));
    }
    // polyfp::str_dump("Final transformation map : ", isl_map_to_str(transformation_map));

    schedule = isl_map_apply_range(isl_map_copy(schedule), isl_map_copy(transformation_map));

    // polyfp::str_dump("Schedule after interchange: ", isl_map_to_str(schedule));

    this->set_schedule(schedule);
}




void compute::split(polyfp::var L0_var, int sizeX)
{
    polyfp::var L0_outer = polyfp::var(generate_new_variable_name());
    polyfp::var L0_inner = polyfp::var(generate_new_variable_name());
    this->split(L0_var, sizeX, L0_outer, L0_inner);
}

void compute::split(polyfp::var L0_var, int sizeX,
        polyfp::var L0_outer, polyfp::var L0_inner)
{
    // polyfp::str_dump("Schedule after interchange: ");
    assert(L0_var.get_name().length() > 0);

    std::vector<std::string> original_loop_level_names =
        this->get_loop_level_names();

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    // polyfp::str_dump("Scget_loop_level_numbers_from_dimension_nameshedule after interchange: ");
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];
    this->assert_names_not_assigned({L0_outer.get_name(), L0_inner.get_name()});
    this->split(L0, sizeX);
    this->set_loop_level_names({L0_outer.get_name(), L0_inner.get_name()});
    // this->update_names(original_loop_level_names, {L0_outer.get_name(), L0_inner.get_name()}, L0, 1);
    // polyfp::str_dump(L0_outer.get_name());
    // polyfp::str_dump(L0_inner.get_name());
}

void compute::split(int L0, int sizeX)
{
    int inDim0 = loop_level_into_dynamic_dimension(L0);

    assert(this->get_schedule() != NULL);
    assert(inDim0 >= 0);
    assert(inDim0 < isl_space_dim(isl_map_get_space(this->get_schedule()), isl_dim_out));
    assert(sizeX >= 1);

    isl_map *schedule = this->get_schedule();
    int duplicate_ID = isl_map_get_static_dim(schedule, 0);

    schedule = isl_map_copy(schedule);
    schedule = isl_map_set_tuple_id(schedule, isl_dim_out,
                                    isl_id_alloc(this->get_ctx(), this->get_name().c_str(), NULL));

    // polyfp::str_dump("Original schedule: ", isl_map_to_str(schedule));
    // polyfp::str_dump("Splitting dimension " + std::to_string(inDim0)
    //                             + " with split size " + std::to_string(sizeX));

    std::string inDim0_str;

    std::string outDim0_str = generate_new_variable_name();
    std::string static_dim_str = generate_new_variable_name();
    std::string outDim1_str = generate_new_variable_name();

    int n_dims = isl_map_dim(this->get_schedule(), isl_dim_out);
    std::vector<isl_id *> dimensions;
    std::vector<std::string> dimensions_str;
    std::string map = "{";

    map = map + this->get_name() + "[";

    for (int i = 0; i < n_dims; i++)
    {
        if (i == 0)
        {
            std::string dim_str = generate_new_variable_name();
            dimensions_str.push_back(dim_str);
            map = map + dim_str;
        }
        else
        {
            std::string dim_str = generate_new_variable_name();
            dimensions_str.push_back(dim_str);
            map = map + dim_str;

            if (i == inDim0)
            {
                inDim0_str = dim_str;
            }
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "] -> " + this->get_name() + "[";

    for (int i = 0; i < n_dims; i++)
    {
        if (i == 0)
        {
            map = map + dimensions_str[i];
            dimensions.push_back(isl_id_alloc(
                                     this->get_ctx(),
                                     dimensions_str[i].c_str(),
                                     NULL));
        }
        else if (i != inDim0)
        {
            map = map + dimensions_str[i];
            dimensions.push_back(isl_id_alloc(
                                     this->get_ctx(),
                                     dimensions_str[i].c_str(),
                                     NULL));
        }
        else
        {
            map = map + outDim0_str + ", " + static_dim_str + ", " + outDim1_str;
            isl_id *id0 = isl_id_alloc(this->get_ctx(),
                                       outDim0_str.c_str(), NULL);
            isl_id *id2 = isl_id_alloc(this->get_ctx(),
                                       static_dim_str.c_str(), NULL);
            isl_id *id1 = isl_id_alloc(this->get_ctx(),
                                       outDim1_str.c_str(), NULL);
            dimensions.push_back(id0);
            dimensions.push_back(id2);
            dimensions.push_back(id1);
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "] : " + dimensions_str[0] + " = " + std::to_string(duplicate_ID) + " and " +
          outDim0_str + " = floor(" + inDim0_str + "/" +
          std::to_string(sizeX) + ") and " + outDim1_str + " = (" +
          inDim0_str + "%" + std::to_string(sizeX) + ") and " + static_dim_str + " = 0}";
    // std::cout<<map;
    isl_map *transformation_map = isl_map_read_from_str(this->get_ctx(), map.c_str());

    for (int i = 0; i < dimensions.size(); i++)
        transformation_map = isl_map_set_dim_id(
                                 transformation_map, isl_dim_out, i, isl_id_copy(dimensions[i]));

    transformation_map = isl_map_set_tuple_id(
                             transformation_map, isl_dim_in,
                             isl_map_get_tuple_id(isl_map_copy(schedule), isl_dim_out));
    isl_id *id_range = isl_id_alloc(this->get_ctx(), this->get_name().c_str(), NULL);
    transformation_map = isl_map_set_tuple_id(transformation_map, isl_dim_out, id_range);

    // polyfp::str_dump("Transformation map : ", isl_map_to_str(transformation_map));

    schedule = isl_map_apply_range(isl_map_copy(schedule), isl_map_copy(transformation_map));

    // polyfp::str_dump("Schedule after splitting: ", isl_map_to_str(schedule));

    this->set_schedule(schedule);

}

void compute::tile(polyfp::var L0, polyfp::var L1,
        polyfp::var L2, int sizeX, int sizeY, int sizeZ)
{
    assert(L0.get_name().length() > 0);
    assert(L1.get_name().length() > 0);
    assert(L2.get_name().length() > 0);

    polyfp::var L0_outer = polyfp::var(generate_new_variable_name());
    polyfp::var L1_outer = polyfp::var(generate_new_variable_name());
    polyfp::var L2_outer = polyfp::var(generate_new_variable_name());
    polyfp::var L0_inner = polyfp::var(generate_new_variable_name());
    polyfp::var L1_inner = polyfp::var(generate_new_variable_name());
    polyfp::var L2_inner = polyfp::var(generate_new_variable_name());

    this->tile(L0, L1, L2, sizeX, sizeY, sizeZ,
               L0_outer, L1_outer, L0_outer, L0_inner, L1_inner, L2_inner);
}

void compute::tile(polyfp::var L0, polyfp::var L1,
        int sizeX, int sizeY)
{
    assert(L0.get_name().length() > 0);
    assert(L1.get_name().length() > 0);

    polyfp::var L0_outer = polyfp::var(generate_new_variable_name());
    polyfp::var L1_outer = polyfp::var(generate_new_variable_name());
    polyfp::var L0_inner = polyfp::var(generate_new_variable_name());
    polyfp::var L1_inner = polyfp::var(generate_new_variable_name());

    this->tile(L0, L1, sizeX, sizeY,
               L0_outer, L1_outer, L0_inner, L1_inner);
}

void compute::tile(polyfp::var L0, polyfp::var L1, polyfp::var L2,
        int sizeX, int sizeY, int sizeZ,
        polyfp::var L0_outer, polyfp::var L1_outer,
        polyfp::var L2_outer, polyfp::var L0_inner,
        polyfp::var L1_inner, polyfp::var L2_inner)
{
    assert(L0.get_name().length() > 0);
    assert(L1.get_name().length() > 0);
    assert(L2.get_name().length() > 0);
    assert(L0_outer.get_name().length() > 0);
    assert(L1_outer.get_name().length() > 0);
    assert(L2_outer.get_name().length() > 0);
    assert(L0_inner.get_name().length() > 0);
    assert(L1_inner.get_name().length() > 0);
    assert(L2_inner.get_name().length() > 0);

    if(sizeX==0 &&sizeY==0&&sizeZ==0)
    {
        return;
    }

    this->assert_names_not_assigned({L0_outer.get_name(), L1_outer.get_name(),
                                     L2_outer.get_name(), L0_inner.get_name(),
                                     L1_inner.get_name(), L2_inner.get_name()});

    std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0.get_name(),
                                                           L1.get_name(),
                                                           L2.get_name()});
    assert(dimensions.size() == 3);

    this->tile(dimensions[0], dimensions[1], dimensions[2],
               sizeX, sizeY, sizeZ);

    if(sizeX == 1 && sizeY == 1 )
    {
        this->update_names(original_loop_level_names, {L0.get_name(), L1.get_name(), L2_outer.get_name(),
                                                       L2_inner.get_name()}, dimensions[0], 3);

    }
    else if(sizeX == 1 && sizeZ == 1 )
    {
        this->update_names(original_loop_level_names, {L0.get_name(), L1_outer.get_name(), L2.get_name(),
                                                       L1_inner.get_name()}, dimensions[0], 3);

    }else if(sizeY == 1 && sizeZ == 1 )
    {
        this->update_names(original_loop_level_names, {L0_outer.get_name(), L1.get_name(), L2.get_name(),
                                                       L0_inner.get_name()}, dimensions[0], 3);

    }else if(sizeX == 1)
    {
        this->update_names(original_loop_level_names, {L0.get_name(), L1_outer.get_name(), L2_outer.get_name(),
                                                    L1_inner.get_name(), L2_inner.get_name()}, dimensions[0], 3);
    }else if(sizeY == 1)
    {
        this->update_names(original_loop_level_names, {L0_outer.get_name(), L1.get_name(), L2_outer.get_name(),
                                                    L0_inner.get_name(), L2_inner.get_name()}, dimensions[0], 3);
    }else if(sizeZ == 1)
    {
        this->update_names(original_loop_level_names, {L0_outer.get_name(), L1_outer.get_name(), L2.get_name(),
                                                    L0_inner.get_name(), L1_inner.get_name()}, dimensions[0], 3);
    }else{
        this->update_names(original_loop_level_names, {L0_outer.get_name(), L1_outer.get_name(), L2_outer.get_name(),
                                                L0_inner.get_name(), L1_inner.get_name(), L2_inner.get_name()}, dimensions[0], 3);
    }

    this->access_map.insert(std::pair(L0.get_name(),L0_inner.get_name()));
    this->access_map.insert(std::pair(L1.get_name(),L1_inner.get_name()));
    this->access_map.insert(std::pair(L2.get_name(),L2_inner.get_name()));
    this->tile_map.insert(std::pair(L0_inner.get_name(),L0_outer.get_name()));
    this->tile_map.insert(std::pair(L1_inner.get_name(),L1_outer.get_name()));
    this->tile_map.insert(std::pair(L2_inner.get_name(),L2_outer.get_name()));
    this->tile_size_map.insert(std::pair(L0_inner.get_name(),sizeX));
    this->tile_size_map.insert(std::pair(L1_inner.get_name(),sizeY));
    this->tile_size_map.insert(std::pair(L2_inner.get_name(),sizeZ));
    this->is_tiled = true;

}

void compute::tile(polyfp::var L0, polyfp::var L1,
      int sizeX, int sizeY,
      polyfp::var L0_outer, polyfp::var L1_outer,
      polyfp::var L0_inner, polyfp::var L1_inner)
{
    assert(L0.get_name().length() > 0);
    assert(L1.get_name().length() > 0);
    assert(L0_outer.get_name().length() > 0);
    assert(L1_outer.get_name().length() > 0);
    assert(L0_inner.get_name().length() > 0);
    assert(L1_inner.get_name().length() > 0);

    std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    this->assert_names_not_assigned({L0_outer.get_name(), L1_outer.get_name(),
                                     L0_inner.get_name(), L1_inner.get_name()});

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0.get_name(),
                                                           L1.get_name()});

    assert(dimensions.size() == 2);

    this->tile(dimensions[0], dimensions[1], sizeX, sizeY);

    // Replace the original dimension name with new dimension names
    if(sizeX == 1)
    {
        this->update_names(original_loop_level_names, {L0.get_name(), L1_outer.get_name(), L1_inner.get_name()}, dimensions[0], 2);
    }
    else if(sizeY == 1)
    {
        this->update_names(original_loop_level_names, {L0_outer.get_name(), L1.get_name(), L0_inner.get_name()}, dimensions[0], 2);
    }else
    {
        this->update_names(original_loop_level_names, {L0_outer.get_name(), L1_outer.get_name(), L0_inner.get_name(),L1_inner.get_name()}, dimensions[0], 2);
    }

    this->access_map.insert(std::pair(L0.get_name(),L0_inner.get_name()));
    this->access_map.insert(std::pair(L1.get_name(),L1_inner.get_name()));
    this->tile_map.insert(std::pair(L0_inner.get_name(),L0_outer.get_name()));
    this->tile_map.insert(std::pair(L1_inner.get_name(),L1_outer.get_name()));
    this->tile_size_map.insert(std::pair(L0_inner.get_name(),sizeX));
    this->tile_size_map.insert(std::pair(L1_inner.get_name(),sizeY));
    this->is_tiled = true;

}

void compute::tile(int L0, int L1, int L2, int sizeX, int sizeY, int sizeZ)
{
    // Check that the two dimensions are consecutive.
    // Tiling only applies on a consecutive band of loop dimensions.
    assert(L1 == L0 + 1);
    assert(L2 == L1 + 1);
    assert((sizeX > 0) && (sizeY > 0) && (sizeZ > 0));
    assert(this->get_iteration_domain() != NULL);

    this->check_dimensions_validity({L0, L1, L2});

    //  Original loops
    //  L0
    //    L1
    //      L2

    this->split(L0, sizeX); // Split L0 into L0 and L0_prime
    // Compute the new L1 and the new L2 and the newly created L0 (called L0 prime)
    int L0_prime = L0 + 1;
    L1 = L1 + 1;
    L2 = L2 + 1;

    //  Loop after transformation
    //  L0
    //    L0_prime
    //      L1
    //        L2

    this->split(L1, sizeY);
    int L1_prime = L1 + 1;
    L2 = L2 + 1;

    //  Loop after transformation
    //  L0
    //    L0_prime
    //      L1
    //        L1_prime
    //          L2

    this->split(L2, sizeZ);

    //  Loop after transformation
    //  L0
    //    L0_prime
    //      L1
    //        L1_prime
    //          L2
    //            L2_prime

    this->interchange(L0_prime, L1);
    // Change the position of L0_prime to the new position
    int temp = L0_prime;
    L0_prime = L1;
    L1 = temp;

    //  Loop after transformation
    //  L0
    //    L1
    //      L0_prime
    //        L1_prime
    //          L2
    //            L2_prime

    this->interchange(L0_prime, L2);
    // Change the position of L0_prime to the new position
    temp = L0_prime;
    L0_prime = L2;
    L2 = temp;

    //  Loop after transformation
    //  L0
    //    L1
    //      L2
    //        L1_prime
    //          L0_prime
    //            L2_prime

    this->interchange(L1_prime, L0_prime);

    //  Loop after transformation
    //  L0
    //    L1
    //      L2
    //        L0_prime
    //          L1_prime
    //            L2_prime
}

void compute::tile(int L0, int L1, int sizeX, int sizeY)
{
    // Check that the two dimensions are consecutive.
    // Tiling only applies on a consecutive band of loop dimensions.
    assert(L1 == L0 + 1);
    assert((sizeX > 0) && (sizeY > 0));
    assert(this->get_iteration_domain() != NULL);
    this->check_dimensions_validity({L0, L1});

    if(sizeX != 1)
    {
        this->split(L0, sizeX);
        this->split(L1 + 1, sizeY);
        this->interchange(L0 + 1, L1 + 1);
    }else
    {
        this->split(L1, sizeY);
    }

}


void compute::skew(polyfp::var L0_var, polyfp::var L1_var,
		       int f_i, int f_j ,
		       polyfp::var new_L0_var, polyfp::var new_L1_var)
{
    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    assert(new_L0_var.get_name().length() > 0);
    assert(new_L1_var.get_name().length() > 0);

    this->assert_names_not_assigned({new_L0_var.get_name(), new_L1_var.get_name()});
    std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name(), L1_var.get_name()});

    this->check_dimensions_validity(dimensions);
    this->is_skewed = true;
    int L0 = dimensions[0];
    int L1 = dimensions[1];
    this->skew(L0, L1, f_i,f_j );
    this->update_names(original_loop_level_names, {new_L0_var.get_name(), new_L1_var.get_name()}, dimensions[0], 2);
    this->access_map.insert(std::pair(L0_var.get_name(),new_L1_var.get_name()));
    this->access_map.insert(std::pair(L1_var.get_name(),new_L0_var.get_name()));
    this->iterator_to_skew = new_L1_var.get_name();
    this->iterator_to_modify = new_L0_var.get_name();
    this->skew_factor = f_j;

}

void compute::skew(int L0 , int L1 , int f_i , int f_j)
{
    if (L0 + 1 != L1)
    {
	    ERROR("Loop levels passed to angle_skew() should be consecutive. The first argument to angle_skew() should be the outer loop level.", true);
    }

    assert(f_j != 0);
    assert(f_i >= 0);
   
    int dim0 = loop_level_into_dynamic_dimension(L0);
    int dim1 = loop_level_into_dynamic_dimension(L1);

    assert(this->get_schedule() != NULL);
    assert(dim0 >= 0);
    assert(dim0 < isl_space_dim(isl_map_get_space(this->get_schedule()), isl_dim_out));
    isl_map *schedule = this->get_schedule();
    int duplicate_ID = isl_map_get_static_dim(schedule, 0);

    schedule = isl_map_copy(schedule);
    schedule = isl_map_set_tuple_id(schedule, isl_dim_out,
                                    isl_id_alloc(this->get_ctx(), this->get_name().c_str(), NULL));
    // polyfp::str_dump("Original schedule: ", isl_map_to_str(schedule));
    // polyfp::str_dump("Angle _ Skewing dimensions " + std::to_string(dim0)
    //                             + " and " + std::to_string(dim1));
    std::string inDim0_str, inDim1_str;
    std::string outDim1_str = generate_new_variable_name();
    std::string outDim0_str = generate_new_variable_name();

    int n_dims = isl_map_dim(this->get_schedule(), isl_dim_out);
    std::vector<isl_id *> dimensions;
    std::vector<std::string> dimensions_str;

    std::string map = "{";
    map = map + this->get_name() + "[";

    for (int i = 0; i < n_dims; i++)
    {
        if (i == 0)
        {
            std::string dim_str = generate_new_variable_name();
            dimensions_str.push_back(dim_str);
            map = map + dim_str;
        }
        else
        {
            std::string dim_str = generate_new_variable_name();
            dimensions_str.push_back(dim_str);
            map = map + dim_str;
            if (i == dim0)
                inDim0_str = dim_str;
            else if (i == dim1)
                inDim1_str = dim_str;
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "] -> " + this->get_name() + "[";

    for (int i = 0; i < n_dims; i++)
    {
        if (i == 0)
        {
            map = map + dimensions_str[i];
            dimensions.push_back(isl_id_alloc(
                                     this->get_ctx(),
                                     dimensions_str[i].c_str(),
                                     NULL));
        }
        else if ((i != dim1) && (i!=dim0))
        {
            map = map + dimensions_str[i];
            dimensions.push_back(isl_id_alloc(
                                     this->get_ctx(),
                                     dimensions_str[i].c_str(),
                                     NULL));
        }
        else // i==dim1
        {
            if(i==dim1){
                  map = map + outDim1_str;
            isl_id *id0 = isl_id_alloc(this->get_ctx(),
                                       outDim1_str.c_str(), NULL);
            dimensions.push_back(id0);
            }
            else{// i== dim 0 
                  map = map + outDim0_str;
            isl_id *id0 = isl_id_alloc(this->get_ctx(),
                                       outDim0_str.c_str(), NULL);
            dimensions.push_back(id0);
            }
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }
    // Computes gcd of f_i and f_j

    int n1 = abs(f_i);
    int n2 = abs(f_j);

    while(n1 != n2)
    {
        if(n1 > n2)
            n1 -= n2;
        else
            n2 -= n1;
    }

    // polyfp::str_dump("The gcd of f_i = "+std::to_string(f_i)+" and fj = "+std::to_string(f_j)+" is pgcd = "+std::to_string(n1));

   

    // Update f_i and f_j to equivalent but prime between themselfs value
    f_i = f_i / n1;
    f_j = f_j / n1;
  
    int gamma = 0;
    int sigma = 1;
    bool found = false;

    if ((f_j == 1) || (f_i == 1))
    {
        gamma = f_i - 1;
        sigma = 1;
        /* Since sigma = 1  then
            f_i - gamma * f_j = 1 & using the previous condition :
             - f_i = 1 : then gamma = 0 (f_i-1) is enough
             - f_j = 1 : then gamma = f_i -1  */
    }
    else
    { 
        if((f_j == - 1) && (f_i > 1))
        {
            gamma = 1;
            sigma = 0;    
        }    
        else
        {   //General case : solving the Linear Diophantine equation & finding basic solution (sigma & gamma) for : f_i* sigma - f_j*gamma = 1 
            int i =0;
            while((i < 100) && (!found))
            {
                if (((sigma * f_i ) % abs(f_j)) ==  1){
                            found = true;
                }
                else{
                    sigma ++;
                    i++;
                }
            };

            if(!found)
            {
                // Detect infinite loop and prevent it in case where f_i and f_j are not prime between themselfs
                ERROR(" Error in solving the Linear Diophantine equation f_i* sigma - f_j*gamma = 1  ", true);
            }

            gamma = ((sigma * f_i) - 1 ) / f_j;
        }
    }
    
    map = map + "] : " + dimensions_str[0] + " = " + std::to_string(duplicate_ID) + " and " +
            outDim0_str + " = (" + inDim0_str + "*"+std::to_string(f_i)+" + "+inDim1_str+"*"+std::to_string(f_j)+" ) and "
          +outDim1_str+" = ("+inDim0_str+"*"+std::to_string(gamma)+" + "+inDim1_str+"*"+std::to_string(sigma)+" ) }";

    // polyfp::str_dump("Transformation angle map (string format) : " + map);

    isl_map *transformation_map = isl_map_read_from_str(this->get_ctx(), map.c_str());

    for (int i = 0; i < dimensions.size(); i++)
        transformation_map = isl_map_set_dim_id(
                                 transformation_map, isl_dim_out, i, isl_id_copy(dimensions[i]));

    transformation_map = isl_map_set_tuple_id(
                             transformation_map, isl_dim_in,
                             isl_map_get_tuple_id(isl_map_copy(schedule), isl_dim_out));
    isl_id *id_range = isl_id_alloc(this->get_ctx(), this->get_name().c_str(), NULL);
    transformation_map = isl_map_set_tuple_id(transformation_map, isl_dim_out, id_range);           
    schedule = isl_map_apply_range(isl_map_copy(schedule), isl_map_copy(transformation_map));
    // polyfp::str_dump("Schedule after transformation is :  ",
    //                             isl_map_to_str(schedule));                            
    this->set_schedule(schedule);
}


void polyfp::compute::after(compute &comp, polyfp::var level)
{
    assert(level.get_name().length() > 0);

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({level.get_name()});

    assert(dimensions.size() == 1);

    int current_level = dimensions[0];
    auto leader_dim_map = comp.iterators_location_map;
    this->after_level = current_level;
    this->ori_after_level = current_level;
    this->after(comp, dimensions[0]);
}

void polyfp::compute::after(compute *comp, polyfp::var level)
{
    assert(level.get_name().length() > 0);
    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({level.get_name()});
    assert(dimensions.size() == 1);
    int current_level = dimensions[0];
    int counter = 0;
    auto leader_dim_map = comp->iterators_location_map;
    this->after_level = current_level;
    this->ori_after_level= current_level;

    this->after(comp, dimensions[0]);

}

void polyfp::compute::after(compute &comp, int level)
{
    auto &graph = this->get_function()->sched_graph;
    auto &edges = graph[&comp];
    auto level_it = edges.find(this);
    edges[this] = level;

    this->get_function()->starting_computations.erase(this);
    // todo
    // this->get_function()->sched_graph_reversed[this][&comp] = level;
    this->after_level = level;
    // this->ori_after_level= level;
    if(level != -1)
    {
        std::vector<polyfp::compute *>::iterator iter2 = this->get_function()->leader_computations.begin();
        while(iter2 != this->get_function()->leader_computations.end())
        {
            if(*iter2 == this)
            {
                iter2 = this->get_function()->leader_computations.erase(iter2);
            }
            else
            {
                iter2++;
            }      
        }
        // this->get_function()->leader_computations.erase(this);
        this->is_leader = false;
        this->is_top_parent = false;
        this->has_a_leader = true;
        this->leader = &comp;

        int component_level = comp.components.size();

        if(component_level !=0)
        {
            std::map<polyfp::compute *, int>::reverse_iterator iter = comp.components.rbegin();
            component_level = iter->second+1;
        }

        auto iter = comp.components.find (this) ;
        if(iter != comp.components.end())
            iter = comp.components.erase (iter);
        comp.components.insert(std::pair(this,component_level));
        comp.update_leader_components(this);
        
    }else if(level == -1)
    {
        this->is_leader = true;
        this->has_a_leader = false;
        this->is_top_parent = false;
        this->leader = NULL; 
        comp.is_leaf = false;
        auto iter = comp.components.find (this) ;
        if(iter != comp.components.end()){
            iter = comp.components.erase (iter);
            comp.delete_leader_components(this);
        }
        std::vector<polyfp::compute *>::iterator iter2 = this->get_function()->leader_computations.begin();
        while(iter2 != this->get_function()->leader_computations.end())
        {
            if(*iter2 == this)
            {
                iter2 = this->get_function()->leader_computations.erase(iter2);
            }
            else
            {
                iter2++;
            }      
        }
        this->get_function()->leader_computations.push_back(this);
        //TODO: check if it is in lead_comp list
        // int current_level = level;
        // int counter = 0;
        // auto dim_list = this->get_loop_level_names();
        // auto leader_dim_map = comp.iterators_location_map;
        // for(int i=0; i<dim_list.size(); i++){
        //     auto fct = global::get_implicit_function();
        //     auto next_level = fct->global_location;
        //     this->iterators_location_map.insert(std::make_pair(dim_list[counter],next_level));
        //     fct->global_location+=1;
        //     counter+=1;
        // }
    }
    
    assert(this->get_function()->sched_graph_reversed[this].size() < 2 &&
            "Node has more than one predecessor.");
    // polyfp::str_dump("sched_graph[" + comp.get_name() + ", " +
    //                              this->get_name() + "] = " + std::to_string(level));                             
}

void polyfp::compute::after(compute *comp, int level)
{
    // polyfp::str_dump("Scheduling " + this->get_name() + " to be executed after " +
    //                             comp.get_name() + " at level " + std::to_string(level));                            

    auto &graph = this->get_function()->sched_graph;
    auto &edges = graph[comp];
    auto level_it = edges.find(this);
    // if (level_it != edges.end())
    // {
    //     if (level_it->second > level)
    //     {
    //         level = level_it->second;
    //     }
    // }

    edges[this] = level;
    

    this->get_function()->starting_computations.erase(this);
    this->after_level = level;
     if(level != -1)
     {
        std::vector<polyfp::compute *>::iterator iter2 = this->get_function()->leader_computations.begin();
        while(iter2 != this->get_function()->leader_computations.end())
        {
            if(*iter2 == this)
            {
                iter2 = this->get_function()->leader_computations.erase(iter2);
            }
            else
            {
                iter2++;
            }      
        }
        this->is_leader = false;
        this->is_top_parent = false;
        this->has_a_leader = true;
        this->leader = comp;

        int component_level = comp->components.size();
        if(component_level !=0)
        {
            std::map<polyfp::compute *, int>::reverse_iterator iter = comp->components.rbegin();
            component_level = iter->second+1;
        }

        auto iter = comp->components.find (this) ;
        if(iter != comp->components.end())
            iter = comp->components.erase (iter);
        comp->components.insert(std::pair(this,component_level));
        comp->update_leader_components(this);
        
    }else if(level == -1)
    {
        this->is_leader = true;
        this->has_a_leader = false;
        this->is_top_parent = false;
        this->leader = NULL; 
        comp->is_leaf = false;
        auto iter = comp->components.find (this) ;
        if(iter != comp->components.end())
        {
            iter = comp->components.erase (iter);
            comp->delete_leader_components(this);
        }
        this->get_function()->leader_computations.push_back(this);
    }
    assert(this->get_function()->sched_graph_reversed[this].size() < 2 &&
            "Node has more than one predecessor.");
}

void polyfp::compute::update_leader_components(polyfp::compute *comp)
{
    if(this->has_a_leader)
    {
        int component_level = this->leader->components.size()+1;
        this->leader->components.insert(std::pair(comp,component_level));
        this->leader->update_leader_components(comp);
    }
}

void polyfp::compute::delete_leader_components(polyfp::compute *comp)
{
    if(this->has_a_leader){
        auto iter = this->leader->components.find (this) ;
        if(iter != this->leader->components.end()){
            iter = this->leader->components.erase (iter);
        }
        // this->leader->components.insert(std::pair(comp,component_level));
        this->leader->update_leader_components(comp);
    }
}

void polyfp::compute::dump_components()
{
    std::string result = "";
    for (auto &edge: this->components)
    {
        result += edge.first->get_name() +"[" + std::to_string(edge.second )+ "]=>";

    }
    result += this->get_name();
    // std::cout<<result<<std::endl;
}

void polyfp::compute::dump_loads_stores()
{
    std::string result = "";

    for (auto &edge: this->map_loadstores)
    {
        result += std::to_string(edge.first )+":[" ;
        for(auto &map: edge.second){
            result+= map.first;
            for(auto &vec: map.second){
                result+= vec.get_name();
            }
        }
        result+=+  "]=>";
    }
        
    result += "root";
    // std::cout<<result<<std::endl;
}

void compute::after_low_level(compute &comp, int level)
{
    // for loop level i return 2*i+1 which represents the
    // the static dimension just after the dynamic dimension that
    // represents the loop level i.
    int dim = loop_level_into_static_dimension(level);

    comp.get_function()->align_schedules();

    assert(this->get_schedule() != NULL);
    assert(dim < (signed int) isl_map_dim(this->get_schedule(), isl_dim_out));
    assert(dim >= compute::root_dimension);

    isl_map *new_sched = NULL;
    for (int i = 1; i<=dim; i=i+2)
    {
        if (i < dim)
        {
            // Get the constant in comp, add +1 to it and set it to sched1
            int order = isl_map_get_static_dim(comp.get_schedule(), i);
            new_sched = isl_map_copy(this->get_schedule());
            new_sched = add_eq_to_schedule_map(i, 0, -1, order, new_sched);
        }
        else // (i == dim)
        {
            // Get the constant in comp, add +1 to it and set it to sched1
            int order = isl_map_get_static_dim(comp.get_schedule(), i);
            new_sched = isl_map_copy(this->get_schedule());
            new_sched = add_eq_to_schedule_map(i, 0, -1, order + 10, new_sched);
        }
        this->set_schedule(new_sched);
    }
    // polyfp::str_dump("Schedule adjusted: ",
    //                             isl_map_to_str(this->get_schedule()));
}



void polyfp::compute::pipeline(polyfp::expr dim, int II)
{
    for(auto &kv: this->get_loop_level_names()){
        if(dim.get_name() == kv){
            int level = this->get_loop_level_number_from_dimension_name(kv);
            this->directive_map.insert(std::pair(kv,"pipeline"));
            std::string c_name = "c"+ std::to_string(level*2+1);
            this->directive_tool_map.insert(std::pair(kv,c_name));
            this->II = II;
        }
    }
}


void polyfp::compute::unroll(polyfp::expr dim, int factor)
{
    this->is_unrolled = true;
    std::string name = dim.get_name();
    auto it = std::find_if(this->unroll_dimension.begin(), this->unroll_dimension.end(),
                            [&](const auto &d) { return d.get_name() == name; });
    if (it == this->unroll_dimension.end()) 
    {
        this->unroll_factor.push_back(factor);
        this->unroll_dimension.push_back(dim);
    }
    // this->unroll_factor.push_back(factor);
    // this->unroll_dimension.push_back(dim);
}

const std::string polyfp::compute::get_unique_name() const
{
    std::stringstream namestream;
    namestream << get_name();
    namestream << "@";
    namestream << (void *)this;
    return namestream.str();
}

void polyfp::compute::gen_time_space_domain()
{
    assert(this->get_iteration_domain() != NULL);
    assert(this->get_schedule() != NULL);

    isl_set *iter = isl_set_copy(this->get_iteration_domain());
    iter = this->intersect_set_with_context(iter);

    time_processor_domain = isl_set_apply(
                                iter,
                                isl_map_copy(this->get_schedule()));
    // polyfp::str_dump("Schedule:", isl_map_to_str(this->get_schedule()));
    // polyfp::str_dump("Generated time-space domain:", isl_set_to_str(time_processor_domain));
}

isl_set *compute::intersect_set_with_context(isl_set *set)
{
    // Unify the space of the context and the "missing" set so that we can intersect them.
    isl_set *context = isl_set_copy(this->get_function()->get_program_context());
    if (context != NULL)
    {
        isl_space *model = isl_set_get_space(isl_set_copy(context));
        set = isl_set_align_params(set, isl_space_copy(model));
        // polyfp::str_dump("Context: ", isl_set_to_str(context));
        // polyfp::str_dump("Set after aligning its parameters with the context parameters: ",
        //                              isl_set_to_str (set));

        isl_id *missing_id1 = NULL;
        if (isl_set_has_tuple_id(set) == isl_bool_true)
        {
            missing_id1 = isl_set_get_tuple_id(set);
        }
        else
        {
            std::string name = isl_set_get_tuple_name(set);
            assert(name.size() > 0);
            missing_id1 = isl_id_alloc(this->get_ctx(), name.c_str(), NULL);
        }

        int nb_dims = isl_set_dim(set, isl_dim_set);
        context = isl_set_add_dims(context, isl_dim_set, nb_dims);
        // isl_set_to_str (context);
        context = isl_set_set_tuple_id(context, isl_id_copy(missing_id1));
        // isl_set_to_str (context);
        set = isl_set_intersect(set, isl_set_copy(context));
        // isl_set_to_str (set);
    }
    return set;
}

}
