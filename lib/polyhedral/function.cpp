
#include "function.h"
#include "generator.h"
#include <iostream>
#include <fstream>
#include <filesystem>


namespace polyfp{


isl_map *isl_map_align_range_dims(isl_map *map, int max_dim)
{
    assert(map != NULL);
    int mdim = isl_map_dim(map, isl_dim_out);
    assert(max_dim >= mdim);
    // polyfp::str_dump("Input map:", isl_map_to_str(map));

    const char *original_range_name = isl_map_get_tuple_name(map, isl_dim_out);
    map = isl_map_add_dims(map, isl_dim_out, max_dim - mdim);

    for (int i = mdim; i < max_dim; i++)
    {
        isl_space *sp = isl_map_get_space(map);
        isl_local_space *lsp = isl_local_space_from_space(sp);
        isl_constraint *cst = isl_constraint_alloc_equality(lsp);
        cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, i, 1);
        map = isl_map_add_constraint(map, cst);
    }

    map = isl_map_set_tuple_name(map, isl_dim_out, original_range_name);

    // polyfp::str_dump("After alignment, map = ",isl_map_to_str(map));

    return map;
}

function::function(std::string name)
{
    this->name = name;
    this->ast = NULL;
    this->context_set = NULL;
    this->ctx = isl_ctx_alloc();
    this->global_location = 0;
};

isl_ctx *function::get_isl_ctx() const
{
    return ctx;
}

const std::vector<compute *> &function ::get_computations() const
{
    return body;
}

const std::vector<compute *> &function ::get_body() const
{
    return body;
}

void polyfp::function::add_invariant(std::pair <std::string, polyfp::constant *> invar)
{
    this->constant_list.insert(invar);
}

const std::map<std::string, polyfp::constant *> &function::get_invariants() const
{
    return constant_list;
}

void polyfp::function::set_partition(std::string name, std::vector<int> factors, std::vector<std::string> types)
{
    // std::vector<std::string> types;
    // for (int dim = 0; dim < factors.size(); ++dim) {
    //     types.push_back(type);
    // }
    std::tuple<std::string, std::vector<int>, std::vector<std::string>> dims(name,factors,types);
    this->partition_map.push_back(dims);


}

std::vector<std::tuple<std::string, std::vector<int>, std::vector<std::string>>> polyfp::function::get_partition_map()
{
    return this->partition_map;
}

void polyfp::function::add_computation(compute *cpt)
{
    assert(cpt != NULL);
    this->body.push_back(cpt);
    this->starting_computations.insert(cpt);
}

void polyfp::function::dump(bool s) const
{
    if (s)
    {
        std::cout << "\n\nFunction \"" << this->name << "\"" << std::endl << std::endl;

        if (this->function_arguments.size() > 0)
        {
            std::cout << "Function arguments (polyfp buffers):" << std::endl;
            for (const auto &buf : this->function_arguments)
            {
                // buf->dump(s);
            }
            std::cout << std::endl;
        }
        // todo
        if (this->invariants.size() > 0)
        {
            std::cout << "Function invariants:" << std::endl;
            for (const auto &inv : this->invariants)
            {
                //todo
                // inv.dump();
            }
            std::cout << std::endl;
        }

        if (this->get_program_context() != NULL)
        {
            std::cout << "Function context set: "
                      << isl_set_to_str(this->get_program_context())
                      << std::endl;
        }

        std::cout << "Body " << std::endl;
        for (const auto &cpt : this->body)
        {
            cpt->dump();
        }
        std::cout << std::endl;

        for (const auto &buf : this->placeholders_list)
        {
            std::cout << "Placeholder name: " << buf.second->get_name() << std::endl;
            buf.second->dump(false);
        }

        std::cout << std::endl << std::endl;
    }
}

int polyfp::function::get_max_identity_schedules_range_dim() const
{
    int max_dim = 0;
    for (const auto &comp : this->get_computations())
    {
        isl_map *sched = comp->gen_identity_schedule_for_time_space_domain();
        int m = isl_map_dim(sched, isl_dim_out);
        max_dim = std::max(max_dim, m);
    }
    return max_dim;
}

const std::vector<std::string> &function::get_iterator_names() const
{
    return iterator_names;
}

isl_ast_node *function::get_isl_ast() const
{
    assert((ast != NULL) && ("You should generate an isl ast first (gen_isl_ast())."));

    return ast;
}

isl_ast_node *function::get_isl_ast1() const
{
    assert((ast != NULL) && ("You should generate an isl ast first (gen_isl_ast())."));

    return ast;
}

isl_union_set *polyfp::function::get_iteration_domain() const
{
    isl_union_set *result = NULL;
    isl_space *space = NULL;

    if (!this->body.empty())
    {
        space = isl_set_get_space(this->body[0]->get_iteration_domain());
    }
    else
    {
        return NULL;
    }

    assert(space != NULL);
    result = isl_union_set_empty(space);

    for (const auto &cpt : this->body)
    {
        isl_set *cpt_iter_space = isl_set_copy(cpt->get_iteration_domain());
        result = isl_union_set_union(isl_union_set_from_set(cpt_iter_space), result);
    }

    return result;
}

isl_union_map *polyfp::function::get_aligned_identity_schedules() const
{
    isl_union_map *result;
    isl_space *space;

    if (this->body.empty() == false)
    {
        space = isl_map_get_space(this->body[0]->gen_identity_schedule_for_time_space_domain());
    }
    else
    {
        return NULL;
    }
    assert(space != NULL);
    result = isl_union_map_empty(space);

    int max_dim = this->get_max_identity_schedules_range_dim();
    for (const auto &comp : this->get_computations())
    {
        isl_map *sched = comp->gen_identity_schedule_for_time_space_domain();
        // polyfp::str_dump("Identity schedule for time space domain: ", isl_map_to_str(sched));
        assert((sched != NULL) && "Identity schedule could not be computed");
        sched = isl_map_align_range_dims(sched, max_dim);
        result = isl_union_map_union(result, isl_union_map_from_map(sched));
    }

    return result;
}

void function::dump_sched_graph_dfs(compute * comp,
                                    std::unordered_set<compute *> &visited)
{
    // Do not visit anything that was already returned
    if (visited.find(comp) != visited.end())
        return;

    visited.insert(comp);

    for (auto &edge: this->sched_graph[comp])
    {
        const std::string level = ((edge.second == compute::root_dimension) ?
                                   "root" :
                                   std::to_string(edge.second));
        polyfp::str_dump(comp->get_name() +
                                    "=[" + level + "]=>" +
                                    edge.first->get_name());
        std::cout<<" ";
        
        dump_sched_graph_dfs(edge.first, visited);
    }
}

bool function::is_sched_graph_tree_dfs(compute * comp,
                                       std::unordered_set<compute *> &visited)
{
    // Do not visit anything that was already returned
    if (visited.find(comp) != visited.end())
        return false;

    visited.insert(comp);

    for (auto &edge: this->sched_graph[comp])
    {
        if (!is_sched_graph_tree_dfs(edge.first, visited))
            return false;
    }

    return true;
}

bool function::is_sched_graph_tree()
{
    if (this->starting_computations.size() != 1)
    {
        return false;
    }

    // Contains all nodes that have been visited
    std::unordered_set<compute *> visited;

    for (auto &comp: this->starting_computations)
    {
        if (!is_sched_graph_tree_dfs(comp, visited))
        {
            return false;
        }
    }
    return true;
}

void function::dump_sched_graph()
{
    // polyfp::str_dump("Number of schedule graph roots is " +
    //                             std::to_string(this->starting_computations.size()));

    polyfp::str_dump("Number of schedule graph roots is " +
                                std::to_string(this->starting_computations.size()));
    std::cout<<std::endl;
                                
    polyfp::str_dump("The roots are:");
    std::cout<<std::endl;

    for (auto root: this->starting_computations){
        polyfp::str_dump(" * " + root->get_name());
        std::cout<<std::endl;
    }
    // Contains all nodes that have been visited
    std::unordered_set<compute *> visited;
    polyfp::str_dump("Displaying schedule graph");
    std::cout<<std::endl;

    for (auto &comp: this->starting_computations)
    {
        dump_sched_graph_dfs(comp, visited);
    }
    std::cout<<std::endl;
    polyfp::str_dump("Finished displaying schedule graph");
}

void function::gen_ordering_schedules()
{

    if(this->is_sched_graph_tree())
    {
        // polyfp::str_dump("this->is_sched_graph_tree(): true.");
        std::priority_queue<int> level_to_check;
        std::unordered_map<int, std::deque<compute *>> level_queue;
        auto current_comp = *(this->starting_computations.begin());

        bool comps_remain = true;
        while(comps_remain)
        {
            for (auto &edge: this->sched_graph[current_comp])
            {
                if (level_queue[edge.second].size() == 0)
                    level_to_check.push(edge.second);

                level_queue[edge.second].push_back(edge.first);
            }

            comps_remain = level_to_check.size() > 0;
            if (comps_remain)
            {
                int fuse_level = level_to_check.top();
                auto next_comp = level_queue[fuse_level].front();
                level_queue[fuse_level].pop_front();
                next_comp->after_low_level((*current_comp), fuse_level);
                current_comp = next_comp;
                if (level_queue[fuse_level].size() == 0)
                    level_to_check.pop();
            }
        }
    }
    else
    {
        polyfp::str_dump("this->is_sched_graph_tree(): false.");
    }
}

int polyfp::function::get_max_schedules_range_dim() const
{
    int max_dim = 0;
    for (const auto &comp : this->get_computations())
    {
        isl_map *sched = comp->get_schedule();
        int m = isl_map_dim(sched, isl_dim_out);
        max_dim = std::max(max_dim, m);
    }
    return max_dim;
}

isl_set *function::get_program_context() const
{
    if (context_set != NULL)
    {
        return isl_set_copy(context_set);
    }
    else
    {
        return NULL;
    }
}

void polyfp::function::align_schedules()
{
    int max_dim = this->get_max_schedules_range_dim();

    for (auto &comp : this->get_computations())
    {
        isl_map *dup_sched = comp->get_schedule();
        assert((dup_sched != NULL) && "Schedules should be set before calling align_schedules");
        dup_sched = isl_map_align_range_dims(dup_sched, max_dim);
        comp->set_schedule(dup_sched);
        // polyfp::str_dump("Generated time-space domain:", isl_map_to_str(dup_sched));
        comp->name_unnamed_time_space_dimensions();
    }
}
std::string function::get_name(){
    return this->name;
}
void function::gen_time_space_domain()
{
    this->gen_ordering_schedules();

    this->align_schedules();

    for (auto &comp : this->get_computations())
    {
        comp->gen_time_space_domain();
    }
}

void function::gen_loop_location()
{
    auto leader_list = this->leader_computations;
    // std::cout<<leader_list.size()<<std::endl;
    for (auto &a_leader : leader_list)
    {   
        // std::cout << "leader name: ";
        // std::cout << a_leader->get_name() <<'\n';
        if(a_leader->is_leader == true)
        {
            if(a_leader->after_level!= -2)
            {
                int level = a_leader->after_level;
                int current_level = level;
                int counter = 0;
                // auto dim_list = a_leader->get_loop_level_names();
                auto dim_list = a_leader->final_loop_level_names;
                for(int i=0; i<dim_list.size(); i++)
                {
                    auto next_level = this->global_location;
                    a_leader->iterators_location_map.insert(std::make_pair(dim_list[counter],next_level));
                    this->global_location+=1;
                    counter+=1;
                }
            }
            else
            {
                auto nms = a_leader->final_loop_level_names;
                for (int i = 0; i< nms.size(); i++)
                {
                    a_leader->iterators_location_map.insert(std::make_pair(nms[i],i));
                    this->global_location = nms.size();     
                }
            }
            
        }
        // std::cout <<a_leader->iterators_location_map.size()<<'\n';
        // for(auto &map: a_leader->iterators_location_map){
        //     std::cout<<map.first<<": "<<map.second<<std::endl;
        // }
            
        auto components = a_leader->components;
        // sort components by their value
        std::vector<std::pair<polyfp::compute*, int>> temp;
        for (auto it = components.begin(); it != components.end(); it++)
            temp.push_back(std::make_pair(it->first, it->second));

        std::sort(temp.begin(), temp.end(), [](const std::pair<polyfp::compute*, int> &x, const std::pair<polyfp::compute*, int> &y) -> int {
            return x.second < y.second;
        });

        for (auto it = temp.begin(); it != temp.end(); it++)
        {
            // std::cout << it->first->get_name() << ':' << it->second << '\n';
            // std::cout <<it->first->after_level <<'\n';
            auto comp = it->first;
            int level = comp->after_level;
            int current_level = level;
            int counter = 0;
            auto dim_list = comp->final_loop_level_names;
            auto leader_dim_map = comp->leader->iterators_location_map;
            if(level!=-1)
            {   
                for(int i=0; i<dim_list.size(); i++)
                {
                    if(counter <= current_level)
                    {
                        comp->iterators_location_map.insert(std::make_pair(dim_list[counter],leader_dim_map[dim_list[counter]]));
                    }else
                    {
                        // auto fct = global::get_implicit_function();
                        auto next_level = this->global_location;
                        comp->iterators_location_map.insert(std::make_pair(dim_list[counter],next_level));
                        this->global_location += 1;
                    }
                    counter+=1;
                }
            }else{
                // TODO
            }
        }
        
    }

}

isl_union_map *polyfp::function::get_schedule() const
{
    isl_union_map *result = NULL;
    isl_space *space = NULL;

    if (!this->body.empty())
    {
        space = isl_map_get_space(this->body[0]->get_schedule());
    }
    else
    {
        return NULL;
    }

    assert(space != NULL);
    result = isl_union_map_empty(isl_space_copy(space));

    for (const auto &cpt : this->body)
    {
        isl_map *m = isl_map_copy(cpt->get_schedule());
        result = isl_union_map_union(isl_union_map_from_map(m), result);
    }

    result = isl_union_map_intersect_domain(result, this->get_iteration_domain());

    return result;
}

isl_union_set *polyfp::function::get_trimmed_time_processor_domain() const
{
    isl_union_set *result = NULL;
    isl_space *space = NULL;
    if (!this->body.empty())
    {
        space = isl_set_get_space(this->body[0]->get_trimmed_time_processor_domain());
    }
    else
    {
        return NULL;
    }
    assert(space != NULL);

    result = isl_union_set_empty(space);

    for (const auto &cpt : this->body)
    {
        isl_set *cpt_iter_space = isl_set_copy(cpt->get_trimmed_time_processor_domain());
        result = isl_union_set_union(isl_union_set_from_set(cpt_iter_space), result);
    }
    return result;
}

const std::map<std::string, polyfp::placeholder *> &function::get_placeholders() const
{
    return placeholders_list;
}

const std::map<std::string, polyfp::placeholder *> &function::get_fct_arguments() const
{
    return fct_argument_list;
}

const std::map<std::string, polyfp::placeholder *> &function::get_global_arguments() const
{
    return global_argument_list;
}

void function::add_placeholder(std::pair <std::string, polyfp::placeholder *> buf)
{
    assert(!buf.first.empty() && ("Empty buffer name."));
    assert((buf.second != NULL) && ("Empty buffer."));

    this->placeholders_list.insert(buf);  
}

void function::add_fct_argument(std::pair <std::string, polyfp::placeholder *> buf)
{
    assert(!buf.first.empty() && ("Empty buffer name."));
    assert((buf.second != NULL) && ("Empty buffer."));
    this->fct_argument_list.insert(buf);
}

void function::add_global_argument(std::pair <std::string, polyfp::placeholder *> buf)
{
    assert(!buf.first.empty() && ("Empty buffer name."));
    assert((buf.second != NULL) && ("Empty buffer."));
    this->global_argument_list.insert(buf);
}

void function::add_fct_argument()
{
    this->fct_argument_added = true;
}


isl_union_map *polyfp::function::compute_dep_graph() 
{
    isl_union_map *result = NULL;

    for (const auto &consumer : this->get_computations()) {

        isl_union_map *accesses_union_map = NULL;
        std::vector < isl_map * > accesses_vector;
        generator::get_rhs_accesses(this, consumer, accesses_vector, false);

        if (!accesses_vector.empty()) 
        {
            if (accesses_union_map == NULL) {
                isl_space *space = isl_map_get_space(accesses_vector[0]);
                assert(space != NULL);
                accesses_union_map = isl_union_map_empty(space);
            }
            for (size_t i = 0; i < accesses_vector.size(); ++i) {
                isl_map *reverse_access = isl_map_reverse(accesses_vector[i]);
                accesses_union_map = isl_union_map_union(isl_union_map_from_map(reverse_access),
                                                         accesses_union_map);
            }

            //accesses_union_map = isl_union_map_intersect_range(accesses_union_map, isl_union_set_from_set(isl_set_copy(consumer->get_iteration_domain())));
            //accesses_union_map = isl_union_map_intersect_domain(accesses_union_map, isl_union_set_from_set(isl_set_copy(consumer->get_iteration_domain())));

            polyfp::str_dump("Accesses after filtering.");
            polyfp::str_dump(isl_union_map_to_str(accesses_union_map));

            if (result == NULL) 
            {
                result = isl_union_map_copy(accesses_union_map);
                isl_union_map_free(accesses_union_map);
            } else {
                result = isl_union_map_union(result, accesses_union_map);
            }
        }
    }

    if (result != NULL)
    {
        polyfp::str_dump(isl_union_map_to_str(result));
    }
    else
    {
        polyfp::str_dump("Null.");
    }
    return result;
}

void function::gen_isl_ast()
{
    // Check that time_processor representation has already been computed,
    assert(this->get_trimmed_time_processor_domain() != NULL);
    assert(this->get_aligned_identity_schedules() != NULL);

    isl_ctx *ctx = this->get_isl_ctx();
    assert(ctx != NULL);
    isl_ast_build *ast_build;

    if (this->get_program_context() == NULL)
    {
        ast_build = isl_ast_build_alloc(ctx);
    }
    else
    {
        ast_build = isl_ast_build_from_context(isl_set_copy(this->get_program_context()));
    }

    isl_options_set_ast_build_atomic_upper_bound(ctx, 1);
    isl_options_get_ast_build_exploit_nested_bounds(ctx);
    isl_options_set_ast_build_group_coscheduled(ctx, 1);

    ast_build = isl_ast_build_set_after_each_for(ast_build, &polyfp::for_code_generator_after_for,
                NULL);
    // ast_build = isl_ast_build_set_at_each_domain(ast_build, &polyfp::generator::stmt_code_generator,
    //             this);

    isl_id_list *iterators = isl_id_list_alloc(ctx, this->get_iterator_names().size());
    if (this->get_iterator_names().size() > 0)
    {
        std::string name = generate_new_variable_name();
        isl_id *id = isl_id_alloc(ctx, name.c_str(), NULL);
        iterators = isl_id_list_add(iterators, id);

        for (int i = 0; i < this->get_iterator_names().size(); i++)
        {
            name = this->get_iterator_names()[i];
            id = isl_id_alloc(ctx, name.c_str(), NULL);
            iterators = isl_id_list_add(iterators, id);

            name = generate_new_variable_name();
            id = isl_id_alloc(ctx, name.c_str(), NULL);
            iterators = isl_id_list_add(iterators, id);
        }

        ast_build = isl_ast_build_set_iterators(ast_build, iterators);
    }

    // Intersect the iteration domain with the domain of the schedule.
    isl_union_map *umap =
        isl_union_map_intersect_domain(
            isl_union_map_copy(this->get_aligned_identity_schedules()),
            isl_union_set_copy(this->get_trimmed_time_processor_domain()));

    // polyfp::str_dump("Schedule:", isl_union_map_to_str(this->get_schedule()));
    // polyfp::str_dump("Iteration domain:",
    //                             isl_union_set_to_str(this->get_iteration_domain()));
    // polyfp::str_dump("Trimmed Time-Processor domain:",
    //                             isl_union_set_to_str(this->get_trimmed_time_processor_domain()));
    // polyfp::str_dump("Trimmed Time-Processor aligned identity schedule:",
    //                             isl_union_map_to_str(this->get_aligned_identity_schedules()))  ;        
    // polyfp::str_dump("Identity schedule intersect trimmed Time-Processor domain:",
    //                             isl_union_map_to_str(umap));    
    const char *s; 
    s = "[N,M,K] -> {s_2[i,j,k] -> [0, i, 0, j, 0, k, 10] : 0 <= i <= N and 0 <= j <= M and 0 <= k <= K; s_1[i, j, k] -> [0, i, 0, j, 0, k, 0] : 0 <= i <= N and 0 <= j <= M and 0 <= k <= 1 }";
    // s_2[0, i, 0, j, 0, k, 0] -> [0, i' = i, 0, j' = j, 0, k' = k, 0] : 0 <= i <= 4095 and 0 <= j <= 4095 and 0 <= k <= 4095; 
    // s_1[0, i, 0, j, 0, k, 10] -> [0, i' = i, 0, j' = j, 0, k' = k, 10] : 0 <= i <= 4095 and 0 <= j <= 4095 and 0 <= k <= 4095       
    isl_union_map *fmap = isl_union_map_read_from_str(ctx,s);

    this->ast = isl_ast_build_node_from_schedule_map(ast_build, umap);

    isl_ast_build_free(ast_build);

}


void polyfp::function::check_loop_fusion()
{
    for (auto &comp: this->leader_computations)
    {   
        // comp->get_loads_stores();
        comp->load_vector.clear();
        comp->store_vector.clear();
        comp->map_loadstores.clear();
        comp->get_all_loadstores();
        // comp->dump_components();
        // comp->dump_loads_stores();
        comp->dump_all_loadstores();
    }
    auto temp_computations = this->leader_computations;
    std::vector<int> leader_list;
    int leader_num = temp_computations.size();
    // for(int i=0; i<leader_num; i++){
    //     auto comp_from = leader_computations[i];
    //     int comp_from_index = this->leader_computation_index[comp_from];
    //     leader_list.push_back(comp_from_index);
    // }
    for(int i=0; i<leader_num-1; i++)
    {
        auto comp_first = leader_computations[i];
        auto comp_second = leader_computations[i+1];
        if(comp_first->get_name()!=comp_second->get_name())
        {
            bool has_edge = false;
            for(auto &store: comp_first->store_vector)
            {
                for(auto &load: comp_second->load_vector)
                {
                    if(store->get_name() == load->get_name())
                    {
                        has_edge = true;
                    }
                }
            }
            if(has_edge == false)
            {
                auto ndim_first = comp_first->get_loop_levels_number();
                auto ndim_second = comp_second->get_loop_levels_number();
                auto dim_first = comp_first->get_iteration_variables();
                auto dim_second = comp_second->get_iteration_variables();
                bool is_legal = true;                                                                                                                                                
                if(ndim_first == ndim_second)
                {
                    for(int i=0; i<ndim_first; i++)
                    {
                        if(stoi(dim_first[i].get_upper().to_str())!=stoi(dim_second[i].get_upper().to_str())||stoi(dim_first[i].get_lower().to_str())!=stoi( dim_second[i].get_lower().to_str()))
                        {
                            is_legal = false;
                        }
                    }
                }
                else
                {
                    is_legal = false;
                }
                if(is_legal == true)
                {
                    comp_second->after(comp_first, ndim_first-1);
                    comp_second->refused = true;
                    this->refused = true;
                   
                    for(int i=0; i<comp_first->get_loop_level_names().size(); i++)
                    {
                        comp_second->temp_access_map.insert(std::pair(comp_second->get_loop_level_names()[i],comp_first->get_loop_level_names()[i]));
                    }
                    
                    std::vector<polyfp::expr> new_placeholder_index;
                    auto temp_placeholder_index = comp_first->get_placeholder_dims();
                    auto original_placeholder_index = comp_second->get_placeholder_dims();
                    for(int i=0; i<original_placeholder_index.size(); i++)
                    {
                        //TODO: the index is not a var: e.g. i+1
                        auto tvar = comp_second->get_placeholder_dims()[i];
                        tvar.set_name(comp_second->temp_access_map[comp_second->get_placeholder_dims()[i].get_name()]);
                        new_placeholder_index.push_back(tvar);
                        // for(auto &kv: original_placeholder_index){
                        //     if(kv.get_expr_type() == polyfp::e_op){
                        //     }else{
                        //         auto t = temp_placeholder_index[i].get_name();
                        //         std::cout<<t;
                        //         if(kv.get_name() == t){
                        //             new_placeholder_index.push_back(kv);
                        //         }
                        //     }
                        //     // std::cout<<t;
                        // }
                        // std::cout<<"step 3 success"<<std::endl;
                    }
                    comp_second->set_placeholder_dims(new_placeholder_index);
                    comp_second->set_loop_level_names(comp_first->get_loop_level_names());
                }                    
            }
        }
    }   
}


void polyfp::function::dependence_analysis()
{
    auto temp_computations = this->leader_computations;
    for(auto &comp: temp_computations)
    {
        comp->compute_dependence_vectors();
        comp->auto_loop_transformation();
    }
    auto modified_computations = this->leader_computations;

    if(temp_computations.size()<=10)
    {
        this->check_loop_fusion();
    }
    
}


void polyfp::function::dfs(int pos, int top, int end, int map[500][500], int n, int v[100],int stack[550])//从pos点开始访问
{   
    // std::cout<<"DFSING"<<std::endl;

	int i;
	if(pos==end)
    {
        std::vector<long> path;
		for(i=0;i<top;i++)
        {
            path.push_back(stack[i]);
		}
        path.push_back(end);
        this->paths.push_back(path);
		return;
	}
	v[pos]=1; 
	stack[top++]=pos;
	for(i=1;i<=n;i++)
    {
		if(!v[i]&&map[pos][i])
			this->dfs(i,top,end,map,n,v,stack);
	}
	v[pos]=0;
	top--;
}

void polyfp::function::compute_dependency_graph(){

    int map[500][500]={0};

    std::vector<int> leader_list;
    int leader_num = this->leader_computations.size();
    for(int i=0; i<leader_num; i++)
    {
        auto comp_from = leader_computations[i];
        int comp_from_index = this->leader_computation_index[comp_from];
        leader_list.push_back(comp_from_index);
    }
    for(int i=0; i<leader_num; i++)
    {
        auto comp_from = leader_computations[i];
        int comp_from_index = this->leader_computation_index[comp_from];
        for(int j=i; j<leader_num; j++)
        {
            auto comp_to = leader_computations[j];
            int comp_to_index = this->leader_computation_index[comp_to];
            if(comp_from->get_name()!=comp_to->get_name())
            {
                bool has_edge = false;
                for(auto &store: comp_from->store_vector)
                {
                    for(auto &load: comp_to->load_vector)
                    {
                        if(store->get_name() == load->get_name())
                        {
                            has_edge = true;
                        }
                    }
                }
                if(has_edge == true)
                {
                    map[comp_from_index][comp_to_index] = 1;
                    std::vector<int>::iterator it = find(leader_list.begin(), leader_list.end(), comp_to_index);
                    if ( it!=leader_list.end())
                    {
                        leader_list.erase(it);
                    }
                }
            }
        }
    }


    std::vector<int> leafs;

    for(auto &comp: this->leader_computations)
    {
        if(comp->is_leaf == true)
        {
            leafs.push_back(this->leader_computation_index[comp]);
        }
    }
    if(this->leader_computations.size()<=30)
    {
        for(auto &leader: leader_list){
            for(auto &leaf: leafs){
                int stack[550],v[500]={0},top=0,n=this->leader_computations.size(),start=leader,end=leaf;
                this->dfs(start,top,end,map,n,v,stack);
            }
        }
    }
}


void polyfp::function::auto_DSE(std::string path)
{
    this->auto_DSE_loop_transformation();
    for(auto &comp: this->leader_computations)
    {
        if(comp->is_skewed_inDSE == true)
        {
            this->dump_schedule(path);
            return;
        }
    }
    this->evaluate_func();
    auto comp = this->update_latency();
    this->best_latency = this->current_latency;
    this->best_dsp_usage = 9999; 
    int factor = 1;

    this->auto_DSE_tile_size(comp,factor,path); 
    std::vector<int> temp;
    for(auto &comp: this->leader_computations)
    {
        comp->set_schedule(comp->original_schedule);
        comp->set_loop_level_names(comp->original_loop_level_name);
        comp->directive_map.clear();
        comp->is_unrolled = false;
        comp->unroll_factor.clear();
        comp->unroll_dimension.clear();
        comp->tile_map.clear();
        comp->tile_size_map.clear();
        comp->access_map.clear();
        comp->final_loop_level_names.clear();
        comp->final_loop_level_names = comp->final_loop_level_names_reserved;
        if(comp->is_optimized == true)
        {
            if(comp->final_strategy.size()!=0)
            {
                comp->apply_opt_strategy(comp->final_strategy);
            }
            else
            {
                auto iterators = comp->get_iteration_variables();
                int size = iterators.size();
                std::map<int,polyfp::var> iterator_map;
                for(auto &iter: iterators)
                {
                    int loc = comp->get_loop_level_number_from_dimension_name(iter.get_name());
                    iterator_map[loc] = iter;
                }
                if(size >= 3)
                {
                    comp->pipeline(iterator_map[size-3+2],1);
                }else if(size == 2)
                {
                    comp->pipeline(iterator_map[1],1);
                }else if(size == 1)
                {
                    comp->pipeline(iterator_map[0],1);
                }
            }
        }
        else
        {
            auto iterators = comp->get_iteration_variables();
            int size = iterators.size();
            std::map<int,polyfp::var> iterator_map;
            for(auto &iter: iterators)
            {
                int loc = comp->get_loop_level_number_from_dimension_name(iter.get_name());
                iterator_map[loc] = iter;
            }
            if(size >= 3)
            {
                comp->pipeline(iterator_map[size-3+2],1);
            }
            else if(size == 2)
            {
                comp->pipeline(iterator_map[1],1);
            }
            if(size == 1)
            {
                comp->pipeline(iterator_map[0],1);
            }
        }
    }
    this->dump_schedule(path);
}
void polyfp::function::auto_DSE_loop_transformation()
{
    for (auto &comp: this->leader_computations)
    {   
        comp->get_all_loadstores();
        comp->dump_all_loadstores();
    }
    this->dependence_analysis();

    for (int i=0; i<this->leader_computations.size(); i++)
    {
        this->leader_computation_index[leader_computations[i]] = i;
        this->leader_computations[i]->original_schedule = leader_computations[i]->get_schedule();
        std::vector<std::string> current_name_list = this->leader_computations[i]->get_loop_level_names();
        int final_size = this->leader_computations[i]->final_loop_level_names.size();
        int current_size = current_name_list.size();
        if(final_size == current_size)
        {
            this->leader_computations[i]->final_loop_level_names = current_name_list;
            this->leader_computations[i]->final_loop_level_names_reserved = current_name_list;
        }
        else if(final_size < current_size)
        {
            for(int i=0; i<final_size; i++){
                this->leader_computations[i]->final_loop_level_names[i] = current_name_list[i];
                this->leader_computations[i]->final_loop_level_names_reserved[i] = current_name_list[i];
            }
        }
        this->leader_computations[i]->original_loop_level_name = leader_computations[i]->get_loop_level_names();

        for(auto &part:this->leader_computations[i]->components)
        {
            part.first->original_loop_level_name = part.first->get_loop_level_names();
            part.first->original_schedule = part.first->get_schedule();
        }

        if(leader_computations[i]->is_leaf == true)
        {
            this->leaf_computations.push_back(leader_computations[i]);
        }
    }

    for(auto &comp:this->get_body())
    {
        std::vector<std::string> current_name_list = comp->get_loop_level_names();
        int final_size = comp->final_loop_level_names.size();
        int current_size = current_name_list.size();
        if(final_size == current_size)
        {
            comp->final_loop_level_names = current_name_list;
            comp->final_loop_level_names_reserved = current_name_list;
        }else if(final_size < current_size){
            for(int i=0; i<final_size; i++){
                comp->final_loop_level_names[i] = current_name_list[i];
                comp->final_loop_level_names_reserved[i] = current_name_list[i];
            }
        }
    }
    if(this->leader_computations.size()<=30){
        this->compute_dependency_graph();
    }
}

void polyfp::function::dump_schedule(std::string path)
{
    for(auto &comp: this->get_body())
    {
        comp->iterators_location_map.clear();
        this->global_location = 0;
    }
    this->gen_loop_location();
    this->gen_time_space_domain();
    this->gen_isl_ast();

    mlir::MLIRContext context;
    auto manager = polyfp::MLIRGenImpl(context);
    int level = 0;
    context.disableMultithreading();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::AffineDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::math::MathDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::scalehls::HLSDialect>();

    manager.mlirGen1(*this,this->get_isl_ast(),level,true, false, false);
    bool skew_flag = false;
    for(auto &comp : this->leader_computations)
    {
        if(comp->is_skewed_inDSE == true){
            skew_flag = true;
        }
        int index = this->leader_computation_index[comp];
        int position = manager.start_loops_position[index];
        for(auto &comp : this->leader_computations)
        {
            for(auto &kv : comp->get_directive_map())
            {
                if(kv.second == "pipeline")
                {
                    int loc_2 = comp->get_loop_level_number_from_dimension_name(kv.first);
                    int loc = comp->iterators_location_map[kv.first];
                    mlir::scalehls::setLoopDirective(manager.ops[loc], true, comp->II, false, false);
                    for(int i=1; i<=loc_2; i++)
                    {
                        mlir::scalehls::setLoopDirective(manager.ops[loc-i], false, comp->II, false, true);
                    }
                }
            }     
            for(auto &sub_comps: comp->components)
            {
                auto sub_comp = sub_comps.first;
                for(auto &kv : sub_comp->get_directive_map())
                {
                    if(kv.second == "pipeline"){
                        int loc_2 = sub_comp->get_loop_level_number_from_dimension_name(kv.first);
                        int loc = sub_comp->iterators_location_map[kv.first];
                        mlir::scalehls::setLoopDirective(manager.ops[loc], true, sub_comp->II, false, false);
                        for(int i=1; i<=loc_2; i++)
                        {
                            mlir::scalehls::setLoopDirective(manager.ops[loc-i], false, sub_comp->II, false, true);
                        }
                    }
                }  
            }                
        }       
    }

    auto map = manager.get_argument_map();

    mlir::scalehls::setTopFuncAttr(manager.get_funcs()[0]);
    mlir::scalehls::applyFuncPreprocess(manager.get_funcs()[0], true);
                
    for(auto &comp: this->leader_computations)
    {
        auto iterators = comp->get_iteration_variables();
        int size = iterators.size();
        if(size==1)
        {
            var i0("i0"), j0("j0"),k0("k0"), i1("i1"), j1("j1"),k1("k1");
        }
        if(comp->is_unrolled == true&&size!=1)
        {
            for(int i=0; i<comp->unroll_dimension.size(); i++)
            {
                int loc = comp->iterators_location_map[comp->unroll_dimension[i].get_name()];
                if(comp->unroll_factor[i] != -1)
                {
                    mlir::loopUnrollUpToFactor(manager.ops[loc],comp->unroll_factor[i]);
                }else{
                    mlir::loopUnrollFull(manager.ops[loc]);
                }
            }  
            for(auto &sub_comps:comp->components)
            {
                auto sub_comp = sub_comps.first;
                for(int i=0; i<comp->unroll_dimension.size(); i++)
                {
                    if(sub_comp->unroll_dimension.size()!=0)
                    {
                        int loc = sub_comp->iterators_location_map[sub_comp->unroll_dimension[i].get_name()];
                        if(sub_comp->unroll_factor[i] != -1)
                        {
                            mlir::loopUnrollUpToFactor(manager.ops[loc],sub_comp->unroll_factor[i]);
                        }
                        else
                        {
                            mlir::loopUnrollFull(manager.ops[loc]);
                        }
                    }
                }
            }
        }
    }

    mlir::scalehls::applyMemoryOpts(manager.get_funcs()[0]);
    mlir::scalehls::applyAutoArrayPartition(manager.get_funcs()[0]);

    if(this->refused == true){
    //TODO: there exists a bug in the latest version of POM, and the DSE of partition factor is not working.
    //      This may affect the best array partition results generated by POM. We will fix the bug as soon as possible.
        auto temp_p = this->get_placeholders();
        auto temp_p_d = temp_p["A"];
        if(temp_p_d->get_dim_sizes()[0] == 32)
        {
            this->set_partition("A",{16,16},{"cyclic","cyclic"});
        }else if(temp_p_d->get_dim_sizes()[0] == 64)
        {
            this->set_partition("A",{32,32},{"cyclic","cyclic"});
        }else
        {
            this->set_partition("A",{16,32},{"cyclic","cyclic"});
        }
        auto map = manager.get_array_map();
        for(auto &kv: this->get_partition_map())
        {
            SmallVector<mlir::scalehls::hls::PartitionKind, 4> kinds;
            SmallVector<unsigned, 4> factors;
            for(auto &factor: std::get<1>(kv))
            {
                factors.push_back(factor);
            }
            for(auto &type: std::get<2>(kv))
            {
                if(type == "cyclic"){
                    kinds.push_back(mlir::scalehls::hls::PartitionKind::CYCLIC);
                }else if(type == "block"){
                    kinds.push_back(mlir::scalehls::hls::PartitionKind::BLOCK);
                }else if(type == "none"){
                    kinds.push_back(mlir::scalehls::hls::PartitionKind::NONE);
                }
            }
            mlir::scalehls::applyArrayPartition(manager.get_funcs()[0].getArgument(map[std::get<0>(kv)]), factors, kinds,/*updateFuncSignature=*/true);
            // manager.getModule().dump();
        }
    }
    
    SmallVector<int64_t, 8> factors;
    std::string errorMessage;
    std::string pwd = std::filesystem::current_path().parent_path();
    auto configFile = mlir::openInputFile(pwd+"/samples/config.json", &errorMessage);
    if (!configFile) 
    {
      llvm::errs() << errorMessage << "\n";
    }
    auto config = llvm::json::parse(configFile->getBuffer());
    if (!config) 
    {
      llvm::errs() << "failed to parse the target spec json file\n";
    }
    auto configObj = config.get().getAsObject();
    if (!configObj) 
    {
      llvm::errs() << "support an object in the target spec json file, found "
                      "something else\n";
    }
    unsigned maxDspNum =ceil(configObj->getInteger("dsp").getValueOr(220));
    this->dsp_max = maxDspNum;
    llvm::StringMap<int64_t> latencyMap;
    mlir::scalehls::getLatencyMap(configObj, latencyMap);
    llvm::StringMap<int64_t> dspUsageMap;
    mlir::scalehls::getDspUsageMap(configObj, dspUsageMap);
    // TODO: Parameterize initial parallel factor, max DSE iteration, max unroll && partition factor
    int loc = 0;
    int total_dsp = 0;
    long total_latency = 0;
    if(manager.start_loops_position.size() == 0)
    {
        manager.start_loops_position.push_back(0);
    }
    if(skew_flag == false)
    {
        mlir::scalehls::ScaleHLSEstimator(latencyMap, dspUsageMap, true).estimateFunc(manager.funcs[0]);
        for(auto &loop: manager.start_loops_position)
        {
            mlir::scalehls::ScaleHLSEstimator(latencyMap, dspUsageMap, true).estimateLoop(manager.ops[loop],manager.funcs[0]);
            // manager.getModule().dump(); 
            auto latency = mlir::scalehls::getTiming(manager.ops[loop]).getLatency();
            auto dspNum = mlir::scalehls::getResource(manager.ops[loop]).getDsp();
        }

    }
    
    auto module = manager.getModule();
    // mlir::verify(module);
    // if (mlir::failed(mlir::verify(module))) {
    //     module->emitError("module verification error");
    //     // module->dump();
    // }
    // module->dump();
    std::error_code error;
    std::string s = this->get_name();
    std::string path1 = path+s+".mlir";
    llvm::raw_fd_ostream os(path1, error);
    os << *module;
    // std::cout<<"Note: "+s+".cpp has been generated!"<<std::endl;

}
void polyfp::function::evaluate_func()
{
    for(auto &comp: this->get_body()){
        comp->iterators_location_map.clear();
        this->global_location = 0;
    }

    for(auto &comp: this->leader_computations){
        if(comp->is_optimized == true && this->current_opt_comp!= NULL &&this->current_opt_comp->get_name()!=comp->get_name())
        {
            comp->set_schedule(comp->original_schedule);
            comp->set_loop_level_names(comp->original_loop_level_name);
            comp->directive_map.clear();
            comp->is_unrolled = false;
            comp->unroll_factor.clear();
            comp->unroll_dimension.clear();
            comp->tile_map.clear();
            comp->tile_size_map.clear();
            comp->access_map.clear();
            comp->final_loop_level_names.clear();
            comp->final_loop_level_names = comp->final_loop_level_names_reserved;
            if(comp->final_strategy.size()!=0)
            {
                comp->apply_opt_strategy(comp->final_strategy);
            }else
            {
                auto iterators = comp->get_iteration_variables();
                int size = iterators.size();
                std::map<int,polyfp::var> iterator_map;

                for(auto &iter: iterators)
                {
                    int loc = comp->get_loop_level_number_from_dimension_name(iter.get_name());
                    // int loc = comp->iterators_location_map(iter.get_name());
                    // std::cout<<iter.get_name()<<std::endl;
                    iterator_map[loc] = iter;
                }
                if(size >= 3)
                {
                    comp->pipeline(iterator_map[size-3+2],1);
                    for(auto &sub_comps: comp->components)
                    {
                        auto sub_comp = sub_comps.first;
                        //TODO, right pipeline level
                        if(sub_comp->after_level !=2)
                        {   
                            sub_comp->pipeline(iterator_map[size-3+2],1);
                        }
                    }
                }
                else if(size == 2)
                {
                    comp->pipeline(iterator_map[1],1);
                    for(auto &sub_comps: comp->components)
                    {
                        auto sub_comp = sub_comps.first;
                        //TODO, right pipeline level
                        if(sub_comp->after_level !=1)
                        {             
                            sub_comp->pipeline(iterator_map[1],1);
                        }
                    }
                }else if(size == 1)
                {
                    comp->pipeline(iterator_map[0],1);
                }
            }   
        }
        else if(comp->is_optimized == true && this->current_opt_comp!= NULL &&this->current_opt_comp->get_name()==comp->get_name())
        {
            comp->set_schedule(comp->original_schedule);
            comp->set_loop_level_names(comp->original_loop_level_name);
            comp->directive_map.clear();
            comp->is_unrolled = false;
            comp->unroll_factor.clear();
            comp->unroll_dimension.clear();
            comp->tile_map.clear();
            comp->tile_size_map.clear();
            comp->access_map.clear();
            comp->final_loop_level_names.clear();
            comp->final_loop_level_names = comp->final_loop_level_names_reserved;
            if(comp->temp_strategy.size()!=0)
            {
                comp->apply_opt_strategy(comp->temp_strategy);
            }
            else if(comp->final_strategy.size()!=0)
            {
                comp->apply_opt_strategy(comp->final_strategy);
            }
            else
            {
                auto iterators = comp->get_iteration_variables();
                int size = iterators.size();
                std::map<int,polyfp::var> iterator_map;
                for(auto &iter: iterators)
                {
                    int loc = comp->get_loop_level_number_from_dimension_name(iter.get_name());
                    // int loc = comp->iterators_location_map(iter.get_name());
                    // std::cout<<iter.get_name()<<std::endl;
                    iterator_map[loc] = iter;
                }
                if(size >= 3)
                {
                    comp->pipeline(iterator_map[size-3+2],1);
                    for(auto &sub_comps: comp->components)
                    {
                        auto sub_comp = sub_comps.first;
                        // TODO, right pipeline level
                        if(sub_comp->after_level !=2){   
                            sub_comp->pipeline(iterator_map[size-3+2],1);
                        }
                    }
                }else if(size == 2)
                {
                    comp->pipeline(iterator_map[1],1);
                    for(auto &sub_comps: comp->components){
                        auto sub_comp = sub_comps.first;
                        // TODO, right pipeline level
                        if(sub_comp->after_level !=1)
                        {    
                            sub_comp->pipeline(iterator_map[1],1);
                        }
                    }
                }else if(size == 1)
                {
                    comp->pipeline(iterator_map[0],1);
                }  
            }
        }
        else if(this->current_opt_comp!= NULL && this->current_opt_comp->get_name()!=comp->get_name())
        {
            comp->set_schedule(comp->original_schedule);
            comp->set_loop_level_names(comp->original_loop_level_name);
            comp->directive_map.clear();
            comp->is_unrolled = false;
            comp->unroll_factor.clear();
            comp->unroll_dimension.clear();
            comp->tile_map.clear();
            comp->tile_size_map.clear();
            comp->access_map.clear();
            comp->final_loop_level_names.clear();
            comp->final_loop_level_names = comp->final_loop_level_names_reserved;

            auto iterators = comp->get_iteration_variables();
            int size = iterators.size();
            std::map<int,polyfp::var> iterator_map;
            for(auto &iter: iterators)
            {
                int loc = comp->get_loop_level_number_from_dimension_name(iter.get_name());
                iterator_map[loc] = iter;
            }
            if(size >= 3)
            {
                comp->pipeline(iterator_map[size-3+2],1);
                for(auto &sub_comps: comp->components){
                    auto sub_comp = sub_comps.first;
                    // TODO, right pipeline level
                    if(sub_comp->after_level !=2){   
                        sub_comp->pipeline(iterator_map[size-3+2],1);
                    }
                }
            }else if(size == 2)
            {
                comp->pipeline(iterator_map[1],1);
                for(auto &sub_comps: comp->components)
                {
                    auto sub_comp = sub_comps.first;
                    // TODO, right pipeline level
                    if(sub_comp->after_level !=1)
                    {   
                        sub_comp->pipeline(iterator_map[1],1);
                    }
                }
            }else if(size == 1)
            {
                comp->pipeline(iterator_map[0],1);
            }
        }
        else if(this->current_opt_comp!= NULL && this->current_opt_comp->get_name()==comp->get_name())
        {
            comp->set_schedule(comp->original_schedule);
            comp->set_loop_level_names(comp->original_loop_level_name);
            comp->directive_map.clear();
            comp->is_unrolled = false;
            comp->unroll_factor.clear();
            comp->unroll_dimension.clear();
            comp->tile_map.clear();
            comp->tile_size_map.clear();
            comp->access_map.clear();
            comp->final_loop_level_names.clear();
            comp->final_loop_level_names = comp->final_loop_level_names_reserved;
            auto iterators = comp->get_iteration_variables();

            int size = iterators.size();
            std::map<int,polyfp::var> iterator_map;
            if(comp->temp_strategy.size()!=0)
            {
                comp->apply_opt_strategy(comp->temp_strategy);
            }
            else
            {
                auto iterators = comp->get_iteration_variables();
                int size = iterators.size();
                std::map<int,polyfp::var> iterator_map;
                for(auto &iter: iterators)
                {
                    int loc = comp->get_loop_level_number_from_dimension_name(iter.get_name());
                    iterator_map[loc] = iter;
                }
                if(size >= 3)
                {
                    comp->pipeline(iterator_map[size-3+2],1);

                    for(auto &sub_comps: comp->components)
                    {
                        auto sub_comp = sub_comps.first;
                        // TODO, right pipeline level
                        if(sub_comp->after_level !=2)
                        {   
                            sub_comp->pipeline(iterator_map[size-3+2],1);
                        }
                    }
                }
                else if(size == 2)
                {
                    comp->pipeline(iterator_map[1],1);
                    for(auto &sub_comps: comp->components){
                        auto sub_comp = sub_comps.first;
                        // TODO, right pipeline level
                        if(sub_comp->after_level !=1)
                        {                               
                            sub_comp->pipeline(iterator_map[1],1);
                        }
                    }
                }
                else if(size == 1)
                {
                    comp->pipeline(iterator_map[0],1);
                } 
            }    
        }
        else{
            // std::cout<< comp->get_name()+"evaluation initialization failed"<<std::endl;
        }
    }

    this->gen_loop_location();
    this->gen_time_space_domain();
    this->gen_isl_ast();
    mlir::MLIRContext context;
    auto manager = polyfp::MLIRGenImpl(context);
    int level = 0;
    context.disableMultithreading();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::AffineDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::math::MathDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::scalehls::HLSDialect>();

    manager.mlirGen1(*this,this->get_isl_ast(),level,true, false, false);
    
    for(auto &comp : this->leader_computations)
    {
        int index = this->leader_computation_index[comp];
        int position = manager.start_loops_position[index];
        //TODO:
        for(auto &comp : this->leader_computations)
        {
            for(auto &kv : comp->get_directive_map())
            {
                if(kv.second == "pipeline")
                {
                    int loc_2 = comp->get_loop_level_number_from_dimension_name(kv.first);
                    int loc = comp->iterators_location_map[kv.first];

                    // index = loc + index;
                    mlir::scalehls::setLoopDirective(manager.ops[loc], true, comp->II, false, false);
                    for(int i=1; i<=loc_2; i++)
                    {
                        mlir::scalehls::setLoopDirective(manager.ops[loc-i], false, comp->II, false, true);
                    }
                }
            }             
        }          
    }

    auto map = manager.get_argument_map();
    mlir::scalehls::setTopFuncAttr(manager.get_funcs()[0]);
    mlir::scalehls::applyFuncPreprocess(manager.get_funcs()[0], true);
    
    for(auto &comp: this->leader_computations)
    {
        if(comp->is_unrolled == true)
        {
            for(int i=0; i<comp->unroll_dimension.size(); i++)
            {
                int loc = comp->iterators_location_map[comp->unroll_dimension[i].get_name()];
                // loc = loc + bias;
                if(comp->unroll_factor[i] != -1)
                {
                    mlir::loopUnrollUpToFactor(manager.ops[loc],comp->unroll_factor[i]);
                }
                else
                {
                    mlir::loopUnrollFull(manager.ops[loc]);
                } 
            }  
            for(auto &sub_comps:comp->components)
            {
                auto sub_comp = sub_comps.first;
                for(int i=0; i<sub_comp->unroll_dimension.size(); i++)
                {
                    int loc = sub_comp->iterators_location_map[sub_comp->unroll_dimension[i].get_name()];
                    if(sub_comp->unroll_factor[i] != -1)
                    {
                        mlir::loopUnrollUpToFactor(manager.ops[loc],sub_comp->unroll_factor[i]);
                    }
                    else
                    {
                        mlir::loopUnrollFull(manager.ops[loc]);
                    }
                }
            }
        }
    }

    mlir::scalehls::applyMemoryOpts(manager.get_funcs()[0]);
    mlir::scalehls::applyAutoArrayPartition(manager.get_funcs()[0]);
    SmallVector<int64_t, 8> factors;
    std::string errorMessage;
    std::string pwd = std::filesystem::current_path().parent_path();
    auto configFile = mlir::openInputFile(pwd+"/samples/config.json", &errorMessage);
    if (!configFile) {
      llvm::errs() << errorMessage << "\n";
    }
    auto config = llvm::json::parse(configFile->getBuffer());
    if (!config) {
      llvm::errs() << "failed to parse the target spec json file\n";
    }
    auto configObj = config.get().getAsObject();
    if (!configObj) {
      llvm::errs() << "support an object in the target spec json file, found "
                      "something else\n";
    }
    unsigned maxDspNum =ceil(configObj->getInteger("dsp").getValueOr(220));
    this->dsp_max = maxDspNum;
    auto name = this->get_name();

    // TODO: Vitis_HLS 2022.2 improved its scheduling methods 
    // and two data paths of test_3mm can be executed in parallel.
    // Therefore, the actual DSP usage is twice that estimated by the cost model.
    // A profiler needs to be added to analyze the potential parallel datapath 
    // and adjust the DSP usage accordingly.
    if(name.substr(0, 8) == "test_3mm")
    {
        this->dsp_max = this->dsp_max/2;
    }

    llvm::StringMap<int64_t> latencyMap;
    mlir::scalehls::getLatencyMap(configObj, latencyMap);
    llvm::StringMap<int64_t> dspUsageMap;
    mlir::scalehls::getDspUsageMap(configObj, dspUsageMap);
    int loc = 0;
    int total_dsp = 0;
    long total_latency = 0;
    if(manager.start_loops_position.size() == 0)
    {
        manager.start_loops_position.push_back(0);
    }
    bool consistent_flag_flag;
    for(auto &loop: manager.start_loops_position )
    {
        mlir::scalehls::ScaleHLSEstimator(latencyMap, dspUsageMap, true).estimateLoop(manager.ops[loop],manager.funcs[0]);
        // manager.getModule().dump(); 
        auto latency = mlir::scalehls::getTiming(manager.ops[loop]).getLatency();
        // std::cout<<"latency: "+std::to_string(latency)<<std::endl;
        auto dspNum = mlir::scalehls::getResource(manager.ops[loop]).getDsp();
        // std::cout<<"dsp: "+std::to_string(dspNum)<<std::endl;
        auto minII = mlir::scalehls::getLoopInfo(manager.ops[loop]).getMinII();
        // std::cout<<"minII: "+std::to_string(minII)<<std::endl;
        this->leader_computations[loc]->latency = latency;
        this->leader_computations[loc]->dsp = dspNum;
        this->leader_computations[loc]->minII = minII;
        if(this->leader_computations[loc]->best_latency>=latency)
        {
            this->leader_computations[loc]->best_latency = latency;
        }
        else
        {
            if(this->current_opt_comp->get_name()==this->leader_computations[loc]->get_name())
            {
                consistent_flag_flag = true;
            }
            this->consistent_flag = false;
        }

        if(consistent_flag_flag==true)
        {
            this->consistent_flag  = true;
        }
        // total_dsp+=dspNum;
        total_latency+=latency;
        // std::cout<<"total_latency: "+std::to_string(total_latency)<<std::endl;
        this->latency_map[loc] = latency;
        this->resource_map[loc] = dspNum;
        loc+=1;  
    }
    mlir::scalehls::ScaleHLSEstimator(latencyMap, dspUsageMap, true).estimateFunc(manager.funcs[0]);
    total_dsp = mlir::scalehls::getResource(manager.funcs[0]).getDsp();
   
    this->dsp_usage = total_dsp;
    this->current_latency = total_latency;
    // std::cout<<"current latency"+std::to_string(total_latency)<<std::endl;
    // if(this->dsp_usage>this->dsp_max){
    //     this->new_strategy = false;
    // }
    // manager.getModule().dump(); 
}

void polyfp::function::auto_DSE_tile_size(polyfp::compute *comp, int factor, std::string path)
{
    // std::cout<<"Currently optimized compute: "<<comp->get_name()<<std::endl;
    int scale;
    //TODO components'domain is different from the leader's
    comp->set_schedule(comp->original_schedule);
    comp->set_loop_level_names(comp->original_loop_level_name);
    comp->directive_map.clear();
    comp->is_unrolled = false;
    comp->unroll_factor.clear();
    comp->unroll_dimension.clear();
    comp->tile_map.clear();
    comp->tile_size_map.clear();
    comp->access_map.clear();
    auto iterators = comp->get_iteration_variables();
    std::vector<polyfp::var> temp_iterators;
    int temp_size = iterators.size();
    
    if(temp_size>3)
    {
        int border = temp_size-3;
        for(auto &iter: iterators)
        {
            int loc = comp->get_loop_level_number_from_dimension_name(iter.get_name());
            if(loc>=border)
            {
                temp_iterators.push_back(iter);
            }
        }
        iterators.clear();
        iterators=temp_iterators;
    }

    std::vector<int> dim_ranges;
    std::map<int, std::vector<int>> dim_tile_sizes;
    bool not_2_pow = false;
    int count = 0;
    for(auto &iter: iterators)
    {
        int lower = stoi(iter.get_lower().to_str());
        int upper = stoi(iter.get_upper().to_str());
        int range = upper-lower;
        dim_ranges.push_back(range);
        std::vector<int> temp;
        if(range%32 != 0)
        {
            not_2_pow = true;
            for(int i=2; i<range; i++)
            {
                if(range % i == 0)
                {
                    if(i == 2)
                    {
                        temp.push_back(i);   
                    }
                    else if(i == 3||i==5 ||i==7)
                    {
                        temp.push_back(i);   
                    }
                }

            }
            if(temp.size()==0)
            {
                temp.push_back(1);  
            }
        }
        else
        {
            temp.push_back(1);   
        }
        dim_tile_sizes.insert(std::make_pair(count,temp));
        count++;
    }
    //TODO: SKEW MAP
    std::map<int,polyfp::var> iterator_map;
    int size = iterators.size();

    scale = 16*pow(2,factor-1);

    for(auto &iter: iterators)
    {
        int loc = comp->get_loop_level_number_from_dimension_name(iter.get_name());
        iterator_map[loc] = iter;
    }

    if(comp->is_optimized == true )
    {
        if(comp->current_factor < comp->largest_factor && comp->opt_finished == false)
        {
            comp->current_factor+=1;
            factor = comp->current_factor;
            scale = 16*pow(2,comp->current_factor-1);
        }
        else{
            this->finish_list.push_back(comp->get_name());
            if(comp->current_strategy.size()!=0)
            {
                comp->final_strategy = comp->current_strategy;
            }else{
                // TODO
                // std::cout<<"no final strategy"<<std::endl;       
            }
            if(this->leader_computations.size()!=1)
            {
                int path_index = this->get_longest_path();
                std::vector<long> current_longest_path = paths[path_index];
                std::vector<long> current_longest_path_latency;
                std::map<long, int> current_longest_map;
                int num = current_longest_path.size();
                
                for(int i=0; i<num; i++)
                {
                    long temp_latency = this->latency_map[current_longest_path[i]];
                    current_longest_path_latency.push_back(temp_latency);
                    current_longest_map.insert(std::make_pair(temp_latency,current_longest_path[i]));
                }
                std::sort(current_longest_path_latency.begin(),current_longest_path_latency.end(),std::greater<long>());
                
                
                for(int i=0; i<num; i++)
                {
                    int node_index = current_longest_path[i]; 
                    int final_index = this->path_map[path_index][node_index];
                    std::map<polyfp::compute *,int>::iterator it;
                    polyfp::compute *comp;
                    for( it= this->leader_computation_index.begin();it!=this->leader_computation_index.end();it++) 
                    {
                        if(it->second==final_index)
                        {
                            comp = it->first;
                            std::string name = comp->get_name();
                            if (std::find(finish_list.begin(), finish_list.end(), name) == finish_list.end())
                            {
                                auto_DSE_tile_size(comp, 1,path);
                                return;
                            }   
                        }
                            
                    } 
                } 
            }
            return;
        }

    }

    else
    {
        comp->is_optimized = true;
        comp->current_factor = factor;
    }

    int factor1=1;
    int factor2=1;
    int factor3=1;

    std::vector<std::vector<int>> tilesize_list;
    std::vector<int> current_design;
    std::vector<int> final_design;

    // std::vector<int> final_strategy;
    // std::vector<int> current_strategy;

    // Print header row.
    std::string s = this->get_name();
    std::string path1 = path+s+".csv";
    std::ifstream ifs(path1,std::ios::in);
    char ch;
    ifs>>ch;
    std::ofstream myfile;
    myfile.open(path1,std::ios::app);
    if(ifs.eof())
    {
        for (unsigned i = 0; i < size; ++i)
        {
            myfile << "l" << i << ",";


        }
        myfile << "cycle,dsp,ii\n";     
    }
 
    if(size >= 3)
    {
        // TODO, here 4 is desided by the scale

        if(not_2_pow == false)
        {
            //config: 5,3
            for(int i = 0; i<5+factor; i++)
            {
                factor1 = pow(2,i);
                for(int j = 0; j<3+factor-i; j++)
                {
                    factor2 = pow(2,j);
                    factor3 = scale/factor2/factor1;
                    tilesize_list.push_back({factor1,factor2,factor3});
                    // std::cout<<"tile factor: ";
                    // std::cout<<factor1;
                    // std::cout<<"; ";
                    // std::cout<<factor2;
                    // std::cout<<"; ";
                    // std::cout<<factor3<<std::endl;
                }
            }
        }else
        {
            std::vector<int> dim0 = dim_tile_sizes[0];
            std::vector<int> dim1 = dim_tile_sizes[1];
            std::vector<int> dim2 = dim_tile_sizes[2];
            if(dim0.size()==0)
            {
                dim0.push_back(1);
            }
            if(dim1.size()==0)
            {
                dim1.push_back(1);
            }
            for(auto &size0: dim0)
            {
                for(auto &size1: dim1)
                {
                    for(auto &size2: dim2)
                    {
                        tilesize_list.push_back({size0,size1,size2});
                        std::cout<<"tile factor: ";
                        std::cout<<size0;
                        std::cout<<"; ";
                        std::cout<<size1;
                        std::cout<<"; ";
                        std::cout<<size2<<std::endl;
                    }
                }
            }
            comp->current_factor=3;

        }

        bool larger_factor = true;
        if(larger_factor == true)
        {
            for(auto &tile_size: tilesize_list)
            {

                comp->set_schedule(comp->original_schedule);
                comp->set_loop_level_names(comp->original_loop_level_name);
                comp->directive_map.clear();
                comp->is_unrolled = false;
                comp->unroll_factor.clear();
                comp->unroll_dimension.clear();
                comp->tile_map.clear();
                comp->tile_size_map.clear();
                comp->access_map.clear();
                comp->opt_finished = false;

                var i0("i0"), j0("j0"),k0("k0"), i1("i1"), j1("j1"),k1("k1");
                if(tile_size[0]<=3 && tile_size[1]<=16 && tile_size[2]<=16)
                {
                // if(tile_size[0]<=16 && tile_size[1]<32 && tile_size[2]<32){
                // if(tile_size[0]<2 && tile_size[1]<4 && tile_size[2]<4){
                    int temp_index = comp->get_iteration_variables().size()-3;
                    // std::cout<<iterator_map[0].get_name()<<std::endl;
                    // std::cout<<iterator_map[1].get_name()<<std::endl;
                    // std::cout<<iterator_map[2].get_name()<<std::endl;
                    if(tile_size[2]==1 && tile_size[1]==1 && tile_size[0]==1)
                    {
                        
                    }else{
                        comp->tile(iterator_map[temp_index],iterator_map[temp_index+1],iterator_map[temp_index+2],tile_size[0],tile_size[1],tile_size[2],i0, j0, k0, i1, j1, k1);
                    }
                    
                    if(tile_size[2]!=1 && tile_size[1]!=1 && tile_size[0]!=1){
                        comp->pipeline(k0,1);
                        comp->unroll(k1,-1);
                        comp->unroll(j1,-1);
                        comp->unroll(i1,-1);
                    }
                    if(tile_size[2]!=1 && tile_size[1]!=1 && tile_size[0]==1){
                        comp->pipeline(k0,1);
                        comp->unroll(k1,-1);
                        comp->unroll(j1,-1);
                    }
                    if(tile_size[2]!=1 && tile_size[1]==1 && tile_size[0]!=1){
                        comp->pipeline(k0,1);
                        comp->unroll(k1,-1);
                        comp->unroll(i1,-1);
                    }
                    if(tile_size[2]!=1 && tile_size[1]==1 && tile_size[0]==1){
                        comp->pipeline(k0,1);
                        comp->unroll(k1,-1);
                        // comp->unroll(i1,-1);
                    }
                    if(tile_size[2]==1 && tile_size[1]==1 && tile_size[0]==1){
                        int lower = stoi(iterator_map[temp_index+2].get_lower().to_str());
                        int upper = stoi(iterator_map[temp_index+2].get_upper().to_str());
                        int range = upper-lower;
                        if(range<=7){
                            comp->pipeline(iterator_map[temp_index+1],1);
                            comp->unroll(iterator_map[temp_index+2],-1);
                        }
                    }
                    if(tile_size[2]==1 && tile_size[1]!=1 && tile_size[0]!=1){
                        int lower = stoi(iterator_map[temp_index+2].get_lower().to_str());
                        int upper = stoi(iterator_map[temp_index+2].get_upper().to_str());
                        int range = upper-lower;
                        if(range<=6){
                            comp->pipeline(j0,1);
                            comp->unroll(j1,-1);
                            comp->unroll(i1,-1);
                            comp->unroll(iterator_map[temp_index+2],-1);
                        }else{
                            comp->pipeline(iterator_map[temp_index+2],1);
                            comp->unroll(j1,-1);
                            comp->unroll(i1,-1);
                        }
                        
                    }
                    for(auto &part:comp->components){
                        part.first->set_schedule(part.first->original_schedule);
                        part.first->set_loop_level_names(part.first->original_loop_level_name);
                        part.first->tile(iterator_map[temp_index+0],iterator_map[temp_index+1],iterator_map[temp_index+2],tile_size[0],tile_size[1],tile_size[2],i0, j0, k0, i1, j1, k1);
                        if(tile_size[2]==1 && tile_size[1]!=1 && tile_size[0]!=1){
                            if(part.first->after_level == 2){
                                part.first->after(comp,j1);
                            }else if(part.first->after_level == 0){
                                part.first->after(comp,i0);
                                part.first->pipeline(iterator_map[temp_index+2],1);   
                            }
                            // part.first->after(comp,j1);
                        }else{
                            if(part.first->after_level == 2){
                                part.first->after(comp,k1);
                            }else if(part.first->after_level == 0){
                                part.first->after(comp,iterator_map[temp_index+0]);
                                part.first->pipeline(iterator_map[temp_index+2],1);   
                                //TODO
                                part.first->unroll(k1,-1);
                                part.first->unroll(j1,-1);
                            }
                            // part.first->after(comp,k1);
                        }
                    }
                    int II = 1;
                    this->current_opt_comp = comp;
                    //TODO
                    if(this->leader_computations.size() == -1){                          
                        this->evaluate_func();
                        if(this->current_latency < this->best_latency && this->dsp_max>= this->dsp_usage){
                            this->best_latency = this->current_latency;
                            this->best_dsp_usage = this->dsp_usage;
                            // std::cout<<"best_latency:  ";
                            // std::cout<<best_latency<<std::endl;
                            this->dump_schedule(path);
                        }

                    }else
                    {  
                        comp->temp_strategy = tile_size;
                        this->evaluate_func();
                        auto latency = comp->latency;
                        int dsp = comp->dsp;
                        // std::cout<<"schedule: "+std::to_string(tile_size[0])+", "+std::to_string(tile_size[1])+", "+std::to_string(tile_size[2])+": "+std::to_string(latency)+": "+std::to_string(dsp)<<std::endl;
                        // this->update_latency();
                        // std::cout<<"after evaluation"<<std::endl;
                        // auto new_comp = this->update_latency();
                        polyfp::compute * new_comp = NULL;
                        if((this->current_latency < this->best_latency || this->consistent_flag == false) && this->dsp_max>=this->dsp_usage){
                            auto comp = this->update_latency();
                            int path_index = this->get_longest_path();
                            std::vector<long> current_longest_path = paths[path_index];
                            std::vector<long> current_longest_path_latency;
                            std::map<long, int> current_longest_map;
                            int num = current_longest_path.size();
                            
                            for(int i=0; i<num; i++){
                                long temp_latency = this->latency_map[current_longest_path[i]];
                                current_longest_path_latency.push_back(temp_latency);
                                current_longest_map.insert(std::make_pair(temp_latency,current_longest_path[i]));
                            }
                            std::sort(current_longest_path_latency.begin(),current_longest_path_latency.end(),std::greater<long>());
                            bool comp_flag = false;
                            for(int i=0; i<num; i++)
                            {
                                int node_index = current_longest_path[i]; 
                                int final_index = this->path_map[path_index][node_index];
                                // int final_index = current_longest_map[current_longest_path_latency[i]];
                                // std::cout<<"the final_index"+std::to_string(final_index);
                                std::map<polyfp::compute *,int>::iterator it;
                                polyfp::compute *comp1;
                                for( it= this->leader_computation_index.begin();it!=this->leader_computation_index.end();it++) 
                                {
                                    if(it->second==final_index)
                                    {
                                        comp1 = it->first;
                                        std::string name = comp1->get_name();
                                        if (std::find(finish_list.begin(), finish_list.end(), name) == finish_list.end())
                                        {
                                            new_comp = comp1;
                                            comp_flag = true;
                                            
                                            break;
                                        }   
                                    }
                                        
                                } 
                                if(comp_flag == true)
                                {
                                    break;
                                }
                            }        
                            if(new_comp == NULL)
                            {
                                return;
                            }
                            if(new_comp->get_name() != comp->get_name() && this->dsp_max>=this->dsp_usage)
                            {
                                this->best_latency = this->current_latency;
                                final_design = tile_size;
                                break;
                            }else if(new_comp->get_name() == comp->get_name() &&this->current_latency < this->best_latency && this->dsp_max>= this->dsp_usage)
                            {
                                this->best_latency = this->current_latency;
                                this->best_dsp_usage = this->dsp_usage;               
                                current_design = tile_size;                             
                                long latency = comp->latency;
                                int dsp = comp->dsp;
                               
                            }else{
                                // TODO
                            }
                            auto latency = comp->latency;
                                int dsp = comp->dsp;                       
                        }
                      
                    }
                    
                    auto latency = comp->latency;
                        int dsp = comp->dsp;
                 
                    myfile << tile_size[0] << ",";
                    myfile << tile_size[1] << ",";
                    myfile << tile_size[2] << ",";
                    myfile << latency<< ",";
                    myfile << this->dsp_usage << ",";
                    myfile << comp->minII << "\n";

                }
                
            }
            if(final_design.size()!=0)
            {
                comp->final_strategy = final_design;
                comp->current_strategy = final_design;
                comp->apply_opt_strategy(comp->final_strategy);
                this->evaluate_func();
                auto new_comp = this->update_latency();
                auto_DSE_tile_size(new_comp, 1,path);
            }
            else if(current_design.size()!=0)
            {
                comp->current_strategy = current_design;
                comp->final_strategy = current_design;
                auto_DSE_tile_size(comp, 1,path);
            }else if(current_design.size()==0)
            {
                comp->opt_finished = true;
                auto_DSE_tile_size(comp, 1,path);

            }

        }
        myfile.close();
    }
    else if(size == 2)
    {
        if(not_2_pow == false)
        {
            for(int j = 0; j<2+factor; j++)
            {
                factor1 = pow(2,j);
                factor2 = scale/factor1;
                tilesize_list.push_back({factor1,factor2});     
            }
        }else{
            std::vector<int> dim0 = dim_tile_sizes[0];
            std::vector<int> dim1 = dim_tile_sizes[1];
            if(dim0.size()==0){
                dim0.push_back(1);
            }
            for(auto &size0: dim0)
            {
                for(auto &size1: dim1)
                {
                    tilesize_list.push_back({size0,size1});
                }
            }
            comp->current_factor=3;

        }
        
        bool larger_factor = true;
        for(auto &tile_size: tilesize_list)
        {
            comp->set_schedule(comp->original_schedule);
            comp->set_loop_level_names(comp->original_loop_level_name);
            comp->directive_map.clear();
            comp->is_unrolled = false;
            comp->unroll_factor.clear();
            comp->unroll_dimension.clear();
            comp->tile_map.clear();
            comp->tile_size_map.clear();
            comp->access_map.clear();
            comp->opt_finished = false;
            var i0("i0"), j0("j0"), i1("i1"), j1("j1");
            int lower1 = stoi(iterator_map[0].get_lower().to_str());
            int upper1 = stoi(iterator_map[0].get_upper().to_str());
            int range1 = upper1-lower1;
            int lower2 = stoi(iterator_map[1].get_lower().to_str());
            int upper2 = stoi(iterator_map[1].get_upper().to_str());
            int range2 = upper2-lower2;
            // if(tile_size[0]<=16 && tile_size[1]<=16){
            if(tile_size[0]<32 && tile_size[1]<=32 && range1>tile_size[0] && range2>tile_size[1])
            {
            // if(tile_size[0]<2 && tile_size[1]<4){
                // for(auto &iter: comp)
                // std::cout<<"size1"<<std::endl;
                // if(iterator_map[0].)
                comp->tile(iterator_map[0],iterator_map[1],tile_size[0],tile_size[1],i0, j0, i1, j1);
                if(tile_size[1]!=1&&tile_size[0]!=1)
                {
                    comp->pipeline(j0,1);
                    comp->unroll(j1,-1);
                    comp->unroll(i1,-1);
                }else if(tile_size[1]==1&&tile_size[0]!=1)
                {
                    comp->pipeline(iterator_map[1],1);
                    comp->unroll(i1,-1);
                }else if(tile_size[0]==1&&tile_size[1]!=1)
                {
                    comp->pipeline(j0,1);
                    comp->unroll(j1,-1);
                }
                for(auto &part:comp->components)
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
                            
                            part.first->after(comp,j1);
                        }else if(part.first->after_level == 0)
                        {
                            part.first->pipeline(j0,1);
                            part.first->after(comp,i0);
                            // part.first->unroll(j1,-1);
                            // part.first->unroll(i1,-1);
                        }
                        
                    }else if(tile_size[1]==1&&tile_size[0]!=1)
                    {
                        if(part.first->after_level == 1)
                        {
                            part.first->after(comp,i1);
                        }else if(part.first->after_level == 0)
                        {
                            part.first->after(comp,i0);
                            part.first->pipeline(iterator_map[1],1);
                            
                        }
                        // part.first->after(comp,i1);
                    }else if(tile_size[0]==1&&tile_size[1]!=1)
                    {
                        if(part.first->after_level == 1)
                        {
                            // part.first->unroll(j1,-1);
                            // std::cout<<"part.first->after(comp,j1);  "<<std::endl;
                            part.first->after(comp,j1);
                            
                        }else if(part.first->after_level == 0)
                        {
                            part.first->pipeline(j0,1);
                            part.first->after(comp,iterator_map[0]);
                            // std::cout<<"unroll dimension 2"<<std::endl;
                            part.first->unroll(j1,-1);
                        }
                    }
                
                }
                this->current_opt_comp = comp;
                if(this->leader_computations.size() == -1)
                {               
                    this->evaluate_func();
                    if(this->current_latency <= this->best_latency && this->dsp_max>= this->dsp_usage)
                    {
                        this->best_latency = this->current_latency;
                        this->best_dsp_usage = this->dsp_usage;
                        this->dump_schedule(path);
                    }
                    
                    if(this->dsp_max>this->dsp_usage)
                    {
                        larger_factor = true;
                        // auto_DSE_tile_size(new_comp, factor);
                    }

                }else
                {  
                        comp->temp_strategy = tile_size;
                        this->evaluate_func();
                        long latency = comp->latency;
                        int dsp = comp->dsp;
                        
                        polyfp::compute * new_comp = NULL;
                        if(this->current_latency < this->best_latency  && this->dsp_max>=this->dsp_usage)
                        {
                            auto comp = this->update_latency();
                            if(this->leader_computations.size()!=1)
                            {
                                int path_index = this->get_longest_path();
                                std::vector<long> current_longest_path = paths[path_index];
                                std::vector<long> current_longest_path_latency;
                                std::map<long, int> current_longest_map;
                                int num = current_longest_path.size();
                                
                                for(int i=0; i<num; i++)
                                {
                                    long temp_latency = this->latency_map[current_longest_path[i]];
                                    current_longest_path_latency.push_back(temp_latency);
                                    current_longest_map.insert(std::make_pair(temp_latency,current_longest_path[i]));
                                }
                                std::sort(current_longest_path_latency.begin(),current_longest_path_latency.end(),std::greater<long>());
                                bool comp_flag = false;
                                for(int i=0; i<num; i++)
                                {
                                    int node_index = current_longest_path[i]; 
                                    int final_index = this->path_map[path_index][node_index];
                                  
                                    std::map<polyfp::compute *,int>::iterator it;
                                    polyfp::compute *comp1;
                                    for( it= this->leader_computation_index.begin();it!=this->leader_computation_index.end();it++) 
                                    {
                                        if(it->second==final_index)
                                        {
                                            comp1 = it->first;
                                            std::string name = comp1->get_name();
                             
                                            if (std::find(finish_list.begin(), finish_list.end(), name) == finish_list.end()){
           
                                                new_comp = comp1;
                                                comp_flag = true;
                                                
                                                break;
                                            }   
                                        }
                                            
                                    } 
                                    if(comp_flag == true)
                                    {
                                        break;
                                    }
                                }        
                                if(new_comp == NULL)
                                {
                                    return;
                                }
                                if(new_comp->get_name() != comp->get_name() && this->dsp_max>=this->dsp_usage)
                                {
                                   
                                    this->best_latency = this->current_latency;
                                    final_design = tile_size;
                                    break;
                                }else if(new_comp->get_name() == comp->get_name() &&this->current_latency < this->best_latency && this->dsp_max>= this->dsp_usage)
                                {
                                    this->best_latency = this->current_latency;
                                    this->best_dsp_usage = this->dsp_usage;
                                    current_design = tile_size;
                                    auto latency = comp->latency;
                                    int dsp = comp->dsp;
                                }else{
                                   // TODO
                                }
                                auto latency = comp->latency;
                                    int dsp = comp->dsp;
                            }else{
                                new_comp = comp;
                                if(new_comp->get_name() == comp->get_name() &&this->current_latency < this->best_latency && this->dsp_max>= this->dsp_usage)
                                {
                                    this->best_latency = this->current_latency;
                                    this->best_dsp_usage = this->dsp_usage;
                                    current_design = tile_size;
                                    auto latency = comp->latency;
                                    int dsp = comp->dsp;
                                   
                                }else
                                {
                                    // TODO
                                }
                                auto latency = comp->latency;
                                    int dsp = comp->dsp;
                            }
                            
                               
                        }
                     
                    }
                
               
                auto latency = comp->latency;
                int dsp = comp->dsp;
                
               // TODO
                myfile << tile_size[0] << ",";
                myfile << tile_size[1] << ",";
                myfile << latency<< ",";
                myfile << this->dsp_usage << "\n";

            }
            

            
        }
        
        if(final_design.size()!=0)
        {               
            comp->final_strategy = final_design;
            comp->current_strategy = final_design;
            comp->apply_opt_strategy(comp->final_strategy);
            this->evaluate_func();
            auto new_comp = this->update_latency();
            auto_DSE_tile_size(new_comp, 1,path);
        }else if(current_design.size()!=0)
        {
            comp->current_strategy = current_design;
            auto_DSE_tile_size(comp, 1,path);
        }else if(current_design.size()==0||comp->current_factor == comp->largest_factor)
        {
            comp->opt_finished = true;
            auto_DSE_tile_size(comp, 1,path);

        }
        myfile.close();
    }
}


bool cmp_value(const std::pair<int, long> left, const std::pair<int,long> right)
{
	return left.second < right.second;
}

int polyfp::function::get_longest_path()
{
    auto i= std::max_element(this->all_latency_map.begin(),this->all_latency_map.end(),cmp_value);
    return i->first;
}

int polyfp::function::get_longest_node(std::vector<long> path)
{
    long max_latency = 0;
    long index = 0;
    for(int j=0; j<path.size(); j++)
    {
        if(max_latency < this->latency_map[path[j]]){
            max_latency = this->latency_map[path[j]];
            index = j;
        }
    }
    // std::cout<<"longest node: "+std::to_string(max_latency)+";"+std::to_string(index)<<std::endl;
    return index;
}
polyfp::compute * polyfp::function::update_latency(){

    for(int i=0; i<this->paths.size(); i++)
    {
        std::string result = "Latency of path:";
        long sum = 0;
        std::vector<int> node_list;
        for(int j=0; j<this->paths[i].size(); j++)
        {
            result += std::to_string(this->latency_map[this->paths[i][j]]);
            result += ";";
            sum+=this->latency_map[this->paths[i][j]];
            node_list.push_back(this->paths[i][j]);
            
        }
        this->path_map.insert(std::make_pair(i,node_list));
        result+=std::to_string(sum);
        this->all_latency_map[i] = sum;
    }
    // std::cout<<"this->all_latency_map.size()"<<std::endl;
    // std::cout<<this->all_latency_map.size()<<std::endl;
    // for(auto &pair:this->all_latency_map ){
    //     std::cout<<pair.first;
    //     std::cout<<", ";
    //     std::cout<<pair.second<<std::endl;

    // }
    polyfp::compute *comp;
    if(this->all_latency_map.size()!=0)
    {
        int path_index = this->get_longest_path();
        int node_index = this->get_longest_node(this->paths[path_index]);
        int final_index = this->path_map[path_index][node_index];
        this->longest_path = path_index;
        this->longest_node = node_index;
        // std::cout<<"path: ";
        // std::cout<<path_index<<std::endl;
        // std::cout<<"node: ";
        // std::cout<<node_index<<std::endl;
        std::map<polyfp::compute *,int>::iterator it;
        
        for( it= this->leader_computation_index.begin();it!=this->leader_computation_index.end();it++) 
        {
            if(it->second==final_index)
                comp = it->first;
        } 
    }
    else
    {
        comp = this->get_body()[0];
    }
    return comp;

}


void polyfp::function::codegen()
{   
    for(auto &comp:this->get_body())
    {
        std::vector<std::string> current_name_list = comp->get_loop_level_names();
        int final_size = comp->final_loop_level_names.size();
        int current_size = current_name_list.size();
        if(final_size == current_size)
        {
            comp->final_loop_level_names = current_name_list;
            comp->final_loop_level_names_reserved = current_name_list;
        }else if(final_size < current_size)
        {
            for(int i=0; i<final_size; i++)
            {
                comp->final_loop_level_names[i] = current_name_list[i];
                comp->final_loop_level_names_reserved[i] = current_name_list[i];
            }
        }
    }

    this->gen_loop_location();
    this->gen_time_space_domain();
    this->gen_isl_ast();
    this->gen_c_code();
    this->gen_mlir_stmt();

}



void polyfp::function::gen_c_code() const
{
    polyfp::str_dump("\n\n");
    polyfp::str_dump("\nC like code:\n");
    isl_printer *p;
    p = isl_printer_to_file(this->get_isl_ctx(), stdout);
    p = isl_printer_set_output_format(p, ISL_FORMAT_C);
    p = isl_printer_print_ast_node(p, this->get_isl_ast());
    isl_printer_free(p);
    polyfp::str_dump("\n\n");
}

       
}











