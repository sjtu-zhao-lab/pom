#ifndef _H_polyfp_CORE_
#define _H_polyfp_CORE_

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

// #include "debug.h"
#include "expr.h"
#include "type.h"
#include "codegen.h"

namespace polyfp{

class compute;
class constant;
class generator;

void init(std::string name);
void init();
void codegen();

compute *get_computation_annotated_in_a_node(isl_ast_node *node);
int loop_level_into_dynamic_dimension(int level);
int loop_level_into_static_dimension(int level);
int dynamic_dimension_into_loop_level(int dim);

isl_map *add_eq_to_schedule_map(int dim0, int in_dim_coefficient, int out_dim_coefficient,
                                int const_conefficient, isl_map *sched);
}

#endif