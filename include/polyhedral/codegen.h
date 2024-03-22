#ifndef _H_polyfp_CODEGEN_
#define _H_polyfp_CODEGEN_

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
// #include "expr.h"
// #include "type.h"
#include "function.h"
#include "compute.h"

namespace polyfp{

class var;
std::string generate_new_variable_name();
polyfp::expr traverse_expr_and_replace_non_affine_accesses(polyfp::compute *comp,
                                                             const polyfp::expr &exp);

}



#endif