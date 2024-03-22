#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/space.h>
#include <isl/constraint.h>
#include <filesystem>
#include <map>
#include <string.h>
#include <stdint.h>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <iostream>
#include <string>

#include "expr.h"
#include "compute.h"
#include "function.h"
#include "core.h"
// #include "mlir/IR/Attributes.h"
#define N 4096

using namespace std;
using namespace polyfp;
int main(){
    std::string name = "test_gemm_"+std::to_string(N);
    init(name);

    auto fct = global::get_implicit_function();
    var i("i", 0 ,N);
    var j("j", 0 ,N);
    var k("k", 0 ,N);
    placeholder A("A",{N,N},p_float32);
    placeholder B("B",{N,N},p_float32);
    placeholder C("C",{N,N},p_float32);
    constant alpha;
    constant beta;
    compute s_1("s_1",{i,j},C(i,j)*beta,C(i,j));
    compute s_2("s_2",{i,j,k},C(i,j)+alpha*A(i,k)*B(k,j),C(i,j));
    var i0("i0"), j0("j0"),k0("k0"), i1("i1"), j1("j1"),k1("k1");
    s_2.after(s_1,-1);
    // s_2.tile(k,j,i,2,2,16,i0, j0, k0, i1, j1,k1);
    // s_2.unroll(k1,-1);
    // s_2.unroll(j1,-1);
    // s_2.unroll(i1,-1);
    // s_1.pipeline(j,1);
    // s_2.pipeline(j,1);
    
    // s_1.tile(k,j,i,2,2,16,i0, j0, k0, i1, j1,k1);
    // s_1.unroll(k1,-1);
    // s_1.unroll(j1,-1);
    // s_1.unroll(i1,-1);
    // s_1.pipeline(k0,1);

    // s.tile(i, j, 4, 4, i0, j0, i1, j1);
    // A.partition({16,2},"cyclic");
    // B.partition({2,2},"cyclic");
    // C.partition({16,2},"cyclic");
    // codegen();
    
    std::string pwd = std::filesystem::current_path().parent_path();
    std::string path = pwd+"/samples/gemm/";
    fct->auto_DSE(path);

}   


