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
    std::string name = "test_2mm_"+std::to_string(N);
    init(name);

    auto *fct = global::get_implicit_function();
    var i("i", 0 ,N);
    var j("j", 0 ,N);
    var k("k", 0 ,N);
    placeholder A("A",{N,N},p_float32);
    placeholder B("B",{N,N},p_float32);
    placeholder C("C",{N,N},p_float32);
    placeholder D("D",{N,N},p_float32);
    placeholder temp("temp",{N,N},p_float32);
    constant alpha(1.6);
    constant beta(3.7);
    constant scalar(3.7);
    compute s_1("s_1",{i,j},scalar,temp(i,j));
    compute s_2("s_2",{i,j,k},temp(i,j)+alpha*A(i,k)*B(k,j),temp(i,j));
    compute s_3("s_3",{i,j},D(i,j)*beta,D(i,j));
    compute s_4("s_4",{i,j,k},D(i,j)+temp(i,k)*C(k,j),D(i,j));
    s_2.after(s_1,-1);
    s_3.after(s_2,-1);
    s_4.after(s_3,-1);
    // var i0("i0"), j0("j0"),k0("k0"), i1("i1"), j1("j1"),k1("k1");
    // s_2.tile(k,i,j,1,2,16,i0, j0, k0, i1, j1,k1);
    // s_4.tile(k,j,i,1,2,16,i0, j0, k0, i1, j1,k1);
    // s_2.unroll(k1,-1);
    // s_2.unroll(j1,-1);
    // s_2.unroll(i1,-1);
    // s_4.unroll(k1,-1);
    // s_4.unroll(j1,-1);
    // s_4.unroll(i1,-1);
    // s_1.pipeline(j,1);
    // s_1.pipeline(j,1);
    // s_2.pipeline(j,1);
    // s_3.pipeline(j,1);
    // s_4.pipeline(j,1);
    // A.partition({16,1},"cyclic");
    // B.partition({1,2},"cyclic");
    // C.partition({1,2},"cyclic");
    // D.partition({16,2},"cyclic");
    // temp.partition({16,2},"cyclic");
    std::string pwd = std::filesystem::current_path().parent_path();
    std::string path = pwd+"/samples/2mm/";
    fct->auto_DSE(path);
    // codegen();
}
