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
    std::string name = "test_3mm_"+std::to_string(N);
    init(name);

    auto *fct = global::get_implicit_function();
    var i("i", 0 ,N);
    var j("j", 0 ,N);
    var k("k", 0 ,N);
    placeholder A("A",{N,N},p_float32);
    placeholder B("B",{N,N},p_float32);
    placeholder C("C",{N,N},p_float32);
    placeholder D("D",{N,N},p_float32);
    placeholder E("E",{N,N},p_float32);
    placeholder F("F",{N,N},p_float32);
    placeholder G("G",{N,N},p_float32);
    constant scalar(0);
    compute s_1("s_1",{i,j},scalar,E(i,j));
    compute s_2("s_2",{i,j,k},E(i,j)+A(i,k)*B(k,j),E(i,j));
    compute s_3("s_3",{i,j},scalar,F(i,j));
    compute s_4("s_4",{i,j,k},F(i,j)+C(i,k)*D(k,j),F(i,j));
    compute s_5("s_5",{i,j},scalar,G(i,j));
    compute s_6("s_6",{i,j,k},G(i,j)+E(i,k)*F(k,j),G(i,j));
    var i0("i0"), j0("j0"),k0("k0"), i1("i1"), j1("j1"),k1("k1");
    s_2.after(s_1,-1);
    s_3.after(s_2,-1);
    s_4.after(s_3,-1);
    s_5.after(s_4,-1);
    s_6.after(s_5,-1);
    // s_3.after(s_1,j);
    // s_2.after(s_1,-1);
    // s_4.after(s_2,k);
    
    // s_2.tile(k,j,i,1,1,16,i0, j0, k0, i1, j1,k1);
    // s_6.tile(j,i,2,16,i0, j0, k0, i1, j1,k1);
    // s_2.unroll(k1,-1);
    // s_2.unroll(j1,-1);
    // s_2.unroll(i1,-1);
    // s_4.unroll(k1,-1);
    // s_4.unroll(j1,-1);
    // s_4.unroll(i1,-1);
    // s_2.pipeline(k0,1);
    // s_4.pipeline(k0,1);
    // A.partition({16,2},"cyclic");
    // B.partition({2,2},"cyclic");
    // C.partition({2,2},"cyclic");
    // D.partition({16,2},"cyclic");
    // temp.partition({16,2},"cyclic");
    std::string pwd = std::filesystem::current_path().parent_path();
    std::string path = pwd+"/samples/3mm/";
    fct->auto_DSE(path);
    // codegen();
}
