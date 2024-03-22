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
#include <filesystem>
#include "expr.h"
#include "compute.h"
#include "function.h"
#include "core.h"
// #include "mlir/IR/Attributes.h"
using namespace std;
using namespace polyfp;
int main(){
    init("test_seidel_4096");
    auto *fct = global::get_implicit_function();
    var i("i", 0 ,4094);
    var j("j", 0 ,4094);
    var k("k", 0 ,4096);

    placeholder A("A",{4096,4096},p_float32);
    placeholder B("B",{4096,4096},p_float32);
    constant factor(9);
    compute s_1("s_1",{k,i,j},(A(i,j+1)+A(i,j)+A(i,j+2)+A(i+1,j)+A(i+1,j+1)+A(i+1,j+2)+A(i+2,j)+A(i+2,j+1)+A(i+2,j+2))/factor,A(i+1,j+1));

    var i0("i0"), j0("j0"),k0("k0"), i1("i1"), j1("j1"),k1("k1");

    std::string pwd = std::filesystem::current_path().parent_path();
    std::string path = pwd+"/samples/seidel/";
    fct->auto_DSE(path);
}

}
