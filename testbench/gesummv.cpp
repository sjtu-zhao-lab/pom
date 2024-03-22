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
    std::string name = "test_gesummv_"+std::to_string(N);
    init(name);
    auto *fct = global::get_implicit_function();
    var i("i", 0 ,N);
    var j("j", 0 ,N);
    var k("k", 0 ,N);
    placeholder temp("temp",{N},p_float32);
    placeholder A("A",{N,N},p_float32);
    placeholder B("B",{N,N},p_float32);
    placeholder x("x",{N},p_float32);
    placeholder y("y",{N},p_float32);
    constant alpha(1.6);
    constant beta(3.7);
       
    compute s_1("s_1",{i,j},temp(i)+A(i,j)*x(j),temp(i));
    compute s_2("s_2",{i,j},y(i)+B(i,j)*x(j),y(i));
    compute s_3("s_3",{i},alpha*temp(i)+beta*y(i),y(i));

    s_2.after(s_1,j);
    s_3.after(s_1,-1);
    
    std::string pwd = std::filesystem::current_path().parent_path();
    std::string path = pwd+"/samples/gesummv/";
    fct->auto_DSE(path);
}
