#include "expr.h"
#include "compute.h"
#include "function.h"
#include "core.h"
#include <filesystem>
using namespace std;
using namespace polyfp;
#define N 4096

int main(){
    std::string name = "test_jacobi_"+std::to_string(N);
    init(name);
    auto *fct = global::get_implicit_function();
    var i("i", 0 ,4094);
    var t("t", 0 ,4096);

    placeholder A("A",{4096},p_float32);
    placeholder B("B",{4096},p_float32);
    constant factor(0.33333);
    compute s_1("s_1",{t,i},(B(i)+B(i+1)+B(i+2))*factor,A(i+1));
    compute s_2("s_2",{t,i},(A(i)+A(i+1)+A(i+2))*factor,B(i+1));
    s_2.after(s_1,t);
    // Choice 1: Hand optimization for users with expertise
    var i0("i0"), j0("j0"),k0("k0"), i1("i1"), j1("j1"),k1("k1");
    // s_1.tile(t,i,1,2,t0,i0,t1,i1);
    // s_2.tile(t,i,1,2,t0,i0,t1,i1);
    // s_1.pipeline(i0,1);
    // s_2.pipeline(i0,1);
    // s_1.unroll(i1,2);
    // s_2.unroll(i1,2);
    // A.partition({4},"cyclic");
    // B.partition({4},"cyclic");
    // codegen();
    // Choice 2: Automatic DSE for users without expertise
    std::string pwd = std::filesystem::current_path().parent_path();
    std::string path = pwd+"/samples/jacobi/";
    fct->auto_DSE(path);

}
