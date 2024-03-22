#include "expr.h"
#include "compute.h"
#include "function.h"
#include "core.h"
#include <filesystem>
using namespace std;
using namespace polyfp;
#define N 4096

int main(){
    std::string name = "test_heat_"+std::to_string(N);
    init(name);
    auto *fct = global::get_implicit_function();
    var i("i", 0 ,4094);
    var k("k", 0 ,4096);

    placeholder A("A",{4096},p_float32);
    placeholder B("B",{4096},p_float32);
    constant factor1(0.125);
    constant factor2(2.0);
    compute s_1("s_1",{k,i},(B(i)-factor2*B(i+1)+B(i+2))*factor1,A(i+1));
    compute s_2("s_2",{k,i},A(i+1),B(i+1));
    var i0("i0"), j0("j0"),k0("k0"), i1("i1"), j1("j1"),k1("k1");

    s_2.after(s_1,k);

    std::string pwd = std::filesystem::current_path().parent_path();
    std::string path = pwd+"/samples/heat/";
    fct->auto_DSE(path);
}
