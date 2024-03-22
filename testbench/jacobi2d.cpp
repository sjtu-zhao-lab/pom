#include "expr.h"
#include "compute.h"
#include "function.h"
#include "core.h"
#include <filesystem>
using namespace std;
using namespace polyfp;
#define N 4096

int main(){
    std::string name = "test_jacobi2d_"+std::to_string(N);
    init(name);
    auto *fct = global::get_implicit_function();
    var i("i", 0 ,4094);
    var j("j", 0 ,4094);
    // var j("j", 1 ,4095);
    var k("k", 0 ,4096);

    placeholder A("A",{4096,4096},p_float32);
    placeholder B("B",{4096,4096},p_float32);
    constant factor(0.2);
    
    compute s_1("s_1",{k,i,j},(A(i+1,j+1)+A(i+1,j)+A(i+1,j+2)+A(i+2,j+1)+A(i,j+1))*factor,B(i+1,j+1));
    compute s_2("s_2",{k,i,j},(B(i+1,j+1)+B(i+1,j)+B(i+1,j+2)+B(i+2,j+1)+B(i,j+1))*factor,A(i+1,j+1));
    var i0("i0"), j0("j0"),k0("k0"), i1("i1"), j1("j1"),k1("k1");

    s_2.after(s_1,k);

    std::string pwd = std::filesystem::current_path().parent_path();
    std::string path = pwd+"/samples/jacobi2d/";
    fct->auto_DSE(path);
}
