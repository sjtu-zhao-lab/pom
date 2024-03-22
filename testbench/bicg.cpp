#include "expr.h"
#include "compute.h"
#include "function.h"
#include "core.h"
#define N 4096
#include <filesystem>
using namespace std;
using namespace polyfp;
int main(){
    std::string name = "test_bicg_"+std::to_string(N);
    init(name);
    auto *fct = global::get_implicit_function();
    var i("i", 0 ,N);
    var j("j", 0 ,N);

    placeholder A("A",{N,N},p_float32);
    placeholder s("s",{N},p_float32);
    placeholder q("q",{N},p_float32);
    placeholder p("p",{N},p_float32);
    placeholder r("r",{N},p_float32);


    compute s_1("s_1",{i,j},s(j)+A(i,j)*r(i),s(j));
    compute s_2("s_2",{i,j},q(i)+A(i,j)*p(j),q(i));
    // compute s_2("s_2",{i,j},q(j)+A(j,i)*p(i),q(j));
    // s_2.interchange(i,j);
    s_2.after(s_1,j);

    var i0("i0"), j0("j0"),k0("k0"), i1("i1"), j1("j1"),k1("k1");
    // s_1.tile(i,j,1,32,i0, j0, i1, j1);
    // s_2.tile(i,j,1,32,i0, j0, i1, j1);
    // s_1.unroll(j1,-1);
    // s_2.unroll(j1,-1);
    // s_2.after(s_1,j1);
    // s_1.pipeline(j,1);
    // s.partition({32},"cyclic");
    // q.partition({32},"cyclic");
    // A.partition({16,16},"cyclic");

    std::string pwd = std::filesystem::current_path().parent_path();
    std::string path = pwd+"/samples/bicg/";
    fct->auto_DSE(path);

}

