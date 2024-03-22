#include "expr.h"
#include "compute.h"
#include "function.h"
#include "core.h"
#include <filesystem>
#define N 4096

using namespace std;
using namespace polyfp;
int main(){
    std::string name = "test_blur_"+std::to_string(N);
    init(name);
    auto *fct = global::get_implicit_function();
    var i("i", 0 ,4094);
    var j("j", 0 ,4094);
    var c("c", 0 ,3);

    placeholder bx("bx",{N,N,3},p_float32);
    placeholder by("by",{N,N,3},p_float32);
    placeholder in("in",{N,N,3},p_float32);
    constant factor(3.0);
    
    compute s_1("s_1",{i,j,c},(in(i,j,c)+in(i,j+1,c)+in(i,j+2,c))/factor,bx(i,j,c));
    compute s_2("s_2",{i,j,c},(bx(i,j,c)+bx(i+1,j,c)+bx(i+2,j,c))/factor,by(i,j,c));
    var i0("i0"), j0("j0"),k0("k0"), i1("i1"), j1("j1"),k1("k1");
    s_2.after(s_1,-1);

    std::string pwd = std::filesystem::current_path().parent_path();
    std::string path = pwd+"/samples/blur/";
    fct->auto_DSE(path);
   
}
