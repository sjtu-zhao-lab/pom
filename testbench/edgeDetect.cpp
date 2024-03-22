#include "expr.h"
#include "compute.h"
#include "function.h"
#include "core.h"
#include <filesystem>
using namespace std;
using namespace polyfp;
#define N 4096
int main(){
    std::string name = "test_edgeDetect_"+std::to_string(N);
    init(name);
    auto *fct = global::get_implicit_function();
    var i("i", 0 ,4094);
    var j("j", 0 ,4094);
    var c("c", 0 ,3);
    placeholder temp("temp",{4096,4096,3},p_float32);
    placeholder src("src",{4096,4096,3},p_float32);
    placeholder out("out",{4096,4096,3},p_float32);
    constant factor(8.0);
    compute s_1("s_1",{i,j,c},(src(i,j,c)+src(i,j+1,c)+src(i,j+2,c)+src(i+1,j,c)+src(i+1,j+2,c)+
                               src(i+2,j,c)+src(i+2,j+1,c)+src(i+2,j+2,c))/factor,temp(i,j,c));
    compute s_2("s_2",{i,j,c},temp(i+1,j+1,c)-temp(i+2,j,c)+
                              temp(i+2,j+1,c)-temp(i+1,j,c),out(i,j,c));
    s_2.after(s_1,-1);

    std::string pwd = std::filesystem::current_path().parent_path();
    std::string path = pwd+"/samples/edgeDetect/";
    fct->auto_DSE(path);

}
