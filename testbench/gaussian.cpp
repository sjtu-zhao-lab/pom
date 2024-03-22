#include "expr.h"
#include "compute.h"
#include "function.h"
#include "core.h"
#include <filesystem>
using namespace std;
using namespace polyfp;
#define N 4096
int main(){
    std::string name = "test_gaussian_"+std::to_string(N);
    init(name);

    auto *fct = global::get_implicit_function();
    var q("q", 0 ,4089);
    var w("w", 0 ,4089);
    var cc("cc", 0 ,3);
    var r("r", 0 ,7);
    var e("e", 0 ,7);

    placeholder temp("temp",{4096,4096,3},p_float32);
    placeholder src("src",{4096,4096,3},p_float32);
    placeholder conv("conv",{4096,4096,3},p_float32);
    placeholder kernelX("kernelX",{7},p_float32);
    placeholder kernelY("kernelY",{7},p_float32);
    constant scalar(0);
    
    compute s_1("s_1",{q,w,cc},scalar,temp(q,w,cc));
    compute s_2("s_2",{q,w,cc},scalar,conv(q,w,cc));
    compute s_3("s_3",{q,w,cc,r},temp(q,w,cc)+src(q + r,w,cc)*kernelX(r),temp(q,w,cc));
    compute s_4("s_4",{q,w,cc,e},conv(q,w,cc)+temp(q,w+e,cc)*kernelY(e),conv(q,w,cc));
    s_2.after(s_1,cc);
    s_3.after(s_1,-1);
    s_4.after(s_3,-1);

    std::string pwd = std::filesystem::current_path().parent_path();
    std::string path = pwd+"/samples/gaussian/";
    fct->auto_DSE(path);
}

