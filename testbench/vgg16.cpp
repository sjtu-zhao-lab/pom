#include "expr.h"
#include "compute.h"
#include "function.h"
#include "core.h"
#include <filesystem>
using namespace std;
using namespace polyfp;
#define K 4 // Size of convolution filter ( FOut xFIn x K x K)
#define N 32 // DATA_SET

int main(){
    init("test_vgg16");
    auto *fct = global::get_implicit_function();
    var o("o", 0 ,64);
    var y("y", 0 ,32); 
    var x("x", 0 ,32); 
    var i("i", 0 ,3);
    var p("p", 0 ,3);
    var q("q", 0 ,3);

    placeholder filter("filter",{64,3,3,3},p_float32);
    placeholder fo1("fo1",{64,32,32},p_float32);//{0,64,32,32}
    placeholder input("input",{3,34,34},p_float32);
    // placeholder temp("temp",{3,32,32},p_float32);
    // placeholder relu("relu",{64,32,32},p_float32);
    // placeholder input1("input1",{64,34,34},p_float32);
    constant scalar(0,p_float32);
    // omit initialisation of input and filter
    compute s_1("s_1",{o,y,x},scalar,fo1(o,y,x));
    compute s_2("s_2",{o,y,x,i,p,q},fo1(o,y,x)+input(i,y+p,x+q)*filter(o,i,p,q),fo1(o,y,x));
    // ReLU = max()
    compute s_3("s_3",{o,y,x},p_max(fo1(o,y,x),scalar),fo1(o,y,x));
    s_2.after(s_1,-1);
    s_3.after(s_2,-1);

    var i2("i2", 0 ,64);
    placeholder fo2("fo2",{64,32,32},p_float32);//{0,64,32,32}
    placeholder filter2("filter2",{64,64,3,3},p_float32);
    compute s_4("s_4",{o,y,x},scalar,fo2(o,y,x));
    compute s_5("s_5",{o,y,x,i2,p,q},fo2(o,y,x)+fo1(i2,y+p,x+q)*filter2(o,i2,p,q),fo2(o,y,x));
    // ReLU = max()
    compute s_6("s_6",{o,y,x},p_max(fo2(o,y,x),scalar),fo2(o,y,x));
    s_4.after(s_3,x);
    s_5.after(s_4,-1);
    s_6.after(s_5,-1);

    var y2("y2", 0 ,16); 
    var x2("x2", 0 ,16); 
    placeholder fo3("fo3",{64,16,16},p_float32);//{0,64,32,32}
    compute s_7("s_7",{o,y2,x2},scalar,fo3(o,y2,x2));
    // o3(o,y2,x2) = Max(o2(o,y2*2+p,x2*2+q), o3(o,y2,x2));
    compute s_8("s_8",{o,y2,x2,p,q},fo2(o,y2*2+p,x2*2+q),fo3(o,y2,x2));
    s_7.after(s_6,-1);
    s_8.after(s_7,-1);

    // Block 2
    var o2("o2", 0 ,128);
    placeholder fo4("fo4",{128,16,16},p_float32);//{0,64,32,32}
    placeholder filter3("filter3",{128,64,3,3},p_float32);
    compute s_9("s_9",{o2,y2,x2},scalar,fo4(o2,y2,x2));
    compute s_10("s_10",{o2,y2,x2,i2,p,q},fo4(o2,y2,x2)+fo3(i2,y2+p,x2+q)*filter3(o2,i2,p,q),fo4(o2,y2,x2));
    //ReLu = max()
    compute s_11("s_11",{o2,y2,x2},p_max(fo4(o2,y2,x2),scalar),fo4(o2,y2,x2));
    s_9.after(s_8,-1);
    s_10.after(s_9,-1);
    s_11.after(s_10,-1);

    var i3("i3", 0 ,128);
    placeholder fo5("fo5",{128,16,16},p_float32);//{0,64,32,32}
    placeholder filter4("filter4",{128,128,3,3},p_float32);
    compute s_12("s_12",{o2,y2,x2},scalar,fo5(o2,y2,x2));
    compute s_13("s_13",{o2,y2,x2,i3,p,q},fo5(o2,y2,x2)+fo4(i3,y2+p,x2+q)*filter4(o2,i3,p,q),fo5(o2,y2,x2));
    //ReLu = max()
    compute s_14("s_14",{o2,y2,x2},p_max(fo5(o2,y2,x2),scalar),fo5(o2,y2,x2));
    s_12.after(s_11,x2);
    s_13.after(s_12,-1);
    s_14.after(s_13,-1);

    var y3("y3", 0 ,8); 
    var x3("x3", 0 ,8); 
    placeholder fo6("fo6",{128,8,8},p_float32);//{0,64,32,32}
    compute s_15("s_15",{o2,y3,x3},scalar,fo6(o2,y3,x3));
    // o3(o,y2,x2) = Max(o2(o,y2*2+p,x2*2+q), o3(o,y2,x2));
    compute s_16("s_16",{o2,y3,x3,p,q},fo5(o2,y3*2+p,x3*2+q),fo6(o2,y3,x3));
    s_15.after(s_14,-1);
    s_16.after(s_15,-1);

    // Block 3
    var o3("o3", 0 ,256);
    placeholder fo7("fo7",{256,8,8},p_float32);//{0,64,32,32}
    placeholder filter5("filter5",{256,128,3,3},p_float32);
    compute s_17("s_17",{o3,y3,x3},scalar,fo7(o3,y3,x3));
    compute s_18("s_18",{o3,y3,x3,i3,p,q},fo7(o3,y3,x3)+fo6(i3,y3+p,x3+q)*filter5(o3,i3,p,q),fo7(o3,y3,x3));
    //ReLu = max()
    compute s_19("s_19",{o3,y3,x3},p_max(fo7(o3,y3,x3),scalar),fo7(o3,y3,x3));
    s_17.after(s_16,-1);
    s_18.after(s_17,-1);
    s_19.after(s_18,-1);

    var i4("i4", 0 ,256);
    placeholder fo8("fo8",{256,8,8},p_float32);//{0,64,32,32}
    placeholder fo9("fo9",{256,8,8},p_float32);//{0,64,32,32}
    placeholder filter6("filter6",{256,256,3,3},p_float32);
    compute s_20("s_20",{o3,y3,x3},scalar,fo8(o3,y3,x3));
    compute s_21("s_21",{o3,y3,x3,i4,p,q},fo8(o3,y3,x3)+fo7(i4,y3+p,x3+q)*filter6(o3,i4,p,q),fo8(o3,y3,x3));
    //ReLu = max()
    compute s_22("s_22",{o3,y3,x3},p_max(fo8(o3,y3,x3),scalar),fo8(o3,y3,x3));
    compute s_23("s_23",{o3,y3,x3},scalar,fo9(o3,y3,x3));
    compute s_24("s_24",{o3,y3,x3,i4,p,q},fo9(o3,y3,x3)+fo8(i4,y3+p,x3+q)*filter6(o3,i4,p,q),fo9(o3,y3,x3));
    //ReLu = max()
    compute s_25("s_25",{o3,y3,x3},p_max(fo9(o3,y3,x3),scalar),fo9(o3,y3,x3));
    s_20.after(s_19,x3);
    s_21.after(s_20,-1);
    s_22.after(s_21,-1);
    s_23.after(s_22,-1);
    s_24.after(s_23,-1);
    s_25.after(s_24,-1);

    var y4("y4", 0 ,4); 
    var x4("x4", 0 ,4); 
    placeholder fo10("fo10",{256,4,4},p_float32);//{0,64,32,32}
    compute s_26("s_26",{o3,y4,x4},scalar,fo10(o3,y4,x4));
    // o3(o,y2,x2) = Max(o2(o,y2*2+p,x2*2+q), o3(o,y2,x2));
    compute s_27("s_27",{o3,y4,x4,p,q},fo9(o3,y4*2+p,x4*2+q),fo10(o3,y4,x4));
    s_26.after(s_25,-1);
    s_27.after(s_26,-1);

    // Block 4
    var o4("o4", 0 ,512);
    placeholder fo11("fo11",{512,4,4},p_float32);//{0,64,32,32}
    placeholder filter7("filter7",{512,256,3,3},p_float32);
    compute s_28("s_28",{o4,y4,x4},scalar,fo11(o4,y4,x4));
    compute s_29("s_29",{o4,y4,x4,i4,p,q},fo11(o4,y4,x4)+fo10(i4,y4+p,x4+q)*filter7(o4,i4,p,q),fo11(o4,y4,x4));
    //ReLu = max()
    compute s_30("s_30",{o4,y4,x4},p_max(fo11(o4,y4,x4),scalar),fo11(o4,y4,x4));
    s_28.after(s_27,-1);
    s_29.after(s_28,-1);
    s_30.after(s_29,-1);

    var i5("i5", 0 ,512);
    placeholder fo12("fo12",{512,8,8},p_float32);//{0,64,32,32}
    placeholder fo13("fo13",{512,8,8},p_float32);//{0,64,32,32}
    placeholder filter8("filter8",{512,512,3,3},p_float32);
    compute s_31("s_31",{o4,y4,x4},scalar,fo12(o4,y4,x4));
    compute s_32("s_32",{o4,y4,x4,i5,p,q},fo12(o4,y4,x4)+fo11(i5,y4+p,x4+q)*filter8(o4,i5,p,q),fo12(o4,y4,x4));
    //ReLu = max()
    compute s_33("s_33",{o4,y4,x4},p_max(fo12(o4,y4,x4),scalar),fo12(o4,y4,x4));
    compute s_34("s_34",{o4,y4,x4},scalar,fo13(o4,y4,x4));
    compute s_35("s_35",{o4,y4,x4,i5,p,q},fo13(o4,y4,x4)+fo12(i5,y4+p,x4+q)*filter8(o4,i5,p,q),fo13(o4,y4,x4));
    //ReLu = max()
    compute s_36("s_36",{o4,y4,x4},p_max(fo13(o4,y4,x4),scalar),fo13(o4,y4,x4));
    s_31.after(s_30,x4);
    s_32.after(s_31,-1);
    s_33.after(s_32,-1);
    s_34.after(s_33,x4);
    s_35.after(s_34,-1);
    s_36.after(s_35,-1);

    var y5("y5", 0 ,2);
    var x5("x5", 0 ,2); 
    placeholder fo14("fo14",{512,2,2},p_float32);//{0,64,32,32}
    compute s_37("s_37",{o4,y5,x5},scalar,fo14(o4,y5,x5));
    // o3(o,y2,x2) = Max(o2(o,y2*2+p,x2*2+q), o3(o,y2,x2));
    compute s_38("s_38",{o4,y5,x5,p,q},fo13(o4,y5*2+p,x5*2+q),fo14(o4,y5,x5));
    s_37.after(s_36,-1);
    s_38.after(s_37,-1);

    // Block 5
    var o5("o5", 0 ,512);
    placeholder fo15("fo15",{512,2,2},p_float32);//{0,64,32,32}
    placeholder filter9("filter9",{512,512,3,3},p_float32);
    compute s_39("s_39",{o5,y5,x5},scalar,fo15(o5,y5,x5));
    compute s_40("s_40",{o5,y5,x5,i5,p,q},fo15(o5,y5,x5)+fo14(i5,y5+p,x5+q)*filter9(o5,i5,p,q),fo15(o5,y5,x5));
    //ReLu = max()
    compute s_41("s_41",{o5,y5,x5},p_max(fo15(o5,y5,x5),scalar),fo15(o5,y5,x5));
    s_39.after(s_38,-1);
    s_40.after(s_39,-1);
    s_41.after(s_40,-1);

    placeholder fo16("fo16",{512,2,2},p_float32);//{0,64,32,32}
    placeholder fo17("fo17",{512,2,2},p_float32);//{0,64,32,32}
    compute s_42("s_42",{o5,y5,x5},scalar,fo16(o5,y5,x5));
    compute s_43("s_43",{o5,y5,x5,i5,p,q},fo16(o5,y5,x5)+fo15(i5,y5+p,x5+q)*filter9(o5,i5,p,q),fo16(o5,y5,x5));
    //ReLu = max()
    compute s_44("s_44",{o5,y5,x5},p_max(fo16(o5,y5,x5),scalar),fo16(o5,y5,x5));
    compute s_45("s_45",{o5,y5,x5},scalar,fo17(o5,y5,x5));
    compute s_46("s_46",{o5,y5,x5,i5,p,q},fo17(o5,y5,x5)+fo16(i5,y5+p,x5+q)*filter9(o5,i5,p,q),fo17(o5,y5,x5));
    //ReLu = max()
    compute s_47("s_47",{o5,y5,x5},p_max(fo17(o5,y5,x5),scalar),fo17(o5,y5,x5));
    s_42.after(s_41,x5);
    s_43.after(s_42,-1);
    s_44.after(s_43,-1);
    s_45.after(s_44,-1);
    s_46.after(s_45,-1);
    s_47.after(s_46,-1);

    // var y5("y5", 0 ,2);
    // var x5("x5", 0 ,2); 
    placeholder fo18("fo18",{512},p_float32);//{0,64,32,32}
    compute s_48("s_48",{o5},scalar,fo18(o5));
    // o3(o,y2,x2) = Max(o2(o,y2*2+p,x2*2+q), o3(o,y2,x2));
    compute s_49("s_49",{o5,p,q},fo17(o5,2+p,2+q),fo18(o5));
    s_48.after(s_47,-1);
    s_49.after(s_48,-1);

    



    fct->auto_DSE_loop_transformation();
    for(auto &comp: fct->leader_computations){
        auto iterators = comp->get_iteration_variables();
        int size = iterators.size();
        if(size>=6){
          comp->apply_opt_strategy({8,1,1});
        }
    }
    std::string pwd = std::filesystem::current_path().parent_path();
    std::string path = pwd+"/samples/vgg16/";
    fct->dump_schedule(path);
}

