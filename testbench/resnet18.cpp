#include "expr.h"
#include "compute.h"
#include "function.h"
#include "core.h"
#include <filesystem>
using namespace std;
using namespace polyfp;
// #define K 4 // Size of convolution filter ( FOut xFIn x K x K)
// #define N 32 // DATA_SET

// polyfp::expr pmax(polyfp::expr left, polyfp::expr right){
//   return expr(polyfp::o_max, left, right);
// }
int main(){
    init("resnet18");
    auto *fct = global::get_implicit_function();
    var o("o", 0 ,64);
    var y("y", 0 ,32); 
    var x("x", 0 ,32); 
    var i("i", 0 ,3);
    var p("p", 0 ,3);
    var q("q", 0 ,3);

    // Block 1.1
    placeholder filter("filter",{64,3,3,3},p_float32);
    placeholder fo1("fo1",{64,32,32},p_float32);//{0,64,32,32}
    placeholder input("input",{3,32,32},p_float32);
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

    placeholder fo3("fo3",{64,32,32},p_float32);//{0,64,32,32}
    compute s_7("s_7",{o,y,x},scalar,fo3(o,y,x));
    compute s_8("s_8",{o,y,x,i2,p,q},fo3(o,y,x)+fo2(i2,y+p,x+q)*filter2(o,i2,p,q),fo3(o,y,x));
    // Residual
    placeholder fo4("fo4",{64,32,32},p_float32);
    compute s_9("s_9",{o,y,x},fo3(o,y,x)+fo1(o,y,x),fo4(o,y,x));
    // ReLU = max()
    compute s_10("s_10",{o,y,x},p_max(fo4(o,y,x),scalar),fo4(o,y,x));
    s_7.after(s_6,x);
    s_8.after(s_7,-1);
    s_9.after(s_8,-1);
    s_10.after(s_9,-1);

    // Block 1.2
    placeholder fo5("fo5",{64,32,32},p_float32);//{0,64,32,32}
    compute s_11("s_11",{o,y,x},scalar,fo5(o,y,x));
    compute s_12("s_12",{o,y,x,i2,p,q},fo5(o,y,x)+fo4(i2,y+p,x+q)*filter2(o,i2,p,q),fo5(o,y,x));
    // ReLU = max()
    compute s_13("s_13",{o,y,x},p_max(fo5(o,y,x),scalar),fo5(o,y,x));   
    s_11.after(s_10,x);
    s_12.after(s_11,-1);
    s_13.after(s_12,-1);

    placeholder fo6("fo6",{64,32,32},p_float32);
    compute s_14("s_14",{o,y,x},scalar,fo6(o,y,x));
    compute s_15("s_15",{o,y,x,i2,p,q},fo6(o,y,x)+fo5(i2,y+p,x+q)*filter2(o,i2,p,q),fo6(o,y,x));
    // Residual
    placeholder fo7("fo7",{64,32,32},p_float32);
    compute s_16("s_16",{o,y,x},fo6(o,y,x)+fo4(o,y,x),fo7(o,y,x));
    // ReLU = max()
    compute s_17("s_17",{o,y,x},p_max(fo7(o,y,x),scalar),fo7(o,y,x));
    s_14.after(s_13,x);
    s_15.after(s_14,-1);
    s_16.after(s_15,-1);
    s_17.after(s_16,-1);


    // Block 2.1
    var o2("o2", 0 ,128);
    var y2("y2", 0 ,16); 
    var x2("x2", 0 ,16); 
    placeholder fo8("fo8",{128,16,16},p_float32);//{0,64,32,32}
    placeholder filter3("filter3",{128,64,3,3},p_float32);
    compute s_18("s_18",{o2,y2,x2},scalar,fo8(o2,y2,x2));
    compute s_19("s_19",{o2,y2,x2,i2,p,q},fo8(o2,y2,x2)+fo7(i2,y2*2+p,x2*2+q)*filter3(o2,i2,p,q),fo8(o2,y2,x2));
    // ReLU = max()
    compute s_20("s_20",{o2,y2,x2},p_max(fo8(o2,y2,x2),scalar),fo8(o2,y2,x2));
    s_18.after(s_17,-1);
    s_19.after(s_18,-1);
    s_20.after(s_19,-1);

    var i3("i3", 0 ,128);
    placeholder fo9("fo9",{128,16,16},p_float32);//{0,64,32,32}
    placeholder filter4("filter4",{128,128,3,3},p_float32);
    compute s_21("s_21",{o2,y2,x2},scalar,fo9(o2,y2,x2));
    compute s_22("s_22",{o2,y2,x2,i3,p,q},fo9(o2,y2,x2)+fo8(i3,y2+p,x2+q)*filter4(o2,i3,p,q),fo9(o2,y2,x2));
    // transform
    placeholder fo10("fo10",{128,16,16},p_float32);//{0,64,32,32}
    placeholder temp1("temp1",{128,64},p_float32);
    compute s_23("s_23",{o2,y2,x2,i2},fo7(i2,y2*2,x2*2)*temp1(o2,i2)+fo10(o2,y2,x2),fo10(o2,y2,x2));
    // Residual
    placeholder fo11("fo11",{128,16,16},p_float32);//{0,64,32,32}
    compute s_24("s_24",{o2,y2,x2},fo10(o2,y2,x2)+fo9(o2,y2,x2),fo11(o2,y2,x2));
    // ReLU = max()
    compute s_25("s_25",{o2,y2,x2},p_max(fo11(o2,y2,x2),scalar),fo11(o2,y2,x2));
    s_21.after(s_20,x2);
    s_22.after(s_21,-1);
    s_23.after(s_22,-1);
    s_24.after(s_23,-1);
    s_25.after(s_24,-1);

    // Block 2.2
    placeholder fo12("fo12",{128,16,16},p_float32);//{0,64,32,32}
    compute s_26("s_26",{o2,y2,x2},scalar,fo12(o2,y2,x2));
    compute s_27("s_27",{o2,y2,x2,i3,p,q},fo12(o2,y2,x2)+fo11(i3,y2+p,x2+q)*filter4(o2,i3,p,q),fo12(o2,y2,x2));
    // ReLU = max()
    compute s_28("s_28",{o2,y2,x2},p_max(fo12(o2,y2,x2),scalar),fo12(o2,y2,x2));
    placeholder fo13("fo13",{128,16,16},p_float32);//{0,64,32,32}
    compute s_29("s_29",{o2,y2,x2},scalar,fo13(o2,y2,x2));
    compute s_30("s_30",{o2,y2,x2,i3,p,q},fo13(o2,y2,x2)+fo12(i3,y2+p,x2+q)*filter4(o2,i3,p,q),fo13(o2,y2,x2));
    // Residual
    placeholder fo14("fo14",{128,16,16},p_float32);//{0,64,32,32}
    compute s_31("s_31",{o2,y2,x2},fo13(o2,y2,x2)+fo11(o2,y2,x2),fo14(o2,y2,x2));
    // ReLU = max()
    compute s_32("s_32",{o2,y2,x2},p_max(fo14(o2,y2,x2),scalar),fo14(o2,y2,x2));
    s_26.after(s_25,x2);
    s_27.after(s_26,-1);
    s_28.after(s_27,-1);
    s_29.after(s_28,x2);
    s_30.after(s_29,-1);
    s_31.after(s_30,-1);
    s_32.after(s_31,-1);

    // Block 3.1
    var o3("o3", 0 ,256);
    var y3("y3", 0 ,8); 
    var x3("x3", 0 ,8); 
    placeholder fo15("fo15",{256,8,8},p_float32);//{0,64,32,32}
    placeholder filter5("filter5",{256,128,3,3},p_float32);
    compute s_33("s_33",{o3,y3,x3},scalar,fo15(o3,y3,x3));
    compute s_34("s_34",{o3,y3,x3,i3,p,q},fo15(o3,y3,x3)+fo14(i3,y3*2+p,x3*2+q)*filter5(o3,i3,p,q),fo15(o3,y3,x3));
    // ReLU = max()
    compute s_35("s_35",{o3,y3,x3},p_max(fo15(o3,y3,x3),scalar),fo15(o3,y3,x3));
    s_33.after(s_32,-1);
    s_34.after(s_33,-1);
    s_35.after(s_34,-1);

    var i4("i4", 0 ,256);
    placeholder fo16("fo16",{256,8,8},p_float32);//{0,64,32,32}
    placeholder filter6("filter6",{256,256,3,3},p_float32);
    compute s_36("s_36",{o3,y3,x3},scalar,fo16(o3,y3,x3));
    compute s_37("s_37",{o3,y3,x3,i4,p,q},fo16(o3,y3,x3)+fo15(i4,y3+p,x3+q)*filter6(o3,i4,p,q),fo16(o3,y3,x3));
    // transform
    placeholder fo17("fo17",{256,8,8},p_float32);//{0,64,32,32}
    placeholder temp2("temp2",{256,128},p_float32);
    compute s_38("s_38",{o3,y3,x3,i3},fo14(i3,y3*2,x3*2)*temp2(o3,i3)+fo17(o3,y3,x3),fo17(o3,y3,x3));
    // Residual
    placeholder fo18("fo18",{256,8,8},p_float32);//{0,64,32,32}
    compute s_39("s_39",{o3,y3,x3},fo17(o3,y3,x3)+fo16(o3,y3,x3),fo18(o3,y3,x3));
    // ReLU = max()
    compute s_40("s_40",{o3,y3,x3},p_max(fo18(o3,y3,x3),scalar),fo18(o3,y3,x3));
    s_36.after(s_35,x3);
    s_37.after(s_36,-1);
    s_38.after(s_37,-1);
    s_39.after(s_38,-1);
    s_40.after(s_39,-1);

    // Block 3.2
    placeholder fo19("fo19",{256,8,8},p_float32);//{0,64,32,32}
    compute s_41("s_41",{o3,y3,x3},scalar,fo19(o3,y3,x3));
    compute s_42("s_42",{o3,y3,x3,i4,p,q},fo19(o3,y3,x3)+fo18(i4,y3+p,x3+q)*filter6(o3,i4,p,q),fo19(o3,y3,x3));
    // ReLU = max()
    compute s_43("s_43",{o3,y3,x3},p_max(fo19(o3,y3,x3),scalar),fo19(o3,y3,x3));
    placeholder fo20("fo20",{256,8,8},p_float32);//{0,64,32,32}
    compute s_44("s_44",{o3,y3,x3},scalar,fo20(o3,y3,x3));
    compute s_45("s_45",{o3,y3,x3,i4,p,q},fo20(o3,y3,x3)+fo19(i4,y3+p,x3+q)*filter6(o3,i4,p,q),fo20(o3,y3,x3));
    // Residual
    placeholder fo21("fo21",{256,8,8},p_float32);//{0,64,32,32}
    compute s_46("s_46",{o3,y3,x3},fo20(o3,y3,x3)+fo18(o3,y3,x3),fo21(o3,y3,x3));
    // ReLU = max()
    compute s_47("s_47",{o3,y3,x3},p_max(fo21(o3,y3,x3),scalar),fo21(o3,y3,x3));
    s_41.after(s_40,x3);
    s_42.after(s_41,-1);
    s_43.after(s_42,-1);
    s_44.after(s_43,-1);
    s_45.after(s_44,-1);
    s_46.after(s_45,-1);
    s_47.after(s_46,-1);

    // Block 4.1
    var o4("o4", 0 ,512);
    var y4("y4", 0 ,4); 
    var x4("x4", 0 ,4); 
    placeholder fo22("fo22",{512,4,4},p_float32);//{0,64,32,32}
    placeholder filter7("filter7",{512,256,3,3},p_float32);
    compute s_48("s_48",{o4,y4,x4},scalar,fo22(o4,y4,x4));
    compute s_49("s_49",{o4,y4,x4,i4,p,q},fo22(o4,y4,x4)+fo21(i4,y4*2+p,x4*2+q)*filter7(o4,i4,p,q),fo22(o4,y4,x4));
    // ReLU = max()
    compute s_50("s_50",{o4,y4,x4},p_max(fo22(o4,y4,x4),scalar),fo22(o4,y4,x4));
    s_48.after(s_47,-1);
    s_49.after(s_48,-1);
    s_50.after(s_49,-1);

    var i5("i5", 0 ,512);
    placeholder fo23("fo23",{512,4,4},p_float32);//{0,64,32,32}
    placeholder filter8("filter8",{512,512,3,3},p_float32);
    compute s_51("s_51",{o4,y4,x4},scalar,fo23(o4,y4,x4));
    compute s_52("s_52",{o4,y4,x4,i5,p,q},fo23(o4,y4,x4)+fo22(i5,y4+p,x4+q)*filter8(o4,i5,p,q),fo23(o4,y4,x4));
    // transform
    placeholder fo24("fo24",{512,4,4},p_float32);//{0,64,32,32}
    placeholder temp3("temp3",{512,256},p_float32);
    compute s_53("s_53",{o4,y4,x4,i4},fo21(i4,y4*2,x4*2)*temp3(o4,i4)+fo24(o4,y4,x4),fo24(o4,y4,x4));
    // Residual
    placeholder fo25("fo25",{512,4,4},p_float32);//{0,64,32,32}
    compute s_54("s_54",{o4,y4,x4},fo24(o4,y4,x4)+fo23(o4,y4,x4),fo25(o4,y4,x4));
    // ReLU = max()
    compute s_55("s_55",{o4,y4,x4},p_max(fo25(o4,y4,x4),scalar),fo25(o4,y4,x4));
    s_51.after(s_50,x4);
    s_52.after(s_51,-1);
    s_53.after(s_52,-1);
    s_54.after(s_53,-1);
    s_55.after(s_54,-1);

    // Block 4.2
    placeholder fo26("fo26",{512,4,4},p_float32);//{0,64,32,32}
    compute s_56("s_56",{o4,y4,x4},scalar,fo26(o4,y4,x4));
    compute s_57("s_57",{o4,y4,x4,i5,p,q},fo26(o4,y4,x4)+fo25(i5,y4+p,x4+q)*filter8(o4,i5,p,q),fo26(o4,y4,x4));
    // ReLU = max()
    compute s_58("s_58",{o4,y4,x4},p_max(fo26(o4,y4,x4),scalar),fo26(o4,y4,x4));
    placeholder fo27("fo27",{512,4,4},p_float32);//{0,64,32,32}
    compute s_59("s_59",{o4,y4,x4},scalar,fo27(o4,y4,x4));
    compute s_60("s_60",{o4,y4,x4,i5,p,q},fo27(o4,y4,x4)+fo26(i5,y4+p,x4+q)*filter8(o4,i5,p,q),fo27(o4,y4,x4));
    // Residual
    placeholder fo28("fo28",{512,4,4},p_float32);//{0,64,32,32}
    compute s_61("s_61",{o4,y4,x4},fo27(o4,y4,x4)+fo25(o4,y4,x4),fo28(o4,y4,x4));
    // ReLU = max()
    compute s_62("s_62",{o4,y4,x4},p_max(fo28(o4,y4,x4),scalar),fo28(o4,y4,x4));
    s_56.after(s_55,x4);
    s_57.after(s_56,-1);
    s_58.after(s_57,-1);
    s_59.after(s_58,x4);
    s_60.after(s_59,-1);
    s_61.after(s_60,-1);
    s_62.after(s_61,-1);


    fct->auto_DSE_loop_transformation();
    int count=0;
    for(auto &comp: fct->leader_computations){
        auto iterators = comp->get_iteration_variables();
        int size = iterators.size();
        if(size>=6){
          comp->apply_opt_strategy({4,1,1});
        }
        if(size==4){
          if(count!=0){
              comp->apply_opt_strategy({2,1,1});
              count+=1;
          }
          count+=1;
          
        }
    }
    std::string pwd = std::filesystem::current_path().parent_path();
    std::string path = pwd+"/samples/resnet18/";
    fct->dump_schedule(path);
}

