#include "generator.h"
#include <string>
#include <iostream>
#include <filesystem>
#include "scalehls/Transforms/Passes.h"
#include "scalehls/Transforms/Utils.h"
#include "scalehls/Transforms/Estimator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MemoryBuffer.h"

namespace polyfp{

polyfp::compute *get_computation_annotated_in_a_node(isl_ast_node *node)
{
    // Retrieve the computes of the node.
    isl_id *comp_id = isl_ast_node_get_annotation(node);
    polyfp::compute *comp = (polyfp::compute *)isl_id_get_user(comp_id);
    isl_id_free(comp_id);
    return comp;
    
}

std::map<std::string,mlir::Value> polyfp::MLIRGenImpl::get_argument_map(){
    return this->argument_map;
}

std::map<std::string,int> polyfp::MLIRGenImpl::get_array_map(){
    return this->array_map;
}


std::vector<mlir::FuncOp>  polyfp::MLIRGenImpl::get_funcs(){
    return this->funcs;
}

//TODO: find out what is the acutal "loc"
int polyfp::MLIRGenImpl::get_iterator_location_from_name(polyfp::compute *comp, polyfp::expr polyfp_expr, std::vector<mlir::Value> &index_values)
{
    auto name_set = comp->get_loop_level_names();
    int loc;
    if (std::find(name_set.begin(), name_set.end(), polyfp_expr.get_name()) == name_set.end() )
    {
        for (auto &kv2: comp->get_access_map())
        {
            if(polyfp_expr.get_name()==kv2.first)
            {
                // loc = comp->get_loop_level_number_from_dimension_name(kv2.second); 
                loc = comp->iterators_location_map[kv2.second];      
            }
        }

        mlir::Value value = ops[loc].getInductionVar();

        if ( std::find(index_values.begin(), index_values.end(), value) == index_values.end())
        {
            index_values.push_back(value);
        }

        loc = std::find(index_values.begin(), index_values.end(), value) - index_values.begin();
    }
    else{
        loc = comp->iterators_location_map[polyfp_expr.get_name()];      
        mlir::Value value = ops[loc].getInductionVar();

        if ( std::find(index_values.begin(), index_values.end(), value) == index_values.end())
        {
            index_values.push_back(value);
        }

        loc = std::find(index_values.begin(), index_values.end(), value) - index_values.begin();
    }
    return loc;
}

mlir::ModuleOp polyfp::MLIRGenImpl::mlirGen1(const polyfp::function &fct, isl_ast_node *isl_node, int &level, bool flag, bool flag2, bool if_flag) 
{
    std::vector<std::pair<std::string, std::string>> generated_stmts;
    isl_ast_node *node=isl_node;
    if (isl_ast_node_get_type(node) == isl_ast_node_for)
    {
        isl_ast_expr *iter = isl_ast_node_for_get_iterator(node);
        isl_id *identifier = isl_ast_expr_get_id(iter);
        std::string name_str(isl_id_get_name(identifier));
        name_map.insert(std::pair(level,name_str));
        isl_ast_expr *init = isl_ast_node_for_get_init(node);
        // std::cout<<isl_ast_expr_to_str(init)<<std::endl;
        isl_ast_expr *cond = isl_ast_node_for_get_cond(node);
        isl_ast_expr *inc = isl_ast_node_for_get_inc(node);
        isl_ast_node *body = isl_ast_node_for_get_body(node);
        int lb_int;
        int ub_int;
        bool vbound_flag = false;
        std::vector<mlir::Value> lb_values;
        std::vector<mlir::AffineExpr> lb_args;
        std::vector<mlir::Value> ub_values;
        std::vector<mlir::AffineExpr> ub_args;

        isl_ast_expr *cond_upper = isl_ast_expr_get_op_arg(cond, 1);

        if (isl_ast_expr_get_type(init) == isl_ast_expr_int)
        {
            lb_int = isl_val_get_num_si(isl_ast_expr_get_val(init));
        }
        else if (isl_ast_expr_get_type(init) == isl_ast_expr_op)
        {
            // std::cout<<"isl_ast_expr_op "<<std::endl;
            vbound_flag = true;
            int nb = isl_ast_expr_get_op_n_arg(init);
            for(int i=0;i<nb;i++)
            {   
                isl_ast_expr *expr_itr = isl_ast_expr_get_op_arg(init, i);
                if (isl_ast_expr_get_type(expr_itr) == isl_ast_expr_int)
                {
                    lb_args.push_back(getAffineConstantExpr(isl_val_get_num_si(isl_ast_expr_get_val(expr_itr)), builder.getContext()));
                }
                else if (isl_ast_expr_get_op_type(expr_itr) == isl_ast_op_sub)
                {
                    // std::cout<<"isl_ast_op_sub "<<std::endl;
                    int nb = isl_ast_expr_get_op_n_arg(init);
                    
                    isl_ast_expr *expr0 = isl_ast_expr_get_op_arg(expr_itr, 0);
                    isl_ast_expr *expr1 = isl_ast_expr_get_op_arg(expr_itr, 1);
                    int sub1;
                    int div1 =1;
                    // isl_ast_expr_to_C_str(expr0);
                    
                    if (isl_ast_expr_get_type(expr0) == isl_ast_expr_id)
                    {
                        isl_id *identifier = isl_ast_expr_get_id(expr0);
                        std::string name_str(isl_id_get_name(identifier));
                        int loc;
                        int index = 0;
                        for(int i=0; i<start_loops_position.size(); i++)
                        {
                            if(start_loops_position[i]>level)
                            {
                                index = start_loops_position[i-1];
                                break;
                            }
                            if(i == start_loops_position.size()-1 )
                            {
                                index = start_loops_position[i];
                                break;
                            }
                        }
                        for (auto &kv4: name_map)
                        {
                            if(name_str==kv4.second)
                            {
                                loc = kv4.first;
                            }
                        }

                        mlir::Value value = ops[loc+index].getInductionVar();
                        lb_values.push_back(value);
                        isl_id_free(identifier);
                    }
                    else
                    {
                        int nb = isl_ast_expr_get_op_n_arg(expr0);
                        
                        isl_ast_expr *expr0_0 = isl_ast_expr_get_op_arg(expr0, 0);
                        isl_ast_expr *expr1_0 = isl_ast_expr_get_op_arg(expr0, 1);
                        if (isl_ast_expr_get_type(expr0_0) == isl_ast_expr_id)
                        {
                            isl_id *identifier = isl_ast_expr_get_id(expr0_0);
                            std::string name_str(isl_id_get_name(identifier));
                            int loc;
                            int index = 0;
                            for(int i=0; i<start_loops_position.size(); i++)
                            {
                                if(start_loops_position[i]>level)
                                {
                                    index = start_loops_position[i-1];
                                    break;
                                }
                                if(i == start_loops_position.size()-1 )
                                {
                                    index = start_loops_position[i];
                                    break;
                                }
                            }
                            for (auto &kv4: name_map)
                            {
                                if(name_str==kv4.second)
                                {
                                    loc = kv4.first;
                                }
                            }
                            mlir::Value value = ops[loc+index].getInductionVar();
                            lb_values.push_back(value);
                            isl_id_free(identifier);
                        }
                        if (isl_ast_expr_get_type(expr1) == isl_ast_expr_int)
                        {   
                            div1 = isl_val_get_num_si(isl_ast_expr_get_val(expr1_0));
                        }

                    }
                    if (isl_ast_expr_get_type(expr1) == isl_ast_expr_int)
                    {   
                        sub1 = isl_val_get_num_si(isl_ast_expr_get_val(expr1));
                    }
                    else{
                        // TODO
                    } 
                    //TODO: find right dimensions
                    lb_args.push_back(builder.getAffineDimExpr(0).floorDiv(div1) - sub1);
                    
                }
                else if(isl_ast_expr_get_op_type(expr_itr) == isl_ast_expr_op)
                {
                    // std::cout<<"a division"<<std::endl;
                    // TODO
                }
               
                else{
                    polyfp::str_dump("Transforming the following expression",
                           (const char *)isl_ast_expr_to_C_str(expr_itr));
                }     
                isl_ast_expr_free(expr_itr);
            }
        }
        auto lb_map = mlir::AffineMap::get(1, 0, ArrayRef<mlir::AffineExpr> (lb_args),builder.getContext());
        mlir::ValueRange lb_vr=llvm::makeArrayRef(lb_values);  
        if (isl_ast_expr_get_type(cond_upper) == isl_ast_expr_int)
        {
            ub_int = isl_val_get_num_si(isl_ast_expr_get_val(cond_upper))+1;
        }
        else if (isl_ast_expr_get_type(cond_upper) == isl_ast_expr_op)
        {
            // std::cout<<"upper bound"<<std::endl;
            vbound_flag = true;
            int nb = isl_ast_expr_get_op_n_arg(cond_upper);
            int sub1;
            int div1=1;
            for(int i=0;i<nb;i++)
            {   
                isl_ast_expr *expr_itr = isl_ast_expr_get_op_arg(cond_upper, i);
                if (isl_ast_expr_get_type(expr_itr) == isl_ast_expr_int)
                {
                    ub_args.push_back(getAffineConstantExpr(isl_val_get_num_si(isl_ast_expr_get_val(expr_itr))+1, builder.getContext()));
                }
                else if (isl_ast_expr_get_op_type(expr_itr) == isl_ast_op_sub)
                {
                    isl_ast_expr *expr0 = isl_ast_expr_get_op_arg(expr_itr, 0);
                    isl_ast_expr *expr1 = isl_ast_expr_get_op_arg(expr_itr, 1);
                    int add1;
                    if (isl_ast_expr_get_type(expr0) == isl_ast_expr_id)
                    {
                        isl_id *identifier = isl_ast_expr_get_id(expr0);
                        std::string name_str(isl_id_get_name(identifier));
                        int loc;
                        for (auto &kv4: name_map)
                        {
                            if(name_str==kv4.second)
                            {
                                loc = kv4.first;
                            }
                        }
                        int index = 0;
                        for(int i=0; i<start_loops_position.size(); i++)
                        {
                            if(start_loops_position[i]>level)
                            {
                                index = start_loops_position[i-1];
                                break;
                            }
                            if(i == start_loops_position.size()-1 )
                            {
                                index = start_loops_position[i];
                                break;
                            }
                        }
                        mlir::Value value = ops[loc+index].getInductionVar();
                        ub_values.push_back(value);
                        isl_id_free(identifier);
                    }  
                    if (isl_ast_expr_get_type(expr1) == isl_ast_expr_int)
                    {   
                        add1 = isl_val_get_num_si(isl_ast_expr_get_val(expr1))+1;
                    }  
                    //TODO: find right dimensions
                    ub_args.push_back(builder.getAffineDimExpr(0) - add1);
                    
                }
                else if (isl_ast_expr_get_type(expr_itr) == isl_ast_expr_id)
                {
                    isl_id *identifier = isl_ast_expr_get_id(expr_itr);
                    std::string name_str(isl_id_get_name(identifier));
                    int loc;
                    for (auto &kv4: name_map)
                    {
                        if(name_str==kv4.second)
                        {
                            loc = kv4.first;
                        }
                    }
                    int index = 0;
                        for(int i=0; i<start_loops_position.size(); i++)
                        {
                            if(start_loops_position[i]>level)
                            {
                                index = start_loops_position[i-1];
                                break;
                            }
                            if(i == start_loops_position.size()-1 )
                            {
                                index = start_loops_position[i];
                                break;
                            }
                        }
                    mlir::Value value = ops[loc+index].getInductionVar();
                    ub_values.push_back(value);
                    isl_id_free(identifier);
                    ub_args.push_back(builder.getAffineDimExpr(0));
                }
                else if(isl_ast_expr_get_type(expr_itr) == isl_ast_expr_op)
                {
                    int nb = isl_ast_expr_get_op_n_arg(expr_itr);
                    isl_ast_expr *expr0_0 = isl_ast_expr_get_op_arg(expr_itr, 0);
                    isl_ast_expr *expr1_0 = isl_ast_expr_get_op_arg(expr_itr, 1);
                    if (isl_ast_expr_get_type(expr0_0) == isl_ast_expr_id)
                    {
                        isl_id *identifier = isl_ast_expr_get_id(expr0_0);
                        std::string name_str(isl_id_get_name(identifier));
                        int loc;
                        int index = 0;
                        for(int i=0; i<start_loops_position.size(); i++)
                        {
                            if(start_loops_position[i]>level)
                            {
                                index = start_loops_position[i-1];
                                break;
                            }
                            if(i == start_loops_position.size()-1 )
                            {
                                index = start_loops_position[i];
                                break;
                            }
                        }
                        for (auto &kv4: name_map)
                        {
                            if(name_str==kv4.second)
                            {
                                loc = kv4.first;
                            }
                        }
                        mlir::Value value = ops[loc+index].getInductionVar();
                        ub_values.push_back(value);
                        isl_id_free(identifier);

                    }
                    if (isl_ast_expr_get_type(expr1_0) == isl_ast_expr_int)
                    {   
                        div1 = isl_val_get_num_si(isl_ast_expr_get_val(expr1_0));
                    }
                    ub_args.push_back(builder.getAffineDimExpr(0).floorDiv(div1));

                }
                else
                {
                    polyfp::str_dump("Transforming the following expression",
                           (const char *)isl_ast_expr_to_C_str(expr_itr));
                }     
                isl_ast_expr_free(expr_itr);
                
            }
        }
        auto ub_map = mlir::AffineMap::get(1, 0, ArrayRef<mlir::AffineExpr> (ub_args),builder.getContext());
        mlir::ValueRange ub_vr=llvm::makeArrayRef(ub_values);
        int step = isl_val_get_num_si(isl_ast_expr_get_val(inc));
        if (isl_ast_node_get_type(node) == isl_ast_node_for)
        {
            if(level== 0)
            {
                start_loops_position.push_back(level);
                std::vector<mlir::Type> types;
                std::string name = polyfp::global::get_implicit_function()->get_name();
                mlir::Location loc = builder.getUnknownLoc();
                auto varLoc = loc;
                llvm::SmallVector<mlir::Type, 4> operandTypes;
                llvm::SmallVector<mlir::Type, 4> operandTypes_temp;
                llvm::SmallVector<mlir::Type, 4> operandTypes_arg;
                mlir::Type t;
                mlir::MemRefType mr;
                for (auto &kv : fct.get_invariants()) 
                {
                    //TODO: more datatype
                    if(kv.second->get_type() == polyfp::p_float64)
                    {
                        t= builder.getF64Type();
                    }
                    if(kv.second->get_type() == polyfp::p_float32)
                    {
                        t= builder.getF32Type();  
                    }
                    if(kv.second->get_type() == polyfp::p_int32)
                    {
                        t= builder.getIntegerType(32); 
                    }
                    operandTypes.push_back(t);
                    argument_list.push_back(kv.first);
                    operandTypes_temp.push_back(mr);
                }
                std::vector<std::string> p_names;

                for (auto &kv : fct.get_fct_arguments()) 
                {
                    unsigned memspace = 0;
                    auto size = kv.second->get_dim_sizes();
                    auto arg_type = kv.second->get_elements_type();
                    p_names.push_back(kv.first);
                    if (arg_type == polyfp::p_uint8 || arg_type == polyfp::p_int8)
                    {
                        t= builder.getIntegerType(8);
                    }else if(arg_type == polyfp::p_uint16 || arg_type == polyfp::p_int16)
                    {
                        t= builder.getIntegerType(16);
                    }else if(arg_type == polyfp::p_uint32 || arg_type == polyfp::p_int32)
                    {
                        t= builder.getIntegerType(32);
                    }else if(arg_type == polyfp::p_uint64 || arg_type == polyfp::p_int64)
                    {
                        t= builder.getIntegerType(64);
                    }else if(arg_type == polyfp::p_float32)
                    {
                        t= builder.getF32Type();
                    }else if(arg_type == polyfp::p_float64)
                    {
                        t= builder.getF64Type();
                    }
                    mr = mlir::MemRefType::get(llvm::makeArrayRef(size), t, {}, memspace);
                    operandTypes.push_back(mr);

                    argument_list.push_back(kv.first);
 
                    array_map.insert(std::make_pair(kv.first,operandTypes.size()-1));
                    
                }

                for (auto &kv : fct.get_placeholders())
                {
                     unsigned memspace = 0;
                     auto size = kv.second->get_dim_sizes();
                     auto arg_type = kv.second->get_elements_type();
                    //  p_names.push_back(kv.first);
                     if (arg_type == polyfp::p_uint8 || arg_type == polyfp::p_int8)
                     {
                         t= builder.getIntegerType(8);
                     }else if(arg_type == polyfp::p_uint16 || arg_type == polyfp::p_int16)
                     {
                         t= builder.getIntegerType(16);
                     }else if(arg_type == polyfp::p_uint32 || arg_type == polyfp::p_int32)
                     {
                         t= builder.getIntegerType(32);
                     }else if(arg_type == polyfp::p_uint64 || arg_type == polyfp::p_int64)
                     {
                         t= builder.getIntegerType(64);
                     }else if(arg_type == polyfp::p_float32)
                     {
                         t= builder.getF32Type();
                     }else if(arg_type == polyfp::p_float64)
                     {
                         t= builder.getF64Type();
                     }
                     mr = mlir::MemRefType::get(llvm::makeArrayRef(size), t, {}, memspace);
                    //  auto op = builder.create<mlir::memref::AllocaOp>(loc, mr);
                    //  op.dump();
                     operandTypes_temp.push_back(mr);
                    //  argument_list.push_back(kv.first);
                    //  array_map.insert(std::make_pair(kv.first,operandTypes_temp.size()-1));
                }

                for (auto &kv : fct.get_global_arguments()) 
                {
                     unsigned memspace = 0;
                     auto size = kv.second->get_dim_sizes();
                     auto arg_type = kv.second->get_elements_type();
                     p_names.push_back(kv.first);
                     if (arg_type == polyfp::p_uint8 || arg_type == polyfp::p_int8)
                     {
                         t= builder.getIntegerType(8);
                     }else if(arg_type == polyfp::p_uint16 || arg_type == polyfp::p_int16)
                     {
                         t= builder.getIntegerType(16);
                     }else if(arg_type == polyfp::p_uint32 || arg_type == polyfp::p_int32)
                     {
                         t= builder.getIntegerType(32);
                     }else if(arg_type == polyfp::p_uint64 || arg_type == polyfp::p_int64)
                     {
                         t= builder.getIntegerType(64);
                     }else if(arg_type == polyfp::p_float32)
                     {
                         t= builder.getF32Type();
                     }else if(arg_type == polyfp::p_float64)
                     {
                         t= builder.getF64Type();
                     }
                     mr = mlir::MemRefType::get(llvm::makeArrayRef(size), t, {}, memspace);
                    //  auto op = builder.create<mlir::memref::AllocaOp>(loc, mr);
                    //  op.dump();
                     operandTypes_arg.push_back(mr);
                     
                     argument_list.push_back(kv.first);
                     array_map.insert(std::make_pair(kv.first,operandTypes_temp.size()-1));
                }
      
                if(flag ==true)
                {

                    mlir::FuncOp myFunc = mlir::FuncOp::create(loc, /*name=*/name, /*type=*/builder.getFunctionType(operandTypes, {}), /*attrs=*/{}); 
                    auto &entryBlock = *myFunc.addEntryBlock();
                    builder.setInsertionPointToStart(&entryBlock);
                    funcs.push_back(myFunc);
                    theModule.push_back(myFunc);

                    int index_pname = 0;
                    int index_pname2 = 0;
                  
                    for(auto &arg: operandTypes_arg)
                    {
                        mlir::MemRefType mr = arg.dyn_cast<mlir::MemRefType>();
                        auto value = builder.create<mlir::memref::AllocaOp>(loc, mr);
                        values.push_back(value);

                    }
                    for(auto &p_name: p_names)
                    {
                        if(index_pname<=fct.get_fct_arguments().size())
                        {
                            auto mem = myFunc.getArgument(index_pname);
                            index_pname+=1;
                            argument_map.insert(std::pair(p_name,mem));
                        }
                        else
                        {
                            auto mem = values[index_pname2];
                            index_pname2+=1;
                            argument_map.insert(std::pair(p_name,mem));
                        }
                    }
                    // theModule.dump();
                    // mlir::MemRefType mr = operandTypes_temp[0].dyn_cast<mlir::MemRefType>();
                    // auto op = builder.create<mlir::memref::AllocaOp>(loc, mr);
                    // for(auto &)
                    //example of builder.getI16IntegerAttr(5)
                    // mlir::Value arg1 = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(),t, builder.getF32ArrayAttr(3.0));
                    // mlir::Value arg2 = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(),t, builder.getF32ArrayAttr(6.7));
                    if(vbound_flag == false)
                    {
                        auto loop = builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), lb_int, ub_int, step);
                        ops.push_back(loop);
                        builder.setInsertionPointAfter(ops[0]);
                        auto return_op = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), ArrayRef<mlir::Value>());
                        builder.setInsertionPointToStart(loop.getBody());
                    }else
                    {
                        auto loop = builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), lb_vr, lb_map, ub_vr, ub_map, step);
                        ops.push_back(loop);
                        builder.setInsertionPointAfter(ops[0]);
                        auto return_op = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), ArrayRef<mlir::Value>());
                        builder.setInsertionPointToStart(loop.getBody());
                    }
                }else{
                    if(vbound_flag == false)
                    {
                        auto loop = builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), lb_int, ub_int, step);
                        ops.push_back(loop);
                        builder.setInsertionPointAfter(ops[0]);
                        builder.setInsertionPointToStart(loop.getBody());
                    }
                    else
                    {
                        auto loop = builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), lb_vr, lb_map, ub_vr, ub_map, step);
                        ops.push_back(loop);
                        builder.setInsertionPointAfter(ops[0]);
                        builder.setInsertionPointToStart(loop.getBody());
                    }
                }  
            }
            else 
            {
                if(vbound_flag == false)
                {
                    auto loop = builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), lb_int, ub_int, step);
                    ops.push_back(loop);
                    builder.setInsertionPointAfter(ops[0]);
                    builder.setInsertionPointToStart(loop.getBody());
                }
                else
                {
                    auto loop = builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), lb_vr, lb_map, ub_vr, ub_map, step);
                    ops.push_back(loop);
                    builder.setInsertionPointAfter(ops[0]);
                    builder.setInsertionPointToStart(loop.getBody());
                }
            }
        }
        isl_ast_expr_free(init);
        isl_ast_expr_free(cond);
        isl_ast_expr_free(inc);
        isl_ast_node_free(body);
        isl_ast_expr_free(cond_upper);

        if (isl_ast_node_get_type(body) == isl_ast_node_for)
        {   
            level = level+1;
            mlirGen1(fct,body,level,false,flag2,if_flag);
        }

        if (isl_ast_node_get_type(body) == isl_ast_node_user)
        {
            mlirGen1(fct,body,level,false,flag2,if_flag);
        }

        else if(isl_ast_node_get_type(body) == isl_ast_node_block)
        {
            mlirGen1(fct,body,level,false,flag2,if_flag);
        }

        else if(isl_ast_node_get_type(body) == isl_ast_node_if)
        {
            // TODO
        }
        else{
            // TODO
        }
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_block)
    {
        // std::cout<<"enter a block node"<<std::endl;
        isl_ast_node_list *list = isl_ast_node_block_get_children(node);
        int current_level = level;
        int children_number = isl_ast_node_list_n_ast_node(list);
        // theModule.dump();
        // std::cout<<"number of children: ";
        // std::cout<<children_number<<std::endl;
        //TODO: current start_loops_position maybe not safe enough, find another way to build it

        if(this->ops.size() == 0)
        {
            for (int i = 0; i <=isl_ast_node_list_n_ast_node(list) - 1; i++)
            {   
                isl_ast_node *child = isl_ast_node_list_get_ast_node(list, i);
                if (isl_ast_node_get_type(child) == isl_ast_node_user)
                {
                    mlirGen1(fct,child,current_level,false,true,if_flag);
                }
                else if (isl_ast_node_get_type(child) == isl_ast_node_for)
                {
                    if(this->ops.size() != 0)
                    {
                        level = level + 1 ;
                        int current_level = level;            
                        start_loops_position.push_back(level);
                        mlirGen1(fct,child,level,false,true,if_flag);
                        builder.setInsertionPointAfter(ops[current_level]);
                    }
                    else{
                        // start_loops_position.push_back(level);
                        mlirGen1(fct,child,level,true, false, false);   
                        builder.setInsertionPointAfter(ops[current_level]);     
                    }
                }
                else if (isl_ast_node_get_type(child) == isl_ast_node_block)
                {
                    mlirGen1(fct,child,level,false,true,if_flag);
                }else if (isl_ast_node_get_type(child) == isl_ast_node_if)
                {
                    mlirGen1(fct,child,level,false,true,if_flag);
                }else{
                    // TODO
                }         
            }
        }
        else
        {
            for (int i = 0; i <=isl_ast_node_list_n_ast_node(list) - 1; i++)
            {   
                isl_ast_node *child = isl_ast_node_list_get_ast_node(list, i);

                if (isl_ast_node_get_type(child) == isl_ast_node_user)
                {
                    mlirGen1(fct,child,current_level,false,true,if_flag);
                }
                else if (isl_ast_node_get_type(child) == isl_ast_node_for)
                {
                    if(this->ops.size() != 0)
                    {
                        level = level + 1 ;
                        int current_level = level;            
                        // start_loops_position.push_back(level);
                        mlirGen1(fct,child,level,false,true,if_flag);
                        builder.setInsertionPointAfter(ops[current_level]);
                    }
                    else
                    {
                        // start_loops_position.push_back(level);
                        mlirGen1(fct,child,level,true, false, false);   
                        builder.setInsertionPointAfter(ops[current_level]);     
                    }
                }
                else if (isl_ast_node_get_type(child) == isl_ast_node_block)
                {
                    mlirGen1(fct,child,level,false,true,if_flag);
                }
                else if (isl_ast_node_get_type(child) == isl_ast_node_if)
                {
                    mlirGen1(fct,child,level,false,true,if_flag);
                }
                else{
                     // TODO
                }         
            }
        }
        
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_user)
    {
        bool flag = true;
        isl_ast_expr *expr = isl_ast_node_user_get_expr(node);
        isl_ast_expr *arg = isl_ast_expr_get_op_arg(expr, 0);
        isl_id *id = isl_ast_expr_get_id(arg);
        std::string computation_name(isl_id_get_name(id));
        isl_id_free(id);
        polyfp::compute *comp;
        int dim_number = 0;
        for (const auto &cpt : fct.get_body())
        {
            if(cpt->get_name()==computation_name)
            {
                comp = cpt;
            }
            if(dim_number < cpt->get_loop_levels_number())
                dim_number = cpt->get_loop_levels_number();
        }
        auto polyfp_expr = comp->get_expr();

        std::string p_name = comp->get_placeholder()->get_name();

        int index_placeholder;
        int index_argument;

        for(int i=0; i<argument_list.size(); i++)
        {
            if(argument_list.at(i) == p_name)
            {
                index_placeholder = i;

            }
        }
        std::vector<mlir::Value> placeholder_index_values;
        SmallVector<mlir::AffineExpr> placeholder_index_args;
        bool placeholder_index_flag = false;
        int count = 0;

        for (auto &kv: comp->get_placeholder_dims())
        {
            int bias = 0;
            if(kv.get_expr_type() == polyfp::e_op)
            {
                //TODO, HANDLE loop skewing
                auto expr0 = kv.get_operand(0);
                auto expr1 = kv.get_operand(1);
                auto left_index = a_print_index(expr0,comp,placeholder_index_values,level);
                auto right_index = a_print_index(expr1,comp,placeholder_index_values,level);

                if(kv.get_op_type() == polyfp::o_sub)
                {
                    placeholder_index_args.push_back(left_index - right_index);
                    placeholder_index_flag = true;
                }

                else if(kv.get_op_type() == polyfp::o_add)
                {
                    placeholder_index_args.push_back(left_index + right_index);
                    placeholder_index_flag = true;
                }
                
                else if(kv.get_op_type() == polyfp::o_mul)
                {
                    placeholder_index_args.push_back(left_index * right_index);
                    placeholder_index_flag = true;    
                }

                else if(kv.get_op_type() == polyfp::o_div)
                {
                    placeholder_index_args.push_back(left_index.floorDiv(right_index));
                    placeholder_index_flag = true;  
                }
            }
            else{
                int loc =0;
                int loc_2 =0;
                std::string tile_name1;
                std::string tile_name2;
                int tile_size;
                auto name_set = comp->get_loop_level_names(); 
                // std::cout<<kv.get_name()<<std::endl;      
                if(std::find(name_set.begin(), name_set.end(), kv.get_name()) == name_set.end())
                {           
                    for (auto &kv2: comp->get_access_map())
                    {
                        if(kv.get_name()==kv2.first)
                        {
                            tile_name1 = kv2.second;
                            loc = comp->iterators_location_map[tile_name1];
                           
                        }
                    }
                    mlir::Value value = ops[loc].getInductionVar();
                    if ( std::find(placeholder_index_values.begin(), placeholder_index_values.end(), value)== placeholder_index_values.end() )
                    {    
                        placeholder_index_values.push_back(value);
                    }
                    
                    if(comp->is_tiled == true)
                    {
                        for (auto &kv3: comp->get_tile_map())
                        {
                            if(tile_name1==kv3.first)
                            {
                                // loc_2 = comp->get_loop_level_number_from_dimension_name(kv3.second);
                                loc = comp->iterators_location_map[kv3.second];
                                tile_name2 = kv3.second;
                            }
                        }
                        for (auto &kv4: comp->get_tile_size_map())
                        {
                            if(tile_name1==kv4.first)
                            {
                                tile_size = kv4.second;
                            }
                        }
                        mlir::Value value2 = ops[loc].getInductionVar();
                        if ( std::find(placeholder_index_values.begin(), placeholder_index_values.end(), value2)== placeholder_index_values.end() )
                        {
                            placeholder_index_values.push_back(value2);
                        }
                        //TODO: find the right dim
                        int index_2 = std::find(placeholder_index_values.begin(), placeholder_index_values.end(), value2) - placeholder_index_values.begin();
                        int index_3 = std::find(placeholder_index_values.begin(), placeholder_index_values.end(), value) - placeholder_index_values.begin();
                        placeholder_index_args.push_back(builder.getAffineDimExpr(index_3)+builder.getAffineDimExpr(index_2)*tile_size);
                        placeholder_index_flag = true;
                    }
                    else if(comp->is_skewed == true)
                    {
                        loc = comp->iterators_location_map[comp->iterator_to_skew];
                        mlir::Value value2 = ops[loc].getInductionVar();
                        if ( std::find(placeholder_index_values.begin(), placeholder_index_values.end(), value2)== placeholder_index_values.end() )
                        {
                            placeholder_index_values.push_back(value2);
                        }
                        //TODO: find the right dim
                        int index_2 = std::find(placeholder_index_values.begin(), placeholder_index_values.end(), value2) - placeholder_index_values.begin();
                        int index_3 = std::find(placeholder_index_values.begin(), placeholder_index_values.end(), value) - placeholder_index_values.begin();
                        // placeholder_index_args.push_back(builder.getAffineDimExpr(index_3)-builder.getAffineDimExpr(index_2)*2);
                        // placeholder_index_args.push_back(builder.getAffineDimExpr(index_3));
                        if(tile_name1 == comp->iterator_to_modify)
                        {
                            placeholder_index_args.push_back(builder.getAffineDimExpr(index_3)-builder.getAffineDimExpr(index_2)*comp->skew_factor);
                        }else
                        {
                            placeholder_index_args.push_back(builder.getAffineDimExpr(index_3));
                        }
                        
                        placeholder_index_flag = true;
                    }
                }
                else{
                    loc = comp->iterators_location_map[kv.get_name()];
                    mlir::Value value = ops[loc].getInductionVar();
                    if (std::find(placeholder_index_values.begin(), placeholder_index_values.end(), value) == placeholder_index_values.end())
                    {
                        placeholder_index_values.push_back(value);
                    }
                    //TODO:
                    int index2 = std::find(placeholder_index_values.begin(), placeholder_index_values.end(), value) - placeholder_index_values.begin();
                    placeholder_index_args.push_back(builder.getAffineDimExpr(index2));
                }
            }
        }
        
        auto placeholder_map = mlir::AffineMap::get(placeholder_index_values.size(), 0, ArrayRef<mlir::AffineExpr> (placeholder_index_args),builder.getContext());
        mlir::ValueRange placeholder_vr=llvm::makeArrayRef(placeholder_index_values);
        mlir::Value mem;
        if(index_placeholder+1<=funcs[0].getNumArguments())
        {
            mem = funcs[0].getArgument(index_placeholder);
            argument_map.insert(std::pair(p_name,mem));
        }
        else{
            mem = values[index_placeholder-funcs[0].getNumArguments()];
            argument_map.insert(std::pair(p_name,mem));
        }

        if (polyfp_expr.get_expr_type() == polyfp::e_var)
        {    

            int index_argument;
            std::string arg_name = polyfp_expr.get_name();
            for(int i=0; i<argument_list.size(); i++)
            {
                if(argument_list.at(i) == arg_name)
                {
                index_argument = i;
                }
            }
            mlir::Value arg_1;
            if(index_argument+1<=funcs[0].getNumArguments())
            {
                arg_1 = funcs[0].getArgument(index_argument);
            }else
            {
                arg_1 = values[index_argument-funcs[0].getNumArguments()];
            }

            // TODO, test all if cases
            if(if_flag == true)
            {
                mlir::Value value = ops[2].getInductionVar();
                SmallVector<mlir::Value, 4> ifOperands;
                ifOperands.push_back(value);
                SmallVector<mlir::AffineExpr, 4> ifExprs;
                ifExprs.push_back(builder.getAffineDimExpr(0));
                SmallVector<bool, 4> ifEqFlags;
                ifEqFlags.push_back(true);
                const auto condition = mlir::IntegerSet::get(1, 0, ifExprs, ifEqFlags);
                auto ifOp =builder.create<mlir::AffineIfOp>(builder.getUnknownLoc(), condition, ifOperands,/*withElseRegion=*/false);
                builder.setInsertionPointToStart(ifOp.getBody());
                auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), arg_1, mem, placeholder_vr);
                builder.setInsertionPointAfter(ifOp);
            }
            else
            {
                if(placeholder_index_flag == true)
                {
                    auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), arg_1, mem, placeholder_map, placeholder_vr);
                    builder.setInsertionPointAfter(store1);
                }else
                {
                    auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), arg_1, mem, placeholder_vr);
                    builder.setInsertionPointAfter(store1);
                }
            }
        }
        else if (polyfp_expr.get_expr_type() == polyfp::e_op && polyfp_expr.get_op_type() != polyfp::o_access && polyfp_expr.get_op_type() != polyfp::o_max )
        {            
            mlir::ValueRange indices = {};
            auto a = polyfp_expr.get_operand(0);
            auto b = polyfp_expr.get_operand(1);
            mlir::BlockArgument left;
            mlir::BlockArgument right;
            mlir::arith::MulFOp allocSize_m;
            mlir::arith::AddFOp allocSize_a;
            // theModule.dump();
            a_print_expr(polyfp_expr, comp, level);

            if(if_flag == true)
            {
                mlir::Value value = ops[2].getInductionVar();
                SmallVector<mlir::Value, 4> ifOperands;
                ifOperands.push_back(value);
                SmallVector<mlir::AffineExpr, 4> ifExprs;
                ifExprs.push_back(builder.getAffineDimExpr(0));
                SmallVector<bool, 4> ifEqFlags;
                ifEqFlags.push_back(true);
                const auto condition = mlir::IntegerSet::get(1, 0, ifExprs, ifEqFlags);
                auto ifOp =builder.create<mlir::AffineIfOp>(builder.getUnknownLoc(), condition, ifOperands,/*withElseRegion=*/false);
                builder.setInsertionPointToStart(ifOp.getBody());
                //TODO: other arithmetic? sub, o_div
                auto the_op = all_current_op.back();
                
                auto index = the_op.index();
                if(index==0)
                {
                    auto op_to_process = std::get<0>(the_op);
                    auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_vr);
                }
                else if(index==1)
                {
                    auto op_to_process = std::get<1>(the_op);
                    auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_vr);
                }else if(index==2)
                {
                    auto op_to_process = std::get<2>(the_op);
                    auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_vr);
                }else if(index==3)
                {
                    auto op_to_process = std::get<3>(the_op);
                    auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_vr);
                }
                builder.setInsertionPointAfter(ifOp);
            }else{
                auto the_op = all_current_op.back();
                auto index = the_op.index();
                if(index==0)
                {
                    auto op_to_process = std::get<0>(the_op);
                    if(placeholder_index_flag == true)
                    {
                        auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_map, placeholder_vr);
                        builder.setInsertionPointAfter(store1);
                    }else{
                        auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_vr);
                        builder.setInsertionPointAfter(store1);
                    } 
                    
                }else if(index==1)
                {
                    auto op_to_process = std::get<1>(the_op);
                    if(placeholder_index_flag == true)
                    {
                        auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_map, placeholder_vr);
                        builder.setInsertionPointAfter(store1);
                    }else
                    {
                        auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_vr);
                        builder.setInsertionPointAfter(store1);
                    }
                }else if(index==2)
                {
                    auto op_to_process = std::get<2>(the_op);
                    if(placeholder_index_flag == true)
                    {
                        auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_map, placeholder_vr);
                        builder.setInsertionPointAfter(store1);
                    }
                    else
                    {
                        auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_vr);
                        builder.setInsertionPointAfter(store1);
                    }
                }else if(index==3)
                {
                    auto op_to_process = std::get<3>(the_op);
                    if(placeholder_index_flag == true)
                    {
                        auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_map, placeholder_vr);
                        builder.setInsertionPointAfter(store1);
                    }
                    else
                    {
                        auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_vr);
                        builder.setInsertionPointAfter(store1);
                    }
                }
            }
        }
        else if(polyfp_expr.get_expr_type() == polyfp::e_op && polyfp_expr.get_op_type() == polyfp::o_access )
        {
            std::string a_name = polyfp_expr.get_name();
            int index_a;
            // if(comp->refused == true){
            //     a_name = comp->temp_access_map[a_name];
            // }
            for(int i=0; i<argument_list.size(); i++)
            {
                if(argument_list.at(i) == a_name)
                    index_a = i;               
            }
            // auto arg_a = funcs[0].getArgument(index_a);
            mlir::Value arg_a;

            if(index_a+1<=funcs[0].getNumArguments())
            {
                arg_a = funcs[0].getArgument(index_a);
            }else
            {
                arg_a = values[index_a-funcs[0].getNumArguments()];
            }
            
            std::vector<mlir::Value> index_values;
            SmallVector<mlir::AffineExpr> index_args;
            bool index_flag = false;
            int count = 0;
            for (auto &kv: polyfp_expr.get_access())
            {
                int bias = 0;
                if(kv.get_expr_type() == polyfp::e_op)
                {
                    auto expr0 = kv.get_operand(0);
                    auto expr1 = kv.get_operand(1);
                    auto left_index = a_print_index(expr0,comp,index_values,level);
                    auto right_index = a_print_index(expr1,comp,index_values,level);
                    if(kv.get_op_type() == polyfp::o_sub)
                    {
                        index_args.push_back(left_index - right_index);
                        index_flag = true;
                    }else if(kv.get_op_type() == polyfp::o_add)
                    {
                        index_args.push_back(left_index + right_index);
                        index_flag = true;  
                    }else if(kv.get_op_type() == polyfp::o_mul)
                    {
                        index_args.push_back(left_index * right_index);
                        index_flag = true;
                    }else if(kv.get_op_type() == polyfp::o_div)
                    {
                        index_args.push_back(left_index.floorDiv(right_index));
                        index_flag = true;
                    }
                }
                else
                {
                    int loc =0;
                    int loc_2 =0;
                    std::string tile_name1;
                    std::string tile_name2;
                    int tile_size;
                    auto name_set = comp->get_loop_level_names();
                    //MOD3
                    auto t_name = kv.get_name();
                    if(comp->refused == true)
                    {
                        t_name = comp->temp_access_map[t_name];
                    }

                    if ( std::find(name_set.begin(), name_set.end(), t_name) == name_set.end() )
                    {
                        for (auto &kv2: comp->get_access_map())
                        {
                            if(t_name==kv2.first){
                                tile_name1 = kv2.second;
                                loc = comp->iterators_location_map[tile_name1];
                                // loc = comp->get_loop_level_number_from_dimension_name(kv2.second);
                            }
                        }
                        
                        mlir::Value value = ops[loc].getInductionVar();
                        if ( std::find(index_values.begin(), index_values.end(), value)== index_values.end() )
                        {
                            index_values.push_back(value);
                        }
                        if(comp->is_tiled ==true)
                        {
                            for (auto &kv3: comp->get_tile_map())
                            {
                                if(tile_name1==kv3.first)
                                {
                                    loc = comp->iterators_location_map[kv3.second];
                                    // loc = comp->get_loop_level_number_from_dimension_name(kv3.second);
                                    tile_name2 = kv3.second;                              
                                }
                            }
                            for (auto &kv4: comp->get_tile_size_map())
                            {
                                if(tile_name1==kv4.first){
                                    tile_size = kv4.second;
                                }
                            }
                            mlir::Value value2 = ops[loc].getInductionVar();
                            if ( std::find(index_values.begin(), index_values.end(), value2)== index_values.end() )
                            {
                                index_values.push_back(value2);
                            }
                            //TODO
                            int index_2 = std::find(index_values.begin(), index_values.end(), value2) - index_values.begin();
                            int index_3 = std::find(index_values.begin(), index_values.end(), value) - index_values.begin();
                            index_args.push_back(builder.getAffineDimExpr(index_3)+builder.getAffineDimExpr(index_2)*tile_size);
                        }
                        else if(comp->is_skewed == true)
                        {                           
                            loc = comp->iterators_location_map[comp->iterator_to_skew];                            
                            mlir::Value value2 = ops[loc].getInductionVar();
                            if ( std::find(index_values.begin(), index_values.end(), value2)== index_values.end() )
                            {                                
                                index_values.push_back(value2);
                            }
                            //TODO: find the right dim
                            int index_2 = std::find(index_values.begin(), index_values.end(), value2) - index_values.begin();
                            int index_3 = std::find(index_values.begin(), index_values.end(), value) - index_values.begin();
                            // placeholder_index_args.push_back(builder.getAffineDimExpr(index_3)-builder.getAffineDimExpr(index_2)*2);
                            // placeholder_index_args.push_back(builder.getAffineDimExpr(index_3));
                            if(tile_name1 == comp->iterator_to_modify)
                            {                               
                                index_args.push_back(builder.getAffineDimExpr(index_3)-builder.getAffineDimExpr(index_2)*comp->skew_factor);
                            }
                            else
                            {
                                index_args.push_back(builder.getAffineDimExpr(index_3));
                            }
                        
                            index_flag = true;
                    }
                    else
                    {
                        index_args.push_back(builder.getAffineDimExpr(loc));
                    }

                    index_flag = true;     

                    }
                    else
                    {
                        if(comp->refused == true)
                        {
                            t_name = comp->temp_access_map[t_name];
                        }
                        else
                        {
                            loc = comp->iterators_location_map[kv.get_name()];
                        }
                        
                        // loc = comp->get_loop_level_number_from_dimension_name(kv.get_name());
                        int index = 0;
                        for(int i=0; i<start_loops_position.size(); i++)
                        {
                            if(start_loops_position[i]>level&&start_loops_position[i-1]<=level)
                            {
                                index = start_loops_position[i-1];
                                break;
                            }
                            if(i == start_loops_position.size()-1 )
                            {
                                index = start_loops_position[i];
                                break;
                            }
                        }
                        mlir::Value value = ops[loc].getInductionVar();
                        if ( std::find(index_values.begin(), index_values.end(), value) == index_values.end())
                        {
                            index_values.push_back(value);
                        }
                        //TODO
                        int index_1 = std::find(index_values.begin(), index_values.end(), value) - index_values.begin();
                        index_args.push_back(builder.getAffineDimExpr(index_1));
                    }
                }
            }
            auto map = mlir::AffineMap::get(index_values.size(), 0, ArrayRef<mlir::AffineExpr> (index_args),builder.getContext());
            mlir::ValueRange vr=llvm::makeArrayRef(index_values);
            //TODO.number of variables
            mlir::AffineLoadOp a;
            if(index_flag == true)
            {
                a = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), arg_a,map,vr);
            }
            else
            {
                a = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), arg_a ,vr);
            }
            if(if_flag == true)
            {
                mlir::Value value = ops[2].getInductionVar();
                SmallVector<mlir::Value, 4> ifOperands;
                ifOperands.push_back(value);
                SmallVector<mlir::AffineExpr, 4> ifExprs;
                ifExprs.push_back(builder.getAffineDimExpr(0));
                SmallVector<bool, 4> ifEqFlags;
                ifEqFlags.push_back(true);
                const auto condition = mlir::IntegerSet::get(1, 0, ifExprs, ifEqFlags);
                auto ifOp =builder.create<mlir::AffineIfOp>(builder.getUnknownLoc(), condition, ifOperands,/*withElseRegion=*/false);
                builder.setInsertionPointToStart(ifOp.getBody());
                auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), a, mem, vr);
                builder.setInsertionPointAfter(ifOp);
            }
            else
            {
                if(placeholder_index_flag == true)
                {
                    auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), a, mem, placeholder_map, placeholder_vr);
                }
                else
                {
                    auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), a, mem, placeholder_vr);
                }   
            }
        }
        else if(polyfp_expr.get_op_type() == polyfp::o_max)
        {
            mlir::ValueRange indices = {};
            auto a = polyfp_expr.get_operand(0);
            auto b = polyfp_expr.get_operand(1);
            mlir::BlockArgument left;
            mlir::BlockArgument right;
            // mlir::arith::MulFOp allocSize_m;
            // mlir::arith::AddFOp allocSize_a;
            // theModule.dump();
            a_print_expr(polyfp_expr, comp, level);

            if(if_flag == true)
            {
                mlir::Value value = ops[2].getInductionVar();
                SmallVector<mlir::Value, 4> ifOperands;
                ifOperands.push_back(value);
                SmallVector<mlir::AffineExpr, 4> ifExprs;
                ifExprs.push_back(builder.getAffineDimExpr(0));
                SmallVector<bool, 4> ifEqFlags;
                ifEqFlags.push_back(true);
                const auto condition = mlir::IntegerSet::get(1, 0, ifExprs, ifEqFlags);
                auto ifOp =builder.create<mlir::AffineIfOp>(builder.getUnknownLoc(), condition, ifOperands,/*withElseRegion=*/false);
                builder.setInsertionPointToStart(ifOp.getBody());
                //TODO: other arithmetic? sub, o_div
                auto the_op = all_current_op.back();
                
                auto index = the_op.index();
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_vr);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_vr);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_vr);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_vr);
                }
                builder.setInsertionPointAfter(ifOp);
            }
            else
            {
                auto the_op = all_current_op.back();
                auto index = the_op.index();
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    if(placeholder_index_flag == true){
                        auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_map, placeholder_vr);
                        builder.setInsertionPointAfter(store1);
                    }else{
                        auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_vr);
                        builder.setInsertionPointAfter(store1);
                    } 
                    
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    if(placeholder_index_flag == true){
                        auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_map, placeholder_vr);
                        builder.setInsertionPointAfter(store1);
                    }else{
                        auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_vr);
                        builder.setInsertionPointAfter(store1);
                    }
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    if(placeholder_index_flag == true){
                        auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_map, placeholder_vr);
                        builder.setInsertionPointAfter(store1);
                    }else{
                        auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_vr);
                        builder.setInsertionPointAfter(store1);
                    }
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    if(placeholder_index_flag == true){
                        auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_map, placeholder_vr);
                        builder.setInsertionPointAfter(store1);
                    }else{
                        auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_vr);
                        builder.setInsertionPointAfter(store1);
                    }
                }
                else if(index==4){
                    auto op_to_process = std::get<4>(the_op);

                    if(placeholder_index_flag == true){
                        auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_map, placeholder_vr);
                        builder.setInsertionPointAfter(store1);
                    }else{
                        auto store1 = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), op_to_process, mem, placeholder_vr);
                        builder.setInsertionPointAfter(store1);

                    }

                }
            }

        }
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_if)
    {
        isl_ast_expr *cond = isl_ast_node_if_get_cond(node);
        isl_ast_node *if_stmt = isl_ast_node_if_get_then(node);
        isl_ast_node *else_stmt = isl_ast_node_if_get_else(node);
        mlirGen1(fct,if_stmt,level,false,true,true);
    }
    return theModule;
}


mlir::AffineExpr polyfp::MLIRGenImpl::a_print_index(polyfp::expr polyfp_expr, polyfp::compute *comp, std::vector<mlir::Value> &index_values, int level){
    mlir::AffineExpr index_value;
    int loc =  0;
    int loc2=0;
    if (polyfp_expr.get_expr_type() == polyfp::e_op){ 
        auto expr0 = polyfp_expr.get_operand(0);
        auto expr1 = polyfp_expr.get_operand(1);
        auto left_index = a_print_index(expr0,comp,index_values,level);
        auto right_index = a_print_index(expr1,comp, index_values,level);
        if(polyfp_expr.get_op_type() == polyfp::o_sub){
            index_value = left_index - right_index;
        }else if(polyfp_expr.get_op_type() == polyfp::o_add){
            index_value = left_index + right_index;
        }else if(polyfp_expr.get_op_type() == polyfp::o_mul){
            index_value = left_index * right_index;
        }else if(polyfp_expr.get_op_type() == polyfp::o_div){
            index_value = left_index.floorDiv(right_index);
        }
    }
    if (polyfp_expr.get_expr_type() == polyfp::e_var){
        loc = get_iterator_location_from_name(comp,polyfp_expr,index_values);
        auto name_set = comp->get_loop_level_names();
        if(comp->is_tiled == true&&std::find(name_set.begin(), name_set.end(), polyfp_expr.get_name()) == name_set.end()){
            int loc_2 =0;
            std::string tile_name1;
            std::string tile_name2;
            int tile_size;
            if(comp->is_tiled ==true)
            {
                for (auto &kv2: comp->get_access_map())
                {
                    if(polyfp_expr.get_name()==kv2.first)
                    {
                        tile_name1 = kv2.second;
                        loc2 = comp->iterators_location_map[kv2.second];
                    }
                }
                for (auto &kv3: comp->get_tile_map())
                {
                    if(tile_name1==kv3.first)
                    {
                        loc = comp->iterators_location_map[kv3.second];
                        tile_name2 = kv3.second;
                    }
                }
                for (auto &kv4: comp->get_tile_size_map()){
                    if(tile_name1==kv4.first)
                    {
                        tile_size = kv4.second;                    
                    }
                }
                
                mlir::Value value2 = ops[loc].getInductionVar();
                mlir::Value value3 = ops[loc2].getInductionVar();
                if ( std::find(index_values.begin(), index_values.end(), value2) == index_values.end())
                {
                    index_values.push_back(value2);
                }

                int index2 = std::find(index_values.begin(), index_values.end(), value2) - index_values.begin();
                //TODO, 这里index3支持了，其他地方可能还有问题
                int index3 = std::find(index_values.begin(), index_values.end(), value3) - index_values.begin();
                index_value = builder.getAffineDimExpr(index3)+builder.getAffineDimExpr(index2)*tile_size;
            }
  
        }else if(comp->is_skewed == true)
        {        
            int loc_2 =0;
            std::string tile_name1;
            std::string tile_name2;
            int tile_size;
            for (auto &kv2: comp->get_access_map())
            {
                if(polyfp_expr.get_name()==kv2.first)
                {
                    tile_name1 = kv2.second;
                    loc = comp->iterators_location_map[tile_name1];
                }
            }
            mlir::Value value = ops[loc].getInductionVar();
            if ( std::find(index_values.begin(), index_values.end(), value)== index_values.end() )
            {
                
                index_values.push_back(value);
            }
           
            loc = comp->iterators_location_map[comp->iterator_to_skew];
            mlir::Value value2 = ops[loc].getInductionVar();
            if ( std::find(index_values.begin(), index_values.end(), value2)== index_values.end() ){
                index_values.push_back(value2);
            }
            //TODO:
            int index_2 = std::find(index_values.begin(), index_values.end(), value2) - index_values.begin();
            int index_3 = std::find(index_values.begin(), index_values.end(), value) - index_values.begin();
            if(tile_name1 == comp->iterator_to_modify){
                index_value=builder.getAffineDimExpr(index_3)-builder.getAffineDimExpr(index_2)*comp->skew_factor;
            }else{
                index_value =builder.getAffineDimExpr(index_3);
            }
            // index_value = builder.getAffineDimExpr(index_3)+builder.getAffineDimExpr(index_2)*tile_size;
        }
        else{
            index_value = builder.getAffineDimExpr(loc);
        }
    }  
    if (polyfp_expr.get_expr_type() == polyfp::e_val)
    {   
        index_value = getAffineConstantExpr(polyfp_expr.get_int_val(),builder.getContext());
    }  
    // index_value = builder.getAffineDimExpr(loc%2);
    
    return index_value;

}


// Dump expression
void polyfp::MLIRGenImpl::a_print_expr(polyfp::expr polyfp_expr, polyfp::compute *comp, int level){
    auto left = polyfp_expr.get_operand(0);
    auto right = polyfp_expr.get_operand(1);
    if ((right.get_op_type() == polyfp::o_access || right.get_expr_type() == polyfp::e_var ) && (left.get_op_type() == polyfp::o_access || left.get_expr_type() == polyfp::e_var))
    {
        // theModule.dump();
        mlir::AffineLoadOp loadedRhs;
        mlir::AffineLoadOp loadedLhs;
        mlir::Value arg_left;
        mlir::Value arg_right;
        if(left.get_op_type() == polyfp::o_access)
        {
            std::string a_name = left.get_name();
            int index_a;
            // if(comp->refused == true){
            //     a_name = comp->temp_access_map[a_name];
            // }
            // std::cout<<a_name<<std::endl;
            for(int i=0; i<argument_list.size(); i++)
            {
                if(argument_list.at(i) == a_name)
                    index_a = i;               
            }

            // auto arg_a = funcs[0].getArgument(index_a);
            mlir::Value arg_a;
            
            if(index_a+1<=funcs[0].getNumArguments())
            {
                arg_a = funcs[0].getArgument(index_a);
            }
            else
            {
                arg_a = values[index_a-funcs[0].getNumArguments()];
            }

            std::vector<mlir::Value> index_values;
            SmallVector<mlir::AffineExpr> index_args;
            bool index_flag = false;

            for (auto &kv: left.get_access())
            {
                int bias = 0;
                if(kv.get_expr_type() == polyfp::e_op)
                {
                    auto expr0 = kv.get_operand(0);
                    auto expr1 = kv.get_operand(1);
                    auto left_index = a_print_index(expr0,comp,index_values,level);
                    auto right_index = a_print_index(expr1,comp,index_values,level);
                    if(kv.get_op_type() == polyfp::o_sub){
                        index_args.push_back(left_index - right_index);
                        index_flag = true;
                    }else if(kv.get_op_type() == polyfp::o_add){
                        index_args.push_back(left_index + right_index);
                        index_flag = true;
                    }else if(kv.get_op_type() == polyfp::o_mul){
                        index_args.push_back(left_index * right_index);
                        index_flag = true;
                    }else if(kv.get_op_type() == polyfp::o_div){
                        index_args.push_back(left_index.floorDiv(right_index));
                        index_flag = true;
                    }
                }
                else
                {
                    int loc =0;
                    int loc_2 =0;
                    std::string tile_name1;
                    std::string tile_name2;
                    int tile_size;
                    auto name_set = comp->get_loop_level_names();
                    auto t_name = kv.get_name();
                    if(comp->refused == true)
                    {
                        t_name = comp->temp_access_map[t_name];
                    }

                    if ( std::find(name_set.begin(), name_set.end(), t_name) == name_set.end() )
                    {                        
                        for (auto &kv2: comp->get_access_map())
                        {
                            if(t_name==kv2.first)
                            {
                                tile_name1 = kv2.second;
                                loc = comp->iterators_location_map[kv2.second];                        
                            }
                        }
                        
                        mlir::Value value = ops[loc].getInductionVar();
                        if ( std::find(index_values.begin(), index_values.end(), value) == index_values.end() )
                        {
                            index_values.push_back(value);
                        }

                        if(comp->is_tiled ==true)
                        {
                            for (auto &kv3: comp->get_tile_map()){
                                
                                if(tile_name1==kv3.first)
                                {
                                    loc = comp->iterators_location_map[kv3.second];                                
                                    tile_name2 = kv3.second;                              
                                }
                            }
                            for (auto &kv4: comp->get_tile_size_map())
                            {
                                
                                if(tile_name1==kv4.first){
                                    tile_size = kv4.second;
                                }
                            }
                            mlir::Value value2 = ops[loc].getInductionVar();
                            if ( std::find(index_values.begin(), index_values.end(), value2)== index_values.end() )
                            {
                                
                                index_values.push_back(value2);
                            }
                            //TODO:
                            int index_2 = std::find(index_values.begin(), index_values.end(), value2) - index_values.begin();
                            
                            int index_3 = std::find(index_values.begin(), index_values.end(), value) - index_values.begin();

                            index_args.push_back(builder.getAffineDimExpr(index_3)+builder.getAffineDimExpr(index_2)*tile_size);
                        }
                        else if(comp->is_skewed == true)
                        {
                            
                            loc = comp->iterators_location_map[comp->iterator_to_skew];
                            
                            mlir::Value value2 = ops[loc].getInductionVar();
                            if ( std::find(index_values.begin(), index_values.end(), value2)== index_values.end() )
                            {                         
                                index_values.push_back(value2);
                            }
                            //TODO: find the right dim
                            int index_2 = std::find(index_values.begin(), index_values.end(), value2) - index_values.begin();
                            int index_3 = std::find(index_values.begin(), index_values.end(), value) - index_values.begin();
                            // placeholder_index_args.push_back(builder.getAffineDimExpr(index_3)-builder.getAffineDimExpr(index_2)*2);
                            // placeholder_index_args.push_back(builder.getAffineDimExpr(index_3));
                            if(tile_name1 == comp->iterator_to_modify)
                            {
                                index_args.push_back(builder.getAffineDimExpr(index_3)-builder.getAffineDimExpr(index_2)*comp->skew_factor);
                            }
                            else
                            {
                                index_args.push_back(builder.getAffineDimExpr(index_3));
                            }
                        
                            index_flag = true;
                        }
                        
                        
                        else{
                            // TODO std::cout<<"something went wrong"<<std::endl;
                        }
                        index_flag = true;              
                    }
                    else
                    {
                        if(comp->refused == true)
                        {
                            loc = comp->iterators_location_map[t_name];
                        }
                        else
                        {
                            loc = comp->iterators_location_map[kv.get_name()];
                        }
                        
                       
                        mlir::Value value = ops[loc].getInductionVar();
                        if ( std::find(index_values.begin(), index_values.end(), value) == index_values.end())
                        {                   
                            index_values.push_back(value);
                        }
                        //TODO index_1 + index ?
                        int index_1 = std::find(index_values.begin(), index_values.end(), value) - index_values.begin();
                        index_args.push_back(builder.getAffineDimExpr(index_1));
                    }
                } 
            }
            auto map = mlir::AffineMap::get(index_values.size(), 0, ArrayRef<mlir::AffineExpr> (index_args),builder.getContext());
            mlir::ValueRange vr=llvm::makeArrayRef(index_values);
            //TODO.number of variables
            mlir::AffineLoadOp a;
            if(index_flag == true)
            {
                loadedLhs = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), arg_a,map,vr);
            }else{
                loadedLhs = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), arg_a ,vr);
            }
        }
        else if(left.get_expr_type() == polyfp::e_var){
            int index_argument;
            std::string arg_name = left.get_name();
           
            for(int i=0; i<argument_list.size(); i++)
            {
                if(argument_list.at(i) == arg_name){
                    index_argument = i;
                }
            }

            if(index_argument+1<=funcs[0].getNumArguments())
            {
                arg_left = funcs[0].getArgument(index_argument);
            }else{
                arg_left = values[index_argument-funcs[0].getNumArguments()];
            }
        }
        if(right.get_op_type() == polyfp::o_access)
        {

            std::string a_name = right.get_name();
            int index_a;
            
            for(int i=0; i<argument_list.size(); i++){
                if(argument_list.at(i) == a_name)
                    index_a = i;               
            }
            // auto arg_a = funcs[0].getArgument(index_a);
            mlir::Value arg_a;
            if(index_a+1<=funcs[0].getNumArguments()){
                arg_a = funcs[0].getArgument(index_a);
            }else{
                arg_a = values[index_a-funcs[0].getNumArguments()];
            }
            std::vector<mlir::Value> index_values;
            SmallVector<mlir::AffineExpr> index_args;
            bool index_flag = false;
            for (auto &kv: right.get_access())
            {
                int bias = 0;
                if(kv.get_expr_type() == polyfp::e_op)
                {
                    auto expr0 = kv.get_operand(0);
                    auto expr1 = kv.get_operand(1);
                    auto left_index = a_print_index(expr0,comp,index_values,level);
                    auto right_index = a_print_index(expr1,comp,index_values,level);
                    if(kv.get_op_type() == polyfp::o_sub){
                        index_args.push_back(left_index - right_index);
                        index_flag = true;
                    }else if(kv.get_op_type() == polyfp::o_add){
                        index_args.push_back(left_index + right_index);
                        index_flag = true;   
                    }else if(kv.get_op_type() == polyfp::o_mul){
                        index_args.push_back(left_index * right_index);
                        index_flag = true;    
                    }else if(kv.get_op_type() == polyfp::o_div){
                        index_args.push_back(left_index.floorDiv(right_index));
                        index_flag = true;   
                    }
                }
                else
                {
                    int loc =0;
                    int loc_2 =0;
                    std::string tile_name1;
                    std::string tile_name2;
                    int tile_size;
                    auto name_set = comp->get_loop_level_names();
                    auto t_name = kv.get_name();
                    if(comp->refused == true)
                    {
                        t_name = comp->temp_access_map[t_name];
                    }
                    if ( std::find(name_set.begin(), name_set.end(), t_name) == name_set.end() )
                    {
                        for (auto &kv2: comp->get_access_map())
                        {
                            if(t_name==kv2.first)
                            {
                                tile_name1 = kv2.second;
                                loc = comp->iterators_location_map[kv2.second];
                                // loc = comp->get_loop_level_number_from_dimension_name(kv2.second);
                            }
                        }
                        int index = 0;
                        for(int i=0; i<start_loops_position.size(); i++)
                        {
                            if(start_loops_position[i]>level&&start_loops_position[i-1]<=level)
                            {
                                index = start_loops_position[i-1];
                                break;
                            }
                            if(i == start_loops_position.size()-1 )
                            {
                                index = start_loops_position[i];
                                break;
                            }
                        }

                        mlir::Value value = ops[loc].getInductionVar();       

                        if ( std::find(index_values.begin(), index_values.end(), value)== index_values.end() )
                        {
                            index_values.push_back(value);
                        }

                        if(comp->is_tiled ==true)
                        {
                            for (auto &kv3: comp->get_tile_map())
                            {
                                if(tile_name1==kv3.first)
                                {
                                    loc = comp->iterators_location_map[kv3.second];
                                    // loc_2 = comp->get_loop_level_number_from_dimension_name(kv3.second);
                                    tile_name2 = kv3.second;
                                }
                            }
                            for (auto &kv4: comp->get_tile_size_map())
                            {
                                if(tile_name1==kv4.first)
                                {
                                    tile_size = kv4.second;
                                }
                            }
                            mlir::Value value2 = ops[loc].getInductionVar();
                            if ( std::find(index_values.begin(), index_values.end(), value2)== index_values.end() )
                            {
                                index_values.push_back(value2);
                            }
                            int index_2 = std::find(index_values.begin(), index_values.end(), value2) - index_values.begin();
                            int index_3 = std::find(index_values.begin(), index_values.end(), value) - index_values.begin();
                            index_args.push_back(builder.getAffineDimExpr(index_3)+builder.getAffineDimExpr(index_2)*tile_size);
                        }
                        else if(comp->is_skewed == true)
                        {
                            
                            loc = comp->iterators_location_map[comp->iterator_to_skew];                           
                            mlir::Value value2 = ops[loc].getInductionVar();
                            if ( std::find(index_values.begin(), index_values.end(), value2)== index_values.end() )
                            {
                                
                                index_values.push_back(value2);
                            }
                            //TODO: find the right dim
                            int index_2 = std::find(index_values.begin(), index_values.end(), value2) - index_values.begin();
                            int index_3 = std::find(index_values.begin(), index_values.end(), value) - index_values.begin();
                            // placeholder_index_args.push_back(builder.getAffineDimExpr(index_3)-builder.getAffineDimExpr(index_2)*2);
                            // placeholder_index_args.push_back(builder.getAffineDimExpr(index_3));
                            if(tile_name1 == comp->iterator_to_modify)
                            {
                                index_args.push_back(builder.getAffineDimExpr(index_3)-builder.getAffineDimExpr(index_2)*comp->skew_factor);
                            }else{
                                index_args.push_back(builder.getAffineDimExpr(index_3));
                            }
                        
                            index_flag = true;
                    }
                    else
                    {
                            // TODO: std::cout<<"something went wrong"<<std::endl;
                    }
                    index_flag = true; 
                      
                    }
                    else
                    {
                        if(comp->refused == true)
                        {
                            loc = comp->iterators_location_map[t_name];
                        }
                        else
                        {
                            loc = comp->iterators_location_map[kv.get_name()];
                        }
                        
                        // loc = comp->get_loop_level_number_from_dimension_name(kv.get_name());
                        int index = 0;
                        for(int i=0; i<start_loops_position.size(); i++){
                            if(start_loops_position[i]>level&&start_loops_position[i-1]<=level){
                                index = start_loops_position[i-1];
                                break;
                            }
                            if(i == start_loops_position.size()-1 ){
                                index = start_loops_position[i];
                                break;
                            }
                        }
                        mlir::Value value = ops[loc].getInductionVar();
                        if(std::find(index_values.begin(), index_values.end(), value) == index_values.end()){
                            index_values.push_back(value);
                        }
                        int index_1 = std::find(index_values.begin(), index_values.end(), value) - index_values.begin();
                        index_args.push_back(builder.getAffineDimExpr(index_1));
                    }
                }      
            }
            auto map = mlir::AffineMap::get(index_values.size(), 0, ArrayRef<mlir::AffineExpr> (index_args),builder.getContext());
            mlir::ValueRange vr=llvm::makeArrayRef(index_values);
            //TODO.number of variables
            mlir::AffineLoadOp a;
            if(index_flag == true){
                loadedRhs = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), arg_a,map,vr);
            }else{
                loadedRhs = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), arg_a ,vr);
            }   
        }
        else if(right.get_expr_type() == polyfp::e_var)
        {
            int index_argument;
            std::string arg_name = right.get_name();
            for(int i=0; i<argument_list.size(); i++){
                if(argument_list.at(i) == arg_name){
                    index_argument = i;
                }
            }
            // arg_right = funcs[0].getArgument(index_argument);
            // mlir::Value arg_right;
            if(index_argument+1<=funcs[0].getNumArguments())
            {
                arg_right = funcs[0].getArgument(index_argument);
            }else{
                arg_right = values[index_argument-funcs[0].getNumArguments()];
            }
        }
        if (right.get_op_type() == polyfp::o_access && left.get_op_type() == polyfp::o_access )
        { 
            if(polyfp_expr.get_op_type() == polyfp::o_add)
            {
                auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), loadedLhs, loadedRhs);
                current_op.push_back(add_1);
                all_current_op.push_back(add_1);
            }else if(polyfp_expr.get_op_type() == polyfp::o_mul)
            {
                auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), loadedLhs, loadedRhs);
                current_op.push_back(mul_1);
                all_current_op.push_back(mul_1);
            }else if(polyfp_expr.get_op_type() == polyfp::o_sub)
            {
                auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), loadedLhs, loadedRhs);
                current_op.push_back(sub_1);
                all_current_op.push_back(sub_1);  
            }else if(polyfp_expr.get_op_type() == polyfp::o_div)
            {
                auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), loadedLhs, loadedRhs);
                current_op.push_back(div_1);
                all_current_op.push_back(div_1);   
            }else if(polyfp_expr.get_op_type() == polyfp::o_max)
            {
                auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), loadedLhs, loadedRhs);
                current_op.push_back(max_1);
                all_current_op.push_back(max_1);   
                // theModule.dump();
            }
        }
        if (right.get_op_type() == polyfp::o_access  && left.get_expr_type() == polyfp::e_var)
        {
            if(polyfp_expr.get_op_type() == polyfp::o_add){
                auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), arg_left, loadedRhs);
                current_op.push_back(add_1);
                all_current_op.push_back(add_1);
            }else if(polyfp_expr.get_op_type() == polyfp::o_mul){
                auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), arg_left, loadedRhs);
                current_op.push_back(mul_1);
                all_current_op.push_back(mul_1);
            }else if(polyfp_expr.get_op_type() == polyfp::o_sub){
                auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), arg_left, loadedRhs);
                current_op.push_back(sub_1);
                all_current_op.push_back(sub_1);
            }else if(polyfp_expr.get_op_type() == polyfp::o_div){
                auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), arg_left, loadedRhs);
                current_op.push_back(div_1);
                all_current_op.push_back(div_1);
            }else if(polyfp_expr.get_op_type() == polyfp::o_max){
                auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), arg_left, loadedRhs);
                current_op.push_back(max_1);
                all_current_op.push_back(max_1);   
            }
        }
        if (left.get_op_type() == polyfp::o_access  && right.get_expr_type() == polyfp::e_var)
        {
            if(polyfp_expr.get_op_type() == polyfp::o_add){
                auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), loadedLhs, arg_right);
                current_op.push_back(add_1);
                all_current_op.push_back(add_1);
            }else if(polyfp_expr.get_op_type() == polyfp::o_mul){
                auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), loadedLhs, arg_right);
                current_op.push_back(mul_1);
                all_current_op.push_back(mul_1);
            }else if(polyfp_expr.get_op_type() == polyfp::o_sub){
                auto sub_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), loadedLhs, arg_right);
                current_op.push_back(sub_1);
                all_current_op.push_back(sub_1);
            }else if(polyfp_expr.get_op_type() == polyfp::o_div){
                auto div_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), loadedLhs, arg_right);
                current_op.push_back(div_1);
                all_current_op.push_back(div_1);
            }else if(polyfp_expr.get_op_type() == polyfp::o_max){
                auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), loadedLhs, arg_right);
                current_op.push_back(max_1);
                all_current_op.push_back(max_1);   
            }
        }
        if (left.get_expr_type() == polyfp::e_var  && right.get_expr_type() == polyfp::e_var)
        {
            if(polyfp_expr.get_op_type() == polyfp::o_add){
                auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), arg_left, arg_right);
                current_op.push_back(add_1);
                all_current_op.push_back(add_1);
            }else if(polyfp_expr.get_op_type() == polyfp::o_mul){
                auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), arg_left, arg_right);
                current_op.push_back(mul_1);
                all_current_op.push_back(mul_1);
            }else if(polyfp_expr.get_op_type() == polyfp::o_sub){
                auto sub_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), arg_left, arg_right);
                current_op.push_back(sub_1);
                all_current_op.push_back(sub_1);
            }else if(polyfp_expr.get_op_type() == polyfp::o_div){
                auto div_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), arg_left, arg_right);
                current_op.push_back(div_1);
                all_current_op.push_back(div_1);
            }else if(polyfp_expr.get_op_type() == polyfp::o_max){
                auto max_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), arg_left, arg_right);
                current_op.push_back(max_1);
                all_current_op.push_back(max_1);   
            }
        }
        // theModule.dump();
    }
    if ((right.get_op_type() == polyfp::o_access || right.get_expr_type() == polyfp::e_var ) && (left.get_op_type() != polyfp::o_access && left.get_expr_type() == polyfp::e_op))
    {
        mlir::AffineLoadOp loadedRhs;
        
        a_print_expr(left,comp,level);
        
        if(right.get_op_type() == polyfp::o_access)
        {
            std::string a_name = right.get_name();
            int index_a;
            for(int i=0; i<argument_list.size(); i++)
            {
                if(argument_list.at(i) == a_name)
                    index_a = i;               
            }
            mlir::Value arg_a;
            if(index_a+1<=funcs[0].getNumArguments())
            {
                arg_a = funcs[0].getArgument(index_a);
            }
            else
            {
                arg_a = values[index_a-funcs[0].getNumArguments()];
            }
            std::vector<mlir::Value> index_values;
            SmallVector<mlir::AffineExpr> index_args;
            bool index_flag = false;
            for (auto &kv: right.get_access())
            {
                int bias = 0;
                if(kv.get_expr_type() == polyfp::e_op)
                {                   
                    auto expr0 = kv.get_operand(0);
                    auto expr1 = kv.get_operand(1);
                    auto left_index = a_print_index(expr0,comp,index_values,level);
                    auto right_index = a_print_index(expr1,comp,index_values,level);
                    if(kv.get_op_type() == polyfp::o_sub){
                        index_args.push_back(left_index - right_index);
                        index_flag = true;
                    }else if(kv.get_op_type() == polyfp::o_add){
                        index_args.push_back(left_index + right_index);
                        index_flag = true;  
                    }else if(kv.get_op_type() == polyfp::o_mul){
                        index_args.push_back(left_index * right_index);
                        index_flag = true; 
                    }else if(kv.get_op_type() == polyfp::o_div){
                        index_args.push_back(left_index.floorDiv(right_index));
                        index_flag = true; 
                    }
                }
                else
                {
                    int loc =0;
                    int loc_2 =0;
                    std::string tile_name1;
                    std::string tile_name2;
                    int tile_size;
                    auto name_set = comp->get_loop_level_names();
                    auto t_name = kv.get_name();
                    if(comp->refused == true)
                    {
                        t_name = comp->temp_access_map[t_name];
                    }
                    if ( std::find(name_set.begin(), name_set.end(), t_name) == name_set.end() )
                    {
                        for (auto &kv2: comp->get_access_map())
                        {
                            if(t_name==kv2.first
                            ){
                                tile_name1 = kv2.second;
                                loc = comp->iterators_location_map[kv2.second];
                                // loc = comp->get_loop_level_number_from_dimension_name(kv2.second);
                            }
                        }
                        int index = 0;
                        for(int i=0; i<start_loops_position.size(); i++)
                        {
                            if(start_loops_position[i]>level&&start_loops_position[i-1]<=level)
                            {
                                index = start_loops_position[i-1];
                                break;
                            }
                            if(i == start_loops_position.size()-1 )
                            {
                                index = start_loops_position[i];
                                break;
                            }
                        }

                        mlir::Value value = ops[loc].getInductionVar();      

                        if(std::find(index_values.begin(), index_values.end(), value)== index_values.end() )
                        {  
                            index_values.push_back(value);
                        }
                        if(comp->is_tiled ==true)
                        {
                            for (auto &kv3: comp->get_tile_map())
                            {
                                if(tile_name1==kv3.first){
                                    loc = comp->iterators_location_map[kv3.second];
                                    // loc_2 = comp->get_loop_level_number_from_dimension_name(kv3.second);
                                    tile_name2 = kv3.second; 
                                }
                            }
                            for (auto &kv4: comp->get_tile_size_map())
                            {
                                if(tile_name1==kv4.first){
                                    tile_size = kv4.second;
                                }
                            }
                            mlir::Value value2 = ops[loc].getInductionVar();
                            if(std::find(index_values.begin(), index_values.end(), value2)== index_values.end() )
                            {   
                                index_values.push_back(value2);
                            }
                            //TODO
                            int index_2 = std::find(index_values.begin(), index_values.end(), value2) - index_values.begin();
                            int index_3 = std::find(index_values.begin(), index_values.end(), value) - index_values.begin();
                            index_args.push_back(builder.getAffineDimExpr(index_3)+builder.getAffineDimExpr(index_2)*tile_size);
                        }
                        else if(comp->is_skewed == true)
                        {
                            
                            loc = comp->iterators_location_map[comp->iterator_to_skew];                            
                            mlir::Value value2 = ops[loc].getInductionVar();
                            if ( std::find(index_values.begin(), index_values.end(), value2)== index_values.end() )
                            {
                                
                                index_values.push_back(value2);
                            }
                            //TODO: find the right dim
                            int index_2 = std::find(index_values.begin(), index_values.end(), value2) - index_values.begin();
                            int index_3 = std::find(index_values.begin(), index_values.end(), value) - index_values.begin();
                            // placeholder_index_args.push_back(builder.getAffineDimExpr(index_3)-builder.getAffineDimExpr(index_2)*2);
                            // placeholder_index_args.push_back(builder.getAffineDimExpr(index_3));
                            if(tile_name1 == comp->iterator_to_modify){
                                index_args.push_back(builder.getAffineDimExpr(index_3)-builder.getAffineDimExpr(index_2)*comp->skew_factor);
                            }else{
                                index_args.push_back(builder.getAffineDimExpr(index_3));
                            }
                        
                            index_flag = true;
                    }
                    else{
                            // TODO
                        }
                        index_flag = true; 

                    }else{
                        if(comp->refused == true){
                            // t_name = comp->temp_access_map[t_name];
                            loc = comp->iterators_location_map[t_name];
                        }else{
                            loc = comp->iterators_location_map[kv.get_name()];
                        }
                        
                        // loc = comp->get_loop_level_number_from_dimension_name(kv.get_name());
                        int index = 0;
                        for(int i=0; i<start_loops_position.size(); i++){
                            if(start_loops_position[i]>level&&start_loops_position[i-1]<=level){
                                index = start_loops_position[i-1];
                                break;
                            }
                            if(i == start_loops_position.size()-1 ){
                                index = start_loops_position[i];
                                break;
                            }
                        }
                        mlir::Value value = ops[loc].getInductionVar();  
                        if ( std::find(index_values.begin(), index_values.end(), value) == index_values.end()){
                            index_values.push_back(value);
                        }
                        //TODO
                        int index_2 = std::find(index_values.begin(), index_values.end(), value) - index_values.begin();
                        index_args.push_back(builder.getAffineDimExpr(index_2));
                    }
                }     
            }
            auto map = mlir::AffineMap::get(index_values.size(), 0, ArrayRef<mlir::AffineExpr> (index_args),builder.getContext());  
            mlir::ValueRange vr=llvm::makeArrayRef(index_values);
            //TODO.number of variables
            mlir::AffineLoadOp a;
            if(index_flag == true){
                loadedRhs = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), arg_a,map,vr);
            }else{
                loadedRhs = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), arg_a ,vr);
            }
            auto the_op = all_current_op.back();
            auto index = the_op.index();
            if(polyfp_expr.get_op_type() == polyfp::o_add)
            {
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(add_1);
                    // add_1.dump();
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(add_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(add_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(add_1);
                }
            }
            else if(polyfp_expr.get_op_type() == polyfp::o_mul)
            {
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(mul_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(mul_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(mul_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(mul_1);
                }
            }
            else if(polyfp_expr.get_op_type() == polyfp::o_sub)
            {
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(sub_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(sub_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(sub_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(sub_1);
                }
            }
            else if(polyfp_expr.get_op_type() == polyfp::o_div)
            {
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(div_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(div_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(div_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(div_1);
                }
            }
            else if(polyfp_expr.get_op_type() == polyfp::o_max)
            {
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(max_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(max_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(max_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(max_1);
                }
            }
        }
        if(right.get_expr_type() == polyfp::e_var)
        {
            int index_argument;
            std::string arg_name = right.get_name();
            for(int i=0; i<argument_list.size(); i++){
                if(argument_list.at(i) == arg_name){
                    index_argument = i;
                }
            }
            // auto arg_right = funcs[0].getArgument(index_argument);
            mlir::Value arg_right;
            if(index_argument+1<=funcs[0].getNumArguments()){
                arg_right = funcs[0].getArgument(index_argument);
            }else{
                arg_right = values[index_argument-funcs[0].getNumArguments()];
            }
            auto the_op = all_current_op.back();
            auto index = the_op.index();
            if(polyfp_expr.get_op_type() == polyfp::o_add)
            {
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,arg_right);
                    all_current_op.push_back(add_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,arg_right);
                    all_current_op.push_back(add_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,arg_right);
                    all_current_op.push_back(add_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,arg_right);
                    all_current_op.push_back(add_1);
                }
            }
            else if(polyfp_expr.get_op_type() == polyfp::o_mul)
            {
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), op_to_process,arg_right);
                    all_current_op.push_back(mul_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), op_to_process,arg_right);
                    all_current_op.push_back(mul_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), op_to_process,arg_right);
                    all_current_op.push_back(mul_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), op_to_process,arg_right);
                    all_current_op.push_back(mul_1);
                }   
            }
            else if(polyfp_expr.get_op_type() == polyfp::o_sub)
            {
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), op_to_process,arg_right);
                    all_current_op.push_back(sub_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), op_to_process,arg_right);
                    all_current_op.push_back(sub_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), op_to_process,arg_right);
                    all_current_op.push_back(sub_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), op_to_process,arg_right);
                    all_current_op.push_back(sub_1);
                }
            }
            else if(polyfp_expr.get_op_type() == polyfp::o_div)
            {
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), op_to_process,arg_right);
                    all_current_op.push_back(div_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), op_to_process,arg_right);
                    all_current_op.push_back(div_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), op_to_process,arg_right);
                    all_current_op.push_back(div_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), op_to_process,arg_right);
                    all_current_op.push_back(div_1);
                }
            }
            else if(polyfp_expr.get_op_type() == polyfp::o_max)
            {
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(max_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(max_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(max_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), loadedRhs, op_to_process);
                    all_current_op.push_back(max_1);
                }
            }
        }    
        // theModule.dump();
    }
    if ((left.get_op_type() == polyfp::o_access || left.get_expr_type() == polyfp::e_var ) && (right.get_op_type() != polyfp::o_access && right.get_expr_type() == polyfp::e_op))
    {
        mlir::AffineLoadOp loadedLhs;
        a_print_expr(right,comp,level);
        if(left.get_op_type() == polyfp::o_access){
            std::string a_name = left.get_name();
            int index_a;
            for(int i=0; i<argument_list.size(); i++){
                if(argument_list.at(i) == a_name)
                    index_a = i;               
            }
            // auto arg_a = funcs[0].getArgument(index_a);
            mlir::Value arg_a;
            if(index_a+1<=funcs[0].getNumArguments()){
                arg_a = funcs[0].getArgument(index_a);
            }else{
                arg_a = values[index_a-funcs[0].getNumArguments()];
            }
            std::vector<mlir::Value> index_values;
            SmallVector<mlir::AffineExpr> index_args;
            bool index_flag = false;
            for (auto &kv: left.get_access()){
                int bias = 0;
                if(kv.get_expr_type() == polyfp::e_op){
                    auto expr0 = kv.get_operand(0);
                    auto expr1 = kv.get_operand(1);
                    auto left_index = a_print_index(expr0,comp,index_values,level);
                    auto right_index = a_print_index(expr1,comp,index_values,level);
                    if(kv.get_op_type() == polyfp::o_sub){               
                        index_args.push_back(left_index - right_index);
                        index_flag = true;
                    }else if(kv.get_op_type() == polyfp::o_add){
                        index_args.push_back(left_index + right_index);
                        index_flag = true;
                    }else if(kv.get_op_type() == polyfp::o_mul){
                        index_args.push_back(left_index * right_index);
                        index_flag = true;  
                    }else if(kv.get_op_type() == polyfp::o_div){
                        index_args.push_back(left_index.floorDiv(right_index));
                        index_flag = true;
                    }
                }
                else{
                    int loc =0;
                    int loc_2 =0;
                    std::string tile_name1;
                    std::string tile_name2;
                    int tile_size;
                    auto name_set = comp->get_loop_level_names();
                    auto t_name = kv.get_name();
                    if(comp->refused == true){
                        t_name = comp->temp_access_map[t_name];
                    }
                    //TODO
                    int index = 0;
                        for(int i=0; i<start_loops_position.size(); i++){
                            if(start_loops_position[i]>level&&start_loops_position[i-1]<=level){
                                index = start_loops_position[i-1];
                                break;
                            }
                            if(i == start_loops_position.size()-1 ){
                                index = start_loops_position[i];
                                break;
                            }
                    }
                    if(std::find(name_set.begin(), name_set.end(), t_name) == name_set.end()){
                        for(auto &kv2: comp->get_access_map()){
                            if(t_name==kv2.first){
                                tile_name1 = kv2.second;
                                loc = comp->iterators_location_map[kv2.second];
                                // loc = comp->get_loop_level_number_from_dimension_name(kv2.second);
                            }
                        }        

                        mlir::Value value = ops[loc].getInductionVar();
                        if(std::find(index_values.begin(), index_values.end(), value)== index_values.end()){
                            index_values.push_back(value);
                        }
                        if(comp->is_tiled ==true){
                            for (auto &kv3: comp->get_tile_map()){
                                if(tile_name1==kv3.first){
                                    loc = comp->iterators_location_map[kv3.second];
                                    // loc_2 = comp->get_loop_level_number_from_dimension_name(kv3.second);
                                    tile_name2 = kv3.second;
                                }
                            }
                            for (auto &kv4: comp->get_tile_size_map()){
                                if(tile_name1==kv4.first){
                                    tile_size = kv4.second;
                                }
                            }
                            mlir::Value value2 = ops[loc].getInductionVar();
                            if(std::find(index_values.begin(), index_values.end(), value2)== index_values.end() ){
                                index_values.push_back(value2);
                            }
                            //TODO
                            int index_2 = std::find(index_values.begin(), index_values.end(), value2) - index_values.begin();
                            int index = std::find(index_values.begin(), index_values.end(), value) - index_values.begin();
                            index_args.push_back(builder.getAffineDimExpr(index)+builder.getAffineDimExpr(index_2)*tile_size);
                        }else if(comp->is_skewed == true){
                            
                            loc = comp->iterators_location_map[comp->iterator_to_skew];
                            
                            mlir::Value value2 = ops[loc].getInductionVar();
                            if ( std::find(index_values.begin(), index_values.end(), value2)== index_values.end() ){
                                
                                index_values.push_back(value2);
                            }
                            //TODO: find the right dim
                            int index_2 = std::find(index_values.begin(), index_values.end(), value2) - index_values.begin();
                            int index_3 = std::find(index_values.begin(), index_values.end(), value) - index_values.begin();
                            // placeholder_index_args.push_back(builder.getAffineDimExpr(index_3)-builder.getAffineDimExpr(index_2)*2);
                            // placeholder_index_args.push_back(builder.getAffineDimExpr(index_3));
                            if(tile_name1 == comp->iterator_to_modify){
                                index_args.push_back(builder.getAffineDimExpr(index_3)-builder.getAffineDimExpr(index_2)*comp->skew_factor);
                            }else{
                                index_args.push_back(builder.getAffineDimExpr(index_3));
                            }
                        
                            index_flag = true;
                    }else{
                            // TODO
                        }
                        index_flag = true; 
                        
                        
                    }else{
                        //TODO
                        // loc = comp->get_loop_level_number_from_dimension_name(kv.get_name());
                        if(comp->refused == true){
                            // t_name = comp->temp_access_map[t_name];
                            loc = comp->iterators_location_map[t_name];
                        }else{
                            loc = comp->iterators_location_map[kv.get_name()];
                        }
                        
                        mlir::Value value = ops[loc].getInductionVar();
                        if(std::find(index_values.begin(), index_values.end(), value) == index_values.end()){
                            index_values.push_back(value);
                        }
                        //TODO
                        int index_1 = std::find(index_values.begin(), index_values.end(), value) - index_values.begin();
                        index_args.push_back(builder.getAffineDimExpr(index_1));
                    }
                }
                
            }
            auto map = mlir::AffineMap::get(index_values.size(), 0, ArrayRef<mlir::AffineExpr> (index_args),builder.getContext());
            mlir::ValueRange vr=llvm::makeArrayRef(index_values);
            //TODO.number of variables
            mlir::AffineLoadOp a;
            if(index_flag == true){
                loadedLhs = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), arg_a,map,vr);
            }else{
                loadedLhs = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), arg_a ,vr);
            }
            auto the_op = all_current_op.back();
            auto index = the_op.index();
            if(polyfp_expr.get_op_type() == polyfp::o_add){
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,loadedLhs);
                    all_current_op.push_back(add_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,loadedLhs);
                    all_current_op.push_back(add_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,loadedLhs);
                    all_current_op.push_back(add_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,loadedLhs);
                    all_current_op.push_back(add_1);
                }
            }
            else if(polyfp_expr.get_op_type() == polyfp::o_mul){
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), op_to_process,loadedLhs);
                    all_current_op.push_back(mul_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), op_to_process,loadedLhs);
                    all_current_op.push_back(mul_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), op_to_process,loadedLhs);
                    all_current_op.push_back(mul_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), op_to_process,loadedLhs);
                    all_current_op.push_back(mul_1);
                } 
            }
            else if(polyfp_expr.get_op_type() == polyfp::o_sub){
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), op_to_process,loadedLhs);
                    all_current_op.push_back(sub_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), op_to_process,loadedLhs);
                    all_current_op.push_back(sub_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), op_to_process,loadedLhs);
                    all_current_op.push_back(sub_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), op_to_process,loadedLhs);
                    all_current_op.push_back(sub_1);
                }      
            }
            else if(polyfp_expr.get_op_type() == polyfp::o_div){
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), op_to_process,loadedLhs);
                    all_current_op.push_back(div_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), op_to_process,loadedLhs);
                    all_current_op.push_back(div_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), op_to_process,loadedLhs);
                    all_current_op.push_back(div_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), op_to_process,loadedLhs);
                    all_current_op.push_back(div_1);
                }  
            }else if(polyfp_expr.get_op_type() == polyfp::o_max){
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(),  op_to_process,loadedLhs);
                    all_current_op.push_back(max_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(),  op_to_process,loadedLhs);
                    all_current_op.push_back(max_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(),  op_to_process,loadedLhs);
                    all_current_op.push_back(max_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(),  op_to_process,loadedLhs);
                    all_current_op.push_back(max_1);
                }
            }
        }
        if(left.get_expr_type() == polyfp::e_var){
            int index_argument;
            std::string arg_name = left.get_name();
            for(int i=0; i<argument_list.size(); i++){
                if(argument_list.at(i) == arg_name){
                    index_argument = i;
                }
            }
            // auto arg_left = funcs[0].getArgument(index_argument);
            mlir::Value arg_left;
            if(index_argument+1<=funcs[0].getNumArguments()){
                arg_left = funcs[0].getArgument(index_argument);
            }else{
                arg_left = values[index_argument-funcs[0].getNumArguments()];
            }
            auto the_op = all_current_op.back();
            auto index = the_op.index();
            if(polyfp_expr.get_op_type() == polyfp::o_add){
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(add_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(add_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(add_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(add_1);
                }
            }
            if(polyfp_expr.get_op_type() == polyfp::o_mul){
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(mul_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(mul_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(mul_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(mul_1);
                }
            }
            if(polyfp_expr.get_op_type() == polyfp::o_sub){
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(sub_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(sub_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(sub_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(sub_1);
                }
            }
            if(polyfp_expr.get_op_type() == polyfp::o_div){
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(div_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(div_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(div_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(div_1);
                }
            }
            else if(polyfp_expr.get_op_type() == polyfp::o_max){
                if(index==0){
                    auto op_to_process = std::get<0>(the_op);
                    auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(max_1);
                }else if(index==1){
                    auto op_to_process = std::get<1>(the_op);
                    auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(max_1);
                }else if(index==2){
                    auto op_to_process = std::get<2>(the_op);
                    auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(max_1);
                }else if(index==3){
                    auto op_to_process = std::get<3>(the_op);
                    auto max_1 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), arg_left, op_to_process);
                    all_current_op.push_back(max_1);
                }
            }
        }
    }
    if ((left.get_op_type() != polyfp::o_access && left.get_expr_type() == polyfp::e_op) && (right.get_op_type() != polyfp::o_access && right.get_expr_type() == polyfp::e_op)){
        mlir::AffineLoadOp loadedLhs;
        a_print_expr(left,comp,level);
        a_print_expr(right,comp,level);
        auto the_op = all_current_op.back();
        auto the_op2 = all_current_op[all_current_op.size()-2];
        auto index = the_op.index();
        auto index2 = the_op2.index();
        //TODO, sub, mul ,div
        if(polyfp_expr.get_op_type() == polyfp::o_add){
            if(index==0){
                auto op_to_process = std::get<0>(the_op);
                if(index2==0){
                    auto op_to_process2 = std::get<0>(the_op2);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,op_to_process2);
                    all_current_op.push_back(add_1);
                }else if(index2==1){
                    auto op_to_process2 = std::get<1>(the_op2);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,op_to_process2);
                    all_current_op.push_back(add_1);
                }
                else if(index2==2){
                    auto op_to_process2 = std::get<2>(the_op2);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,op_to_process2);
                    all_current_op.push_back(add_1);
                }
                else if(index2==3){
                    auto op_to_process2 = std::get<3>(the_op2);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,op_to_process2);
                    all_current_op.push_back(add_1);
                }
                
            }else if(index==1){
                auto op_to_process = std::get<1>(the_op);
                if(index2==0){
                    auto op_to_process2 = std::get<0>(the_op2);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,op_to_process2);
                    all_current_op.push_back(add_1);
                }else if(index2==1){
                    auto op_to_process2 = std::get<1>(the_op2);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,op_to_process2);
                    all_current_op.push_back(add_1);
                }
                else if(index2==2){
                    auto op_to_process2 = std::get<2>(the_op2);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,op_to_process2);
                    all_current_op.push_back(add_1);
                }
                else if(index2==3){
                    auto op_to_process2 = std::get<3>(the_op2);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,op_to_process2);
                    all_current_op.push_back(add_1);
                }
            }else if(index==2){
                auto op_to_process = std::get<2>(the_op);
                if(index2==0){
                    auto op_to_process2 = std::get<0>(the_op2);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,op_to_process2);
                    all_current_op.push_back(add_1);
                }else if(index2==1){
                    auto op_to_process2 = std::get<1>(the_op2);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,op_to_process2);
                    all_current_op.push_back(add_1);
                }
                else if(index2==2){
                    auto op_to_process2 = std::get<2>(the_op2);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,op_to_process2);
                    all_current_op.push_back(add_1);
                }
                else if(index2==3){
                    auto op_to_process2 = std::get<3>(the_op2);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,op_to_process2);
                    all_current_op.push_back(add_1);
                }
            }else if(index==3){
                auto op_to_process = std::get<3>(the_op);
                if(index2==0){
                    auto op_to_process2 = std::get<0>(the_op2);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,op_to_process2);
                    all_current_op.push_back(add_1);
                }else if(index2==1){
                    auto op_to_process2 = std::get<1>(the_op2);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,op_to_process2);
                    all_current_op.push_back(add_1);
                }
                else if(index2==2){
                    auto op_to_process2 = std::get<2>(the_op2);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,op_to_process2);
                    all_current_op.push_back(add_1);
                }
                else if(index2==3){
                    auto op_to_process2 = std::get<3>(the_op2);
                    auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), op_to_process,op_to_process2);
                    all_current_op.push_back(add_1);
                }
            }
        }
        // if(polyfp_expr.get_op_type() == polyfp::o_add){
        //     auto add_1 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), add_op[add_op.size()-2], add_op.back());
        //     all_current_op.push_back(add_1);
        // }
        // else if(polyfp_expr.get_op_type() == polyfp::o_mul){
        //     auto mul_1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), mul_op[mul_op.size()-2], mul_op.back());
        //     all_current_op.push_back(mul_1);
        // }
        // else if(polyfp_expr.get_op_type() == polyfp::o_sub){
        //     auto sub_1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), mul_op[mul_op.size()-2], mul_op.back());
        //     all_current_op.push_back(sub_1);
        // }
        // else if(polyfp_expr.get_op_type() == polyfp::o_div){
        //     auto div_1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), mul_op[mul_op.size()-2], mul_op.back());
        //     all_current_op.push_back(div_1);
        // }
    }
}

mlir::OwningOpRef<mlir::ModuleOp> mlirGen2(mlir::MLIRContext &context, polyfp::function &fct, isl_ast_node *node, int &level) {
    auto manager = MLIRGenImpl(context); 
    manager.mlirGen1(fct,node,level,true, false, false);
    // manager.getModule().dump();
    for(auto &comp : fct.leader_computations)
    {
        for(auto &kv : comp->get_directive_map())
        {
            if(kv.second == "pipeline")
            {
                int loc_2 = comp->get_loop_level_number_from_dimension_name(kv.first);
                int loc = comp->iterators_location_map[kv.first];
                // index = loc + index;
                mlir::scalehls::setLoopDirective(manager.ops[loc], true, comp->II, false, false);
                for(int i=1; i<=loc_2; i++)
                {
                    mlir::scalehls::setLoopDirective(manager.ops[loc-i], false, comp->II, false, true);
                }
            }
        }             
    }
    auto map = manager.get_array_map();
    mlir::scalehls::setTopFuncAttr(manager.get_funcs()[0]);
    for(auto &kv: fct.get_partition_map())
    {
        SmallVector<mlir::scalehls::hls::PartitionKind, 4> kinds;
        SmallVector<unsigned, 4> factors;
        for(auto &factor: std::get<1>(kv))
        {
            factors.push_back(factor);
        }
        for(auto &type: std::get<2>(kv))
        {
            if(type == "cyclic"){
                kinds.push_back(mlir::scalehls::hls::PartitionKind::CYCLIC);
            }else if(type == "block"){
                kinds.push_back(mlir::scalehls::hls::PartitionKind::BLOCK);
            }else if(type == "none"){
                kinds.push_back(mlir::scalehls::hls::PartitionKind::NONE);
            }
        }
        mlir::scalehls::applyArrayPartition(manager.get_funcs()[0].getArgument(map[std::get<0>(kv)]), factors, kinds,/*updateFuncSignature=*/true);
        // manager.getModule().dump();
    }
    // manager.getModule().dump();
    // mlir::scalehls::applyFuncPreprocess(manager.get_funcs()[0], true);         
    mlir::scalehls::applyFuncPreprocess(manager.get_funcs()[0], true);
    // mlir::scalehls::setFuncDirective(manager.get_funcs()[0], false, 1, true);
    for(auto &comp: fct.leader_computations)
    {
        if(comp->is_unrolled == true)
        {
            for(int i=0; i<comp->unroll_dimension.size(); i++)
            {
                // int bias = comp->get_loop_level_number_from_dimension_name(comp->unroll_dimension[i].get_name());
                // int loc = fct.leader_computation_index[comp];
                int loc = comp->iterators_location_map[comp->unroll_dimension[i].get_name()];
                if(comp->unroll_factor[i] != -1)
                {
                    mlir::loopUnrollUpToFactor(manager.ops[loc],comp->unroll_factor[i]);
                }
                else
                {
                    mlir::loopUnrollFull(manager.ops[loc]);
                }
            }  
        }
    }
    // 
    mlir::scalehls::applyMemoryOpts(manager.get_funcs()[0]);

 
manager.getModule().dump();

    // Read target specification JSON file.
    std::string errorMessage;
    std::string pwd = std::filesystem::current_path().parent_path();
    auto configFile = mlir::openInputFile(pwd+"/samples/config.json", &errorMessage);
    if (!configFile) {
        llvm::errs() << errorMessage << "\n";
    }
    // Parse JSON file into memory.
    auto config = llvm::json::parse(configFile->getBuffer());
    if (!config) {
        llvm::errs() << "failed to parse the target spec json file\n";
    }
    auto configObj = config.get().getAsObject();
    if (!configObj) {
        llvm::errs() << "support an object in the target spec json file, found "
                      "something else\n";
    }
    // Collect profiling latency and DSP usage data, where default values are
    // based on Xilinx PYNQ-Z1 board.
    llvm::StringMap<int64_t> latencyMap;
    mlir::scalehls::getLatencyMap(configObj, latencyMap);
    llvm::StringMap<int64_t> dspUsageMap;
    mlir::scalehls::getDspUsageMap(configObj, dspUsageMap);
   
    int dspNum;
    return manager.getModule();
}

mlir::ModuleOp polyfp::MLIRGenImpl::getModule()
{
    return this->theModule;
} 
void gen_mlir(polyfp::function &fct, isl_ast_node *node, int &level)
{
    mlir::MLIRContext context;
    context.disableMultithreading();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::AffineDialect>();
    context.getOrLoadDialect<mlir::math::MathDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::scalehls::HLSDialect>();
    mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen2(context, fct, node, level);
    mlir::verify(*module, false);
    if (failed(mlir::verify(*module, false))) 
    {
        module->emitError("module verification error");
    }
    // module->dump();
    std::error_code error;
    std::string s = fct.get_name();
    std::string pwd = std::filesystem::current_path().parent_path();
    std::string path = pwd+"/samples/"+s+"/"+s+".mlir";
    llvm::raw_fd_ostream os(path, error);
    os << *module;

}
void function::gen_mlir_stmt(){   
    int level = 0;
    gen_mlir(*this,this->get_isl_ast(),level);
}



}
