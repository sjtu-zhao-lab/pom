#ifndef _H_polyfp_EXPR_
#define _H_polyfp_EXPR_
#include<algorithm>

#include <map>
#include <unordered_map>
#include <vector>
#include <assert.h>
#include "debug.h"
#include "type.h"

namespace polyfp
{
class function;
class compute;

std::string generate_new_variable_name();
std::string str_from_polyfp_type_expr(polyfp::expr_t type);
std::string str_polyfp_type_op(polyfp::op_t type);
std::string str_from_polyfp_type_primitive(polyfp::primitive_t type);

// class placeholder;
class expr;
class var;
class global;

template <typename T>
using only_integral = typename std::enable_if<std::is_integral<T>::value, expr>::type;

class global
{
private:

    static primitive_t loop_iterator_type;
    static function *implicit_fct;

public:

    static std::string generate_new_placeholder_name()
    {
        static int counter = 0;
        return "b" + std::to_string(counter++);
    }

    static std::string generate_new_constant_name()
    {
        static int counter = 0;
        return "C" + std::to_string(counter++);
    }

    static function *get_implicit_function()
    {
        return global::implicit_fct;
    }

    static void set_implicit_function(function *fct)
    {
        global::implicit_fct = fct;
    }

    // TODO: The default data type
    static void set_default_polyfp_options()
    {
        global::loop_iterator_type = p_float32;
    }

    static void set_loop_iterator_type(primitive_t t) {
        global::loop_iterator_type = t;
    }

    static primitive_t get_loop_iterator_data_type()
    {
        return global::loop_iterator_type;
    }

    global()
    {
        set_default_polyfp_options();
    }
};


class expr
{
    friend class var;
    friend class computation;
    friend class generator;
    friend class p_max;

    // The type of the operator.
    polyfp::op_t _operator;
    std::vector<polyfp::expr> op;

    union
    {
        uint8_t     uint8_value;
        int8_t      int8_value;
        uint16_t    uint16_value;
        int16_t     int16_value;
        uint32_t    uint32_value;
        int32_t     int32_value;
        uint64_t    uint64_value;
        int64_t     int64_value;
        float       float32_value;
        double      float64_value;
    };


    // e.g. {i, j}
    std::vector<polyfp::expr> access_vector;

    bool defined;

protected:

    std::string name;

    polyfp::primitive_t dtype;

    polyfp::expr_t etype;

public:

  
    polyfp::compute *owner;

    // Create an undefined expression.
    expr()
    {
        this->defined = false;

        this->_operator = polyfp::o_none;
        this->etype = polyfp::e_none;
        this->dtype = polyfp::p_none;
    }

    // Create an undefined expression with type.
    expr(polyfp::primitive_t dtype)
    {
        this->defined = false;
        this->_operator = polyfp::o_none;
        this->etype = polyfp::e_none;
        this->dtype = dtype;
    }


    /**
      * Create an expression for a unary operator that applies
      * on a variable. For example: allocate(A) or free(B).
      */
    expr(polyfp::op_t o, std::string name)
    {
        this->_operator = o;
        this->etype = polyfp::e_op;
        this->dtype = polyfp::p_none;
        this->defined = true;
        this->name = name;
    }

    /**
     * Construct an expression for a binary operator.
     */
    expr(polyfp::op_t o, polyfp::expr expr0, polyfp::expr expr1)
    {
        if (expr0.get_data_type() != expr1.get_data_type())
        {
            polyfp::str_dump("Binary operation between two expressions of different types:\n");
            expr0.dump(false);
            polyfp::str_dump(" (" + str_from_polyfp_type_primitive(expr0.get_data_type()) + ")");
            polyfp::str_dump(" and ");
            expr1.dump(false);
            polyfp::str_dump(" (" + str_from_polyfp_type_primitive(expr1.get_data_type()) + ")");
            polyfp::str_dump("\n");
            ERROR("\nThe two expressions should be of the same type. Use casting to elevate the type of one expression to the other.\n", true);
        }

        this->_operator = o;
        this->etype = polyfp::e_op;
        this->dtype = expr0.get_data_type();
        this->defined = true;

        this->op.push_back(expr0);
        this->op.push_back(expr1);
    }


    // Construct an access
    expr(polyfp::op_t o, std::string name,
         std::vector<polyfp::expr> vec,
         polyfp::primitive_t type)
    {
        assert(((o == polyfp::o_access) || (o == polyfp::o_placeholder)) &&
               "The operator is not an access or a placeholder operator.");

        assert(vec.size() > 0);
        assert(name.size() > 0);

        this->_operator = o;
        this->etype = polyfp::e_op;
        this->dtype = type;
        this->defined = true;

        if (o == polyfp::o_access || o == polyfp::o_placeholder)
        {
            this->set_access(vec);
        }
        else
        {
            ERROR("Type of operator is not o_access or o_placeholder, or o_lin_index.", true);
        }

        this->name = name;
    }

    // Construct an unsigned 8-bit integer expression.
    expr(uint8_t val)
    {
        this->etype = polyfp::e_val;
        this->_operator = polyfp::o_none;
        this->defined = true;
        this->dtype = polyfp::p_uint8;
        this->uint8_value = val;
    }

    // Construct a signed 8-bit integer expression.
    expr(int8_t val)
    {
        this->etype = polyfp::e_val;
        this->_operator = polyfp::o_none;
        this->defined = true;
        this->dtype = polyfp::p_int8;
        this->int8_value = val;
    }

    // Construct an unsigned 16-bit integer expression.
    expr(uint16_t val)
    {
        this->defined = true;
        this->etype = polyfp::e_val;
        this->_operator = polyfp::o_none;
        this->dtype = polyfp::p_uint16;
        this->uint16_value = val;
    }

    expr(int16_t val)
    {
        this->defined = true;
        this->etype = polyfp::e_val;
        this->_operator = polyfp::o_none;

        this->dtype = polyfp::p_int16;
        this->int16_value = val;
    }

    expr(uint32_t val)
    {
        this->etype = polyfp::e_val;
        this->_operator = polyfp::o_none;
        this->defined = true;

        this->dtype = polyfp::p_uint32;
        this->uint32_value = val;
    }

    expr(int32_t val)
    {
        this->etype = polyfp::e_val;
        this->_operator = polyfp::o_none;
        this->defined = true;

        this->dtype = polyfp::p_int32;
        this->int32_value = val;
    }

    expr(uint64_t val)
    {
        this->etype = polyfp::e_val;
        this->_operator = polyfp::o_none;
        this->defined = true;

        this->dtype = polyfp::p_uint64;
        this->uint64_value = val;
    }

    expr(int64_t val)
    {
        this->etype = polyfp::e_val;
        this->_operator = polyfp::o_none;
        this->defined = true;

        this->dtype = polyfp::p_int64;
        this->int64_value = val;
    }

    expr(float val)
    {
        this->etype = polyfp::e_val;
        this->_operator = polyfp::o_none;
        this->defined = true;

        this->dtype = polyfp::p_float32;
        this->float32_value = val;
    }

    polyfp::expr copy() const;

    expr(double val)
    {
        this->etype = polyfp::e_val;
        this->_operator = polyfp::o_none;
        this->defined = true;

        this->dtype = polyfp::p_float64;
        this->float64_value = val;
    }

    uint8_t get_uint8_value() const
    {
        assert(this->get_expr_type() == polyfp::e_val);
        assert(this->get_data_type() == polyfp::p_uint8);

        return uint8_value;
    }

    int8_t get_int8_value() const
    {
        assert(this->get_expr_type() == polyfp::e_val);
        assert(this->get_data_type() == polyfp::p_int8);

        return int8_value;
    }

    uint16_t get_uint16_value() const
    {
        assert(this->get_expr_type() == polyfp::e_val);
        assert(this->get_data_type() == polyfp::p_uint16);

        return uint16_value;
    }

    int16_t get_int16_value() const
    {
        assert(this->get_expr_type() == polyfp::e_val);
        assert(this->get_data_type() == polyfp::p_int16);

        return int16_value;
    }

    uint32_t get_uint32_value() const
    {
        assert(this->get_expr_type() == polyfp::e_val);
        assert(this->get_data_type() == polyfp::p_uint32);

        return uint32_value;
    }

    int32_t get_int32_value() const
    {
        assert(this->get_expr_type() == polyfp::e_val);
        assert(this->get_data_type() == polyfp::p_int32);

        return int32_value;
    }

    uint64_t get_uint64_value() const
    {
        assert(this->get_expr_type() == polyfp::e_val);
        assert(this->get_data_type() == polyfp::p_uint64);

        return uint64_value;
    }

    int64_t get_int64_value() const
    {
        assert(this->get_expr_type() == polyfp::e_val);
        assert(this->get_data_type() == polyfp::p_int64);

        return int64_value;
    }

    float get_float32_value() const
    {
        assert(this->get_expr_type() == polyfp::e_val);
        assert(this->get_data_type() == polyfp::p_float32);

        return float32_value;
    }

    double get_float64_value() const
    {
        assert(this->get_expr_type() == polyfp::e_val);
        assert(this->get_data_type() == polyfp::p_float64);

        return float64_value;
    }

    int64_t get_int_val() const
    {
        assert(this->get_expr_type() == polyfp::e_val);

        int64_t result = 0;

        if (this->get_data_type() == polyfp::p_uint8)
        {
            result = this->get_uint8_value();
        }
        else if (this->get_data_type() == polyfp::p_int8)
        {
            result = this->get_int8_value();
        }
        else if (this->get_data_type() == polyfp::p_uint16)
        {
            result = this->get_uint16_value();
        }
        else if (this->get_data_type() == polyfp::p_int16)
        {
            result = this->get_int16_value();
        }
        else if (this->get_data_type() == polyfp::p_uint32)
        {
            result = this->get_uint32_value();
        }
        else if (this->get_data_type() == polyfp::p_int32)
        {
            result = this->get_int32_value();
        }
        else if (this->get_data_type() == polyfp::p_uint64)
        {
            result = this->get_uint64_value();
        }
        else if (this->get_data_type() == polyfp::p_int64)
        {
            result = this->get_int64_value();
        }
        else if (this->get_data_type() == polyfp::p_float32)
        {
            result = this->get_float32_value();
        }
        else if (this->get_data_type() == polyfp::p_float64)
        {
            result = this->get_float64_value();
        }
        else
        {
            ERROR("Calling get_int_val() on a non integer expression.", true);
        }

        return result;
    }

    double get_double_val() const
    {
        assert(this->get_expr_type() == polyfp::e_val);

        double result = 0;

        if (this->get_data_type() == polyfp::p_float32)
        {
            result = this->get_float32_value();
        }
        else if (this->get_data_type() == polyfp::p_float64)
        {
            result = this->get_float64_value();
        }
        else
        {
            ERROR("Calling get_double_val() on a non double expression.", true);
        }

        return result;
    }

    /**
      * Return the value of the \p i 'th operand of the expression.
      * \p i can be 0, 1 or 2.
      */
    const polyfp::expr &get_operand(int i) const
    {
        assert(this->get_expr_type() == polyfp::e_op);
        assert((i < (int)this->op.size()) && "Operand index is out of bounds.");

        return this->op[i];
    }

    // Return the number of arguments of the operator.
    int get_n_arg() const
    {
        assert(this->get_expr_type() == polyfp::e_op);

        return this->op.size();
    }

    polyfp::expr_t get_expr_type() const
    {
        return etype;
    }

    polyfp::primitive_t get_data_type() const
    {
        return dtype;
    }

    const std::string &get_name() const
    {
        assert(
            (this->get_expr_type() == polyfp::e_var) ||
               (this->get_op_type() == polyfp::o_access) ||
               (this->get_op_type() == polyfp::o_placeholder));
        return name;
    }

    void set_name(std::string &name)
    {
        assert((this->get_expr_type() == polyfp::e_var) ||
               (this->get_op_type() == polyfp::o_access));

        this->name = name;
    }

    polyfp::expr replace_op_in_expr(const std::string &to_replace,
                                      const std::string &replace_with)
    {
        if (this->name == to_replace) {
            this->name = replace_with;
            return *this;
        }
        for (int i = 0; i < this->op.size(); i++) {
            polyfp::expr operand = this->get_operand(i);
            this->op[i] = operand.replace_op_in_expr(to_replace, replace_with);
        }
        return *this;
    }

    // Get the type of the operator (polyfp::op_t)
    polyfp::op_t get_op_type() const
    {
        return _operator;
    }

    // e.g. For a placeholder access A[i+1,j], it will return {i+1, j}
    const std::vector<polyfp::expr> &get_access() const
    {
        assert(this->get_expr_type() == polyfp::e_op);
        assert(this->get_op_type() == polyfp::o_access || this->get_op_type() == polyfp::o_placeholder);

        return access_vector;
    }



    // Get the number of dimensions in the access vector.
    int get_n_dim_access() const
    {
        assert(this->get_expr_type() == polyfp::e_op);
        assert(this->get_op_type() == polyfp::o_access);

        return access_vector.size();
    }

    bool is_defined() const
    {
        return defined;
    }


    bool is_equal(polyfp::expr e) const
    {
        bool equal = true;

        if ((this->_operator != e._operator) ||
            (this->op.size() != e.op.size()) ||
            (this->access_vector.size()   != e.access_vector.size())   ||
            (this->defined != e.defined)     ||
            (this->name != e.name)           ||
            (this->dtype != e.dtype)         ||
            (this->etype != e.etype))
        {
            equal = false;
            return equal;
        }

        for (int i = 0; i < this->access_vector.size(); i++)
            equal = equal && this->access_vector[i].is_equal(e.access_vector[i]);

        for (int i = 0; i < this->op.size(); i++)
            equal = equal && this->op[i].is_equal(e.op[i]);

        if ((this->etype == e_val) && (e.etype == e_val))
        {
            if (this->get_int_val() != e.get_int_val())
                equal = false;
            if ((this->get_data_type() == polyfp::p_float32) ||
                (this->get_data_type() == polyfp::p_float64))
                if (this->get_double_val() != e.get_double_val())
                    equal = false;
        }

        return equal;
    }

    bool is_integer() const
    {
        return this->get_expr_type() == e_val &&
                (this->get_data_type() == p_uint8 ||
                 this->get_data_type() == p_uint16 ||
                 this->get_data_type() == p_uint32 ||
                 this->get_data_type() == p_uint64 ||
                 this->get_data_type() == p_int16 ||
                 this->get_data_type() == p_int32 ||
                 this->get_data_type() == p_int8 ||
                 this->get_data_type() == p_int64);
    }


    expr operator+(polyfp::expr other) const;
    expr operator-(polyfp::expr other) const;
    expr operator/(polyfp::expr other) const;
    expr operator*(polyfp::expr other) const;
    expr operator%(polyfp::expr other) const;
    expr operator>>(polyfp::expr other) const;

    // TODO: Extensions
    // Expression multiplied by (-1).

    polyfp::expr& operator=(polyfp::expr const &);

    void set_access(std::vector<polyfp::expr> vector)
    {
        access_vector = vector;
    }

    void set_access_dimension(int i, polyfp::expr acc)
    {
        assert((i < (int)this->access_vector.size()) && "index is out of bounds.");
        access_vector[i] = acc;
    }

    void get_access_vector(std::vector<polyfp::expr> &loads) const{

        switch (this->etype){
            case polyfp::e_op:
            {
                if (this->get_n_arg() > 0)
                {
                    for (int i = 0; i < this->get_n_arg(); i++)
                    {
                        this->op[i].get_access_vector(loads);
                    }
                }
                if ((this->get_op_type() == polyfp::o_access))
                {
                    // std::cout << "Access to " +  this->get_name() + ". Access expressions:" << std::endl;
                    loads.push_back(*this);
                }
                
                break;
            }
            case (polyfp::e_val):
            {
                // TODO: 
                // if (this->get_data_type() == polyfp::p_uint8)
                // {
                //     std::cout << "Value:" << this->get_uint8_value() << std::endl;
                // }
                // else if (this->get_data_type() == polyfp::p_int8)
                // {
                //     std::cout << "Value:" << this->get_int8_value() << std::endl;
                // }
                // else if (this->get_data_type() == polyfp::p_uint16)
                // {
                //     std::cout << "Value:" << this->get_uint16_value() << std::endl;
                // }
                // else if (this->get_data_type() == polyfp::p_int16)
                // {
                //     std::cout << "Value:" << this->get_int16_value() << std::endl;
                // }
                // else if (this->get_data_type() == polyfp::p_uint32)
                // {
                //     std::cout << "Value:" << this->get_uint32_value() << std::endl;
                // }
                // else if (this->get_data_type() == polyfp::p_int32)
                // {
                //     std::cout << "Value:" << this->get_int32_value() << std::endl;
                // }
                // else if (this->get_data_type() == polyfp::p_uint64)
                // {
                //     std::cout << "Value:" << this->get_uint64_value() << std::endl;
                // }
                // else if (this->get_data_type() == polyfp::p_int64)
                // {
                //     std::cout << "Value:" << this->get_int64_value() << std::endl;
                // }
                // else if (this->get_data_type() == polyfp::p_float32)
                // {
                //     std::cout << "Value:" << this->get_float32_value() << std::endl;
                // }
                // else if (this->get_data_type() == polyfp::p_float64)
                // {
                //     std::cout << "Value:" << this->get_float64_value() << std::endl;
                // }
                break;
            }
            case (polyfp::e_var):
            {
                // TODO:
                // std::cout << "Var name:" << this->get_name() << std::endl;
                // std::cout << "Expression value type:" << str_from_polyfp_type_primitive(this->dtype) << std::endl;
                break;
            }
        } 

    }

    void dump(bool exhaustive) const
    {
        if (this->get_expr_type() != e_none)
        {
            if (exhaustive == true)
            {
                if (this->is_defined())
                {
                    std::cout << "Expression:" << std::endl;
                    std::cout << "Expression type:" << str_from_polyfp_type_expr(this->etype) << std::endl;
                    switch (this->etype)
                    {
                    case polyfp::e_op:
                    {
                        std::cout << "Expression operator type:" << str_polyfp_type_op(this->_operator) << std::endl;
                        if (this->get_n_arg() > 0)
                        {
                            std::cout << "Number of operands:" << this->get_n_arg() << std::endl;
                            std::cout << "Dumping the operands:" << std::endl;
                            for (int i = 0; i < this->get_n_arg(); i++)
                            {
                                std::cout << "Operand " << std::to_string(i) << "." << std::endl;
                                this->op[i].dump(exhaustive);
                            }
                        }
                        if ((this->get_op_type() == polyfp::o_access))
                        {
                            std::cout << "Access to " +  this->get_name() + ". Access expressions:" << std::endl;
                            for (const auto &e : this->get_access())
                            {
                                e.dump(exhaustive);
                            }
                        }
                        
                        break;
                    }
                    case (polyfp::e_val):
                    {
                        std::cout << "Expression value type:" << str_from_polyfp_type_primitive(this->dtype) << std::endl;

                        if (this->get_data_type() == polyfp::p_uint8)
                        {
                            std::cout << "Value:" << this->get_uint8_value() << std::endl;
                        }
                        else if (this->get_data_type() == polyfp::p_int8)
                        {
                            std::cout << "Value:" << this->get_int8_value() << std::endl;
                        }
                        else if (this->get_data_type() == polyfp::p_uint16)
                        {
                            std::cout << "Value:" << this->get_uint16_value() << std::endl;
                        }
                        else if (this->get_data_type() == polyfp::p_int16)
                        {
                            std::cout << "Value:" << this->get_int16_value() << std::endl;
                        }
                        else if (this->get_data_type() == polyfp::p_uint32)
                        {
                            std::cout << "Value:" << this->get_uint32_value() << std::endl;
                        }
                        else if (this->get_data_type() == polyfp::p_int32)
                        {
                            std::cout << "Value:" << this->get_int32_value() << std::endl;
                        }
                        else if (this->get_data_type() == polyfp::p_uint64)
                        {
                            std::cout << "Value:" << this->get_uint64_value() << std::endl;
                        }
                        else if (this->get_data_type() == polyfp::p_int64)
                        {
                            std::cout << "Value:" << this->get_int64_value() << std::endl;
                        }
                        else if (this->get_data_type() == polyfp::p_float32)
                        {
                            std::cout << "Value:" << this->get_float32_value() << std::endl;
                        }
                        else if (this->get_data_type() == polyfp::p_float64)
                        {
                            std::cout << "Value:" << this->get_float64_value() << std::endl;
                        }
                        break;
                    }
                    case (polyfp::e_var):
                    {
                        std::cout << "Var name:" << this->get_name() << std::endl;
                        std::cout << "Expression value type:" << str_from_polyfp_type_primitive(this->dtype) << std::endl;
                        break;
                    }

                    }
                }
            }
            else
            {   std::cout << "dump expression"<<std::endl;
                std::cout << this->to_str();
            }
        }
    }

    bool is_constant() const
    {
        if (this->get_expr_type() == polyfp::e_val)
            return true;
        else
            return false;
    }

    int get_dependence_vector() const{
        // TODO: a more general method to calculate dependence vector
        // Already supported: A(i+4,j-5)-> A(i,j-1)
        // Not supported: A(2*i,j), A(i+j, j+9)
        int temp;
        if (this->get_expr_type() == e_op){
            switch (this->get_op_type()){
                case polyfp::o_add:
                    if ((this->get_operand(0).get_expr_type() == polyfp::e_val)){
                        temp = this->get_operand(0).get_int_val();
                    }else if((this->get_operand(1).get_expr_type() == polyfp::e_val)){
                        temp = this->get_operand(1).get_int_val();
                    }else{
                        std::cout<<"not supported type"<<std::endl;
                        return false;
                    }
                case polyfp::o_sub:
                    if ((this->get_operand(0).get_expr_type() == polyfp::e_val)){
                        temp = -(this->get_operand(1).get_int_val());
                    }else if((this->get_operand(1).get_expr_type() == polyfp::e_val)){
                        temp = -(this->get_operand(1).get_int_val());
                    }else{
                        std::cout<<"not supported type"<<std::endl;
                        return false;
                    }
            }
        }else if(this->get_expr_type() == e_var){
            temp = 0;
            

        }else{
            std::cout<<"not supported type"<<std::endl;
            return false;
        }
        return temp;


    }



    // Simplify the expression.
    polyfp::expr simplify() const
    {
        if (this->get_expr_type() != e_none)
        {
            switch (this->etype)
            {
                case polyfp::e_op:
                {
                    switch (this->get_op_type())
                    {
                    case polyfp::o_max:
                        return *this;
                    case polyfp::o_min:
                        return *this;
                    case polyfp::o_add:
                        this->get_operand(0).simplify();
                        this->get_operand(1).simplify();
                        if ((this->get_operand(0).get_expr_type() == polyfp::e_val) && (this->get_operand(1).get_expr_type() == polyfp::e_val))
                            if ((this->get_operand(0).get_data_type() == polyfp::p_int32))
                                return expr(this->get_operand(0).get_int_val() + this->get_operand(1).get_int_val());
                    case polyfp::o_sub:
                        this->get_operand(0).simplify();
                        this->get_operand(1).simplify();
                        if ((this->get_operand(0).get_expr_type() == polyfp::e_val) && (this->get_operand(1).get_expr_type() == polyfp::e_val))
                            if ((this->get_operand(0).get_data_type() == polyfp::p_int32))
                                return expr(this->get_operand(0).get_int_val() - this->get_operand(1).get_int_val());
                    case polyfp::o_mul:
                        this->get_operand(0).simplify();
                        this->get_operand(1).simplify();
                        if ((this->get_operand(0).get_expr_type() == polyfp::e_val) && (this->get_operand(1).get_expr_type() == polyfp::e_val))
                            if ((this->get_operand(0).get_data_type() == polyfp::p_int32))
                                return expr(this->get_operand(0).get_int_val() * this->get_operand(1).get_int_val());
                    case polyfp::o_div:
                        return *this;
                    case polyfp::o_mod:
                        return *this;
                    case polyfp::o_access:
                        return *this;
                    default:
                        ERROR("Simplifying an unsupported polyfp expression.", 1);
                    }
                    break;
                }
                case (polyfp::e_val):
                {
                    return *this;
                }
                case (polyfp::e_var):
                {
                    return *this;
                }
                default:
                    ERROR("Expression type not supported.", true);
            }
        }

        return *this;
    }
#include <iostream>

    std::string to_str() const
    {
        std::string str = std::string("");

        if (this->get_expr_type() != e_none)
        {
            // std::cout<<this->get_expr_type();
                switch (this->etype)
                {
                case polyfp::e_op:
                {
                    switch (this->get_op_type())
                    {
                    case polyfp::o_max:
                        str +=  "max(" + this->get_operand(0).to_str();
                        str +=  ", " + this->get_operand(1).to_str();
                        str +=  ")";
                        break;
                    case polyfp::o_min:
                        str +=  "min(" + this->get_operand(0).to_str();
                        str +=  ", " + this->get_operand(1).to_str();
                        str +=  ")";
                        break;
                    case polyfp::o_add:
                        str +=  "(" + this->get_operand(0).to_str();
                        str +=  " + " + this->get_operand(1).to_str();
                        str +=  ")";
                        break;
                    case polyfp::o_sub:
                        str +=  "(" + this->get_operand(0).to_str();
                        str +=  " - " + this->get_operand(1).to_str();
                        str +=  ")";
                        break;
                    case polyfp::o_mul:
                        str +=  "(" + this->get_operand(0).to_str();
                        str +=  " * " + this->get_operand(1).to_str();
                        str +=  ")";
                        break;
                    case polyfp::o_div:
                        str +=  "(" + this->get_operand(0).to_str();
                        str +=  " / " + this->get_operand(1).to_str();
                        str +=  ")";
                        break;
                    case polyfp::o_mod:
                        str +=  "(" + this->get_operand(0).to_str();
                        str +=  " % " + this->get_operand(1).to_str();
                        str +=  ")";
                        break;
                    case polyfp::o_access:
                    case polyfp::o_placeholder:
                        str +=  this->get_name() + "(";
                        for (int k = 0; k < this->get_access().size(); k++)
                        {
                            if (k != 0)
                            {
                                str +=  ", ";
                            }
                            str += this->get_access()[k].to_str();
                        }
                        str +=  ")";
                        break;
                    default:
                        ERROR("Dumping an unsupported polyfp expression.", 1);
                    }
                    break;
                }
                case (polyfp::e_val):
                {
                    if (this->get_data_type() == polyfp::p_uint8)
                    {
                        str +=  std::to_string((int)this->get_uint8_value());
                    }
                    else if (this->get_data_type() == polyfp::p_int8)
                    {
                        str +=  std::to_string((int)this->get_int8_value());
                    }
                    else if (this->get_data_type() == polyfp::p_uint16)
                    {
                        str +=  std::to_string(this->get_uint16_value());
                    }
                    else if (this->get_data_type() == polyfp::p_int16)
                    {
                        str +=  std::to_string(this->get_int16_value());
                    }
                    else if (this->get_data_type() == polyfp::p_uint32)
                    {
                        str +=  std::to_string(this->get_uint32_value());
                    }
                    else if (this->get_data_type() == polyfp::p_int32)
                    {
                        str +=  std::to_string(this->get_int32_value());
                    }
                    else if (this->get_data_type() == polyfp::p_uint64)
                    {
                        str +=  std::to_string(this->get_uint64_value());
                    }
                    else if (this->get_data_type() == polyfp::p_int64)
                    {
                        str +=  std::to_string(this->get_int64_value());
                    }
                    else if (this->get_data_type() == polyfp::p_float32)
                    {
                        str +=  std::to_string(this->get_float32_value());
                    }
                    else if (this->get_data_type() == polyfp::p_float64)
                    {
                        str +=  std::to_string(this->get_float64_value());
                    }
                    break;
                }
                case (polyfp::e_var):
                {
                    str += this->get_name();
                    break;
                }

                default:
                    ERROR("Expression type not supported.", true);
                }
            }

          return str;
        }

};


class var: public polyfp::expr
{
    friend compute;
private:

    static std::unordered_map<std::string, var> declared_vars;
    expr lower;
    expr upper;

public:

    // Return the upper bound of this variable.
    expr get_upper()
    {
	    return upper;
    }

    expr get_lower()
    {
	    return lower;
    }

    var(std::string name);

    var(std::string name, polyfp::primitive_t type);

    var(std::string name, int lower_bound, int upper_bound) : var(name)
    {
        lower = expr((int32_t) lower_bound);
        upper = expr((int32_t) upper_bound);
        // flag = 0;

    }
    var(std::string name, expr lower_bound, expr upper_bound) : var(name)
    {
        lower = lower_bound;
        upper = upper_bound;
        // flag = 0;
    }
     var(std::string name, int lower_bound, expr upper_bound) : var(name)
    {
        lower = expr((int32_t) lower_bound);
        upper = upper_bound;
        // flag = 0;

    }

    var(): var(generate_new_variable_name()) {}

    void show(){
        std::cout << "Saved variable " << this->name << " of type " << str_from_polyfp_type_primitive(this->dtype)<<std::endl;
    }


};


class constant: public polyfp::expr
{
    friend compute;
    friend function;
private:
    expr value;
    float float_value;
    polyfp::primitive_t datatype;
    polyfp::function *func;
    
public:


    constant(float value = 0, polyfp::primitive_t t = p_float32, polyfp::function *fct = global::get_implicit_function());
    polyfp::primitive_t get_type() const;
    
};

class p_max: public polyfp::expr
{
    friend compute;
    friend function;
private:
    expr left_value;
    expr right_value;
    polyfp::function *func;

    
public:
    p_max( polyfp::expr value1, polyfp::expr value2, polyfp::op_t o = polyfp::o_max, polyfp::function *fct = global::get_implicit_function());

};


/**
  * Takes in a primitive value \p val, and returns an expression
  * of polyfp type \p tT that represents \p val.
  */
template <typename cT>
expr value_cast(primitive_t tT, cT val) {

    switch (tT) {
        case p_int8:
            return expr{static_cast<int8_t>(val)};
        case p_uint8:
            return expr{static_cast<uint8_t>(val)};
        case p_int16:
            return expr{static_cast<int16_t>(val)};
        case p_uint16:
            return expr{static_cast<uint16_t>(val)};
        case p_int32:
            return expr{static_cast<int32_t>(val)};
        case p_uint32:
            return expr{static_cast<uint32_t>(val)};
        case p_int64:
            return expr{static_cast<int64_t>(val)};
        case p_uint64:
            return expr{static_cast<uint64_t>(val)};
        case p_float32:
            return expr{static_cast<float>(val)};
        case p_float64:
            return expr{static_cast<double>(val)};
        default:
            throw std::invalid_argument{"Type not supported"};
    }
}



template <typename T>
only_integral<T> operator+(const polyfp::expr &e, T val)
{
    return e + value_cast(e.get_data_type(), val);
}

template <typename T>
only_integral<T> operator+(T val, const polyfp::expr &e)
{
    return value_cast(e.get_data_type(), val) + e;
}

template <typename T>
only_integral<T> operator-(const polyfp::expr &e, T val)
{
    return e - value_cast(e.get_data_type(), val);
}

template <typename T>
only_integral<T> operator-(T val, const polyfp::expr &e)
{
    return value_cast(e.get_data_type(), val) - e;
}

template <typename T>
only_integral<T> operator/(const polyfp::expr &e, T val)
{
    return e / expr{val};
}

template <typename T>
only_integral<T> operator/(T val, const polyfp::expr &e)
{
    return expr{val} / e;
}

template <typename T>
only_integral<T> operator*(const polyfp::expr &e, T val)
{
    return e * value_cast(e.get_data_type(), val);
}

template <typename T>
only_integral<T> operator*(T val, const polyfp::expr &e)
{
    return value_cast(e.get_data_type(), val) * e;
}

template <typename T>
only_integral<T> operator%(const polyfp::expr &e, T val)
{
    return e % expr{val};
}

template <typename T>
only_integral<T> operator%(T val, const polyfp::expr &e)
{
    return expr{val} % e;
}

}


#endif
