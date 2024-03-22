#include "expr.h"
#include "function.h";
// #include <polyfp/core.h>
// #include "function.h"
namespace polyfp
{

polyfp::expr& polyfp::expr::operator=(polyfp::expr const & e)
{
    this->_operator = e._operator;
    this->op = e.op;
    this->access_vector = e.access_vector;
    this->defined = e.defined;
    this->name = e.name;
    this->dtype = e.dtype;
    this->etype = e.etype;

    // Copy the integer value
    if (e.get_expr_type() == polyfp::e_val)
    {
        if (e.get_data_type() == polyfp::p_uint8)
        {
            this->uint8_value = e.get_uint8_value();
        }
        else if (e.get_data_type() == polyfp::p_int8)
        {
            this->int8_value = e.get_int8_value();
        }
        else if (e.get_data_type() == polyfp::p_uint16)
        {
            this->uint16_value = e.get_uint16_value();
        }
        else if (e.get_data_type() == polyfp::p_int16)
        {
            this->int16_value = e.get_int16_value();
        }
        else if (e.get_data_type() == polyfp::p_uint32)
        {
            this->uint32_value = e.get_uint32_value();
        }
        else if (e.get_data_type() == polyfp::p_int32)
        {
            this->int32_value = e.get_int32_value();
        }
        else if (e.get_data_type() == polyfp::p_uint64)
        {
            this->uint64_value = e.get_uint64_value();
        }
        else if (e.get_data_type() == polyfp::p_int64)
        {
            this->int64_value = e.get_int64_value();
        }
        else if (e.get_data_type() == polyfp::p_float32)
        {
            this->float32_value = e.get_float32_value();
        }
        else if (e.get_data_type() == polyfp::p_float64)
        {
            this->float64_value = e.get_float64_value();
        }
    }
    return *this;
}
// todo
// polyfp::expr polyfp::expr::substitute(std::vector<std::pair<var, expr>> substitutions) const
// {
//     for (auto &substitution: substitutions)
//         if (this->is_equal(substitution.first))
//             return substitution.second;


//     return apply_to_operands([&substitutions](const expr& e){
//         return e.substitute(substitutions);
//     });
// }

// polyfp::expr polyfp::expr::substitute_access(std::string original, std::string substitute) const {
//     expr && result = this->apply_to_operands([&original, &substitute](const expr& e){
//         return e.substitute_access(original, substitute);
//     });
//     if (result.get_op_type() == o_access && result.name == original)
//     {
//         result.name = substitute;
//     }
//     return result;
// }

polyfp::var::var(std::string name)


{
    assert(!name.empty());

    auto declared = var::declared_vars.find(name);


    if (declared != var::declared_vars.end())
    {
        *this = declared->second;
    }
    else
    {
        this->name = name;
        this->etype = polyfp::e_var;
        this->dtype = global::get_loop_iterator_data_type();
        // this->defined = true;
        if (true)
        {
            var::declared_vars.insert(std::make_pair(name, *this));
           
        }
    }
}

polyfp::var::var(std::string name, polyfp::primitive_t type)
{
    assert(!name.empty());

    auto declared = var::declared_vars.find(name);

    if (declared != var::declared_vars.end())
    {
        assert(declared->second.dtype == type);
        *this = declared->second;
    }
    else
    {
        this->name = name;
        this->etype = polyfp::e_var;
        this->dtype = type;
        // this->defined = true;
        if (true)
        {
            var::declared_vars.insert(std::make_pair(name, *this));
            
        }
    }
}


polyfp::constant::constant(float value, polyfp::primitive_t t, polyfp::function *fct):
                         float_value(value), func(fct), datatype(t){

        this->name = global::generate_new_constant_name();
        this->etype = polyfp::e_var;
        this->dtype = t;
        // fct->add_invariant(*this);
        fct->add_invariant(std::pair<std::string, polyfp::constant *>(name, this));

}

polyfp::primitive_t constant::get_type() const
{
    return dtype;
}


polyfp::p_max::p_max(polyfp::expr value1, polyfp::expr value2, polyfp::op_t o, polyfp::function *fct){

        this->left_value = value1;
        this->right_value = value2;
        this->func = fct;
        this->_operator = o;
        this->etype = polyfp::e_op;
        this->op.push_back(value1);
        this->op.push_back(value2);
        // fct->add_invariant(*this);
        // fct->add_invariant(std::pair<std::string, polyfp::constant *>(name, this));

}

polyfp::expr polyfp::expr::copy() const
{
    return (*this);
}

expr polyfp::expr::operator+(polyfp::expr other) const {
    return polyfp::expr{o_add, *this, other};
}

expr polyfp::expr::operator-(polyfp::expr other) const {
    return polyfp::expr{o_sub, *this, other};
}

expr polyfp::expr::operator*(polyfp::expr other) const {
    return polyfp::expr{o_mul, *this, other};
}

expr polyfp::expr::operator/(polyfp::expr other) const {
    return polyfp::expr{o_div, *this, other};
}

expr polyfp::expr::operator%(polyfp::expr other) const {
    return polyfp::expr{o_mod, *this, other};
}

// todo
// expr memcpy(const buffer &from, const buffer &to) {
//     return expr(o_memcpy, var(p_void_ptr, from.get_name()), var(p_void_ptr, to.get_name()));
// }

// expr allocate(const buffer &b)
// {
//     return expr{o_allocate, b.get_name()};
// }


}
