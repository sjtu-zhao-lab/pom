#ifndef _H_PolyFP_TYPE_
#define _H_PolyFP_TYPE_

#include <string.h>
#include <stdint.h>

namespace polyfp
{


// Type of expression
enum expr_t
{
    e_val,          // literal value, like 1, 2.4, 10, ...
    e_var,          // a variable of a primitive type (i.e., an identifier holding one value),
    e_op,           // an operation: add, mul, div, ...
    e_none          // undefined expression. The existence of an expression of e_none type means an error.
};

enum primitive_t
{
    p_uint8,
    p_uint16,
    p_uint32,
    p_uint64,
    p_int8,
    p_int16,
    p_int32,
    p_int64,
    p_float32,
    p_float64,
    // p_boolean,
    p_none
};


// Type of operator
enum op_t
{
    o_add,
    o_sub,
    o_mul,
    o_div,
    o_mod,
    o_max,
    o_min,
    o_access,
    o_placeholder,
    o_none,
};

}

#endif
