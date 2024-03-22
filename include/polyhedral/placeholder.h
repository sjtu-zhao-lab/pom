#ifndef _H_polyfp_PLACEHOLDER_
#define _H_polyfp_PLACEHOLDER_
// #include "compute.h"
#include "function.h"
#include "expr.h"
#include <map>

/**
  * A class that represents placeholders.
  *
  * placeholders have two use cases:
  * - used to store the results of computations, and
  * - used to represent input arguments to functions.
  */

namespace polyfp{
  class compute;

static std::string generate_new_p_operator_name()
{
    static int counter = 0;
    return "p" + std::to_string(counter++);
}


class placeholder
{
    friend compute;
    friend function;
    // friend generator;

private:

    /**
      * The sizes of the dimensions of the placeholder.  Assuming the following
      * placeholder buf[N0][N1][N2], dim_sizes should be {N0, N1, N2}.
      */
    std::vector<int64_t> dim_sizes;

    /**
      * The polyfp function where this placeholder is declared or where the
      * placeholder is an argument.
      */
    polyfp::function *fct;

    /**
      * The name of the placeholder.
      * placeholder names should not start with _ (an underscore).
      * Names starting with _ are reserved names.
      */
    std::string name;

    /**
      * The type of the elements of the placeholder.
      */
    polyfp::primitive_t type;



protected:
    /**
     * Set the size of a dimension of the placeholder.
     */
    void set_dim_size(int dim, int size);

public:
    /**
      * \brief Default polyfp constructor
      */
    placeholder();

    /**
      * A polyfp placeholder is equivalent to an array in C.
      *
      * placeholders have two use cases:
      * - Used to store the results of computes, and
      * - Used to represent input arguments to functions.
      *
      * \p name is the name of the placeholder.
      *
      * \p dim_sizes is a vector of polyfp expressions that represent the
      * size of each dimension in the placeholder.
      * Assuming we want to declare the placeholder buf[N0][N1][N2],
      * then the vector of sizes should be {N0, N1, N2}.
      * placeholder dimensions in polyfp have the same semantics as in
      * C/C++.
      *
      * \p type is the type of the elements of the placeholder.
      * It must be a primitive type (i.e. p_uint8, p_uint16, ...).
      * Possible types are declared in \ref polyfp::primitive_t
      * (in type.h).
      *
      * \p fct is a pointer to a polyfp function where the placeholder is
      * declared or used.  If this argument is not provided (which is
      * the common case), the function that was created automatically
      * during polyfp initialization will be used (we call that
      * function the "implicit function").
      */
    placeholder(std::string name, std::vector<int64_t> dim_sizes,
           polyfp::primitive_t type,
           polyfp::function *fct = global::get_implicit_function());

    
    void dump(bool exhaustive) const;

    const std::string &get_name() const;

    // Get the number of dimensions of the placeholder.
    int get_n_dims() const;

    polyfp::primitive_t get_elements_type() const;
    void partition(std::vector<int> factors, std::string type);
    void partition(std::vector<int> factors, std::vector<std::string> type);

    const std::vector<int64_t> &get_dim_sizes() const;

    template<typename... Args> polyfp::expr operator()(Args... args)
    {
        // TODO move to cpp
        std::vector<polyfp::expr> access_expressions{std::forward<Args>(args)...};
        if (access_expressions.size() != this->get_n_dims())
        {
            polyfp::str_dump("Error - Incorrect access: " + this->get_name() + "(");
            for (int i = 0; i < access_expressions.size(); i++)
            {
                polyfp::expr e = access_expressions[i];
                e.dump(false);
                if (i != access_expressions.size() - 1)
                    polyfp::str_dump(", ");
            }
            polyfp::str_dump(").\n");
            polyfp::str_dump("The number of access dimensions does not match that used in the declaration of " + this->get_name() + ".\n\n");
            exit(1);
        }
            return polyfp::expr(polyfp::o_access,
                                  this->get_name(),
                                  access_expressions,
                                  this->get_elements_type());
        // }
    }


    operator expr();


};







}



#endif