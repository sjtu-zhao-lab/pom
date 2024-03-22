#include "placeholder.h"
#include <iostream>
namespace polyfp{


polyfp::placeholder::placeholder(std::string name, std::vector<int64_t> dim_sizes,
                         polyfp::primitive_t type, polyfp::function *fct):
                         dim_sizes(dim_sizes), fct(fct),
                         name(name), type(type)
{
    if(fct->fct_argument_added == false)
    {
        fct->add_fct_argument(std::pair<std::string, polyfp::placeholder *>(name, this));
        fct->add_placeholder(std::pair<std::string, polyfp::placeholder *>(name, this));
    }
    else
    {
        fct->add_global_argument(std::pair<std::string, polyfp::placeholder *>(name, this));
        fct->add_placeholder(std::pair<std::string, polyfp::placeholder *>(name, this));
    }   
    
}

void placeholder::partition(std::vector<int> factors, std::string type){
    //TODO: CHECK DIMENSIONS AND WARNING
    std::vector<std::string> types;
    for (int dim = 0; dim < factors.size(); ++dim) {
        types.push_back(type);
    }
    this->fct->set_partition(this->get_name(),factors,types);

}

// TODO
void placeholder::partition(std::vector<int> factors, std::vector<std::string> types){
    //TODO: CHECK DIMENSIONS AND WARNING
    this->fct->set_partition(this->get_name(),factors,types);

}


const std::string &placeholder::get_name() const
{
    return name;
}


int placeholder::get_n_dims() const
{
    return this->get_dim_sizes().size();
}


polyfp::primitive_t placeholder::get_elements_type() const
{
    return type;
}


const std::vector<int64_t> &placeholder::get_dim_sizes() const
{
    return dim_sizes;
}

void polyfp::placeholder::dump(bool exhaustive) const
{
    if (exhaustive)
    {
        std::cout << "Buffer \"" << this->name
                  << "\", Number of dimensions: " << this->get_n_dims()
                  << std::endl;

        std::cout << "Dimension sizes: ";
        for (const auto &size : dim_sizes)
        {
            std::cout << "    ";
        }
        std::cout << std::endl;

        std::cout << "Elements type: "
                  << str_from_polyfp_type_primitive(this->type) << std::endl;

        std::cout << std::endl << std::endl;
    }
}
// const std::string &p_max::get_name() const
// {
//     return name;
// }
// polyfp::p_max::p_max(polyfp::expr expr1, polyfp::expr expr2)
// {

//     this->arg_list.push_back(expr1);
//     this->arg_list.push_back(expr2);
// }

// int p_max::get_n_args() const
// {
//     return this->arg_list.size();
// }


}

