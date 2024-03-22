#include <iostream>
#include <sstream>

namespace polyfp
{

int polyfp_indentation = 0;

void str_dump(const std::string &str)
{
    std::cout << str;
}

void str_dump(const std::string &str, const char *str2)
{
    std::cout << str << " " << str2;
}

void str_dump(const char *str, const char *str2)
{
    std::cout << str << " " << str2<<std::endl;
}

void print_indentation()
{
    for (int polyfp_indent = 0; polyfp_indent < polyfp::polyfp_indentation; polyfp_indent++)
    {
        if (polyfp_indent % 4 == 0)
        {
            str_dump("|");
        }
        else
        {
            str_dump(" ");
        }
    }
}

}
