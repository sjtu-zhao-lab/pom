#ifndef _H_DEBUG_
#define _H_DEBUG_

#include <iostream>


namespace polyfp
{

void str_dump(const std::string &str);
void str_dump(const std::string &str, const char *str2);
void str_dump(const char *str, const char *str2);
void print_indentation();

extern int polyfp_indentation;

} // namespace polyfp

#define ERROR(message, exit_program) {                      \
    std::cerr << "Error in " << __FILE__ << ":"             \
              << __LINE__ << " - " << message << std::endl; \
    if (exit_program)                                       \
    {                                                       \
        exit(1);                                            \
    }                                                       \
}

#endif
