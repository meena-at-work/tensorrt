#include <iostream>

#include <tensorflow/c/c_api.h>

#include "tftrt_nn_runner.h"
int main(void)
{
    std::cout << "Hello from the CPP example of TFTRT, using TF version: " << TF_Version() << std::endl;

    return 0;
}
