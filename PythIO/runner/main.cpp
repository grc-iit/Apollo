#include <bits/stdint-intn.h>
#include <iostream>
#include "cppflow/include/cppflow/cppflow.h"
#include "cppflow/include/cppflow/raw_ops.h"
#include "tensorflow/c/c_api.h"

int main(int argc, char *argv[]){
	cppflow::model model(argv[1]);
	std::vector<float> input_vec = {10, 20, 30, 40, 50};
	std::vector<int64_t> shape = {5, 1, 1};
	auto input = cppflow::tensor(input_vec, shape);
	// auto shape = input.shape();
	// std::vector<long> shape_data = shape.get_data<long>();
	// auto in_data = input.get_data<long>();
	// for (auto i : in_data)
	//         std::cout << i << std::endl;
	// std::cout<<shape_data[0]<< " " << shape_data[1] << std::endl;
	// auto re_input = cppflow::reshape(input, (shape_data[0], 1, shape_data[1]));
	auto op = model(input);
	return 0;
}
