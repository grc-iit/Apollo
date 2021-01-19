

load_model:
	g++  -I./cppflow/include -Ilibtensorflow2/include -L libtensorflow2/lib/ -l tensorflow -std=gnu++17 -o runner main.cpp

default:
	load_model
