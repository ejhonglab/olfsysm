all:
	g++ -std=c++14 model.cpp -Ofast -fopenmp -o model

debug:
	g++ -std=c++14 model.cpp -g -o model

lib:
	g++ -std=c++14 -fPIC -shared model.cpp -Ofast -fopenmp -o model.so
