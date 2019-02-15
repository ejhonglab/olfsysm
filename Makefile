all:
	g++ -std=c++11 model.cpp -Ofast -fopenmp -o model

debug:
	g++ -std=c++11 model.cpp -g -o model

lib:
	g++ -std=c++11 -fPIC -shared model.cpp -Ofast -fopenmp -o model.so
