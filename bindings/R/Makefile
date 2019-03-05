all: clean
	cp ../../libolfsysm/bin/libolfsysm.a ./olfsysm/src/
	cp ../../libolfsysm/api/olfsysm.hpp ./olfsysm/src/
	R CMD INSTALL "olfsysm"
	rm ./olfsysm/src/libolfsysm.a
	rm ./olfsysm/src/olfsysm.hpp

clean:
	-rm ./olfsysm/src/hook.o
	-rm ./olfsysm/src/libolfsysm.a
	-rm ./olfsysm/src/olfsysm.so
	-rm ./olfsysm/src/olfsysm.hpp

.PHONY: all install clean
