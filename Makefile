all: bindings

lib:
	$(MAKE) -C ./libolfsysm

bindings: lib
	$(MAKE) -C ./bindings

install:
	$(MAKE) -C ./bindings install

.PHONY: all lib bindings
