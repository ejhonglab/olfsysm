all: build

build: Rbuild

install: Rinstall

Rbuild:
	$(MAKE) -C ./R

Rinstall: Rbuild
	$(MAKE) -C ./R install

.PHONY: all install Rbuild Rinstall
