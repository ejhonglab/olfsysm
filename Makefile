all:
	$(MAKE) -C ./libolfsysm
	$(MAKE) -C ./bindings

debug:
	$(MAKE) -C ./libolfsysm debug=1
	$(MAKE) -C ./bindings

.PHONY: all debug
