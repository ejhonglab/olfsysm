CXX = g++
CXXFLAGS = -std=c++17 -fopenmp -fpic -I./api/ -I./include/ -c

AR = ar
ARFLAGS = rvcs

SRCDIR = ./src
TGTDIR = ./bin
OBJDIR = ./build
APIDIR = ./api

SRCNAMES = olfsysm.cpp
SOURCES = $(SRCNAMES:%.cpp=$(SRCDIR)/%.cpp)
OBJECTS = $(SRCNAMES:%.cpp=$(OBJDIR)/%.o)

TARGET = $(TGTDIR)/libolfsysm.a

debug ?= 0
ifeq ($(debug), 1)
	DEBUG_FLAGS = -DDEBUG -Og -ggdb3
else
	DEBUG_FLAGS = -Ofast -funroll-loops -march=native
endif

all: $(TARGET)

clean:
	-rm -r $(TGTDIR)
	-rm -r $(OBJDIR)

$(TGTDIR):
	mkdir -p $(TGTDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(TARGET): $(OBJECTS) | $(TGTDIR)
	$(AR) $(ARFLAGS) $(TARGET) $(OBJECTS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(DEBUG_FLAGS) $< -o $@

$(OBJDIR)/olfsysm.o: $(APIDIR)/olfsysm.hpp

.PHONY: all clean
