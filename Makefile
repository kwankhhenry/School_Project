#Parameters and Flags
CC	:= g++
CFLAGS := -Wall -fPIC#-std=c++#-g
LFLAGS := -shared
TARGET := main

# Directory Paths
SRCDIR := src
BUILDDIR := build
LIBDIR := ../lib

SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name '*.$(SRCEXT)')
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))

# Include and Linker options
INCLUDE := -I include	# Specify custom include paths
LIBRARY := -L lib		# Specify custom library paths
LINKING := -l lnk		# Specify custom libraries to include (.so (dynamic) or .a (static)).  E.g. -l mine = libmine.so

# all: library.cpp main.cpp
# $@ evaluates all output objects "all"
# $< evaluates the first upcoming file "library.cpp"
# $^ evaluates all cpp files "library.cpp main.cpp"

$(TARGET): $(OBJECTS)
	@echo " Linking..."
	@echo " $(CC) -o $(TARGET) $^ $(LIBRARY) $(LINKING)"; 
		$(CC) -o $(TARGET) $^ $(LIBRARY) $(LINKING)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@echo " Building..."
	@mkdir -p $(BUILDDIR)
	@echo " $(CC) $(CFLAGS) -o $@ -c $< $(INCLUDE)"; 
		$(CC) $(CFLAGS) -o $@ -c $< $(INCLUDE)

lib: $(OBJECTS)
	@echo " Creating shared library..."
	@mkdir -p $(LIBDIR)
	@echo " $(CC) $(LFLAGS) -o $(LIBDIR)/lib$(TARGET).so $^";
		$(CC) $(LFLAGS) -o $(LIBDIR)/lib$(TARGET).so $^

clean:
	@echo " Cleaning...";
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; 
		$(RM) -r $(BUILDDIR) $(TARGET)
	@echo " $(RM) -r $(LIBDIR)/lib$(TARGET).so"; 
		$(RM) -r $(LIBDIR)/lib$(TARGET).so
