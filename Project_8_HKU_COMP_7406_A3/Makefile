#Parameters and Flags
CC	:= g++
CFLAGS := -Wall#-g
SRCDIR := src
BUILDDIR := build
TARGET := main

SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name '*.$(SRCEXT)')
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
LIB := -L ~/CPP_Library/boost_1_67_0/lib		#Specify library Path
LNK := -l boost_regex #Specify xxx suffix libxxx.a
INC := -I ~/CPP_Library/boost_1_67_0/	#Specify include Path

$(TARGET): $(OBJECTS)
	@echo " Linking..."
	@echo " $(CC) -o $(TARGET) $^ $(LIB) $(LNK)"; 
		$(CC) -o $(TARGET) $^ $(LIB) $(LNK)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@echo " Building..."
	@mkdir -p $(BUILDDIR)
	@echo " $(CC) $(CFLAGS) $(INC) -o $@ -c $<"; 
		$(CC) $(CFLAGS) $(INC) -o $@ -c $<

clean:
	@echo " Cleaning...";
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; 
		$(RM) -r $(BUILDDIR) $(TARGET)
