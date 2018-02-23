#Parameters and Flags
CC	:= g++
CFLAGS := -Wall#-g
SRCDIR := src
BUILDDIR := build
TARGET := main

SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name '*.$(SRCEXT)')
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
LIB := -L lib
LNK := -l lnk
INC := -I include

$(TARGET): $(OBJECTS)
	@echo " Linking..."
	@echo " $(CC) -o $(TARGET) $^ $(LIB)"; 
		$(CC) -o $(TARGET) $^ $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@echo " Building..."
	@mkdir -p $(BUILDDIR)
	@echo " $(CC) $(CFLAGS) $(INC) -o $@ -c $<"; 
		$(CC) $(CFLAGS) $(INC) -o $@ -c $<

clean:
	@echo " Cleaning...";
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; 
		$(RM) -r $(BUILDDIR) $(TARGET)
