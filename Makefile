CC = g++
CFLAGS = -g -Wall

LIBBGS = -I'$(CURDIR)/lib/' -L'$(CURDIR)/lib/' -lbgs
OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(LIBBGS) $(OPENCV)

SRCS = main.cpp
PROG = runBGS

$(PROG) : $(SRCS)
    $(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS);

