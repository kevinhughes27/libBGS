#-------------------------------------------------
#
# Project created by QtCreator 2012-05-23T11:44:55
#
#-------------------------------------------------

QT       -= core gui

TARGET = bgs
TEMPLATE = lib
#CONFIG += staticlib

LIBS +=`pkg-config opencv --cflags --libs`

SOURCES += \
    WrenGA.cpp \
    PoppeGMM.cpp \
    GrimsonGMM.cpp \
    Eigenbackground.cpp \
    AdaptiveMedian.cpp \
    Mean.cpp \
    PratiMediod.cpp \
    ZivkovicGMM.cpp \
    SimpleFrameDifferencing.cpp

HEADERS += \
    WrenGA.hpp \
    PoppeGMM.hpp \
    GrimsonGMM.hpp \
    Eigenbackground.hpp \
    BgsParams.hpp \
    PratiMediod.hpp \
    Mean.hpp \
    AdaptiveMedian.hpp \
    Bgs.hpp \
    ZivkovicGMM.hpp \
    libBGS.h \
    SimpleFrameDifferencing.hpp

unix:!symbian {
    maemo5 {
        target.path = /opt/usr/lib
    } else {
        target.path = /usr/lib
    }
    INSTALLS += target
}

OTHER_FILES += \
    bgs.pro.user
