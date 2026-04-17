CXX = g++
BASE_CXXFLAGS = -std=c++20 -Wall
CXXFLAGS = $(BASE_CXXFLAGS)
TEST_CXXFLAGS = $(BASE_CXXFLAGS)
INCLUDE_PATH = ./maxlab/include
CLOSEDLOOP_INCLUDE_PATH = ./closedloop/include
LIB_PATH = ./maxlab/lib
BUILD_PATH = ./build
EXAMPLE_PATH = ./examples
CLOSEDLOOP_PATH = ./closedloop
MAXLAB_RUNTIME_LIB ?= $(HOME)/MaxLab/lib

EXAMPLE_INCLUDES = -I$(INCLUDE_PATH)
CLOSEDLOOP_INCLUDES = -I$(INCLUDE_PATH) -I$(CLOSEDLOOP_INCLUDE_PATH)

LDFLAGS = -L$(LIB_PATH) -lmaxlab -pthread
RPATH_FLAGS =

USE_QT ?= 1
ifneq ($(strip $(USE_QT)),0)
QT_QMAKE ?= $(firstword $(wildcard $(CONDA_PREFIX)/bin/qmake6 $(CONDA_PREFIX)/bin/qmake))
ifeq ($(strip $(QT_QMAKE)),)
QT_QMAKE := $(shell command -v qmake6 2>/dev/null || command -v qmake 2>/dev/null)
endif

QT_HEADERS ?=
QT_LIBDIR ?=
ifneq ($(strip $(QT_QMAKE)),)
ifeq ($(strip $(QT_HEADERS)),)
QT_HEADERS := $(shell $(QT_QMAKE) -query QT_INSTALL_HEADERS 2>/dev/null)
endif
ifeq ($(strip $(QT_LIBDIR)),)
QT_LIBDIR := $(shell $(QT_QMAKE) -query QT_INSTALL_LIBS 2>/dev/null)
endif
endif

ifeq ($(strip $(QT_HEADERS)),)
QT_HEADERS := /usr/include/x86_64-linux-gnu/qt6
endif
ifeq ($(strip $(QT_LIBDIR)),)
QT_LIBDIR := /usr/lib/x86_64-linux-gnu
endif

ifeq ($(wildcard $(QT_HEADERS)/QtWidgets/QWidget),)
$(error Qt6 headers not found at $(QT_HEADERS). Activate an environment with qmake6 or set QT_HEADERS=/path/to/include/qt6)
endif
ifeq ($(wildcard $(QT_LIBDIR)/libQt6Widgets.so*),)
$(error Qt6 libraries not found at $(QT_LIBDIR). Activate an environment with qmake6 or set QT_LIBDIR=/path/to/lib)
endif

QT_CFLAGS = -I$(QT_HEADERS) -I$(QT_HEADERS)/QtCore -I$(QT_HEADERS)/QtWidgets -I$(QT_HEADERS)/QtGui -fPIC
QT_LIBS = $(QT_LIBDIR)/libQt6Widgets.so $(QT_LIBDIR)/libQt6Gui.so $(QT_LIBDIR)/libQt6Core.so
CXXFLAGS += -DUSE_QT $(QT_CFLAGS)
LDFLAGS += $(QT_LIBS)
RPATH_FLAGS += -Wl,-rpath,$(QT_LIBDIR)
endif

ifneq ($(wildcard $(MAXLAB_RUNTIME_LIB)/libstdc++.so),)
LDFLAGS += -L$(MAXLAB_RUNTIME_LIB)
RPATH_FLAGS += -Wl,-rpath,$(MAXLAB_RUNTIME_LIB)
endif

LDFLAGS += $(RPATH_FLAGS)

CLOSEDLOOP_SRCS = maxone_with_filter.cpp runtime_config.cpp runtime_timing.cpp runtime_logging.cpp motor_decoder.cpp spike_detection.cpp ponggame.cpp filter.cpp gamewindow.cpp
CLOSEDLOOP_OBJS = $(addprefix $(BUILD_PATH)/,$(CLOSEDLOOP_SRCS:.cpp=.o))

all: example_raw example_filtered maxone_with_filter
.PHONY: all

test_runtime_config:
		mkdir -p $(BUILD_PATH)
		$(CXX) $(TEST_CXXFLAGS) closedloop/tests/runtime_config_test.cpp closedloop/runtime_config.cpp -o $(BUILD_PATH)/runtime_config_test $(CLOSEDLOOP_INCLUDES)
		$(BUILD_PATH)/runtime_config_test
.PHONY: test_runtime_config

test_motor_decoder:
		mkdir -p $(BUILD_PATH)
		$(CXX) $(TEST_CXXFLAGS) closedloop/tests/motor_decoder_test.cpp closedloop/motor_decoder.cpp -o $(BUILD_PATH)/motor_decoder_test $(CLOSEDLOOP_INCLUDES)
		$(BUILD_PATH)/motor_decoder_test
.PHONY: test_motor_decoder

test_ponggame:
		mkdir -p $(BUILD_PATH)
		$(CXX) $(TEST_CXXFLAGS) closedloop/tests/ponggame_test.cpp closedloop/ponggame.cpp -o $(BUILD_PATH)/ponggame_test $(CLOSEDLOOP_INCLUDES)
		$(BUILD_PATH)/ponggame_test
.PHONY: test_ponggame

test_gamewindow:
		mkdir -p $(BUILD_PATH)
		$(CXX) $(CXXFLAGS) closedloop/tests/gamewindow_test.cpp closedloop/gamewindow.cpp closedloop/ponggame.cpp -o $(BUILD_PATH)/gamewindow_test $(CLOSEDLOOP_INCLUDES) $(QT_LIBS) -L$(QT_LIBDIR) -L$(MAXLAB_RUNTIME_LIB) $(RPATH_FLAGS)
		QT_QPA_PLATFORM=offscreen $(BUILD_PATH)/gamewindow_test
.PHONY: test_gamewindow

test_runtime_timing:
		mkdir -p $(BUILD_PATH)
		$(CXX) $(TEST_CXXFLAGS) closedloop/tests/runtime_timing_test.cpp closedloop/runtime_timing.cpp -o $(BUILD_PATH)/runtime_timing_test $(CLOSEDLOOP_INCLUDES)
		$(BUILD_PATH)/runtime_timing_test
.PHONY: test_runtime_timing

test_runtime_logging:
		mkdir -p $(BUILD_PATH)
		$(CXX) $(TEST_CXXFLAGS) closedloop/tests/runtime_logging_test.cpp closedloop/runtime_logging.cpp -o $(BUILD_PATH)/runtime_logging_test $(CLOSEDLOOP_INCLUDES)
		$(BUILD_PATH)/runtime_logging_test
.PHONY: test_runtime_logging

example_raw: $(BUILD_PATH)/process_raw_data.o
		$(CXX) $^ -o $(BUILD_PATH)/$@ $(EXAMPLE_INCLUDES) $(LDFLAGS)

example_filtered: $(BUILD_PATH)/process_spike_events.o
		$(CXX) $^ -o $(BUILD_PATH)/$@ $(EXAMPLE_INCLUDES) $(LDFLAGS)

maxone_with_filter: $(CLOSEDLOOP_OBJS)
		$(CXX) $^ -o $(BUILD_PATH)/$@ $(CLOSEDLOOP_INCLUDES) $(LDFLAGS)

$(BUILD_PATH)/process_raw_data.o: $(EXAMPLE_PATH)/process_raw_data.cpp
		mkdir -p $(BUILD_PATH)
		$(CXX) $(CXXFLAGS) -c $< -o $@ $(EXAMPLE_INCLUDES)

$(BUILD_PATH)/process_spike_events.o: $(EXAMPLE_PATH)/process_spike_events.cpp
		mkdir -p $(BUILD_PATH)
		$(CXX) $(CXXFLAGS) -c $< -o $@ $(EXAMPLE_INCLUDES)

$(BUILD_PATH)/%.o: $(CLOSEDLOOP_PATH)/%.cpp
		mkdir -p $(BUILD_PATH)
		$(CXX) $(CXXFLAGS) -c $< -o $@ $(CLOSEDLOOP_INCLUDES)

clean:
		rm -r $(BUILD_PATH)
