CXX = g++
CXXFLAGS = -std=c++20 -Wall
INCLUDE_PATH = ./maxlab/include
CLOSEDLOOP_INCLUDE_PATH = ./closedloop/include
LIB_PATH = ./maxlab/lib
BUILD_PATH = ./build
EXAMPLE_PATH = ./examples
CLOSEDLOOP_PATH = ./closedloop

EXAMPLE_INCLUDES = -I$(INCLUDE_PATH)
CLOSEDLOOP_INCLUDES = -I$(INCLUDE_PATH) -I$(CLOSEDLOOP_INCLUDE_PATH)

LDFLAGS = -L$(LIB_PATH) -lmaxlab -pthread

CLOSEDLOOP_SRCS = maxone_with_filter.cpp spike_detection.cpp ponggame.cpp filter.cpp gamewindow.cpp
CLOSEDLOOP_OBJS = $(addprefix $(BUILD_PATH)/,$(CLOSEDLOOP_SRCS:.cpp=.o))

all: example_raw example_filtered maxone_with_filter
.PHONY: all

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
