CXX = g++
CXXFLAGS = -std=c++20 -Wall
INCLUDE_PATH = ./maxlab/include
LIB_PATH = ./maxlab/lib
BUILD_PATH = ./build
EXAMPLE_PATH = ./examples

LDFLAGS = -L$(LIB_PATH) -lmaxlab -pthread

all: example_raw example_filtered
.PHONY: all

example_raw: $(BUILD_PATH)/process_raw_data.o
		$(CXX) $^ -o $(BUILD_PATH)/$@ -I$(INCLUDE_PATH) $(LDFLAGS)

example_filtered: $(BUILD_PATH)/process_spike_events.o
		$(CXX) $^ -o $(BUILD_PATH)/$@ -I$(INCLUDE_PATH) $(LDFLAGS)

$(BUILD_PATH)/process_raw_data.o: $(EXAMPLE_PATH)/process_raw_data.cpp
		mkdir -p $(BUILD_PATH)
		$(CXX) $(CXXFLAGS) -c $< -o $@ -I$(INCLUDE_PATH)

$(BUILD_PATH)/process_spike_events.o: $(EXAMPLE_PATH)/process_spike_events.cpp
		mkdir -p $(BUILD_PATH)
		$(CXX) $(CXXFLAGS) -c $< -o $@ -I$(INCLUDE_PATH)

clean:
		rm -r $(BUILD_PATH)
