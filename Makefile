HIPCC = hipcc
MPICXX = mpicxx

export OMPI_CXX = $(HIPCC)

MPICXXFLAGS = -O3 -DUSE_HIP -w

SRC_DIR = src
BUILD_DIR = build
LOG_DIR = log

TASK1_SRC = $(SRC_DIR)/task1.cu
TASK2_SRC = $(SRC_DIR)/task2.cu

TASK1_BINARY = $(BUILD_DIR)/task1
TASK2_BINARY = $(BUILD_DIR)/task2

BINARIES = $(TASK1_BINARY) $(TASK2_BINARY)

LOG_FILE = $(LOG_DIR)/build.log

all: $(BINARIES)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(LOG_DIR):
	@mkdir -p $(LOG_DIR)

$(TASK1_BINARY): $(TASK1_SRC) | $(BUILD_DIR)
	$(MPICXX) $(MPICXXFLAGS) $< -o $@

$(TASK2_BINARY): $(TASK2_SRC) | $(BUILD_DIR)
	$(MPICXX) $(MPICXXFLAGS) $< -o $@

clean:
	@echo "Cleaning build and log files..." | tee -a $(LOG_FILE)
	rm -rf $(BUILD_DIR)/* $(LOG_FILE)
	rm -rf $(LOG_DIR)/* $(LOG_FILE)
	rm -rf *.out *.err
	# rm -rf trace/ *.csv *.db *.txt *.json

run: $(BINARIES) $(LOG_DIR)
	@echo "Running task1 with HIP..." | tee -a $(LOG_FILE)
	$(TASK1_BINARY) > $(LOG_DIR)/task1.out 2>&1
	# srun -N 1 -p devel --time=0:01:00 $(TASK1_BINARY) > $(LOG_DIR)/task1.out 2>&1
	@echo "Running task2 with HIP and MPI..." | tee -a $(LOG_FILE)
	# srun -N 2 --ntasks=8 -p devel --time=0:1:00 $(TASK2_BINARY) > $(LOG_DIR)/task2.out 2>&1
	mpirun -np 8 $(TASK2_BINARY) > $(LOG_DIR)/task2.out 2>&1

.PHONY: all clean run
