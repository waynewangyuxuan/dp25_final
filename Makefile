# Makefile for dp25_final project

.PHONY: task1 task2 task3 task4 task5 clean setup

# Setup virtual environment
setup:
	chmod +x setup.sh
	./setup.sh

# Activate virtual environment
# Usage: source venv/bin/activate

# Task 1 - Baseline evaluation
task1:
	python experiments/task1_baseline.py

# Task 2 - FGSM attack
task2:
	python experiments/task2_fgsm.py

# Task 3 - Improved attacks
task3:
	python experiments/task3_pgd_full.py

# Task 4 - Patch attacks
task4:
	python experiments/task4_pgd_patch.py

# Task 5 - Transferability
task5:
	python experiments/task5_transfer.py

# Run all tasks
all: task1 task2 task3 task4 task5

# Clean generated data
clean:
	rm -rf data/adv_test_set_*
	rm -rf logs/*.json logs/*.csv
	rm -rf figures/*.png 