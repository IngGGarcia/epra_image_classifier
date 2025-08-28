# EPRA Image Classifier - Unified Command Interface
# ================================================

# Default Python interpreter
PYTHON := python3

# Default configuration files
BINARY_CONFIG := configs/binary_config.yaml
MULTICLASS_CONFIG := configs/multiclass_config.yaml

# Default directories
DATA_DIR := data/
OUTPUT_DIR := outputs/
BINARY_OUTPUT_DIR := outputs/binary
MULTICLASS_OUTPUT_DIR := outputs/multiclass
SCRIPTS_DIR := scripts/

# Default model parameters
EPOCHS := 50
BATCH_SIZE := 32
LEARNING_RATE := 0.001

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m  # No Color

.PHONY: help install train-binary train-binary-resume train-multiclass train-multiclass-resume predict predict-multiclass predict-multiclass-batch evaluate clean setup-dirs lint test prepare-multiclass quick-multiclass-pipeline

# Default target
.DEFAULT_GOAL := help

## help: Show this help message
help:
	@echo "$(BLUE)EPRA Image Classifier - Available Commands$(NC)"
	@echo "=========================================="
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  install          Install package and dependencies"
	@echo "  setup-dirs       Create necessary directories"
	@echo ""
	@echo "$(GREEN)Training:$(NC)"
	@echo "  train-binary         Train binary violence classification model"
	@echo "  train-binary-resume  Resume binary training from checkpoint"
	@echo "  train-multiclass     Train multiclass violence classification model"
	@echo "  train-multiclass-resume Resume multiclass training from checkpoint"
	@echo "  train-custom         Train with custom configuration"
	@echo ""
	@echo "$(GREEN)Data Preparation:$(NC)"
	@echo "  prepare-multiclass   Prepare multiclass data from binary model predictions"
	@echo "  quick-multiclass-pipeline Complete pipeline: binary → multiclass data → multiclass training"
	@echo ""
	@echo "$(GREEN)Inference:$(NC)"
	@echo "  predict          Run binary inference on images"
	@echo "  predict-batch    Run binary batch inference"
	@echo "  predict-multiclass      Run multiclass inference on images"
	@echo "  predict-multiclass-batch Run multiclass batch inference"
	@echo ""
	@echo "$(GREEN)Evaluation:$(NC)"
	@echo "  evaluate         Evaluate model on test set"
	@echo "  evaluate-detailed Create detailed evaluation report"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  lint             Run code linting"
	@echo "  test             Run tests"
	@echo "  clean            Clean generated files"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make train-binary                    # Train binary model using data/raw"
	@echo "  make prepare-multiclass              # Convert binary predictions to multiclass data"
	@echo "  make train-multiclass                # Train multiclass model using prepared data"
	@echo "  make quick-multiclass-pipeline       # Complete pipeline after binary training"
	@echo "  make train-binary-resume             # Resume from latest checkpoint"
	@echo "  make predict CHECKPOINT=outputs/binary/best_model.pth INPUT=image.jpg"
	@echo "  make predict-multiclass INPUT=image.jpg          # Uses multiclass model automatically"
	@echo "  make predict-multiclass-batch INPUT=test_images/ # Batch multiclass with reports"
	@echo "  make evaluate CHECKPOINT=outputs/binary/best_model.pth DATA_DIR=data/test_subset"

## install: Install package and dependencies
install:
	@echo "$(BLUE)Installing EPRA Image Classifier...$(NC)"
	pip install -e .
	@echo "$(GREEN)Installation completed!$(NC)"

## setup-dirs: Create necessary directories
setup-dirs:
	@echo "$(BLUE)Creating project directories...$(NC)"
	mkdir -p $(DATA_DIR)
	mkdir -p $(OUTPUT_DIR)/checkpoints $(OUTPUT_DIR)/logs $(OUTPUT_DIR)/results
	mkdir -p logs/
	mkdir -p plots/
	@echo "$(GREEN)Directories created!$(NC)"

## train-binary: Train binary violence classification model
train-binary:
	@echo "$(BLUE)Training binary violence classification model...$(NC)"
	@echo "$(BLUE)Output directory: $(BINARY_OUTPUT_DIR)$(NC)"
	@mkdir -p $(BINARY_OUTPUT_DIR)/checkpoints $(BINARY_OUTPUT_DIR)/logs $(BINARY_OUTPUT_DIR)/results
	$(PYTHON) $(SCRIPTS_DIR)/train.py \
		--config $(BINARY_CONFIG) \
		--data-dir data/raw \
		--model-type binary \
		--output-dir $(BINARY_OUTPUT_DIR) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--learning-rate $(LEARNING_RATE) \
		--experiment-name binary_$(shell date +%Y%m%d_%H%M%S)
	@echo "$(GREEN)Binary training completed! Model saved to $(BINARY_OUTPUT_DIR)$(NC)"

## train-binary-resume: Resume binary training from checkpoint
train-binary-resume:
	@echo "$(BLUE)Resuming binary violence classification training...$(NC)"
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "$(YELLOW)No CHECKPOINT specified, using latest checkpoint...$(NC)"; \
		LATEST_CHECKPOINT=$$(ls -t $(BINARY_OUTPUT_DIR)/checkpoints/checkpoint_epoch_*.pth 2>/dev/null | head -1); \
		if [ -z "$$LATEST_CHECKPOINT" ]; then \
			echo "$(RED)Error: No checkpoints found in $(BINARY_OUTPUT_DIR)/checkpoints/$(NC)"; \
			exit 1; \
		fi; \
		echo "$(BLUE)Using checkpoint: $$LATEST_CHECKPOINT$(NC)"; \
		CHECKPOINT_PATH=$$LATEST_CHECKPOINT; \
	else \
		CHECKPOINT_PATH=$(CHECKPOINT); \
	fi; \
	echo "$(BLUE)Output directory: $(BINARY_OUTPUT_DIR)$(NC)"; \
	mkdir -p $(BINARY_OUTPUT_DIR)/checkpoints $(BINARY_OUTPUT_DIR)/logs $(BINARY_OUTPUT_DIR)/results; \
	$(PYTHON) $(SCRIPTS_DIR)/train.py \
		--config $(BINARY_CONFIG) \
		--data-dir data/raw \
		--model-type binary \
		--output-dir $(BINARY_OUTPUT_DIR) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--learning-rate $(LEARNING_RATE) \
		--experiment-name binary_resume_$(shell date +%Y%m%d_%H%M%S) \
		--resume $$CHECKPOINT_PATH
	@echo "$(GREEN)Binary training resumed and completed! Model saved to $(BINARY_OUTPUT_DIR)$(NC)"

## train-multiclass: Train multiclass violence classification model  
train-multiclass:
	@echo "$(BLUE)Training multiclass violence classification model...$(NC)"
	@echo "$(BLUE)Output directory: $(MULTICLASS_OUTPUT_DIR)$(NC)"
	@mkdir -p $(MULTICLASS_OUTPUT_DIR)/checkpoints $(MULTICLASS_OUTPUT_DIR)/logs $(MULTICLASS_OUTPUT_DIR)/results
	$(PYTHON) $(SCRIPTS_DIR)/train.py \
		--config $(MULTICLASS_CONFIG) \
		--data-dir data/multiclass_data \
		--model-type multiclass \
		--output-dir $(MULTICLASS_OUTPUT_DIR) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--learning-rate $(LEARNING_RATE) \
		--experiment-name multiclass_$(shell date +%Y%m%d_%H%M%S)
	@echo "$(GREEN)Multiclass training completed! Model saved to $(MULTICLASS_OUTPUT_DIR)$(NC)"

## train-multiclass-resume: Resume multiclass training from checkpoint
train-multiclass-resume:
	@echo "$(BLUE)Resuming multiclass violence classification training...$(NC)"
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "$(YELLOW)No CHECKPOINT specified, using latest checkpoint...$(NC)"; \
		LATEST_CHECKPOINT=$$(ls -t $(MULTICLASS_OUTPUT_DIR)/checkpoints/checkpoint_epoch_*.pth 2>/dev/null | head -1); \
		if [ -z "$$LATEST_CHECKPOINT" ]; then \
			echo "$(RED)Error: No checkpoints found in $(MULTICLASS_OUTPUT_DIR)/checkpoints/$(NC)"; \
			exit 1; \
		fi; \
		echo "$(BLUE)Using checkpoint: $$LATEST_CHECKPOINT$(NC)"; \
		CHECKPOINT_PATH=$$LATEST_CHECKPOINT; \
	else \
		CHECKPOINT_PATH=$(CHECKPOINT); \
	fi; \
	echo "$(BLUE)Output directory: $(MULTICLASS_OUTPUT_DIR)$(NC)"; \
	mkdir -p $(MULTICLASS_OUTPUT_DIR)/checkpoints $(MULTICLASS_OUTPUT_DIR)/logs $(MULTICLASS_OUTPUT_DIR)/results; \
	$(PYTHON) $(SCRIPTS_DIR)/train.py \
		--config $(MULTICLASS_CONFIG) \
		--data-dir data/multiclass_data \
		--model-type multiclass \
		--output-dir $(MULTICLASS_OUTPUT_DIR) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--learning-rate $(LEARNING_RATE) \
		--experiment-name multiclass_resume_$(shell date +%Y%m%d_%H%M%S) \
		--resume $$CHECKPOINT_PATH
	@echo "$(GREEN)Multiclass training resumed and completed! Model saved to $(MULTICLASS_OUTPUT_DIR)$(NC)"

## train-custom: Train with custom configuration
train-custom:
	@echo "$(BLUE)Training with custom configuration...$(NC)"
	@if [ -z "$(CONFIG)" ]; then \
		echo "$(RED)Error: CONFIG not specified. Usage: make train-custom CONFIG=path/to/config.yaml DATA_DIR=path/to/data$(NC)"; \
		exit 1; \
	fi
	@if [ -z "$(DATA_DIR)" ]; then \
		echo "$(RED)Error: DATA_DIR not specified. Usage: make train-custom CONFIG=path/to/config.yaml DATA_DIR=path/to/data$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPTS_DIR)/train.py \
		--config $(CONFIG) \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--experiment-name custom_$(shell date +%Y%m%d_%H%M%S) \
		$(EXTRA_ARGS)
	@echo "$(GREEN)Custom training completed!$(NC)"

## predict: Run inference on single image or directory
predict:
	@echo "$(BLUE)Running inference...$(NC)"
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "$(RED)Error: CHECKPOINT not specified. Usage: make predict CHECKPOINT=path/to/model.pth INPUT=path/to/image$(NC)"; \
		exit 1; \
	fi
	@if [ -z "$(INPUT)" ]; then \
		echo "$(RED)Error: INPUT not specified. Usage: make predict CHECKPOINT=path/to/model.pth INPUT=path/to/image$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPTS_DIR)/predict.py \
		--config $(BINARY_CONFIG) \
		--checkpoint $(CHECKPOINT) \
		--input $(INPUT) \
		--output $(BINARY_OUTPUT_DIR)/predictions_$(shell date +%Y%m%d_%H%M%S).json \
		--return-probabilities \
		$(if $(VERBOSE),--verbose) \
		$(EXTRA_ARGS)
	@echo "$(GREEN)Inference completed!$(NC)"

## predict-batch: Run batch inference with detailed analysis
predict-batch:
	@echo "$(BLUE)Running batch inference with analysis...$(NC)"
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "$(RED)Error: CHECKPOINT not specified. Usage: make predict-batch CHECKPOINT=path/to/model.pth INPUT=path/to/images$(NC)"; \
		exit 1; \
	fi
	@if [ -z "$(INPUT)" ]; then \
		echo "$(RED)Error: INPUT not specified. Usage: make predict-batch CHECKPOINT=path/to/model.pth INPUT=path/to/images$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPTS_DIR)/predict.py \
		--config $(BINARY_CONFIG) \
		--checkpoint $(CHECKPOINT) \
		--input $(INPUT) \
		--output $(BINARY_OUTPUT_DIR)/batch_predictions_$(shell date +%Y%m%d_%H%M%S).json \
		--output-dir $(BINARY_OUTPUT_DIR)/batch_analysis_$(shell date +%Y%m%d_%H%M%S) \
		--batch-size $(BATCH_SIZE) \
		--return-probabilities \
		--aggregate-results \
		--create-report \
		$(if $(VERBOSE),--verbose) \
		$(EXTRA_ARGS)
	@echo "$(GREEN)Batch inference completed!$(NC)"

## predict-multiclass: Run multiclass inference on single image or directory
predict-multiclass:
	@echo "$(BLUE)Running multiclass inference...$(NC)"
	@if [ -z "$(INPUT)" ]; then \
		echo "$(RED)Error: INPUT not specified. Usage: make predict-multiclass INPUT=path/to/image$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPTS_DIR)/predict.py \
		--config $(MULTICLASS_CONFIG) \
		--checkpoint $(MULTICLASS_OUTPUT_DIR)/best_model.pth \
		--input $(INPUT) \
		--output $(MULTICLASS_OUTPUT_DIR)/predictions_$(shell date +%Y%m%d_%H%M%S).json \
		--return-probabilities \
		$(if $(VERBOSE),--verbose) \
		$(EXTRA_ARGS)
	@echo "$(GREEN)Multiclass inference completed!$(NC)"

## predict-multiclass-batch: Run batch multiclass inference with detailed analysis
predict-multiclass-batch:
	@echo "$(BLUE)Running batch multiclass inference with analysis...$(NC)"
	@if [ -z "$(INPUT)" ]; then \
		echo "$(RED)Error: INPUT not specified. Usage: make predict-multiclass-batch INPUT=path/to/images$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPTS_DIR)/predict.py \
		--config $(MULTICLASS_CONFIG) \
		--checkpoint $(MULTICLASS_OUTPUT_DIR)/best_model.pth \
		--input $(INPUT) \
		--output $(MULTICLASS_OUTPUT_DIR)/batch_predictions_$(shell date +%Y%m%d_%H%M%S).json \
		--output-dir $(MULTICLASS_OUTPUT_DIR)/batch_analysis_$(shell date +%Y%m%d_%H%M%S) \
		--batch-size $(BATCH_SIZE) \
		--return-probabilities \
		--aggregate-results \
		--create-report \
		$(if $(VERBOSE),--verbose) \
		$(EXTRA_ARGS)
	@echo "$(GREEN)Multiclass batch inference completed!$(NC)"

## evaluate: Evaluate model on test set
evaluate:
	@echo "$(BLUE)Evaluating model...$(NC)"
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "$(RED)Error: CHECKPOINT not specified. Usage: make evaluate CHECKPOINT=path/to/model.pth DATA_DIR=path/to/test$(NC)"; \
		exit 1; \
	fi
	@if [ -z "$(DATA_DIR)" ]; then \
		echo "$(RED)Error: DATA_DIR not specified. Usage: make evaluate CHECKPOINT=path/to/model.pth DATA_DIR=path/to/test$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPTS_DIR)/evaluate.py \
		--config $(BINARY_CONFIG) \
		--checkpoint $(CHECKPOINT) \
		--data-dir $(DATA_DIR) \
		--output-dir $(BINARY_OUTPUT_DIR)/evaluation_$(shell date +%Y%m%d_%H%M%S) \
		--batch-size $(BATCH_SIZE) \
		$(if $(VERBOSE),--verbose) \
		$(EXTRA_ARGS)
	@echo "$(GREEN)Evaluation completed!$(NC)"

## evaluate-detailed: Create detailed evaluation report with visualizations
evaluate-detailed:
	@echo "$(BLUE)Creating detailed evaluation report...$(NC)"
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "$(RED)Error: CHECKPOINT not specified. Usage: make evaluate-detailed CHECKPOINT=path/to/model.pth DATA_DIR=path/to/test$(NC)"; \
		exit 1; \
	fi
	@if [ -z "$(DATA_DIR)" ]; then \
		echo "$(RED)Error: DATA_DIR not specified. Usage: make evaluate-detailed CHECKPOINT=path/to/model.pth DATA_DIR=path/to/test$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPTS_DIR)/evaluate.py \
		--config $(BINARY_CONFIG) \
		--checkpoint $(CHECKPOINT) \
		--data-dir $(DATA_DIR) \
		--output-dir $(BINARY_OUTPUT_DIR)/detailed_evaluation_$(shell date +%Y%m%d_%H%M%S) \
		--batch-size $(BATCH_SIZE) \
		--save-predictions \
		--create-visualizations \
		--verbose \
		$(EXTRA_ARGS)
	@echo "$(GREEN)Detailed evaluation completed!$(NC)"

## lint: Run code linting
lint:
	@echo "$(BLUE)Running code linting...$(NC)"
	@if command -v ruff >/dev/null 2>&1; then \
		echo "Running ruff..."; \
		ruff check src/ scripts/ --fix; \
		ruff format src/ scripts/; \
	else \
		echo "$(YELLOW)Ruff not found, trying flake8...$(NC)"; \
		if command -v flake8 >/dev/null 2>&1; then \
			flake8 src/ scripts/; \
		else \
			echo "$(YELLOW)No linter found. Install ruff: pip install ruff$(NC)"; \
		fi \
	fi
	@echo "$(GREEN)Linting completed!$(NC)"

## test: Run tests
test:
	@echo "$(BLUE)Running tests...$(NC)"
	@if [ -d "tests" ]; then \
		$(PYTHON) -m pytest tests/ -v $(if $(VERBOSE),--verbose) $(EXTRA_ARGS); \
	else \
		echo "$(YELLOW)No tests directory found$(NC)"; \
	fi
	@echo "$(GREEN)Tests completed!$(NC)"

## clean: Clean generated files
clean:
	@echo "$(BLUE)Cleaning generated files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true
	@echo "$(GREEN)Cleanup completed!$(NC)"

# Quick start targets
## quick-binary: Quick binary training with defaults
quick-binary: setup-dirs
	@echo "$(BLUE)Quick binary training setup...$(NC)"
	@if [ ! -f "$(BINARY_CONFIG)" ]; then \
		echo "$(YELLOW)Warning: Binary config not found at $(BINARY_CONFIG)$(NC)"; \
	fi
	@mkdir -p $(BINARY_OUTPUT_DIR)/checkpoints $(BINARY_OUTPUT_DIR)/logs $(BINARY_OUTPUT_DIR)/results
	@echo "$(GREEN)Ready! Run: make train-binary$(NC)"

## quick-multiclass: Quick multiclass training with defaults  
quick-multiclass: setup-dirs
	@echo "$(BLUE)Quick multiclass training setup...$(NC)"
	@if [ ! -f "$(MULTICLASS_CONFIG)" ]; then \
		echo "$(YELLOW)Warning: Multiclass config not found at $(MULTICLASS_CONFIG)$(NC)"; \
	fi
	@mkdir -p $(MULTICLASS_OUTPUT_DIR)/checkpoints $(MULTICLASS_OUTPUT_DIR)/logs $(MULTICLASS_OUTPUT_DIR)/results
	@echo "$(GREEN)Ready! Run: make train-multiclass$(NC)"

# Development targets
.PHONY: dev-install dev-setup
dev-install:
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	pip install -e ".[dev]"

dev-setup: dev-install setup-dirs
	@echo "$(BLUE)Development environment setup completed!$(NC)"

# Docker targets (if Dockerfile exists)
.PHONY: docker-build docker-run
docker-build:
	@if [ -f "Dockerfile" ]; then \
		echo "$(BLUE)Building Docker image...$(NC)"; \
		docker build -t epra-classifier .; \
	else \
		echo "$(YELLOW)No Dockerfile found$(NC)"; \
	fi

docker-run:
	@if command -v docker >/dev/null 2>&1; then \
		echo "$(BLUE)Running Docker container...$(NC)"; \
		docker run -it --rm epra-classifier; \
	else \
		echo "$(YELLOW)Docker not found$(NC)"; \
	fi

# Data preparation targets
## prepare-multiclass: Prepare multiclass data from binary model predictions
prepare-multiclass:
	@echo "$(BLUE)Preparing multiclass data from binary model...$(NC)"
	@if [ ! -f "outputs/binary/best_model.pth" ]; then \
		echo "$(RED)Error: Binary model not found. Run 'make train-binary' first.$(NC)"; \
		exit 1; \
	fi
	@if [ ! -d "data/raw" ]; then \
		echo "$(RED)Error: Source data directory 'data/raw' not found.$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) scripts/prepare_multiclass_data.py \
		--binary-model outputs/binary/best_model.pth \
		--source-dir data/raw \
		--target-dir data/multiclass_data \
		--config configs/binary_config.yaml
	@echo "$(GREEN)Multiclass data preparation completed! Ready for multiclass training.$(NC)"

## quick-multiclass-pipeline: Complete pipeline from binary model to multiclass training
quick-multiclass-pipeline: prepare-multiclass train-multiclass
	@echo "$(GREEN)Complete multiclass pipeline finished!$(NC)"
	@echo "$(GREEN)Binary model trained → Multiclass data prepared → Multiclass model trained$(NC)"
