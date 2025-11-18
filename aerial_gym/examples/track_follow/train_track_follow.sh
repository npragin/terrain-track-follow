#!/usr/bin/env bash

# Training script for track_follow_task using rl_games (PPO)
# This script trains a policy to track and follow a moving target in the procedural forest environment

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (3 levels up from examples/track_follow)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Default arguments
TASK="track_follow_task"
CONFIG_FILE="ppo_track_follow.yaml"
NUM_ENVS=16
HEADLESS="True"
EXPERIMENT_NAME="track_follow_training"
SEED=-1
TRACK_WANDB=true
WANDB_PROJECT="track_follow"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --headless)
            HEADLESS="$2"
            shift 2
            ;;
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --track)
            TRACK_WANDB=true
            shift 1
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            TRACK_WANDB=true  # Enable tracking if project is specified
            shift 2
            ;;
        --play)
            PLAY_MODE=true
            shift 1
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num_envs NUM          Number of parallel environments (default: 16, max: 16 for track_follow_task)"
            echo "  --headless BOOL         Run without display (default: True)"
            echo "  --experiment_name NAME  Name for this experiment (default: track_follow_training)"
            echo "                          For --play mode, this is used to find checkpoints in runs/<experiment_name>/"
            echo "  --seed SEED             Random seed (default: -1 for random)"
            echo "  --config FILE           Config file name (default: ppo_track_follow.yaml)"
            echo "  --track                 Enable Weights & Biases tracking"
            echo "  --wandb-project NAME    WandB project name (default: track_follow)"
            echo "  --play                  Test/play mode (uses experiment_name to find checkpoint)"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Train the policy"
            echo "  $0"
            echo ""
            echo "  # Train with custom experiment name"
            echo "  $0 --experiment_name my_experiment"
            echo ""
            echo "  # Test a trained policy (finds best checkpoint in runs/my_experiment/)"
            echo "  $0 --play --experiment_name my_experiment"
            echo ""
            echo "  # Train with WandB tracking"
            echo "  $0 --track --wandb-project my_project"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Change to the rl_games directory
cd "$PROJECT_ROOT/aerial_gym/rl_training/rl_games"

# Build the command
CMD="python runner.py --file=$CONFIG_FILE --task=$TASK --num_envs=$NUM_ENVS --headless=$HEADLESS"

if [ "$SEED" != "-1" ]; then
    CMD="$CMD --seed=$SEED"
fi

if [ -n "$EXPERIMENT_NAME" ]; then
    CMD="$CMD --experiment_name=$EXPERIMENT_NAME"
fi

if [ "$TRACK_WANDB" = true ]; then
    CMD="$CMD --track --wandb-project-name=$WANDB_PROJECT"
fi

if [ "$PLAY_MODE" = true ]; then
    # Find checkpoint in runs/<experiment_name>/ directory
    RUNS_DIR="runs/$EXPERIMENT_NAME"
    
    if [ ! -d "$RUNS_DIR" ]; then
        echo "Error: Experiment directory not found: $RUNS_DIR"
        echo "Available experiments in runs/:"
        ls -1 runs/ 2>/dev/null || echo "  (runs/ directory is empty or doesn't exist)"
        exit 1
    fi
    
    # Look for best checkpoint first, then latest checkpoint
    CHECKPOINT=""
    if [ -f "$RUNS_DIR/best.pth" ]; then
        CHECKPOINT="$RUNS_DIR/best.pth"
    elif [ -f "$RUNS_DIR/model_best.pth" ]; then
        CHECKPOINT="$RUNS_DIR/model_best.pth"
    else
        # Find the latest checkpoint file (portable method)
        LATEST=$(ls -t "$RUNS_DIR"/*.pth 2>/dev/null | head -1)
        if [ -n "$LATEST" ] && [ -f "$LATEST" ]; then
            CHECKPOINT="$LATEST"
        fi
    fi
    
    if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
        echo "Error: No checkpoint found in $RUNS_DIR"
        echo "Available files:"
        ls -lh "$RUNS_DIR" 2>/dev/null || echo "  (directory is empty)"
        exit 1
    fi
    
    CMD="$CMD --play --checkpoint=$CHECKPOINT"
    echo "=========================================="
    echo "Testing trained policy"
    echo "Experiment: $EXPERIMENT_NAME"
    echo "Checkpoint: $CHECKPOINT"
    echo "=========================================="
else
    CMD="$CMD --train"
    echo "=========================================="
    echo "Training track_follow_task"
    echo "Experiment: $EXPERIMENT_NAME"
    echo "Environments: $NUM_ENVS"
    echo "Headless: $HEADLESS"
    echo "Config: $CONFIG_FILE"
    if [ "$TRACK_WANDB" = true ]; then
        echo "WandB Project: $WANDB_PROJECT"
    fi
    echo "=========================================="
fi

# Print the command and execute
echo "Running: $CMD"
echo ""

eval $CMD

