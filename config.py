# Configurations (mostly optimized through empirical testing)
INPUT_SIZE = 784
HIDDEN_LAYER_1_SIZE = 312
HIDDEN_LAYER_2_SIZE = 55
OUTPUT_CLASSES = 10
IMAGE_SHAPE = (28, 28)

NUM_ITERATIONS = 50000
INITIAL_LEARNING_RATE = 0.1
DROPOUT_RATE = 0.25
DECAY_RATE = 0.0001
ENABLE_DROPOUT = False

# Labels and paths
# This program has been modified to also work with mnist Fashion dataset,
# so these following configs are in case I wish to switch between datasets and trained models
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
param_dir = "model_params"
dataset_path = "data/train.csv"
generalized_accuracy_filepath = f"{param_dir}/generalized_accuracy.csv"


# Training options
train_model = False
resume_autosave_state = False  # Set to True to resume from a saved state