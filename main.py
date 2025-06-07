from model_training import *
from data_preprocessing import *
from model_eval import *

# Trains model using gradient descent with learning rate decay and early stopping
def gradient_descent(X, Y, iterations, initial_lr, decay_rate, patience, X_test, Y_test, resume=False):
    params, start_iteration = initialize_training(resume, iterations)
    alpha = initial_lr
    best_accuracy = 0
    patience_counter = 0

    train_accuracies, val_accuracies, iterations_list = [], [], []

    for i in range(start_iteration, start_iteration + iterations):
        params, cache = perform_training_step(params, X, Y, alpha)
        if i % 50 == 0:
            alpha, accuracy, val_accuracy = evaluate_model(i, params, cache, X, Y, X_test, Y_test, initial_lr, decay_rate, train_accuracies, val_accuracies, iterations_list)
            print(f"Iteration {i}: Train Accuracy {accuracy:.4f}, Validation Accuracy {val_accuracy:.4f}")

            best_accuracy, patience_counter = handle_early_stopping(i, val_accuracy, best_accuracy, patience_counter, patience, params)

    plot_convergence_curve(iterations_list, train_accuracies, val_accuracies)
    return params

# Evals model after training by showing confusion matrix as well as 5 random individual predictions
def test_model(X_test, Y_test, params, val_set_size):
    test_predictions = make_predictions(X_test, params)
    generalized_accuracy = get_accuracy(test_predictions, Y_test)

    print(f"\nGeneralized Accuracy: {generalized_accuracy:.4f}")

    for i in range(5):
        evaluate_single_prediction(np.random.randint(0, val_set_size), X_test, Y_test, params)

    plot_confusion_matrix(Y_test, test_predictions, labels)

# Loads params, trains model, evals model, and optionally saves model params
def main():
    X_train, Y_train, X_test, Y_test, val_set_size = load_and_preprocess_data()
    _, m_train = X_train.shape

    if train_model:
        patience = 50
        if resume_autosave_state:
            params = gradient_descent(X_train, Y_train, NUM_ITERATIONS, INITIAL_LEARNING_RATE, DECAY_RATE, X_test, Y_test, patience, resume=True)
        else:
            params = gradient_descent(X_train, Y_train, NUM_ITERATIONS, INITIAL_LEARNING_RATE, DECAY_RATE, X_test, Y_test, patience)

        test_predictions = make_predictions(X_test, params)
        generalized_accuracy = get_accuracy(test_predictions, Y_test)
        print(f"\nGeneralized Accuracy after training: {generalized_accuracy:.4f}")
        previous_generalized_accuracy = read_generalized_accuracy(generalized_accuracy_filepath)

        if generalized_accuracy > previous_generalized_accuracy + 0.003:
            save_params(params, param_dir)
            save_generalized_accuracy(generalized_accuracy_filepath, generalized_accuracy)
            print(f"New parameters saved with generalized accuracy: {generalized_accuracy:.4f}")
        else:
            print(
                f"Parameters not saved. Improvement not sufficient: {generalized_accuracy - previous_generalized_accuracy:.4f}")
    else:
        params, _ = load_params(param_dir)

    test_model(X_test, Y_test, params, val_set_size)

if __name__ == "__main__":
    main()