def estimate_total_parameters(input_size, output_size, hidden_layers):
    """
    Estimate total trainable parameters in a feedforward MLP.

    Parameters:
    - input_size (int): Number of input features
    - output_size (int): Number of output neurons (e.g., 1 for regression)
    - hidden_layers (list of int): Neurons in each hidden layer

    Returns:
    - total_params (int): Total trainable parameters
    """
    total_params = 0
    prev_size = input_size

    for layer_size in hidden_layers:
        weights = prev_size * layer_size
        biases = layer_size
        total_params += weights + biases
        prev_size = layer_size

    total_params += (prev_size * output_size) + output_size

    return total_params


# Example usage:
input_features = 9
output_targets = 1
hidden_config = [400, 200, 100, 50]

total = estimate_total_parameters(input_features, output_targets, hidden_config)
print(f"Total parameters: {total:,}")

# Check if under budget
budget = 100_000
if total <= budget:
    print("✅ Fits")
