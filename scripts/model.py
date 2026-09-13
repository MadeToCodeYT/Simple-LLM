import json
import parameters

def save_model(
    w_Q,
    w_K,
    w_V,
    weights_1,
    biases_1,
    weights_2,
    biases_2,
    output_weights,
    output_biases,
    embedding_table,
    positional_table
) -> None:
    data = {
        "w_Q": w_Q,
        "w_K": w_K,
        "w_V": w_V,
        "weights_1": weights_1,
        "biases_1": biases_1,
        "weights_2": weights_2,
        "biases_2": biases_2,
        "output_weights": output_weights,
        "output_biases": output_biases,
        "embedding_table": embedding_table,
        "positional_table": positional_table,
    }

    with open("model/model.json", "w") as file:
        json.dump(data, file, indent=4)

def load_model() -> None:
    with open("model/model.json", "r") as file:
        data = json.load(file)

    parameters.w_Q = data["w_Q"]
    parameters.w_K = data["w_K"]
    parameters.w_V = data["w_V"]
    parameters.weights_1 = data["weights_1"]
    parameters.biases_1 = data["biases_1"]
    parameters.weights_2 = data["weights_2"]
    parameters.biases_2 = data["biases_2"]
    parameters.output_weights = data["output_weights"]
    parameters.output_biases = data["output_biases"]
    parameters.embedding_table = data["embedding_table"]
    parameters.positional_table = data["positional_table"]