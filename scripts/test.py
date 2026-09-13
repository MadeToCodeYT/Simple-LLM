import model
import parameters

original = parameters.w_Q[0][0]

model.save_model(
    parameters.w_Q,
    parameters.w_K,
    parameters.w_V,
    parameters.weights_1,
    parameters.biases_1,
    parameters.weights_2,
    parameters.biases_2,
    parameters.output_weights,
    parameters.output_biases,
    parameters.embedding_table,
    parameters.positional_table
)

parameters.w_Q[0][0] = 999999.0

print("Changed:", parameters.w_Q[0][0])

model.load_model()

print("Loaded:", parameters.w_Q[0][0])
print("Original:", original)