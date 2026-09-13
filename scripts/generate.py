import model
import transformer

model.load_model()

prompt = "Hello"

result = transformer.generate(
    prompt,
    num_tokens=50
)

print(result)