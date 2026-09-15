import model
import transformer

model.load_model()

while True:
    prompt = input("User: ")

    if prompt.lower() == "exit":
        break

    result = transformer.generate(
        prompt,
        num_tokens=50
    )

    print("Large Language Model:")
    print(result + "\n")