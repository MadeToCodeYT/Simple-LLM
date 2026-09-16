import model
import transformer

model.load_model()

while True:
    prompt = input("User: ")

    if prompt.lower() == "exit":
        break

    result = transformer.generate(
        f"User: {prompt}\nLarge Language Model:""",
        num_tokens=50
    )

    print("Large Language Model: ", end="")
    print(result + "\n")