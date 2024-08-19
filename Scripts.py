import sys

def generate_response(prompt):
    return f"AI Response to: {prompt}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
        response = generate_response(prompt)
        print(response)
    else:
        print("No prompt provided.")
