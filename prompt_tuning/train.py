from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("klue", "nli")
    prompt_input_dataset = {}
