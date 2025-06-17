import torch
import json
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_config(file_path='microsoft_inference_phi3.json'):
    """Load and validate configuration from JSON file"""
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
        
        required_keys = ['model_name', 'device_map', 'torch_dtype',
                         'trust_remote_code', 'messages', 'generation_args']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required key in config: {key}")
        
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {file_path}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in config file")

def main():
    try:
        # Load configuration
        config = load_config()
        torch.manual_seed(0)

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            device_map=config["device_map"],
            torch_dtype=getattr(torch, config["torch_dtype"]) if config["torch_dtype"] != "auto" else "auto",
            trust_remote_code=config["trust_remote_code"],
            attn_implementation="eager"
        )

        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

        # Set pad token if not already defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize the prompt
        inputs = tokenizer(config["messages"], return_tensors="pt").to(model.device)

        # Inject pad_token_id into generation_args to prevent issues
        generation_args = config["generation_args"]
        generation_args["pad_token_id"] = tokenizer.pad_token_id

        # Generate output
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                use_cache=False,
                **generation_args
            )

        # Decode and print the result
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("\n=== Generated Output ===\n")
        print(output_text)

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        if isinstance(e, KeyError):
            print("Please check your config file has all required keys.")
        elif isinstance(e, FileNotFoundError):
            print("Please ensure the config file exists at the specified path.")
        elif isinstance(e, json.JSONDecodeError):
            print("Please check your config file has valid JSON format.")

if __name__ == "__main__":
    main()
