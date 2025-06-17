import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_config(file_path='microsoft_inference.json'):
    """Load and validate configuration from JSON file"""
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
        
        # Validate required keys
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
        
        # Set manual seed for reproducibility
        torch.random.manual_seed(0)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            device_map=config["device_map"],
            torch_dtype=getattr(torch, config["torch_dtype"]) if config["torch_dtype"] != "auto" else "auto",
            trust_remote_code=config["trust_remote_code"]
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

        # Setup pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        # Generate response
        output = pipe(config["messages"], **config["generation_args"])
        print(output[0]['generated_text'])
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        if isinstance(e, KeyError):
            print("Please check your config file has all required keys.")
        elif isinstance(e, FileNotFoundError):
            print("Please ensure the config file exists at the specified path.")
        elif isinstance(e, json.JSONDecodeError):
            print("Please check your config file has valid JSON format.")

if __name__ == "__main__":
    main()