from src.train import train_model
from src.evaluate import evaluate_model

def main():
    print("Training the model...")
    train_model()
    
    print("Evaluating the model...")
    evaluate_model()

if __name__ == "__main__":
    main()
