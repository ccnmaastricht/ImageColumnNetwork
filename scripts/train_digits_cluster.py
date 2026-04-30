from column_ode_digits import train_digit_classification
import torch

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_digit_classification(
        digits_to_include=[0,1],
        device=device,
        batch_size=16,
        nr_epochs=50,
        lr=1e-1,
        lambda_suppression=1e-1,
        lambda_magnitude=1e-2,
        lambda_ei=5e-2
    )