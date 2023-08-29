import torch
from src.utils.data_utils import label_to_string

def predict_code(model, image, max_len, eos, transform, device):
    # Set to evaluation mode
    model.eval()
    # Apply transformations (if needed)
    if transform: image = transform(image)    
    # Initialize a tensor full of 'EOS' id values
    rec_targets = torch.full((1, max_len), eos, dtype=torch.int32)
    # Move to device
    image = image.unsqueeze(0).to(device)
    rec_targets = rec_targets.to(device)
    max_len = torch.tensor([max_len]).to(device)
    # Predict label
    pred_rec = model((image, rec_targets, max_len))['output']['pred_rec'][0]
    # Convert predicted label to string
    predicted_string = label_to_string(pred_rec)
    # Return prediction
    return predicted_string