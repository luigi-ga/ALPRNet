import torch
from src.utils.data_utils import label_to_string


# Function to compute the accuracy on the validation set
def compute_accuracy(model, valid_loader, loss_weights, device):
	# Set the model to evaluation mode
	model.eval()

	# Initialize values
	running_loss = 0.0
	correct_predictions = 0
	total_samples = 0

	# Disable gradient computation during validation
	with torch.no_grad():
		# Iterate over the validation set
		for i, (images, labels, labels_len) in enumerate(valid_loader):
			# Move inputs and targets to the appropriate device
			images, labels, labels_len = images.to(device), labels.to(device), labels_len.to(device)
			# Output prediction
			output_dict = model((images, labels, labels_len))	
			# Convert prediction and labels to strings
			pred_list = [label_to_string(pred) for pred in output_dict['output']['pred_rec']]
			targ_list = [label_to_string(lbl) for lbl in labels]

			# Build accuracy list
			acc_list = [(pred == targ) for pred, targ in zip(pred_list, targ_list)]
			# Update total samples and correct predictions
			total_samples += len(acc_list)
			correct_predictions += sum(acc_list)

			# Initialize loss dictionary and total loss
			loss_dict = {}
			loss = 0
			# Iterate over output losses to compute the loss
			for k, losses in output_dict['losses'].items():
				# Compute loss mean
				losses = losses.mean(dim=0, keepdim=True)
				# Update total loss (weighted sum)
				loss += loss_weights[k] * losses
				# Store loss in loss_dict
				loss_dict[k] = losses.item()
			# Update running loss
			running_loss += loss.item()

	# Compute the average loss
	average_loss = running_loss / len(valid_loader)
	# Calculate the validation accuracy
	validation_accuracy = correct_predictions / total_samples

	# Return average loss and validation accuracy
	return average_loss, validation_accuracy


# Function that trains the mode
def train(model, optimizer, es, train_loader, valid_loader, num_epochs, loss_weights, grad_clip, device):
	# Initialize training results
	res = {
		'Train Losses': list(),
		'Valid Losses': list(),
		'Valid Accuracies': list()
	}

	# Iterate over epochs
	for epoch in range(num_epochs):
		# Set model to training mode
		model.train()		
		# Initialize the running loss for this epoch
		running_loss = 0.0		

		# Iterate over the dataloader
		for i, (images, labels, labels_len) in enumerate(train_loader):
			# Move inputs and targets to the appropriate device
			images, labels, labels_len = images.to(device), labels.to(device), labels_len.to(device)
			# Zero the gradients for this batch
			optimizer.zero_grad()
			# Forward pass
			output_dict = model((images, labels, labels_len))
			
			# Initialize loss dictionary and total loss
			loss_dict = {}
			loss = 0
			# Iterate over output losses to compute the loss
			for k, losses in output_dict['losses'].items():
				# Compute loss mean
				losses = losses.mean(dim=0, keepdim=True)
				# Update total loss (weighted sum)
				loss += loss_weights[k] * losses
				# Store loss in loss_dict
				loss_dict[k] = losses.item()

			# Backpropagation
			loss.backward()
			# Gradient clipping (to avoid large gradients)
			if grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
			# Update the model's parameters
			optimizer.step()
			# Update running loss
			running_loss += loss.item()
			
			# Print the training progress for each batch
			if i % 20 == 0:
				print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(train_loader)}], Train Loss: {loss.item():.4f}")

		# Compute the average loss for the epoch
		train_loss = running_loss / len(train_loader)
		# Compute validation loss and accuracy
		valid_loss, valid_accuracy = compute_accuracy(model, valid_loader, loss_weights, device)

		# Print the report
		print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}\n\n")
		
		# Update results
		res['Train Losses'].append(train_loss)
		res['Valid Losses'].append(valid_loss)
		res['Valid Accuracies'].append(valid_accuracy)

		# Check early stopping
		if es(valid_loss): break

	# Return results
	return res



