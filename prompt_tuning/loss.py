def get_loss(inputs, prompt_model, criterion, device):
    inputs = inputs.to(device)
    logits = prompt_model(inputs)
    labels = inputs.label
    loss = criterion(logits, labels)
    return loss
