import torch


def get_accuracy(outputs, labels):
    if outputs.dim() == 2 and outputs.shape[-1] > 1:
        return get_multiclass_accuracy(outputs, labels)
    else:
        preds = torch.sign(outputs).view(-1)
        correct = (preds == labels).sum()
        acc = correct
    return acc.item()


def get_multiclass_accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=-1)
    correct = (preds == labels).sum()
    acc = correct
    return acc


def train_epoch(model, dloader, loss_fn, optimizer, device, classify=True, label_index=0):
    running_loss = 0.0
    if classify:
        running_acc = 0.0
    for i, data in enumerate(dloader):
        if len(data.y.shape) > 1:
            labels = data.y[:, label_index].view(-1, 1)
            labels = labels.float()
        else:
            labels = data.y
        if loss_fn.__class__.__name__ == 'CrossEntropyLoss' and -1 in labels:
            labels = (labels + 1) / 2
            labels = labels.long()
        inputs = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if classify:
            running_acc += get_accuracy(outputs, labels)

    if classify:
        return running_loss / len(dloader), running_acc / len(dloader.dataset)
    else:
        return running_loss / len(dloader), -1


def test_epoch(model, dloader, loss_fn, device, classify=True, label_index=0):
    with torch.no_grad():
        running_loss = 0.0

        if classify:
            running_acc = 0.0
        model.eval()
        for i, data in enumerate(dloader):
            if len(data.y.shape) > 1:
                labels = data.y[:, label_index].view(-1, 1)
                labels = labels.float()
            else:
                labels = data.y
            if loss_fn.__class__.__name__ == 'CrossEntropyLoss' and -1 in labels:
                labels = (labels + 1) / 2
                labels = labels.long()
            inputs = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model.forward(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            if classify:
                running_acc += get_accuracy(outputs, labels)

        if classify:
            return running_loss / len(dloader), running_acc / len(dloader.dataset)
        else:
            return running_loss / len(dloader), -1


def ExpLoss(preds, labels):
    labels = labels.view(preds.shape)
    loss = (- labels * preds).exp()
    loss = loss.mean()
    return loss
