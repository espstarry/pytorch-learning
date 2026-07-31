import torch


# Goal: learn y = 2x + 1 from a few examples.
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = torch.tensor([3.0, 5.0, 7.0, 9.0])

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)

learning_rate = 0.01
optimizer = torch.optim.SGD([w, b], learning_rate)

for step in range(2000):
    # TODO 2: compute prediction for y = w*x + b
    y_pred = w * x + b

    # TODO 3: mean squared error.
    # Hint: subtract y_true, square, then take mean.
    loss = ((y_pred - y_true) ** 2).mean()

    # TODO 4: backward pass.
    # Surface action: call something on loss.
    # Real side effect: w.grad and b.grad are filled/accumulated.
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 10 == 0:
        print(
            f"step={step:03d}",
            f"w={w.item():.4f}",
            f"b={b.item():.4f}",
            f"loss={loss.item():.6f}",
        )

print("\nFinal:")
print("w should be close to 2.0:", w.item())
print("b should be close to 1.0:", b.item())
