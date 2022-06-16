import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader
from simple_unet import UNet


model = UNet(128, 128, sigmoid_act=True)
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

f = h5py.File("train_features.hdf5", "r")
x = torch.Tensor(f['feature_x'][:1000, :, :500])
y = torch.Tensor(f['feature_y'][:1000, :, :500])
dset = TensorDataset(x, y)

loader = DataLoader(dataset=dset, batch_size=16, shuffle=True, drop_last=True)

for epoch in range(10):
    for step, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        optimizer.zero_grad()
        out = model(batch_x)
        # shape_b, shape_c, shape_l = out.shape
        # out = out.reshape(shape_b, -1).softmax(-1).reshape(shape_b, shape_c, -1)
        # loss = loss_fn(out, batch_y)
        loss = loss_fn(out * batch_x, batch_y)
        if step % 10 == 0:
            baseline = loss_fn(batch_x, batch_y)
            print(epoch, step, loss.item(), baseline.item())
        loss.backward()
        optimizer.step()

torch.save(model, 'net_track.pth')
