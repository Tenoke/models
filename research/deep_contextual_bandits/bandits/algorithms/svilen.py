# import sys, os
# import struct
# import heapq
# import base64
# import random
# import numpy as np

import gpytorch
# from gpytorch.models import AbstractVariationalGP
# from gpytorch.variational import CholeskyVariationalDistribution
# from gpytorch.variational import VariationalStrategy
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model


# import os
# import pickle
# import numpy as np
# import PIL.Image
# import dnnlib
# import dnnlib.tflib as tflib
# import config


import torch
import torch.nn as nn
import random
# import torch.nn.functional as F
# import torch.optim as optim

# from sklearn.model_selection import train_test_split



# Dep net
class Net(nn.Module):

    def __init__(self, input_d):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_d, int(input_d/2))

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return x


# GP model
class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, input_d=0):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            # self.net = Net(input_d)

        def forward(self, x):
            # projected_x = self.net(x)

            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Svilen():
    def __init__(self, name, hparams):
        self.name = name
        self.hparams = hparams
        self.x = []
        self.y = []
        self.model = None

        print(hparams)

    def update(self, context, actions, rewards):
        self.x.append(context)
        self.y.append(actions if rewards > 0 else abs(actions-1))
        self.model = GPRegressionModel(torch.Tensor(self.x), torch.Tensor(self.y), GaussianLikelihood())
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

        print('c', context)
        print(len(context))
        print('a', actions)
        print ('r', rewards)

    def action(self, context):
        print('a-c', context)
        if not self.model:
            return random.randint(0, self.hparams.num_actions-1)

        print(self.model(torch.Tensor(context).unsqueeze(0)).mean)
        print(self.model(torch.Tensor(context).unsqueeze(0)).mean())
        return self.model(torch.Tensor(context).unsqueeze(0)).mean
# dx = torch.Tensor(train_x).cuda()
# dy = torch.Tensor(train_y).cuda()

# # Train the net + GP

# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = GPRegressionModel(dx, dy, likelihood).cuda()

# model.train()
# likelihood.train()

# # Use the adam optimizer
# optimizer = torch.optim.Adam([
#     {'params': model.net.parameters()},
#     {'params': model.covar_module.parameters()},
#     {'params': model.mean_module.parameters()},
#     {'params': model.likelihood.parameters()},
# ], lr=0.01)

# # "Loss" for GPs - the marginal log likelihood
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# for i in range(1000):
#     # Zero backprop gradients
#     optimizer.zero_grad()
#     # Get output from model
#     output = model(dx)
#     # Calc loss and backprop derivatives
#     loss = -mll(output, dy)
#     loss.backward()
#     print('Iter %d/%d - Loss: %.3f' % (i + 1, 100, loss.item()))
#     optimizer.step()

# tflib.init_tf()
# #
# ## Load pre-trained network.

# input_test = torch.randn(1, 512, device = 'cuda').requires_grad_()

# fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
# images = Gs.run(np.array(input_test.detach().cpu().numpy()), None, truncation_psi=0.7, randomize_noise = True, output_transform=fmt)
# PIL.Image.fromarray(images[0], 'RGB').save("yatty/" + "start" + ".png")

# optimizer = optim.Adam([input_test], lr = 0.01)
# criterion = nn.MSELoss()

# model.eval()
# likelihood.eval()

# # Optimize outputs + print images

# with gpytorch.settings.debug(False):
#     frame = 0
#     for f in range(200000):

#         # this part prints an image of intermediate optimziations
#         if f % 100 == 0:
#             fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
#             images = Gs.run(np.array(input_test.detach().cpu().numpy()), None, truncation_psi=0.7, randomize_noise = True, output_transform=fmt)
#             PIL.Image.fromarray(images[0], 'RGB').save("yatty/" + str(frame) + ".png")
#             frame += 1
#         optimizer.zero_grad()
#         output = likelihood(model(input_test))
#         loss = - torch.mean(output.mean + output.variance)
#         loss.backward()
#         optimizer.step()

#         if f % 100 == 0:
#             print(output)
#             print(loss)
#             print(f)
