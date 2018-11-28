from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import image_gen_seis_data
from tf_unet import unet
from tf_unet import util
import time
from scipy.ndimage import gaussian_filter

plt.rcParams['image.cmap'] = 'bone'
nx = 2000
ny = 1000

L  = 5
FR = 32
TI = 50
E  = 20
opt="adam"

generator = image_gen_seis_data.DigitSeisDataProvider(nx, ny, numFiles=TI, mostCrossings=True)


x_test, y_test = generator(1)
fig, ax = plt.subplots(1,2, sharey=True, figsize=(8,4))
ax[0].imshow(x_test[0,...,0], aspect="auto")
ax[1].imshow(y_test[0,...,1], aspect="auto")
plt.savefig('Test_data_example.png')
plt.close()


OPT="adam"
start_time = time.time()
print("Image size: {}x{}".format(nx,ny))
net = unet.Unet(channels=generator.channels, n_class=generator.n_class,
                layers=L, features_root=FR, cost_kwargs=dict(regularizer=0.001))

# cost_kwargs=dict(regularizer=0.001)
# , opt_kwargs=dict(momentum=0.2)
trainer = unet.Trainer(net, optimizer=OPT, opt_kwargs=dict(beta1=0.91))
path = trainer.train(generator, "./unet_trained", training_iters=TI, epochs=E,
                        display_step=2, dropout=0.5)
# dropout=0.5
end_time = time.time()
print('Training took {} minutes'.format((end_time - start_time)/60))

x_test, y_test = generator(1)
prediction = net.predict("./unet_trained/model.ckpt", x_test)
prediction2 = net.predict("./unet_trained/model.ckpt", gaussian_filter(x_test, 3))

# Z = np.zeros([ny, nx])
# Z[ prediction[0,:,:,1]>0.7] = 1
# fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12,5))
# ax[0].imshow(x_test[0,...,0], aspect="auto")
# ax[1].imshow(y_test[0,...,1], aspect="auto")
# ax[2].imshow(prediction[0,:,:,1], aspect="auto")
# ax[0].set_title("Input")
# ax[1].set_title("Ground truth")
# ax[2].set_title("Prediction")
# fig.tight_layout()
# plt.savefig('Test_results_L{}_FR{}_TI{}_E{}_{}.png'.format(L,FR,TI,E,OPT))
# plt.close()

# fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12,5))
# ax[0].imshow(image[...,0], aspect="auto")
# ax[1].imshow(gaussian_filter(image[...,0], 1), aspect="auto")
#
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
ax[0].imshow(x_test[0,...,0], aspect="auto")
ax[1].imshow(prediction[0,:,:,1], aspect="auto")
im = ax[2].imshow(prediction2[0,:,:,1], aspect="auto")
plt.colorbar(im)
plt.savefig('Prediction_results.png')
# plt.imshow(prediction[0,:,:,1])
# plt.colorbar()
# plt.savefig('Prediction_results.png')
# plt.close()
