from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import image_gen_seis_data
from tf_unet import unet
from tf_unet import util

plt.rcParams['image.cmap'] = 'bone'
nx = 1000
ny = 1000

generator = image_gen_seis_data.DigitSeisDataProvider(nx, ny)
x_test, y_test = generator(1)
fig, ax = plt.subplots(1,2, sharey=True, figsize=(8,4))
ax[0].imshow(x_test[0,...,0], aspect="auto")
ax[1].imshow(y_test[0,...,1], aspect="auto")
plt.savefig('Test_data_example.png')
plt.close()


net = unet.Unet(channels=generator.channels, n_class=generator.n_class,
                layers=3, features_root=64)

trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
path = trainer.train(generator, "./unet_trained", training_iters=32, epochs=10,
                        display_step=2)


x_test, y_test = generator(1)
prediction = net.predict("./unet_trained/model.ckpt", x_test)

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
ax[0].imshow(x_test[0,...,0], aspect="auto")
ax[1].imshow(y_test[0,...,1], aspect="auto")
mask = prediction[0,...,1] > 0.4
ax[2].imshow(mask, aspect="auto")
ax[0].set_title("Input")
ax[1].set_title("Ground truth")
ax[2].set_title("Prediction")
fig.tight_layout()
plt.savefig('Test_results.png')
plt.close()

plt.imshow(prediction[0,:,:,1])
plt.colorbar()
plt.savefig('Prediction_results.png')
plt.close()
