import glob

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.optimizers import Adam
from networks.siamese_network.SiameseModel import SiameseModel
from networks.siamese_network.SiameseLoader import SiameseLoader
import os
from time import time

initial_epoch = 0
ckpt_period = 10
n_epochs_to_train = 1000
ckpts = glob.glob("checkpoints/*.hdf5")

if len(ckpts) != 0:
    latest_ckpt = max(ckpts, key=os.path.getctime)
    print("loading from checkpoint: ", latest_ckpt)
    initial_epoch = int(latest_ckpt[latest_ckpt.find("-epoch-") + len("-epoch-"):latest_ckpt.rfind("-lr-")])
    model = load_model(latest_ckpt)
else:
    model = SiameseModel().build()

optimizer = Adam(0.000006)
model.compile(loss='binary_crossentropy', optimizer=optimizer)

os.makedirs("checkpoints", exist_ok=True)
file_path = "checkpoints/flowchroma-epoch-{epoch:05d}-lr-" + "-train_loss-{loss:.4f}-val_loss-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(file_path,
                             monitor=['loss', 'val_loss'],
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='min',
                             period=ckpt_period)

tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=0)

if n_epochs_to_train <= initial_epoch:
    n_epochs_to_train += initial_epoch

image_loader = SiameseLoader('./data')
model.fit_generator(generator=image_loader.generate(32),
                    steps_per_epoch=4,
                    epochs=n_epochs_to_train,
                    validation_data=image_loader.generate_val(32),
                    validation_steps=1,
                    callbacks=[checkpoint, tensorboard],
                    use_multiprocessing=True,
                    initial_epoch=initial_epoch,
                    workers=6)


