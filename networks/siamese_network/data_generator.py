from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

left_generator = datagen.flow_from_directory(
    'data/train_new',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)
left_generator = datagen.flow_from_directory(
    'data/train_new',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

for data,label in left_generator:
    print(label.shape)
print(type(left_generator))
