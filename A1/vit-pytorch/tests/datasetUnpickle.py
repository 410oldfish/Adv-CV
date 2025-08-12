from SavePickleToImg import save_cifar10_images

data_batch_path = '../cifar-10-python/cifar-10-batches-py/test_batch'
save_dir = '../examples/data/test'

save_cifar10_images(data_batch_path, save_dir)
