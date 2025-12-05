from my.task import full_connected_task, conv_task,full_connected_task_for_image

db_seed = 20
net_seed = 40
test_size = 0.3
epochs = 250
batch_size = 100
lr = 0.1
# full_connected_task(epochs, lr, batch_size, net_seed=net_seed, db_seed=db_seed, test_size=test_size)
# full_connected_task_for_image(epochs, lr, batch_size, net_seed=net_seed, db_seed=db_seed, test_size=test_size)
conv_task(epochs,lr,batch_size)
