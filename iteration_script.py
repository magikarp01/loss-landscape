import os

for iter in range(10, 40, 2):
    os.system('python plot_surface.py --iter=' + str(iter) + ' --x=-2:2:101 --y=-2:2:101 --trainloader deep_learning/maze_9_dataloader_200.pth --testloader deep_learning/maze_13_dataloader_200.pth --model ms_dt_20 --model_file MS_DT_20.pth --model_file2 MS_DT_20_2.pth --model_file3 MS_DT_20_3.pth --loss_name crossentropy --dir_type weights --plot --cuda --dataset mazes_dt')
    os.system('python h52vtp.py --surf_file iteration_models/iter_' + str(iter) + 'MS_DT_20.pt_MS_DT_20.pth__2.pt_MS_DT_20_2.pth_weightsMS_DT_20_3.pt_MS_DT_20_3.pth.h5_[-2.0,2.0,101]x[-2.0,2.0,101].h5 --surf_name train_loss --zmax  10 --log')
