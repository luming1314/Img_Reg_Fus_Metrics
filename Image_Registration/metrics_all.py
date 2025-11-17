import numpy as np

from metrics import *
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES']='0'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='nirscene_png', help="[nirscene_png, Road]")
parser.add_argument('--method', type=str, default='+', help="[AUNet_v1, not, SuperFusion, UMF-CMGR, CrossRAFT, GLU-Net, RedFeat]")
parser.add_argument('--root_in', type=str, default='../results/test/Reg', help="Please fill in the path for A root_in")
parser.add_argument('--save_excel', type=str, default='../results/test/Reg', help="Please fill in the path for A root_in")
parser.add_argument('--root_gt', type=str, default='../../../SuperFusion-main/dataset/test', help="Please fill in the path for A root_gt")
parser.add_argument('--use_ir', action='store_true', help="Set whether to use the original image")

if __name__ == '__main__':

    mse_metric  = MSE().cuda()
    lncc_metric = LNCC().cuda()
    ncc_metric  = NCC().cuda()
    rmi_metric  = RMI().cuda()
    mi_metric = MI().cuda()
    ssim_metric = SSIM().cuda()
    psnr_metric = PSNR().cuda()
    mae_metric = MAE().cuda()
    mee_metric = MEE().cuda()

    opts = parser.parse_args()
    method_all = ['Unregistered', 'AUNet_v1', 'SuperFusion', 'UMF-CMGR', 'CrossRAFT', 'GLU-Net', 'RedFeat']
    root_gt = os.path.join(opts.root_gt, opts.dataset, 'vi')
    root_in = ''
    MSE_List_mean, NCC_List_mean, MI_List_mean, SSIM_List_mean, MAE_List_mean, MEE_List_mean, PSNR_List_mean = [], [], [], [], [], [], []

    MSE_List_std, NCC_List_std, MI_List_std, SSIM_List_std, MAE_List_std, MEE_List_std, PSNR_List_std = [], [], [], [], [], [], []
    if opts.method == '+':
        for name in method_all:
            if name == 'Unregistered':
                if opts.use_ir:
                    root_in = os.path.join(opts.root_gt, opts.dataset, 'ir')
                else:
                    root_in = os.path.join(opts.root_gt, opts.dataset, 'ir_warp')
            else:
                root_in = os.path.join(opts.root_in, name, opts.dataset + '_{}'.format(name))

            MSE_list, NCC_list, MI_list, SSIM_list, MAE_list, MEE_list, PSNR_list \
                = calc_img_metrics(
                mse_metric,
                ncc_metric,
                lncc_metric,
                mi_metric,
                ssim_metric,
                psnr_metric,
                mae_metric,
                mee_metric,
                root_in, root_gt)
            print('This method is '+ name)

            # MSE_List.append(str((sum(MSE_list)/len(MSE_list)).item()))
            # NCC_List.append(str((sum(NCC_list)/len(NCC_list)).item()))
            # MI_List.append(str((sum(MI_list)/len(MI_list)).item()))
            # SSIM_List.append(str((sum(SSIM_list)/len(SSIM_list)).item()))
            # MAE_List.append(str((sum(MAE_list)/len(MAE_list)).item()))
            # MEE_List.append(str((sum(MEE_list)/len(MEE_list)).item()))
            # PSNR_List.append(str((sum(PSNR_list)/len(PSNR_list)).item()))

            MSE_List_mean.append(np.mean(MSE_list))
            NCC_List_mean.append(np.mean(NCC_list))
            MI_List_mean.append(np.mean(MI_list))
            SSIM_List_mean.append(np.mean(SSIM_list))
            MAE_List_mean.append(np.mean(MAE_list))
            MEE_List_mean.append(np.mean(MEE_list))
            PSNR_List_mean.append(np.mean(PSNR_list))

            MSE_List_std.append(np.std(MSE_list))
            NCC_List_std.append(np.std(NCC_list))
            MI_List_std.append(np.std(MI_list))
            SSIM_List_std.append(np.std(SSIM_list))
            MAE_List_std.append(np.std(MAE_list))
            MEE_List_std.append(np.std(MEE_list))
            PSNR_List_std.append(np.std(PSNR_list))

        data_mean = {
            'method': method_all,
            'MSE': MSE_List_mean,
            'NCC': NCC_List_mean,
            'MI': MI_List_mean,
            'SSIM': SSIM_List_mean,
            'MAE': MAE_List_mean,
            'MEE': MEE_List_mean,
            'PSNR': PSNR_List_mean
        }

        data_std = {
            'method': method_all,
            'MSE': MSE_List_std,
            'NCC': NCC_List_std,
            'MI': MI_List_std,
            'SSIM': SSIM_List_std,
            'MAE': MAE_List_std,
            'MEE': MEE_List_std,
            'PSNR': PSNR_List_std
        }

        # 将数据转换为pandas DataFrame
        df_mean = pd.DataFrame(data_mean)
        df_std = pd.DataFrame(data_std)

        # 保存到Excel文件
        filename = os.path.join(opts.save_excel, 'reg_{}.xlsx'.format(opts.dataset))

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df_mean.to_excel(writer, sheet_name='mean', index=False)
            df_std.to_excel(writer, sheet_name='std', index=False)


        print(f"Data saved to {filename}")

        pass
    else:
        raise ValueError("后续开发.")




