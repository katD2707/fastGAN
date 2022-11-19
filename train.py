from discriminator import Discriminator
from generator import Generator
from utils import weights_init, copy_G_params, load_params, get_dir
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import InfiniteSamplerWrapper
from torch import optim
from tqdm import tqdm
from learning import train_d
from diffaug import DiffAugment
import torchvision.utils as vutils
import torch.nn.functional as F
import config

POLICY = 'color,translation'


def train(args):
    data_root = args.data_root
    total_iterations = args.iter
    checkpoint_path = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    nz = args.nz
    ngf = args.ngf
    nlr = args.nlr
    nbeta1 = args.nbeta1
    current_iteration = args.start_iter
    saved_model_folder, saved_image_folder = get_dir(args)

    device = torch.device("cpu")
    if args.use_cuda:
        device = torch.device("cuda:0")

    # Dataset
    transform_list = [
        transforms.Resize((int(im_size), int(im_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    trans = transforms.Compose(transform_list)

    if 'lmdb' in args.data_root:
        from datasets import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    else:
        from datasets import ImageFolder
        dataset = ImageFolder(root=data_root, transform=trans)

    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 sampler=InfiniteSamplerWrapper(dataset), num_workers=args.num_workers,
                                 pin_memory=True))

    # Generator
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    netG.apply(weights_init)
    netG.to(device)

    # Discriminator
    netD = Discriminator(ndf=args.ndf, im_size=im_size)
    netD.apply(weights_init)
    netD.to(device)

    # For EMA update
    avg_param_G = copy_G_params(netG)

    # Sampling after training step
    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)

    # Optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    # Load checkpoint
    if checkpoint_path != 'None':
        ckpt = torch.load(checkpoint_path)
        netG.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['g'].items()})
        netD.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['d'].items()})
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint_path.split('_')[-1].split('.')[0])
        del ckpt

    for iteration in tqdm(range(current_iteration, total_iterations + 1)):
        real_image = next(dataloader)
        real_image = real_image.to(device)
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)

        fake_images = netG(noise)

        real_image = DiffAugment(real_image, policy=POLICY)
        fake_images = [DiffAugment(fake, policy=POLICY) for fake in fake_images]

        ## 2. train Discriminator
        netD.zero_grad()

        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_image, label="real")
        train_d(netD, [fi.detach() for fi in fake_images], label="fake")
        optimizerD.step()

        ## 3. train Generator
        netG.zero_grad()
        pred_g = netD(fake_images, "fake")
        err_g = -pred_g.mean()

        err_g.backward()
        optimizerG.step()

        ### EMA Generator update
        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f" % (err_dr, -err_g.item()))

        if iteration % (args.save_interval * 10) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), saved_image_folder + '/%d.jpg' % iteration,
                                  nrow=4)
                vutils.save_image(torch.cat([
                    F.interpolate(real_image, 128),
                    rec_img_all, rec_img_small,
                    rec_img_part]).add(1).mul(0.5), saved_image_folder + '/rec_%d.jpg' % iteration)
            load_params(netG, backup_para)

        if iteration % (args.save_interval * 50) == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g': netG.state_dict(), 'd': netD.state_dict()}, saved_model_folder + '/%d.pth' % iteration)
            load_params(netG, backup_para)
            torch.save({'g': netG.state_dict(),
                        'd': netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder + '/all_%d.pth' % iteration)

if __name__ == "__main__":
    args = config.parse_args()
    print(args)

    train(args)