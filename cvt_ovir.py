import os
import emoji
import random
import argparse
import openvino as ov

arg = argparse.ArgumentParser(description='depth completion')
arg.add_argument('-p', '--project_name', type=str, default='inference')
arg.add_argument('-c', '--configuration', type=str, default='val_iccv.yml')
arg = arg.parse_args()
from configs import get as get_cfg
config = get_cfg(arg)

# MODULES
from model import get as get_model
from utility import *

# MINIMIZE RANDOMNESS
torch.manual_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)


def cvt_ovir(args):

    # NETWORK
    print(emoji.emojize('Prepare model... :writing_hand:', variant="emoji_type"), end=' ')
    model = get_model(args)
    net = model(args)
   

    # LOAD MODEL
    print(emoji.emojize('Load model... :writing_hand:', variant="emoji_type"), end=' ')
    if len(args.test_model) != 0:
        assert os.path.exists(args.test_model), \
            "file not found: {}".format(args.test_model)

        checkpoint_ = torch.load(args.test_model, map_location='cpu')
        model = remove_moudle(checkpoint_)
        key_m, key_u = net.load_state_dict(model, strict=True)

        if key_u:
            print('Unexpected keys :')
            print(key_u)

        if key_m:
            print('Missing keys :')
            print(key_m)

    net.eval()
    print('Done!')

    dep = torch.randn(1,1,256,1216)
    rgb = torch.randn(1,3,256,1216)
    ip = torch.randn(1,1,256,1216)
    dep_clear = torch.randn(1,1,256,1216)

    ov_model = ov.convert_model(net,example_input=(dep, rgb, ip, dep_clear))
    ov.save_model(ov_model, f"ov_models/{args.test_model.split('/')[-1].split('.')[0].lower()}.xml")

    emodel = ov.compile_model(ov_model, "CPU")
    # print(emodel([dep, rgb, ip, dep_clear]))
    # print(net(dep, rgb, ip, dep_clear))
    print(f'Convert Done! ')
    print(f"OpenVINO IR saved to ov_models/{args.test_model.split('/')[-1].split('.')[0].lower()}.xml")


if __name__ == '__main__':
    cvt_ovir(config)
