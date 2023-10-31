import torch

def load_model_weights(model, weights_file):
    print("=> Initializing model '{}'".format(weights_file))
    pre_model = torch.load(weights_file, map_location="cpu")
    # rename moco pre-trained keys
    print("=> Initializing feature model '{}'".format(weights_file))
    state_dict = pre_model['state_dict']
    for k in list(state_dict.keys()):
        # Initialize the feature module with encoder_q of moco.
        # if k.startswith('module.encoder_q'):
        #     # remove prefix
        #     state_dict["module.encoder_q.{}".format(k[len('module.encoder_q.'):])] = state_dict[k]
        # delete renamed or unused k
        if not k.startswith('module.encoder_q'):
            del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    # msg = model.load_state_dict(state_dict)
    print(msg)
    # optimizer.load_state_dict(pre_model['optimizer'])
