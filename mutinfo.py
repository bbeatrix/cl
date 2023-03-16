import math
import torch

from scipy.special import loggamma


device = "cpu"
batch_size = 64

def entropy(samples):
    N = len(samples)
    d = len(samples[0])
    distance = torch.cdist(samples, samples, p=2)
    distance = distance.masked_select(~torch.eye(N, dtype=bool)).view(N, N - 1)
    epsilon = 1e-10
    R = torch.min(distance, 1).values + epsilon
    Y = torch.mean(math.log(N - 1) + d * torch.log(R))
    pi = 3.14
    B = (d/2) * math.log(pi) - loggamma(d/2 + 1)
    euler = 0.577
    return Y + B + euler


def mutual_information(device, model1, model2, sample_size, dt_loader, index):
    assert model1.feats_dim == model2.feats_dim, "The models' feat dims should be identical"
    output_dim = model1.feats_dim

    representation1 = torch.zeros(sample_size, output_dim).to(device) #itt args.num_dim ki lett cserélve hardcodeolt 320-ra
    representation2 = torch.zeros(sample_size, output_dim).to(device) #TODO: ez, de paraméteresen
    representation_joint = torch.zeros(sample_size, output_dim * 2).to(device)

    for i, (x_batch, _) in enumerate(dt_loader):
        if i * batch_size >= sample_size:
            break
        x_batch = x_batch.to(device)
        with torch.no_grad():
            r1 = model1.features(x_batch)
            r2 = model2.features(x_batch)
        h = len(r1)
        representation1[i * batch_size: i*batch_size+h] = r1
        representation2[i * batch_size: i*batch_size+h] = r2
        

    representation1 = representation1 - torch.mean(representation1, axis=0)
    representation2 = representation2 - torch.mean(representation2, axis=0)
    representation1 = representation1 / torch.norm(representation1)
    representation2 = representation2 / torch.norm(representation2)
    representation_joint = torch.cat((representation1, representation2), 1)
    
    representation1 = representation1.to("cpu")
    representation2 = representation2.to("cpu")
    representation_joint = representation_joint.to("cpu")

    return entropy(representation1)  + entropy(representation2) - entropy(representation_joint)


