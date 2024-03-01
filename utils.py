import torch
from PIL import Image
import os


def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def save_model(encoder, decoder, discri, args, e='end', batch_id='none'):
    encoder.eval().cpu()
    encoder_model_filename = "encoder_epoch_" + e + "_batch_id_" + batch_id + ".pth"
    encoder_model_path = os.path.join(args.checkpoint_model_dir, encoder_model_filename)
    torch.save(encoder.state_dict(), encoder_model_path)

    decoder.eval().cpu()
    decoder_model_filename = "decoder_epoch_" + e + "_batch_id_" + batch_id + ".pth"
    decoder_model_path = os.path.join(args.checkpoint_model_dir, decoder_model_filename)
    torch.save(decoder.state_dict(), decoder_model_path)

    discri.eval().cpu()
    discri_model_filename = "discri_epoch_" + e + "_batch_id_" + batch_id + ".pth"
    discri_model_path = os.path.join(args.checkpoint_model_dir, discri_model_filename)
    torch.save(encoder.state_dict(), discri_model_path)

