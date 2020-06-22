#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import copy
import os.path as osp
import json
import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
import albumentations as al
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw
from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False
import glob
import os


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def load_images(image_paths, input_size, use_crop):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path, input_size, use_crop)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images


def get_classtable():
    classes = []
    with open("samples/synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes


def preprocess(image_path, input_size, use_crop):
    if use_crop:
        resize = [al.Resize(int(input_size * 1.1), int(input_size * 1.1)),
                  al.CenterCrop(height=input_size, width=input_size)]
    else:
        resize = [al.Resize(input_size, input_size)]
    transform = al.Compose(resize + [
        al.Normalize(),
        ToTensorV2()
    ])

    im = Image.open(image_path).convert("RGB")
    raw_image = np.array(im)
    image = transform(image=raw_image)['image']
    # raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (input_size,) * 2)
    # image = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


def load_checkpoint(checkpoint_path, model, use_gpu=True):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    if use_gpu:
        checkpoint_dict = torch.load(checkpoint_path)
    else:
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    pretrained_dict = checkpoint_dict['state_dict']
    model.load_state_dict(pretrained_dict)
    optimizer_state = checkpoint_dict['optimizer']
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}".format(
        checkpoint_path, iteration))
    return model, optimizer_state, learning_rate, iteration


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


@click.group()
@click.pass_context
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option("-i", "--image-paths", type=str, required=True)
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-t", "--target-layer", type=str, required=True)
@click.option("-k", "--topk", type=int, default=3)
@click.option("-m", "--model_path", type=str, default=None)
@click.option("-s", "--input_size", type=int, default=560)
@click.option("-n", "--num_classes", type=int, default=3)
@click.option("-b", "--batch_size", type=int, default=10)
@click.option("-p", "--pretrained", type=bool, default=True)
@click.option("-u", "--use_crop", type=bool, default=True)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("-c", "--classes_json", type=str, default='["normal", "warning", "disease"]')
@click.option("--cuda/--cpu", default=True)
def demo1(image_paths, target_layer, arch, topk, model_path, input_size, num_classes, batch_size, pretrained,
          use_crop, output_dir, classes_json, cuda):
    """
    Visualize model responses given multiple images
    """
    os.makedirs(output_dir, exist_ok=True)

    device = get_device(cuda)

    # Synset words
    # classes = get_classtable()
    classes = json.loads(classes_json)
    # Model from torchvision
    model = models.__dict__[arch](pretrained=pretrained, num_classes=num_classes)
    if model_path:
        model, _, _, _ = load_checkpoint(model_path, model)
    model.to(device)
    model.eval()

    # Images
    image_paths = glob.glob(os.path.join(image_paths, "*"))

    image_paths_list = chunks(image_paths, batch_size)

    gcam = GradCAM(model=model)
    gbp = GuidedBackPropagation(model=model)
    bp = BackPropagation(model=model)
    # deconv = Deconvnet(model=model)

    for image_idx, image_paths in enumerate(image_paths_list):
        images, raw_images = load_images(image_paths, input_size, use_crop)
        image_file_names = [os.path.splitext(os.path.basename(fn))[0] for fn in image_paths]
        images = torch.stack(images).to(device)

        """
        Common usage:
        1. Wrap your model with visualization classes defined in grad_cam.py
        2. Run forward() with images
        3. Run backward() with a list of specific classes
        4. Run generate() to export results
        """

        # =========================================================================
        # print("Vanilla Backpropagation:")

        probs, ids = bp.forward(images)  # sorted

        # for i in range(topk):
        #     bp.backward(ids=ids[:, [i]])
        #     gradients = bp.generate()
        #
        #     # Save results as image files
        #     for j in range(len(images)):
        #         print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))
        #
        #         save_gradient(
        #             filename=osp.join(
        #                 output_dir,
        #                 "{}-{}-{}-{}-vanilla-{}.png".format(image_file_names[j], image_idx, j, arch,
        #                                                     classes[ids[j, i]]),
        #             ),
        #             gradient=gradients[j],
        #         )
        #
        # # Remove all the hook function in the "model"
        # bp.remove_hook()

        # =========================================================================
        # print("Deconvolution:")
        #
        # _ = deconv.forward(images)
        #
        # for i in range(topk):
        #     deconv.backward(ids=ids[:, [i]])
        #     gradients = deconv.generate()
        #
        #     for j in range(len(images)):
        #         print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))
        #
        #         save_gradient(
        #             filename=osp.join(
        #                 output_dir,
        #                 "{}-{}-{}-{}-deconvnet-{}.png".format(image_file_names[j], image_idx, j, arch,
        #                                                       classes[ids[j, i]]),
        #             ),
        #             gradient=gradients[j],
        #         )
        #
        # deconv.remove_hook()

        # =========================================================================
        print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

        _ = gcam.forward(images)

        _ = gbp.forward(images)

        for i in range(topk):
            # Guided Backpropagation
            gbp.backward(ids=ids[:, [i]])
            gradients = gbp.generate()

            # Grad-CAM
            gcam.backward(ids=ids[:, [i]])
            regions = gcam.generate(target_layer=target_layer)

            for j in range(len(images)):
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                # Guided Backpropagation
                # save_gradient(
                #     filename=osp.join(
                #         output_dir,
                #         "{}-{}={}-{}-guided-{}.png".format(image_file_names[j], image_idx, j, arch, classes[ids[j, i]]),
                #     ),
                #     gradient=gradients[j],
                # )

                grad_cam_path = osp.join(
                    output_dir,
                    "{}-{}-{}-{}-gradcam-{}-{}.png".format(
                        image_file_names[j], image_idx, j, arch, target_layer, classes[ids[j, i]]
                    ))
                # Grad-CAM
                save_gradcam(
                    filename=grad_cam_path,
                    gcam=regions[j, 0],
                    raw_image=raw_images[j],
                )

                guided_grad_cam_path = osp.join(
                    output_dir,
                    "{}-{}-{}-{}-guided_gradcam-{}-{}.png".format(
                        image_file_names[j], image_idx, j, arch, target_layer, classes[ids[j, i]]
                    ))
                # Guided Grad-CAM
                save_gradient(
                    filename=guided_grad_cam_path,
                    gradient=torch.mul(regions, gradients)[j],
                )

                grad_cam_im = Image.open(grad_cam_path)
                guided_grad_cam_im = Image.open(guided_grad_cam_path)

                im_h = cv2.hconcat([raw_images[j], np.array(grad_cam_im), np.array(guided_grad_cam_im)])
                concat_im = Image.fromarray(im_h)
                concat_w, concat_h = concat_im.size
                bg_im = Image.new("RGBA", (concat_w, concat_h + 80), (0, 0, 0, 255))
                bg_im.paste(concat_im, (0, 80))
                d = ImageDraw.Draw(bg_im)
                d.text((5, 1), classes[ids[j, i]], fill='white')
                bg_im.save("{}-{}-{}-{}-{}.png".format(
                        image_file_names[j], image_idx, j, arch, classes[ids[j, i]]
                    ), format="png")
                os.unlink(grad_cam_path)
                os.unlink(guided_grad_cam_path)
                # cv2.imwrite(output_dir + list_name[i], im_h)

        del images
        del probs
        del ids


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
def demo2(image_paths, output_dir, cuda):
    """
    Generate Grad-CAM at different layers of ResNet-152
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model
    model = models.resnet152(pretrained=True)
    model.to(device)
    model.eval()

    # The four residual layers
    target_layers = ["relu", "layer1", "layer2", "layer3", "layer4"]
    target_class = 243  # "bull mastif"

    # Images
    images, raw_images = load_images(image_paths)
    images = torch.stack(images).to(device)

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    ids_ = torch.LongTensor([[target_class]] * len(images)).to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-CAM @{}".format(target_layer))

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print(
                "\t#{}: {} ({:.5f})".format(
                    j, classes[target_class], float(probs[ids == target_class])
                )
            )

            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "resnet152", target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-k", "--topk", type=int, default=3)
@click.option("-s", "--stride", type=int, default=1)
@click.option("-b", "--n-batches", type=int, default=128)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
def demo3(image_paths, arch, topk, stride, n_batches, output_dir, cuda):
    """
    Generate occlusion sensitivity maps
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    model = models.__dict__[arch](pretrained=True)
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    # Images
    images, _ = load_images(image_paths)
    images = torch.stack(images).to(device)

    print("Occlusion Sensitivity:")

    patche_sizes = [10, 15, 25, 35, 45, 90]

    logits = model(images)
    probs = F.softmax(logits, dim=1)
    probs, ids = probs.sort(dim=1, descending=True)

    for i in range(topk):
        for p in patche_sizes:
            print("Patch:", p)
            sensitivity = occlusion_sensitivity(
                model, images, ids[:, [i]], patch=p, stride=stride, n_batches=n_batches
            )

            # Save results as image files
            for j in range(len(images)):
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                save_sensitivity(
                    filename=osp.join(
                        output_dir,
                        "{}-{}-sensitivity-{}-{}.png".format(
                            j, arch, p, classes[ids[j, i]]
                        ),
                    ),
                    maps=sensitivity[j],
                )


if __name__ == "__main__":
    main()
