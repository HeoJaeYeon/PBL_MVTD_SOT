import random
import numpy as np
import math
import cv2 as cv
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvisf


class Transform:
    """A set of transformations, used for e.g. data augmentation.
    Args of constructor:
        transforms: An arbitrary number of transformations, derived from the TransformBase class.
                    They are applied in the order they are given.

    The Transform object can jointly transform images, bounding boxes and segmentation masks.
    This is done by calling the object with the following key-word arguments (all are optional).

    The following arguments are inputs to be transformed. They are either supplied as a single instance, or a list of instances.
        image  -  Image
        coords  -  2xN dimensional Tensor of 2D image coordinates [y, x]
        bbox  -  Bounding box on the form [x, y, w, h]
        mask  -  Segmentation mask with discrete classes

    The following parameters can be supplied with calling the transform object:
        joint [Bool]  -  If True then transform all images/coords/bbox/mask in the list jointly using the same transformation.
                         Otherwise each tuple (images, coords, bbox, mask) will be transformed independently using
                         different random rolls. Default: True.
        new_roll [Bool]  -  If False, then no new random roll is performed, and the saved result from the previous roll
                            is used instead. Default: True.

    Check the DiMPProcessing class for examples.
    """

    def __init__(self, *transforms):
        if len(transforms) == 1 and isinstance(transforms[0], (list, tuple)):
            transforms = transforms[0]
        self.transforms = transforms
        self._valid_inputs = ['image', 'coords', 'bbox', 'mask', 'att']
        self._valid_args = ['joint', 'new_roll']
        self._valid_all = self._valid_inputs + self._valid_args

    def __call__(self, **inputs):
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        for v in inputs.keys():
            if v not in self._valid_all:
                raise ValueError('Incorrect input \"{}\" to transform. Only supports inputs {} and arguments {}.'.format(v, self._valid_inputs, self._valid_args))

        joint_mode = inputs.get('joint', True)
        new_roll = inputs.get('new_roll', True)

        if not joint_mode:
            out = zip(*[self(**inp) for inp in self._split_inputs(inputs)])
            return tuple(list(o) for o in out)

        out = {k: v for k, v in inputs.items() if k in self._valid_inputs}

        for t in self.transforms:
            out = t(**out, joint=joint_mode, new_roll=new_roll)
        if len(var_names) == 1:
            return out[var_names[0]]
        # Make sure order is correct
        return tuple(out[v] for v in var_names)

    def _split_inputs(self, inputs):
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        split_inputs = [{k: v for k, v in zip(var_names, vals)} for vals in zip(*[inputs[vn] for vn in var_names])]
        for arg_name, arg_val in filter(lambda it: it[0]!='joint' and it[0] in self._valid_args, inputs.items()):
            if isinstance(arg_val, list):
                for inp, av in zip(split_inputs, arg_val):
                    inp[arg_name] = av
            else:
                for inp in split_inputs:
                    inp[arg_name] = arg_val
        return split_inputs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class TransformBase:
    """Base class for transformation objects. See the Transform class for details."""
    def __init__(self):
        """2020.12.24 Add 'att' to valid inputs"""
        self._valid_inputs = ['image', 'coords', 'bbox', 'mask', 'att']
        self._valid_args = ['new_roll']
        self._valid_all = self._valid_inputs + self._valid_args
        self._rand_params = None

    def __call__(self, **inputs):
        # Split input
        input_vars = {k: v for k, v in inputs.items() if k in self._valid_inputs}
        input_args = {k: v for k, v in inputs.items() if k in self._valid_args}

        # Roll random parameters for the transform
        if input_args.get('new_roll', True):
            rand_params = self.roll()
            if rand_params is None:
                rand_params = ()
            elif not isinstance(rand_params, tuple):
                rand_params = (rand_params,)
            self._rand_params = rand_params

        outputs = dict()
        for var_name, var in input_vars.items():
            if var is not None:
                transform_func = getattr(self, 'transform_' + var_name)
                if var_name in ['coords', 'bbox']:
                    params = (self._get_image_size(input_vars),) + self._rand_params
                else:
                    params = self._rand_params
                if isinstance(var, (list, tuple)):
                    outputs[var_name] = [transform_func(x, *params) for x in var]
                else:
                    outputs[var_name] = transform_func(var, *params)
        return outputs

    def _get_image_size(self, inputs):
        im = None
        for var_name in ['image', 'mask']:
            if inputs.get(var_name) is not None:
                im = inputs[var_name]
                break
        if im is None:
            return None
        if isinstance(im, (list, tuple)):
            im = im[0]
        if isinstance(im, np.ndarray):
            return im.shape[:2]
        if torch.is_tensor(im):
            return (im.shape[-2], im.shape[-1])
        raise Exception('Unknown image type')

    def roll(self):
        return None

    def transform_image(self, image, *rand_params):
        """Must be deterministic"""
        return image

    def transform_coords(self, coords, image_shape, *rand_params):
        """Must be deterministic"""
        return coords

    def transform_bbox(self, bbox, image_shape, *rand_params):
        """Assumes [x, y, w, h]"""
        # Check if not overloaded
        if self.transform_coords.__code__ == TransformBase.transform_coords.__code__:
            return bbox

        coord = bbox.clone().view(-1,2).t().flip(0)

        x1 = coord[1, 0]
        x2 = coord[1, 0] + coord[1, 1]

        y1 = coord[0, 0]
        y2 = coord[0, 0] + coord[0, 1]

        coord_all = torch.tensor([[y1, y1, y2, y2], [x1, x2, x2, x1]])

        coord_transf = self.transform_coords(coord_all, image_shape, *rand_params).flip(0)
        tl = torch.min(coord_transf, dim=1)[0]
        sz = torch.max(coord_transf, dim=1)[0] - tl
        bbox_out = torch.cat((tl, sz), dim=-1).reshape(bbox.shape)
        return bbox_out

    def transform_mask(self, mask, *rand_params):
        """Must be deterministic"""
        return mask

    def transform_att(self, att, *rand_params):
        """2020.12.24 Added to deal with attention masks"""
        return att


class ToTensor(TransformBase):
    """Convert to a Tensor"""

    def transform_image(self, image):
        # handle numpy array
        if image.ndim == 2:
            image = image[:, :, None]

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(image, torch.ByteTensor):
            return image.float().div(255)
        else:
            return image

    def transfrom_mask(self, mask):
        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask)

    def transform_att(self, att):
        if isinstance(att, np.ndarray):
            return torch.from_numpy(att).to(torch.bool)
        elif isinstance(att, torch.Tensor):
            return att.to(torch.bool)
        else:
            raise ValueError ("dtype must be np.ndarray or torch.Tensor")


class ToTensorAndJitter(TransformBase):
    """Convert to a Tensor and jitter brightness"""
    def __init__(self, brightness_jitter=0.0, normalize=True):
        super().__init__()
        self.brightness_jitter = brightness_jitter
        self.normalize = normalize

    def roll(self):
        return np.random.uniform(max(0, 1 - self.brightness_jitter), 1 + self.brightness_jitter)

    def transform_image(self, image, brightness_factor):
        # handle numpy array
        image = torch.from_numpy(image.transpose((2, 0, 1)))

        # backward compatibility
        if self.normalize:
            return image.float().mul(brightness_factor/255.0).clamp(0.0, 1.0)
        else:
            return image.float().mul(brightness_factor).clamp(0.0, 255.0)

    def transform_mask(self, mask, brightness_factor):
        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask)
        else:
            return mask
    def transform_att(self, att, brightness_factor):
        if isinstance(att, np.ndarray):
            return torch.from_numpy(att).to(torch.bool)
        elif isinstance(att, torch.Tensor):
            return att.to(torch.bool)
        else:
            raise ValueError ("dtype must be np.ndarray or torch.Tensor")


class Normalize(TransformBase):
    """Normalize image"""
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def transform_image(self, image):
        return tvisf.normalize(image, self.mean, self.std, self.inplace)


class ToGrayscale(TransformBase):
    """Converts image to grayscale with probability"""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability
        self.color_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_grayscale):
        if do_grayscale:
            if torch.is_tensor(image):
                raise NotImplementedError('Implement torch variant.')
            img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            return np.stack([img_gray, img_gray, img_gray], axis=2)
            # return np.repeat(np.sum(img * self.color_weights, axis=2, keepdims=True).astype(np.uint8), 3, axis=2)
        return image


class ToBGR(TransformBase):
    """Converts image to BGR"""
    def transform_image(self, image):
        if torch.is_tensor(image):
            raise NotImplementedError('Implement torch variant.')
        img_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return img_bgr


class RandomHorizontalFlip(TransformBase):
    """Horizontally flip image randomly with a probability p."""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_flip):
        if do_flip:
            if torch.is_tensor(image):
                return image.flip((2,))
            return np.fliplr(image).copy()
        return image

    def transform_coords(self, coords, image_shape, do_flip):
        if do_flip:
            coords_flip = coords.clone()
            coords_flip[1,:] = (image_shape[1] - 1) - coords[1,:]
            return coords_flip
        return coords

    def transform_mask(self, mask, do_flip):
        if do_flip:
            if torch.is_tensor(mask):
                return mask.flip((-1,))
            return np.fliplr(mask).copy()
        return mask

    def transform_att(self, att, do_flip):
        if do_flip:
            if torch.is_tensor(att):
                return att.flip((-1,))
            return np.fliplr(att).copy()
        return att


class RandomHorizontalFlip_Norm(RandomHorizontalFlip):
    """Horizontally flip image randomly with a probability p.
    The difference is that the coord is normalized to [0,1]"""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability

    def transform_coords(self, coords, image_shape, do_flip):
        """we should use 1 rather than image_shape"""
        if do_flip:
            coords_flip = coords.clone()
            coords_flip[1,:] = 1 - coords[1,:]
            return coords_flip
        return coords


# ======== Photometric augmentations for maritime scenes ========

def _clamp_like_input(img: torch.Tensor) -> torch.Tensor:
    """
    Clamp to the value range of the input (estimated from its max).
    After ToTensorAndJitter(normalize=True) images are usually in [0, 1].
    """
    if not torch.is_tensor(img):
        return img
    if img.dtype.is_floating_point:
        # usually [0, 1]; a [0, 255] image would have a max far above 1.5
        max_val = float(img.max().detach().cpu()) if img.numel() > 0 else 1.0
        if max_val > 1.5:
            return img.clamp(0.0, 255.0)
        return img.clamp(0.0, 1.0)
    return img


class RandomContrast(TransformBase):
    """
    Random contrast adjustment.
    - torch tensor (C,H,W)
    - assumes values in [0, 1]
    """
    def __init__(self, probability=0.3, min_factor=0.6, max_factor=1.4):
        super().__init__()
        self.probability = probability
        self.min_factor = min_factor
        self.max_factor = max_factor

    def roll(self):
        do = random.random() < self.probability
        if not do:
            return False, 1.0
        factor = random.uniform(self.min_factor, self.max_factor)
        return True, factor

    def transform_image(self, image, do_contrast, factor):
        if not do_contrast:
            return image

        # torch tensor path
        if torch.is_tensor(image):
            img = image
            # (C,H,W)
            if img.dim() == 3:
                mean = img.mean(dim=(1, 2), keepdim=True)
                out = (img - mean) * factor + mean
            else:
                # other shapes
                mean = img.mean()
                out = (img - mean) * factor + mean
            return _clamp_like_input(out)

        # numpy path (fallback)
        if isinstance(image, np.ndarray):
            img = image.astype(np.float32)
            mean = img.mean(axis=(0, 1), keepdims=True)
            out = (img - mean) * factor + mean
            out = np.clip(out, 0, 255).astype(image.dtype)
            return out

        return image


class RandomGamma(TransformBase):
    """
    Random gamma correction.
    gamma < 1 => brighter, gamma > 1 => darker / different contrast
    """
    def __init__(self, probability=0.3, min_gamma=0.7, max_gamma=1.5, eps=1e-6):
        super().__init__()
        self.probability = probability
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.eps = eps

    def roll(self):
        do = random.random() < self.probability
        if not do:
            return False, 1.0
        gamma = random.uniform(self.min_gamma, self.max_gamma)
        return True, gamma

    def transform_image(self, image, do_gamma, gamma):
        if not do_gamma:
            return image

        if torch.is_tensor(image):
            img = image
            max_val = float(img.max().detach().cpu()) if img.numel() > 0 else 1.0

            # [0, 1] range
            if img.dtype.is_floating_point and max_val <= 1.5:
                out = torch.pow(img.clamp_min(self.eps), gamma)
                return out.clamp(0.0, 1.0)

            # [0, 255] range
            if img.dtype.is_floating_point:
                img01 = (img / 255.0).clamp_min(self.eps)
                out01 = torch.pow(img01, gamma)
                out = out01 * 255.0
                return out.clamp(0.0, 255.0)

            return img

        if isinstance(image, np.ndarray):
            img = image.astype(np.float32) / 255.0
            img = np.clip(img, self.eps, 1.0)
            out = np.power(img, gamma) * 255.0
            out = np.clip(out, 0, 255).astype(image.dtype)
            return out

        return image


class RandomCLAHE(TransformBase):
    """
    Weak CLAHE to enhance local contrast.
    - OpenCV based
    - inputs are usually torch tensors (after ToTensorAndJitter),
      so the torch <-> numpy conversion is handled internally
    """
    def __init__(self, probability=0.15, clip_limit=2.0, tile_grid_size=(8, 8)):
        super().__init__()
        self.probability = probability
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def roll(self):
        do = random.random() < self.probability
        return do

    def _apply_clahe_rgb_uint8(self, img_rgb_u8: np.ndarray) -> np.ndarray:
        # apply CLAHE to the L channel in LAB space
        lab = cv.cvtColor(img_rgb_u8, cv.COLOR_RGB2LAB)
        l, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l2 = clahe.apply(l)
        lab2 = cv.merge([l2, a, b])
        out = cv.cvtColor(lab2, cv.COLOR_LAB2RGB)
        return out

    def transform_image(self, image, do_clahe):
        if not do_clahe:
            return image

        # torch tensor path
        if torch.is_tensor(image):
            img = image
            device = img.device
            dtype = img.dtype

            # (C,H,W) -> (H,W,C)
            if img.dim() == 3 and img.shape[0] in [1, 3]:
                max_val = float(img.max().detach().cpu()) if img.numel() > 0 else 1.0

                if dtype.is_floating_point and max_val <= 1.5:
                    img_u8 = (img.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8)
                else:
                    img_u8 = img.detach().cpu().clamp(0, 255).to(torch.uint8)

                np_img = img_u8.permute(1, 2, 0).numpy()

                # expand 1 channel to 3 channels
                if np_img.shape[2] == 1:
                    np_img = np.repeat(np_img, 3, axis=2)

                out_np = self._apply_clahe_rgb_uint8(np_img)

                out_t = torch.from_numpy(out_np).permute(2, 0, 1).to(device=device)

                # back to float in [0, 1]
                out_t = out_t.to(torch.float32) / 255.0
                out_t = out_t.to(dtype)

                return out_t.clamp(0.0, 1.0)

            return img

        # numpy path
        if isinstance(image, np.ndarray):
            img = image
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=2)
            out = self._apply_clahe_rgb_uint8(img.astype(np.uint8))
            return out.astype(image.dtype)

        return image
