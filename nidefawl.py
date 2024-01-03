import torch
import numpy as np
import folder_paths
import comfy
import gc
from PIL import Image, ImageFilter
from nodes import MAX_RESOLUTION, VAELoader

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

DEFAULT_SCRIPT = 'RESULT = (vae.decode(samples["samples"]), )'

class PythonScript:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_script"

    CATEGORY = "_for_testing"
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}, "optional": {
            "samples": ("LATENT", ),
            "vae": ("VAE", ),
            "clip": ("CLIP", ),
            "text": ("STRING", {"default": DEFAULT_SCRIPT, "multiline": True}),
            "width": ("INT", {"default": 1280.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 768.0, "min": 0, "max": MAX_RESOLUTION}),
            }}

    def run_script(self, samples=None, vae=None, clip=None, text=None, width=None, height=None):
        SCRIPT = text if text is not None and len(text) > 0 else DEFAULT_SCRIPT
        # print(SCRIPT)
        r = compile(SCRIPT, "<string>", "exec")
        ctxt = {"RESULT": None, "samples": samples, "vae": vae, "clip": clip, "width": width, "height": height}
        eval(r, ctxt)
        # print(ctxt["RESULT"])
        return ctxt["RESULT"]

# Modified version of WAS_Bounded_Image_Blend_With_Mask
# This version uses lists as inputs for batched processing
class Blend_Images_With_Bounded_Masks:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "target": ("IMAGE",),
                "target_mask": ("MASK",),
                "target_bounds": ("IMAGE_BOUNDS",),
                "source": ("IMAGE",),
                "blend_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "feathering": ("INT", {"default": 16, "min": 0, "max": 0xffffffffffffffff}),
                "blend_to_single": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_images_with_bounded_masks"
    
    CATEGORY = "_for_testing"
    OUTPUT_IS_LIST = (False,)
    INPUT_IS_LIST = True
    
    def blend_images_with_bounded_masks(self, target, target_mask, target_bounds, source, blend_factor, feathering, blend_to_single):
        # Ensure we are working with batches
        target_mask = target_mask[0] if len(target_mask) == 1 and target_mask[0].dim() == 3 else target_mask
        blend_factor = blend_factor[0] if len(blend_factor) == 1 else blend_factor
        feathering = feathering[0] if len(feathering) == 1 else feathering
        blend_to_single = blend_to_single[0] if len(blend_to_single) == 1 else blend_to_single

        # If number of target masks and source images don't match, then only the first mask will be used on 
        # the source images, otherwise, each mask will be used for each source image 1 to 1
        # Simarly, if the number of target images and source images don't match then
        # all source images will be applied only to the first target, otherwise they will be applied 
        # 1 to 1
        tgt_mask_len = 1 if len(target_mask) != len(source) else len(source)
        tgt_len = 1 if len(target) != len(source) else len(source)
        bounds_len = 1 if len(target_bounds) != len(source) else len(source)

        tgt_arr = [tensor2pil(tgt) for tgt in target[:tgt_len]]
        src_arr = [tensor2pil(src) for src in source]
        tgt_mask_arr=[]

        # Convert Target Mask(s) to grayscale image format
        for m_idx in range(tgt_mask_len):
            np_array = np.clip((target_mask[m_idx].cpu().numpy().squeeze() * 255.0), 0, 255)
            tgt_mask_arr.append(Image.fromarray((np_array).astype(np.uint8), mode='L'))

        result_tensors = []
        for idx in range(len(src_arr)):
            src = src_arr[idx]
            # If only one target image, then ensure it is the only one used
            if (tgt_len == 1 and idx == 0) or tgt_len > 1:
                tgt = tgt_arr[idx]

            # If only one bounds, no need to extract and calculate more than once
            if (bounds_len == 1 and idx == 0) or bounds_len > 1:
                # Extract the target bounds
                rmin, rmax, cmin, cmax = target_bounds[idx][0]

                # Calculate the dimensions of the target bounds
                height, width = (rmax - rmin + 1, cmax - cmin + 1)

            # If only one mask, then ensure that is the only the first is used
            if (tgt_mask_len == 1 and idx == 0) or tgt_mask_len > 1:
                tgt_mask = tgt_mask_arr[idx]

            # If only one mask and one bounds, then mask only needs to
            #   be extended once because all targets will be the same size
            if (tgt_mask_len == 1 and bounds_len == 1 and idx == 0) or \
                (tgt_mask_len > 1 or bounds_len > 1):

                # This is an imperfect, but easy way to determine if  the mask based on the
                #   target image or source image. If not target, assume source. If neither, 
                #   then it's not going to look right regardless
                if (tgt_mask.size != tgt.size):
                    # Create the blend mask with the same size as the target image
                    mask_extended_canvas = Image.new('L', tgt.size, 0)

                    # Paste the mask portion into the extended mask at the target bounds position
                    mask_extended_canvas.paste(tgt_mask, (cmin, rmin))

                    tgt_mask = mask_extended_canvas

                # Apply feathering (Gaussian blur) to the blend mask if feather_amount is greater than 0
                if feathering > 0:
                    tgt_mask = tgt_mask.filter(ImageFilter.GaussianBlur(radius=feathering))

                # Apply blending factor to the tgt mask now that it has been extended
                tgt_mask = tgt_mask.point(lambda p: p * blend_factor)

            # Resize the source image to match the dimensions of the target bounds
            src_resized = src.resize((width, height), Image.Resampling.LANCZOS)

            # Create a blank image with the same size and mode as the target
            src_positioned = Image.new(tgt.mode, tgt.size)

            # Paste the source image onto the blank image using the target
            src_positioned.paste(src_resized, (cmin, rmin))

            # Blend the source and target images using the blend mask
            result = Image.composite(src_positioned, tgt, tgt_mask)

            if not blend_to_single:
              # Convert the result back to a PyTorch tensor
              result_tensors.append(pil2tensor(result))
            else:
              tgt = result

        if blend_to_single:
          result_tensors.append(pil2tensor(tgt))

        return (torch.cat(result_tensors, dim=0),)


# Modified version of WAS_Bounded_Image_Crop_With_Mask
# Added make_square and quantize_to options
class Crop_Images_With_Masks:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding_left": ("INT", {"default": 64, "min": 0, "max": 0xffffffffffffffff}),
                "padding_right": ("INT", {"default": 64, "min": 0, "max": 0xffffffffffffffff}),
                "padding_top": ("INT", {"default": 64, "min": 0, "max": 0xffffffffffffffff}),
                "padding_bottom": ("INT", {"default": 64, "min": 0, "max": 0xffffffffffffffff}),
                "make_square": ("BOOLEAN", {"default": False}),
                "quantize_to": ("INT", {"default": 8, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE_BOUNDS",)
    FUNCTION = "crop_images_with_masks"
    
    CATEGORY = "_for_testing"
    
    def fix_bb(self, p1, p2, make_square, quantize_to, image_bounds):
      x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
      x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])


      bounds_min = [0, 0]
      bounds_max = image_bounds

      if make_square:
        width = x2 - x1
        height = y2 - y1
        if width > height:
          height_to_square = width - height
          half_diff = height_to_square // 2
          y1 -= half_diff
          y2 += half_diff
          if height_to_square % 2 != 0:
            if y1 > bounds_min[1]:
              y1 -= 1
            else:
              y2 += 1
        else:
          width_to_square = height - width
          half_diff = width_to_square // 2
          x1 -= half_diff
          x2 += half_diff
          if width_to_square % 2 != 0:
            if x1 > bounds_min[0]:
              x1 -= 1
            else:
              x2 += 1
        # check for overlap with current image bounds and move the box if necessary
        if x1 < bounds_min[0]:
          x2 += bounds_min[0] - x1
          x1 = bounds_min[0]
        if y1 < bounds_min[1]:
          y2 += bounds_min[1] - y1
          y1 = bounds_min[1]
        if x2 > bounds_max[0]:
          x1 -= x2 - bounds_max[0]
          x2 = bounds_max[0]
        if y2 > bounds_max[1]:
          y1 -= y2 - bounds_max[1]
          y2 = bounds_max[1]
        

      if quantize_to > 0:
        width = x2 - x1
        height = y2 - y1
        if width % quantize_to != 0:
          width_to_q = (quantize_to - (width % quantize_to))
          half_q = width_to_q // 2
          x1 -= half_q
          x2 += half_q
          if width_to_q % 2 != 0:
            if x1 > bounds_min[0]:
              x1 -= 1
            else:
              x2 += 1
        if height % quantize_to != 0:
          half_q = (quantize_to - (height % quantize_to)) // 2
          y1 -= half_q
          y2 += half_q
          if height % 2 != 0:
            if y1 > bounds_min[1]:
              y1 -= 1
            else:
              y2 += 1
        # check for overlap with current image bounds and move the box if necessary
        if x1 < bounds_min[0]:
          x2 += bounds_min[0] - x1
          x1 = bounds_min[0]
        if y1 < bounds_min[1]:
          y2 += bounds_min[1] - y1
          y1 = bounds_min[1]
        if x2 > bounds_max[0]:
          x1 -= x2 - bounds_max[0]
          x2 = bounds_max[0]
        if y2 > bounds_max[1]:
          y1 -= y2 - bounds_max[1]
          y2 = bounds_max[1]

      # clamp to image bounds
      x1 = max(x1, bounds_min[0])
      y1 = max(y1, bounds_min[1])
      x2 = min(x2, bounds_max[0])
      y2 = min(y2, bounds_max[1])
      return ([x1, y1], [x2, y2])

    def crop_images_with_masks(self, image, mask, padding_left, padding_right, padding_top, padding_bottom, make_square, quantize_to):
        # Ensure we are working with batches
        image = image.unsqueeze(0) if image.dim() == 3 else image
        mask = mask.unsqueeze(0) if mask.dim() == 2 else mask

        # If number of masks and images don't match, then only the first mask will be used on 
        # the images, otherwise, each mask will be used for each image 1 to 1
        mask_len = len(mask)
        image_len = len(image)

        cropped_images = []
        all_bounds = []
        for i in range(max(mask_len, image_len)):
            # Single mask or multiple?
            if (mask_len == 1 and i == 0) or mask_len > 0:
                rows = torch.any(mask[i], dim=1)
                cols = torch.any(mask[i], dim=0)
                row_bounds = torch.where(rows)[0]
                if len(row_bounds) == 0:
                    continue
                col_bounds = torch.where(cols)[0]
                if len(col_bounds) == 0:
                    continue

                rmin, rmax = row_bounds[[0, -1]]
                cmin, cmax = col_bounds[[0, -1]]

                rmin = max(rmin - padding_top, 0)
                rmax = min(rmax + padding_bottom, mask[i].shape[0] - 1)
                cmin = max(cmin - padding_left, 0)
                cmax = min(cmax + padding_right, mask[i].shape[1] - 1)

                # Fix the bounding box to be square if necessary
                p1, p2 = self.fix_bb([cmin, rmin], [cmax + 1, rmax + 1], make_square, quantize_to, [mask[i].shape[1], mask[i].shape[0]])
                cmin, rmin = p1
                cmax, rmax = p2[0] - 1, p2[1] - 1

            # Even if only a single mask, create a bounds for each cropped image
            all_bounds.append([rmin, rmax, cmin, cmax])
            cropped_images.append(image[i][rmin:rmax+1, cmin:cmax+1, :])
        if len(cropped_images) == 0:
            empty_img = torch.zeros((1, 8, 8, 3))
            return empty_img, [[0, 7, 0, 7]]
        return torch.stack(cropped_images), all_bounds
        #     # Even if only a single mask, create a bounds for each cropped image
        #     all_bounds.append([rmin, rmax, cmin, cmax])
        #     cropped_images.append(image[i % image_len][rmin:rmax+1, cmin:cmax+1, :])
  
        # return (torch.cat(cropped_images, dim=0), all_bounds)


# Modified version of VAELoader from comfyui
# This loader allows setting the datatype of the loaded VAE and the device type
class VAELoaderDataType():
    data_types = ["fp32", "fp16", "bf16"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (VAELoader.vae_list(),),
                "data_type": (s.data_types,),
                "device_type": (["GPU", "CPU"],),
            }
        }
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"

    CATEGORY = "loaders"

    def load_vae(self, vae_name, data_type, device_type):
        if vae_name in ["taesd", "taesdxl"]:
            sd = VAELoader.load_taesd(vae_name)
        else:
            vae_path = folder_paths.get_full_path("vae", vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
        device = torch.device(torch.cuda.current_device())
        if device_type == "CPU":
            device = torch.device("cpu")
        dt = torch.float32
        if data_type == "fp16":
            dt = torch.float16
        elif data_type == "bf16":
            dt = torch.bfloat16
        vae = comfy.sd.VAE(sd=sd, dtype=dt, device=device)
        return (vae,)

# https://github.com/comfyanonymous/ComfyUI_experiments
class ModelSamplerTonemapNoiseTest:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "custom_node_experiments"

    def patch(self, model, multiplier):
        
        def sampler_tonemap_reinhard(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            noise_pred = (cond - uncond)
            noise_pred_vector_magnitude = (torch.linalg.vector_norm(noise_pred, dim=(1)) + 0.0000000001)[:,None]
            noise_pred /= noise_pred_vector_magnitude

            mean = torch.mean(noise_pred_vector_magnitude, dim=(1,2,3), keepdim=True)
            std = torch.std(noise_pred_vector_magnitude, dim=(1,2,3), keepdim=True)

            top = (std * 3 + mean) * multiplier

            #reinhard
            noise_pred_vector_magnitude *= (1.0 / top)
            new_magnitude = noise_pred_vector_magnitude / (noise_pred_vector_magnitude + 1.0)
            new_magnitude *= top

            return uncond + noise_pred * new_magnitude * cond_scale

        m = model.clone()
        m.set_model_sampler_cfg_function(sampler_tonemap_reinhard)
        return (m, )

# https://github.com/ntdviet/comfyui-ext
class gcLatentTunnel:
    @classmethod
    def INPUT_TYPES(s):
      return {
        "required": {
			  "samples": ("LATENT",),
		  }
	  }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "gcTunnel"
    CATEGORY = "latent"

    def gcTunnel(self, samples):
        print("Garbage collecting...")
        s = samples.copy()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return (s,)


# https://github.com/comfyanonymous/ComfyUI_experiments
class ReferenceOnlySimple:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "reference": ("LATENT",),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})
                              }}

    RETURN_TYPES = ("MODEL", "LATENT")
    FUNCTION = "reference_only"

    CATEGORY = "custom_node_experiments"

    def reference_only(self, model, reference, batch_size):
        model_reference = model.clone()
        size_latent = list(reference["samples"].shape)
        size_latent[0] = batch_size
        latent = {}
        latent["samples"] = torch.zeros(size_latent)

        batch = latent["samples"].shape[0] + reference["samples"].shape[0]
        def reference_apply(q, k, v, extra_options):
            k = k.clone().repeat(1, 2, 1)

            for o in range(0, q.shape[0], batch):
                for x in range(1, batch):
                    k[x + o, q.shape[1]:] = q[o,:]

            return q, k, k

        model_reference.set_model_attn1_patch(reference_apply)
        out_latent = torch.cat((reference["samples"], latent["samples"]))
        if "noise_mask" in latent:
            mask = latent["noise_mask"]
        else:
            mask = torch.ones((64,64), dtype=torch.float32, device="cpu")

        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
        if mask.shape[0] < latent["samples"].shape[0]:
            print(latent["samples"].shape, mask.shape)
            mask = mask.repeat(latent["samples"].shape[0], 1, 1)

        out_mask = torch.zeros((1,mask.shape[1],mask.shape[2]), dtype=torch.float32, device="cpu")
        return (model_reference, {"samples": out_latent, "noise_mask": torch.cat((out_mask, mask))})

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return int(hex_color, 16)

class EmptyImageWithColor:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                              "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "color": ("COLOR", {"default": "#000000"}),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"

    CATEGORY = "image"

    def generate(self, width, height, batch_size=1, color=0):
        color = hex_to_rgb(color)
        r = torch.full([batch_size, height, width, 1], ((color >> 16) & 0xFF) / 0xFF)
        g = torch.full([batch_size, height, width, 1], ((color >> 8) & 0xFF) / 0xFF)
        b = torch.full([batch_size, height, width, 1], ((color) & 0xFF) / 0xFF)
        return (torch.cat((r, g, b), dim=-1), )

class MaskFromColor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "color": ("COLOR", { "default": "#FFFFFF", }),
                "threshold": ("INT", { "default": 0, "min": 0, "max": 127, "step": 1, }),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, color, threshold):
        temp = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)
        color = hex_to_rgb(color)
        color = torch.tensor([(color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF], dtype=torch.int)
        lower_bound = (color - threshold).clamp(min=0)
        upper_bound = (color + threshold).clamp(max=255)
        lower_bound = lower_bound.view(1, 1, 1, 3)
        upper_bound = upper_bound.view(1, 1, 1, 3)
        mask = (temp >= lower_bound) & (temp <= upper_bound)
        mask = mask.all(dim=-1)
        mask = mask.float()
        return (mask, )

class SetLatentCustomNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "noise": ("LATENT",),   }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "set_custom_noise"

    CATEGORY = "latent/noise"

    def set_custom_noise(self, samples, noise):
        s = samples.copy()
        if noise['samples'].shape[0] != samples['samples'].shape[0] and noise['samples'].shape[0] == 1:
          s["noise_custom"] = noise['samples'].clone().repeat(samples['samples'].shape[0], 1, 1, 1)
        else:
          s["noise_custom"] = noise['samples'].clone()
        return (s,)

# Converts a latent to an image without decoding
class LatentToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",), }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "latent_to_image"
   
    CATEGORY = "latent"
   
    def latent_to_image(self, samples):
        return (samples["samples"],)

class LatentNoiseStd:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "noise_std": ("FLOAT", {"default": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "get_latent_noise_std"

    CATEGORY = "WAS Suite/Latent/Generate"

    def get_latent_noise_std(self, samples, noise_std, seed):
        s = samples.copy()
        torch.manual_seed(seed)
        noise = torch.randn_like(s["samples"]) * noise_std
        s["samples"] = s["samples"] + noise
        return (s,)

NODE_CLASS_MAPPINGS = {
    "PythonScript": PythonScript,
    "BlendImagesWithBoundedMasks": Blend_Images_With_Bounded_Masks,
    "CropImagesWithMasks": Crop_Images_With_Masks,
    "VAELoaderDataType": VAELoaderDataType,
    "ModelSamplerTonemapNoiseTest": ModelSamplerTonemapNoiseTest,
	  "gcLatentTunnel": gcLatentTunnel,
    "ReferenceOnlySimple": ReferenceOnlySimple,
    "EmptyImageWithColor": EmptyImageWithColor,
    "MaskFromColor": MaskFromColor,
    "SetLatentCustomNoise": SetLatentCustomNoise,
}
