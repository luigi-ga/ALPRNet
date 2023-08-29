
from kornia.filters import MotionBlur
from torchvision import transforms

class MotionBlurTransform:
    def __init__(self, kernel_size, angle, direction):
        self.motion_blur = MotionBlur(kernel_size=kernel_size, angle=angle, direction=direction)

    def __call__(self, img):
        image_tensor = transforms.ToTensor()(img).unsqueeze(0)
        blurred_image_tensor = self.motion_blur(image_tensor)
        blurred_image_pil = transforms.ToPILImage()(blurred_image_tensor.squeeze())
        return blurred_image_pil