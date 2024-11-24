from ultralytics import YOLO
import torch


class FinetuneYolo(YOLO):
    def load_backbone(self, ckptPath):
        """
        Transfers backbone parameters with matching names and shapes from 'weights' to model.
        """
        backboneWeights = torch.load(ckptPath)
        self.model.load_state_dict(backboneWeights, strict=False)
        return self
    
    def freeze_backbone(self, freeze):
        # Freeze backbone params
        freeze = [f'model.{x}.' for x in  range(freeze)]  # layers to freeze
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze):
                v.requires_grad = False
        return self
    
    def unfreeze_backbone(self):
        # unfreeze backbone params
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        return self


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov8s.yaml', help='model.yaml')
    parser.add_argument('--ckptPath', type=str, default=r"D:\yp\yolo_finetuning\weights\yolov8s.pt", help='The path of checkpoints')
    opt = parser.parse_args()

    model = FinetuneYolo(opt.cfg).load_backbone(opt.ckptPath)  # build from YAML and transfer weights
    model.freeze_backbone(10)
    model.train(data=r'D:\yp\yolo_finetuning\data_small\data.yaml', epochs=50, imgsz=768, batch=16)