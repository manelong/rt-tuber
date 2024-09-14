# """Copyright(c) 2023 lyuwenyu. All Rights Reserved.
# """
# import os
# import sys
# import time
# from turtle import st

# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# import torch
# import torch.nn as nn 
# import torchvision.transforms as T

# import numpy as np 
# from PIL import Image, ImageDraw
# import random

# from src.core import YAMLConfig

# def draw(images, labels, boxes, scores, thrh = 0.6):
#     for i, im in enumerate(images):
#         draw = ImageDraw.Draw(im)

#         scr = scores[i]
#         lab = labels[i][scr > thrh]
#         box = boxes[i][scr > thrh]

#         for b in box:
#             draw.rectangle(list(b), outline='red',)
#             draw.text((b[0], b[1]), text=str(lab[i].item()), fill='blue', )

#         im.save(f'results_{i}.jpg')


# def main(args, ):
#     """main
#     """
#     cfg = YAMLConfig(args.config, resume=args.resume)

#     if args.resume:
#         checkpoint = torch.load(args.resume, map_location='cpu') 
#         if 'ema' in checkpoint:
#             state = checkpoint['ema']['module']
#         else:
#             state = checkpoint['model']
#     else:
#         raise AttributeError('Only support resume to load model.state_dict by now.')

#     # NOTE load train mode state -> convert to deploy mode
#     # cfg.model.load_state_dict(state)

#     class Model(nn.Module):
#         def __init__(self, ) -> None:
#             super().__init__()
#             self.model = cfg.model.deploy()
#             self.postprocessor = cfg.postprocessor.deploy()
            
#         def forward(self, images, target, orig_target_sizes):
#             start_time = time.time()
#             outputs = self.model(images, target)
#             print(f'Inference time: {time.time() - start_time:.4f}s')
#             start_time = time.time()
#             outputs = self.postprocessor(outputs, orig_target_sizes)
#             print(f'Postprocess time: {time.time() - start_time:.4f}s')
#             return outputs
        
#     model = Model().to(args.device)
#     model.train()
#     # print(model)
    
#     input  =  [torch.randn(1,3,640,640).to(args.device) for _ in range(args.input_len)]
#     # input = torch.randn(1, 3, 640, 640).to(args.device)
#     # 定义一个随机生成的目标
#     targets = [
#         {
#             'labels': torch.tensor([random.randint(0, 2)]).to(args.device),  # 随机选择0、1、2中的一个作为标签
#             'boxes': torch.tensor([[1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]).to(args.device)  # 生成一个随机的边界框
#         }
#     ]

#     output = model(input, targets, torch.tensor([[640, 640]]).to(args.device))

#     # print(output) 



#     # im_pil = Image.open(args.im_file).convert('RGB')
#     # w, h = im_pil.size
#     # orig_size = torch.tensor([w, h])[None].to(args.device)

#     # transforms = T.Compose([
#     #     T.Resize((640, 640)),
#     #     T.ToTensor(),
#     # ])
#     # im_data = transforms(im_pil)[None].to(args.device)

#     # output = model(im_data, orig_size)
#     # labels, boxes, scores = output

#     # draw([im_pil], labels, boxes, scores)


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-c', '--config', type=str, default='/home/sports/data/code/rtDETRv2/RT-DETR-main/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml')
#     parser.add_argument('-r', '--resume', type=str, default='/home/sports/data/code/rtDETR/RT-DETR/rtdetrv2_pytorch/pretrained/rtdetrv2_r50vd_m_7x_coco_ema.pth')
#     parser.add_argument('-f', '--im-file', type=str, )
#     parser.add_argument('-d', '--device', type=str, default='cuda')
#     parser.add_argument('-K', '--input_len', type=int, default=6, help='K frames as input')
#     parser.add_argument('--test-only', action='store_true', default=False)
#     args = parser.parse_args()
#     main(args)
