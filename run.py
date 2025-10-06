from PIL import Image
import depth_pro
import torch
import numpy as np
import argparse
import os
import os.path as osp
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--input_dir", type=str, default="/mnt/localssd/data/coco/train2017", help="data directory containing all images to run")
    parser.add_argument("-o", "--output_dir", type=str, default="/mnt/localssd/data/coco/annotations/depth-pro_pred", help="output directory")
    parser.add_argument("-n", "--num_workers", type=int, default=1, help="Number of workers for data loading")
    parser.add_argument("-i", "--worker_id", type=int, default=0, help="Worker ID")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    num_processes = args.num_workers
    process_id = args.worker_id
    device = 'cuda:' + str(process_id % num_processes)

    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms()
    model = model.to(device)
    model.eval()

    image_paths = [osp.join(args.input_dir, f) for f in os.listdir(args.input_dir)]
    assert len(image_paths) == len([x for x in image_paths if x.endswith('.jpg')])

    if num_processes > 1:
        image_paths = image_paths[process_id::num_processes]
        print(f"Job idx={process_id}/N={num_processes}: processing {len(image_paths)} images. ({image_paths[0]} ...)")

    for image_path in tqdm(image_paths):
        # Load and preprocess an image.
        image, _, f_px = depth_pro.load_rgb(image_path)
        image = transform(image).to(device)

        # Run inference.
        with torch.no_grad():
            prediction = model.infer(image, f_px=f_px)
            depth = prediction["depth"]  # Depth in [m].
            focallength_px = prediction["focallength_px"]  # Focal length in pixels.
        
        depth_np = depth.squeeze().cpu().numpy()
        output_path = osp.join(args.output_dir, osp.basename(image_path).replace('.jpg', '.npz'))
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            np.savez(f, depth=depth_np, f_px=focallength_px.item())
        
        # vis_img = np.asarray(Image.open(image_path).convert('RGB'))
        # load_depth = np.load(output_path)['depth']
        # max_depth = np.percentile(load_depth, 95)
        # vis_depth = (load_depth - load_depth.min()) / (max_depth - load_depth.min()) 
        # vis_depth = np.clip(vis_depth, 0, 1)
        # vis_depth = (vis_depth * 255).astype(np.uint8)
        # vis_depth = np.stack([vis_depth]*3, axis=-1)
        # vis = np.concatenate([vis_img, vis_depth], axis=1)
        # Image.fromarray(vis).save('tmp_'+osp.basename(image_path))
        # breakpoint()