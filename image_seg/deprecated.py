class MobilityDataset(Dataset): 
    """ deprecated: use SamH5Dataset instead!"""
    def __init__(self, root_dir, transform=None, num_masks=9, is_train=True, split=0.9):
        self.root_dir = root_dir
        self.transform = transform
        self.num_masks = num_masks
        self.is_train = is_train
        self.split = split
        self.filenames = self.load_filenames(root_dir, is_train, split)
        
    def load_filenames(self, root_dir, is_train=True, split=0.9):
        filenames = []
        for obj in natsorted(os.listdir(root_dir)):
            obj_dir = join(root_dir, obj)
            obj_filenames = []
            for folder in natsorted(os.listdir(obj_dir)):
                folder_dir = join(obj_dir, folder)

                for view in os.listdir(folder_dir):
                    view_dir = join(folder_dir, view) # /local/real/mandi/sam_dataset/Safe/xxx/view_0/
                    rgb_fname = join(view_dir, "rgb.png")
                    mask_fnames = natsorted(glob(join(view_dir, "mask_*.png")))[:self.num_masks]
                    if os.path.exists(rgb_fname) and len(mask_fnames) > 0:
                        obj_filenames.append(
                            {
                                "obj": obj,
                                "folder": folder,
                                "view": view,
                                "rgb_fname": rgb_fname,
                                "mask_fnames": mask_fnames,
                            }
                        )
            # determiniscally shuffle filenames
            random.seed(42)
            random.shuffle(obj_filenames)
            if len(obj_filenames) > 0:
                num_files = int(len(obj_filenames)* split)
                num_files = max(num_files, 1)
                num_files = min(num_files, len(obj_filenames)) 
                if is_train:
                    filenames.extend(obj_filenames[:num_files])
                else:
                    filenames.extend(obj_filenames[num_files:])
                
        print(f"Train Set? {is_train} - Found {len(filenames)} image-mask pairs")
        return filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx): 
        image_path = self.filenames[idx]["rgb_fname"] 
        image = Image.open(image_path).convert("RGB")
        image = np.array(image, dtype=np.float32)  
        masks = []
        masks_path = self.filenames[idx]["mask_fnames"]
        for mask_path in masks_path:
            mask = Image.open(mask_path)
            mask = np.array(mask, dtype=np.float32)
            masks.append(mask)
        if len(masks) < self.num_masks:
            masks.extend([
                np.zeros(masks[0].shape) for _ in range(self.num_masks - len(masks))
            ])
            
        masks = np.stack(masks, axis=0)

        rgb, masks, original_size = self.transform(image, masks)
        
        return dict(
            image=rgb,
            masks=masks, 
            # original_size=original_size,
            fname=image_path,
        )

class SamPointsDataset(Dataset): 
    """ 
    Deprecated: use SamH5Dataset instead!
    For each image, samples points for each mask
    """
    def __init__(self, root_dir, transform=None, prompts_per_mask=6, is_train=True, split=0.9, seed=42, filter_masks=True):
        self.root_dir = root_dir
        self.transform = transform
        self.prompts_per_mask = prompts_per_mask # for each (image, mask) pair, how many point prompts to sample
        self.is_train = is_train
        self.split = split
        self.filenames = self.load_filenames(root_dir, is_train, split, filter_masks)
        self.np_random = np.random.RandomState(seed)
        
    def load_filenames(self, root_dir, is_train=True, split=0.9, filter_masks=True):
        """ try load each image-mask pair separately"""
        filenames = []
        for obj in natsorted(os.listdir(root_dir)):
            obj_dir = join(root_dir, obj)
            obj_filenames = []
            for folder in natsorted(os.listdir(obj_dir)):
                folder_dir = join(obj_dir, folder) 
                for view in os.listdir(folder_dir):
                    view_dir = join(folder_dir, view) # /local/real/mandi/blender_dataset_v3/test/Table/xxx/loop_0/
                    rgb_fname = join(view_dir, "rgb.png")
                    mask_fnames = natsorted(glob(join(view_dir, "mask_*.png")))
                    if filter_masks: # skip the near empty masks
                        mask_fnames = [m for m in mask_fnames if np.sum(np.array(Image.open(m))) > 10]
                    if os.path.exists(rgb_fname) and len(mask_fnames) > 0: 
                        obj_filenames.append(
                            {
                                "obj": obj,
                                "folder": folder,
                                "view": view,
                                "rgb_fname": rgb_fname,
                                "mask_fnames": mask_fnames,
                            }
                        )
            # determiniscally shuffle filenames
            random.seed(42)
            random.shuffle(obj_filenames)
            if len(obj_filenames) > 0:
                num_files = int(len(obj_filenames)* split)
                num_files = max(num_files, 1)
                num_files = min(num_files, len(obj_filenames)) 
                if is_train:
                    filenames.extend(obj_filenames[:num_files])
                else:
                    filenames.extend(obj_filenames[num_files:])

        print(f"Train Set? {is_train} - Found {len(filenames)} image-all_its_mask pairs")
        return filenames

    def __len__(self):
        return len(self.filenames)

    def sample_points(self, mask, num_points):
        """ uniformly sample foreground points on a mask, note here assumes one point per prompt per mask"""
        pos_points = np.where(mask > 0) 
        # if no points, sample from the whole image
        if len(pos_points[0]) == 0:
            pos_points = np.where(mask >= 0) 
        replace = len(pos_points[0]) < num_points
        point_idxs = self.np_random.choice(len(pos_points[0]), size=num_points, replace=replace)
        point_coords = np.stack([pos_points[0][point_idxs], pos_points[1][point_idxs]], axis=1) # shape num_points, 2   
        point_labels = mask[point_coords[:,0], point_coords[:,1]]

        point_coords = torch.from_numpy(point_coords).float()
        point_labels = torch.from_numpy(point_labels).float()
        point_coords, point_labels = point_coords.unsqueeze(1), point_labels.unsqueeze(1) # shape (num_points, 1, 2), (num_points, 1)
        return point_coords, point_labels

    def sample_one_point(self, mask):
        pos_points = np.where(mask > 0) 
        # if no points, sample from the whole image
        if len(pos_points[0]) == 0:
            pos_points = np.where(mask >= 0) 
        point_idxs = self.np_random.choice(len(pos_points[0]), size=1, replace=False)
        coord = np.stack([pos_points[0][point_idxs], pos_points[1][point_idxs]], axis=1) # shape 1, 2   
        label = mask[coord[:,0], coord[:,1]]
        return coord, label

    def __getitem__(self, idx): 
        """ each sample is one RGB + sample multiple masks * multiple different point prompts """
        image_path = self.filenames[idx]["rgb_fname"] 
        image = Image.open(image_path).convert("RGB")
        image = np.array(image, dtype=np.float32)  
    
        mask_fnames = self.filenames[idx]["mask_fnames"]
        mask_idxs = self.np_random.choice(len(mask_fnames), size=self.prompts_per_mask, replace=True)
        mask_imgs = [Image.open(mask_fnames[i]) for i in mask_idxs]
        masks = [np.array(mask, dtype=np.float32) for mask in mask_imgs] 
        
        point_coords, point_labels = [], []
        for mask in masks:
            coord, label = self.sample_one_point(mask)
            point_coords.append(coord)
            point_labels.append(label)
        point_coords = np.stack(point_coords, axis=0) # shape num_masks, 1, 2
        point_labels = np.stack(point_labels, axis=0) # shape num_masks, 1

        masks = np.stack(masks, axis=0)
        rgb, masks, original_size = self.transform(image, masks)
        
        # save as imgs
        # for i, mask in enumerate(masks):
        #     mask_cp = deepcopy(mask)
        #     # convert to 3-channel rgb:
        #     mask_cp = mask_cp[None, :].repeat(3,1,1).numpy() * 255
        #     x = int(point_coords[i,0,0].item())
        #     y = int(point_coords[i,0,1].item())
        #     mask_cp[:, x, y] = np.array([255,0,0])
        #     img = Image.fromarray(mask_cp.transpose(1,2,0).astype(np.uint8))
        #     img.save(f"test_{idx}_{i}.png")
        # breakpoint()

        return dict(
            image=rgb,
            masks=masks,
            point_coords=point_coords,
            point_labels=point_labels,
            # original_size=original_size,
            fname=image_path,
        )

