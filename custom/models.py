#custom.models
import os
import gc
import torch
import random

from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from anomalib.models.image.patchcore.lightning_model import logger
from anomalib.models.components import KCenterGreedy
from anomalib.models import Patchcore
from anomalib.models.image.efficient_ad.lightning_model import EfficientAd
from anomalib.models.image.efficient_ad.torch_model import reduce_tensor_elems

# from .dinomaly.lightning_model import Dinomaly
import custom.detector as Detector

__all__ = [
    # "Dinomaly",
    "Detector",
]

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class CustomPatchcoreModel(PatchcoreModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_bank = torch.tensor([], device='cuda') # 메모리 뱅크를 GPU에 관리

    def subsample_embedding(self, embeddings: list[torch.Tensor], sampling_ratio: float) -> None:
        coreset_list = []
        logger.info(f"Starting subsample_embedding with {len(embeddings)} embeddings.")
        for idx, embedding in enumerate(embeddings):
            embedding = embedding.cuda()
            sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
            coreset = sampler.sample_coreset()
            coreset_list.append(coreset.cpu())
            logger.info(f"Coreset sampled for embedding {idx+1}, Shape: {coreset.shape}")
            
            del idx
            del embedding
            del sampler
            
            torch.cuda.empty_cache()
            gc.collect()
            
        self.memory_bank = torch.vstack(coreset_list).cuda() # 메모리 뱅크에 CPU로 코어셋 저장
        
        del embeddings
        
        torch.cuda.empty_cache()
        gc.collect()
        
    def nearest_neighbors(self, embedding: torch.Tensor, n_neighbors: int):
        distances = self.euclidean_dist(embedding, self.memory_bank)  # 메모리 뱅크를 GPU에서 접근
        
        del embedding  
        
        torch.cuda.empty_cache()
        gc.collect()

        if n_neighbors == 1:
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores, locations
    
    @staticmethod
    def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device = x.device
        y = y.to(device)
        
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)  # |x|
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)  # |y|
        
        res = torch.zeros((x.shape[0], y.shape[0]), device=device) 
        
        if x.shape[0] >= 2**15 or y.shape[0] >= 2**15:
            if len(x.shape) == 3:
                x = x.view(-1, x.shape[-1])  
            if len(y.shape) == 3:  
                y = y.view(-1, y.shape[-1]) 
            
            n = len(res)
            batch_size = int(n / 16)
            for i in range(0, n, batch_size):
                x_batch = x[i : i + batch_size]
                res[i : i + batch_size] = (
                    x_batch.pow(2).sum(dim=1, keepdim=True)  # |x_batch|
                    - 2 * torch.matmul(x_batch, y.transpose(-2, -1))  # - 2 * x_batch @ y.T
                    + y_norm.transpose(-2, -1)  # |y|.T
                ) 
        else: # row distance can be rewritten as sqrt(|x| - 2 * x @ y.T + |y|.T)
            res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
            
        return res.clamp_min_(0).sqrt_()
    
class Patchcore(Patchcore):
    def __init__(self, *args, **kwargs):
        pre_trained = kwargs.pop("pre_trained", True)
        
        super().__init__(*args, **kwargs)
        
        self.model = CustomPatchcoreModel(
            backbone=self.model.backbone,
            layers=self.model.layers,
            num_neighbors=self.model.num_neighbors,
            pre_trained=pre_trained,
        )

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> None:
        del args, kwargs  # These variables are not used.

        embedding = self.model(batch['image'])
        self.embeddings.append(embedding.cpu())
        
        del embedding
        
        torch.cuda.empty_cache()
        gc.collect()        

        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        logger.info("Applying core-set subsampling to get the embedding.")
        self.model.subsample_embedding(self.embeddings, self.coreset_sampling_ratio)
        
'''============================================================================================================'''
    
class EfficientAd(EfficientAd):
    def _get_quantiles_of_maps(self, maps: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        num_maps = len(maps) // 2
        sampled_maps = random.sample(maps, num_maps)
        
        flat = []
        logger.info(f"Starting get_quantiles of {len(sampled_maps)} maps.")
        for idx, map in enumerate(sampled_maps):
            map_flat = reduce_tensor_elems(map, m=2**10)
            flat.append(map_flat)
            logger.info(f"map_elems reduced by flatten {idx+1}, Shape: {map_flat.shape}")
            
            # del idx
            # del map

        maps_flat = torch.flatten(torch.cat(flat))
        qa = torch.quantile(maps_flat, q=0.9).to(self.device)
        qb = torch.quantile(maps_flat, q=0.995).to(self.device)
        
        torch.cuda.empty_cache()
        gc.collect()  
        
        return qa, qb