from .nerf import NeRFDataset
from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .nerfpp import NeRFPPDataset
from .rtmv import RTMVDataset
from .spot import SpotDataset
from .brics import BRICSDataset
from .spot_online import SpotOnlineDataset
from .spot_vis import SpotVisDataset

dataset_dict = {'nerf': NeRFDataset,
                'nsvf': NSVFDataset,
                'colmap': ColmapDataset,
                'nerfpp': NeRFPPDataset,
                'rtmv': RTMVDataset,
                "spot": SpotDataset,
                "brics": BRICSDataset,
                'spot_online': SpotOnlineDataset,
                'spot_vis': SpotVisDataset}
