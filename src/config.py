import os
from omegaconf import OmegaConf

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yml'))

SETTINGS = OmegaConf.load(CONFIG_PATH)

OmegaConf.set_struct(SETTINGS, False)

SETTINGS.map_width_px = SETTINGS.grid_width * SETTINGS.block_size
SETTINGS.map_height_px = SETTINGS.grid_height * SETTINGS.block_size
SETTINGS.window_width = SETTINGS.map_width_px + SETTINGS.sidebar_width
SETTINGS.window_height = SETTINGS.map_height_px

OmegaConf.set_struct(SETTINGS, True)