from examples.sim_arrange_boxes.config import TrainConfig as ArrangeBoxesTrainConfig
# from examples.sim_arrange_boxes.config import TrainConfig as ArrangeBoxesTrainConfig
from examples.sim_pick.config import TrainConfig as PickTrainConfig

CONFIG_MAPPING = {
    "arrange_boxes": ArrangeBoxesTrainConfig,
    # "arrange_boxes": ArrangeBoxesTrainConfig,
    "pick": PickTrainConfig,
}


"""
#from examples.sim_arrange_boxes.config import TrainConfig as ArrangeBoxesTrainConfig
from examples.sim_pick.config import TrainConfig as PickTrainConfig

CONFIG_MAPPING = {
    "arrange_boxes": ArrangeBoxesTrainConfig,
    #"pick": PickTrainConfig,
}"""
