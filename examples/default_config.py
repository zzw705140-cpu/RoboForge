from abc import abstractmethod

class DefaultConfig:
    """所有任务共享的默认训练配置。"""

    # 每次 BC/SAC 参数更新使用的 transition 数量
    batch_size: int = 64

    # Replay buffer 允许保存的最大 transition 数量
    replay_buffer_capacity: int = 200000

    # 训练指标记录间隔
    log_period: int = 10

    # RLPD/SAC 训练参数
    cta_ratio: int = 2                    #Critic与Actor的更新比例为2，critic2次，actor1次
    discount: float = 0.97                #折扣因子
    max_steps: int = 1_000_000            #单个episode长度
    random_steps: int = 0                 #直接从专家示范数据开始，不随机探索
    training_starts: int = 100            #累计100条transition后，learner开始训练
    steps_per_update: int = 50            #每训练50步向actor发布一次最新的网络参数

    # 设为 0 时关闭对应的周期性持久化。
    checkpoint_period: int = 0            #隔多少训练步保存一次模型参数
    buffer_period: int = 0                #控制每隔多少个环境交互步保存一次经验池数据

    # "resnet":            从头训练 ResNet-10
    # "resnet-pretrained": 使用预训练并冻结的 ResNet-10
    # 现在表示使用已经训练好的ResNet-10参数
    encoder_type: str = "resnet-pretrained"

    @abstractmethod
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        raise NotImplementedError
