from pytorch_tabnet.tab_model import TabNetClassifier
from config.TabNetConfig import *

# TabNet模型
tabnet_model = TabNetClassifier(
    n_d=8,
    n_a=8,
    n_steps=3,
    gamma=1.3,
    n_independent=2,
    n_shared=2,
    optimizer_params=dict(lr=2e-2),
    verbose=0,
)