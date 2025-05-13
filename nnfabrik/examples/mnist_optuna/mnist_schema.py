import os
import datajoint as dj
dj.config['enable_python_native_blobs'] = True
if not "stores" in dj.config:
    dj.config["stores"] = {}

dj.config['database.host'] = os.environ['DJ_LOCALHOST']
dj.config['database.user'] = os.environ['DJ_LOCALUSER']
dj.config['database.password'] = os.environ['DJ_LOCALPASS']  
dj.config["stores"]["minio"] = {  # store in local folder
        "protocol": "file",
        "location": "/home/neurd/workspace/DNN/nnfabrik_utils/trainedModels_mnist"
}

    
from nnfabrik.main import my_nnfabrik
from nnfabrik.templates.trained_model import TrainedOptunaModelBase #TrainedModelBase
## define nnfabrik tables here
model_schema_name = "nnfabrik_mnist"
nnfabrik_module = my_nnfabrik(
    model_schema_name,
    context = None,
)
Fabrikant, Seed, Model, Dataset, Trainer = nnfabrik_module.Fabrikant, nnfabrik_module.Seed,\
nnfabrik_module.Model, nnfabrik_module.Dataset, nnfabrik_module.Trainer

schema = dj.Schema(model_schema_name)
@schema
class TrainedModel(TrainedOptunaModelBase):
    table_comment = "mnist trained models with extra tuna"
    nnfabrik = nnfabrik_module

