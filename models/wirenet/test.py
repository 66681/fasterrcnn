from models.wirenet.wirepoint_dataset import WirePointDataset
from models.config.config_tool import read_yaml

# image_file = "D:/python/PycharmProjects/data"
#
# label_file = "D:/python/PycharmProjects/data/labels/train"
# dataset_test = WireDataset(image_file)
# dataset_test.show(0)
# for i in dataset_test:
#     print(i)
cfg = 'wirenet.yaml'
cfg = read_yaml(cfg)
print(f'cfg:{cfg}')
print(cfg['model']['n_dyn_negl'])
# net = WirepointPredictor()

dataset = WirePointDataset(dataset_path=cfg['io']['datadir'], dataset_type='val')
# dataset.show(0)

for i in range(len(dataset)):
    dataset.show(i)


