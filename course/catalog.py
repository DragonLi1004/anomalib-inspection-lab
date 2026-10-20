"""Display catalog; stable experiment IDs and metric records remain unchanged."""
DATASETS = {
    'bottle': {'label': '瓶子', 'scope': '正式评估', 'examples': '大面积破损、小面积破损、污染'},
    'cable': {'label': '电缆', 'scope': '正式评估', 'examples': '导线弯曲、错位、缺失、绝缘层切伤或穿孔'},
    'capsule': {'label': '胶囊', 'scope': '正式评估', 'examples': '裂纹、印字异常、穿孔、划伤、挤压变形'},
    'hazelnut_toy': {'label': '榛子（演示）', 'scope': '小样本演示', 'examples': '颜色异常、裂纹'},
}
METHODS = {
    'baseline': {'name': 'PatchCore-R18-384', 'label': '基线模型', 'status': '默认检测方案'},
    'pca192': {'name': 'PatchCore-R18-PCA192', 'label': 'PCA 192 维实验', 'status': '实验方案 · 存在精度损失'},
    'pca96': {'name': 'PatchCore-R18-PCA96', 'label': 'PCA 96 维实验', 'status': '实验方案 · 存在精度损失'},
}
OPTIMIZATIONS = [
    {'id': 'lossless-pack-v1', 'name': '无损模型打包 v1', 'scope': '部署存储',
     'status': '已验证', 'benefit': '9 模型组合包缩小 75.63%；恢复文件字节一致'},
    {'id': 'session-reuse-v1', 'name': '推理会话复用 v1', 'scope': '连续检测服务',
     'status': '已验证', 'benefit': '本机复用请求延迟降低 97.55%–97.91%；首次请求仍需加载'},
    {'id': 'exact-search-chunking', 'name': '精确检索分块实验', 'scope': '检索实现',
     'status': '未采用', 'benefit': '检测输出不变，但耗时增加且显存收益不稳定'},
]


def describe_run(row):
    """Human-readable alias, never used to load/rename a model or join experiments."""
    dataset = row.get('dataset', 'hazelnut_toy')
    method = row.get('method', 'baseline')
    info = METHODS.get(method)
    backbone = row.get('backbone')
    supported = backbone in ('resnet18', 'resnet18.tv_in1k')
    model_name = info['name'] if info and supported else f"PatchCore / {backbone or '骨干未记录'} / {method}"
    dataset_label = DATASETS.get(dataset, {}).get('label', dataset)
    method_label = info['label'] if info else method
    status = info['status'] if info else '未登记方法'
    protocol_label = '配对协议 v2' if row.get('protocol_version') == 'paired-coreset-v2' else '早期记录'
    run_id = row.get('id', row.get('run_id', ''))
    display_name = f"{dataset_label}｜{model_name}｜种子 {row.get('seed', '未记录')}｜{protocol_label}｜{run_id[-15:]}"
    return {'dataset_label': dataset_label, 'model_name': model_name, 'method_label': method_label,
            'method_status': status, 'display_name': display_name, 'protocol_label': protocol_label}


def catalog():
    return {'datasets': DATASETS, 'methods': METHODS, 'optimizations': OPTIMIZATIONS}
