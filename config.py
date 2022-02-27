class Config(object):
    data_path = './data'  # 文件存放路径
    train_file = '/测试用.txt'
    val_file = '/val.txt'
    test_file = '/test.txt'
    lr = 1e-3
    weight_decay = 1e-4
    use_gpu = False
    epoch = 20
    batch_size = 8
    maxlen = 125  # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    save_every_epoch = 20  # 每20个batch 可视化一次
    model_path = './checkpoints/model2'  # 模型保存路径
    train = True
    val = True
    test = True
    embedding_dim = 32

