from datasets import load_dataset
bookcorpus_dataset = load_dataset('bookcorpus')

# 查看数据集中包含的split。常见的split有'train', 'test', 'validation'。
print(bookcorpus_dataset.keys())

# 查看某个split（如'train'）的数据大小
print(len(bookcorpus_dataset['train'])*0.01)

# 查看数据集的一些样例
print(bookcorpus_dataset['train'][0])