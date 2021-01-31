## ResNet20 pruning

Данный репозиторий содержит код для обучения модели ResNet20 на датасете CIFAR-10, а также реализацию базового метода pruning'а.

Filter-level метод pruning'а из следующих двух шагов:

1. Для каждого сверточного слоя кластеризуется набор сверток, его представляющий (тензор весов), при этом кластеризация проводится отдельно для каждого слоя.
2. Каждая исходная свертка в определенном слое заменяется на центроид кластера, к которому она была отнесена на шаге 1.

Код реализации моделей и вспомогательных методов находится в файлах ```resnet.py```, ```prune_model.py``` и ```train_utils.py```, ```pruning_utils.py``` соответственно.
Результаты обучения модели и экспериментов с методом pruning'а находятся в файле ```Pruning.ipynb```.