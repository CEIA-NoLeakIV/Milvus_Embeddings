import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from .layers import GDC

def create_resnet50(embedding_dim=512, pretrained=True):
    """
    Cria um backbone ResNet50 para extração de features,
    substituindo a camada final 'fc' por uma camada de embedding.
    """
    # Carrega o ResNet50. 
    # 'weights=ResNet50_Weights.DEFAULT' carrega os pesos pré-treinados da ImageNet.
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = resnet50(weights=weights)

    # A camada original 'fc' do ResNet50 entra com 2048 features e sai com 1000 (classes).
    # Vamos substituí-la!

    # Opção A: Simples nn.Linear (como no seu Keras ResNet)
    # in_features = model.fc.in_features # (é 2048)
    # model.fc = nn.Linear(in_features, embedding_dim)

    # Opção B: Usar GDC (Global Depthwise Conv) + FC (como no MobileNetV1 do Repo 2)
    # Isso é geralmente melhor para reconhecimento facial.
    in_features = model.fc.in_features # (é 2048)
    model.fc = nn.Sequential(
        nn.Linear(in_features, embedding_dim, bias=False),
        nn.BatchNorm1d(embedding_dim)
    )

    # O GDC (utils.layers.GDC) espera um input 4D (feature map)
    # O ResNet5a padrão já faz o AvgPool, então vamos usar a Opção B (Linear + BN).

    print(f"Backbone ResNet50 criado. Camada 'fc' substituída por Linear({in_features}, {embedding_dim}) + BN1d.")

    return model