"""
Face Recognition API - Runner Script
=====================================
Script para iniciar a API Flask de reconhecimento facial.

Uso:
    python run_api.py
    python run_api.py --host 0.0.0.0 --port 5000 --debug
"""

import argparse
import sys
import os

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.api import create_app


def parse_args():
    parser = argparse.ArgumentParser(
        description="Face Recognition API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
    python run_api.py                           # Inicia em localhost:5000
    python run_api.py --port 8080               # Inicia na porta 8080
    python run_api.py --host 0.0.0.0 --debug    # Modo debug, acessível externamente
        """
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host para bind do servidor (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Porta do servidor (default: 5000)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Ativar modo debug (hot reload)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("       FACE RECOGNITION API")
    print("=" * 60)
    print(f"  Host:  {args.host}")
    print(f"  Port:  {args.port}")
    print(f"  Debug: {args.debug}")
    print("=" * 60)
    print()
    print("Endpoints disponíveis:")
    print(f"  POST /api/embedding          - Gerar embedding de uma imagem")
    print(f"  POST /api/embeddings/batch   - Gerar embeddings em lote")
    print(f"  POST /api/milvus/insert      - Inserir embeddings no Milvus")
    print(f"  GET  /api/health             - Health check")
    print(f"  GET  /api/models             - Listar modelos disponíveis")
    print()
    print("=" * 60)
    
    app = create_app()
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == "__main__":
    main()