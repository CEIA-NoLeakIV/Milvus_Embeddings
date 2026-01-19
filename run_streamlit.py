#!/usr/bin/env python3
"""
Face Recognition - Streamlit Interface Runner
==============================================
Script para iniciar a interface Streamlit de busca por similaridade.

Uso:
    python run_streamlit.py
    python run_streamlit.py --port 8501
"""

import argparse
import subprocess
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Face Recognition Streamlit Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
    python run_streamlit.py                    # Inicia em localhost:8501
    python run_streamlit.py --port 8502        # Inicia na porta 8502
        """
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Porta do servidor Streamlit (default: 8501)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host para bind do servidor (default: localhost)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Caminho para o app Streamlit
    app_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "streamlit_app",
        "app.py"
    )
    
    if not os.path.exists(app_path):
        print(f"Erro: Arquivo não encontrado: {app_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("       FACE RECOGNITION - STREAMLIT INTERFACE")
    print("=" * 60)
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  URL:  http://{args.host}:{args.port}")
    print("=" * 60)
    print()
    
    # Comando para executar o Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        app_path,
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true"
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServidor encerrado.")
    except Exception as e:
        print(f"Erro ao iniciar Streamlit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()