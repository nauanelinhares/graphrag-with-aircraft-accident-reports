# Estudo sobre o uso de Modelos de Linguagem e Grafos de Conhecimento com RAG no Tráfego Aéreo Brasileiro

Este projeto implementa uma abordagem para auxiliar o monitoramento e a tomada de decisões no tráfego aéreo brasileiro, utilizando Modelos de Linguagem de Grande Escala (LLMs) combinados com Grafos de Conhecimento e Recuperação de Informação Aumentada (RAG).

## Objetivo

O objetivo é desenvolver um sistema de consulta capaz de responder perguntas relacionadas ao tráfego aéreo no Brasil, de consultas simples (como previsões meteorológicas) a questões complexas (como causas de acidentes aéreos).

## Estrutura do Projeto

### 1. Componentes Principais

- **Modelos de Linguagem (LLMs):** Utiliza LLMs para processamento de linguagem natural, suportando tarefas como consulta de dados meteorológicos e análise de padrões de voo.
- **Recuperação Aumentada (RAG):** Melhora a precisão das respostas dos LLMs integrando dados específicos do contexto.
- **Grafos de Conhecimento:** Representam relações complexas entre entidades e permitem respostas contextualizadas, aplicando informações sobre normas de aviação e dados de tráfego aéreo.

### 2. Frameworks e Ferramentas

- **Python**: Linguagem base para a implementação.
- **Streamlit**: Interface de visualização dos dados em tempo real.
- **Nebula Graph**: Gerenciamento e consulta de grafos de conhecimento.
- **LlamaHub e Ollama**: Conexão e execução de modelos de linguagem, incluindo comparações entre modelos open-source e proprietários.
- **Cohere Rerank**: Aperfeiçoamento de respostas do sistema de busca.

## Arquitetura

A arquitetura se baseia em um fluxo de dados entre:

1. **Processamento de Dados**: Indexação e transformação dos dados brutos (PDFs, HTML) para um formato de grafo.
2. **LLMs com RAG**: Respostas a consultas dos usuários, contextualizadas com dados recuperados dos grafos e RAG.
3. **Exibição de Resultados**: Visualização no Streamlit.

## Dados Utilizados

- **Dados de voo (VRA)**: Histórico de atrasos, condições de voo e análises meteorológicas.
- **Normas Aeronáuticas (RBAC, ICA, NOGEF)**: Para consultas sobre regulamentação.
- **Relatórios de Segurança (SIPAER e CENIPA)**: Estatísticas de acidentes e incidentes no tráfego aéreo.

## Funcionalidades

- Consulta em tempo real sobre condições de voo e normas regulatórias.
- Análise de condições meteorológicas com base em dados METAR e TAF.
- Identificação de padrões de risco e causas de atrasos ou acidentes.

## Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/usuario/projeto-tg

   ```

2. Instale as dependências:

   ```bash
   pip install -r requirements.txt

   ```

3. Execute o projeto:
   ```bash
   streamlit run streamlit/main.py
   ```

---

