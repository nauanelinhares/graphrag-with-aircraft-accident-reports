import requests
from lxml import html
import os
import random

# URL da página que queremos fazer o scraping


for page in range(120):
    num_of_files = len([name for name in os.listdir('Acidentes') if os.path.isfile(os.path.join('Acidentes', name))])

    url = f'https://sistema.cenipa.fab.mil.br/cenipa/paginas/relatorios/relatorios?&&?&pag={page}'

    # Fazendo a requisição HTTP para obter o conteúdo da página
    response = requests.get(url)
    response.raise_for_status()  # Verifica se a requisição foi bem-sucedida

    # Parseando o conteúdo HTML da página
    tree = html.fromstring(response.content)

    # Gerando 20 números aleatórios de 1 a 120
    random_reports = random.sample(range(1, 121), 20)

    # Loop para baixar os relatórios variando o tr[]
    # Encontrando todos os links com o título "Relatório Final em Português"
    links = tree.xpath('//a[@title="Relatório Final em Português"]')

    base_url = 'https://sistema.cenipa.fab.mil.br/cenipa/paginas/relatorios/'


    # Count the number of files in the 'Acidentes' directory

    for i, link in enumerate(links, start=1):
        if(i in random_reports):
            report_url = base_url + link.get('href')
            report_response = requests.get(report_url)
            report_response.raise_for_status()
            
            # Salvando o relatório em um arquivo
            with open(f'Acidentes/report_{num_of_files+i}.pdf', 'wb') as file:
                file.write(report_response.content)
            print(f'Relatório {num_of_files+i} baixado com sucesso.')
