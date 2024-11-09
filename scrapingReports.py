import requests
from lxml import html
import os
import random
import pandas as pd

# URL da página que queremos fazer o scraping

scrapingInCipraer = True 
scrapingInToCsv = False

def getPdfsFromCenipa(link, i):
    report_response = requests.get(link)
    report_response.raise_for_status()
    with open(f'Acidentes/report_{i}.pdf', 'wb') as file:
        file.write(report_response.content)
        print(f'Relatório {i} baixado com sucesso.')



def getPDFsLinksFromCenipa(cenipaLink):
    response = requests.get(cenipaLink)
    response.raise_for_status()  # Verifica se a requisição foi bem-sucedida

        # Parseando o conteúdo HTML da página
    tree = html.fromstring(response.content)
    links = tree.xpath('//a[@title="Relatório Final em Português"]')
    base_url = 'https://sistema.cenipa.fab.mil.br/cenipa/paginas/relatorios/'
    
    for i, link in enumerate(links, start=1):
        link_url =  link.get('href')
        report_url = base_url +link_url
        getPdfsFromCenipa(report_url, i)




if scrapingInToCsv:
    df = pd.read_csv('ocorrencias.csv')
    i = 0 
    for index, row in df.iterrows():
        i = i + 1
        if pd.isna(row['Historico']):
            url = row['Link']
            response = requests.get(url)
            response.raise_for_status()  # V
            tree = html.fromstring(response.content)
            
            #  to x path<p class="text-justify">
            
            elements = tree.xpath('//p[@class="text-justify"]')
            
            if elements:  
                for element in elements:
                    # print(element.text_content())
                    row['Historico'] = element.text_content()
                    df.at[index, 'Historico'] = element.text_content()
            # else:
            #     cenipaLinks = tree.xpath('//a[contains(text(), "Clique aqui")]')
            #     if cenipaLinks:  
            #         for link in cenipaLinks:
            #             getPDFsLinksFromCenipa(link.get('href'))
        print(i)
    df.to_csv('ocorrencias_atualizadas.csv', index=False)


if scrapingInCipraer:
    url = 'https://dedalo.sti.fab.mil.br/ocorrencia/77866'
    response = requests.get(url)
    response.raise_for_status()  # V
    tree = html.fromstring(response.content)
    cenipaLinks = tree.xpath('//a[contains(text(), "Clique aqui")]')
    if cenipaLinks:  
        for link in cenipaLinks:
            getPDFsLinksFromCenipa(link.get('href'))

            

