import PyPDF2
import os

def merge_pdfs(pdf_list, output_path):
    pdf_merger = PyPDF2.PdfMerger()
    
    for pdf in pdf_list:
        pdf_merger.append(pdf)
    
    with open(output_path, 'wb') as output_pdf:
        pdf_merger.write(output_pdf)

if __name__ == "__main__":
    # Caminho da pasta contendo os arquivos PDF
    folder_path = '/home/nauanelinhares/tg1/Acidentes'
    
    # Lista de arquivos PDF a serem combinados
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    # Caminho do arquivo PDF de saída
    output_pdf_path = '/home/nauanelinhares/tg1/saida.pdf'
    
    merge_pdfs(pdf_files, output_pdf_path)
    print(f'PDFs combinados em {output_pdf_path}')
